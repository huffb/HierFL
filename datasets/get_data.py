from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _dataset_root(args) -> Path:
    return _project_root() / args.dataset_root


def _load_dataset(args):
    dataset_name = args.dataset.lower()
    dataset_root = _dataset_root(args)
    dataset_root.mkdir(parents=True, exist_ok=True)

    if dataset_name == "mnist":
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )
        train_dataset = datasets.MNIST(
            root=dataset_root, train=True, download=True, transform=transform
        )
        test_dataset = datasets.MNIST(
            root=dataset_root, train=False, download=True, transform=transform
        )
        num_classes = 10
    elif dataset_name == "fmnist":
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.2860,), (0.3530,)),
            ]
        )
        train_dataset = datasets.FashionMNIST(
            root=dataset_root, train=True, download=True, transform=transform
        )
        test_dataset = datasets.FashionMNIST(
            root=dataset_root, train=False, download=True, transform=transform
        )
        num_classes = 10
    elif dataset_name == "cifar10":
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )
        train_dataset = datasets.CIFAR10(
            root=dataset_root, train=True, download=True, transform=transform
        )
        test_dataset = datasets.CIFAR10(
            root=dataset_root, train=False, download=True, transform=transform
        )
        num_classes = 10
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    return train_dataset, test_dataset, num_classes


def _extract_targets(dataset) -> np.ndarray:
    targets = getattr(dataset, "targets", None)
    if targets is None:
        raise ValueError("Dataset does not expose targets")
    return np.asarray(targets, dtype=np.int64)


def _split_evenly(indices: np.ndarray, parts: int) -> List[np.ndarray]:
    return [np.asarray(split, dtype=np.int64) for split in np.array_split(indices, parts)]


def _build_iid_splits(targets: np.ndarray, num_clients: int, seed: int) -> List[np.ndarray]:
    rng = np.random.default_rng(seed)
    indices = np.arange(len(targets))
    rng.shuffle(indices)
    return _split_evenly(indices, num_clients)


def _build_one_class_splits(
    targets: np.ndarray, num_clients: int, num_classes: int, seed: int
) -> List[np.ndarray]:
    rng = np.random.default_rng(seed)
    class_indices = []
    for label in range(num_classes):
        label_indices = np.where(targets == label)[0]
        rng.shuffle(label_indices)
        class_indices.append(label_indices)

    client_splits: List[List[np.ndarray]] = [[] for _ in range(num_clients)]
    for label in range(num_classes):
        assigned_clients = list(range(label, num_clients, num_classes))
        splits = np.array_split(class_indices[label], len(assigned_clients))
        for client_id, split in zip(assigned_clients, splits):
            if len(split) > 0:
                client_splits[client_id].append(np.asarray(split, dtype=np.int64))

    return [
        np.concatenate(chunks).astype(np.int64) if chunks else np.empty(0, dtype=np.int64)
        for chunks in client_splits
    ]


def _build_shard_splits(
    targets: np.ndarray,
    num_clients: int,
    classes_per_client: int,
    seed: int,
) -> List[np.ndarray]:
    rng = np.random.default_rng(seed)
    sorted_indices = np.argsort(targets, kind="stable")
    num_shards = max(num_clients * max(classes_per_client, 1), num_clients)
    shards = [
        np.asarray(shard, dtype=np.int64) for shard in np.array_split(sorted_indices, num_shards)
    ]
    rng.shuffle(shards)

    client_splits: List[List[np.ndarray]] = [[] for _ in range(num_clients)]
    shard_cursor = 0
    for client_id in range(num_clients):
        for _ in range(max(classes_per_client, 1)):
            if shard_cursor >= len(shards):
                break
            client_splits[client_id].append(shards[shard_cursor])
            shard_cursor += 1

    while shard_cursor < len(shards):
        client_id = shard_cursor % num_clients
        client_splits[client_id].append(shards[shard_cursor])
        shard_cursor += 1

    return [
        np.concatenate(chunks).astype(np.int64) if chunks else np.empty(0, dtype=np.int64)
        for chunks in client_splits
    ]


def _build_client_splits(
    targets: np.ndarray, args, num_classes: int, seed_offset: int
) -> List[np.ndarray]:
    if args.iid == 1:
        return _build_iid_splits(targets, args.num_clients, args.seed + seed_offset)
    if args.iid == -2:
        return _build_one_class_splits(
            targets, args.num_clients, num_classes, args.seed + seed_offset
        )

    classes_per_client = args.classes_per_client if args.iid == 0 else 2
    return _build_shard_splits(
        targets,
        args.num_clients,
        classes_per_client=classes_per_client,
        seed=args.seed + seed_offset,
    )


def _make_loaders(dataset, client_splits: Sequence[np.ndarray], batch_size: int, train: bool):
    loaders = []
    for split in client_splits:
        subset = Subset(dataset, split.tolist())
        loaders.append(
            DataLoader(
                subset,
                batch_size=batch_size,
                shuffle=train,
                drop_last=False,
            )
        )
    return loaders


def get_dataloaders(args):
    train_dataset, test_dataset, num_classes = _load_dataset(args)
    train_targets = _extract_targets(train_dataset)
    test_targets = _extract_targets(test_dataset)

    train_splits = _build_client_splits(train_targets, args, num_classes, seed_offset=0)
    test_splits = _build_client_splits(test_targets, args, num_classes, seed_offset=10_000)

    train_loaders = _make_loaders(train_dataset, train_splits, args.batch_size, train=True)
    test_loaders = _make_loaders(test_dataset, test_splits, args.batch_size, train=False)
    v_train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False
    )
    v_test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False
    )
    return train_loaders, test_loaders, v_train_loader, v_test_loader


def _iter_labels_from_dataset(dataset) -> Iterable[int]:
    if isinstance(dataset, Subset):
        subset_targets = _extract_targets(dataset.dataset)
        for index in dataset.indices:
            yield int(subset_targets[index])
        return

    for label in _extract_targets(dataset):
        yield int(label)


def show_distribution(data_loader, args):
    labels = list(_iter_labels_from_dataset(data_loader.dataset))
    num_classes = max(args.output_channels, 10)
    distribution = np.zeros(num_classes, dtype=np.int64)
    for label in labels:
        if 0 <= label < num_classes:
            distribution[label] += 1
    return distribution
