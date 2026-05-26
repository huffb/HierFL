import argparse

import torch


def add_dataset_args(parser):
    parser.add_argument(
        '--dataset',
        type=str,
        default='mnist',
        help='Dataset name: mnist, fmnist, cifar10.',
    )
    parser.add_argument(
        '--model',
        type=str,
        default='lenet',
        help='Model name. mnist/fmnist: logistic, lenet, cnn_3; cifar10: resnet18, cnn_complex.',
    )
    parser.add_argument(
        '--input_channels',
        type=int,
        default=1,
        help='Input channels. mnist/fmnist: 1, cifar10: 3.',
    )
    parser.add_argument(
        '--output_channels',
        type=int,
        default=10,
        help='Number of output classes.',
    )
    parser.add_argument(
        '--dataset_root',
        type=str,
        default='data',
        help='Dataset root folder.',
    )
    parser.add_argument(
        '--show_dis',
        type=int,
        default=0,
        help='Whether to print client distribution statistics.',
    )


def add_training_args(parser):
    parser.add_argument(
        '--batch_size',
        type=int,
        default=20,
        help='Client-side batch size.',
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=0.1,
        help='Learning rate of client SGD.',
    )
    parser.add_argument(
        '--lr_decay',
        type=float,
        default=0.995,
        help='Exponential learning-rate decay factor.',
    )
    parser.add_argument(
        '--lr_decay_epoch',
        type=int,
        default=1,
        help='Apply learning-rate decay every N epochs.',
    )
    parser.add_argument(
        '--momentum',
        type=float,
        default=0.95,
        help='SGD momentum.',
    )
    parser.add_argument(
        '--weight_decay',
        type=float,
        default=0,
        help='Weight decay coefficient.',
    )
    parser.add_argument(
        '--SGD_clip',
        type=float,
        default=10,
        help='Gradient clipping norm for local SGD.',
    )
    parser.add_argument(
        '--verbose',
        type=int,
        default=0,
        help='Verbose mode for progress output.',
    )


def add_federated_args(parser):
    parser.add_argument(
        '--num_communication',
        type=int,
        default=10,
        help='Number of cloud communication rounds.',
    )
    parser.add_argument(
        '--num_local_update',
        type=int,
        default=1,
        help='Number of client local updates (tau_1).',
    )
    parser.add_argument(
        '--num_edge_aggregation',
        type=int,
        default=1,
        help='Number of edge aggregations before cloud aggregation (tau_2).',
    )
    parser.add_argument(
        '--iid',
        type=int,
        default=-2,
        help='Data distribution mode: 1, 0, -1, -2.',
    )
    parser.add_argument(
        '--edgeiid',
        type=int,
        default=0,
        help='Edge distribution mode when iid=-2: 1 edge-iid, 0 edge-non-iid.',
    )
    parser.add_argument(
        '--classes_per_client',
        type=int,
        default=5,
        help='Classes per client under artificial non-iid partitioning.',
    )
    parser.add_argument(
        '--avg',
        type=int,
        default=2,
        help='Aggregation strategy: 0=favg, 1=simavg, 2=lwavg.',
    )
    parser.add_argument(
        '--num_clients',
        type=int,
        default=10,
        help='Total number of clients.',
    )
    parser.add_argument(
        '--num_edges',
        type=int,
        default=1,
        help='Total number of edge servers.',
    )
    parser.add_argument(
        '--frac',
        type=float,
        default=1,
        help='Fraction of clients participating per round.',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=1,
        help='Random seed.',
    )
    parser.add_argument(
        '--gpu',
        type=int,
        default=0,
        help='GPU index to use when CUDA is available.',
    )
    parser.add_argument(
        '--mtl_model',
        type=int,
        default=0,
        help='Use multi-task model layout.',
    )
    parser.add_argument(
        '--global_model',
        type=int,
        default=1,
        help='Use the same global/shared model for all clients.',
    )
    parser.add_argument(
        '--local_model',
        type=int,
        default=0,
        help='Reserved local-model flag.',
    )


def add_privacy_args(parser):
    parser.add_argument(
        '--client_add_noise',
        type=int,
        default=0,
        help='Client-side DP noise switch: 1 enable, 0 disable.',
    )
    parser.add_argument(
        '--client_sepsilon',
        type=float,
        default=3,
        help='DP epsilon for client shared/conv layers.',
    )
    parser.add_argument(
        '--client_depsilon',
        type=float,
        default=5,
        help='DP epsilon for client dense/private layers.',
    )
    parser.add_argument(
        '--client_delta',
        type=float,
        default=1e-9,
        help='DP delta for client-side Gaussian noise.',
    )
    parser.add_argument(
        '--edge_add_noise',
        type=int,
        default=0,
        help='Edge-side DP noise switch: 1 enable, 0 disable.',
    )
    parser.add_argument(
        '--edge_sepsilon',
        type=float,
        default=0.01,
        help='DP epsilon for edge shared/conv layers.',
    )
    parser.add_argument(
        '--edge_depsilon',
        type=float,
        default=0.05,
        help='DP epsilon for edge dense/private layers.',
    )
    parser.add_argument(
        '--edge_delta',
        type=float,
        default=1e-6,
        help='DP delta for edge-side Gaussian noise.',
    )
    parser.add_argument(
        '--DP_SDG',
        type=int,
        default=0,
        help='Enable DP-SGD on clients: 1 enable, 0 disable.',
    )


def build_parser():
    parser = argparse.ArgumentParser(description='Hierarchical federated learning training options.')
    add_dataset_args(parser)
    add_training_args(parser)
    add_federated_args(parser)
    add_privacy_args(parser)
    return parser


def args_parser():
    args = build_parser().parse_args()
    args.cuda = torch.cuda.is_available()
    args.client_shared_epsilon = args.client_sepsilon
    args.client_private_epsilon = args.client_depsilon
    args.edge_shared_epsilon = args.edge_sepsilon
    args.edge_private_epsilon = args.edge_depsilon
    args.dp_sgd = args.DP_SDG
    return args
