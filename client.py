import copy

import torch
from Gaussian_Add import Model_Noise_Add
from models.initialize_model import initialize_model
from torch.autograd import Variable


class Client:

    def __init__(self, id, train_loader, test_loader, args, device):
        self.id = id
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.model = initialize_model(args, device)
        self.receiver_buffer = {}
        self.batch_size = args.batch_size
        self.global_model = copy.deepcopy(self.model.shared_layers.state_dict())
        self.epoch = 0
        self.clock = []

    def local_update(self, num_iter, device, args):
        itered_num = 0
        total_loss = 0.0
        end = False
        num_clip = args.SGD_clip
        dataset_size = len(self.train_loader.dataset)

        for _ in range(1000):
            for data in self.train_loader:
                inputs, labels = data
                inputs = Variable(inputs).to(device)
                labels = Variable(labels).to(device)
                batch_loss = self.model.optimize_model(
                    input_batch=inputs,
                    label_batch=labels,
                    clip=num_clip,
                    args=args,
                    dataset_size=dataset_size,
                )
                total_loss += batch_loss
                itered_num += 1
                if itered_num >= num_iter:
                    end = True
                    self.epoch += 1
                    self.model.exp_lr_sheduler(epoch=self.epoch)
                    break

            if end:
                break

            self.epoch += 1
            self.model.exp_lr_sheduler(epoch=self.epoch)
            self.model.print_current_lr()

        if args.client_add_noise == 1:
            lr = args.lr
            clip = num_clip
            sensitivity = 2 * lr * clip / dataset_size
            state_dict = self.model.shared_layers.state_dict()
            n = args.num_communication * args.num_edge_aggregation
            Model_Noise_Add(
                delta=args.client_delta,
                sepsilon=args.client_shared_epsilon / n,
                depsilon=args.client_private_epsilon / n,
                model=args.model,
                w=state_dict.items(),
                sensitivity=sensitivity,
            )

        return total_loss / max(itered_num, 1)

    def test_model(self, device):
        correct = 0.0
        total = 0.0
        with torch.no_grad():
            for data in self.test_loader:
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = self.model.test_model(input_batch=inputs)
                _, predict = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predict == labels).sum().item()
        return correct, total

    def send_to_edgeserver(self, edgeserver):
        edgeserver.receive_from_client(
            client_id=self.id,
            cshared_state_dict=copy.deepcopy(self.model.shared_layers.state_dict()),
        )
        return None

    def receive_from_edgeserver(self, shared_state_dict):
        self.receiver_buffer = shared_state_dict
        self.global_model = copy.deepcopy(shared_state_dict)
        return None

    def sync_with_edgeserver(self):
        self.model.update_model(self.receiver_buffer)
        return None
