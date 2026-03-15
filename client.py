# The structure of the client
# Should include following funcitons
# 1. Client intialization, dataloaders, model(include optimizer)
# 2. Client model update
# 3. Client send updates to server
# 4. Client receives updates from server
# 5. Client modify local model based on the feedback from the server
from torch.autograd import Variable
import torch
from models.initialize_model import initialize_model
import copy
from Gaussian_Add import Model_Noise_Add

class Client():

    def __init__(self, id, train_loader, test_loader, args, device):
        self.id = id
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.model = initialize_model(args, device)
        # copy.deepcopy(self.model.shared_layers.state_dict())
        self.receiver_buffer = {}
        self.batch_size = args.batch_size
        self.global_model = copy.deepcopy(self.model.shared_layers.state_dict())
        #record local update epoch
        self.epoch = 0
        # record the time
        self.clock = []

    def local_update(self, num_iter, device, args):
        itered_num = 0
        loss = 0.0
        end = False
        num_clip = args.SGD_clip
        dataset_size = len(self.train_loader.dataset)
        # mu = 0.1
        # prev_avg_loss = float('inf')
        # the upperbound selected in the following is because it is expected that one local update will never reach 1000
        for epoch in range(1000):
            for data in self.train_loader:
                inputs, labels = data
                inputs = Variable(inputs).to(device)
                labels = Variable(labels).to(device)
                loss += self.model.optimize_model(input_batch=inputs,
                                                  label_batch=labels,clip=num_clip,
                                                  args=args,dataset_size=dataset_size
                                                  )
                # proximal_term = 0.0
                # for name, param in self.model.shared_layers.named_parameters():
                #     global_param = self.global_model[name]
                #     proximal_term += (mu / 2) * torch.norm(param - global_param) ** 2
                # 总的损失为正常损失加上近端项
                # loss += proximal_term
                itered_num += 1
                loss /= num_iter
                if itered_num >= num_iter:
                    end = True
                    # print(f"Iterer number {itered_num}")
                    self.epoch += 1
                    self.model.exp_lr_sheduler(epoch=self.epoch)
                    # self.model.print_current_lr()
                    break
            if end: break
            self.epoch += 1
            # loss_decrease_rate = (prev_avg_loss - loss)
            # self.model.Loss_lr_sheduler(loss = loss_decrease_rate)
            # prev_avg_loss = loss
            # if loss_decrease_rate < 0.01:  # 举例：如果损失下降速率小于1%，则降低学习率
            #     current_lr = self.model.get_current_lr()  # 假设模型中有获取当前学习率的方法
            #     new_lr = current_lr * 0.5  # 例如将学习率减半
            #     self.model.set_lr(new_lr)  # 假设模型中有设置学习率的方法
            # prev_avg_loss = loss  # 更新上一个epoch的平均损失
            self.model.exp_lr_sheduler(epoch = self.epoch)
            self.model.print_current_lr()
        # print(itered_num)
        # print(f'The {self.epoch}')
        if args.client_add_noise ==1:
            print("3333333333333333333333")
            lr = args.lr
            clip = num_clip
            dataset_size = len(self.train_loader.dataset)
            sensitivity = 2 * lr * clip / dataset_size
            state_dict = self.model.shared_layers.state_dict()
            n = args.num_communication * args.num_edge_aggregation
            Model_Noise_Add(delta=args.client_delta, sepsilon=args.client_sepsilon / n,
                            depsilon=args.client_depsilon / n,
                            model=args.model, w=state_dict.items(), sensitivity=sensitivity)
        # self.model.shared_layers.load_state_dict(w)
        # print(self.model.shared_layers.state_dict().size())
        # for name, param in self.model.shared_layers.state_dict().items():
        #     print(f"Layer Name: {name}")
        # print(self.model.shared_layers.state_dict().items())

        return loss

    def test_model(self, device):
        correct = 0.0
        total = 0.0
        with torch.no_grad():
            for data in self.test_loader:
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = self.model.test_model(input_batch= inputs)
                _, predict = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predict == labels).sum().item()
        return correct, total

    def send_to_edgeserver(self, edgeserver):
        edgeserver.receive_from_client(client_id= self.id,
                                        cshared_state_dict = copy.deepcopy(self.model.shared_layers.state_dict())
                                        )
        return None


    def receive_from_edgeserver(self, shared_state_dict):
        self.receiver_buffer = shared_state_dict
        self.global_model = copy.deepcopy(shared_state_dict)
        return None

    def sync_with_edgeserver(self):
        """
        The global has already been stored in the buffer
        :return: None
        """
        # self.model.shared_layers.load_state_dict(self.receiver_buffer)
        self.model.update_model(self.receiver_buffer)
        return None

