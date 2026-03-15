# The structure of the edge server
# THe edge should include following funcitons
# 1. Server initialization
# 2. Server receives updates from the client
# 3. Server sends the aggregated information back to clients
# 4. Server sends the updates to the cloud server
# 5. Server receives the aggregated information from the cloud server

import copy
from average import average_weights
from noise_insert import add_differential_privacy_noise
from Gaussian_Add import Model_Noise_Add
from SimAvg import similarity_weighted_aggregation
from LWA import similarity_layer_weighted_aggregation
class Edge():

    def __init__(self, id, cids, shared_layers):
        """
        id: edge id
        cids: ids of the clients under this edge
        receiver_buffer: buffer for the received updates from selected clients
        shared_state_dict: state dict for shared network
        id_registration: participated clients in this round of traning
        sample_registration: number of samples of the participated clients in this round of training
        all_trainsample_num: the training samples for all the clients under this edge
        shared_state_dict: the dictionary of the shared state dict
        clock: record the time after each aggregation
        :param id: Index of the edge
        :param cids: Indexes of all the clients under this edge
        :param shared_layers: Structure of the shared layers
        :return:
        """
        self.id = id
        self.cids = cids
        self.receiver_buffer = {}
        self.shared_state_dict = {}
        self.id_registration = []
        self.sample_registration = {}
        self.all_trainsample_num = 0
        self.shared_state_dict = shared_layers.state_dict()
        self.noised_state_dict = []
        self.previous_state_dict = [] #2.19 存储上一轮中的模型
        self.clock = []

    def refresh_edgeserver(self):
        self.receiver_buffer.clear()
        del self.id_registration[:]
        self.sample_registration.clear()
        return None

    def client_register(self, client):
        self.id_registration.append(client.id)
        self.sample_registration[client.id] = len(client.train_loader.dataset)
        return None

    def receive_from_client(self, client_id, cshared_state_dict):
        self.receiver_buffer[client_id] = cshared_state_dict
        return None

    def addnoise(self,args):
        received_dict = [dict for dict in self.receiver_buffer.values()]
        sample_num = [snum for snum in self.sample_registration.values()]
        self.noised_state_dict = add_differential_privacy_noise(w=received_dict,
                                                s_num=sample_num,epsilon=args.client_epsilon,delta=args.client_delta,
                                                num_clients=sample_num)
    def aggregate(self, args):
        """
        Using the old aggregation funciton
        :param args:
        :return:
        """
        received_dict = [dict for dict in self.receiver_buffer.values()]
        sample_num = [snum for snum in self.sample_registration.values()]
        if args.avg == 0:
            self.shared_state_dict = average_weights(w=received_dict,
                                                 s_num=sample_num)
        elif args.avg == 1:
            # print(self.previous_state_dict)

            self.shared_state_dict = similarity_weighted_aggregation(received_dict=received_dict,
                                                                     previous_state_dict=self.previous_state_dict,
                                                                     sample_num=sample_num
                                                                )
        elif args.avg == 2:
            self.shared_state_dict = similarity_layer_weighted_aggregation(received_dict=received_dict,
                                                                     previous_state_dict=self.previous_state_dict,
                                                                     sample_num=sample_num)
        self.previous_state_dict = copy.deepcopy(self.shared_state_dict) #存储本轮的模型作为下一轮中的聚合参数


    def send_to_client(self, client):
        client.receive_from_edgeserver(copy.deepcopy(self.shared_state_dict))
        return None

    def send_to_cloudserver(self, cloud,args):
        if args.client_add_noise == 1:
            print("222222222222222222222")
            sample_num = [snum for snum in self.sample_registration.values()]
            lr = args.lr
            clip = args.SGD_clip
            dataset_size = sum(sample_num)
            sensitivity = 2 * lr * clip / dataset_size
            state_dict = self.shared_state_dict
            n = args.num_edge_aggregation
            Model_Noise_Add(delta=args.client_delta, sepsilon=args.edge_sepsilon / n,
                            depsilon=args.edge_depsilon / n,
                            model=args.model, w=state_dict.items(), sensitivity=sensitivity)
        cloud.receive_from_edge(edge_id=self.id,
                                eshared_state_dict=copy.deepcopy(
                                    self.shared_state_dict))
        return None

    def receive_from_cloudserver(self, shared_state_dict):
        self.shared_state_dict = shared_state_dict
        return None

