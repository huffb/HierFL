# The structure of the edge server
# 1. Server initialization
# 2. Server receives updates from the client
# 3. Server sends the aggregated information back to clients
# 4. Server sends the updates to the cloud server
# 5. Server receives the aggregated information from the cloud server

import copy

from average import average_weights
from Gaussian_Add import Model_Noise_Add
from LWA import similarity_layer_weighted_aggregation
from SimAvg import similarity_weighted_aggregation
from noise_insert import add_differential_privacy_noise


class Edge:
    def __init__(self, id, cids, shared_layers):
        self.id = id
        self.cids = cids
        self.receiver_buffer = {}
        self.shared_state_dict = shared_layers.state_dict()
        self.id_registration = []
        self.sample_registration = {}
        self.all_trainsample_num = 0
        self.noised_state_dict = []
        self.previous_state_dict = []
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

    def addnoise(self, args):
        received_dict = [state_dict for state_dict in self.receiver_buffer.values()]
        sample_num = [sample for sample in self.sample_registration.values()]
        self.noised_state_dict = add_differential_privacy_noise(
            w=received_dict,
            s_num=sample_num,
            epsilon=args.client_private_epsilon,
            delta=args.client_delta,
            num_clients=sample_num,
        )

    def aggregate(self, args):
        received_dict = [state_dict for state_dict in self.receiver_buffer.values()]
        sample_num = [sample for sample in self.sample_registration.values()]
        if args.avg == 0:
            self.shared_state_dict = average_weights(w=received_dict, s_num=sample_num)
        elif args.avg == 1:
            self.shared_state_dict = similarity_weighted_aggregation(
                received_dict=received_dict,
                previous_state_dict=self.previous_state_dict,
                sample_num=sample_num,
            )
        elif args.avg == 2:
            self.shared_state_dict = similarity_layer_weighted_aggregation(
                received_dict=received_dict,
                previous_state_dict=self.previous_state_dict,
                sample_num=sample_num,
            )
        self.previous_state_dict = copy.deepcopy(self.shared_state_dict)

    def send_to_client(self, client):
        client.receive_from_edgeserver(copy.deepcopy(self.shared_state_dict))
        return None

    def send_to_cloudserver(self, cloud, args):
        if args.edge_add_noise == 1:
            print("222222222222222222222")
            sample_num = [sample for sample in self.sample_registration.values()]
            dataset_size = sum(sample_num)
            sensitivity = 2 * args.lr * args.SGD_clip / dataset_size
            Model_Noise_Add(
                delta=args.edge_delta,
                sepsilon=args.edge_shared_epsilon / args.num_edge_aggregation,
                depsilon=args.edge_private_epsilon / args.num_edge_aggregation,
                model=args.model,
                w=self.shared_state_dict.items(),
                sensitivity=sensitivity,
            )
        cloud.receive_from_edge(
            edge_id=self.id,
            eshared_state_dict=copy.deepcopy(self.shared_state_dict),
        )
        return None

    def receive_from_cloudserver(self, shared_state_dict):
        self.shared_state_dict = shared_state_dict
        return None
