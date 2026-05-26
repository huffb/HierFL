# The structure of the server
# 1. Server initialization
# 2. Server receives updates from edges
# 3. Server sends the aggregated information back to edges

import copy

from average import average_weights
from LWA import similarity_layer_weighted_aggregation
from SimAvg import similarity_weighted_aggregation
from noise_insert import add_differential_privacy_noise


class Cloud:
    def __init__(self, shared_layers):
        self.receiver_buffer = {}
        self.shared_state_dict = {}
        self.id_registration = []
        self.sample_registration = {}
        self.clock = []
        self.noised_state_dict = []
        self.previous_state_dict = None

    def refresh_cloudserver(self):
        self.receiver_buffer.clear()
        del self.id_registration[:]
        self.sample_registration.clear()
        return None

    def edge_register(self, edge):
        self.id_registration.append(edge.id)
        self.sample_registration[edge.id] = edge.all_trainsample_num
        return None

    def receive_from_edge(self, edge_id, eshared_state_dict):
        self.receiver_buffer[edge_id] = eshared_state_dict
        return None

    def addnoise(self, args):
        received_dict = [state_dict for state_dict in self.receiver_buffer.values()]
        sample_num = [sample for sample in self.sample_registration.values()]
        self.noised_state_dict = add_differential_privacy_noise(
            w=received_dict,
            s_num=sample_num,
            epsilon=args.edge_private_epsilon,
            delta=args.edge_delta,
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
        return None

    def send_to_edge(self, edge):
        edge.receive_from_cloudserver(copy.deepcopy(self.shared_state_dict))
        return None
