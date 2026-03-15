import copy
import torch
from torch import nn

def average_weights(w, s_num):
    #copy the first client's weights
    total_sample_num = sum(s_num)
    temp_sample_num = s_num[0]
    w_avg = copy.deepcopy(w[0])
    # for i in range(len(s_num)):
    #     print(f"客户端{i} :{s_num[i] /total_sample_num}")
    for k in w_avg.keys():  #the nn layer loop
        for i in range(1, len(w)):   #the client loop
            w_avg[k] = w_avg[k].float()#5.6
            w_avg[k] += torch.mul(w[i][k], s_num[i]/temp_sample_num)

        w_avg[k] = torch.mul(w_avg[k], temp_sample_num/total_sample_num)
    return w_avg