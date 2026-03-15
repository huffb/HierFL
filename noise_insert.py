import torch
import torch
import math
import numpy as np
from options import args_parser
def add_differential_privacy_noise(w, s_num, epsilon,delta,num_clients):
    """
    为每个客户端的神经网络权重添加差分隐私噪声。

    参数:
    w -- 神经网络权重的列表，每个元素是一个字典，字典的键是层名，值是权重矩阵。
    s_num -- 每个客户端的样本数量列表。
    epsilon -- 差分隐私的隐私预算参数，较小的epsilon提供更强的隐私保护。

    返回:
    带有差分隐私噪声的权重列表。
    """
    noised_w = []  # 存储添加了噪声的权重
    total_sample_num = sum(s_num)  # 所有客户端样本的总数
    # delta = 1e-6
    # 计算每个权重的标准差，
    # delta = 1 / (2 * epsilon * total_sample_num)
    # std_dev1 = (math.sqrt(2 * math.log(1.25 / delta))) / (epsilon * math.sqrt(total_sample_num))#0.2217-0.49
    K = 100
    std_dev0 = (math.sqrt(2 * math.log(1.25 / delta))) / (epsilon * total_sample_num /100)
    std_dev1 = (math.sqrt(2 * math.log(1.25 / delta))) / (epsilon * total_sample_num /30)
    std_dev2 = (math.sqrt(2 * math.log(1.25 / delta))) / (epsilon * total_sample_num/30)
    # 生成高斯噪声
    f = 0.9
    for i, client_w in enumerate(w):
        noised_client_w = {}  # 当前客户端的带有噪声的权重
        for k, weights in client_w.items():#k :each loop
            #将噪声添加到权重上
            if(k == 'conv1.weight' or k =='conv1.bias'):
                # flat = weights.view(-1)
                # frac = int(f * weights.numel())
                # noise0 = torch.randn(flat.size()) * std_dev0
                # w1 = flat[:frac]+noise0[:frac]
                # noise = torch.randn(flat.size()) * std_dev2
                # w2 = flat[frac:]+noise[frac:]
                # noised_weights = torch.cat((w1,w2),dim=0)
                # noised_weights = noised_weights.view(weights.size())
                noise = torch.randn(weights.size()) * std_dev0
                noised_weights = weights + noise
            else:
                noise = torch.randn(weights.size()) * std_dev2
                noised_weights = weights + noise
                # noised_weights = weights
            # noise = torch.randn(weights.size()) * std_dev1
            # noised_weights = weights + noise
            # 更新带有噪声的权重
            noised_client_w[k] = noised_weights
        # noised_w.append(noised_client_w)
        w[i] = noised_client_w
    return w
