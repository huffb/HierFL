import copy
import torch
from torch.nn.utils import parameters_to_vector
from average import average_weights
import numpy as np
import torch.nn.functional as F

def l2_distance(model1, model2):
    vec1 = parameters_to_vector(model1.values())
    vec2 = parameters_to_vector(model2.values())
    return torch.norm(vec1 - vec2)


def cosine_similarity(model1, model2):
    vec1 = parameters_to_vector(model1.values())
    vec2 = parameters_to_vector(model2.values())
    return torch.dot(vec1, vec2) / (torch.norm(vec1) * torch.norm(vec2) + 1e-8)


def non_linear_mapping(similarity, alpha):
    return alpha * (1 - torch.exp(-torch.exp(-alpha * (similarity - 1))))


def calculate_weights(sample_num, similarities):
    exp_similarities = [torch.exp(sim) for sim in similarities]
    denominator = sum([s * es for s, es in zip(sample_num, exp_similarities)])
    return [s * es / denominator for s, es in zip(sample_num, exp_similarities)]
    # similarities_tensor = torch.tensor(similarities, dtype=torch.float32)
    # weights = F.softmax(similarities_tensor, dim=0)
    return weights.tolist()


def compute_client_gradients(client_models, global_model):
    """计算每个客户端的梯度（当前模型参数 - 全局模型参数）"""
    gradients = []
    for model in client_models:
        grad = {}
        for k in model:
            grad[k] = model[k] - global_model[k]
        gradients.append(grad)
    return gradients


def compute_global_gradient(client_gradients, sample_num):
    """计算全局梯度（加权平均所有客户端梯度）"""
    return average_weights(client_gradients, sample_num)


def similarity_weighted_aggregation(received_dict, previous_state_dict, sample_num):
    alpha = 15.0
    if not previous_state_dict:
        return average_weights(w=received_dict, s_num=sample_num)

    # 1. 计算客户端梯度
    client_gradients = compute_client_gradients(received_dict, previous_state_dict)

    # 2. 计算全局梯度
    global_gradient = compute_global_gradient(client_gradients, sample_num)

    # 3. 计算梯度相似度
    similarities = []
    for grad in client_gradients:
        sim = cosine_similarity(grad, global_gradient)
        sim = non_linear_mapping(sim.clamp(min=0.0), alpha)  # 限制相似度非负
        similarities.append(sim)

    # 4. 计算聚合权重
    weights = calculate_weights(sample_num, similarities)

    print("每个客户端的权重:")
    for i, weight in enumerate(weights):
        print(f"客户端 {i + 1}: {weight.item()}")

    # 5. 梯度加权聚合
    aggregated_grad = copy.deepcopy(client_gradients[0])
    for key in aggregated_grad:
        aggregated_grad[key] = torch.zeros_like(aggregated_grad[key])
        for i in range(len(client_gradients)):
            aggregated_grad[key] += weights[i] * client_gradients[i][key]

    # 6. 更新全局模型：global_{t+1} = global_t + aggregated_grad
    new_global_model = copy.deepcopy(previous_state_dict)
    for key in new_global_model:
        new_global_model[key] += aggregated_grad[key]

    return new_global_model