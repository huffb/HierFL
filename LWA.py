# import copy
# import torch
# from torch.nn.utils import parameters_to_vector
# from average import average_weights
#
# def layer_cosine_similarity(layer1, layer2):
#     """
#     计算两个模型层之间的余弦相似度
#     """
#     vec1 = layer1.view(-1)
#     vec2 = layer2.view(-1)
#     dot_product = torch.dot(vec1, vec2)
#     norm1 = torch.norm(vec1)
#     norm2 = torch.norm(vec2)
#     similarity = dot_product / (norm1 * norm2)
#     return similarity
#
# def non_linear_mapping(similarity, alpha=5.0):
#     return alpha * (1 - torch.exp(-torch.exp(-alpha * (similarity - 1))))
#
# def calculate_layer_weights(sample_num, layer_similarities):
#     weights = []
#     exp_similarities = [torch.exp(similarity) for similarity in layer_similarities]
#     denominator = sum([sample * exp_similarity for sample, exp_similarity in zip(sample_num, exp_similarities)])
#     for sample, exp_similarity in zip(sample_num, exp_similarities):
#         weights.append(sample * exp_similarity / denominator)
#     return weights
#
# def similarity_layer_weighted_aggregation(received_dict, previous_state_dict, sample_num):
#     """
#     根据与 previous_state_dict 的相似度进行加权聚合
#     """
#     alpha = 10.0
#     aggregated_dict = copy.deepcopy(received_dict[0])
#     if not previous_state_dict:
#         # 如果 previous_state_dict 为空，简单平均聚合
#         averaged_dict = average_weights(w=received_dict,
#                                         s_num=sample_num)
#         return averaged_dict
#     else:
#         for key in aggregated_dict.keys():
#             # 计算每一层的相似度
#             layer_similarities = []
#             for i in range(len(received_dict)):
#                 similarity = layer_cosine_similarity(received_dict[i][key], previous_state_dict[key])
#                 similarity = non_linear_mapping(similarity, alpha)
#                 layer_similarities.append(similarity)
#
#             # 计算每一层的权重
#             layer_weights = calculate_layer_weights(sample_num, layer_similarities)
#
#             # 按层加权聚合
#             aggregated_dict[key] = layer_weights[0] * received_dict[0][key].float()
#             for i in range(1, len(received_dict)):
#                 aggregated_dict[key] += layer_weights[i] * received_dict[i][key]
#
#         return aggregated_dict
#
# # def average_weights(w, s_num):
# #     #copy the first client's weights
# #     total_sample_num = sum(s_num)
# #     temp_sample_num = s_num[0]
# #     w_avg = copy.deepcopy(w[0])
# #     for k in w_avg.keys():  #the nn layer loop
# #         for i in range(1, len(w)):   #the client loop
# #             w_avg[k] = w_avg[k].float()#5.6
# #             w_avg[k] += torch.mul(w[i][k], s_num[i]/temp_sample_num)
# #         w_avg[k] = torch.mul(w_avg[k], temp_sample_num/total_sample_num)
# #     return w_avg
import copy
import torch
from torch.nn.utils import parameters_to_vector
from average import average_weights
import torch.nn.functional as F


def calculate_layer_gradient(current_layer, previous_layer):
    """计算单层参数的梯度（当前参数 - 上一轮参数）"""
    return current_layer - previous_layer


def layer_gradient_similarity(grad1, grad2):
    """计算两个梯度向量的余弦相似度"""
    vec1 = grad1.view(-1)
    vec2 = grad2.view(-1)
    dot_product = torch.dot(vec1, vec2)
    norm1 = torch.norm(vec1)
    norm2 = torch.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return torch.tensor(0.0)  # 处理零向量情况
    sim = dot_product / (norm1 * norm2)
    sim = torch.clamp(sim, -1, 1)
    # angle = torch.arccos(sim)
    # return angle
    return dot_product / (norm1 * norm2)


def non_linear_mapping(similarity, alpha=5.0):
    """非线性映射函数增强相似度区分度"""
    return alpha * (1 - torch.exp(-torch.exp(-alpha * (similarity - 1))))
def calculate_layer_weights(sample_num, layer_similarities):
    """根据相似度和数据量计算归一化权重"""
    total_samples = sum(sample_num)
    # weighted_sims = [s * n for s, n in zip(layer_similarities, sample_num)]
    # denominator = sum(weighted_sims)
    # if denominator == 0:
    #     return [1 / len(sample_num)] * len(sample_num)  # 均匀分布保底
    similarities_tensor = torch.tensor(layer_similarities, dtype=torch.float32)
    weights = F.softmax(similarities_tensor, dim=0)
    return weights.tolist()
    # return [s * n / denominator for s, n in zip(layer_similarities, sample_num)]


def similarity_layer_weighted_aggregation(received_dict, previous_state_dict, sample_num):
    """
    基于梯度相似度的层次自适应加权聚合
    Args:
        received_dict: 客户端模型参数列表 [client1_dict, client2_dict, ...]
        previous_state_dict: 上一轮全局模型参数
        sample_num: 客户端数据量列表
    """
    aggregated_dict = copy.deepcopy(received_dict[0])
    if not previous_state_dict:
        return average_weights(received_dict, sample_num)

    # 计算全局梯度参考
    global_gradients = {}
    for key in previous_state_dict:
        global_grad = torch.zeros_like(previous_state_dict[key])
        total_samples = sum(sample_num)
        for i, client_params in enumerate(received_dict):
            client_grad = calculate_layer_gradient(client_params[key], previous_state_dict[key])
            global_grad += client_grad * (sample_num[i] / total_samples)
        global_gradients[key] = global_grad

    # 逐层计算相似度并加权聚合
    for key in aggregated_dict.keys():
        layer_similarities = []
        client_gradients = []
        layer_sim = []

        # 计算每个客户端的梯度
        for i in range(len(received_dict)):
            client_grad = calculate_layer_gradient(received_dict[i][key], previous_state_dict[key])
            client_gradients.append(client_grad)

            # 计算与全局梯度的相似度
            sim = layer_gradient_similarity(client_grad, global_gradients[key])
            layer_sim.append(sim)
            sim = non_linear_mapping(sim, alpha=10.0)  # 增强对比
            layer_similarities.append(sim)

        # 计算归一化权重
        layer_weights = calculate_layer_weights(sample_num, layer_similarities)

        # 打印层次名称、客户端的权重和相似度
        # print(f"层次名称: {key}")
        # print("客户端的权重:", end=" ")
        # for i, weight in enumerate(layer_weights):
        #     print(f"客户端{i + 1}的权重: {weight}", end=" ")
        # print("---------------------------------------")
        # print("客户端的余弦相似度:", end=" ")
        # for i, sim in enumerate(layer_sim):
        #     print(f"客户端{i + 1}的相似度: {sim}", end=" ")
        # print("---------------------------------------")
        # print("客户端的相似度放大后:", end=" ")
        # for i, sim in enumerate(layer_similarities):
        #     print(f"客户端{i + 1}的相似度: {sim}", end=" ")
        # print("---------------------------------------")

        # 加权聚合
        aggregated_dict[key] = torch.zeros_like(previous_state_dict[key])
        for i in range(len(received_dict)):
            aggregated_dict[key] += client_gradients[i] * layer_weights[i]

        # 更新为新的全局参数：上一轮参数 + 聚合梯度
        aggregated_dict[key] = previous_state_dict[key] + aggregated_dict[key]

    return aggregated_dict