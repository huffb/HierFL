import argparse
import torch

def args_parser():
    parser = argparse.ArgumentParser()
    # dataset and model
    parser.add_argument(
        '--dataset',
        type = str,
        default = 'mnist',
        # default='mnist',
        # default='mnist',
        help = 'name of the dataset: mnist, cifar10,fmnist'
    )
    parser.add_argument(
        '--model',
        type = str,
        # default = 'cnn',4.22
        default = 'lenet',
        help='name of model. mnist: logistic, lenet,cnn_3; cifar10: resnet18, cnn_complex'
    )
    parser.add_argument(
        '--input_channels',
        type = int,
        default = 1,
        help = 'input channels. mnist:1, cifar10 :3'
    )
    parser.add_argument(
        '--output_channels',
        type = int,
        default = 10,
        help = 'output channels'
    )
    #nn training hyper parameter
    parser.add_argument(
        '--batch_size',
        type = int,
        default = 20,
        help = 'batch size when trained on client'
        #B
    )
    parser.add_argument(
        '--num_communication',#client-edge-cloud communication times.
        type = int,
        default=10,
        help = 'number of communication rounds with the cloud server'

    )
    parser.add_argument(
        '--num_local_update',#how many times local updata then communicate with edge
        type=int,
        default=1,
        help='number of local update (tau_1)'

    )
    parser.add_argument(
        '--num_edge_aggregation',  #
        type=int,
        default=1,
        help='number of edge aggregation (tau_2)'

    )
    # setting for federeated learning  及仅仅修改这两个没有什么区别啊
    parser.add_argument(
        '--iid',
        type=int,
        default=-2,
        help='distribution of the data, 1,0,-1, -2(one-class)'
    )
    parser.add_argument(
        '--edgeiid',
        type=int,
        default=0,
        help='distribution of the data under edges, 1 (edgeiid),0 (edgeniid) (used only when iid = -2)'
    )
    parser.add_argument(
        '--classes_per_client',
        type=int,
        default=5,
        # 原本为2 修改这个也没啥区别 人工n-idd有点失败
        help='under artificial non-iid distribution, the classes per client'
    )
    parser.add_argument(
        '--avg',#聚合策略
        type=int,
        default=2,
        help='favg:0, simavg:1,lwavg:2'
    )
    parser.add_argument(
        '--num_clients',#客户端数量
        type=int,
        default=10,
        help='number of all available clients'
    )
    parser.add_argument(
        '--num_edges',#边缘服务器数量
        type=int,
        default=1,
        help='number of edges'
    )
    parser.add_argument(
        '--lr',#学习率
        type = float,
        default = 0.1,
        help = 'learning rate of the SGD when trained on client'
    )
    parser.add_argument(
        '--lr_decay',
        type = float,
        default= '0.995',
        help = 'lr decay rate'
    )
    parser.add_argument(
        '--lr_decay_epoch',
        type = int,
        default=1,
        help= 'lr decay epoch'
    )
    parser.add_argument(
        '--momentum',
        type = float,
        default = 0.95,
        help = 'SGD momentum'
    )
    parser.add_argument(
        '--weight_decay',
        type = float,
        default = 0,
        help= 'The weight decay rate'
    )
    parser.add_argument(
        '--verbose',
        type = int,
        default = 0,
        help = 'verbose for print progress bar'
    )

    # 会有影响 但不成正比。。。。
    parser.add_argument(
        '--frac',
        type = float,
        default = 1,
        help = 'fraction of participated clients'
    )

    parser.add_argument(
        '--seed',
        type = int,
        default = 1,
        help = 'random seed (defaul: 1)'
    )
    parser.add_argument(
        '--dataset_root',
        type = str,
        default = 'data',
        help = 'dataset root folder'
    )
    parser.add_argument(
        '--show_dis',
        type= int,
        default= 0,
        help='whether to show distribution'
    )

    parser.add_argument(
        '--gpu',
        type = int,
        default=0,
        help = 'GPU to be selected, 0, 1, 2, 3'
    )

    parser.add_argument(
        '--mtl_model',
        default=0,
        type = int
    )
    parser.add_argument(
        '--global_model',
        default=1,
        type=int
    )
    parser.add_argument(
        '--local_model',
        default=0,
        type=int
    )
    parser.add_argument(
        '--client_add_noise',#差分隐私噪声
        type=int,
        default=0,
        help='Add dp noise: 1 to add noise, 0 to not add noise.'
    )
    parser.add_argument(
        '--client_sepsilon',
        type=float,
        default=3,
        help='Mean (mu) of the Gaussian noise. Defaults to 0.0.'
    )
    parser.add_argument(
        '--client_depsilon', #客户端深层隐私预算
        type=float,
        default=5,
        help='Mean (mu) of the Gaussian noise. Defaults to 0.0.'
    )
    # 添加高斯噪声的标准差参数，默认1.0
    parser.add_argument(
        '--client_delta',
        type=float,
        default=1e-9,
        help='Standard deviation  of the Gaussian noise.'
    )
    parser.add_argument(
        '--edge_add_noise',#边缘噪声添加选择
        type=int,
        default=0,
        help='Add dp noise: 1 to add noise, 0 to not add noise.'
    )
    parser.add_argument(
        '--edge_sepsilon',
        type=float,
        default=0.01,
        help='Mean (mu) of the Gaussian noise. Defaults to 0.0.'
    )
    parser.add_argument(
        '--edge_depsilon',
        type=float,
        default=0.05,
        help='Mean (mu) of the Gaussian noise. Defaults to 0.0.'
    )
    # 添加高斯噪声的标准差参数，默认为1.0
    parser.add_argument(
        '--edge_delta',
        type=float,
        default=1e-6,
        help='Standard deviation  of the Gaussian noise.'
    )
    parser.add_argument(
        '--SGD_clip',
        type=float,
        default=10,
        help='SGD clip.'
    )
    parser.add_argument(
        '--DP_SDG',
        type=int,
        default=0,
        help='if DP_SDG:0,1'
    )
    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    return args
