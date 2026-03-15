import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np


def visualize_mnist():
    # 定义数据预处理（这里仅转换为张量）
    transform = transforms.Compose([transforms.ToTensor()])

    # 加载MNIST数据集（假设数据已下载在指定路径）
    mnist_trainset = torchvision.datasets.MNIST(root='D:/work/hfl/HierFL-master/data/mnist', train=True,
                                                download=False, transform=transform)
    mnist_trainloader = torch.utils.data.DataLoader(mnist_trainset, batch_size=100,
                                                    shuffle=True, num_workers=2)

    # 获取一个批次的数据
    dataiter = iter(mnist_trainloader)
    images, labels = next(dataiter)

    # 调整图像尺寸以适应展示
    images = images.view(100, 1, 28, 28)

    # 创建一个图像网格
    grid = torchvision.utils.make_grid(images, nrow=10)

    # 反归一化（如果有归一化操作）
    grid = grid * 0.5 + 0.5
    npimg = grid.numpy()

    # 显示并保存图像
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.axis('off')
    plt.savefig('fmnist_visualization.png',dpi=300)
    plt.show()


if __name__ == '__main__':
    # 如果代码会被冻结（打包成可执行文件），需要这一行
    # from multiprocessing import freeze_support
    # freeze_support()
    visualize_mnist()
#
# import torch
# import torchvision
# import torchvision.transforms as transforms
# import matplotlib.pyplot as plt
# import numpy as np
#
#
# def visualize_cifar10():
#     # 定义数据预处理（这里仅转换为张量）
#     transform = transforms.Compose([transforms.ToTensor()])
#
#     # 加载CIFAR - 10数据集（假设数据已下载在指定路径）
#     cifar_trainset = torchvision.datasets.CIFAR10(root='D:/work/hfl/HierFL-master/data/cifar10', train=True,
#                                                  download=False, transform=transform)
#     cifar_trainloader = torch.utils.data.DataLoader(cifar_trainset, batch_size=100,
#                                                     shuffle=True, num_workers=2)
#
#     # 获取一个批次的数据
#     dataiter = iter(cifar_trainloader)
#     images, labels = next(dataiter)
#
#     # 创建一个图像网格
#     grid = torchvision.utils.make_grid(images, nrow=10)
#
#     # 反归一化（如果有归一化操作）
#     grid = grid / 2 + 0.5
#     npimg = grid.numpy()
#
#     # 显示并保存图像
#     plt.imshow(np.transpose(npimg, (1, 2, 0)))
#     plt.axis('off')
#     plt.savefig('cifar10_visualization.png')
#     plt.show()
#
#
# if __name__ == '__main__':
#     # 如果代码会被冻结（打包成可执行文件），需要这一行
#     # from multiprocessing import freeze_support
#     # freeze_support()
#     visualize_cifar10()
# import torch
# import torchvision
# import torchvision.transforms as transforms
# import matplotlib.pyplot as plt
# import numpy as np
#
#
# def visualize_fmnist():
#     # 定义数据预处理（这里仅转换为张量）
#     transform = transforms.Compose([transforms.ToTensor()])
#
#     # 加载FashionMNIST数据集（假设数据已下载在指定路径）
#     fmnist_trainset = torchvision.datasets.FashionMNIST(root='D:/work/hfl/HierFL-master/data/fmnist', train=True,
#                                                         download=False, transform=transform)
#     fmnist_trainloader = torch.utils.data.DataLoader(fmnist_trainset, batch_size=100,
#                                                      shuffle=True, num_workers=2)
#
#     # 获取一个批次的数据
#     dataiter = iter(fmnist_trainloader)
#     images, labels = next(dataiter)
#
#     # 调整图像尺寸以适应展示（FashionMNIST图像和MNIST一样是单通道28x28）
#     images = images.view(100, 1, 28, 28)
#
#     # 创建一个图像网格
#     grid = torchvision.utils.make_grid(images, nrow=10)
#
#     # 反归一化（如果有归一化操作）
#     grid = grid / 2 + 0.5
#     npimg = grid.numpy()
#
#     # 显示并保存图像
#     plt.imshow(np.transpose(npimg, (1, 2, 0)))
#     plt.axis('off')
#     plt.savefig('fmnist_visualization.png')
#     plt.show()
#
#
# if __name__ == '__main__':
#     # 如果代码会被冻结（打包成可执行文件），需要这一行
#     # from multiprocessing import freeze_support
#     # freeze_support()
#     visualize_fmnist()