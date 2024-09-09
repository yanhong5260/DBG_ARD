import os.path as osp
import sys
sys.path.insert(0, osp.join(osp.dirname(osp.abspath(__file__)), '../'))
# from pyexpat import model



# from DataSet.myCIFAR10 import CIFAR10
# from DataSet.myMNIST import MNIST
# from DataSet.myCIFAR100 import CIFAR100

import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.backends.cudnn as cudnn
import torch
import torch.utils.data as Data
import numpy as np
from utils.loggerUtils import logger

# 加载模型
from Model.LeNet5 import LeNet5 as LeNet
# from Model.LeNet import Net
from Model.ResNet import Model as ResNet
from Model.VGG import VGG16
from Model.GoogLeNet import Model as GoogLeNet

from Model.ResNetAll import ResNet18
from Model.ResNetAll import ResNet50
from Model.resnet_KD import resnet18 as ResNet18_KD
# from Model.ResNetAll_One import ResNet18_One


DataSetPath = "/home/ubuntu/datas/code/deep/DataSetRoot"

def get_model(model_name,dataset_name):
    input_channel = 3
    num_classes = 10
    if dataset_name == 'MNIST' or dataset_name == 'MYMNIST':  # (1,28,28)
            input_channel = 1
            num_classes = 10
    elif dataset_name == 'FashionMNIST': # (1,28,28)
            input_channel = 1
            num_classes = 10
    elif dataset_name == 'SVHN': # (3,32,32)
            input_channel = 3
            num_classes = 10
    elif dataset_name == 'CIFAR10' or dataset_name == 'MYCIFAR10': # (3,32,32)
            input_channel = 3
            num_classes = 10
    elif dataset_name == 'CIFAR100' or dataset_name == 'MYCIFAR100': # (3,32,32)
            input_channel = 3
            num_classes = 100



    if model_name == 'LeNet':
        model =LeNet(input_channel = input_channel,num_classes = num_classes)
    # elif model_name == 'Net':
    #     model =Net(input_channel = input_channel,num_classes = num_classes)
    elif model_name == 'ResNet':
        model =ResNet(input_channel = input_channel,num_classes = num_classes)
    elif model_name == 'ResNetKD':
        model =ResNet18_KD(input_channel = input_channel,num_classes = num_classes)
    elif model_name == 'ResNet18':
        model =ResNet18(input_channel = input_channel,num_classes = num_classes)
    elif model_name == 'ResNet50':
        model =ResNet50(input_channel = input_channel,num_classes = num_classes)
    elif model_name == 'VGG':
        model =VGG16(input_channel = input_channel,num_classes = num_classes)
    elif model_name == 'GoogLeNet':
        model =GoogLeNet(input_channel = input_channel,num_classes = num_classes)
    
    return model





def get_dataset(dataset_name,batch_size = 64,num_workers=0):
    if dataset_name == 'MNIST':
        return get_MNIST(batch_size,num_workers)
    # elif dataset_name == 'MYMNIST':
    #     return get_MYMNIST(batch_size,num_workers)
    elif dataset_name == 'FashionMNIST':
        return get_FashionMNIST(batch_size,num_workers)
    elif dataset_name == 'SVHN':
        return get_SVHN(batch_size,num_workers)
    # elif dataset_name == 'MYCIFAR10':
    #     return get_mycifar(batch_size,num_workers)
    elif dataset_name == 'CIFAR10':
        return get_cifar(batch_size,num_workers)
    elif dataset_name == 'CIFAR100':
        return get_cifar100(batch_size,num_workers)
    # elif dataset_name == 'MYCIFAR100':
    #     return get_mycifar100(batch_size,num_workers)





def get_MNIST(batch_size,num_workers):
    # 数据集加载
    cudnn.benchmark = True
    transform = transforms.Compose(
        [transforms.ToTensor(),  # 将PILImage转换为张量 # 将PILImage转换为张量
        #  transforms.Normalize((0.1307,), (0.3081,))
        ]  # 将[0, 1]归一化到[-1, 1]
    )
    train_data = datasets.MNIST(root=f'{DataSetPath}/MNIST', train=True,
                                        download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
                                            shuffle=True, num_workers=0)

    test_data = datasets.MNIST(root=f'{DataSetPath}/MNIST', train=False,
                                        download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size,
                                            shuffle=False, num_workers=0)

    return train_data, test_data, train_loader, test_loader


# def get_MYMNIST(batch_size,num_workers):
#     # 数据集加载
#     cudnn.benchmark = True
#     transform = transforms.Compose(
#         [transforms.ToTensor(),  # 将PILImage转换为张量 # 将PILImage转换为张量
#         #  transforms.Normalize((0.1307,), (0.3081,))
#         ]  # 将[0, 1]归一化到[-1, 1]
#     )
#     train_data = MNIST(root='f'{DataSetPath}/MNIST', train=True,
#                                         download=True, transform=transform)
#     train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
#                                             shuffle=True, num_workers=0)

#     test_data = MNIST(root=f'{DataSetPath}/MNIST', train=False,
#                                         download=True, transform=transform)
#     test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size,
#                                             shuffle=False, num_workers=0)

#     return train_data, test_data, train_loader, test_loader


def get_FashionMNIST(batch_size,num_workers):
    # 数据集加载
    cudnn.benchmark = True
    transform = transforms.Compose(
        [transforms.ToTensor(),  # 将PILImage转换为张量 # 将PILImage转换为张量
        ]  # 将[0, 1]归一化到[-1, 1]
    )
    train_data = datasets.FashionMNIST(root=f'{DataSetPath}/FashionMNIST', train=True,
                                        download=False, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
                                            shuffle=True, num_workers=0)

    test_data = datasets.FashionMNIST(root=f'{DataSetPath}/FashionMNIST', train=False,
                                        download=False, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size,
                                            shuffle=False, num_workers=0)

    return train_data, test_data, train_loader, test_loader




def get_SVHN(batch_size,num_workers):
    # 数据集加载
    cudnn.benchmark = True
    transform = transforms.Compose(
        [transforms.ToTensor(),  # 将PILImage转换为张量 # 将PILImage转换为张量
        ]  # 将[0, 1]归一化到[-1, 1]
    )

    train_data = datasets.SVHN(
        root=f'{DataSetPath}/SVHN',
        split='train',
        download=True,
        transform=transform
    )
    
    test_data = datasets.SVHN(
        root=f'{DataSetPath}/SVHN',
        split='test',
        download=True,
        transform=transform
    )
    
    # define train loader
    train_loader = Data.DataLoader(
        dataset=train_data,
        shuffle=True,
        batch_size=batch_size
    )
    
    # define test loader
    test_loader = Data.DataLoader(
        dataset=test_data,
        shuffle=True,
        batch_size=batch_size
    )

    return train_data, test_data, train_loader, test_loader


def get_cifar(batch_size,num_workers):
     # 数据集加载
    cudnn.benchmark = True
    transform = transforms.Compose(
        [transforms.ToTensor(),  # 将PILImage转换为张量 # 将PILImage转换为张量
        #  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 将[0, 1]归一化到[-1, 1]
        ]
    )

    train_data = datasets.CIFAR10(
        root=f'{DataSetPath}/CIFAR10',
        train=True,
        download=True,
        transform=transform
    )
    
    test_data = datasets.CIFAR10(
        root=f'{DataSetPath}/CIFAR10',
        train=False,
        download=True,
        transform=transform
    )
    
    # define train loader
    train_loader = Data.DataLoader(
        dataset=train_data,
        shuffle=True,
        batch_size=batch_size
    )
    
    # define test loader
    test_loader = Data.DataLoader(
        dataset=test_data,
        shuffle=True,
        batch_size=batch_size
    )

    return train_data, test_data, train_loader, test_loader





# def get_mycifar(batch_size,num_workers):
#     # 数据集加载
#     cudnn.benchmark = True
#     transform = transforms.Compose(
#         [transforms.ToTensor(),  # 将PILImage转换为张量 # 将PILImage转换为张量
#         ]  # 将[0, 1]归一化到[-1, 1]
#     )
#     train_data = CIFAR10(root='DataSet/CIFAR10',
#                                           # root表示cifar10的数据存放目录，使用torchvision可直接下载cifar10数据集，也可直接在https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz这里下载（链接来自cifar10官网）
#                                           train=True,
#                                           download=False,
#                                           transform=transform  # 按照上面定义的transform格式转换下载的数据
#                                           )
#     test_data = CIFAR10(root='DataSet/CIFAR10',
#                                             train=False,
#                                             download=False,
#                                             transform=transform)

#     train_loader = torch.utils.data.DataLoader(train_data,
#                                             batch_size=batch_size,  # 每个batch载入的图片数量，默认为1
#                                             shuffle=True,
#                                             num_workers=0  # 载入训练数据所需的子任务数
#                                             )
#     test_loader = torch.utils.data.DataLoader(test_data,
#                                             batch_size=batch_size,
#                                             shuffle=False,
#                                             num_workers=0)                                        

#     return train_data, test_data, train_loader, test_loader


def get_cifar100(batch_size,num_workers):
     # 数据集加载
    cudnn.benchmark = True
    transform = transforms.Compose(
        [transforms.ToTensor(),  # 将PILImage转换为张量 # 将PILImage转换为张量
        #  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 将[0, 1]归一化到[-1, 1]
        ]
    )

    train_data = datasets.CIFAR100(
        root=f'{DataSetPath}/CIFAR100',
        train=True,
        download=True,
        transform=transform
    )
    
    test_data = datasets.CIFAR100(
        root=f'{DataSetPath}/CIFAR100',
        train=False,
        download=True,
        transform=transform
    )
    
    # define train loader
    train_loader = Data.DataLoader(
        dataset=train_data,
        shuffle=True,
        batch_size=batch_size
    )
    
    # define test loader
    test_loader = Data.DataLoader(
        dataset=test_data,
        shuffle=True,
        batch_size=batch_size
    )

    return train_data, test_data, train_loader, test_loader


# def get_mycifar100(batch_size,num_workers):
#      # 数据集加载
#     cudnn.benchmark = True
#     transform = transforms.Compose(
#         [transforms.ToTensor(),  # 将PILImage转换为张量 # 将PILImage转换为张量
#         #  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 将[0, 1]归一化到[-1, 1]
#         ]
#     )

#     train_data = CIFAR100(
#         root='DataSet/CIFAR100',
#         train=True,
#         download=True,
#         transform=transform
#     )
    
#     test_data = CIFAR100(
#         root='DataSet/CIFAR100',
#         train=False,
#         download=True,
#         transform=transform
#     )
    
#     # define train loader
#     train_loader = Data.DataLoader(
#         dataset=train_data,
#         shuffle=True,
#         batch_size=batch_size
#     )
    
#     # define test loader
#     test_loader = Data.DataLoader(
#         dataset=test_data,
#         shuffle=True,
#         batch_size=batch_size
#     )

#     return train_data, test_data, train_loader, test_loader

# get_dataset("SVHN",50,0)

def concatenate_data(old_datasets,add_data,add_label,datasets_name):
    """合并数据集

    Args:
        old_datasets (_type_): 原始数据集
        add_data (_type_): 需要增加的图片
        add_label (_type_): 需要增加的标签
        datasets_name (_type_): 数据集名称

    Returns:
        _type_: 合并后的数据集图片和标签
    """
    
    new_data,new_label = None,None
    if datasets_name == 'FashionMNIST' or datasets_name == 'MNIST':
        mid_data = (add_data*255).squeeze(0).clone().detach()
        mid_data = mid_data.type(torch.uint8).cpu()
        # mid_data = (mid_data.squeeze(0).cpu())*255
        # new_data = torch.cat((old_datasets.data,mid_data.round()))  # .round()四舍五入  .trunc 取整数部分
        new_data = torch.cat((old_datasets.data,mid_data)) 
        new_label = torch.cat((old_datasets.targets,add_label))
        old_datasets.targets = new_label
    # elif datasets_name == 'MYCIFAR10' or datasets_name == 'CIFAR10':
    #     # new_data = np.concatenate((old_datasets.data,add_data.transpose(0,2,3,1)))
    #     new_data = np.concatenate((old_datasets.data,add_data))
    #     new_label = np.concatenate((old_datasets.targets,add_label.numpy().tolist())).
    #     # new_label =np.concatenate((old_datasets.targets,np.expand_dims(np.array(add_label.item()),axis=0)))
    #     old_datasets.targets = new_label
    elif datasets_name == 'MYCIFAR10' or datasets_name == 'CIFAR10' or datasets_name == 'CIFAR100':
        mid_data = (add_data*255).round().clone().detach().cpu()
        mid_data = mid_data.type(torch.uint8).numpy().transpose(0,2,3,1)
        new_data = np.concatenate((old_datasets.data,mid_data))
        # new_data = np.concatenate((old_datasets.data,np.expand_dims(add_data,0)))
        new_label = np.concatenate((old_datasets.targets,add_label.numpy().tolist()))
        # new_label =np.concatenate((old_datasets.targets,np.expand_dims(np.array(add_label.item()),axis=0)))
        old_datasets.targets = new_label
    elif datasets_name == 'SVHN':
        # x.numpy()  x.detach().numpy() 主要区别在于是否使用detach()，也就是返回的新变量是否需要计算梯度。
        new_data = np.concatenate((old_datasets.data,add_data))
        new_label = np.concatenate((old_datasets.labels,add_label.numpy()))
        old_datasets.labels = new_label
        

    old_datasets.data = new_data                           
              
    # if len(mid_sample.shape) == len(train_data.data.shape):
    #         train_data_new.data = np.concatenate((train_data.data,mid_sample))
    # else:
    #     train_data_new.data = np.concatenate((train_data.data,mid_sample.squeeze(0)))
    # # 添加 label
    # # train_data_new.targets = np.concatenate((train_data.targets,torch.unsqueeze(borderline_labels[0], dim=0)))
    # if hasattr(train_data,'targets'):
    #     train_data_new.targets = np.concatenate((train_data.targets,torch.unsqueeze(borderline_labels[0], dim=0)))
    # else:
    #     train_data_new.labels = np.concatenate((train_data.labels,np.expand_dims(np.array(borderline_labels[0].item()),axis=0)))
    return old_datasets



def concatenate_data_adv(old_datasets,add_data,add_label,datasets_name,fun_name):
    """合并数据集

    Args:
        old_datasets (_type_): 原始数据集
        add_data (_type_): 需要增加的图片
        add_label (_type_): 需要增加的标签
        datasets_name (_type_): 数据集名称

    Returns:
        _type_: 合并后的数据集图片和标签
    """
    
    new_data,new_label = None,None
    if datasets_name == 'FashionMNIST' or datasets_name == 'MNIST':
        mid_data = (add_data*255).squeeze(0).clone().detach()
        mid_data = mid_data.type(torch.uint8).cpu()
        # mid_data = (mid_data.squeeze(0).cpu())*255
        # new_data = torch.cat((old_datasets.data,mid_data.round()))  # .round()四舍五入  .trunc 取整数部分
        new_data = torch.cat((old_datasets.data,mid_data)) 
        new_label = torch.cat((old_datasets.targets,add_label))
        old_datasets.targets = new_label
    # elif datasets_name == 'MYCIFAR10' or datasets_name == 'CIFAR10':
    #     # new_data = np.concatenate((old_datasets.data,add_data.transpose(0,2,3,1)))
    #     new_data = np.concatenate((old_datasets.data,add_data))
    #     new_label = np.concatenate((old_datasets.targets,add_label.numpy().tolist())).
    #     # new_label =np.concatenate((old_datasets.targets,np.expand_dims(np.array(add_label.item()),axis=0)))
    #     old_datasets.targets = new_label
    elif datasets_name == 'MYCIFAR10' or datasets_name == 'CIFAR10' or datasets_name == 'CIFAR100':
        if fun_name == 'adv':
            mid_data = (add_data*255).round().clone().detach().cpu()
            mid_data = mid_data.type(torch.uint8).numpy().transpose(0,2,3,1)
            new_data = np.concatenate((old_datasets.data,mid_data))
            # new_data = np.concatenate((old_datasets.data,np.expand_dims(add_data,0)))
            # new_label = np.concatenate((old_datasets.targets,np.expand_dims(np.array(add_label.item()),axis=0)))
            # new_label =np.concatenate((old_datasets.targets,np.expand_dims(np.array(add_label.item()),axis=0)))
        elif fun_name == 'orig':
            mid_data = (add_data*255).round().clone().detach().cpu()
            mid_data = mid_data.type(torch.uint8).numpy().transpose(2,1,0)
            new_data = np.concatenate((old_datasets.data,np.expand_dims(mid_data,0)))

        new_label = np.concatenate((old_datasets.targets,np.expand_dims(np.array(add_label.item()),axis=0)))
        old_datasets.targets = new_label
    elif datasets_name == 'SVHN':
        # x.numpy()  x.detach().numpy() 主要区别在于是否使用detach()，也就是返回的新变量是否需要计算梯度。
        new_data = np.concatenate((old_datasets.data,add_data))
        new_label = np.concatenate((old_datasets.labels,add_label.numpy()))
        old_datasets.labels = new_label
        

    old_datasets.data = new_data                           
              
    # if len(mid_sample.shape) == len(train_data.data.shape):
    #         train_data_new.data = np.concatenate((train_data.data,mid_sample))
    # else:
    #     train_data_new.data = np.concatenate((train_data.data,mid_sample.squeeze(0)))
    # # 添加 label
    # # train_data_new.targets = np.concatenate((train_data.targets,torch.unsqueeze(borderline_labels[0], dim=0)))
    # if hasattr(train_data,'targets'):
    #     train_data_new.targets = np.concatenate((train_data.targets,torch.unsqueeze(borderline_labels[0], dim=0)))
    # else:
    #     train_data_new.labels = np.concatenate((train_data.labels,np.expand_dims(np.array(borderline_labels[0].item()),axis=0)))
    return old_datasets
