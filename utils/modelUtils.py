import os
import numpy as np
import pandas as pd
import os.path as osp
import sys
sys.path.insert(0, osp.join(osp.dirname(osp.abspath(__file__)), '../'))

# from DeepFool_master.deepfool import deepfool
import matplotlib.pyplot as plt
import random
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.backends.cudnn as cudnn
# from model.VGG import VGG16
# 加载自定义的CIFAR10
# from DataSet.myCIFAR10 import CIFAR10
import sys
from utils.loggerUtils import logger
import utils.config as config 
args = config.args

from torchattacks.attacks.fgsm import FGSM
from torchattacks.attacks.pgd import PGD
from torchattacks.attacks.cw import CW
from torchattacks.attacks.deepfool import DeepFool
from torchattacks.attacks.bim import BIM
from torchattacks.attacks.autoattack import AutoAttack
from torchattacks.attacks.autoattack import FAB


"""
自定义GetLoader
"""
class GetLoader(torch.utils.data.Dataset):
    # 初始化函数，得到数据
    def __init__(self, data_root, data_label, transforms=None):
        self.data = data_root
        self.label = data_label
        self.transforms = transforms

    # index是根据batchsize划分数据后得到的索引，最后将data和对应的labels进行一起返回
    def __getitem__(self, index):
        if self.transforms is not None:
            data = self.data[index]
            data = torch.squeeze(data)
            # data = Image.fromarray(data)
            data = self.transforms(data)
        else:
            data = self.data[index]
        labels = self.label[index]
        labels = torch.squeeze(labels)
        return data, labels

    # 该函数返回数据大小长度，目的是DataLoader方便划分，如果不知道大小，DataLoader会一脸懵逼
    def __len__(self):
        return len(self.data)


class GetLoaderChange(torch.utils.data.Dataset):
    # 初始化函数，得到数据
    def __init__(self, data_root, data_label, transforms=None):
        self.data = data_root
        self.label = data_label
        self.transforms = transforms

    # index是根据batchsize划分数据后得到的索引，最后将data和对应的labels进行一起返回
    def __getitem__(self, index):
        if self.transforms is not None:
            data = self.data[index]
            data = torch.squeeze(data)
            # data = Image.fromarray(data)
            data = self.transforms(data)
        else:
            data = self.data[index]
        labels = self.label[index]
        labels = torch.squeeze(labels)
        return data, labels,index

    # 该函数返回数据大小长度，目的是DataLoader方便划分，如果不知道大小，DataLoader会一脸懵逼
    def __len__(self):
        return len(self.data)

"""
测试模型在良性样本的准确率
"""
def modelTest(model,name,data_loader,device):
    logger.info("开始测试模型的性能...")
    model = model.to(device)
    model.eval()
    currAcc = 0
    with torch.no_grad():
        correct = 0
        total = 0
        for index,data in enumerate(data_loader):
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        currAcc = correct / total
        logger.info('{} Accuracy of the model on the test images: {} %'.format(name,100 * correct / total))
    return currAcc


def modelTestChange(model,name,data_loader,device):
    logger.info("开始测试模型的性能...")
    model = model.to(device)
    model.eval()
    currAcc = 0
    with torch.no_grad():
        correct = 0
        total = 0
        for index,data in enumerate(data_loader):
            images, labels, _ = data
            images = images.to(device)
            labels = labels.to(device)
            outputs,_ = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        currAcc = correct / total
        logger.info('{} Accuracy of the model on the test images: {} %'.format(name,100 * correct / total))
    return currAcc


def modelTestTwo(model,name,data_loader,device):
    logger.info("开始测试模型的性能...")
    model = model.to(device)
    model.eval()
    currAcc = 0
    with torch.no_grad():
        correct = 0
        total = 0
        for index,data in enumerate(data_loader):
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        currAcc = correct / total
        logger.info('{} Accuracy of the model on the test images: {} %'.format(name,100 * correct / total))
    return currAcc

"""
训练模型
"""
def trainModel(model,train_loader,test_loader,save_path,name,epoch_nums=100,device=0):
    # 模型加载
    model = model.to(device)
    total_step = len(train_loader)
    #logger.info(model)
    criterion = nn.CrossEntropyLoss()  # 定义损失函数：交叉熵
    criterion = criterion.to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)  # 定义优化方法：随机梯度下降

    bestAcc= 0
    model_best = model
    for epoch in range(epoch_nums):
        logger.info("开始训练...")
        model.train()
        ave_loss = 0
        for batch_idx, data in enumerate(train_loader, 0):
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            # logger.info(images.shape)
            outputs,_ = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ave_loss = ave_loss + loss.item()
            if (batch_idx + 1) % 100 == 0:
                logger.info('Epoch [{}/{}], Step [{}/{}], Loss {:.4f}'.format(epoch + 1, epoch_nums, batch_idx + 1, total_step,
                                                                       loss.item()))
        
        ave_loss = ave_loss/total_step

        logger.info("测试test_loader...")
        currAcc= modelTest(model,name,test_loader,device)
        
        if bestAcc < currAcc:
            bestAcc = currAcc
            model_best = model
            # 保存整个模型 包含网络结构

            # 判断文件夹路径是否存在，如果不存在，则创建，此处是创建多级目录
            file_dir = os.path.split(save_path)[0]
            if not os.path.isdir(file_dir):
                os.makedirs(file_dir)

            torch.save(model_best, save_path)
            # 只保存键值对
            # torch.save(model.state_dict(), f'checkpoint/ResNet/cifar10_ResNet_origModel2.pkl')
            logger.info(f'model saved!!! path is {save_path}')
    return model_best

def trainModelChange(model,train_loader,test_loader,save_path,name,epoch_nums=50,device=0):
    # 模型加载
    model = model.to(device)
    total_step = len(train_loader)
    #logger.info(model)
    criterion = nn.CrossEntropyLoss()  # 定义损失函数：交叉熵
    criterion = criterion.to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)  # 定义优化方法：随机梯度下降

    bestAcc= 0
    model_best = model
    for epoch in range(epoch_nums):
        logger.info("开始训练...")
        model.train()
        ave_loss = 0
        for batch_idx, data in enumerate(train_loader, 0):
            images, labels,_ = data
            images = images.to(device)
            labels = labels.to(device)
            # logger.info(images.shape)
            outputs,_ = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ave_loss = ave_loss + loss.item()
            if (batch_idx + 1) % 100 == 0:
                logger.info('Epoch [{}/{}], Step [{}/{}], Loss {:.4f}'.format(epoch + 1, epoch_nums, batch_idx + 1, total_step,
                                                                       loss.item()))
        
        ave_loss = ave_loss/total_step

        logger.info("测试test_loader...")
        currAcc= modelTestChange(model,name,test_loader,device)
        
        if bestAcc < currAcc:
            bestAcc = currAcc
            model_best = model
            # 保存整个模型 包含网络结构

            # 判断文件夹路径是否存在，如果不存在，则创建，此处是创建多级目录
            file_dir = os.path.split(save_path)[0]
            if not os.path.isdir(file_dir):
                os.makedirs(file_dir)

            torch.save(model, save_path)
            # 只保存键值对
            # torch.save(model.state_dict(), f'checkpoint/ResNet/cifar10_ResNet_origModel2.pkl')
            logger.info(f'model saved!!! path is {save_path}')
    return model_best



def getAttack(name,model):
    if name == 'FGSM' :
        return FGSM(model, eps=0.007)
    elif name == "PGD":
        return PGD(model, eps=8 / 255, alpha=1 / 255, steps=40, random_start=True)
    elif name == "CW":
        return CW(model, c=1e-4, kappa=0, steps=1000, lr=0.01)
    elif name == "DeepFool":
        return DeepFool(model, steps=50, overshoot=0.02)
    elif name == "BIM":
        return BIM(model, eps=4 / 255, alpha=1 / 255, steps=0)
    elif name == "AutoAttack":
        return AutoAttack(model, norm='Linf', eps=.3, version='standard', n_classes=10, seed=None, verbose=False)
    elif name == "FAB":
        return FAB(model, norm='Linf', steps=100, eps=None, n_restarts=1, alpha_max=0.1, eta=1.05, beta=0.9, verbose=False, seed=0, targeted=False, n_classes=10)


def getAutoAdvData(model,test_loader,save_path):


    name = "AutoAttack"

    logger.info(len(test_loader))
    
    attack = getAttack(name,model) 
    # adv_image = []
    # orig_label = []
    adv_image = None
    orig_label = None
    logger.info(f"---------------开始生成对抗样本{save_path}_{name}-------------------")
    q = 0
    k=0
    for step, data in enumerate(tqdm(test_loader)):
        q = q + 1
        X, Y = data
        X = X.to('cuda').float()  # torch.Size([64, 1, 28, 28])
        # logger.info("X", X.shape)

        adv_x = attack(X, Y)  # torch.Size([64, 1, 28, 28])
        # adv_image.append(adv_x)
        # orig_label.append(Y)
        if adv_image == None:
            adv_image = adv_x.cpu()
        else:
            adv_image = torch.cat([adv_image.cpu(),adv_x.cpu()],dim=0).cpu()
        if orig_label == None:
            orig_label = Y.cpu()
        else:
            orig_label = torch.cat([orig_label.cpu(),Y.cpu()],dim=0).cpu()
        # print(q)
        # print(int((len(test_loader)+1)/2))
        # if(q==int((len(test_loader)+1)/2) or q==len(test_loader)):
        # if(q==int((len(test_loader))/2) or q==len(test_loader)): 
        # torch.cuda.memory_reserved(0)
        gpu_memory_all = 12288
        gpu_memory_current = torch.cuda.memory_reserved(0) / 1048576
        # if(q*test_loader.batch_size%5000==0):
        # 当前现存大于0.75时保存一次
        if gpu_memory_current > gpu_memory_all * 0.75:
            k=k+1  
            logger.info(f"---------------保存第{k}个文件-------------------")
            logger.info(f"adv_image：{len(adv_image)}")
            logger.info(f"orig_label：{len(orig_label)}")

            logger.info(f"adv_image：{adv_image.shape}")
            logger.info(f"orig_label：{orig_label.shape}")

            adv_image = adv_image.cpu().detach().numpy()
            orig_label = orig_label.cpu().detach().numpy()

            np.savez(f'{save_path}_AutoAttack_{k}_test', image=adv_image, label=orig_label)
            # np.savez(f'{save_path}_{attack.attack}', image=adv_image.cpu(), label=orig_label.cpu())
            logger.info(f'第{k}个对抗样本保存成功！~~~')
            del adv_image,orig_label
            adv_image = None
            orig_label = None
            torch.cuda.empty_cache()
    
    logger.info("---------------保存最后一个文件-------------------")
    logger.info(f"adv_image：{len(adv_image)}")
    logger.info(f"orig_label：{len(orig_label)}")

    logger.info(f"adv_image：{adv_image.shape}")
    logger.info(f"orig_label：{orig_label.shape}")

    adv_image = adv_image.cpu().detach().numpy()
    orig_label = orig_label.cpu().detach().numpy()

    np.savez(f'{save_path}_AutoAttack_{k+1}_test', image=adv_image, label=orig_label)
    # np.savez(f'{save_path}_{attack.attack}', image=adv_image.cpu(), label=orig_label.cpu())
    logger.info(f'{save_path}_AutoAttack对抗样本保存成功！~~~')
    
        
                


        


def getAdvData(model,test_loader,save_path):

    # attack1 = FGSM(model, eps=0.007)
    # attack2 = BIM(model, eps=4 / 255, alpha=1 / 255, steps=0)
    # attack3 = PGD(model, eps=8 / 255, alpha=1 / 255, steps=40, random_start=True)
    # attack4 = CW(model, c=1e-4, kappa=0, steps=1000, lr=0.01)
    # attack5 = DeepFool(model, steps=50, overshoot=0.02)

    # attack_list = [attack1,attack2,attack3,attack4,attack5]
    # attack_list = [attack4,attack5]
    # attack_list = ["FGSM","BIM","PGD","CW","AutoAttack","DeepFool"]
    # attack_list = ["AutoAttack"]
    attack_list = ["FGSM","BIM","PGD","CW","DeepFool"]

    logger.info(len(test_loader))
    k=0
    for name in attack_list:
        attack = getAttack(name,model)
        k=k+1
        # adv_image = []
        # orig_label = []
        adv_image = None
        orig_label = None
        logger.info(f"---------------开始生成对抗样本{save_path}/{attack.attack}-------------------")
        q = 0
        for step, data in enumerate(tqdm(test_loader)):
            q = q + 1
            X, Y = data
            X = X.to('cuda').float()  # torch.Size([64, 1, 28, 28])
            # logger.info("X", X.shape)

            if(name=="DeepFool"):
                adv_x_DeepFool = attack(X, Y)
                adv_x = adv_x_DeepFool[0]
            else:
                adv_x = attack(X, Y)  # torch.Size([64, 1, 28, 28])
            # adv_image.append(adv_x)
            # orig_label.append(Y)
            # print(adv_x.shape)


            if adv_image == None:
                adv_image = adv_x.cpu()
            else:
                adv_image = torch.cat([adv_image.cpu(),adv_x.cpu()],dim=0).cpu()
            if orig_label == None:
                orig_label = Y.cpu()
            else:
                orig_label = torch.cat([orig_label.cpu(),Y.cpu()],dim=0).cpu()
            torch.cuda.empty_cache()

            # if(q==len(test_loader)):

            #     adv_image = None
            #     orig_label = None
                


        logger.info("---------------结束-------------------")
        logger.info(f"adv_image：{len(adv_image)}")
        logger.info(f"orig_label：{len(orig_label)}")

        # images = [aa.tolist() for aa in adv_image]  # 列表中元素由tensor变成列表
        # labels = [aa.tolist() for aa in orig_label]  # 列表中元素由tensor变成列表

        # images = torch.tensor(images)
        # labels = torch.tensor(labels)

        logger.info(f"adv_image：{adv_image.shape}")
        logger.info(f"orig_label：{orig_label.shape}")

        adv_image = adv_image.cpu().detach().numpy()
        orig_label = orig_label.cpu().detach().numpy()

        np.savez(f'{save_path}_{attack.attack}', image=adv_image, label=orig_label)
        # np.savez(f'{save_path}_{attack.attack}', image=adv_image.cpu(), label=orig_label.cpu())
        logger.info(f'{attack.attack}对抗样本保存成功！~~~')


def getAdvDataChange(model,test_loader,save_path):

    # attack1 = FGSM(model, eps=0.007)
    # attack2 = BIM(model, eps=4 / 255, alpha=1 / 255, steps=0)
    # attack3 = PGD(model, eps=8 / 255, alpha=1 / 255, steps=40, random_start=True)
    # attack4 = CW(model, c=1e-4, kappa=0, steps=1000, lr=0.01)
    # attack5 = DeepFool(model, steps=50, overshoot=0.02)

    # attack_list = [attack1,attack2,attack3,attack4,attack5]
    # attack_list = [attack4,attack5]
    # attack_list = ["FGSM","BIM","PGD","CW","AutoAttack","DeepFool"]
    # attack_list = ["AutoAttack"]
    attack_list = ["FGSM","BIM","PGD","CW","DeepFool"]

    logger.info(len(test_loader))
    k=0
    for name in attack_list:
        attack = getAttack(name,model)
        k=k+1
        # adv_image = []
        # orig_label = []
        adv_image = None
        orig_label = None
        new_index_data = None
        logger.info(f"---------------开始生成对抗样本{save_path}{attack.attack}-------------------")
        q = 0
        for step, data in enumerate(tqdm(test_loader)):
            q = q + 1
            X, Y,index_data = data
            X = X.to('cuda').float()  # torch.Size([64, 1, 28, 28])
            # logger.info("X", X.shape)

            if(name=="DeepFool"):
                adv_x_DeepFool = attack(X, Y)
                adv_x = adv_x_DeepFool[0]
            else:
                adv_x = attack(X, Y)  # torch.Size([64, 1, 28, 28])
            # adv_image.append(adv_x)
            # orig_label.append(Y)
            if adv_image == None:
                adv_image = adv_x.cpu()
            else:
                adv_image = torch.cat([adv_image.cpu(),adv_x.cpu()],dim=0).cpu()
            if orig_label == None:
                orig_label = Y.cpu()
            else:
                orig_label = torch.cat([orig_label.cpu(),Y.cpu()],dim=0).cpu()
            if new_index_data == None:
                new_index_data = index_data.cpu()
            else:
                new_index_data = torch.cat([new_index_data.cpu(),index_data.cpu()],dim=0).cpu()
            torch.cuda.empty_cache()

        logger.info("---------------结束-------------------")
        logger.info(f"adv_image：{len(adv_image)}")
        logger.info(f"orig_label：{len(orig_label)}")

        logger.info(f"adv_image：{adv_image.shape}")
        logger.info(f"orig_label：{orig_label.shape}")

        adv_image = adv_image.cpu().detach().numpy()
        orig_label = orig_label.cpu().detach().numpy()
        new_index_data = new_index_data.cpu().detach().numpy()

        np.savez(f'{save_path}/Change_{attack.attack}', image=adv_image, label=orig_label, index = new_index_data)
        # np.savez(f'{save_path}_{attack.attack}', image=adv_image.cpu(), label=orig_label.cpu())
        logger.info(f'Change_{attack.attack}对抗样本保存成功！~~~')



def loadAdv(name):
    logger.info(f"*************获取对抗样本数据{name}... *****************")
    # MyDataSet = np.load(f"checkpoint/advData/cifar10_VGG16_advData10000_torchattacks_{name}.npz")
    # save_path=f'checkpoint/{args.dataset}/{args.model}/AdvData/{args.model}_{name}.npz'
    # save_path=f'checkpoint/{args.dataset}/{args.model}/AdvData/_{name}.npz'
    # save_path=f'checkpoint/{args.dataset}/{args.model}/AdvData/AdvData{name}.npz'
    save_path=f'checkpoint/{args.dataset}/{args.model}/AdvData/{args.model}_{name}.npz'
    
    MyDataSet = np.load(save_path)

    images = MyDataSet['image']
    labels = MyDataSet['label']
    # logger.info(labels)

    Cifar_Adv_image = [aa.tolist() for aa in images]  # 列表中元素由tensor变成列表
    Cifar_Adv_label = [aa.tolist() for aa in labels]  # 列表中元素由tensor变成列表

    Cifar_Adv_image = torch.tensor(Cifar_Adv_image) #(1000,3,32,32)
    Cifar_Adv_label = torch.tensor(Cifar_Adv_label)
    logger.info(type(Cifar_Adv_image))
    logger.info(Cifar_Adv_image.shape)

    logger.info("Cifar_Adv_label",type(Cifar_Adv_label))
    logger.info("Cifar_Adv_label",len(Cifar_Adv_label))
    return Cifar_Adv_image,Cifar_Adv_label


def loadAdvChange(name):
    logger.info(f"*************获取对抗样本数据{name}... *****************")
    # MyDataSet = np.load(f"checkpoint/advData/cifar10_VGG16_advData10000_torchattacks_{name}.npz")
    # save_path=f'checkpoint/{args.dataset}/{args.model}/AdvData/{args.model}_{name}.npz'
    save_path=f'checkpoint/{args.dataset}/{args.model}/AdvData/{name}.npz'
    
    MyDataSet = np.load(save_path)

    images = MyDataSet['image']
    labels = MyDataSet['label']
    indexs = MyDataSet['index']
    # logger.info(labels)

    Adv_image = [aa.tolist() for aa in images]  # 列表中元素由tensor变成列表
    Adv_label = [aa.tolist() for aa in labels]  # 列表中元素由tensor变成列表
    Adv_index = [aa.tolist() for aa in indexs]  # 列表中元素由tensor变成列表

    Adv_image = torch.tensor(Adv_image) #(1000,3,32,32)
    Adv_label = torch.tensor(Adv_label)
    Adv_index = torch.tensor(Adv_index)
    logger.info(type(Adv_image))
    logger.info(Adv_image.shape)

    logger.info("Cifar_Adv_label",type(Adv_label))
    logger.info("Cifar_Adv_label",len(Adv_label))
    return Adv_image,Adv_label,Adv_index



"""
测试模型在对抗样本的准确率
"""
def modelAdvTest(model,BATCH_SIZE,device):
    transform_train = transforms.Compose([
        transforms.ToPILImage(),
        # transforms.RandomHorizontalFlip(),
        # transforms.RandomGrayscale(),
        transforms.ToTensor()
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    # 'BIM','CW','DeepFool','FGSM','PGD'
    names = [
        'FGSM','BIM','PGD','DeepFool','CW'
    ]
    # names = [
    #     'AutoAttack_lbgat'
    # ]
    for name in names:
        Cifar_Adv_image,Cifar_Adv_label = loadAdv(name)
        train_data_new = GetLoader(Cifar_Adv_image, Cifar_Adv_label, transforms=transform_train)
        # 使用DataLoader进行数据分批，dataset代表传入的数据集，batch_size表示每个batch有多少个样本
        my_loader = DataLoader(dataset=train_data_new, batch_size=BATCH_SIZE, shuffle=True)
        modelTest(model,name,my_loader,device)


def modelAdvTestChange(model,BATCH_SIZE,device):
    transform_train = transforms.Compose([
        transforms.ToPILImage(),
        # transforms.RandomHorizontalFlip(),
        # transforms.RandomGrayscale(),
        transforms.ToTensor()
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    # 'BIM','CW','DeepFool','FGSM','PGD'
    names = [
        'Change_FGSM','Change_BIM','Change_PGD','Change_DeepFool','Change_CW'
    ]
    # names = [
    #     'AutoAttack_lbgat'
    # ]
    for name in names:
        Cifar_Adv_image,Cifar_Adv_label,Adv_index = loadAdvChange(name)
        train_data_new =  GetLoaderChange(Cifar_Adv_image, Cifar_Adv_label, transforms=transform_train)
        # 使用DataLoader进行数据分批，dataset代表传入的数据集，batch_size表示每个batch有多少个样本
        my_loader = DataLoader(dataset=train_data_new, batch_size=BATCH_SIZE, shuffle=True)
        modelTestChange(model,name,my_loader,device)

    
def modelAutoAdvTest(model,BATCH_SIZE,device):
    transform_train = transforms.Compose([
        transforms.ToPILImage(),
        # transforms.RandomHorizontalFlip(),
        # transforms.RandomGrayscale(),
        transforms.ToTensor()
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    logger.info(f"----------------测试AutoAttack-------------------")
    # checkpoint/CIFAR10/VGG/_AutoAttack_1_test.npz
    name = '_AutoAttack'
    for i in range(2):      
        path = name+"_"+str(i+1)+"_test"
        Cifar_Adv_image,Cifar_Adv_label = loadAdv(path)
        train_data_new = GetLoader(Cifar_Adv_image, Cifar_Adv_label, transforms=transform_train)
        # 使用DataLoader进行数据分批，dataset代表传入的数据集，batch_size表示每个batch有多少个样本
        my_loader = DataLoader(dataset=train_data_new, batch_size=BATCH_SIZE, shuffle=True)
        modelTest(model,name,my_loader,device)
    
    