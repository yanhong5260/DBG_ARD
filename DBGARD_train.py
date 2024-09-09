import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.optim as optim
from torchvision import  transforms
import math

from utils.loggerUtils import MyLogger
import utils.config as config
from DBGARD.losses import getMid, distance_loss
from models.wideresnet import wideresnet
from models.resnet_trades import ResNet18
from Attacks.attackPGD_test import attack_pgd
from DBGARD.thickness import calculate_thicknessBatch

# 设置随机种子以保证结果的一致性
def set_seed(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

# KL散度损失函数
def kl_loss(a, b):
    loss = -a * b + torch.log(b + 1e-5) * b
    return loss

# 评估模型在测试集上的性能
def eval_test(model, device, test_loader):
    logger.info("*********************eval_test*************************")
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    test_accuracy = correct / len(test_loader.dataset)
    logger.info('Test: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * test_accuracy))
    return test_loss, test_accuracy

# 主训练函数
def train(args):
    # 设置保存路径
    save_path = f'zcheckpoint/CIFAR10/ResNet18/DBGARD'
    
    # 数据预处理
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])

    # 加载数据
    trainset = torchvision.datasets.CIFAR10(root='data/CIFAR10', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='data/CIFAR10', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    # 初始化模型
    student = ResNet18(num_classes=10)
    student = torch.nn.DataParallel(student).to(args.device)
    student.train()
    optimizer = optim.SGD(student.parameters(), lr=0.1, momentum=0.9, weight_decay=2e-4)

    teacher = wideresnet()
    teacher.load_state_dict(torch.load('./weights/model_cifar_wrn.pt'))
    teacher = torch.nn.DataParallel(teacher).to(args.device)
    teacher.eval()

    # 训练循环
    p = 5/6.0
    bestAcc = 0
    total_step = len(trainloader)

    for epoch in range(1, args.epochs + 1):
        logger.info(f"---------------------------{epoch}--------------------------")
        for step, (train_batch_data, train_batch_labels) in enumerate(trainloader):
            student.train()
            train_batch_data = train_batch_data.float().to(args.device)
            train_batch_labels = train_batch_labels.to(args.device)
            optimizer.zero_grad()

            with torch.no_grad():
                teacher_logits = teacher(train_batch_data)
  
            student.train()
            nat_logits = student(train_batch_data)
            KD_loss = kl_loss(F.log_softmax(nat_logits, dim=1), F.softmax(teacher_logits.detach(), dim=1))
            KD_loss = torch.mean(KD_loss)
            

            if epoch > 19:
                bd_logits = getMid(student, teacher_logits, train_batch_data, train_batch_labels, optimizer, args.device, step_size=2/255.0, epsilon=args.epsilon, perturb_steps=10)
                BD_loss = kl_loss(F.log_softmax(bd_logits, dim=1), F.softmax(teacher_logits.detach(), dim=1))
                BD_loss = torch.mean(BD_loss)
                mid_loss = distance_loss(nat_logits, bd_logits, args.device)
                if epoch % 5 == 0 or epoch == 20:
                    bd_thickness_s = calculate_thicknessBatch(train_batch_data, train_batch_labels, student, 10, 50, True, 0, 0.75)
                    bd_thickness_t = calculate_thicknessBatch(train_batch_data, train_batch_labels, teacher, 10, 50, True, 0, 0.75)
                    logger.info(f"{step}")
                    logger.info(f"----边界厚度均值bd_thickness_s为：{np.mean(bd_thickness_s)}")
                    logger.info(f"----边界厚度均值bd_thickness_t为：{np.mean(bd_thickness_t)}")
                    p = 1 / (1 + math.exp(4 * (np.mean(bd_thickness_s) - np.mean(bd_thickness_t))))
                if bd_thickness_t > bd_thickness_s:
                    loss = p * BD_loss + (1 - p) * KD_loss + mid_loss
                else:
                    loss = (1 - p) * BD_loss + p * KD_loss + mid_loss
            else:
                loss = KD_loss
            
            loss.backward()
            optimizer.step()

            if step % 100 == 0:
                # logger.info(f"mid_loss为：{mid_loss}")
                logger.info(f"loss为：{loss}")
                logger.info('Epoch [{}/{}], Step [{}/{}], Loss {:.4f}'.format(epoch, args.epochs, step, total_step, loss.item()))

        # 评估
        test_loss, test_accuracy = eval_test(student, args.device, testloader)

        if epoch % 10 == 0:
            robust_accs = []
            for test_batch_data, test_batch_labels in testloader:
                test_batch_data = test_batch_data.float().to(args.device)
                test_batch_labels = test_batch_labels.to(args.device)
                test_ifgsm_data = attack_pgd(student, test_batch_data, test_batch_labels, attack_iters=20, step_size=0.003, epsilon=8.0/255.0)
                logits_adv = student(test_ifgsm_data)
                predictions = np.argmax(logits_adv.cpu().detach().numpy(), axis=1)
                predictions = predictions - test_batch_labels.cpu().detach().numpy()
                robust_accs.extend(predictions.tolist())

            robust_acc = np.mean(np.array(robust_accs) == 0)
            logger.info(f'robust acc is {robust_acc}')

            if bestAcc < robust_acc:
                bestAcc = robust_acc
                model_best = student
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                torch.save(model_best.state_dict(), save_path + "_best_dict.pt")
                logger.info(f'model saved!!! path is {save_path}')

        if epoch % 20 == 0 or epoch % 50 == 0:
            torch.save(student.state_dict(), save_path + str(epoch) + "_robust" + str(robust_acc) + "_dict.pt")
            logger.info(f'model saved!!! path is {save_path}')

        if epoch in [76, 91, 101]:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.1

if __name__ == "__main__":
    # 初始化日志
    logger = MyLogger(os.path.basename(__file__).split(".")[0])
    
    # 解析参数
    args = config.args
    
    # 设置随机种子
    set_seed()
    
    # 开始训练
    train(args)
