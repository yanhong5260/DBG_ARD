import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from DBGARD.getBoundarySample import get_boundarySample

def squared_l2_norm(x):
    flattened = x.view(x.unsqueeze(0).shape[0], -1)
    return (flattened ** 2).sum(1)

def l2_norm(x):
    return squared_l2_norm(x).sqrt()

def angular_distance(logits1, logits2):
    numerator = logits1.mul(logits2).sum(1)
    logits1_l2norm = logits1.mul(logits1).sum(1).sqrt()
    logits2_l2norm = logits2.mul(logits2).sum(1).sqrt()
    denominator = logits1_l2norm.mul(logits2_l2norm)
    for i, _ in enumerate(numerator):
        if numerator[i] > denominator[i]:
            numerator[i] = denominator[i]
    D = torch.sub(1.0, torch.abs(torch.div(numerator, denominator)))
    return D

def distance_loss(orig, mid, device):
    dis = angular_distance(orig, mid)
    dis = dis.mean()
    norm = 1
    dis_loss = dis + 0.001 * norm
    return dis_loss

def getMid(model, teacher_logits, x_natural, y, optimizer, device, step_size=0.003, epsilon=0.031, perturb_steps=10, beta=6.0):
    criterion_kl = nn.KLDivLoss(size_average=False, reduce=False)
    model.eval()
    
    borderline_samples = get_boundarySample(x_natural, y, model, device) 
    x_adv = borderline_samples.clone()

    for _ in range(perturb_steps):
        x_adv.requires_grad_()
        with torch.enable_grad():
            loss_kl = criterion_kl(F.log_softmax(model(x_adv), dim=1),
                                   F.softmax(teacher_logits, dim=1))
            loss_kl = torch.sum(loss_kl)
        grad = torch.autograd.grad(loss_kl, [x_adv])[0]
        x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
        x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
        x_adv = torch.clamp(x_adv, 0.0, 1.0)

    model.train()

    x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)
    optimizer.zero_grad()
    logits = model(x_adv)
    return logits
