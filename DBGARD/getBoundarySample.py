import os
import numpy as np
import torch
import torch.nn.functional as F
import os.path as osp
import sys
sys.path.insert(0, osp.join(osp.dirname(osp.abspath(__file__)), '../'))

from utils.loggerUtils import MyLogger
import utils.config as config
from torchattacks.attacks.deepfool import DeepFool
from utils.modelUtils import getAttack

logger = MyLogger(os.path.basename(__file__).split(".")[0])
args = config.args

def boundarySample_search(X, Y, adv_x, model, device):
    borderline_samples = []
    borderline_labels = []
    x1 = X
    x2 = adv_x
    
    Z1 = model(X.to(device))
    probs_x1 = F.softmax(Z1, dim=1).cpu().detach().numpy()
    pred_x1 = np.argmax(probs_x1, axis=1)

    with torch.no_grad():
        for k in range(50):
            middle_sample = (x1 + x2) / 2
            Zmid = model(middle_sample.to(device))
            probs_middle = F.softmax(Zmid, dim=1).cpu().numpy().flatten()
            pred_middle = np.argmax(probs_middle)
            pred_middle_sort = np.sort(probs_middle)

            if (pred_middle_sort[-1] - pred_middle_sort[-2]) < 0.1 and pred_middle == pred_x1:
                borderline_samples.append(middle_sample)
                borderline_labels.append(Y)
                break

            if pred_middle == pred_x1:
                x1 = middle_sample
                pred_x1 = pred_middle
            else:
                x2 = middle_sample

            if len(borderline_samples) >= 1:
                break

    return borderline_samples, borderline_labels


def get_boundarySample(X, Y, model, device):
    attack_r = DeepFool(model, steps=50, overshoot=0.02)
    pert_image, r_list = attack_r(X, Y)
    
    r_mid = [i for i in r_list if i != 0]
    # TODO : 如果r_mid为空，则返回0
    p = np.percentile(r_mid, 50) if len(r_mid) > 0 else 0
    borderline_samples = []
    
    for x_index, r in enumerate(r_list):
        if 0 < r < p:
            adv_x = pert_image[x_index].unsqueeze(0)
            bd_image, _ = boundarySample_search(X[x_index].unsqueeze(0), Y[x_index].unsqueeze(0), adv_x, model, device)
            bd_sample = bd_image[0].squeeze(0) if bd_image else adv_x.squeeze(0)
        else:
            bd_sample = X[x_index]
        
        borderline_samples.append(bd_sample)

    borderline_samples = torch.stack([t for t in borderline_samples])
    return borderline_samples
