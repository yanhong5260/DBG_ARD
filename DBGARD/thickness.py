import numpy as np

import torch
import torch.nn as nn
import utils.config as config 
from torchattacks.attacks.pgdl2 import PGDL2

args = config.args

def calculate_thickness(train_loader, model, num_classes, num_measurements, num_points, class_pair, alpha, beta):
    softmax1 = nn.Softmax()
    temp_hist = []
    
    for i, (images, labels) in enumerate(train_loader):
        if i >= num_measurements:
            break

        model.eval()
        images, labels = images.cuda(), labels.cuda()

        output = model(images)
        pred = output.data.max(1, keepdim=True)[1]

        labels_change = torch.randint(1, num_classes, (labels.shape[0],)).cuda()
        wrong_labels = torch.remainder(labels_change + labels, num_classes)
        PGD_attack = PGDL2(model, eps=5.0, alpha=1, steps=20, random_start=True)
        adv_images = PGD_attack.__call__(images, wrong_labels)

        for data_ind in range(labels.shape[0]):
            x1, x2 = images[data_ind], adv_images[data_ind]
            dist = torch.norm(x1 - x2, p=2)

            new_batch = []
            for lmbd in np.linspace(0, 1.0, num=num_points):
                new_batch.append(x1 * lmbd + x2 * (1 - lmbd))
            new_batch = torch.stack(new_batch)

            model.eval()
            y_new_batch = softmax1(model(new_batch))

            if not class_pair:
                y_new_batch = y_new_batch[:, pred[data_ind]].detach().cpu().numpy().flatten()
            else:
                y_original_class = y_new_batch[:, pred[data_ind]].squeeze()
                y_target_class = y_new_batch[:, wrong_labels[data_ind]]
                y_new_batch = y_original_class - y_target_class
                y_new_batch = y_new_batch.detach().cpu().numpy().flatten()

            boundary_thickness = np.logical_and((beta > y_new_batch), (alpha < y_new_batch))
            boundary_thickness = dist.item() * np.sum(boundary_thickness) / num_points
            temp_hist.append(boundary_thickness)
    
    return temp_hist

def calculate_thicknessBatch(X, Y, model, num_classes, num_points, class_pair, alpha, beta):
    softmax1 = nn.Softmax()
    temp_hist = []
    
    model.eval()
    images, labels = X.cuda(), Y.cuda()

    output = model(images)
    pred = output.data.max(1, keepdim=True)[1]

    labels_change = torch.randint(1, num_classes, (labels.shape[0],)).cuda()
    wrong_labels = torch.remainder(labels_change + labels, num_classes)
    PGD_attack = PGDL2(model, eps=5.0, alpha=1, steps=20, random_start=True)
    adv_images = PGD_attack.__call__(images, wrong_labels)

    for data_ind in range(labels.shape[0]):
        x1, x2 = images[data_ind], adv_images[data_ind]
        dist = torch.norm(x1 - x2, p=2)

        new_batch = []
        for lmbd in np.linspace(0, 1.0, num=num_points):
            new_batch.append(x1 * lmbd + x2 * (1 - lmbd))
        new_batch = torch.stack(new_batch)

        model.eval()
        y_new_batch = softmax1(model(new_batch))

        if not class_pair:
            y_new_batch = y_new_batch[:, pred[data_ind]].detach().cpu().numpy().flatten()
        else:
            y_original_class = y_new_batch[:, pred[data_ind]].squeeze()
            y_target_class = y_new_batch[:, wrong_labels[data_ind]]
            y_new_batch = y_original_class - y_target_class
            y_new_batch = y_new_batch.detach().cpu().numpy().flatten()

        boundary_thickness = np.logical_and((beta > y_new_batch), (alpha < y_new_batch))
        boundary_thickness = dist.item() * np.sum(boundary_thickness) / num_points
        temp_hist.append(boundary_thickness)

    return temp_hist
