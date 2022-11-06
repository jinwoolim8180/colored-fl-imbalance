from statistics import mean, stdev
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np

def evaluate(model, test_dataset, device, args):
    total_outputs = prediction(model, test_dataset, device, args)
    labels = test_dataset.targets
    acc = evaluate_accuracy(total_outputs, labels, device)
    accs = [round(acc, 4) for acc in evaluate_label_accs(total_outputs, labels)]
    model.to(torch.device('cpu'))
    return [acc, accs]

def prediction(model, test_dataset, device, args):
    model.to(device)
    model = nn.DataParallel(model)
    model.eval()
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=0)
    total_outputs = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            total_outputs.append(outputs)
    total_outputs = torch.cat(total_outputs)
    _, pred_labels = torch.max(total_outputs, 1)
    return pred_labels

def evaluate_accuracy(pred_labels, labels, device):
    return torch.sum(torch.eq(pred_labels, labels.to(device))).item() / len(labels)

def evaluate_label_accs(pred_labels, labels):
    pred_labels = pred_labels.cpu()
    subgroups = []
    subgroup_accs = []
    for l in torch.unique(labels):
        subgroups.append((labels == l).nonzero().squeeze())
    for indices in subgroups:
        subgroup_accs.append(evaluate_accuracy(pred_labels[indices], labels[indices], torch.device('cpu')))
    return subgroup_accs