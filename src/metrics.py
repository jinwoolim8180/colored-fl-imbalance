from statistics import mean, stdev
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, average_precision_score
import numpy as np

def evaluate(model, test_dataset, device, args):
    total_outputs = prediction(model, test_dataset, device, args)
    acc = evaluate_accuracy(total_outputs, test_dataset.labels, device)
    eo = evaluate_eo(total_outputs, test_dataset.labels, test_dataset.protected_variables)
    accs = [round(acc, 4) for acc in evaluate_group_accs(total_outputs, test_dataset.labels, test_dataset.protected_variables)]
    # ap = evaluate_ap(total_outputs, test_dataset.labels, test_dataset.protected_variables)
    ap = 0
    model.to(torch.device('cpu'))
    return [acc, accs], (eo, ap)

def prediction(model, test_dataset, device, args):
    model.to(device)
    model = nn.DataParallel(model)
    model.eval()
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=0)
    total_outputs = []
    with torch.no_grad():
        for _, (inputs, labels, _) in enumerate(test_loader):
            if args.dataset in ['compas', 'adult']:
                inputs_cat, inputs_num, labels = inputs[0].to(device), inputs[1].to(device), labels.to(device)
                outputs = model(inputs_cat, inputs_num)
                outputs = F.softmax(outputs, dim=1)
            else:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
            total_outputs.append(outputs)
    total_outputs = torch.cat(total_outputs)
    _, pred_labels = torch.max(total_outputs, 1)
    return pred_labels

def evaluate_accuracy(pred_labels, labels, device):
    return torch.sum(torch.eq(pred_labels, labels.to(device))).item() / len(labels)

def evaluate_group_accs(pred_labels, labels, protected_variables):
    pred_labels = pred_labels.cpu()
    subgroups = []
    uniq_pvs = torch.unique(protected_variables)
    uniq_labels = torch.unique(labels)
    subgroup_accs = []
    for v in uniq_pvs:
        for l in uniq_labels:
            subgroups.append(((protected_variables == v) & (labels == l)).nonzero().squeeze())
    for indices in subgroups:
        subgroup_accs.append(evaluate_accuracy(pred_labels[indices], labels[indices], torch.device('cpu')))
    print('avg:', mean(subgroup_accs))
    return subgroup_accs

def evaluate_ap(pred_labels, labels, protected_variables):
    return average_precision_score(labels.cpu(), pred_labels.cpu())

def evaluate_eo(pred_labels, labels, protected_variables):
    pred_labels = pred_labels.cpu()
    subgroups = []
    uniq_labels = labels.unique()
    subgroup_rates = [[] for _ in range(len(uniq_labels))]
    for v in torch.unique(protected_variables):
        subgroups.append((protected_variables == v).nonzero().squeeze())
    for indices in subgroups:
        matrix = confusion_matrix(labels[indices], pred_labels[indices], labels=uniq_labels)
        for i in range(len(uniq_labels)):
            if matrix[i].sum() != 0:
                rate = matrix[i][i] / matrix[i].sum()
                subgroup_rates[i].append(rate)
    eo = sum([max(rates) - min(rates) for rates in subgroup_rates])
    # eo = mean([stdev(rates) for rates in subgroup_rates])
    return eo