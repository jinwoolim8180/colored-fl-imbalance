from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from torch import optim
import time
import random
import os
from tqdm import tqdm
from torchtext import data
import numpy as np
import torch.nn.functional as F
from torch.utils import data
import pandas as pd
from torch.autograd import Variable
from torch.autograd import grad

import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
sys.path.append('../models')
import models
import dataset
import arguments


torch.set_num_threads(8)
torch.manual_seed(1)
random.seed(1)


def get_accuracy(truth, pred):
    assert len(truth) == len(pred)
    right = 0
    for i in range(len(truth)):
        if truth[i] == pred[i]:
            right += 1.0
    return right / len(truth)

# Representation neutralization


def feature_neutralization(r_batch, p_batch, y_batch, a_batch):
    category1_bias1 = []
    category1_bias2 = []
    category2_bias1 = []
    category2_bias2 = []

    for i in range(a_batch.shape[0]):
        if y_batch[i].cpu().detach().numpy() == 0 and a_batch[i].cpu().detach().numpy() == 0:
            category1_bias1.append([r_batch[i], p_batch[i]])
        elif y_batch[i].cpu().detach().numpy() == 0 and a_batch[i].cpu().detach().numpy() == 1:
            category1_bias2.append([r_batch[i], p_batch[i]])
        elif y_batch[i].cpu().detach().numpy() == 1 and a_batch[i].cpu().detach().numpy() == 0:
            category2_bias1.append([r_batch[i], p_batch[i]])
        elif y_batch[i].cpu().detach().numpy() == 1 and a_batch[i].cpu().detach().numpy() == 1:
            category2_bias2.append([r_batch[i], p_batch[i]])

    neutralization_repre_5 = torch.zeros(a_batch.shape[0], HIDDEN_DIM)
    neutralization_repre_6 = torch.zeros(a_batch.shape[0], HIDDEN_DIM)
    neutralization_repre_7 = torch.zeros(a_batch.shape[0], HIDDEN_DIM)
    neutralization_repre_8 = torch.zeros(a_batch.shape[0], HIDDEN_DIM)
    neutralization_repre_9 = torch.zeros(a_batch.shape[0], HIDDEN_DIM)

    neutralization_probability5 = torch.zeros(a_batch.shape[0], 2)

    for i in range(a_batch.shape[0]):
        if y_batch[i].cpu().detach().numpy() == 0 and a_batch[i].cpu().detach().numpy() == 0:
            if len(category1_bias2) != 0:
                neutralization_sample = random.choice(category1_bias2)
            else:
                neutralization_sample = random.choice(category1_bias1)

        elif y_batch[i].cpu().detach().numpy() == 0 and a_batch[i].cpu().detach().numpy() == 1:
            if len(category1_bias1) != 0:
                neutralization_sample = random.choice(category1_bias1)
            else:
                neutralization_sample = random.choice(category1_bias2)

        elif y_batch[i].cpu().detach().numpy() == 1 and a_batch[i].cpu().detach().numpy() == 0:
            if len(category2_bias2) != 0:
                neutralization_sample = random.choice(category2_bias2)
            else:
                neutralization_sample = random.choice(category2_bias1)

        elif y_batch[i].cpu().detach().numpy() == 1 and a_batch[i].cpu().detach().numpy() == 1:
            if len(category2_bias1) != 0:
                neutralization_sample = random.choice(category2_bias1)
            else:
                neutralization_sample = random.choice(category2_bias2)
        neutralization_repre_5[i] = 0.5 * \
            r_batch[i] + 0.5 * neutralization_sample[0]
        neutralization_repre_6[i] = 0.6 * \
            r_batch[i] + 0.4 * neutralization_sample[0]
        neutralization_repre_7[i] = 0.7 * \
            r_batch[i] + 0.3 * neutralization_sample[0]
        neutralization_repre_8[i] = 0.8 * \
            r_batch[i] + 0.2 * neutralization_sample[0]
        neutralization_repre_9[i] = 0.9 * \
            r_batch[i] + 0.1 * neutralization_sample[0]

        neutralization_probability5[i] = 0.5 * \
            p_batch[i] + 0.5 * neutralization_sample[1]
    neutralization_repre_5 = neutralization_repre_5.cuda()
    neutralization_repre_6 = neutralization_repre_6.cuda()
    neutralization_repre_7 = neutralization_repre_7.cuda()
    neutralization_repre_8 = neutralization_repre_8.cuda()
    neutralization_repre_9 = neutralization_repre_9.cuda()
    return neutralization_repre_5, neutralization_repre_6, neutralization_repre_7, neutralization_repre_8, neutralization_repre_9, neutralization_probability5


def train_epoch_progress(model, classification_head, train_iter, loss_function, optimizer, optimizer_head, epoch):
    model.train()
    classification_head.train()
    if epoch >= EPOCHS_first_stage:
        for param in model.parameters():
            param.requires_grad = False
        for param in classification_head.parameters():
            param.requires_grad = True
    avg_loss = 0.0
    truth_res = []
    pred_res = []
    softmax = nn.Softmax(dim=1)

    for batch in tqdm(train_iter, desc='Train epoch '+str(epoch+1)):
        # Pre-processing the data
        sent, label, sensitive_label = batch[0], batch[1], batch[2]
        sent = sent.type(torch.cuda.FloatTensor)
        label = label.type(torch.cuda.LongTensor)
        sent = Variable(sent)
        label = Variable(label)
        label = label.cpu()
        truth_res += list(label.data.numpy())
        sensitive_label = Variable(sensitive_label)

        # Get probability of the original input from the biased model
        sent = sent.cuda()
        pred, representation = model(sent)
        pred_softmax = softmax(pred/temperature)  # temperature scaling
        pred_softmax = pred_softmax.cpu()

        # The first training stage
        if epoch < EPOCHS_first_stage:
            # Calculating accuracy
            pred = F.log_softmax(pred)
            pred = pred.cpu()
            pred_label = pred.data.max(1)[1].numpy()
            pred_res += [x for x in pred_label]

            # Update the parameters for the whole model
            model.zero_grad()
            loss = loss_function(pred, label)
            avg_loss += loss.data[0]
            loss.backward()
            optimizer.step()

        # The second training stage
        if epoch >= EPOCHS_first_stage:
            # Get the interpolated features and probablity
            neutra_repre_5, neutra_repre_6, neutra_repre_7, neutra_repre_8, neutra_repre_9, neutra_probability5 = feature_neutralization(
                representation, pred_softmax, label, sensitive_label)

            # Using knowledge distillation loss as in equation 1
            pred_neutra = classification_head(neutra_repre_5)
            pred_neutra = softmax(pred_neutra)
            pred_neutra = pred_neutra.cpu()
            loss = kd_loss_function(neutra_probability5, pred_neutra)

            # Add regularization as is done in equation 3
            augmented_list = []
            augmented_list.append(neutra_repre_6)
            augmented_list.append(neutra_repre_7)
            augmented_list.append(neutra_repre_8)
            augmented_list.append(neutra_repre_9)
            difference_sum = 0
            for i in range(4):
                pred_augmented = classification_head(augmented_list[i])
                pred_augmented = softmax(pred_augmented)
                pred_augmented = pred_augmented.cpu()
                difference_sum += torch.abs(pred_augmented - pred_neutra)

            # Linearly combine two losses as is done in equation 4
            loss += alpha * torch.sum(difference_sum)
            avg_loss += loss.data[0]

            # Calculating accuracy
            pred, representation = model(sent)
            pred = classification_head(representation)
            pred = F.log_softmax(pred)
            pred = pred.cpu()
            pred_label = pred.data.max(1)[1].numpy()
            pred_res += [x for x in pred_label]

            # Update the classification head parameters
            classification_head.zero_grad()
            loss.backward()
            optimizer_head.step()

    avg_loss /= len(train_iter)
    acc = get_accuracy(truth_res, pred_res)
    return avg_loss, acc


def evaluate_first_stage(model, data, loss_function, name):
    model.eval()
    avg_loss = 0.0
    truth_res = []
    pred_res = []
    for batch in data:
        sent, label, sensitive_label = batch[0], batch[1], batch[2]
        sent = sent.type(torch.cuda.FloatTensor)
        label = label.type(torch.cuda.LongTensor)
        sent = Variable(sent)
        label = Variable(label)
        truth_res += list(label.data)
        model.batch_size = len(label.data)
        sent = sent.cuda()
        pred, representation = model(sent)
        pred = F.log_softmax(pred)
        pred = pred.cpu()
        pred_label = pred.data.max(1)[1].numpy()
        pred_res += [x for x in pred_label]
        label = label.cpu()
        loss = loss_function(pred, label)
        avg_loss += loss.data[0]
    avg_loss /= len(data)
    acc = get_accuracy(truth_res, pred_res)
    print(name + ': loss %.2f acc %.1f' % (avg_loss, acc*100))
    return acc


def evaluate_second_stage(model, classification_head, data, loss_function, name):
    model.eval()
    classification_head.eval()
    avg_loss = 0.0
    truth_res = []
    pred_res = []
    for batch in data:
        sent, label, sensitive_label = batch[0], batch[1], batch[2]
        sent = sent.type(torch.cuda.FloatTensor)
        label = label.type(torch.cuda.LongTensor)
        sent = Variable(sent)
        label = Variable(label)
        truth_res += list(label.cpu().data.numpy())
        model.batch_size = len(label.data)
        sent = sent.cuda()
        pred, representation = model(sent)
        pred = classification_head(representation)
        pred = F.log_softmax(pred)
        pred = pred.cpu()
        pred_label = pred.data.max(1)[1].numpy()
        pred_res += [x for x in pred_label]
        label = label.cpu()
        loss = loss_function(pred, label)
        avg_loss += loss.data[0]
    avg_loss /= len(data)
    acc = get_accuracy(truth_res, pred_res)
    print(name + ': loss %.2f acc %.1f' % (avg_loss, acc*100))
    return acc

EPOCHS_first_stage = 9
EPOCHS_second_stage = 4
USE_GPU = torch.cuda.is_available()
HIDDEN_DIM = 50
BATCH_SIZE = 64
INPUT_DIM = 138
alpha = 0.035  # This is the value in Equation 4 to control the fairness accuracy trade-off
temperature = 5.0  # Temperature scaling

train_dataset, train_subgroups = dataset.get_dataset(args, 'train')
val_dataset, val_subgroups = dataset.get_dataset(args, 'val')
test_dataset, test_subgroups = dataset.get_dataset(args, 'test')

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=8, shuffle=True)

train_model = models.get_model(args)
test_model = models.get_model(args)

classification_head = nn.Linear(512, 2)
if USE_GPU:
    model = model.cuda()
    classification_head = classification_head.cuda()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
optimizer_head = optim.Adam(classification_head.parameters(), lr=1e-3)
loss_function = nn.NLLLoss()
kd_loss_function = nn.MSELoss()


print('Training...')
for epoch in range(EPOCHS_first_stage+EPOCHS_second_stage):
    avg_loss, acc = train_epoch_progress(
        model, classification_head, train_iter, loss_function, optimizer, optimizer_head, epoch)
    tqdm.write('Train: loss %.2f acc %.1f' % (avg_loss, acc*100))
dev_acc = evaluate_second_stage(
    model, classification_head, dev_iter, loss_function, 'Dev')


torch.save(model, 'model.pkl')
torch.save(classification_head, 'classification_head.pkl')

test_acc = evaluate_second_stage(
    model, classification_head, test_iter, loss_function, 'Final Test')
