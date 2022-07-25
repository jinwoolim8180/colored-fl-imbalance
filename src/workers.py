import copy
from itertools import accumulate
import time
from client import Client
import pickle
import os
import numpy as np
import torch

from metrics import evaluate
from dataset import get_dataset
from models import get_model
from torch.utils.data import DataLoader, Subset

def gpu_train_worker(trainQ, resultQ, device, train_dataset, args):
    model = get_model(args)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=args.lr_decay)

    while True:
        msg = trainQ.get()

        if msg == 'kill':
            break
        else:
            client: Client = msg['client']
            model_parameters = msg['model_parameters']
            model.load_state_dict(model_parameters)
            while scheduler.state_dict()['_step_count'] < msg['round']:
                scheduler.step()
            delta, weight = client.train(device, msg['lr'], model, train_dataset, optimizer)
            result = {'delta': copy.deepcopy(delta), 'id': client.nodeID, 'weight': weight}
            result['prev_grads'] = copy.deepcopy(client.prev_grads)
            resultQ.put(result)
            del client
            del delta
        del msg
    del model

def gpu_test_worker(testQ, device, args):
    test_dataset = get_dataset(args, 'test')
    model = get_model(args)
    if not args.start_from_checkpoint:
        acc_list = []
        fair_list = []
    else:
        with open('../save/checkpoint/result.pkl', 'rb') as f:
            acc_list = pickle.load(f)
    while True:
        msg = testQ.get()

        if msg == 'kill':
            break

        else:
            model_parameters = msg['model_parameters']
            model.load_state_dict(model_parameters)
            round = msg['round']
        
            acc = evaluate(model, test_dataset, device, args)
            acc_list.append(acc)
            print("Round: {} / Avg Acc: {} / Label Acc: {}".format(round, acc[0], acc[1]))

        if round == args.checkpoint_round:
            with open('../save/checkpoint/result.pkl', 'wb') as f:
                pickle.dump(acc_list, f)

    file_name = '../save/results/{}/R[{}]FA[{}]FLA[{}]LR[{}]LD[{}]E[{}]FR[{}]C[{}]T[{}]WD[{}]D[{}]_dynamic.pkl'.\
        format(args.dataset, args.round, args.feddyn_alpha, args.feddyn_lambda, args.lr, args.lr_decay, args.local_epoch, args.fraction, args.clients, args.target, args.weight_decay, args.diversity_ratio)

    with open(file_name, 'wb') as f:
        pickle.dump(acc_list, f)

    torch.save(model.state_dict(), '../save/models/{}_fl.pt'.format(args.dataset, args.round))