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

def optimizer_to(optim, device):
    for param in optim.state.values():
        # Not sure there are any global tensors in the state dict
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)

def gpu_train_worker(trainQ, resultQ, device, train_dataset, args):
    model = get_model(args)
    if args.dataset == 'cmnist':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=args.lr_decay)

    roundIdx = 0
    while True:
        msg = trainQ.get()

        if msg == 'kill':
            break
        else:
            client: Client = msg['client']
            model_parameters = msg['model_parameters']
            # if args.fedbn == 1:
                # model_parameters.update(client.bn_params)
            model.load_state_dict(model_parameters)
            if client.optimizer_params == None:
                optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
                scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=150, gamma=args.lr_decay)
            else:
                optimizer.load_state_dict(client.optimizer_params)
                optimizer_to(optimizer, device)
                scheduler.load_state_dict(client.scheduler_params)
                while scheduler.state_dict()['_step_count'] < msg['round']:
                    scheduler.step()
            delta, weight = client.train(device, msg['lr'], model, train_dataset, optimizer)
            optimizer_to(optimizer, torch.device('cpu'))
            result = {'delta': copy.deepcopy(delta), 'id': client.nodeID, 'weight': weight, 'pv_uniq': client.pv_uniq, 'optimizer_params': copy.deepcopy(optimizer.state_dict()), 'scheduler_params': copy.deepcopy(scheduler.state_dict())}
            result['prev_grads'] = copy.deepcopy(client.prev_grads)
            # if args.fedbn == 1:
                # result['bn_params'] = copy.deepcopy(client.bn_params)
            resultQ.put(result)

            # if roundIdx != msg['round']:
            #     roundIdx = msg['round']
            #     scheduler.step()
            #     if roundIdx % 100 == 0:

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
            result = pickle.load(f)
            acc_list = result['acc']
            fair_list = result['fair']
    while True:
        msg = testQ.get()

        if msg == 'kill':
            break

        else:
            model_parameters = msg['model_parameters']
            model.load_state_dict(model_parameters)
            round = msg['round']
        
            acc, fair = evaluate(model, test_dataset, device, args)
            acc_list.append(acc)
            fair_list.append(fair)
            print("Round: {} / Accuracy: {} / Fairness: {}".format(round, acc[0], fair))

        if round == args.checkpoint_round:
            with open('../save/checkpoint/result.pkl', 'wb') as f:
                pickle.dump({'acc': acc_list, 'fair': fair_list}, f)

    file_name = '../save/results/{}/R[{}]FA[{}]FLA[{}]LR[{}]LD[{}]E[{}]FR[{}]C[{}]T[{}]WD[{}]D[{}]_dynamic.pkl'.\
        format(args.dataset, args.round, args.feddyn_alpha, args.feddyn_lambda, args.lr, args.lr_decay, args.local_epoch, args.fraction, args.clients, args.target, args.weight_decay, args.diversity_ratio)

    with open(file_name, 'wb') as f:
        pickle.dump({'acc': acc_list, 'fair': fair_list}, f)

    torch.save(model.state_dict(), '../save/models/{}_fl.pt'.format(args.dataset, args.round))