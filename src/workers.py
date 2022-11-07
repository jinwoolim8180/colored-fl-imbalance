import copy
from client import Client
import pickle
import torch
from pathlib import Path

from evaluate import evaluate
from dataset import get_dataset
from models import get_model

def gpu_train_worker(trainQ, resultQ, device, train_dataset, args):
    model = get_model(args)
    while True:
        msg = trainQ.get()

        if msg == 'kill':
            break
        else:
            lr = args.lr * args.lr_decay_rate**(msg['round'] // args.lr_decay_step_size)
            if args.dataset == 'femnst':
                optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            else:
                optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=args.weight_decay, momentum=0.0)
            client: Client = msg['client']
            model_parameters = msg['model_parameters']
            model.load_state_dict(model_parameters)
            delta, weight, loss = client.train(device, model, train_dataset, optimizer)
            result = {'delta': copy.deepcopy(delta), 'id': client.nodeID, 'weight': weight, 'loss': loss}
            resultQ.put(result)
            del client
            del delta
            del loss
            del weight
        del msg
    del model

def gpu_test_worker(testQ, device, args):
    test_dataset = get_dataset(args, 'test')
    model = get_model(args)
    if not args.resume_checkpoint:
        acc_list = []
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
            with open('../save/checkpoint/result_.pkl', 'wb') as f:
                pickle.dump(acc_list, f)

    Path(f"../save/results/{args.dataset}").mkdir(parents=True, exist_ok=True)

    file_name = f'../save/results/{args.dataset}/R[{args.round}]LR[{args.lr}]LD[{args.lr_decay_rate}]E[{args.local_epoch}]FR[{args.fraction}]C[{args.clients}]WD[{args.weight_decay}]RH[{args.rho}B[{args.beta}]M[{args.n_minority_classes}]C[{args.climb}]L[{args.loss}]'
    if args.loss == 'fl':
        file_name += f'FL[{args.focal_loss}]'
    if args.climb:
        file_name += f'LLR[{args.lambda_lr}]EP[{args.epsilon}]'
    file_name += '.pkl'

    with open(file_name, 'wb') as f:
        pickle.dump(acc_list, f)