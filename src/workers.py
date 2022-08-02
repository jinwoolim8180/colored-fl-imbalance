import copy
from client import Client
import pickle
import torch

from evaluate import evaluate
from dataset import get_dataset
from models import get_model

def gpu_train_worker(trainQ, resultQ, device, train_dataset, args):
    model = get_model(args)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_decay_step_size, gamma=args.lr_decay_rate)

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
            delta, weight = client.train(device, model, train_dataset, optimizer)
            result = {'delta': copy.deepcopy(delta), 'id': client.nodeID, 'weight': weight}
            resultQ.put(result)
            del client
            del delta
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

    file_name = '../save/results/{}/R[{}]FA[{}]FLA[{}]LR[{}]LD[{}]E[{}]FR[{}]C[{}]T[{}]WD[{}]D[{}]_dynamic.pkl'.\
        format(args.dataset, args.round, args.feddyn_alpha, args.feddyn_lambda, args.lr, args.lr_decay, args.local_epoch, args.fraction, args.clients, args.target, args.weight_decay, args.diversity_ratio)

    with open(file_name, 'wb') as f:
        pickle.dump(acc_list, f)