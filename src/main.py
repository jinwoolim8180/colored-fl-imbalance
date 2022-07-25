from typing import List
import torch
import numpy as np
import time
import arguments
import copy
import random
import os
import torch.multiprocessing as mp
import queue

from client import Client
from server import Server
from workers import *
from dataset import split_client_indices
import models
import torch.backends.cudnn as cudnn

torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)
np.random.seed(0)
cudnn.benchmark = False
cudnn.deterministic = True
random.seed(0)

if __name__ == "__main__":
    mp.set_start_method('spawn')
    if torch.cuda.is_available():
        n_devices = torch.cuda.device_count()
        devices = [torch.device("cuda:{}".format(i)) for i in range(n_devices)]
        cuda = True
        
    else:
        n_devices = 1
        devices = [torch.device('cpu')]
        cuda = False

    os.environ["OMP_NUM_THREADS"] = "1"

    parser = arguments.parser()
    parser.add_argument('--clients', type=int, default=2)
    parser.add_argument('--fraction', type=float, default=1.0)
    parser.add_argument('--local_epoch', type=int, default=4)
    parser.add_argument('--distribution', type=str, default='seperate')
    parser.add_argument('--n_procs', type=int, default=1)
    parser.add_argument('--feddyn_alpha', type=float, default=0.01)
    parser.add_argument('--feddyn_lambda', type=float, default=0.0)
    parser.add_argument('--q', type=float, default=0)
    parser.add_argument('--L', type=float, default=1)
    parser.add_argument('--noniid_beta', type=float, default= 0.3)
    parser.add_argument('--checkpoint_round', type=int, default=300)
    parser.add_argument('--start_from_checkpoint', type=int, default=0)

    args = parser.parse_args()
    
    print("> Setting:", args)

    n_train_processes = n_devices * args.n_procs
    trainIDs = ["Train Worker : {}".format(i) for i in range(n_train_processes)]
    trainQ = mp.Queue()
    resultQ = mp.Queue()
    testQ = mp.Queue()

    # processes list
    processes = []

    train_dataset = get_dataset(args, 'train')
    if not args.start_from_checkpoint:
        indices = split_client_indices(train_dataset, args)

        # create pseudo server
        server = Server(args)

        # for FedDyn optimizer
        prev_grads = {}
        model = models.get_model(args)
        with torch.no_grad():
            prev_grads = {k: torch.zeros(v.numel()) for (k, v) in model.named_parameters() if v.requires_grad}

        # create pseudo clients
        clients: List[Client] = []
        for i in range(10):
            clients.append(Client(i, indices[i], copy.deepcopy(prev_grads), args))
    
    else:
        with open('../save/checkpoint/state.pkl', 'rb') as f:
            checkpoint = pickle.load(f)
            clients = checkpoint['clients']
            server = checkpoint['server']

    # create train processes
    for i, trainID in enumerate(trainIDs):
        p = mp.Process(target=gpu_train_worker, args=(trainQ, resultQ, devices[i%n_devices], train_dataset, args))
        p.start()
        processes.append(p)
        time.sleep(0.1)
    
    # create test process
    p = mp.Process(target=gpu_test_worker, args=(testQ, devices[0], args))
    p.start()
    processes.append(p)
    time.sleep(5)

    lr = args.lr
    n_trainees = int(len(clients)*args.fraction)

    for roundIdx in range(1, args.round+1):
        if args.start_from_checkpoint:
            roundIdx += args.start_from_checkpoint
        cur_time = time.time()
        print(f"Round {roundIdx}", end=', ')
        # if roundIdx == 200:
        #     server.lamb = [0 for _ in range(10)]

        # Randomly selected clients
        if roundIdx == 1:
            trainees = [clients[i] for i in np.random.choice(np.arange(len(clients)), len(clients), replace=False)]
        else:
            trainees = [clients[i] for i in np.random.choice(np.arange(len(clients)), n_trainees, replace=False)]
        
        count = 0
        for i, client in enumerate(trainees):
            for _ in range(args.clients // 10):
                # download model
                model_parameters = server.model_parameters
                if roundIdx != 1:
                    client.prev_grads = copy.deepcopy(server.prev_grads[client.pv_uniq])
                    client.optimizer_params = copy.deepcopy(server.optimizer_params[client.pv_uniq])
                    client.scheduler_params = copy.deepcopy(server.scheduler_params[client.pv_uniq])
                    client.lamb = server.lamb[client.pv_uniq]
                count += 1
                trainQ.put({'round': roundIdx, 'type': 'train', 'client': copy.deepcopy(client), 'lr':lr, 'model_parameters': copy.deepcopy(model_parameters)})
        for _ in range(count):
            msg = resultQ.get()
            delta = msg['delta']
            weight = msg['weight']
            client_id = msg['id']
            client = clients[client_id]
            pv_uniq = msg['pv_uniq']
            client.pv_uniq = pv_uniq
            optimizer_params = msg['optimizer_params']
            scheduler_params = msg['scheduler_params']

            # set prev_grads for FedDyn regularizer
            prev_grads = msg['prev_grads']
            # client.prev_grads = prev_grads
            
            # upload weights to server
            server.update_client_param(delta, client_id, prev_grads, pv_uniq, optimizer_params, scheduler_params)
            del msg
        # aggregate uploaded weights
        server.aggregate()
        if roundIdx % 1 == 0:
            testQ.put({'round': roundIdx, 'model_parameters': copy.deepcopy(server.model_parameters)})
        # file_name = '../save/models/round:{}_feddyn:{}_debias:{}.pt'.format(roundIdx, args.feddyn, args.debias)
        # torch.save(server.model_parameters, file_name)
        print(f"Elapsed Time : {(time.time()-cur_time):.1f}")

        if roundIdx == args.checkpoint_round:
            with open('../save/checkpoint/state.pkl', 'wb') as f:
                pickle.dump({'clients': clients, 'server': server}, f)
        
        if roundIdx == args.round:
            break

    for _ in range(n_train_processes):
        trainQ.put('kill')
    testQ.put('kill')

    # Train finished
    time.sleep(5)

    # # create test process
    # p = mp.Process(target=gpu_test_worker, args=(testQ, devices[0], args))
    # p.start()
    # processes.append(p)
    # time.sleep(5)

    # # Test start
    # for i in range(preQ.qsize()):
    #     testQ.put(preQ.get())

    for p in processes:
        p.join()
