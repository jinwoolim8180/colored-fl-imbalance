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
    if not os.path.isdir('../save/checkpoint/'):
        os.mkdir('../save/checkpoint/')
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
    parser.add_argument('--clients', type=int, default=10)
    parser.add_argument('--fraction', type=float, default=1.0)
    parser.add_argument('--local_epoch', type=int, default=4)
    parser.add_argument('--distribution', type=str, default='iid')
    parser.add_argument('--n_procs', type=int, default=1)
    parser.add_argument('--weighted_avg', type=int, default=0)
    parser.add_argument('--beta', type=float, default= 0.3)
    parser.add_argument('--checkpoint_round', type=int, default=300)
    parser.add_argument('--resume_checkpoint', type=int, default=0)
    parser.add_argument('--lambda_lr', type=float, default=1.0)
    parser.add_argument('--epsilon', type=int, default=1e-03)
    parser.add_argument('--climb', type=int, default=0)

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
    if not args.resume_checkpoint:
        start_round = 0
        indices = split_client_indices(train_dataset, args)

        # create pseudo server
        server = Server(args)

        # for FedDyn optimizer
        model = models.get_model(args)

        # create pseudo clients
        clients: List[Client] = []
        for i in range(args.clients):
            clients.append(Client(i, indices[i], args))
    
    else:
        with open('../save/checkpoint/state.pkl', 'rb') as f:
            checkpoint = pickle.load(f)
            clients = checkpoint['clients']
            server = checkpoint['server']
            start_round = checkpoint['round']

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

    n_trainees = int(len(clients)*args.fraction)

    for roundIdx in range(start_round + 1, args.round+1):
        cur_time = time.time()
        print(f"Round {roundIdx}", end=', ')

        # Randomly selected clients
        trainees = [clients[i] for i in np.random.choice(np.arange(len(clients)), n_trainees, replace=False)]
        
        count = 0
        for i, client in enumerate(trainees):
            model_parameters = server.model_parameters
            count += 1
            trainQ.put({'round': roundIdx, 'type': 'train', 'client': copy.deepcopy(client), 'model_parameters': copy.deepcopy(model_parameters)})

        for _ in range(count):
            msg = resultQ.get()
            delta = msg['delta']
            weight = msg['weight']
            client_id = msg['id']
            loss = msg['loss']

            # upload weights to server
            server.update_client_param(client_id, delta, weight, loss)
            del msg
        # aggregate uploaded weights
        server.aggregate()
        if roundIdx % 5 == 0:
            testQ.put({'round': roundIdx, 'model_parameters': copy.deepcopy(server.model_parameters)})
            print(f"Elapsed Time : {(time.time()-cur_time):.1f}")

        if roundIdx == args.checkpoint_round:
            with open('../save/checkpoint/state_.pkl', 'wb') as f:
                pickle.dump({'clients': clients, 'server': server, 'round': roundIdx}, f)
        
    for _ in range(n_train_processes):
        trainQ.put('kill')
    testQ.put('kill')

    time.sleep(5)

    for p in processes:
        p.join()
