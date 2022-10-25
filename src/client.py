import torch
import torch.nn as nn
import copy
from torch.utils.data import DataLoader, Subset
from loss import BalancedSoftmax, FocalLoss

class Client:
    def __init__(self, nodeID, node_indices, args):
        self.nodeID = nodeID
        self.node_indices = node_indices
        self.args = args

    def train(self, device, model: nn.Module, total_train_dataset, optimizer):
        model.to(device)

        train_loader = DataLoader(Subset(total_train_dataset, self.node_indices), self.args.batch_size, shuffle=True, num_workers=0)

        criterion = nn.CrossEntropyLoss()
        test_loss = 0.
        with torch.no_grad():
            model.eval()
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                test_loss += criterion(outputs, labels)

        old_param = copy.deepcopy(model.state_dict())
        labels = total_train_dataset.targets[self.node_indices]

        if self.args.loss == 'ce':
            criterion = nn.CrossEntropyLoss()
        
        elif self.args.loss == 'fl':
            criterion = FocalLoss(gamma=self.args.focal_loss)

        elif self.args.loss == 'bs':
            n_samples = [(labels == i).count_nonzero() for i in labels.unique()]
            criterion = BalancedSoftmax(n_samples)


        for _ in range(self.args.local_epoch):
            for inputs, labels in train_loader:
                model.train()
                optimizer.zero_grad()
                inputs, labels = inputs.to(device), labels.to(device)
                output = model(inputs)
                loss: torch.Tensor = criterion(output, labels)   # true loss
                loss.backward()
                optimizer.step()

        delta = {k: v.sub(old_param[k]).to(torch.device('cpu')) for k, v in model.state_dict().items()}
        if self.args.weighted_avg == 0:
            weight = len(self.node_indices)
        else:
            weight = 1
        model.to(torch.device('cpu'))
        return delta, weight, test_loss.to(torch.device('cpu'))
