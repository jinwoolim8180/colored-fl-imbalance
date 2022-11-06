import argparse

def parser():
    parser = argparse.ArgumentParser(description='Some hyperparameters')

    parser.add_argument('--round', type=int, default=500, help='number of rounds')
    parser.add_argument('--dataset',  type=str, default='cifar10', help='type of dataset')
    parser.add_argument('--batch_size', type=int, default=1024, help='size of batch')
    parser.add_argument('--lr', type=float, default=0.1, help='learning rate')
    parser.add_argument('--lr_decay_step_size', type=int, default=150, help='')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='0.1, 0.01')
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--focal_loss', type=float, default= 0, help='gamma of focal loss')
    parser.add_argument('--loss', type=str, default='ce', help='ce, fl, bs')
    parser.add_argument('--n_minority_classes', type=int, default=0)
    parser.add_argument('--rho', type=int, default=1)
                          
    return parser

if __name__ == "__main__":
    args = parser()
    print(args)
