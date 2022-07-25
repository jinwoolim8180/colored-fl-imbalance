import argparse

def parser():
    parser = argparse.ArgumentParser(description='Some hyperparameters')

    parser.add_argument('--round', type=int, default=500,
                        help='number of rounds')
    parser.add_argument('--dataset',  type=str, default='celeba',
                        help='type of dataset')
    parser.add_argument('--target', type=str, default='Smiling')
    parser.add_argument('--batch_size', type=int, default=1024, 
                        help='size of batch')
    parser.add_argument('--lr', type=float, default=0.1,
                        help='learning rate')
    parser.add_argument('--lr_decay', type=float, default=0.1,
                        help='0.992, 0.998')
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--bias_level', type=int, default=0,
                        help='level of bias(higher means more biased)')
    parser.add_argument('--group_imbalance', type=int, default=1,
                        help='level of imbalance in group(higher means more imbalance)')
    parser.add_argument('--focal_loss', type=float, default= 0,
                        help='gamma of focal loss')
    parser.add_argument('--cutting', type=int, default= 0,
                        help='gamma of focal loss')
    parser.add_argument('--diversity_ratio', type=str, default="0.5pct")
                          
    return parser

if __name__ == "__main__":
    args = parser()
    print(args)
