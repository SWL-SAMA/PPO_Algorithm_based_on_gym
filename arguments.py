import argparse
parser=argparse.ArgumentParser()
parser.add_argument('--run_mode',type=str,default='test',help='Train or test PPO module')
parser.add_argument('--train_num',type=int,default=5000,help='Train epochs')
args = parser.parse_args()