import argparse
import os


"""
2022.09.25 —— by yh
修改PATH路径
"""
# PATH = '/home/ubuntu/data/code/deep/DeepDIG-new/DeepDIGCode'

"""
2022.09.25 —— by yh
修改参数
parser.add_argument("--dataset",default='CIFAR10')
parser.add_argument("--pre-trained-model",default='ResNet')
parser.add_argument('--pre-trained-model-input-shape', type=str, default="3;32;32",
                    help='shape of the input data to pre trained model')
parser.add_argument("--num-samples-trajectory", type=int, required=False, default=10,
                    help="Number of samples generated in the trajectory line between x(t)=t*x0+(1-t)*x1")
"""
parser = argparse.ArgumentParser(description='Arguments of DeepDIG project')
# parser.add_argument("--project-dir",default=PATH)
parser.add_argument("--dataset",default='CIFAR10',type=str,help = 'MYCIFAR10 \ CIFAR10 \ MNIST')
parser.add_argument("--model",type=str,default='ResNet')
parser.add_argument("--device",default=0,type=int,help = 'cuda device')
parser.add_argument("--batch_size",default=64,type=int,help = 'batch_size')
parser.add_argument("--epochs",type=int,default=50)
parser.add_argument("--num_workers",default=0,type=int,help = 'num_workers')
parser.add_argument("--save_path",type=str,default='origModel')
parser.add_argument('--beta', default=6, type=float,
                    help='regularization, i.e., 1/lambda in TRADES')
# parser.add_argument("--pre-trained-model",default='ResNet')
# parser.add_argument("--dropout", type=float, required=False, default=0.0, help="Ratio of dropout")
# parser.add_argument("--lr", type=float, required=False, default=0.01, help="Learning rate")
# parser.add_argument('--batch_size', type=int, default=32,
#                     help='input batch size for training (default: 128)')
# parser.add_argument("--middle-point-threshold", type=float, required=False, default=0.0001,
#                     help="Parameter beta in Algorithm 1")

# parser.add_argument('--pre-trained-model-input-shape', type=str, default="3;32;32",
#                     help='shape of the input data to pre trained model')

parser.add_argument('--epsilon', default=0.031,
                    help='perturbation')
parser.add_argument('--num-steps', default=20, type=int,
                    help='perturb number of steps')
parser.add_argument('--step-size', default=0.003,
                    help='perturb step size')


args = parser.parse_args()
