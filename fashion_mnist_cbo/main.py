import os
import torch
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter
from model import FashionCNN, FashionFC
from solver import CBOSolver
import argparse
from tqdm import tqdm

EPOCH_REPORT_TEMPLATE = """
[train] train_loss: {train_loss}
[train] train_point_acc: {train_acc}
[val]   val_loss: {val_loss}
[val]   val_point_acc: {val_acc}
"""

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, help='number of epochs', default=100)
parser.add_argument('--restarts', type=int, help='number of epochs', default=10)
parser.add_argument('--particle_batch_size', type=int, help='particle batch size', default=10)
parser.add_argument('--data_batch_size', type=int, help='data batch size', default=50)
parser.add_argument('--N', type=int, help='number of particles', default=100)
parser.add_argument('--lmda', type=float, help='drift rate', default=1)
parser.add_argument('--gamma', type=float, help='learning rate', default=0.1)
parser.add_argument('--sigma', type=float, help='noise rate', default=np.sqrt(0.1))
parser.add_argument('--beta', type=float, help='weight constant', default=1)
parser.add_argument('--eps', type=float, help='stoping criterion', default=1e-10)
parser.add_argument('--exp', type=str, help='log directory', default='exp_logs')

args = parser.parse_args()

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,),(0.5,),)])

train_set = datasets.FashionMNIST('./data', download=True, train=True, transform=transform)
test_set = datasets.FashionMNIST('./data', download=True, train=False, transform=transform)

indices = list(range(len(train_set)))
np.random.shuffle(indices)
split = int(np.floor(0.8 * len(train_set)))
train_sample = SubsetRandomSampler(indices[:split])
valid_sample = SubsetRandomSampler(indices[split:])

train_loader_eval = torch.utils.data.DataLoader(train_set, sampler=train_sample, batch_size=args.data_batch_size)
train_loader = torch.utils.data.DataLoader(train_set, sampler=train_sample, batch_size=args.data_batch_size, drop_last=True)
val_loader = torch.utils.data.DataLoader(train_set, sampler=valid_sample, batch_size=args.data_batch_size)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.data_batch_size)

os.makedirs(os.path.join(args.exp, 'tensorboard'), exist_ok=True)
logger = SummaryWriter(os.path.join(args.exp, 'tensorboard'))
net = FashionFC()
net.eval()
solver = CBOSolver(net, N=args.N, lmda=args.lmda, 
                   sigma=args.sigma, gamma=args.gamma, 
                   beta=args.beta, logger=logger, 
                   eps=args.eps, batch_size=args.particle_batch_size)
epoch_break = False
for epoch in range(1, args.epoch + 1):
    print("Epoch {} starting ...".format(epoch))
    solver.sigma = solver.sigma / np.log(epoch + 1)
    for data, target in tqdm(train_loader):
        stop_crit = solver.step(data, target)
        '''if stop_crit:
            epoch_break=True
            break'''
    solver.evaluate(train_loader_eval, split="train")
    solver.evaluate(val_loader, split="val")
    stats = solver.dump_logs(epoch)
    print("-----------------Epoch {} summary----------------------".format(epoch))
    print(EPOCH_REPORT_TEMPLATE.format(
        train_loss=stats["train_loss"][-1], 
        val_loss=stats["val_loss"][-1], 
        train_acc=stats["train_acc"][-1], 
        val_acc=stats["val_acc"][-1]
        )
    )
    '''if epoch_break:
        break'''
