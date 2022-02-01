import torch
import torch.nn.functional as F
from model import FashionCNN
import timeit
import numpy as np
from copy import deepcopy
from tqdm import tqdm

class CBOSolver(object):

    def __init__(self, model, N=100, lmda=1, sigma=np.sqrt(0.1), gamma=0.1, beta = 50, logger=None, eps=1e-4):
        self.N = N
        self.lmda = lmda
        self.sigma = sigma
        self.gamma = gamma
        self.beta = beta
        self.eps = eps
        self.particles = {}
        self.consensus_weights = {}
        self.model = model
        self.remainder_idx = []
        self.losses = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": []
        }
        self.logger = logger
        self.d = 0
        for key, val in model.state_dict().items():
            self.d += torch.prod(torch.tensor(val.shape)).item()
            if key not in self.particles:
                self.particles[key] = []
            for _ in range(N):
                self.particles[key].append(torch.randn(val.shape).to(val.device).unsqueeze(0))
            self.particles[key] = torch.cat(self.particles[key], dim=0)

    @torch.no_grad()    
    def update_consensus(self, idx, losses):
        prev_consensus = deepcopy(self.consensus_weights)
        weights = torch.exp(-1 * self.beta * torch.tensor(losses))
        cons_norm = 0
        for key in self.particles.keys():
            self.consensus_weights[key] = torch.sum((self.particles[key][idx].T * weights).T, dim=0) / torch.sum(weights)
            if len(prev_consensus.keys()) != 0:
                cons_norm += torch.square(self.consensus_weights[key] - prev_consensus[key]).sum().item()
        if len(prev_consensus.keys()) == 0:
            cons_norm = np.inf

        return (cons_norm / self.d) <= self.eps

    @torch.no_grad()  
    def update_particle(self, idx):
        for key in self.particles.keys():
            z = torch.randn(self.particles[key][idx].shape).to(self.particles[key][idx].device)
            self.particles[key][idx] = self.particles[key][idx] -\
                 self.lmda * self.gamma * (self.particles[key][idx] - self.consensus_weights[key]) +\
                 self.sigma * np.sqrt(self.gamma) * (self.particles[key][idx] - self.consensus_weights[key]) * z
    
    def get_state_dict(self, i):
        weight = {}
        for key, val in self.particles.items():
            weight[key] = val[i]

        return weight

    @torch.no_grad()
    def step(self, x, targets):
        if len(np.unique(self.remainder_idx)) < len(x):
            self.remainder_idx.extend(list(range(self.N)))
        unique_idx, counts = np.unique(self.remainder_idx, return_counts=True)
        counts = counts / counts.sum()
        idx = np.random.choice(unique_idx, size=len(x), replace=False, p=counts)
        self.remainder_idx = list(set(self.remainder_idx).difference(set(idx)))
        losses = []
        for ind, i in enumerate(idx):
            self.model.load_state_dict(self.get_state_dict(i))
            out = self.model(x[ind])
            loss = F.cross_entropy(out, targets[ind])
            losses.append(loss)
        stop_crit = self.update_consensus(idx, losses)
        self.update_particle(idx)

        return stop_crit

    @torch.no_grad()
    def evaluate(self, dataloader, split="train"):
        self.model.load_state_dict(self.consensus_weights)
        total_loss = 0
        num_correct = 0
        num_data = 0
        print("Starting evaluation on {} set ...".format(split))
        for data, target in tqdm(dataloader):
            out = self.model(data)
            loss = F.cross_entropy(out, target, reduction='sum')
            total_loss += loss.item()
            pred_c = torch.argmax(out, dim=1)
            num_correct += (pred_c == target).sum().item()
            num_data += data.shape[0]
        total_loss = total_loss / num_data
        acc = num_correct / num_data
        self.losses["{}_loss".format(split)].append(total_loss)
        self.losses["{}_acc".format(split)].append(acc)
    
    def dump_logs(self, epoch):
        self.logger.add_scalars(
            "log/loss",
            {
                "train": self.losses["train_loss"][epoch-1],
                "val": self.losses["val_loss"][epoch-1]
            },
            epoch
        )
        self.logger.add_scalars(
            "log/acc",
            {
                "train_acc": self.losses["train_acc"][epoch-1],
                "val_acc": self.losses["val_acc"][epoch-1]
            },
            epoch
        )

        return self.losses

if __name__ == '__main__':
    net = FashionCNN()
    time_start = timeit.default_timer()
    solver = CBOSolver(net)
    print('Particle init took: {} seconds'.format(timeit.default_timer() - time_start))