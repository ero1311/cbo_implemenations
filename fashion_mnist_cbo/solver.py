import torch
import torch.nn.functional as F
from model import FashionCNN
import timeit
import numpy as np
from copy import deepcopy
from tqdm import tqdm

class CBOSolver(object):

    def __init__(self, model, N=100, lmda=1, sigma=np.sqrt(0.1), gamma=0.1, beta = 50, logger=None, eps=1e-4, batch_size=10):
        self.N = N
        self.lmda = lmda
        self.sigma = sigma
        self.gamma = gamma
        self.beta = beta
        self.eps = eps
        self.batch_size=batch_size
        self.particles = {}
        self.consensus_weights = {}
        self.model = model
        self.optimizer_sgd = torch.optim.SGD(self.model.parameters(), lr=0.01)
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
        '''prev_consensus = deepcopy(self.consensus_weights)
        weights = torch.exp(-1 * self.beta * torch.tensor(losses))
        cons_norm = 0
        for key in self.particles.keys():
            self.consensus_weights[key] = torch.sum((self.particles[key][idx].T * weights).T, dim=0) / torch.sum(weights)
            if len(prev_consensus.keys()) != 0:
                cons_norm += torch.square(self.consensus_weights[key] - prev_consensus[key]).sum().item()
        if len(prev_consensus.keys()) == 0:
            cons_norm = np.inf'''
        
        prev_consensus = deepcopy(self.consensus_weights)
        best_id = idx[np.argmin(losses)]
        cons_norm = 0
        for key in self.particles.keys():
            self.consensus_weights[key] = self.particles[key][best_id].clone()
            if len(prev_consensus.keys()) != 0:
                cons_norm += torch.square(self.consensus_weights[key] - prev_consensus[key]).sum().item()
        if len(prev_consensus.keys()) == 0:
            cons_norm = np.inf

        return (cons_norm / self.d) <= self.eps

    @torch.no_grad()  
    def update_particle(self, idx, brown_mot=False):
        for key in self.particles.keys():
            self.particles[key][idx] = self.particles[key][idx] -\
                 self.lmda * self.gamma * (self.particles[key][idx] - self.consensus_weights[key])
            if brown_mot:
                z = torch.randn(self.particles[key][idx].shape).to(self.particles[key][idx].device)
                self.particles[key][idx] += self.sigma * np.sqrt(self.gamma) * z
    
    def get_state_dict(self, i):
        weight = {}
        for key, val in self.particles.items():
            weight[key] = val[i]

        return weight

    @torch.no_grad()
    def step(self, x, targets):
        if len(np.unique(self.remainder_idx)) < self.batch_size:
            self.remainder_idx.extend(list(range(self.N)))
        unique_idx, counts = np.unique(self.remainder_idx, return_counts=True)
        counts = counts / counts.sum()
        idx = np.random.choice(unique_idx, size=self.batch_size, replace=False, p=counts)
        self.remainder_idx = list(set(self.remainder_idx).difference(set(idx)))
        losses = []
        for i in idx:
            self.model.load_state_dict(self.get_state_dict(i))
            out = self.model(x)
            loss = F.cross_entropy(out, targets)
            losses.append(loss)
        stop_crit = self.update_consensus(idx, losses)
        self.update_particle(idx, stop_crit)

        return stop_crit
    
    def step_sgd(self, x, targets):
        self.model.train()
        self.optimizer_sgd.zero_grad()
        out = self.model(x)
        loss = F.cross_entropy(out, targets)
        loss.backward()
        self.optimizer_sgd.step()
        self.consensus_weights = self.model.state_dict()
        self.model.eval()

        return False


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
            num_correct += torch.eq(pred_c, target).sum().item()
            num_data += data.shape[0]
        print(num_data)
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