from math import cos
from numpy.lib.shape_base import tile
import torch
import torch.nn as nn
import numpy as np
import plotly.graph_objects as go

def plot_ackley(axis_npoints = 1000):
    coords = torch.linspace(-40, 40, axis_npoints)
    grid = torch.cartesian_prod(coords, coords)
    cost = Ackley()
    z = cost(grid)
    fig = go.Figure(data=[go.Surface(z=z.reshape(axis_npoints, axis_npoints), x=coords.numpy(), y=coords.numpy())])
    fig.update_layout(title="Ackley function a=20, b=0.2, c=2pi", autosize=False,
                    width=1000, height=1000,
                    margin=dict(l=65, r=50, b=65, t=90))
    fig.show()

def plot_rastrigin(axis_npoints = 1000):
    coords = torch.linspace(-5.12, 5.12, axis_npoints)
    grid = torch.cartesian_prod(coords, coords)
    cost = Rastrigin()
    z = cost(grid)
    fig = go.Figure(data=[go.Surface(z=z.reshape(axis_npoints, axis_npoints), x=coords.numpy(), y=coords.numpy())])
    fig.update_layout(title="Rastrigin function A=10", autosize=False,
                    width=1000, height=1000,
                    margin=dict(l=65, r=50, b=65, t=90))
    fig.show()

def plot_griewank(axis_npoints = 1000):
    coords = torch.linspace(-5, 5, axis_npoints)
    grid = torch.cartesian_prod(coords, coords)
    cost = Griewank()
    z = cost(grid)
    fig = go.Figure(data=[go.Surface(z=z.reshape(axis_npoints, axis_npoints), x=coords.numpy(), y=coords.numpy())])
    fig.update_layout(title="Griewank function", autosize=False,
                    width=1000, height=1000,
                    margin=dict(l=65, r=50, b=65, t=90))
    fig.show()

def plot_zakharov(axis_npoints = 1000):
    coords = torch.linspace(-5, 10, axis_npoints)
    grid = torch.cartesian_prod(coords, coords)
    cost = Zakharov()
    z = cost(grid)
    fig = go.Figure(data=[go.Surface(z=z.reshape(axis_npoints, axis_npoints), x=coords.numpy(), y=coords.numpy())])
    fig.update_layout(title="Zakharov function", autosize=False,
                    width=1000, height=1000,
                    margin=dict(l=65, r=50, b=65, t=90))
    fig.show()

class Ackley:
    def __init__(self, a=20, b=0.2, c=2*np.pi):
        self.a = a
        self.b = b
        self.c = c

    def __call__(self, x):
        _, d = x.shape
        f = -self.a * torch.exp(-self.b * torch.linalg.norm(x, ord=2, dim=1) / np.sqrt(d)) - torch.exp(torch.sum(torch.cos(self.c * x), dim=1) / d) + self.a + np.exp(1.0)
        return f

class Rastrigin:
    def __init__(self, A=10):
        self.A = A
    
    def __call__(self, x):
        _, d = x.shape
        return self.A * d + torch.sum(torch.square(x) - 10 * torch.cos(2 * np.pi * x), dim=1)

class Griewank:
    def __call__(self, x):
        n, d = x.shape
        denom = torch.repeat_interleave(torch.arange(1, d+1).view(1, -1), repeats=n, dim=0)
        denom = torch.sqrt(denom.float())
        return torch.sum(x / 4000, dim=1) - torch.prod(torch.cos(x / denom), dim=1) + 1

class Zakharov:
    def __call__(self, x):
        n, d = x.shape
        multiplier = torch.repeat_interleave(torch.arange(1, d+1).view(1, -1), repeats=n, dim=0)
        square = torch.square(torch.sum(0.5 * multiplier * x, dim=1))
        return torch.sum(torch.square(x), dim=1) + square + torch.square(square)

