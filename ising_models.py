import torch
import itertools
import numpy as np
import igraph as ig
import torch.nn as nn
import torch.distributions as dists
import matplotlib.pyplot as plt

class LatticeIsingModel(nn.Module):
    """
        Ising model is defined as
            p(x) \propto exp(x^T J x + b^T x)
    """

    def __init__(self, dim, init_sigma=.15, init_bias=0., learn_G=False, learn_sigma=False, learn_bias=False,
                 lattice_dim=2):
        super().__init__()
        g = ig.Graph.Lattice(dim=[dim] * lattice_dim, circular=True)  # Boundary conditions
        A = np.asarray(g.get_adjacency().data)  # g.get_sparse_adjacency()
        self.G = nn.Parameter(torch.tensor(A).float(), requires_grad=learn_G)
        self.sigma = nn.Parameter(torch.tensor(init_sigma).float(), requires_grad=learn_sigma)
        self.bias = nn.Parameter(torch.ones((dim ** lattice_dim,)).float() * init_bias, requires_grad=learn_bias)
        self.init_dist = dists.Bernoulli(logits=2 * self.bias)
        self.data_dim = dim ** lattice_dim

    def init_sample(self, n):
        return self.init_dist.sample((n,))

    @property
    def J(self):
        return self.G * self.sigma

    def forward(self, x):
        """
        :param x: (batch_size, data_dim)
        :return: (batch_size,) log probability log p(x) \propto x^T J x + b^T x
        """
        if len(x.size()) > 2:
            x = x.view(x.size(0), -1)

        x = (2 * x) - 1

        xg = x @ self.J
        xgx = (xg * x).sum(-1)
        b = (self.bias[None, :] * x).sum(-1)
        return xgx + b
    
    def log_prob(self, x):
        return self.forward(x.float())

    @torch.no_grad()
    def get_diff_log_prob(self, x, t):
        """
        get the diff log ratio: [\log p_t (y) - \log p_t (x)]_{y=N(x)}, where N(x) is the neighbor of x
        :param x: (batch_size, data_dim)
        :param t: (batch_size)
        :return: (batch_size, data_dim)
        """
        x_ = ((2 * x) - 1).float()
        J = self.J
        diff = x_ @ (J + J.T) - torch.diag(J)[None, :] + self.bias[None, :]
        diff = t[:, None] * (- 2. * x_) * diff

        B, D = x.size()
        log_ratio = torch.zeros(B, D, 2).to(x.device)
        log_ratio.scatter_(-1, 1-x.unsqueeze(-1), diff.unsqueeze(-1))
        return log_ratio.detach()

def load_ising_models(args):
    dim = args.ising_dim
    sigma = args.ising_sigma
    bias = args.ising_bias

    model = LatticeIsingModel(dim=dim, init_sigma=sigma, init_bias=bias)
    args.discrete_dim = dim**2
    model = model.to(args.device)
    model.eval()

    if args.ising_dim == 5:
        A = model.J
        b = model.bias
        lst=torch.tensor(list(itertools.product([-1.0, 1.0], repeat=dim**2))).to(args.device)
        f = lambda x: torch.exp((x @ A * x).sum(-1)  + torch.sum(b*x,dim=-1))
        flst = f(lst)
        Z = flst.sum()
        plst = flst / Z
        gt_mean = torch.sum(lst*plst.unsqueeze(1).expand(-1,lst.size(1)),0)
        args.gt_mean = gt_mean
        del lst, flst, plst, Z
    
    return model
