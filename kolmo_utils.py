import torch
import numpy as np
from itertools import cycle
import torch.nn.functional as F
from einops import rearrange, repeat
from torch.utils.data import Dataset, DataLoader
from torch.distributions.categorical import Categorical

class UniformInitialDist:
    def __init__(self, D, S, device):
        self.D = D
        self.S = S
        self.device = device

    def sample(self, n):
        return torch.randint(0, self.S, (n, self.D)).to(self.device)
    
    def log_prob(self, x):
        return np.log((1 / self.S)) * self.D

@torch.no_grad()
def make_forward_step_probs(B, D, S, t_c, t_n, xt, model, args):
    delta_xt = torch.zeros((B, D, S)).to(args.device)
    delta_xt = delta_xt.scatter_(-1, xt[:, :, None], 1.0)
    
    dt = t_n - t_c
    t_ = t_c * torch.ones((B,)).to(args.device)
    G_t = model(xt, t_)
    R_t = F.relu(G_t)
    step_probs = delta_xt + (R_t * dt)

    step_probs = step_probs.clamp(max=1.0)
    step_probs.scatter_(-1, xt[:, :, None], 0.0)
    step_probs.scatter_(-1, xt[:, :, None], (1.0 - step_probs.sum(dim=-1, keepdim=True))) 
    step_probs = step_probs.clamp(min=0)
    return step_probs

def generate_samples_using_euler_method(v_theta, initial_samples, ts, args):
    B, D = initial_samples.shape
    S = args.vocab_size
    samples = initial_samples
    t_prev = ts[:-1]
    t_next = ts[1:]

    samples_list = [initial_samples]
    for t_p, t_n in zip(t_prev, t_next):
        forward_step_probs = make_forward_step_probs(B, D, S, t_p.item(), t_n.item(), samples, v_theta, args)
        forward_dist = Categorical(forward_step_probs)
        samples = forward_dist.sample()
        samples_list.append(samples)

    samples = torch.stack(samples_list, dim=0)
    return samples

@torch.no_grad()
def generate_train_data_using_rate_matrix(model, num_samples, ts, sample_initial, time_derivative_log_density, target_density, args):

    initial_samples = sample_initial(num_samples)
    samples = generate_samples_using_euler_method(model, initial_samples, ts, args)  # (T, N, D)
    samples = rearrange(samples, 't n d -> n t d')    # (N, T, D)

    N, T, D = samples.shape
    ts_ = ts.to(samples.device)
    ts = repeat(ts_, 't -> n t', n=N)
    ts = rearrange(ts, 'n t -> (n t)')
    samples = rearrange(samples, 'n t d -> (n t) d')

    dt_log_unormalised_density = time_derivative_log_density(samples, ts)
    dt_log_unormalised_density = rearrange(dt_log_unormalised_density, '(n t) -> n t', n=N)
   
    G_t = model(samples, ts)
    G_t = G_t.scatter_(-1, samples[..., None], 0.0)
    G_t_plus = F.relu(G_t)
    neg_G_t_plus = F.relu(-G_t)
    G_t_plus = rearrange(G_t_plus, '(n t) d s -> n t d s', n=N)
    neg_G_t_plus = rearrange(neg_G_t_plus, '(n t) d s -> n t d s', n=N)


    log_ratio = target_density.get_diff_log_prob(samples, ts)
    log_ratio = log_ratio.scatter_(-1, samples[..., None], 0.0)
    log_ratio.clamp_(max=5)
    log_ratio = rearrange(log_ratio, '(n t) d s -> n t d s', n=N)

    stein_eq = (G_t_plus - neg_G_t_plus * log_ratio.exp()).reshape(N, T, -1).sum(dim=-1)    # (N, T)
    dt_log_Zt_ = dt_log_unormalised_density + stein_eq
    dt_log_Zt_ = dt_log_Zt_.detach()
    dt_log_Zt = dt_log_Zt_.mean(dim=0)


    samples = rearrange(samples, '(n t) d -> n t d', n=N).cpu()
    dt_log_Zt = dt_log_Zt.cpu() # (T)
    ts_ = ts_.cpu() # (T)

    dataset = getattr(args, 'dataset', OnlineData(args.N, args.T, update_dt_log_Zt=False))
    dataset.update_data(ts_, samples, dt_log_Zt)
    args.dataset = dataset

    dataset_iterator = iter(cycle(DataLoader(dataset, batch_size=args.batch_size, shuffle=True)))
    return dataset_iterator

@torch.no_grad()
def make_forward_step_probs_with_weight_updating(weights, t_c, t_n, xt, model, time_derivative_log_density, target_density, args):
    B, D = xt.shape
    S = args.vocab_size
    
    delta_xt = torch.zeros((B, D, S)).to(args.device)
    delta_xt = delta_xt.scatter_(-1, xt[:, :, None], 1.0)
    
    dt = t_n - t_c
    t_ = t_c * torch.ones((B,)).to(args.device)
    G_t = model(xt, t_)
    G_t_plus = F.relu(G_t)
    neg_G_t_plus = F.relu(-G_t)
    step_probs = delta_xt + (G_t_plus * dt)

    step_probs = step_probs.clamp(max=1.0)
    step_probs.scatter_(-1, xt[:, :, None], 0.0)
    step_probs.scatter_(-1, xt[:, :, None], (1.0 - step_probs.sum(dim=-1, keepdim=True))) 
    step_probs = step_probs.clamp(min=0)

    dt_log_unormalised_density = time_derivative_log_density(xt, t_)
    log_ratio = target_density.get_diff_log_prob(xt, t_)
    stein_eq = (G_t_plus - neg_G_t_plus * log_ratio.exp()).reshape(B, -1).sum(dim=-1)
    At = stein_eq + dt_log_unormalised_density
    weights = weights + dt * At

    return step_probs, weights

@torch.no_grad()
def generate_samples_with_importance_weights(model, num_samples, sample_initial, dt_logpt, target_density, ts, args, return_all=False):
    samples = sample_initial(num_samples)
    weights = torch.zeros(num_samples,).to(args.device)
    t_prev = ts[:-1]
    t_next = ts[1:]

    samples_list = [samples]
    weights_list = [weights]
    for t_p, t_n in zip(t_prev, t_next):
        forward_step_probs, weights = make_forward_step_probs_with_weight_updating(weights, t_p.item(), t_n.item(), samples, model, dt_logpt, target_density, args)
        forward_dist = Categorical(forward_step_probs)
        samples = forward_dist.sample()
        samples_list.append(samples)
        weights_list.append(weights)
    
    if return_all:
        samples = torch.stack(samples_list, dim=0)
        weights = torch.stack(weights_list, dim=0)
        return samples, weights

    return samples, weights # NOTE: weights are not normalised


class DataBuffer():
    def __init__(self, bs, update_dt_log_Zt=True):
        self.ts = []
        self.samples = []
        self.time_idx = []
        self.dt_log_Zt = []
        self.max_size = 1024 // bs
        self.update_dt_log_Zt = update_dt_log_Zt

    def update_buffer(self, ts, samples, time_idx, dt_log_Zt):
        self.ts.append(ts)
        self.samples.append(samples)
        self.time_idx.append(time_idx)
        if self.update_dt_log_Zt:
            self.dt_log_Zt.append(dt_log_Zt)
        else:
            self.dt_log_Zt = dt_log_Zt

        if len(self.ts) > self.max_size:
            self.ts.pop(0)
            self.samples.pop(0)
            self.time_idx.pop(0)
            if self.update_dt_log_Zt: self.dt_log_Zt.pop(0)

    def get_training_data(self):
        ts = torch.cat(self.ts, dim=0)
        samples = torch.cat(self.samples, dim=0)
        time_idx = torch.cat(self.time_idx, dim=0)
        if self.update_dt_log_Zt:
            dt_log_Zt = torch.cat(self.dt_log_Zt, dim=0).mean(dim=0)
        else:
            dt_log_Zt = self.dt_log_Zt

        ts = rearrange(ts, 'n t -> (n t)')
        time_idx = rearrange(time_idx, 'n t -> (n t)')
        samples = rearrange(samples, 'n t d -> (n t) d')

        perm = torch.randperm(samples.size(0))
        ts = ts[perm]
        time_idx = time_idx[perm]
        samples = samples[perm]

        return ts, samples, time_idx, dt_log_Zt

class OnlineData(Dataset):
    def __init__(self, bs, T, update_dt_log_Zt=True):
        self.T = T
        self.ts = None
        self.samples = None
        self.time_idx = None
        self.dt_log_Zt = None
        self.buffer = DataBuffer(bs, update_dt_log_Zt)

    def update_data(self, ts, samples, dt_log_Zt):
        N, T, D = samples.shape
        time_idx = torch.arange(self.T)
        time_idx = repeat(time_idx, 't -> n t', n=N)
        ts = repeat(ts, 't -> n t', n=N)

        self.buffer.update_buffer(ts, samples, time_idx, dt_log_Zt)
        ts, samples, time_idx, dt_log_Zt = self.buffer.get_training_data()

        self.ts = ts
        self.samples = samples
        self.time_idx = time_idx
        self.dt_log_Zt = dt_log_Zt

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        time_idx = self.time_idx[idx]

        data = self.samples[idx]
        ts = self.ts[idx]
        dt_log_Zt = self.dt_log_Zt[time_idx]
        return ts, data, dt_log_Zt
