import copy
import torch
import random
import numpy as np
from itertools import chain
from collections import deque
from torch.utils.data import Dataset
from dm_control import suite, manipulation
nn = torch.nn
F = nn.functional
td = torch.distributions


def build_mlp(*sizes, act=nn.ELU):
    mlp = []
    for i in range(1, len(sizes)):
        mlp.append(nn.Linear(sizes[i-1], sizes[i]))
        mlp.append(act())
    return nn.Sequential(*mlp[:-1])


def grads_sum(model):
    s = 0
    for p in model.parameters():
        if p.grad is not None:
            s += p.grad.pow(2).sum().item()
    return np.sqrt(s)


def make_env(name, **kwargs):
    if name in manipulation.ALL:
        return manipulation.load(name)
    domain, task = name.split('_', 1)
    if domain == 'ball':
        domain = 'ball_in_cup'
        task = 'catch'
    return suite.load(domain, task, **kwargs)


def simulate(env, policy, training):
    # done flags might be useful for another learning alg
    obs = env.reset()
    done = False
    observations, actions, rewards, dones, log_probs = [[] for _ in range(5)]
    while not done:
        action, log_prob = policy(obs, training)
        new_obs, reward, done = env.step(action)
        observations.append(obs)
        actions.append(action)
        dones.append([done])
        rewards.append(np.float32([reward]))
        log_probs.append(log_prob)
        obs = new_obs

    tr = dict(
        observations=observations,
        actions=actions,
        rewards=rewards,
        done_flags=dones,
        log_probs=log_probs,
    )
    for k, v in tr.items():
        tr[k] = np.stack(v)
    return tr


# TODO: compressed save
class TrajectoryBuffer(Dataset):
    def __init__(self, capacity, seq_len):
        self._data = deque(maxlen=capacity)
        self.seq_len = seq_len

    def add(self, trajectory):
        if len(trajectory['actions']) < self.seq_len:
            return
        self._data.append(trajectory)

    def __getitem__(self, idx):
        tr = self._data[idx]
        start = random.randint(0, len(tr['actions']) - self.seq_len)
        return {k: v[start:start+self.seq_len] for k, v in tr.items()}

    def __len__(self):
        return len(self._data)

    def sample_subset(self, size):
        idx = np.random.randint(0, len(self._data), size=size)
        return torch.utils.data.Subset(self, idx)


class TruncatedTanhTransform(td.transforms.TanhTransform):
    _lim = .9999

    def _inverse(self, y):
        y = torch.clamp(y, min=-self._lim, max=self._lim)
        return y.atanh()


@torch.no_grad()
def soft_update(target, online, rho):
    for pt, po in zip(target.parameters(), online.parameters()):
        pt.data.copy_(rho * pt.data + (1. - rho) * po.data)


def retrace(resids, cs, discount, disclam):
    cs = torch.cat((cs[1:], torch.ones_like(cs[-1:])))
    cs *= disclam
    resids, cs = map(lambda t: t.flip(0), (resids, cs))
    deltas = []
    last_val = torch.zeros_like(resids[0])
    for r, c in zip(resids, cs):
        last_val = r + last_val * discount * c
        deltas.append(last_val)
    return torch.stack(deltas).flip(0)


def make_param_group(*modules):
    return nn.ParameterList(chain(*map(nn.Module.parameters, modules)))


def make_targets(*modules):
    return map(lambda m: copy.deepcopy(m).requires_grad_(False), modules)
