import torch
import numpy as np
from collections import deque
import random
from torch.utils.data import Dataset
from dm_control import suite, manipulation
import math
import dataclasses
from abc import abstractmethod, ABC
from ruamel.yaml import YAML
nn = torch.nn
F = nn.functional
td = torch.distributions
ACT_LIM = .9997


def build_mlp(sizes, act=nn.ELU):
    mlp = []
    for i in range(1, len(sizes)):
        mlp.append(nn.Linear(sizes[i-1], sizes[i]))
        mlp.append(act())
    return nn.Sequential(*mlp[:-1])


def grads_sum(model):
    s = 0
    for p in model.parameters():
        if p.grad is not None:
            s += p.grad.sum().item()
    return s


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
    prev_state = None
    observations, actions, rewards, log_probs, states = [], [], [], [], []  # states=recurrent hidden
    while not done:
        if torch.is_tensor(prev_state):
            states.append(prev_state.detach().cpu().flatten().numpy())
            action, log_prob, prev_state = policy(obs, prev_state, training)
        else:
            action, log_prob, prev_state = policy(obs, prev_state, training)
            states.append(torch.zeros_like(prev_state).detach().cpu().flatten().numpy())
        new_obs, reward, done = env.step(action)
        observations.append(obs)
        actions.append(action)
        rewards.append([reward])
        log_probs.append(log_prob)
        obs = new_obs

    tr = dict(
        observations=observations,
        actions=actions,
        rewards=rewards,
        states=states,
        log_probs=log_probs,
    )
    for k, v in tr.items():
        tr[k] = np.stack(v)
    return tr


class TrajectoryBuffer(Dataset):
    def __init__(self, capacity, seq_len):
        self._data = deque(maxlen=capacity)
        self.seq_len = seq_len

    def add(self, trajectory):
        if len(trajectory['actions']) < self.seq_len:
            return
        self._data.append(trajectory)

    def __getitem__(self, idx):
        tr = random.choice(self._data)
        start = random.randint(0, len(tr['actions']) - self.seq_len)
        return {k: v[start:start+self.seq_len] for k, v in tr.items()}

    def __len__(self):
        return len(self._data)


class TanhTransform(td.transforms.TanhTransform):
    lim = ACT_LIM
    
    def _inverse(self, y):
        y = torch.clamp(y, min=-self.lim, max=self.lim)
        return torch.atanh(y)


def soft_update(target, online, rho):
    for pt, po in zip(target.parameters(), online.parameters()):
        pt.copy_(rho * pt + (1. - rho) * po)


def gve(rewards, next_values, discount, disclam):
    target_values = []
    last_val = next_values[-1]
    for r, v in zip(rewards.flip(0), next_values.flip(0)):
        last_val = r + discount*(disclam*last_val + (1.-disclam)*v)
        target_values.append(last_val)
    return torch.stack(target_values).flip(0)


def retrace(values, resids, cs, discount, disclam):
    cs = torch.cat((cs[1:], torch.ones_like(cs[0])[None]))
    cs *= disclam
    resids, cs = map(lambda t: t.flip(0), (resids, cs))
    deltas = []
    last_val = torch.zeros_like(resids[0])
    for r, c in zip(resids, cs):
        last_val = r + last_val * discount * c
        deltas.append(last_val)
    return values + torch.stack(deltas).flip(0)


@dataclasses.dataclass
class BaseConfig(ABC):
    def save(self, file_path):
        yaml = YAML()
        with open(file_path, 'w') as f:
            yaml.dump(dataclasses.asdict(self), f)

    def load(self, file_path):
        yaml = YAML()
        with open(file_path) as f:
            config_dict = yaml.load(f)
        return dataclasses.replace(self, **config_dict)

    def __post_init__(self):
        for field in dataclasses.fields(self):
            value = getattr(self, field.name)
            value = field.type(value)
            setattr(self, field.name, value)
