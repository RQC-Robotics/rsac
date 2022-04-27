from dataclasses import dataclass, field
from .agent import RSAC
from . import wrappers, utils
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import pathlib
import datetime
from collections import deque
import numpy as np
import pdb

torch.autograd.set_detect_anomaly(True)


@dataclass
class Config(utils.BaseConfig):
    discount: float = .99
    disclam: float = 1.
    num_samples: int = 16
    action_repeat: int = 2

    critic_layers: tuple = (256, 256)
    actor_layers: tuple = (256, 256)
    hidden_dim: int = 256
    obs_emb_dim: int = 64
    init_log_alpha: float = 1.
    init_std: float = 2.
    mean_scale: float = 5.
    spr_coef: float = 2.
    spr_depth: int = 5

    critic_lr: float = 1e-3
    actor_lr: float = 1e-3
    dual_lr: float = 1e-2
    encoder_lr: float = 1e-3
    weight_decay: float = 1e-7

    critic_tau: float = .995
    actor_tau: float = .995
    encoder_tau: float = .995

    total_steps: int = 2 * 10 ** 6
    training_steps: int = 200
    seq_len: int = 50
    eval_freq: int = 20000
    max_grad: float = 100.
    batch_size: int = 50
    buffer_size: int = 1000
    burn_in: int = -1
    bptt: int = -1


    # PointNet
    pn_number: int = 600
    pn_layers: tuple = (32, 64, 128)
    pn_dropout: float = 0.

    task: str = 'walker_stand'
    aux_loss: str = 'None'
    logdir: str = 'logdir'
    device: str = 'cuda'
    observe: str = 'point_cloud'


class RLAlg:
    def __init__(self, config):

        self.config = config
        self.env, obs_spec, act_spec = self._make_env()
        self._task_path = pathlib.Path(config.logdir).joinpath(
            f'./{config.task}/{config.observe}/{config.aux_loss}')
        self.callback = SummaryWriter(log_dir=self._task_path)
        self.agent = RSAC(obs_spec, act_spec, config, self.callback)
        self.buffer = utils.TrajectoryBuffer(config.buffer_size, seq_len=config.seq_len)
        self.interactions_count = 0

    def learn(self):
        self.config.save(self._task_path / 'config.yml')

        def policy(obs, state, training):
            obs = torch.from_numpy(obs[None]).to(self.agent.device)
            action, log_prob, state = self.agent.policy(obs, state, training)
            action, log_prob = map(lambda t: t.detach().cpu().numpy().flatten(), (action, log_prob))
            return action, log_prob, state

        while self.interactions_count < self.config.total_steps:
            tr = utils.simulate(self.env, policy, True)
            self.buffer.add(tr)
            self.interactions_count += 1000

            dl = DataLoader(self.buffer, batch_size=self.config.batch_size)
            self.agent.train()
            for i, tr in enumerate(dl):
                obs, actions, rewards, log_probs, hidden_states = map(lambda k: tr[k].to(self.agent.device).transpose(0, 1),
                    ('observations', 'actions', 'rewards', 'log_probs', 'states'))
                self.agent.step(obs, actions, rewards, log_probs, hidden_states)
                if i > self.config.training_steps:
                    break

            if self.interactions_count % self.config.eval_freq == 0:
                self.agent.eval()
                scores = [utils.simulate(self.env, policy, False)['rewards'].sum() for _ in range(10)]
                self.callback.add_scalar('test/eval_reward', np.mean(scores), self.interactions_count)
                self.callback.add_scalar('test/eval_std', np.std(scores), self.interactions_count)

            if self.interactions_count % (5*self.config.eval_freq) == 0:
                self.save()

    def save(self):
        torch.save({
            'interactions': self.interactions_count,
            'agent': self.agent.state_dict(),
            'optim': self.agent.optim.state_dict(),
        }, self._task_path / 'checkpoint')

    def load(self, path):
        path = pathlib.Path(path)
        self.config = self.config.load(path / 'config.yml')
        if (path / 'checkpoint').exists():
            chkp = torch.load(path / 'checkpoint')
            with torch.no_grad():
                self.agent.load_state_dict(chkp['agent'])
                self.agent.optim.load_state_dict(chkp['optim'])
            self.interactions_count = chkp['interactions']

    def _make_env(self):
        env = utils.make_env(self.config.task)
        if self.config.observe == 'states':
            env = wrappers.StatesWrapper(env)
        elif self.config.observe in wrappers.PixelsWrapper.channels.keys():
            env = wrappers.PixelsWrapper(env, mode=self.config.observe)
        elif self.config.observe == 'point_cloud':
            env = wrappers.PointCloudWrapper(env, pn_number=self.config.pn_number)
        else:
            raise NotImplementedError
        env = wrappers.FrameSkip(env, self.config.action_repeat)
        return env, env.observation_spec(), env.action_spec()
