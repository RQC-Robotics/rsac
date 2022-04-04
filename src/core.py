from dataclasses import dataclass, field
from .agent import RSAC
from . import wrappers, utils
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import pathlib
import datetime
from tqdm.notebook import trange
from collections import deque
from statistics import mean
import pdb

torch.autograd.set_detect_anomaly(True)


@dataclass
class Config(utils.AbstractConfig):
    discount: float = .99
    disclam: float = .95
    num_samples: int = 6
    action_repeat: int = 2

    critic_layers: tuple = (256, 256)
    actor_layers: tuple = (256, 256) #field(default_factory=lambda: 2*[256])
    hidden_dim: int = 256
    critic_heads: int = 2
    init_log_alpha: float = 1.
    init_std: float = 3.
    mean_scale: float = 5.
    spr_coef: float = 1
    spr_depth: int = 5

    critic_lr: float = 1e-3
    actor_lr: float = 1e-3
    dual_lr: float = 1e-3
    critic_tau: float = .99
    actor_tau: float = .99
    encoder_tau: float = .99

    total_steps: int = 10 ** 7
    training_steps: int = 200
    seq_len: int = 50
    eval_freq: int = 10
    max_grad: float = 100.
    batch_size: int = 30
    buffer_size: int = 1000
    burn_in: int = 15


    # PointNet
    pn_number: int = 600
    pn_layers: tuple = (64, 128, 256)#field(default_factory=lambda: [64, 128])
    pn_dropout: float = 0.
    pn_emb_dim: int = 64

    task: str = 'walker_stand'
    aux_loss: str = 'contrastive'
    logdir: str = 'logdir/contrastive'
    device: str = 'cuda'

    def __post_init__(self):
        spi = self.training_steps * self.batch_size * self.seq_len / 1000.
        print(f'Samples per insert (SPI): {spi: .1f}')


class RLAlg:
    def __init__(self, config):
        self._c = config
        self._env, act_dim, obs_dim = self._make_env()
        self._task_path = pathlib.Path(config.logdir).joinpath(
            f'./RSAC2/{config.task}/{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}')
        self.callback = SummaryWriter(log_dir=self._task_path)
        config.save(self._task_path / 'config')
        self.agent = RSAC(obs_dim, act_dim, config, self.callback)
        self.buffer = utils.TrajectoryBuffer(config.buffer_size, seq_len=config.seq_len)

    def learn(self):
        logs = deque(maxlen=10)

        def policy(obs, state, training):
            obs = torch.from_numpy(obs[None]).to(self.agent.device)
            action, state = self.agent.policy(obs, state, training)
            return action.cpu().detach().numpy().flatten(), state

        t = 0
        with trange(self._c.total_steps) as pbar:
            while True:
                tr = utils.simulate(self._env, policy, True)
                self.buffer.add(tr)
                t += 1

                dl = DataLoader(self.buffer, batch_size=self._c.batch_size)
                self.agent.train()
                for i, tr in enumerate(dl):
                    obs, actions, rewards, hidden_states = map(lambda k: tr[k].to(self.agent.device).transpose(0, 1),
                        ('observations', 'actions', 'rewards', 'states'))
                    self.agent.step(obs, actions, rewards, hidden_states)
                    if i > self._c.training_steps:
                        break

                if t % self._c.eval_freq == 0:
                    self.agent.eval()
                    score = mean([utils.simulate(self._env, policy, False)['rewards'].sum() for _ in range(5)])
                    logs.append(score)
                    pbar.update(1000*t - pbar.n)
                    pbar.set_postfix(score=score, mean10=mean(logs))
                    self.callback.add_scalar('test/eval_reward', mean(logs), pbar.n)

    def _make_env(self):
        env = utils.make_env(self._c.task)
        #env = wrappers.dmWrapper(env)
        env = wrappers.depthMapWrapper(env, device=self._c.device, points=self._c.pn_number, camera_id=1)
        env = wrappers.FrameSkip(env, self._c.action_repeat)
        act_dim = env.action_space.shape[0]
        obs_dim = env.observation_space.shape[0]
        return env, act_dim, obs_dim
