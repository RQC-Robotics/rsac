from dataclasses import dataclass, field
from .agent import RSAC
from . import wrappers, utils
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import pathlib
import datetime
#from tqdm.notebook import trange
from tqdm import trange
from collections import deque
from statistics import mean, stdev
import pdb

torch.autograd.set_detect_anomaly(True)


@dataclass
class Config(utils.AbstractConfig):
    discount: float = .99
    disclam: float = .95
    num_samples: int = 6
    action_repeat: int = 2

    critic_layers: tuple = (256, 256)
    actor_layers: tuple = (255, 256)
    hidden_dim: int = 512
    obs_emb_dim: int = 256
    init_log_alpha: float = 1.
    init_std: float = 3.
    mean_scale: float = 5.
    spr_coef: float = 1
    spr_depth: int = 5

    critic_lr: float = 1e-3
    actor_lr: float = 1e-3
    dual_lr: float = 1e-3
    critic_tau: float = .995
    actor_tau: float = .995
    encoder_tau: float = .995

    total_steps: int = 10 ** 7
    training_steps: int = 200
    seq_len: int = 50
    eval_freq: int = 10000
    max_grad: float = 100.
    batch_size: int = 20
    buffer_size: int = 200
    burn_in: int = 15


    # PointNet
    pn_number: int = 600
    pn_layers: tuple = (64, 128)#field(default_factory=lambda: [64, 128])
    pn_dropout: float = 0.

    task: str = 'walker_stand'
    aux_loss: str = 'None'
    logdir: str = 'logdir/'
    device: str = 'cuda'
    observe: str = 'pixels'

    def __post_init__(self):
        spi = self.training_steps * self.batch_size * self.seq_len / 1000.
        #print(f'Samples per insert (SPI): {spi: .1f}')


class RLAlg:
    def __init__(self, config):
        self._c = config
        self.env, act_dim, obs_dim = self._make_env()
        # self._task_path = pathlib.Path(config.logdir).joinpath(
        #     f'./{config.task}/{config.observe}/{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}')
        self._task_path = pathlib.Path(config.logdir).joinpath(
            f'./{config.task}/{config.observe}/{config.aux_loss}')
        self.callback = SummaryWriter(log_dir=self._task_path)
        self.agent = RSAC(obs_dim, act_dim, config, self.callback)
        self.buffer = utils.TrajectoryBuffer(config.buffer_size, seq_len=config.seq_len)
        self.interactions_count = 0

    def learn(self):
        self._c.save(self._task_path / 'config')
        logs = deque(maxlen=10)

        def policy(obs, state, training):
            obs = torch.from_numpy(obs[None]).to(self.agent.device)
            action, state = self.agent.policy(obs, state, training)
            return action.cpu().detach().numpy().flatten(), state

        with trange(self._c.total_steps) as pbar:
            while True:
                pbar.n = self.interactions_count
                tr = utils.simulate(self.env, policy, True)
                self.buffer.add(tr)
                self.interactions_count += 1000

                dl = DataLoader(self.buffer, batch_size=self._c.batch_size)
                self.agent.train()
                for i, tr in enumerate(dl):
                    obs, actions, rewards, hidden_states = map(lambda k: tr[k].to(self.agent.device).transpose(0, 1),
                        ('observations', 'actions', 'rewards', 'states'))
                    self.agent.step(obs, actions, rewards, hidden_states)
                    if i > self._c.training_steps:
                        break

                if self.interactions_count % self._c.eval_freq == 0:
                    self.agent.eval()
                    scores = [utils.simulate(self.env, policy, False)['rewards'].sum() for _ in range(5)]
                    score = mean(scores)
                    logs.append(score)
                    #pbar.update(self.interactions_count - pbar.n)
                    pbar.set_postfix(score=score, mean10=mean(logs))
                    self.callback.add_scalar('test/eval_reward', mean(logs), pbar.n)
                    self.save()

    def save(self):
        torch.save({
            'interactions': self.interactions_count,
            'agent': self.agent.state_dict(),
            'actor_optim': self.agent.actor_optim.state_dict(),
            'critic_optim': self.agent.critic_optim.state_dict(),
            'dual_optim': self.agent.dual_optim.state_dict(),
        }, self._task_path / 'checkpoint')

    def load(self):
        self._c.load(self._task_path / 'config')
        chkp = torch.load(self._task_path / 'checkpoint')
        with torch.no_grad():
            self.agent.load_state_dict(chkp['agent'])
            self.agent.actor_optim.load_state_dict(chkp['actor_optim'])
            self.agent.critic_optim.load_state_dict(chkp['critic_optim'])
            self.agent.dual_optim.load_state_dict(chkp['dual_optim'])
        self.interactions_count = chkp['interactions']

    def _make_env(self):
        env = utils.make_env(self._c.task)
        if self._c.observe == 'states':
            env = wrappers.dmWrapper(env)
        elif self._c.observe == 'pixels':
            env = wrappers.PixelsToGym(env)
        elif self._c.observe == 'point_cloud':
            env = wrappers.depthMapWrapper(env, device=self._c.device, points=self._c.pn_number, camera_id=1)
        env = wrappers.FrameSkip(env, self._c.action_repeat)
        act_dim = env.action_space.shape[0]
        obs_dim = env.observation_space.shape[0]  # only used for states as observations
        return env, act_dim, obs_dim
