from .agent import RSAC
from . import wrappers, utils
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from .config import Config
import pathlib
import numpy as np
import pickle


class RLAlg:
    def __init__(self, config):

        self.config = config
        self.env = self.make_env()
        self._task_path = pathlib.Path(config.logdir).joinpath(
            f'./{config.task}/{config.observe}/{config.aux_loss}')
        self.callback = SummaryWriter(log_dir=self._task_path)
        self.agent = RSAC(self.env, config, self.callback)
        self.buffer = utils.TrajectoryBuffer(config.buffer_size, seq_len=config.seq_len)
        self.interactions_count = 0

    def learn(self):
        def update_buffer():
            tr = utils.simulate(self.env, self.policy, True)
            self.buffer.add(tr)
            self.interactions_count += 1000

        # prefill
        [update_buffer() for _ in range(10)]

        while self.interactions_count < self.config.total_steps:
            update_buffer()
            dl = DataLoader(self.buffer, batch_size=self.config.batch_size, drop_last=True)
            self.agent.train()
            for i, tr in enumerate(dl):
                obs, actions, rewards, log_probs = map(
                    lambda k: tr[k].to(self.agent.device).transpose(0, 1),
                    ('observations', 'actions', 'rewards', 'log_probs')
                )
                self.agent.step(obs, actions, rewards, log_probs)
                if i == self.config.training_steps:
                    break

            if self.interactions_count % self.config.eval_freq == 0:
                self.agent.eval()
                scores = [utils.simulate(self.env, self.policy, False)['rewards'].sum() for _ in range(10)]
                self.callback.add_scalar('test/eval_reward', np.mean(scores), self.interactions_count)
                self.callback.add_scalar('test/eval_std', np.std(scores), self.interactions_count)

            if self.interactions_count % (5*self.config.eval_freq) == 0:
                self.save()

    def save(self):
        self.config.save(self._task_path / 'config.yml')
        torch.save({
            'interactions': self.interactions_count,
            'params': self.agent.state_dict(),
            'optim': self.agent.optim.state_dict(),
        }, self._task_path / 'checkpoint')
        # with open(self._task_path / 'buffer', 'wb') as buffer:
        #     pickle.dump(self.buffer, buffer)

    @classmethod
    def load(cls, path, **kwargs):
        path = pathlib.Path(path)
        #[f.unlink() for f in path.iterdir() if f.match('*tfevents*')]
        config = Config.load(path / 'config.yml', **kwargs)
        alg = cls(config)

        if (path / 'checkpoint').exists():
            chkp = torch.load(path / 'checkpoint',
                              map_location=torch.device(config.device if torch.cuda.is_available() else 'cpu'))
            with torch.no_grad():
                alg.agent.load_state_dict(chkp['params'], strict=False)
                alg.agent.optim.load_state_dict(chkp['optim'])
            alg.interactions_count = chkp['interactions']

        if (path / 'buffer').exists():
            with open(path / 'buffer', 'rb') as b:
                alg.buffer = pickle.load(b)
        return alg

    def make_env(self):
        env = utils.make_env(self.config.task)
        if self.config.observe == 'states':
            env = wrappers.StatesWrapper(env)
        elif self.config.observe in wrappers.PixelsWrapper.channels.keys():
            env = wrappers.PixelsWrapper(env, mode=self.config.observe)
        elif self.config.observe == 'point_cloud':
            env = wrappers.PointCloudWrapper(env, pn_number=self.config.pn_number, static_camera=False)
        else:
            raise NotImplementedError
        env = wrappers.ActionRepeat(env, self.config.action_repeat)
        env = wrappers.FrameStack(env, self.config.frames_stack)
        return env

    def policy(self, obs, training):
        obs = torch.from_numpy(obs[None]).to(self.agent.device)
        action, log_prob = self.agent.policy(obs, training)
        action, log_prob = map(lambda t: t.detach().cpu().numpy().flatten(), (action, log_prob))
        return action, log_prob
