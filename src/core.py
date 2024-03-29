import time
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
        utils.set_seed(config.seed)
        self.config = config
        self.env = self.make_env(random=config.seed)
        self.task_path = pathlib.Path(config.logdir)
        self.callback = SummaryWriter(log_dir=str(self.task_path))
        self.agent = RSAC(self.env, config, self.callback)
        self.buffer = utils.TrajectoryBuffer(config.buffer_size, seq_len=config.seq_len+1)
        self.interactions_count = 0

    def learn(self):
        dur = time.time()
        while self.interactions_count < self.config.total_steps:
            tr = utils.simulate(self.env, self.policy, True)
            self.buffer.add(tr)
            self.interactions_count += self.config.action_repeat * len(tr['actions'])

            dl = DataLoader(
                self.buffer.sample_subset(
                    self.config.spi * len(tr['actions']) // self.config.seq_len
                ),
                batch_size=self.config.batch_size,
            )

            for tr in dl:
                obs, actions, rewards, done_flags, log_probs = map(
                    lambda k: tr[k].to(self.agent.device).transpose(0, 1),
                    ('observations', 'actions', 'rewards', 'done_flags', 'log_probs')
                )
                self.agent.step(obs, actions, rewards, done_flags, log_probs)

            if self.interactions_count % self.config.eval_freq == 0:
                scores = [utils.simulate(self.make_env(), self.policy, False)['rewards'].sum()
                          for _ in range(10)]
                self.callback.add_scalar('test/eval_reward', np.mean(scores),
                                         self.interactions_count)
                self.callback.add_scalar('test/eval_std', np.std(scores), self.interactions_count)

                self.save()

        dur = time.time() - dur
        self.callback.add_hparams(
            {k: v for k, v in vars(self.config).items() if
             any(map(lambda t: isinstance(v, t), (int, float, bool)))},
            dict(duration=dur, score=np.mean(scores))
        )

    def save(self):
        self.config.save(self.task_path / 'config.yml')
        torch.save({
            'interactions': self.interactions_count,
            'params': self.agent.state_dict(),
            'optim': self.agent.optim.state_dict(),
        }, self.task_path / 'checkpoint')
        # TODO: restore buffer saving
        # with open(self.task_path / 'buffer', 'wb') as buffer:
        #     pickle.dump(self.buffer, buffer)

    @classmethod
    def load(cls, path, **kwargs):
        path = pathlib.Path(path)
        #[f.unlink() for f in path.iterdir() if f.match('*tfevents*')]
        config = Config.load(path / 'config.yml', **kwargs)
        alg = cls(config)

        if (path / 'checkpoint').exists():
            chkp = torch.load(
                path / 'checkpoint',
                map_location=torch.device(config.device if torch.cuda.is_available() else 'cpu')
            )
            with torch.no_grad():
                alg.agent.load_state_dict(chkp['params'], strict=False)
                alg.agent.optim.load_state_dict(chkp['optim'])
            alg.interactions_count = chkp['interactions']

        if (path / 'buffer').exists():
            with open(path / 'buffer', 'rb') as b:
                alg.buffer = pickle.load(b)
        return alg

    def make_env(self, **task_kwargs):
        env = utils.make_env(self.config.task, **task_kwargs)
        if self.config.observe == 'states':
            env = wrappers.StatesWrapper(env)
            env = wrappers.ActionRepeat(env, self.config.action_repeat)
        elif self.config.observe in wrappers.PixelsWrapper.channels.keys():
            env = wrappers.PixelsWrapper(env, mode=self.config.observe)
            env = wrappers.ActionRepeat(env, self.config.action_repeat)
            env = wrappers.FrameStack(env, self.config.frames_stack, stack=False)
        elif self.config.observe == 'point_cloud':
            # env = wrappers.PointCloudWrapper(
            #     env,
            #     pn_number=self.config.pn_number,
            #     render_kwargs=dict(camera_id=0, height=240, width=320),
            #     downsample=self.config.downsample,
            #     apply_segmentation=True,
            # )
            env = wrappers.PointCloudWrapperV2(
                env,
                pn_number=self.config.pn_number,
                stride=self.config.stride,
                render_kwargs=dict(camera_id=0, height=240, width=320)
            )
            env = wrappers.ActionRepeat(env, self.config.action_repeat)
            env = wrappers.FrameStack(env, self.config.frames_stack, stack=True)
        else:
            raise NotImplementedError

        return env

    def policy(self, obs, training):
        obs = torch.from_numpy(obs[None]).to(self.agent.device)
        action, log_prob = self.agent.policy(obs, training)
        action, log_prob = map(lambda t: t.detach().cpu().numpy().flatten(), (action, log_prob))
        return action, log_prob
