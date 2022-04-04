import numpy as np
import gym
from gym.spaces import Box
import torch
from .utils import PointCloudGenerator
import ctypes
import pathlib
from collections import defaultdict
from dm_control.mujoco.wrapper import MjvOption


class Wrapper:
    """ Partially solves problem with  compatibility"""

    def __init__(self, env):
        self.env = env
        self.observation_space, self.action_space = self._infer_spaces(env)

    def observation(self, timestamp):
        return timestamp

    def reward(self, timestamp):
        return np.float32(timestamp.reward)

    def done(self, timestamp):
        return timestamp.last()

    def step(self, action):
        timestamp = self.env.step(action)
        obs = self.observation(timestamp)
        r = self.reward(timestamp)
        d = self.done(timestamp)
        return obs, r, d, None

    def reset(self):
        return self.observation(self.env.reset())

    @staticmethod
    def _infer_spaces(env):
        lim = float('inf')
        spec = env.action_spec()
        action_space = Box(low=spec.minimum.astype(np.float32), dtype=np.float32,
                           high=spec.maximum.astype(np.float32), shape=spec.shape)
        ar = list(env.observation_spec().values())[0]
        obs_sample = np.concatenate(list(map(lambda ar: ar.generate_value() if ar.shape != () else [1],
                                             env.observation_spec().values())))
        obs_space = Box(low=-lim, high=lim, shape=obs_sample.shape, dtype=ar.dtype)
        return obs_space, action_space

    def __getattr__(self, item):
        return getattr(self.env, item)


class dmWrapper(Wrapper):
    def observation(self, timestamp):
        obs = np.array([])
        for v in timestamp.observation.values():
            if not v.ndim:
                v = v[None]
            obs = np.concatenate((obs, v))
        return obs.astype(np.float32)


class FrameSkip(gym.Wrapper):
    def __init__(self, env, frames_number):
        super().__init__(env)
        self.fn = frames_number

    def step(self, action):
        R = 0
        for i in range(self.fn):
            next_obs, reward, done, info = self.env.step(action)
            R += reward
            if done:
                break
        return np.float32(next_obs), np.float32(R), done, info

    def reset(self):
        return np.float32(self.env.reset())


class depthMapWrapper(Wrapper):

    def __init__(self, env,
                 camera_id=1,
                 height=240,
                 width=320,
                 device='cpu',
                 return_pos=False,
                 points=1000,
                 ):
        super().__init__(env)
        self.env = env
        self.points = points
        self._depth_kwargs = dict(camera_id=camera_id, height=height, width=width,
                                  depth=True, scene_option=self._prepare_scene())
        self.return_pos = return_pos
        self.pcg = PointCloudGenerator(**self.pc_params, device=device)

    def observation(self, timestamp):
        depth = self.env.physics.render(**self._depth_kwargs)
        pc = self.pcg.get_PC(depth)
        pc = self._segmentation(pc)
        if self.return_pos:
            pos = self.env.physics.position()
            return pc, pos
        return pc.detach().cpu().numpy()

    def _segmentation(self, pc):
        dist_thresh = 19
        pc = pc[pc[..., 2] < dist_thresh] # smth like infty cutting
        if self.points:
            amount = pc.size(-2)
            if amount > self.points:
                ind = torch.randperm(amount, device=self.pcg.device)[:self.points]
                pc = torch.index_select(pc, -2, ind)
            elif amount < self.points:
                zeros = torch.zeros(self.points - amount, *pc.shape[1:], device=self.pcg.device)
                pc = torch.cat([pc, zeros])
        return pc

    def _prepare_scene(self):
        scene = MjvOption()
        scene.ptr.contents.flags = (ctypes.c_uint8*22)(0)

        return scene

    @property
    def pc_params(self):
        # device
        fovy = self.env.physics.model.cam_fovy[0]
        return dict(
            camera_fovy=fovy,
            image_height=self._depth_kwargs.get('height') or 240,
            image_width=self._depth_kwargs.get('width') or 320)


class Monitor(Wrapper):
    def __init__(self, env, path, render_kwargs={'camera_id': 1}):
        self.env = env
        self.path = pathlib.Path(path)
        self.render_kwargs = render_kwargs
        self._data = defaultdict(list)

    def step(self, action):
        timestamp = self.env.step(action)
        image = self._render()
        depth = self._render(depth=True)
        state = self.env.physics.state()
        self._data['states'].append(state)
        self._data['depth_maps'].append(depth)
        self._data['images'].append(image)
        return timestamp

    def _render(self, **kwargs):
        kw = self.render_kwargs.copy()
        kw.update(kwargs)
        return self.env.physics.render(**kw)

    def save(self, path_dir):
        """
        save data to path_dir in the desired format
        """
        pass

    def free(self):
        self._data = defaultdict(list)

    @property
    def data(self):
        return self._data
