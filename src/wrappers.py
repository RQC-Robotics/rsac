import numpy as np
import gym
from gym.spaces import Box
import torch
from .utils import PointCloudGenerator
import ctypes
import pathlib
from collections import defaultdict
from dm_control.mujoco import wrapper
from dm_control.mujoco.engine import Camera
from dm_control.mujoco.wrapper.mjbindings import enums


class Wrapper:
    """ Partially solves problem with  compatibility"""

    def __init__(self, env):
        self.env = env
        self._observation_space, self._action_space = self._infer_spaces(env)

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

        obs_space = Box(low=-lim, high=lim, shape=obs_sample.shape, dtype=np.float32)#ar.dtype)
        return obs_space, action_space

    def __getattr__(self, item):
        return getattr(self.env, item)

    @property
    def unwrapped(self):
        env = self
        while hasattr(env, 'env'):
            env = env.env
        return env

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space


class dmWrapper(Wrapper):
    def observation(self, timestamp):
        obs = np.array([])
        for v in timestamp.observation.values():
            if not v.ndim:
                v = v[None]
            obs = np.concatenate((obs, v))
        return obs.astype(np.float32)


class FrameSkip(Wrapper):
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
        return np.float32(next_obs), np.float32(R), done, info  # np.float32(next_obs)

    def reset(self):
        return np.float32(self.env.reset())


class depthMapWrapper(Wrapper):

    def __init__(self, env,
                 camera_id=0,
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
        scene = wrapper.MjvOption()
        scene.flags = (ctypes.c_uint8*22)(0)

        return scene

    @property
    def pc_params(self):
        # device
        fovy = self.env.physics.model.cam_fovy[0]
        return dict(
            camera_fovy=fovy,
            image_height=self._depth_kwargs.get('height', 240),
            image_width=self._depth_kwargs.get('width', 320)
        )


class Monitor(Wrapper):
    def __init__(self, env, path, render_kwargs={'camera_id': 0}):
        self.env = env
        self.path = pathlib.Path(path)
        self.render_kwargs = render_kwargs
        self._data = defaultdict(list)

    def reset(self):
        self._data = defaultdict(list)
        return self.env.reset()

    def step(self, action):
        obs, r, d, _ = self.env.step(action)
        #image = self._render()
        #depth = self._render(depth=True)
        state = self.env.physics.state()
        self._data['states'].append(state)
        #self._data['depth_maps'].append(depth)
        #self._data['images'].append(image)
        self._data['observations'].append(obs)
        return obs, r, d, _

    def _render(self, **kwargs):
        kw = self.render_kwargs.copy()
        kw.update(kwargs)
        return self.env.physics.render(**kw)

    def save(self, path_dir):
        """
        save data to path_dir in desired format
        """
        pass

    @property
    def data(self):
        data = {}
        for k, v in self._data.items():
            if isinstance(v[0], np.ndarray):
                data[k] = torch.from_numpy(np.stack(v))
            else:
                data[k] = torch.stack(v)
        return data


class PixelsToGym(Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = wrapper.pixels.Wrapper(self.env, render_kwargs={'camera_id': 0, 'height': 64, 'width': 64})

    def observation(self, timestamp):
        obs = timestamp.observation['pixels']
        obs = np.array(obs) / 255.
        obs = np.array(obs)
        return obs.transpose((2, 1, 0))

    @property
    def observation_space(self):
        # correspondent space have to be extracted from the dm_control API -> gym API
        return Box(low=0., high=1., shape=(64, 64, 3))


class PointCloudWrapper(Wrapper):
    def __init__(self, env, pn_number=1000, render_kwargs=None, threshold=10):
        self._assert_kwargs(render_kwargs)
        super().__init__(env)

        self.render_kwargs = render_kwargs or dict(camera_id=0)
        self.scene_option = wrapper.MjvOption()
        self.scene_option.flags[enums.mjtVisFlag.mjVIS_STATIC] = 0  # wrong segmentation for some envs
        self.threshold = threshold
        self.pn_number = pn_number

    def observation(self, timestamp):
        depth_map = self.env.physics.render(depth=True, **self.render_kwargs)
        inv_mat = self.inverse_matrix()
        width = self.render_kwargs.get('width', 320)
        height = self.render_kwargs.get('height', 240)
        plane = np.concatenate((np.mgrid[:height, :width], depth_map[None]))
        point_cloud = np.einsum('ij, jkl->kli', inv_mat, plane).reshape(-1, 3)
        mask = self._segmentation_mask()
        # z-axis mask to fix poor segmentation
        z_mask = point_cloud[..., 2] < self.threshold
        selected_points = point_cloud[mask & z_mask]
        return self._to_fixed_number(selected_points)

    def inverse_matrix(self):
        # one could reuse the matrix if camera remains static
        camera = Camera(self.env.physics, **self.render_kwargs)
        image, focal, _, _ = camera.matrices()
        cx = image[0, 2]
        cy = image[1, 2]
        f_inv = 1. / focal[1, 1]
        inv_mat = np.array([
            [-f_inv, f_inv ** 2, cy * f_inv ** 2 + cx * f_inv],
            [0, f_inv, -f_inv * cy],
            [0, 0, 1.]
        ])
        return inv_mat

    def _segmentation_mask(self):
        seg = self.env.physics.render(segmentation=True, **self.render_kwargs, scene_option=self.scene_option)
        model_id, obj_type = np.split(seg, 2, -1)
        return (obj_type != -1).flatten()

    def _to_fixed_number(self, pc):
        n = len(pc)
        if n < self.pn_number:
            return np.pad(pc, ((0, self.pn_number - pc.shape[-2]), (0, 0)), mode='edge')
        else:
            return np.random.permutation(pc)[:self.pn_number]

    @staticmethod
    def _assert_kwargs(kwargs):
        if kwargs is None:
            return
        keys = kwargs.keys()
        assert 'camera_id' in keys
        assert 'depth' not in keys
        assert 'segmentation' not in keys
