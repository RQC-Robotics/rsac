import torch
import numpy as np
import ctypes
import pathlib
from collections import defaultdict
from dm_control.mujoco import wrapper
from dm_control.mujoco.engine import Camera
from dm_control.mujoco.wrapper.mjbindings import enums
from dm_env import specs


class Wrapper:
    def __init__(self, env):
        self.env = env

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
        return obs, r, d

    def reset(self):
        return self.observation(self.env.reset())

    def __getattr__(self, item):
        return getattr(self.env, item)

    @property
    def unwrapped(self):
        if hasattr(self.env, 'unwrapped'):
            return self.env.unwrapped
        else:
            return self.env


class StatesWrapper(Wrapper):
    """ Converts OrderedDict obs to 1-dim np.ndarray[np.float32]. """
    def __init__(self, env):
        super().__init__(env)
        self._observation_spec = self._infer_obs_specs(env)

    def observation(self, timestamp):
        obs = np.array([])
        for v in timestamp.observation.values():
            if not v.ndim:
                v = v[None]
            obs = np.concatenate((obs, v.flatten()))
        return obs.astype(np.float32)

    @staticmethod
    def _infer_obs_specs(env) -> specs.Array:
        dim = sum((np.prod(ar.shape) for ar in env.observation_spec().values()))
        return specs.Array(shape=(dim,), dtype=np.float32, name='states')

    def observation_spec(self):
        return self._observation_spec


class FrameSkip(Wrapper):
    def __init__(self, env, frames_number):
        super().__init__(env)
        self.fn = frames_number

    def step(self, action):
        R = 0
        for i in range(self.fn):
            next_obs, reward, done= self.env.step(action)
            R += reward
            if done:
                break
        return np.float32(next_obs), np.float32(R), done  # np.float32(next_obs)

    def reset(self):
        return np.float32(self.env.reset())


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
        state = self.physics.state()
        self._data['states'].append(state)
        #self._data['depth_maps'].append(depth)
        #self._data['images'].append(image)
        self._data['observations'].append(obs)
        return obs, r, d, _

    def _render(self, **kwargs):
        kw = self.render_kwargs.copy()
        kw.update(kwargs)
        return self.physics.render(**kw)

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


class PixelsWrapper(Wrapper):
    channels = dict(rgb=3, rgbd=4, d=1, g=1, gd=2)

    def __init__(self, env, render_kwargs=None, mode='rgb'):
        super().__init__(env)
        self.render_kwargs = render_kwargs or dict(camera_id=0, height=84, width=84)
        self.mode = mode
        self._gs_coef = np.array([0.299, 0.587, 0.114])

    def observation(self, timestamp):
        # depth could be normalized /depth.max()
        if self.mode != 'd':
            rgb = self.physics.render(**self.render_kwargs).astype(np.float32)
            rgb /= 255.
        obs = ()
        if self.mode in ('rgb', 'rgbd'):
            obs += (rgb - .5,)
        if self.mode in ('rgbd', 'd', 'gd'):
            depth = self.physics.render(depth=True, **self.render_kwargs)
            obs += (depth[..., np.newaxis],)
        if self.mode in ('g', 'gd'):
            g = rgb @ self._gs_coef
            obs += (g[..., np.newaxis],)
        obs = np.concatenate(obs, -1)
        return obs.transpose((2, 1, 0)).astype(np.float32)

    def observation_spec(self):
        shape = (
            self.channels[self.mode],
            self.render_kwargs.get('height', 240),
            self.render_kwargs.get('width', 320)
        )
        return specs.Array(shape=shape, dtype=np.float32, name=self.mode)


class PointCloudWrapper(Wrapper):
    def __init__(self, env, pn_number=1000, render_kwargs=None, static_camera=True):
        super().__init__(env)

        self.render_kwargs = render_kwargs or dict(camera_id=0)
        self.scene_option = wrapper.MjvOption()
        self.scene_option.flags[enums.mjtVisFlag.mjVIS_STATIC] = 0  # results in wrong segmentation for some envs
        self.pn_number = pn_number
        self.static_camera = static_camera
        self._partial_sum = None
        if static_camera:
            self._inverse_matrix = self.inverse_matrix()

    def observation(self, timestamp):
        # scene_option shouldn't be used in depth_map however it removes contour for some envs meanwhile broking other
        depth_map = self.physics.render(depth=True, **self.render_kwargs, scene_option=self.scene_option)
        inv_mat = self._inverse_matrix if self.static_camera else self.inverse_matrix()
        point_cloud = self._get_point_cloud(inv_mat, depth_map)
        segmentation_mask = self._segmentation_mask()
        mask = self._mask(point_cloud)  # additional mask if needed
        selected_points = point_cloud[segmentation_mask & mask]
        return self._to_fixed_number(selected_points).astype(np.float32)

    def inverse_matrix(self):
        # one could reuse the matrix if a camera remains static
        camera = Camera(self.physics, **self.render_kwargs)
        image, focal, _, _ = camera.matrices()
        inv_mat = np.linalg.inv((image@focal)[:, :-1])
        return inv_mat
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
        seg = self.physics.render(segmentation=True, **self.render_kwargs, scene_option=self.scene_option)
        model_id, obj_type = np.split(seg, 2, -1)
        return (obj_type != -1).flatten()

    def _to_fixed_number(self, pc):
        if len(pc) < self.pn_number:
            return np.pad(pc, ((0, self.pn_number - pc.shape[-2]), (0, 0)), mode='edge')
        else:
            return np.random.permutation(pc)[:self.pn_number]

    def _get_point_cloud(self, mat, depth_map):
        dot_product = lambda x, y: np.einsum('ij, jhw-> hwi', x, y)
        if not self.static_camera or self._partial_sum is None:
            width = self.render_kwargs.get('width', 320)
            height = self.render_kwargs.get('height', 240)
            grid = np.mgrid[:height, :width]
            self._partial_sum = dot_product(mat[:, :-1], grid)

        residual_sum = dot_product(mat[:, -1:], depth_map[np.newaxis])
        return np.reshape(self._partial_sum + residual_sum, (-1, 3))

    def _mask(self, point_cloud):
        """ Heuristic to cut outliers """
        threshold = np.quantile(point_cloud[..., 2], .99)  # assuming object is connected and compact
        return point_cloud[..., 2] < min(threshold, 10)

    def observation_spec(self):
        return specs.Array(shape=(self.pn_number, 3), dtype=np.float32, name='point_cloud')
