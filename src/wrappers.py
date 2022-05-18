import numpy as np
from collections import defaultdict, deque
from dm_env import specs
from dm_control.mujoco import wrapper
from dm_control.mujoco.wrapper.mjbindings import enums


class Wrapper:
    def __init__(self, env):
        self.env = env

    def observation(self, timestamp):
        return timestamp.observation

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
        obs = []
        for v in timestamp.observation.values():
            if v.ndim == 0:
                v = v[None]
            obs.append(v.flatten())
        obs = np.concatenate(obs)
        return obs.astype(np.float32)

    @staticmethod
    def _infer_obs_specs(env) -> specs.Array:
        dim = sum((np.prod(ar.shape) for ar in env.observation_spec().values()))
        return specs.Array(shape=(dim,), dtype=np.float32, name='states')

    def observation_spec(self):
        return self._observation_spec


class ActionRepeat(Wrapper):
    def __init__(self, env, frames_number: int):
        assert frames_number > 0
        super().__init__(env)
        self.fn = frames_number

    def step(self, action):
        R = 0
        for i in range(self.fn):
            next_obs, reward, done = self.env.step(action)
            R += reward
            if done:
                break
        return np.float32(next_obs), np.float32(R), done

    def reset(self):
        return np.float32(self.env.reset())


class FrameStack(Wrapper):
    def __init__(self, env, frame_number: int = 1):
        super().__init__(env)
        self.fn = frame_number
        self._state = None

    def reset(self):
        obs = self.env.reset()
        self._state = deque(self.fn * [obs], maxlen=self.fn)
        return self.observation(None)

    def step(self, action):
        obs, r, d = self.env.step(action)
        self._state.append(obs)
        return self.observation(None), r, d

    def observation(self, timestamp):
        return np.stack(self._state)

    def observation_spec(self):
        spec = self.env.observation_spec()
        return spec.replace(
            shape=(self.fn, *spec.shape),
            name=f'{self.fn}_stacked_{spec.name}'
        )


class Monitor(Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self._data = defaultdict(list)

    def reset(self):
        self._data = defaultdict(list)
        return self.env.reset()

    def step(self, action):
        obs, r, d = self.env.step(action)
        state = self.physics.state()
        self._data['states'].append(state)
        self._data['observations'].append(obs)
        return obs, r, d

    @property
    def data(self):
        return {k: np.array(v) for k, v in self._data.items()}


class PixelsWrapper(Wrapper):
    channels = dict(rgb=3, rgbd=4, d=1, g=1, gd=2)

    def __init__(self, env, render_kwargs=None, mode='rgb'):
        super().__init__(env)
        self.render_kwargs = render_kwargs or dict(camera_id=0, height=84, width=84)
        self.mode = mode
        self._gs_coef = np.array([0.299, 0.587, 0.114])

    def observation(self, timestamp):
        if self.mode != 'd':
            rgb = self.physics.render(**self.render_kwargs).astype(np.float32)
            rgb /= 255.
        obs = ()
        if 'rgb' in self.mode:
            obs += (rgb - .5,)
        if 'd' in self.mode:
            depth = self.physics.render(depth=True, **self.render_kwargs)
            obs += (depth[..., np.newaxis],)
        if 'rgb' not in self.mode and 'g' in self.mode:
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
    def __init__(self, env, pn_number=1000, render_kwargs=None, static_camera=True, as_pixels=False):
        super().__init__(env)
        self.render_kwargs = render_kwargs or dict(camera_id=0, height=240, width=320)
        self.pn_number = pn_number

        self.scene_option = wrapper.MjvOption()
        self.scene_option.flags[enums.mjtVisFlag.mjVIS_STATIC] = 0  # results in wrong segmentation for some envs

        self.static_camera = static_camera
        self.as_pixels = as_pixels
        self._partial_sum = None
        if static_camera:
            self._inverse_matrix = self.inverse_matrix()

    def observation(self, timestamp):
        depth_map = self.physics.render(depth=True, **self.render_kwargs, scene_option=self.scene_option)
        point_cloud = self._get_point_cloud(depth_map)
        if self.as_pixels:
            # TODO: decide if segmentation or another mask is needed
            return point_cloud
        point_cloud = np.reshape(point_cloud, (-1, 3))
        segmentation_mask = self._segmentation_mask()
        # TODO: fix orientation so mask can be used
        mask = self._mask(point_cloud)  # additional mask if needed
        selected_points = point_cloud[segmentation_mask & mask]
        return self._to_fixed_number(selected_points).astype(np.float32)

    def inverse_matrix(self):
        # one could reuse the matrix if a camera remains static
        cam_id, height, width = map(self.render_kwargs.get, ('camera_id', 'height', 'width'))
        fov = self.physics.model.cam_fovy[cam_id]
        rotation = self.physics.data.cam_xmat[cam_id].reshape(3, 3)
        cx = (width - 1)/2.
        cy = (height - 1)/2.
        f_inv = 2.*np.tan(np.deg2rad(fov)/2.)/height
        inv_mat = np.array([
            [f_inv, 0, -cx*f_inv],
            [0, f_inv, -f_inv*cy],
            [0, 0, 1.]
        ])
        return rotation@inv_mat

    def _segmentation_mask(self):
        seg = self.physics.render(segmentation=True, **self.render_kwargs, scene_option=self.scene_option)
        model_id, obj_type = np.split(seg, 2, -1)
        return (obj_type != -1).flatten()

    def _to_fixed_number(self, pc):
        n = len(pc)
        if n < self.pn_number:
            return np.pad(pc, ((0, self.pn_number - n), (0, 0)), mode='edge')
        else:
            return np.random.permutation(pc)[:self.pn_number]

    def _get_point_cloud(self, depth_map):
        cam_id = self.render_kwargs['camera_id']
        inv_mat = self._inverse_matrix if self.static_camera else self.inverse_matrix()
        dot_product = lambda x, y: np.einsum('ij, jhw->hwi', x, y)

        if not self.static_camera or self._partial_sum is None:
            width = self.render_kwargs.get('width', 320)
            height = self.render_kwargs.get('height', 240)
            grid = 1 + np.mgrid[:height, :width]
            self._partial_sum = dot_product(inv_mat[:, :-1], grid) + self.physics.data.cam_xpos[cam_id]

        residual_sum = dot_product(inv_mat[:, -1:], depth_map[np.newaxis])
        return self._partial_sum + residual_sum

    def _mask(self, point_cloud):
        """ Heuristic to cut outliers """
        threshold = np.quantile(point_cloud[..., 2], .99)
        return point_cloud[..., 2] < max(threshold, 10)

    def observation_spec(self):
        return specs.Array(shape=(self.pn_number, 3), dtype=np.float32, name='point_cloud')
