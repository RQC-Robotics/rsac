import numpy as np
from collections import defaultdict, deque
from dm_env import specs
from dm_control.mujoco import wrapper
from dm_control.mujoco.wrapper.mjbindings import enums


class Wrapper:
    def __init__(self, env):
        self.env = env

    @staticmethod
    def observation(timestamp):
        return timestamp.observation

    @staticmethod
    def reward(timestamp):
        return np.float32(timestamp.reward)

    @staticmethod
    def done(timestamp):
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
    def __init__(self, env, frames_number: int, discount: float = 1.):
        assert frames_number > 0
        super().__init__(env)
        self.fn = frames_number
        self.discount = discount

    def step(self, action):
        R = 0
        discount = 1.
        for i in range(self.fn):
            next_obs, reward, done = self.env.step(action)
            R += discount*reward
            discount *= self.discount
            if done:
                break
        return np.float32(next_obs), np.float32(R), done

    def reset(self):
        return np.float32(self.env.reset())


class FrameStack(Wrapper):
    def __init__(self, env, num_frames: int = 1, stack=True):
        super().__init__(env)
        self.fn = num_frames
        self.stack = stack
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
        if self.stack:
            return np.stack(self._state)
        else:
            return np.concatenate(self._state)

    def observation_spec(self):
        spec = self.env.observation_spec()
        shape = spec.shape
        shape = (self.fn, *shape) if self.stack else (self.fn * shape[0], *shape[1:])
        return spec.replace(
            shape=shape,
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
    _grayscale_coef = np.array([0.299, 0.587, 0.114])

    def __init__(self, env, render_kwargs=None, mode='rgb'):
        super().__init__(env)
        self.render_kwargs = render_kwargs or dict(camera_id=0, height=84, width=84)
        self.mode = mode

    def observation(self, timestamp):
        if self.mode != 'd':
            rgb = self.physics.render(**self.render_kwargs).astype(np.float32)
            rgb /= 255.
        obs = ()
        if 'rgb' in self.mode:
            obs += (rgb - .5,)
        if 'd' in self.mode:
            depth = self.physics.render(depth=True, **self.render_kwargs)
            depth = np.where(depth > 10., 0., depth)  # truncate depth
            depth /= depth.max()
            obs += (depth[..., np.newaxis],)
        if 'rgb' not in self.mode and 'g' in self.mode:
            g = rgb @ self._grayscale_coef
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
    def __init__(
            self,
            env,
            pn_number: int = 1000,
            render_kwargs=None,
            downsample=1,
            apply_segmentation=False
    ):
        super().__init__(env)
        self.render_kwargs = render_kwargs or dict(camera_id=0, height=240, width=320)
        assert all(map(lambda k: k in self.render_kwargs, ('camera_id', 'height', 'width')))
        self.pn_number = pn_number

        self.scene_option = wrapper.MjvOption()
        if apply_segmentation:
            self.scene_option.flags[enums.mjtVisFlag.mjVIS_STATIC] = 0

        self.downsample = downsample

    def observation(self, timestamp):
        depth_map = self.physics.render(depth=True, **self.render_kwargs,
                                        scene_option=self.scene_option)
        point_cloud = self._get_point_cloud(depth_map)
        point_cloud = np.reshape(point_cloud, (-1, 3))

        segmentation_mask = self._segmentation_mask()
        # TODO: fix orientation so mask can be used
        mask = self._mask(point_cloud)  # additional mask if needed
        selected_points = point_cloud[segmentation_mask & mask][::self.downsample]
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
        return rotation.T@inv_mat

    def _segmentation_mask(self):
        seg = self.physics.render(segmentation=True, **self.render_kwargs,
                                  scene_option=self.scene_option)
        model_id, obj_type = np.split(seg, 2, -1)
        return (obj_type != -1).flatten()

    def _to_fixed_number(self, pc):
        if self.pn_number:
            n = len(pc)
            if n == 0:
                pc = np.zeros((1, 3))
            elif n <= self.pn_number:
                pc = np.pad(pc, ((0, self.pn_number - n), (0, 0)), mode='edge')
            else:
                pc = np.random.permutation(pc)[:self.pn_number]
        return pc

    def _get_point_cloud(self, depth_map):
        width = self.render_kwargs['width']
        height = self.render_kwargs['height']
        grid = 1. + np.mgrid[:height, :width]  # TODO: somehting is wrong here
        grid = np.concatenate((grid, depth_map[None]), axis=0)
        return np.einsum('ij, jhw->hwi', self.inverse_matrix(), grid)

    def _mask(self, point_cloud):
        """ Heuristic to cut outliers """
        return point_cloud[..., 2] < 10.

    def observation_spec(self):
        return specs.Array(shape=(self.pn_number, 3), dtype=np.float32, name='point_cloud')


class PointCloudWrapperV2(Wrapper):
    def __init__(self, env, pn_number: int = 1000, render_kwargs=None, stride: int = 1):
        super().__init__(env)
        self.render_kwargs = render_kwargs or dict(camera_id=0, height=84, width=84)
        assert all(map(lambda k: k in self.render_kwargs, ('camera_id', 'height', 'width')))

        self._grid = 1. + np.mgrid[:self.render_kwargs['height'], :self.render_kwargs['width']]

        self.stride = stride
        self.pn_number = pn_number
        self._selected_geoms = np.array(self._segment_by_name(
            env.physics, ('ground', 'wall', 'floor'), **self.render_kwargs
        ))

    def observation(self, timestep):
        depth = self.env.physics.render(depth=True, **self.render_kwargs)
        pcd = self._point_cloud_from_depth(depth)
        mask = self._mask(pcd)
        pcd = pcd[mask][::self.stride]
        return self._to_fixed_number(pcd).astype(np.float32)

    def _point_cloud_from_depth(self, depth):
        f_inv, cx, cy = self._inverse_intrinsic_matrix_params()
        x, y = (depth * self._grid)
        x = (x - cx) * f_inv
        y = (y - cy) * f_inv

        pc = np.stack((x, y, depth), axis=-1)
        return pc.reshape(-1, 3)
        # rot_mat = self.env.physics.data.cam_xmat[self.render_kwargs['camera_id']].reshape(3, 3)
        # return np.einsum('ij, hwi->hwj', rot_mat, pc).reshape(-1, 3)

    def _to_fixed_number(self, pc):
        n = len(pc)
        if n == 0:
            pc = np.zeros((self.pn_number, 3))
        elif n <= self.pn_number:
            pc = np.pad(pc, ((0, self.pn_number - n), (0, 0)), mode='edge')
        else:
            pc = np.random.permutation(pc)[:self.pn_number]
        return pc

    def _inverse_intrinsic_matrix_params(self):
        height = self.render_kwargs['height']
        cx = (height - 1) / 2.
        cy = (self.render_kwargs['width'] - 1) / 2.
        fov = self.env.physics.model.cam_fovy[self.render_kwargs['camera_id']]
        f_inv = 2 * np.tan(np.deg2rad(fov) / 2.) / height
        return f_inv, cx, cy

    def _mask(self, point_cloud):
        seg = self.env.physics.render(segmentation=True, **self.render_kwargs)
        segmentation = np.isin(seg[..., 0].flatten(), self._selected_geoms)
        truncate = point_cloud[..., 2] < 10.
        return np.logical_and(segmentation, truncate)

    def observation_spec(self):
        return specs.Array(shape=(self.pn_number, 3), dtype=np.float32, name='point_cloud')

    @staticmethod
    def _segment_by_name(physics, bad_geoms_names, **render_kwargs):
        geom_ids = physics.render(segmentation=True, **render_kwargs)[..., 0]

        def _predicate(geom_id):
            return all(
                map(
                    lambda name: name not in physics.model.id2name(geom_id, 'geom'),
                    bad_geoms_names
                )
            )

        return list(filter(_predicate, np.unique(geom_ids).tolist()))
