from abc import ABC
import dataclasses
from ruamel.yaml import YAML


@dataclasses.dataclass
class BaseConfig(ABC):
    def save(self, file_path):
        yaml = YAML()
        with open(file_path, 'w') as f:
            yaml.dump(dataclasses.asdict(self), f)

    @classmethod
    def load(cls, file_path, **kwargs):
        yaml = YAML()
        with open(file_path) as f:
            config_dict = yaml.load(f)
        config_dict.update(kwargs)

        fields = tuple(map(lambda field: field.name, dataclasses.fields(cls)))
        config_dict = {k: v for k, v in config_dict.items() if k in fields}

        return cls(**config_dict)

    def __post_init__(self):
        for field in dataclasses.fields(self):
            value = getattr(self, field.name)
            value = field.type(value)
            setattr(self, field.name, value)


@dataclasses.dataclass
class Config(BaseConfig):
    # algo
    discount: float = .99
    disclam: float = 1.
    num_samples: int = 16
    action_repeat: int = 2
    frames_stack: int = 3
    init_temperature: float = .1
    target_ent_per_dim: float = -1.

    # architecture
    critic_layers: tuple = (256, 256)
    actor_layers: tuple = (256, 256)
    obs_emb_dim: int = 50
    mean_scale: float = 1.

    # PointNet
    pn_number: int = 100
    pn_layers: tuple = (64, 128, 256)
    downsample: int = 5

    # train
    rl_lr: float = 3e-4
    ae_lr: float = 3e-4
    dual_lr: float = 1e-4
    weight_decay: float = 0.
    critic_tau: float = .01
    encoder_tau: float = .01
    max_grad: float = 40.

    total_steps: int = 2*10**6
    spi: int = 128
    seq_len: int = 8
    batch_size: int = 16
    eval_freq: int = 20000
    buffer_size: int = 1000

    # task
    seed: int = 0
    task: str = 'walker_stand'
    aux_loss: str = 'None'
    logdir: str = 'logdir'
    device: str = 'cuda'
    observe: str = 'point_cloud'
    debug: bool = True
