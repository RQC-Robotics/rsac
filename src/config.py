from abc import ABC
import dataclasses
from ruamel.yaml import YAML


@dataclasses.dataclass
class BaseConfig(ABC):
    def save(self, file_path):
        yaml = YAML()
        with open(file_path, 'w') as f:
            yaml.dump(dataclasses.asdict(self), f)

    def load(self, file_path, **kwargs):
        yaml = YAML()
        with open(file_path) as f:
            config_dict = yaml.load(f)
        config_dict.update(kwargs)
        return dataclasses.replace(self, **config_dict)

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
    num_samples: int = 32
    action_repeat: int = 2
    spr_coef: float = 2.
    spr_depth: int = 5
    init_log_alpha: float = -4.

    # architecture
    critic_layers: tuple = (256, 256)
    actor_layers: tuple = (256, 256)
    hidden_dim: int = 256
    obs_emb_dim: int = 64
    mean_scale: float = 5.

    # PointNet
    pn_number: int = 600
    pn_layers: tuple = (32, 32, 32, 32)
    pn_dropout: float = 0.

    # train
    rl_lr: float = 3e-4
    ae_lr: float = 3e-4
    dual_lr: float = 1e-2
    weight_decay: float = 1e-7
    critic_tau: float = .995
    actor_tau: float = .995
    encoder_tau: float = .995
    max_grad: float = 50.

    total_steps: int = 4 * 10 ** 6
    training_steps: int = 100
    seq_len: int = 10
    batch_size: int = 128
    eval_freq: int = 20000
    buffer_size: int = 1000
    burn_in: int = 40

    # task
    task: str = 'walker_stand'
    aux_loss: str = 'None'
    logdir: str = 'logdir'
    device: str = 'cuda'
    observe: str = 'point_cloud'

    def __post_init__(self):
        super().__post_init__()
        self.spi = self.batch_size * self.training_steps * self.seq_len / 1000.
