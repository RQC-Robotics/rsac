import torch
from .utils import build_mlp, TanhTransform
from torchvision import transforms as T
nn = torch.nn
F = nn.functional
td = torch.distributions


class Critic(nn.Module):
    def __init__(self, in_features, layers):
        super().__init__()
        self.qs = nn.ModuleList([build_mlp((in_features, *layers, 1)) for _ in range(2)])

    def forward(self, obs, action):
        x = torch.cat([obs, action], -1)
        qs = [head(x) for head in self.qs]
        return torch.cat(qs, -1)


class Actor(nn.Module):
    def __init__(self, in_features, out_features, layers, mean_scale=1, init_std=2.):
        super().__init__()
        self.mean_scale = mean_scale
        self.mlp = build_mlp((in_features, *layers, 2*out_features))
        self.init_std = torch.log(torch.tensor(init_std).exp() - 1.)

    def forward(self, x):
        x = self.mlp(x)
        mu, std = x.chunk(2, -1)
        mu = self.mean_scale * torch.tanh(mu / self.mean_scale)
        std = torch.maximum(std, torch.full_like(std, -18.))
        std = F.softplus(std + self.init_std) + 1e-7
        return self.get_dist(mu, std)

    @staticmethod
    def get_dist(mu, std):
        dist = td.Normal(mu, std)
        dist = td.transformed_distribution.TransformedDistribution(dist, TanhTransform())
        dist = td.Independent(dist, 1)
        return dist


class DummyEncoder(nn.Linear):
    def forward(self, x):
        return super().forward(x), None


class PointCloudDecoder(nn.Module):
    def __init__(self, in_features, pn_number, depth=32, act=nn.ELU):
        super().__init__()

        self.deconvs = nn.Sequential(
            nn.Linear(in_features, 2*depth*pn_number),
            act(),
            nn.Unflatten(-1, (pn_number, 2*depth)),
            nn.Linear(2*depth, depth),
            act(),
            nn.Linear(depth, 3)
        )

    def forward(self, x):
        return self.deconvs(x)


class PointCloudEncoder(nn.Module):
    def __init__(self, in_features, out_features, layers, dropout=0., act=nn.ELU):
        super().__init__()
        self.convs = nn.Sequential()

        sizes = (in_features,) + layers
        for i in range(len(sizes)-1):
            block = nn.Sequential(
                nn.Linear(sizes[i], sizes[i+1]),
                act(),
                nn.Dropout(dropout)
            )
            self.convs.add_module(f'conv{i}', block)

        self.fc = nn.Sequential(
            nn.Linear(sizes[-1], out_features),
            nn.LayerNorm(out_features),
            nn.Tanh()
        )

        self.soft = False

    def forward(self, x):
        x = self.convs(x)
        # try soft sampling
        if self.soft:
            indices = F.gumbel_softmax(x, hard=True).argmax(-2)
            values = torch.gather(x, -2, indices.unsqueeze(-2)).squeeze(-2)
        else:
            values, indices = torch.max(x, -2)
        return self.fc(values), indices


class PixelEncoder(nn.Module):
    def __init__(self, in_channels=3, out_features=64, depth=32, act=nn.ELU):
        super().__init__()

        self.convs = nn.Sequential(
            T.Normalize(.5, 1),
            nn.Conv2d(in_channels, depth, 3, 2),
            act(),
            nn.Conv2d(depth, depth, 3, 1),
            act(),
            nn.Conv2d(depth, depth, 3, 1),
            act(),
            nn.Flatten(),
            nn.Linear(depth*27*27, out_features),  # 37 if size = 84, 27 if 64
            nn.LayerNorm(out_features),
            nn.Tanh()
        )

    def forward(self, img):
        reshape = img.ndimension() > 4  # hide temporal axis
        if reshape:
            seq_len, batch_size = img.shape[:2]
            img = img.flatten(0, 1)
        img = self.convs(img)
        if reshape:
            img = img.reshape(seq_len, batch_size, -1)
        return img, None


class PixelDecoder(nn.Module):
    def __init__(self, in_features, out_channels=3, depth=32, act=nn.ELU):
        super().__init__()
        self.deconvs = nn.Sequential(
            nn.Linear(in_features, depth*27*27),
            act(),
            nn.Unflatten(-1, (depth, 27, 27)),
            nn.ConvTranspose2d(depth, depth, 3, 1),
            act(),
            nn.ConvTranspose2d(depth, depth, 3, 1),
            act(),
            nn.ConvTranspose2d(depth, out_channels, 3, 2, output_padding=1),
            T.Normalize(-.5, 1)
        )

    def forward(self, x):
        reshape = x.ndimension() > 2  # hide temporal axis
        if reshape:
            seq_len, batch_size = x.shape[:2]
            x = x.flatten(0, 1)
        img = self.deconvs(x)
        if reshape:
            img = img.reshape(seq_len, batch_size, 3, 64, 64)
        return img
