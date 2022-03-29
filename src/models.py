import torch
from .utils import build_mlp, TanhTransform
nn = torch.nn
F = nn.functional
td = torch.distributions


class Critic(nn.Module):
    """ proper double critic with separate body"""
    def __init__(self, in_features, heads, layers):
        super().__init__()
        self.qs = nn.ModuleList([build_mlp([in_features] + layers + [1]) for _ in range(heads)])

    def forward(self, obs, action):
        x = torch.cat([obs, action], -1)
        qs = [head(x) for head in self.qs]
        return torch.cat(qs, -1)


class Actor(nn.Module):
    def __init__(self, in_features, out_features, layers, mean_scale=1, init_std=2.):
        super().__init__()
        self.mean_scale = mean_scale
        self.mlp = build_mlp([in_features] + layers + [2*out_features])
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


# class DummyEncoder(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.dummy_param = nn.Parameter(torch.zeros(1))
#
#     def forward(self, x):
#         return x


class PointCloudEncoder(nn.Module):
    def __init__(self, in_features, depth, layers, dropout=0.):
        super().__init__()
        coef = 2 ** (layers - 1)
        self.model = nn.ModuleList([nn.Sequential(nn.Linear(in_features, coef*depth),
                                                  nn.ReLU(inplace=True), nn.Dropout(dropout))])
        for i in range(layers-1):
            m = nn.Sequential(nn.Linear(coef*depth, coef // 2 * depth), nn.ReLU(inplace=True), nn.Dropout(dropout))
            self.model.append(m)
            coef //= 2

        self.fc = nn.Sequential(
            nn.Linear(depth, depth),
            nn.LayerNorm([depth]),
            nn.Tanh(),
        )

    def forward(self, x):
        for layer in self.model:
            x = layer(x)
        values, idx = torch.max(x, -2)
        return self.fc(values)


class PointCloudDecoder(nn.Module):
    def __init__(self, in_features, out_features, depth, layers, pn_number):
        super().__init__()

        self.coef = 2**(layers-1)
        self.fc = nn.Sequential(
            nn.Linear(in_features, self.coef * depth * pn_number),
            nn.ELU(inplace=True),
        )

        self.deconvs = nn.ModuleList([nn.Unflatten(-1, (pn_number, self.coef * depth))])
        for _ in range(layers-1):
            self.deconvs.append(nn.Linear(self.coef * depth, self.coef * depth // 2))
            self.deconvs.append(nn.ELU(inplace=True))
            self.coef //= 2

        self.deconvs.append(nn.Linear(depth, out_features))

    def forward(self, x):
        x = self.fc(x)
        for layer in self.deconvs:
            x = layer(x)
        return x


class PointCloudEncoderv2(nn.Module):
    def __init__(self, in_features, depth, layers, dropout=0., act=nn.ELU):
        super().__init__()
        coef = 1
        self.convs = nn.ModuleList([])
        for i in range(layers):
            next_coef = coef << 1

            if i == 0:
                linear = nn.Linear(in_features, next_coef * depth)
            else:
                linear = nn.Linear(coef * depth, next_coef * depth)
            m = nn.Sequential(
                linear,
                act(),
                nn.Dropout(dropout),
            )
            self.convs.append(m)
            coef = next_coef

    def forward(self, x):
        # features = []
        for module in self.convs:
            x = module(x)
            # features.append(x)
        values, idx = torch.max(x, -2)
        return values
