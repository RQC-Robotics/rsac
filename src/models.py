import torch
from .utils import build_mlp, TruncatedTanhTransform
nn = torch.nn
F = nn.functional
td = torch.distributions


class LayerNormTanhEmbedding(nn.Module):
    """MLP embedded with LayerNorm (applied to the last dim) and tanh."""
    def __init__(self, *layers, act=nn.ELU):
        super().__init__()
        self.emb = nn.Sequential(
            build_mlp(*layers, act=act),
            nn.LayerNorm(layers[-1]),
            nn.Tanh()
        )

    def forward(self, x):
        return self.emb(x)
    
    
# class LayerNormTanhMLP(nn.Module):
#     """One layer LayerNormTanhEmbedding before MLP."""
#     def __init__(self, *layers, act=nn.ReLU):
#         super().__init__()
#         self.net = nn.Sequential(
#             LayerNormTanhEmbedding(layers[0], layers[1]),
#             build_mlp(*layers[1:], act=act)
#         )
#
#     def forward(self, inp):
#         return self.net(inp)


class Critic(nn.Module):
    def __init__(self, in_features, layers):
        super().__init__()
        self.qs = nn.ModuleList([build_mlp(in_features, *layers, 1) for _ in range(2)])

    def forward(self, obs, action):
        x = torch.cat([obs, action], -1)
        qs = [head(x) for head in self.qs]
        return torch.cat(qs, -1)


class Actor(nn.Module):
    def __init__(self, in_features, out_features, layers, mean_scale=1, init_std=1.):
        super().__init__()
        self.mean_scale = mean_scale
        self.mlp = build_mlp(in_features, *layers, 2*out_features)
        self.init_std = torch.log(torch.tensor(init_std).exp() - 1.)

    def forward(self, x):
        x = self.mlp(x)
        mu, std = x.chunk(2, -1)
        mu = self.mean_scale * torch.tanh(mu / self.mean_scale)
        std = torch.maximum(std, torch.full_like(std, -18.))
        std = F.softplus(std + self.init_std) + 1e-4
        return self.get_dist(mu, std)

    @staticmethod
    def get_dist(mu, std):
        dist = td.Normal(mu, std)
        dist = td.transformed_distribution.TransformedDistribution(dist, TruncatedTanhTransform())
        return td.Independent(dist, 1)


class PointCloudDecoder(nn.Module):
    def __init__(self, in_features, pn_number, layers, act=nn.ReLU):
        super().__init__()

        layers = layers + (3,)
        self.deconvs = nn.Sequential(
            nn.Linear(in_features, pn_number*layers[0]),
            nn.Unflatten(-1, (pn_number, layers[0])),
        )
        for i in range(len(layers)-1):
            block = nn.Sequential(
                act(),
                nn.Linear(layers[i], layers[i+1]),
            )
            self.deconvs.add_module(f'deconv{i}', block)

    def forward(self, x):
        return self.deconvs(x)


class PointCloudEncoderGlobal(nn.Module):
    """PointNet with an option to process global features of selected points."""
    def __init__(self, in_features, out_features, layers, act=nn.ReLU, features_from_layers=(0,)):
        super().__init__()

        layers = (in_features,) + layers
        self.layers = nn.ModuleList()
        for i in range(len(layers) - 1):
            block = nn.Sequential(
                nn.Linear(layers[i], layers[i + 1]),
                act(),
            )
            self.layers.append(block)

        if isinstance(features_from_layers, int):
            features_from_layers = (features_from_layers, )
        self.selected_layers = features_from_layers
        
        self.fc_size = layers[-1] * (1 + sum([layers[i] for i in self.selected_layers]))
        self.fc = LayerNormTanhEmbedding(self.fc_size, out_features)

    def forward(self, x):
        features = [x]
        for layer in self.layers:
            x = layer(x)
            features.append(x)

        values, indices = x.max(-2)
        if len(self.selected_layers):
            selected_features = torch.cat(
                [self._gather(features[ind], indices) for ind in self.selected_layers],
                -1)
            values = torch.cat((values.unsqueeze(-1), selected_features), -1).flatten(-2)
        return self.fc(values)

    @staticmethod
    def _gather(features, indices):
        indices = torch.repeat_interleave(indices.unsqueeze(-1), features.size(-1), -1)
        return torch.gather(features, -2, indices)


class PixelEncoder(nn.Module):
    def __init__(self, in_channels=3, out_features=64, depth=32, act=nn.ReLU):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(in_channels, depth, 3, 2),
            act(),
            nn.Conv2d(depth, depth, 3, 1),
            act(),
            # nn.Flatten(),
            # LayerNormTanhEmbedding(depth*39*39, out_features)
            nn.Conv2d(depth, depth, 3, 1),
            act(),
            nn.Conv2d(depth, depth, 3, 1),
            act(),
            nn.Flatten(),
            LayerNormTanhEmbedding(depth*35*35, out_features)
        )

    def forward(self, img):
        prefix_shape = img.shape[:-3]
        img = img.flatten(0, len(prefix_shape)-1)
        img = self.convs(img)
        img = img.reshape(*prefix_shape, -1)
        return img


class PixelDecoder(nn.Module):
    def __init__(self, in_features, out_channels=3, depth=32, act=nn.ReLU):
        super().__init__()
        dim = 35 # 39 - for two conv layers, 35 for 4 layers
        self.deconvs = nn.Sequential(
            nn.Linear(in_features, depth*dim**2),
            act(),
            nn.Unflatten(-1, (depth, dim, dim)),
            nn.ConvTranspose2d(depth, depth, 3, 1),
            act(),
            nn.ConvTranspose2d(depth, depth, 3, 1),
            act(),
            nn.ConvTranspose2d(depth, depth, 3, 1),
            act(),
            nn.ConvTranspose2d(depth, out_channels, 3, 2, output_padding=1),
        )

    def forward(self, x):
        prefix_shape = x.shape[:-1]
        x = x.flatten(0, len(prefix_shape)-1)
        img = self.deconvs(x)
        img = img.reshape(*prefix_shape, 3, 84, 84)
        return img
