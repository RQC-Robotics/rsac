import pathlib
import argparse
from collections import defaultdict
from . import utils, wrappers, core
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, Dataset
nn = torch.nn


class PCA(nn.Module):
    def __init__(self, alg, lr=1e-3):
        super().__init__()
        self.encoder = alg.agent.encoder
        self.device = alg.agent.device
        self.head = nn.Linear(alg.config.obs_emb_dim, len(alg.env.physics.state())).to(alg.agent.device)
        self.optim = torch.optim.SGD(self.head.parameters(), lr=lr)

    def forward(self, observations):
        with torch.no_grad():
            obs_emb, _ = self.encoder(observations)
        return self.head(obs_emb)

    def learn(self, observations, states):
        states_pred = self(observations)
        loss = (states_pred - states).pow(2).mean()
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        return loss.item()


class DictDataset(Dataset):
    def __init__(self, data):
        self._data = data

    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self._data.items()}

    def __len__(self):
        return len(self._data['states'])


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('path', help='Path to directory containing weights and config.')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=10)
    return parser.parse_args()


def train_pca(path: str, lr: float, epochs: int):
    path = pathlib.Path(path)
    config = core.Config()
    config = config.load(path / 'config.yml')
    alg = core.RLAlg(config)
    pca = PCA(alg, lr)
    monitor = wrappers.Monitor(alg.env)

    def policy(obs, state, training):
        obs = torch.from_numpy(obs[None]).to(alg.agent.device)
        action, state = alg.agent.policy(obs, state, training)
        return action.cpu().detach().numpy().flatten(), state

    for _ in range(epochs):
        tr = utils.simulate(monitor, policy, False)
        ds = DictDataset(monitor.data)
        dl = DataLoader(ds, batch_size=50, shuffle=True)
        loss = 0
        for item in dl:
            obs, states = map(lambda k: item[k].to(alg.agent.device), ('observations', 'states'))
            loss += pca.learn(obs, states)
        print(f'Epoch loss: {loss}')

    _ = utils.simulate(monitor, policy, False)
    data = monitor.data
    states_pred = pca(data['observations'].to(alg.agent.device)).detach().cpu().numpy()
    states = data['states']
    for i in range(states_pred.shape[-1]):
        plt.figure(figsize=(10, 6))
        plt.plot(states_pred[:, i])
        plt.plot(states[:, i])
        plt.show()

    return pca


if __name__ == "__main__":
    args = parse_args()
    pca = train_pca(args.path, args.lr, args.epochs)

