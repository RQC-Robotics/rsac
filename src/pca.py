import torch
import pathlib
import argparse
import matplotlib.pyplot as plt
from . import utils, wrappers
from .config import Config
from .core import RLAlg
from torch.utils.data import DataLoader, Dataset
nn = torch.nn


class PCA(nn.Module):
    def __init__(self, alg, lr=1e-3):
        super().__init__()
        self.alg = alg
        self.head = nn.Linear(alg.config.hidden_dim, len(alg.env.physics.state())).to(alg.agent.device)
        self.optim = torch.optim.SGD(self.head.parameters(), lr=lr)

    def observe(self, obs, hidden):
        if not torch.is_tensor(hidden):
            hidden = torch.zeros(obs.size(0), self.alg.config.hidden_dim, device=self.alg.agent.device)
        with torch.no_grad():
            obs, _ = self.alg.agent.encoder(obs)
            hidden = self.alg.agent.cell(obs, hidden)
        return self.head(hidden), hidden

    def learn(self, observations, states):
        hidden = None
        states_pred = []
        for obs in observations:
            state, hidden = self.observe(obs, hidden)
            states_pred.append(state)
        states_pred = torch.stack(states_pred)
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
    parser.add_argument('--batch_size', type=int, default=50)
    return parser.parse_args()


def train_pca(path, lr, epochs, batch_size):
    path = pathlib.Path(path)
    config = Config()
    config = config.load(path / 'config.yml')
    alg = RLAlg(config)
    alg.load(path)
    pca = PCA(alg, lr)
    monitor = wrappers.Monitor(alg.env)

    def get_data(monitor, detach=False):
        utils.simulate(monitor, alg.policy, training=False)
        data = monitor.data
        obs, states = map(lambda k: torch.from_numpy(data[k]).to(alg.agent.device).unsqueeze(1),
                          ('observations', 'states'))
        if detach:
            obs, states = map(lambda t: t.detach().cpu().numpy(), (obs, states))
        return obs, states

    for _ in range(epochs):
        observations, states = get_data(monitor)
        loss = pca.learn(observations, states)
        print(f'Epoch loss: {loss}')

    states_pred, states = map(lambda t: t.detach().cpu().numpy(), (pca(obs), states))
    for i in range(states_pred.shape[-1]):
        plt.figure(figsize=(10, 6))
        plt.plot(states_pred[:, i])
        plt.plot(states[:, i])
        plt.show()

    return pca


if __name__ == "__main__":
    args = parse_args()
    pca = train_pca(args.path, args.lr, args.epochs, args.batch_size)

