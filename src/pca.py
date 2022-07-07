import torch
import pathlib
import argparse
import matplotlib.pyplot as plt
from . import utils, wrappers
from .config import Config
from .core import RLAlg
nn = torch.nn


class PCA(nn.Module):
    def __init__(self, alg, lr=1e-3):
        super().__init__()
        self.alg = alg
        self.head = nn.Linear(alg.config.hidden_dim, len(alg.env.physics.state())).to(alg.agent.device)
        self.optim = torch.optim.Adam(self.head.parameters(), lr=lr)

    def observe(self, obs, hidden):
        with torch.no_grad():
            obs = self.alg.agent.encoder(obs)
            hidden = self.alg.agent.cell(obs, hidden)
        return self.head(hidden), hidden

    def learn(self, observations, states):
        states_pred = self(observations)
        loss = (states_pred - states).pow(2).mean()
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        return loss.item()

    def forward(self, observations, hidden=None):
        if not torch.is_tensor(hidden):
            hidden = torch.zeros(observations.size(1), self.alg.config.hidden_dim, device=self.alg.agent.device)
        states_pred = []
        for obs in observations:
            state, hidden = self.observe(obs, hidden)
            states_pred.append(state)
        return torch.stack(states_pred)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('path', help='Path to directory containing weights and config.')
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--batch_size', type=int, default=5)
    return parser.parse_args()


def train_pca(path, lr, epochs, batch_size):
    alg = RLAlg.load(path)
    alg.agent.requires_grad_(False)
    pca = PCA(alg, lr)
    monitor = wrappers.Monitor(alg.env)

    def get_data(n=1):
        obs_batch, states_batch = [], []
        for _ in range(n):
            utils.simulate(monitor, alg.policy, training=False)
            data = monitor.data
            obs, states = map(lambda k: torch.from_numpy(data[k]).to(alg.agent.device).unsqueeze(1),
                              ('observations', 'states'))
            obs_batch.append(obs)
            states_batch.append(states)
        return map(lambda x: torch.cat(x, 1), (obs_batch, states_batch))

    for i in range(epochs):
        observations, states = get_data(batch_size)
        loss = pca.learn(observations, states)
        print(f'Epoch {i} loss: {loss}')

    observations, states = get_data()
    states_pred, states = map(lambda t: t.detach().squeeze(1).cpu().numpy(),
                              (pca(observations), states))
    for i in range(states_pred.shape[-1]):
        plt.figure(figsize=(10, 6))
        plt.title(f'proprietary state[{i}]')
        plt.plot(states_pred[:, i], label='pred')
        plt.plot(states[:, i], label='truth')
        plt.xlabel('timestep')
        plt.legend()
        plt.savefig(f'coord{i}.pdf', format='pdf')

    return pca


if __name__ == "__main__":
    args = parse_args()
    pca = train_pca(args.path, args.lr, args.epochs, args.batch_size)

