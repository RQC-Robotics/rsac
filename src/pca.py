import torch
from torch.utils.data import DataLoader, Dataset
from collections import defaultdict
from . import utils, wrappers
nn = torch.nn


class PCA(nn.Module):
    def __init__(self, alg):
        super().__init__()
        self.encoder = alg.agent.encoder
        self.device = alg.agent.device
        self.head = nn.Linear(alg.config.obs_emb_dim, len(alg.env.physics.state())).to(self.device)
        self.optim = torch.optim.SGD(self.head.parameters(), lr=1e-3)

    def forward(self, observations):
        if not torch.is_tensor(observations):
            observations = torch.from_numpy(observations)
        observations = observations.to(self.device)
        with torch.no_grad():
            obs_emb, _ = self.encoder(observations)
        return self.head(obs_emb)

    def learn(self, observations, states):
        states_pred = self(observations)
        loss = (states_pred - states.to(self.device)).pow(2).mean()
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



import matplotlib.pyplot as plt
from IPython.display import clear_output # shouldn't present

def encoder_pca(alg, epochs=10):
    pca = PCA(alg)
    monitor = wrappers.Monitor(alg.env, '.')

    def policy(obs, state, training):
        obs = torch.from_numpy(obs[None]).to(pca.device)
        action, state = alg.agent.policy(obs, state, training)
        return action.cpu().detach().numpy().flatten(), state

    logs = []
    for _ in range(epochs):
        tr = utils.simulate(monitor, policy, False)
        ds = DictDataset(monitor.data)
        dl = DataLoader(ds, batch_size=10, shuffle=True)
        loss = 0
        for item in dl:
            loss += pca.learn(item['observations'], item['states'])
        logs.append(loss)

        clear_output(wait=True)
        plt.plot(logs)
        plt.show()

    clear_output(wait=True)
    tr = utils.simulate(monitor, policy, False)
    data = monitor.data
    states_pred = pca(data['observations'].to(pca.device)).detach().cpu().numpy()
    states = data['states'].detach().cpu().numpy()
    for i in range(states_pred.shape[-1]):
        plt.figure(figsize=(10, 6))
        plt.plot(states_pred[:, i])
        plt.plot(states[:, i])
        plt.show()



