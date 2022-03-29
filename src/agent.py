import torch
import pdb
from . import models, utils
from copy import deepcopy
from torch.nn.utils import clip_grad_norm_
from itertools import chain
from pytorch3d.loss import chamfer_distance

torch.autograd.set_detect_anomaly(True)
td = torch.distributions
nn = torch.nn
F = nn.functional


class RSAC(nn.Module):
    def __init__(self, obs_dim, act_dim, config, callback):
        super().__init__()
        self.obs_dim, self.act_dim = obs_dim, act_dim
        self.callback = callback
        self._c = config
        self._step = 0
        self._build()

    @torch.no_grad()
    def policy(self, obs, state, training):
        if not torch.is_tensor(state):
            state = self.init_hidden(obs.size(0))
        obs = self.encoder(obs)
        state = self.cell(obs, state)
        dist = self.actor(state)
        if training:
            action = dist.sample()
        else:
            action = dist.sample([100]).mean(0)
        return action, state

    def step(self, obs, actions, rewards, hidden_states):
        self._critic_parameters.requires_grad_(True)
        obs_emb = self.encoder(obs)
        states = self.cell_roll(self.cell, obs_emb, hidden_states[0])
        target_obs_emb = self._target_encoder(obs)
        target_states = self.cell_roll(self._target_cell, target_obs_emb, hidden_states[0])
        assert not target_states.requires_grad
        next_states = target_states.roll(-1, 0)

        alpha = torch.maximum(self._log_alpha, torch.full_like(self._log_alpha, -18))
        alpha = F.softplus(alpha) + 1e-7

        rl_loss = self._policy_learning(states, actions, rewards, target_states, alpha)
        auxiliary_loss = self._auxiliary_loss(obs, actions, states, target_states)
        critic_loss = rl_loss + self._c.spr_coef*auxiliary_loss

        self._critic_optim.zero_grad()
        critic_loss.backward()
        clip_grad_norm_(self._critic_parameters, self._c.max_grad)
        self.callback.add_scalar('train/auxiliary_loss', auxiliary_loss.item(), self._step)
        self.callback.add_scalar('train/critic_loss', critic_loss.item(), self._step)
        self.callback.add_scalar('train/critic_grads', utils.grads_sum(self.critic), self._step)
        self.callback.add_scalar('train/encoder_grads', utils.grads_sum(self.encoder), self._step)
        self.callback.add_scalar('train/cell_grads', utils.grads_sum(self.cell), self._step)

        self._critic_parameters.requires_grad_(False)
        actor_loss, dual_loss = self._policy_improvement(states.detach(), alpha)
        self._actor_optim.zero_grad()
        self._dual_optim.zero_grad()
        actor_loss.backward()
        dual_loss.backward()
        clip_grad_norm_(self.actor.parameters(), self._c.max_grad)
        self.callback.add_scalar('train/actor_loss', actor_loss.item(), self._step)
        self.callback.add_scalar('train/actor_grads', utils.grads_sum(self.actor), self._step)
        self._critic_optim.step()
        self._actor_optim.step()
        self._dual_optim.step()
        self.update_target()
        self._step += 1

    # def _policy_learning(self, obs, actions, rewards, next_obs, alpha):
    #
    #     target_dist = self._target_actor(next_obs)
    #     next_actions = target_dist.sample([self._c.num_samples])
    #     log_prob = target_dist.log_prob(next_actions).unsqueeze(-1)
    #     next_q_values = self._target_critic(next_obs[None].expand(self._c.num_samples, *obs.shape),
    #                                         next_actions).min(-1, keepdim=True).values
    #     target_values = next_q_values - alpha.detach()*log_prob
    #
    #     #target_values = utils.gve(rewards, target_values, self._c.discount, self._c.disclam)
    #
    #     q_values = self.critic(obs, actions)
    #     resids = rewards + self._c.discount*target_values.mean(0) - q_values
    #     resids = resids[:-1]
    #     loss = self._masked_discount(resids)*resids.pow(2)
    #
    #     self.callback.add_scalar('train/mean_reward', torch.mean(rewards).item(), self._step)
    #     self.callback.add_scalar('train/mean_value', q_values.mean().item(), self._step)
    #     return loss.mean()

    def _policy_learning(self, states, actions, rewards, target_states, alpha):
        # pdb.set_trace()
        target_dist = self._target_actor(target_states)
        sampled_actions = target_dist.sample([self._c.num_samples])
        log_prob = target_dist.log_prob(sampled_actions).unsqueeze(-1)
        target_values = self._target_critic(target_states[None].expand(self._c.num_samples, *states.shape),
                                            sampled_actions).min(-1, keepdim=True).values
        target_values = target_values - alpha.detach() * log_prob

        target_values = utils.gve(rewards, target_values.mean(0),
                                  self._c.discount, self._c.disclam)

        q_values = self.critic(states, actions)
        # resids = rewards + self._c.discount*target_values.mean(0) - q_values
        resids = q_values[:-1] - target_values
        discount = self._masked_discount(resids)
        loss = discount * resids.pow(2)

        self.callback.add_scalar('train/mean_reward', rewards.mean().item(), self._step)
        self.callback.add_scalar('train/mean_value', q_values.mean().item(), self._step)
        return loss.mean()

    def _policy_improvement(self, obs, alpha):
        # pdb.set_trace()
        dist = self.actor(obs)
        actions = dist.rsample([self._c.num_samples])
        log_prob = dist.log_prob(actions)
        q_values = self.critic(obs[None].expand(self._c.num_samples, *obs.shape), actions).min(-1).values

        with torch.no_grad():
            ent = -log_prob.mean()
            self.callback.add_scalar('train/actor_entropy', ent.item(), self._step)
            self.callback.add_scalar('train/alpha', alpha.item(), self._step)

        actor_loss = (alpha.detach() * log_prob - q_values).mean(0)
        actor_loss = self._masked_discount(actor_loss) * actor_loss
        dual_loss = - alpha * (log_prob.detach() + self._target_entropy)
        return actor_loss.mean(), dual_loss.mean()

    def _auxiliary_loss(self, observations, actions, states, target_states):
        if self._c.aux_loss == 'None':
            return torch.tensor(0., requires_grad=True)
        elif self._c.aux_loss == 'reconstruction':
            obs_pred = self.decoder(states)
            return chamfer_distance(observations.flatten(0, 1), obs_pred.flatten(0, 1))[0]
        elif self._c.aux_loss == 'contrastive':
            #pdb.set_trace()
            # todo make sure states could be reused and not computed twice
            actions = actions.unfold(0, self._c.spr_depth, 1)
            actions = actions.flatten(0, 1).movedim(-1, 0)
            states = states[:-self._c.spr_depth + 1].flatten(0, 1)
            target_states = target_states.roll(self._c.spr_depth, 0)[:-self._c.spr_depth+1].flatten(0, 1)

            predicted_states = self.cell_roll(self.dm, actions, states)
            predicted_states = self.projection(predicted_states)
            predicted_states = self.prediction(predicted_states)
            target_states = self._target_projection(target_states)

            contrastive_loss = - self.cos_sim(predicted_states, target_states)
            return contrastive_loss.mean()
        else:
            raise NotImplementedError

    @staticmethod
    def cell_roll(cell, inp, state):
        states = []
        for x in inp:
            state = cell(x, state)
            states.append(state)
        return torch.stack(states)

    def _masked_discount(self, tensor):
        m = min(max(self._c.burn_in, 0), tensor.size(0))
        mask = torch.cat([torch.zeros(m, device=self.device), torch.ones(tensor.size(0) - m, device=self.device)])
        discount = self._c.discount ** torch.arange(tensor.size(0), device=self.device)
        mask = mask * discount
        while mask.ndimension() != tensor.ndimension():
            mask = mask.unsqueeze(-1)
        return mask

    def _build(self):
        self.device = torch.device(self._c.device if torch.cuda.is_available() else 'cpu')
        self.cell = nn.GRUCell(self._c.emb_dim, self._c.hidden_dim)
        self.actor = models.Actor(self._c.hidden_dim, self.act_dim, layers=self._c.actor_layers,
                                  mean_scale=self._c.mean_scale, init_std=self._c.init_std)

        self.critic = models.Critic(self._c.hidden_dim + self.act_dim, 2, self._c.critic_layers)

        # SPR
        self.dm = nn.GRUCell(self.act_dim, self._c.hidden_dim)
        self.projection = nn.Linear(self._c.hidden_dim, self._c.hidden_dim)
        self.prediction = nn.Linear(self._c.hidden_dim, self._c.hidden_dim)
        self.cos_sim = nn.CosineSimilarity(dim=-1)

        self.encoder = models.PointCloudEncoderv2(3, self._c.pn_depth, self._c.pn_layers)
        self.decoder = models.PointCloudDecoder(in_features=self._c.hidden_dim, depth=self._c.pn_depth,
                                                layers=self._c.pn_layers, pn_number=self._c.pn_number, out_features=3)
        self._log_alpha = nn.Parameter(torch.tensor(self._c.init_log_alpha).float())
        self._target_encoder, self._target_actor, self._target_critic, self._target_cell, self._target_projection \
            = map(lambda m: deepcopy(m).requires_grad_(False),
                  (self.encoder, self.actor, self.critic, self.cell, self.projection))

        self._critic_parameters = nn.ParameterList(
            chain(*map(nn.Module.parameters,
                       [
                           self.encoder,
                           self.cell,
                           self.dm,
                           self.projection,
                           self.prediction,
                           self.decoder,
                           self.critic
                       ]
                       )
                  )
        )
        self._critic_optim = torch.optim.Adam(self._critic_parameters, self._c.critic_lr)
        self._actor_optim = torch.optim.Adam(self.actor.parameters(), self._c.actor_lr)
        self._dual_optim = torch.optim.Adam([self._log_alpha], self._c.dual_lr)
        self._target_entropy = -self.act_dim
        self.to(self.device)

    @torch.no_grad()
    def update_target(self):
        utils.soft_update(self._target_encoder, self.encoder, self._c.critic_tau)
        utils.soft_update(self._target_cell, self.cell, self._c.critic_tau)
        utils.soft_update(self._target_critic, self.critic, self._c.critic_tau)
        utils.soft_update(self._target_projection, self.projection, self._c.critic_tau)
        utils.soft_update(self._target_actor, self.actor, self._c.actor_tau)

    def init_hidden(self, batch_size=1):
        return torch.zeros((batch_size, self._c.hidden_dim), device=self.device)
