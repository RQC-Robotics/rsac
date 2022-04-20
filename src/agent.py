import torch
import pdb
from . import models, utils, wrappers
from copy import deepcopy
from torch.nn.utils import clip_grad_norm_
from itertools import chain
from pytorch3d.loss import chamfer_distance

torch.autograd.set_detect_anomaly(True)
td = torch.distributions
nn = torch.nn
F = nn.functional


class RSAC(nn.Module):
    def __init__(self, obs_spec, act_spec, config, callback):
        super().__init__()
        self.obs_spec = obs_spec
        self.act_dim = act_spec.shape[0]
        self.callback = callback
        self._c = config
        self._step = 0
        self._build()

    @torch.no_grad()
    def policy(self, obs, state, training):
        if not torch.is_tensor(state):
            state = self.init_hidden(obs.size(0))
        obs, _ = self._target_encoder(obs)
        state = self._target_cell(obs, state)
        dist = self._target_actor(state)
        if training:
            action = dist.sample()
            action = action + self._c.expl_noise*torch.randn_like(action)
        else:
            action = dist.sample([100]).mean(0)
        action = torch.clamp(action, -utils.ACT_LIM, utils.ACT_LIM)
        log_prob = dist.log_prob(action)
        return action, log_prob, state

    def step(self, obs, actions, rewards, log_probs, hidden_states):
        assert not log_probs.isnan().any()
        #it may be better to store next_obs in the buffer to not shift batch every time
        obs_emb, _ = self.encoder(obs)
        target_obs_emb, _ = self._target_encoder(obs)
        states = self.cell_roll(self.cell, obs_emb, hidden_states[0], bptt=self._c.bptt)
        target_states = self.cell_roll(self._target_cell, target_obs_emb, hidden_states[0])

        alpha = torch.maximum(self._log_alpha, torch.full_like(self._log_alpha, -20.))
        alpha = F.softplus(alpha) + 1e-8

        rl_loss = self._policy_learning(states, actions, rewards, log_probs, target_states, alpha.detach())
        auxiliary_loss = self._auxiliary_loss(obs, actions, obs_emb, target_obs_emb)
        critic_loss = rl_loss + auxiliary_loss

        self.critic_optim.zero_grad()
        critic_loss.backward()
        clip_grad_norm_(self._critic_parameters, self._c.max_grad)
        self.callback.add_scalar('train/auxiliary_loss', auxiliary_loss.item(), self._step)
        self.callback.add_scalar('train/critic_loss', rl_loss.item(), self._step)
        self.callback.add_scalar('train/critic_grads', utils.grads_sum(self.critic), self._step)
        self.callback.add_scalar('train/encoder_grads', utils.grads_sum(self.encoder), self._step)
        self.callback.add_scalar('train/cell_grads', utils.grads_sum(self.cell), self._step)

        #self._critic_parameters.requires_grad_(False)
        # check if target not online
        actor_loss, dual_loss = self._policy_improvement(target_states, alpha)
        self.actor_optim.zero_grad()
        self.dual_optim.zero_grad()
        actor_loss.backward()
        dual_loss.backward()
        clip_grad_norm_(self.actor.parameters(), self._c.max_grad)
        self.callback.add_scalar('train/actor_loss', actor_loss.item(), self._step)
        self.callback.add_scalar('train/actor_grads', utils.grads_sum(self.actor), self._step)
        self.critic_optim.step()
        self.actor_optim.step()
        self.dual_optim.step()
        self.update_targets()
        self._step += 1

    def _policy_learning(self, states, actions, rewards, behaviour_log_probs, target_states, alpha):
        states, actions, rewards, behaviour_log_probs, target_states, next_states = \
            map(lambda t: t[:-1],
                (states, actions, rewards, behaviour_log_probs, target_states, target_states.roll(-1, 0)))

        with torch.no_grad():  # none of ops require grads but still
            target_dist = self._target_actor(next_states)
            sampled_actions = target_dist.sample([self._c.num_samples])
            next_log_probs = target_dist.log_prob(sampled_actions).unsqueeze(-1)
            next_values = self._target_critic(next_states[None].expand(self._c.num_samples, *states.shape),
                                              sampled_actions).min(-1, keepdim=True).values
            next_values = torch.mean(next_values - alpha * next_log_probs, 0)
            # values = self._target_critic(target_states, actions).min(-1, keepdim=True).values
            # log_probs = self._target_actor(target_states).log_prob(actions).unsqueeze(-1)
            #
            # resids = rewards + self._c.munchausen * torch.clamp(alpha*log_probs, min=-1., max=0.) \
            #          + self._c.discount*next_values - values
            #
            # cs = torch.minimum(torch.ones_like(log_probs), (log_probs - behaviour_log_probs).exp())
            #
            # target_values = utils.retrace(values, resids, cs, self._c.discount, self._c.disclam)
            # GVE uses on-policy target from the buffer
            target_values = utils.gve(rewards, next_values, self._c.discount, 1. - self._c.disclam)

        q_values = self.critic(states, actions)

        loss = (q_values - target_values).pow(2)
        loss = self._masked_discount(loss, self._c.burn_in)*loss

        self.callback.add_scalar('train/mean_reward', rewards.mean().item(), self._step)
        self.callback.add_scalar('train/mean_value', q_values.mean().item(), self._step)
        #self.callback.add_scalar('train/mean_retrace_weight', cs.mean().item(), self._step)
        #self.callback.add_scalar('train/mean_retrace_delta', (target_values - values).mean().item(), self._step)
        return loss.mean()

    def _policy_improvement(self, states, alpha):
        # pdb.set_trace()
        assert not states.requires_grad
        dist = self.actor(states)
        actions = dist.rsample([self._c.num_samples])
        log_prob = dist.log_prob(actions)
        q_values = self._target_critic(states[None].expand(self._c.num_samples, *states.shape), actions).min(-1).values

        with torch.no_grad():
            ent = -log_prob.mean()
            self.callback.add_scalar('train/actor_entropy', ent.item(), self._step)
            self.callback.add_scalar('train/alpha', alpha.item(), self._step)

        actor_loss = torch.mean(alpha.detach() * log_prob - q_values, 0)
        actor_loss = self._masked_discount(actor_loss) * actor_loss
        dual_loss = - alpha * (log_prob.detach() + self._target_entropy)
        return actor_loss.mean(), dual_loss.mean()

    def _auxiliary_loss(self, obs, actions, obs_emb, target_obs_emb):
        # todo check l2 reg
        if self._c.aux_loss == 'None':
            return torch.tensor(0., requires_grad=True)
        elif self._c.aux_loss == 'reconstruction':
            obs_pred = self.decoder(obs_emb)
            if self._c.observe == 'point_cloud':
                loss = chamfer_distance(obs.flatten(0, 1), obs_pred.flatten(0, 1))[0]
            else:
                loss = (obs_pred - obs).pow(2).mean()
            return loss
        elif self._c.aux_loss == 'contrastive':
            actions = actions[:-1].unfold(0, self._c.spr_depth, 1).flatten(0, 1).movedim(-1, 0)
            obs_emb = obs_emb[:-self._c.spr_depth].flatten(0, 1)

            pred_embeds = self.cell_roll(self.dm, actions, obs_emb)
            pred_embeds = self.projection(pred_embeds)
            pred_embeds = self.prediction(pred_embeds)

            target_obs_emb = self._target_projection(target_obs_emb[1:])
            target_obs_emb = target_obs_emb.unfold(0, self._c.spr_depth, 1).flatten(0, 1).movedim(-1, 0)

            contrastive_loss = - self.cos_sim(pred_embeds, target_obs_emb)
            return self._c.spr_coef*contrastive_loss.mean()

        else:
            raise NotImplementedError

    @staticmethod
    def cell_roll(cell, inp, state, bptt=-1):
        states = []
        for i, x in enumerate(inp):
            if bptt > 0 and i % bptt == 0:
                state = state.detach()
            state = cell(x, state)
            states.append(state)
        return torch.stack(states)

    def _masked_discount(self, x, size=-1):
        size = min(max(size, 0), x.size(0))
        mask = torch.cat([torch.zeros(size, device=self.device), torch.ones(x.size(0) - size, device=self.device)])
        discount = self._c.discount ** torch.arange(x.size(0), device=self.device)
        mask = mask * discount
        while mask.ndimension() != x.ndimension():
            mask = mask.unsqueeze(-1)
        return mask

    def _build(self):
        emb = self._c.obs_emb_dim
        self.device = torch.device(self._c.device if torch.cuda.is_available() else 'cpu')
        # RL
        self.cell = nn.GRUCell(emb, self._c.hidden_dim)
        self.actor = models.Actor(self._c.hidden_dim, self.act_dim, layers=self._c.actor_layers,
                                  mean_scale=self._c.mean_scale, init_std=self._c.init_std)

        self.critic = models.Critic(self._c.hidden_dim + self.act_dim, self._c.critic_layers)

        # SPR
        self.dm = nn.GRUCell(self.act_dim, emb)
        self.projection = nn.Sequential(
            nn.Linear(emb, emb),
            nn.ELU(),
            nn.Linear(emb, emb),
        )
        self.prediction = nn.Linear(emb, emb)
        self.cos_sim = nn.CosineSimilarity(dim=-1)

        obs_dim = self.obs_spec.shape[0]
        if self._c.observe == 'states':
            self.encoder = models.DummyEncoder(obs_dim, emb)
            self.decoder = nn.Linear(emb, obs_dim)
        elif self._c.observe in wrappers.PixelsWrapper.channels.keys():
            self.encoder = models.PixelEncoder(obs_dim, emb)
            self.decoder = models.PixelDecoder(emb, obs_dim)
        elif self._c.observe == 'point_cloud':
            # self.encoder = models.PointCloudEncoder(3, emb, layers=self._c.pn_layers,
            #                                         dropout=self._c.pn_dropout)
            self.encoder = models.PointCloudEncoderGlobal(3, emb, sizes=self._c.pn_layers,
                                                          dropout=self._c.pn_dropout, features_from_layers=())
            self.decoder = models.PointCloudDecoder(emb, layers=self._c.pn_layers, pn_number=self._c.pn_number)

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
        self.critic_optim = torch.optim.Adam(self._critic_parameters, self._c.critic_lr)
        self.actor_optim = torch.optim.Adam(self.actor.parameters(), self._c.actor_lr)
        self.dual_optim = torch.optim.Adam([self._log_alpha], self._c.dual_lr)
        self._target_entropy = -self.act_dim
        self.to(self.device)

    @torch.no_grad()
    def update_targets(self):
        utils.soft_update(self._target_encoder, self.encoder, self._c.encoder_tau)
        utils.soft_update(self._target_projection, self.projection, self._c.encoder_tau)
        utils.soft_update(self._target_cell, self.cell, self._c.critic_tau)
        utils.soft_update(self._target_critic, self.critic, self._c.critic_tau)
        utils.soft_update(self._target_actor, self.actor, self._c.actor_tau)

    def init_hidden(self, batch_size=1):
        return torch.zeros((batch_size, self._c.hidden_dim), device=self.device)
