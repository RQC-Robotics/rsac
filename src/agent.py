import torch
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
            state = torch.zeros((obs.size(0), self._c.hidden_dim), device=self.device)
        obs, _ = self._target_encoder(obs)
        state = self._target_cell(obs, state)
        dist = self._target_actor(state)
        if training:
            action = dist.sample()
        else:
            action = dist.sample([100]).mean(0)
        action = torch.clamp(action, -utils.ACT_LIM, utils.ACT_LIM)
        log_prob = dist.log_prob(action)
        return action, log_prob, state

    def step(self, obs, actions, rewards, log_probs, hidden_states):
        obs_emb, _ = self.encoder(obs)
        target_obs_emb, _ = self._target_encoder(obs)
        states = self.cell_roll(self.cell, obs_emb, hidden_states[0], bptt=self._c.bptt)
        target_states = self.cell_roll(self._target_cell, target_obs_emb, hidden_states[0])

        alpha = torch.maximum(self._log_alpha, torch.full_like(self._log_alpha, -20.))
        alpha = F.softplus(alpha) + 1e-8

        rl_loss = self._policy_learning(states, actions, rewards, log_probs, target_states, alpha.detach())
        auxiliary_loss = self._auxiliary_loss(obs, actions, states, target_states)
        actor_loss, dual_loss = self._policy_improvement(target_states, alpha)
        model_loss = rl_loss + auxiliary_loss + actor_loss + dual_loss

        self.optim.zero_grad()
        model_loss.backward()
        clip_grad_norm_(self.parameters(), self._c.max_grad)
        self.optim.step()
        self.callback.add_scalar('train/actor_loss', actor_loss.item(), self._step)
        self.callback.add_scalar('train/auxiliary_loss', auxiliary_loss.item(), self._step)
        self.callback.add_scalar('train/critic_loss', rl_loss.item(), self._step)
        # self.callback.add_scalar('train/actor_grads', utils.grads_sum(self.actor), self._step)
        # self.callback.add_scalar('train/critic_grads', utils.grads_sum(self.critic), self._step)
        # self.callback.add_scalar('train/encoder_grads', utils.grads_sum(self.encoder), self._step)
        # self.callback.add_scalar('train/cell_grads', utils.grads_sum(self.cell), self._step)
        self.update_targets()
        self._step += 1

    def _policy_learning(self, states, actions, rewards, behaviour_log_probs, target_states, alpha):
        with torch.no_grad():
            target_dist = self._target_actor(target_states)
            sampled_actions = target_dist.sample([self._c.num_samples])
            sampled_log_probs = target_dist.log_prob(sampled_actions).unsqueeze(-1)
            values = self._target_critic(torch.repeat_interleave(target_states[None], self._c.num_samples, 0),
                                         sampled_actions).min(-1, keepdim=True).values
            soft_values = torch.mean(values - alpha*sampled_log_probs, 0)
            target_values = self._target_critic(target_states, actions).min(-1, keepdim=True).values
            log_probs = target_dist.log_prob(actions).unsqueeze(-1)
            cs = torch.minimum(torch.ones_like(log_probs[0]), (log_probs - behaviour_log_probs).exp())
            deltas = rewards + self._c.discount*soft_values.roll(-1, 0) - target_values
            target_values, deltas, cs = map(lambda t: t[:-1], (target_values, deltas, cs))
            target_values = utils.retrace(target_values, deltas, cs, self._c.discount, self._c.disclam)

        q_values = self.critic(states, actions)

        loss = (q_values[:-1] - target_values).pow(2)

        self.callback.add_scalar('train/mean_reward', rewards.mean().item(), self._step)
        self.callback.add_scalar('train/mean_value', q_values.mean().item(), self._step)
        # self.callback.add_scalar('train/retrace_weight', cs.mean(), self._step)
        # self.callback.add_scalar('train/mean_deltas', deltas.mean(), self._step)
        return loss.mean()

    def _policy_improvement(self, states, alpha):
        dist = self.actor(states)
        actions = dist.rsample([self._c.num_samples])
        log_prob = dist.log_prob(actions)
        q_values = self._target_critic(torch.repeat_interleave(states[None], self._c.num_samples, 0),
                                       actions).min(-1).values

        with torch.no_grad():
            ent = -log_prob.mean()
            self.callback.add_scalar('train/actor_entropy', ent.item(), self._step)
            self.callback.add_scalar('train/alpha', alpha.item(), self._step)

        actor_loss = torch.mean(alpha.detach() * log_prob - q_values, 0)
        actor_loss = self._masked_discount(actor_loss, self._c.burn_in) * actor_loss
        dual_loss = - alpha * (log_prob.detach() + self._target_entropy)
        return actor_loss.mean(), dual_loss.mean()

    def _auxiliary_loss(self, obs, actions, states_emb, target_states_emb):
        # todo check l2 reg
        if self._c.aux_loss == 'None':
            return torch.tensor(0.)
        elif self._c.aux_loss == 'reconstruction':
            obs_pred = self.decoder(states_emb)
            if self._c.observe == 'point_cloud':
                loss = chamfer_distance(obs.flatten(0, 1), obs_pred.flatten(0, 1))[0]
            else:
                loss = (obs_pred - obs).pow(2)
            return loss.mean()
        elif self._c.aux_loss == 'contrastive':
            #import pdb; pdb.set_trace()
            actions = actions[:-1].unfold(0, self._c.spr_depth, 1).flatten(0, 1).movedim(-1, 0)
            states_emb = states_emb[:-self._c.spr_depth].flatten(0, 1)

            pred_embeds = self.cell_roll(self.dm, actions, states_emb)
            pred_embeds = self.projection(pred_embeds)
            pred_embeds = self.prediction(pred_embeds)

            target_states_emb = self._target_projection(target_states_emb[1:])
            target_states_emb = target_states_emb.unfold(0, self._c.spr_depth, 1).flatten(0, 1).movedim(-1, 0)

            contrastive_loss = - self.cos_sim(pred_embeds, target_states_emb)
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
        # TODO: weight decay, value-function, make contrastive dimensions larger
        emb = self._c.obs_emb_dim
        hidden = self._c.hidden_dim
        self.device = torch.device(self._c.device if torch.cuda.is_available() else 'cpu')
        # RL
        self.cell = nn.GRUCell(emb, hidden)
        self.actor = models.Actor(hidden, self.act_dim, layers=self._c.actor_layers,
                                  mean_scale=self._c.mean_scale, init_std=self._c.init_std)

        self.value = utils.build_mlp((hidden, *self._c.critic_layers, 1))
        self.critic = models.Critic(hidden + self.act_dim, self._c.critic_layers)

        # SPR
        self.dm = nn.GRUCell(self.act_dim, hidden)
        self.projection = utils.build_mlp((hidden, emb, emb))
        self.prediction = utils.build_mlp((emb, emb, emb))
        self.cos_sim = nn.CosineSimilarity(dim=-1)

        obs_dim = self.obs_spec.shape[0]
        if self._c.observe == 'states':
            self.encoder = models.DummyEncoder(obs_dim, emb)
            self.decoder = nn.Linear(hidden, obs_dim)
        elif self._c.observe in wrappers.PixelsWrapper.channels.keys():
            self.encoder = models.PixelEncoder(obs_dim, emb)
            self.decoder = models.PixelDecoder(hidden, obs_dim)
        elif self._c.observe == 'point_cloud':
            # self.encoder = models.PointCloudEncoder(3, emb, layers=self._c.pn_layers,
            #                                         dropout=self._c.pn_dropout)
            self.encoder = models.PointCloudEncoderGlobal(3, emb, sizes=self._c.pn_layers,
                                                          dropout=self._c.pn_dropout, features_from_layers=())
            self.decoder = models.PointCloudDecoder(hidden, layers=self._c.pn_layers, pn_number=self._c.pn_number)

        self._log_alpha = nn.Parameter(torch.tensor(self._c.init_log_alpha))

        self._target_encoder, self._target_actor, self._target_value,\
            self._target_critic, self._target_cell, self._target_projection =\
            map(lambda m: deepcopy(m).requires_grad_(False),
                (self.encoder, self.actor, self.value, self.critic, self.cell, self.projection))

        def param_group(*modules):
            return nn.ParameterList(chain(*map(nn.Module.parameters, modules)))
        self._rl_params = param_group(self.cell, self.critic, self.value)
        self._ae_params = param_group(self.encoder, self.dm, self.projection, self.prediction, self.decoder)

        self.optim = torch.optim.Adam([
            {'params': self.actor.parameters(), 'lr': self._c.actor_lr},
            {'params': self._rl_params, 'lr': self._c.critic_lr},
            {'params': self._ae_params, 'lr': self._c.encoder_lr, 'weight_decay': self._c.weight_decay},
            {'params': [self._log_alpha], 'lr': self._c.dual_lr}
        ])
        self._target_entropy = -self.act_dim
        self.to(self.device)

    @torch.no_grad()
    def update_targets(self):
        # utils.soft_update(self._target_encoder, self.encoder, self._c.encoder_tau)
        # utils.soft_update(self._target_projection, self.projection, self._c.encoder_tau)
        # utils.soft_update(self._target_cell, self.cell, self._c.critic_tau)
        # utils.soft_update(self._target_critic, self.critic, self._c.critic_tau)
        # utils.soft_update(self._target_actor, self.actor, self._c.actor_tau)
        def predicate(t):
            return self._step % t == 0
        utils.hard_update(self._target_encoder, self.encoder, predicate(self._c.encoder_update))
        utils.hard_update(self._target_projection, self.projection, predicate(self._c.encoder_update))
        utils.hard_update(self._target_actor, self.actor, predicate(self._c.actor_update))
        utils.hard_update(self._target_value, self.value, predicate(self._c.critic_update))
        utils.hard_update(self._target_cell, self.cell, predicate(self._c.critic_update))
        utils.hard_update(self._target_critic, self.critic, predicate(self._c.critic_update))

