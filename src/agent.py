import torch
from torch.nn.utils import clip_grad_norm_
from pytorch3d.loss import chamfer_distance
from . import models, utils, wrappers
td = torch.distributions
nn = torch.nn
F = nn.functional


class RSAC(nn.Module):
    def __init__(self, env, config, callback):
        super().__init__()
        self.env = env
        self.callback = callback
        self._c = config
        self._step = 0
        self._build()

    @torch.no_grad()
    def policy(self, obs, state, training):
        if not torch.is_tensor(state):
            state = torch.zeros((obs.size(0), self._c.hidden_dim), device=self.device)

        obs = self.encoder(obs)
        state = self.cell(obs, state)
        dist = self.actor(state)

        if training:
            action = dist.sample()
        else:
            action = torch.tanh(dist.base_dist.mean)

        log_prob = dist.log_prob(action).sum(-1)
        return action, log_prob, state

    def step(self, obs, actions, rewards, dones, log_probs, hidden_states):
        # burn_in
        init_hidden = hidden_states[0]
        if self._c.burn_in > 0:
            target_obs_emb = self._target_encoder(obs[:self._c.burn_in])
            init_hidden = self.cell_roll(self._target_cell, target_obs_emb, init_hidden)[-1]
            obs, actions, rewards, dones, log_probs = map(
                lambda t: t[self._c.burn_in:],
                (obs, actions, rewards, dones, log_probs)
            )

        obs_emb = self.encoder(obs)
        target_obs_emb = self._target_encoder(obs)
        states = self.cell_roll(self.cell, obs_emb, init_hidden)
        target_states = self.cell_roll(self._target_cell, target_obs_emb, init_hidden)

        alpha = torch.maximum(self._log_alpha, torch.full_like(self._log_alpha, -18.))
        alpha = F.softplus(alpha) + 1e-8

        online_policy = self.actor(states.detach())
        sampled_actions = online_policy.rsample([self._c.num_samples])
        sampled_log_probs = online_policy.log_prob(sampled_actions).sum(-1, keepdim=True)

        critic_loss = self._policy_learning(
            states, actions, rewards, dones, log_probs, target_states,
            online_policy, sampled_actions, sampled_log_probs, alpha
        )

        auxiliary_loss = self._auxiliary_loss(obs, states)

        actor_loss, dual_loss = self._policy_improvement(
            states.detach(), sampled_actions, sampled_log_probs, alpha)
        model_loss = critic_loss + auxiliary_loss + actor_loss + dual_loss

        self.optim.zero_grad()
        model_loss.backward()
        clip_grad_norm_(self._rl_params, self._c.max_grad)
        clip_grad_norm_(self._ae_params, self._c.max_grad)
        self.optim.step()

        if self._c.debug:
            self.callback.add_scalar('train/actor_loss', actor_loss, self._step)
            self.callback.add_scalar('train/auxiliary_loss', auxiliary_loss, self._step)
            self.callback.add_scalar('train/critic_loss', critic_loss, self._step)
            self.callback.add_scalar('train/actor_grads', utils.grads_sum(self.actor), self._step)
            self.callback.add_scalar('train/critic_grads', utils.grads_sum(self.critic), self._step)
            self.callback.add_scalar(
                'train/encoder_grads', utils.grads_sum(self.encoder), self._step)
            self.callback.add_scalar('train/cell_grads', utils.grads_sum(self.cell), self._step)
        self._update_targets()
        self._step += 1

    def _policy_learning(
            self,
            states,
            actions,
            rewards,
            dones,
            behaviour_log_probs,
            target_states,
            online_policy,
            sampled_actions,
            sampled_log_probs,
            alpha
    ):
        del dones  # not used for continuous control tasks
        with torch.no_grad():
            q_values = self._target_critic(
                torch.repeat_interleave(target_states[None], self._c.num_samples, 0),
                sampled_actions
            ).min(-1, keepdim=True).values

            soft_values = torch.mean(
                q_values - alpha * sampled_log_probs, 0)
            target_q_values = self._target_critic(
                target_states, actions).min(-1, keepdim=True).values

            log_probs = online_policy.log_prob(actions).sum(-1, keepdim=True)
            cs = torch.minimum(torch.tensor(1.), (log_probs - behaviour_log_probs).exp())

            deltas = rewards + self._c.discount * soft_values.roll(-1, 0) - target_q_values
            target_q_values, deltas, cs = map(
                lambda t: t[:-1], (target_q_values, deltas, cs))
            deltas = utils.retrace(deltas, cs, self._c.discount, self._c.disclam)
            target_q_values += deltas

        q_values = self.critic(states, actions)
        loss = (q_values[:-1] - target_q_values).pow(2)
        loss = self._sequence_discount(loss) * loss

        if self._c.debug:
            self.callback.add_scalar('train/mean_reward',
                                     rewards.detach().mean() / self._c.action_repeat, self._step)
            self.callback.add_scalar('train/mean_value', q_values.detach().mean(), self._step)
            self.callback.add_scalar('train/retrace_weight', cs.detach().mean(), self._step)
            self.callback.add_scalar('train/mean_deltas', deltas.detach().mean(), self._step)
        return loss.mean()

    def _policy_improvement(
            self,
            states,
            actions,
            log_probs,
            alpha
    ):
        log_probs = log_probs.squeeze(-1)
        self.critic.requires_grad_(False)
        q_values = self.critic(
            torch.repeat_interleave(states[None], self._c.num_samples, 0),
            actions
        ).min(-1).values
        self.critic.requires_grad_(True)

        if self._c.debug:
            ent = -log_probs.detach().mean()
            self.callback.add_scalar('train/actor_entropy', ent, self._step)
            self.callback.add_scalar('train/alpha', alpha.detach().mean(), self._step)
        actor_loss = alpha.detach() * log_probs - q_values
        actor_loss = self._sequence_discount(actor_loss) * actor_loss
        dual_loss = - alpha * (log_probs.detach() + self._target_entropy)
        return actor_loss.mean(), dual_loss.mean()

    def _auxiliary_loss(self, obs, states_emb):
        # todo check l2 reg; introduce lagrange multipliers
        if self._c.aux_loss == 'None':
            return torch.tensor(0.)
        elif self._c.aux_loss == 'reconstruction':
            obs_pred = self.decoder(states_emb)
            if self._c.observe == 'point_cloud':
                loss = chamfer_distance(obs.flatten(0, 1), obs_pred.flatten(0, 1))[0]
            else:
                loss = (obs_pred - obs).pow(2)
            return loss.mean()
        else:
            raise NotImplementedError

    @staticmethod
    def cell_roll(cell, inp, state):
        states = []
        for x in inp:
            state = cell(x, state)
            states.append(state)
        return torch.stack(states)

    def _sequence_discount(self, x):
        discount = self._c.discount ** torch.arange(x.size(0), device=self.device)
        shape = (x.ndimension() - 1) * (1,)
        return discount.reshape(-1, *shape)

    def _build(self):
        emb = self._c.obs_emb_dim
        hidden = self._c.hidden_dim
        act_dim = self.env.action_spec().shape[0]
        obs_spec = self.env.observation_spec()
        self.device = torch.device(self._c.device if torch.cuda.is_available() else 'cpu')

        # RL
        self.cell = nn.GRUCell(emb, hidden)
        self.actor = models.Actor(hidden, act_dim, layers=self._c.actor_layers,
                                  mean_scale=self._c.mean_scale)

        self.critic = models.Critic(hidden + act_dim, self._c.critic_layers)

        # Encoder+decoder
        obs_dim = obs_spec.shape[0]
        if self._c.observe == 'states':
            self.encoder = models.LayerNormTanhEmbedding(obs_dim, emb)
            self.decoder = nn.Linear(emb, obs_dim)
        elif self._c.observe in wrappers.PixelsWrapper.channels.keys():
            self.encoder = models.PixelsEncoder(obs_dim, emb)
            self.decoder = models.PixelsDecoder(emb, obs_dim)
        elif self._c.observe == 'point_cloud':
            self.encoder = models.PointCloudEncoder(emb, layers=self._c.pn_layers,
                                               features_from_layers=())
            self.decoder = models.PointCloudDecoder(hidden, layers=self._c.pn_layers,
                                               pn_number=self._c.pn_number)
        else:
            raise NotImplementedError

        init_log_alpha = torch.log(torch.tensor(self._c.init_temperature).exp() - 1.)
        self._log_alpha = nn.Parameter(init_log_alpha)

        self._target_encoder, self._target_critic, self._target_cell =\
            utils.make_targets(self.encoder, self.critic, self.cell)

        self._rl_params = utils.make_param_group(self.critic, self.actor)
        self._ae_params = utils.make_param_group(self.encoder, self.cell, self.decoder)

        self.optim = torch.optim.Adam([
            {'params': self._rl_params, 'lr': self._c.rl_lr},
            {'params': self._ae_params, 'lr': self._c.ae_lr, 'weight_decay': self._c.weight_decay},
            {'params': [self._log_alpha], 'lr': self._c.dual_lr}
        ])
        self._target_entropy = self._c.target_ent_per_dim * act_dim
        self.apply(utils.weight_init)
        self.to(self.device)

    @torch.no_grad()
    def _update_targets(self):
        utils.soft_update(self._target_encoder, self.encoder, self._c.encoder_tau)
        utils.soft_update(self._target_cell, self.cell, self._c.encoder_tau)
        utils.soft_update(self._target_critic, self.critic, self._c.critic_tau)
