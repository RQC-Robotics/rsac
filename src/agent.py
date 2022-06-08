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
    def policy(self, obs, training):
        state = self._target_encoder(obs)
        dist = self._target_actor(state)

        if training:
            action = dist.sample()
        else:
            action = dist.sample([100]).mean(0)
        log_prob = dist.log_prob(action).sum(-1)
        return action, log_prob

    def step(self, obs, actions, rewards, dones, log_probs):
        states = self.encoder(obs)
        target_states = self._target_encoder(obs)

        alpha = torch.maximum(self._log_alpha, torch.full_like(self._log_alpha, -18.))
        alpha = F.softplus(alpha) + 1e-7

        rl_loss = self._policy_learning(states, actions, rewards, dones, log_probs,
                                        target_states, alpha.detach())
        auxiliary_loss = self._auxiliary_loss(obs, states)
        actor_loss, dual_loss = self._policy_improvement(states.detach(), alpha)
        model_loss = rl_loss + auxiliary_loss + actor_loss + dual_loss

        self.optim.zero_grad()
        model_loss.backward()
        clip_grad_norm_(self._rl_params, self._c.max_grad)
        clip_grad_norm_(self._ae_params, self._c.max_grad)
        self.optim.step()

        if self._c.debug:
            self.callback.add_scalar('train/actor_loss', actor_loss, self._step)
            self.callback.add_scalar('train/auxiliary_loss', auxiliary_loss, self._step)
            self.callback.add_scalar('train/critic_loss', rl_loss, self._step)
            self.callback.add_scalar('train/actor_grads', utils.grads_sum(self.actor), self._step)
            self.callback.add_scalar('train/critic_grads', utils.grads_sum(self.critic), self._step)
            self.callback.add_scalar('train/encoder_grads',
                                     utils.grads_sum(self.encoder),
                                     self._step)
        self._update_targets()
        self._step += 1

    def _policy_learning(self, states, actions, rewards, dones, behaviour_log_probs, target_states,
                         alpha):
        with torch.no_grad():
            target_dist = self._target_actor(target_states)
            sampled_actions = target_dist.sample()
            sampled_log_probs = target_dist.log_prob(sampled_actions)

            q_values = self._target_critic(
                target_states, sampled_actions).min(-1, keepdim=True).values

            soft_values = q_values - (alpha * sampled_log_probs).sum(-1, keepdim=True)
            target_q_values = self._target_critic(
                target_states, actions).min(-1, keepdim=True).values

            log_probs = target_dist.log_prob(actions).sum(-1, keepdim=True)
            cs = torch.minimum(torch.tensor(1.), (log_probs - behaviour_log_probs).exp())
            dones = dones.float()
            deltas = rewards + self._c.discount * (1. - dones) * soft_values.roll(-1, 0) \
                     - target_q_values
            # Bootstrapped value is known(=0) if the last transition is terminal
            #   so we can use it in Q-value update, otherwise mask it.
            deltas[-1] *= dones[-1]
            deltas = utils.retrace(deltas, cs, self._c.discount, self._c.disclam)
            target_q_values += deltas

        q_values = self.critic(states, actions)
        loss = (q_values - target_q_values).pow(2)
        loss = self._sequence_discount(loss) * loss

        if self._c.debug:
            self.callback.add_scalar('train/mean_reward',
                                     rewards.detach().mean() / self._c.action_repeat, self._step)
            self.callback.add_scalar('train/mean_value', q_values.detach().mean(), self._step)
            self.callback.add_scalar('train/retrace_weight', cs.detach().mean(), self._step)
            self.callback.add_scalar('train/mean_deltas', deltas.detach().mean(), self._step)
        return loss.mean()

    def _policy_improvement(self, states, alpha):
        dist = self.actor(states)
        actions = dist.rsample()
        log_prob = dist.log_prob(actions)

        self.critic.requires_grad_(False)
        q_values = self.critic(states, actions).min(-1).values
        self.critic.requires_grad_(True)

        if self._c.debug:
            ent = -log_prob.detach().sum(-1).mean()
            self.callback.add_scalar('train/actor_entropy', ent, self._step)
            self.callback.add_scalar('train/alpha', alpha.detach().mean(), self._step)

        actor_loss = (alpha.detach() * log_prob).sum(-1) - q_values
        actor_loss = self._sequence_discount(actor_loss) * actor_loss
        dual_loss = - alpha * (log_prob.detach() + self._c.target_ent_per_dim)
        return actor_loss.mean(), dual_loss.mean()

    def _auxiliary_loss(self, obs, states_emb):
        # todo check l2 reg; introduce lagrange multipliers
        if self._c.aux_loss == 'None':
            return torch.tensor(0.)
        elif self._c.aux_loss == 'reconstruction':
            obs_pred = self.decoder(states_emb)
            if self._c.observe == 'point_cloud':
                loss = chamfer_distance(obs.flatten(0, 2), obs_pred.flatten(0, 2))[0]
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
        self.act_dim = self.env.action_spec().shape[0]
        obs_spec = self.env.observation_spec()
        self.device = torch.device(self._c.device if torch.cuda.is_available() else 'cpu')

        # RL
        self.actor = models.Actor(emb, self.act_dim, layers=self._c.actor_layers,
                                  mean_scale=self._c.mean_scale)

        self.critic = models.Critic(emb + self.act_dim, self._c.critic_layers)

        # Encoder+decoder
        obs_dim = obs_spec.shape[0]
        if self._c.observe == 'states':
            self.encoder = models.LayerNormTanhEmbedding(obs_dim, emb)
            self.decoder = nn.Linear(emb, obs_dim)
        elif self._c.observe in wrappers.PixelsWrapper.channels.keys():
            self.encoder = models.PixelsEncoder(obs_dim, emb)
            self.decoder = models.PixelsDecoder(emb, obs_dim)
        elif self._c.observe == 'point_cloud':
            frames_stack = obs_spec.shape[0]
            frame_encoder = models.PointCloudEncoder(emb, layers=self._c.pn_layers,
                                                     features_from_layers=())
            frame_decoder = models.PointCloudDecoder(emb, layers=self._c.pn_layers,
                                                     pn_number=self._c.pn_number)
            self.encoder = nn.Sequential(
                frame_encoder,
                nn.Flatten(-2),
                models.LayerNormTanhEmbedding(frames_stack * emb, emb)
            )
            self.decoder = nn.Sequential(
                nn.Linear(emb, frames_stack * emb),
                nn.Unflatten(-1, (frames_stack, emb)),
                frame_decoder
            )
        else:
            raise NotImplementedError

        self._log_alpha = nn.Parameter(torch.full((self.act_dim,), self._c.init_log_alpha))

        self._target_encoder, self._target_actor, self._target_critic = \
            utils.make_targets(self.encoder, self.actor, self.critic)

        self._rl_params = utils.make_param_group(self.critic, self.actor)
        self._ae_params = utils.make_param_group(self.encoder, self.decoder)

        self.optim = torch.optim.Adam([
            {'params': self._rl_params, 'lr': self._c.rl_lr},
            {'params': self._ae_params, 'lr': self._c.ae_lr, 'weight_decay': self._c.weight_decay},
            {'params': [self._log_alpha], 'lr': self._c.dual_lr}
        ])
        self.to(self.device)

    @torch.no_grad()
    def _update_targets(self):
        utils.soft_update(self._target_encoder, self.encoder, self._c.encoder_tau)
        utils.soft_update(self._target_critic, self.critic, self._c.critic_tau)
        utils.soft_update(self._target_actor, self.actor, self._c.actor_tau)
