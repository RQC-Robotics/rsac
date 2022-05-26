import torch
from . import models, utils, wrappers
from torch.nn.utils import clip_grad_norm_
from pytorch3d.loss import chamfer_distance
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
        log_prob = dist.log_prob(action)
        return action, log_prob

    def step(self, obs, actions, rewards, log_probs):
        states = self.encoder(obs)
        target_states = self._target_encoder(obs)

        alpha = torch.maximum(self._log_alpha, torch.full_like(self._log_alpha, -20.))
        alpha = F.softplus(alpha) + 1e-8

        rl_loss = self._policy_learning(states, actions, rewards, log_probs, target_states, alpha.detach())
        auxiliary_loss = self._auxiliary_loss(obs, actions, states, target_states)
        actor_loss, dual_loss = self._policy_improvement(target_states, alpha)
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
            self.callback.add_scalar('train/encoder_grads', utils.grads_sum(self.encoder), self._step)
            self.callback.add_scalar('train/cell_grads', utils.grads_sum(self.cell), self._step)
        self._update_targets()
        self._step += 1

    def _policy_learning(self, states, actions, rewards, behaviour_log_probs, target_states, alpha):
        with torch.no_grad():
            target_dist = self._target_actor(target_states)
            sampled_actions = target_dist.sample([self._c.num_samples])
            sampled_log_probs = target_dist.log_prob(sampled_actions).unsqueeze(-1)
            q_values = self._target_critic(
                torch.repeat_interleave(target_states[None], self._c.num_samples, 0),
                sampled_actions
            ).min(-1, keepdim=True).values
            soft_values = torch.mean(q_values - alpha*sampled_log_probs, 0)
            target_q_values = self._target_critic(target_states, actions).min(-1, keepdim=True).values

            log_probs = target_dist.log_prob(actions).unsqueeze(-1)
            cs = torch.minimum(torch.tensor(1.), (log_probs - behaviour_log_probs).exp())
            deltas = rewards + self._c.discount*soft_values.roll(-1, 0) - target_q_values
            target_q_values, deltas, cs = map(lambda t: t[:-1], (target_q_values, deltas, cs))
            deltas = utils.retrace(deltas, cs, self._c.discount, self._c.disclam)
            target_q_values += deltas

        q_values = self.critic(states, actions)
        loss = (q_values[:-1] - target_q_values).pow(2)
        loss = self._sequence_discount(loss)*loss

        if self._c.debug:
            self.callback.add_scalar('train/mean_reward', rewards.mean() / self._c.action_repeat, self._step)
            self.callback.add_scalar('train/mean_value', q_values.mean(), self._step)
            self.callback.add_scalar('train/retrace_weight', cs.mean(), self._step)
            self.callback.add_scalar('train/mean_deltas', deltas.mean(), self._step)
        return loss.mean()

    def _policy_improvement(self, states, alpha):
        dist = self.actor(states)
        actions = dist.rsample([self._c.num_samples])
        log_prob = dist.log_prob(actions)
        # todo here it is possible to separate mean and std and learn them with different learning rates:
        #   it can help exploration in exchange for more computations
        q_values = self._target_critic(
            torch.repeat_interleave(states[None], self._c.num_samples, 0),
            actions
        ).min(-1).values

        if self._c.debug:
            ent = -log_prob.detach().mean()
            self.callback.add_scalar('train/actor_entropy', ent, self._step)
            self.callback.add_scalar('train/alpha', alpha, self._step)
        
        actor_loss = torch.mean(alpha.detach() * log_prob - q_values, 0)
        actor_loss = self._sequence_discount(actor_loss)*actor_loss
        dual_loss = - alpha * (log_prob.detach().mean() + self._target_entropy)
        return actor_loss.mean(), dual_loss

    def _auxiliary_loss(self, obs, actions, states_emb, target_states_emb):
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
        elif self._c.aux_loss == 'contrastive':
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
    def cell_roll(cell, inp, state):
        states = []
        for i, x in enumerate(inp):
            state = cell(x, state)
            states.append(state)
        return torch.stack(states)

    def _sequence_discount(self, x):
        discount = self._c.discount ** torch.arange(x.size(0), device=self.device)
        shape = (x.ndimension() - 1) * (1,)
        return discount.reshape(-1, *shape)

    def _build(self):
        # TODO: weight decay, value-function, make contrastive dimensions larger
        emb = self._c.obs_emb_dim
        self.act_dim = self.env.action_spec().shape[0]
        obs_spec = self.env.observation_spec()
        self.device = torch.device(self._c.device if torch.cuda.is_available() else 'cpu')

        # RL
        self.actor = models.Actor(emb, self.act_dim, layers=self._c.actor_layers,
                                  mean_scale=self._c.mean_scale)

        self.critic = models.Critic(emb + self.act_dim, self._c.critic_layers)

        # SPR
        self.dm = nn.GRUCell(self.act_dim, emb)
        self.projection = utils.build_mlp(emb, emb, emb)
        self.prediction = utils.build_mlp(emb, emb, emb)
        self.cos_sim = nn.CosineSimilarity(dim=-1)

        frames_stack, obs_dim, *_ = obs_spec.shape
        if self._c.observe == 'states':
            encoder = models.TanhLayerNormEmbedding(obs_dim, emb)
            decoder = nn.Linear(emb, obs_dim)
        elif self._c.observe in wrappers.PixelsWrapper.channels.keys():
            encoder = models.PixelEncoder(obs_dim, emb)
            decoder = models.PixelDecoder(emb, obs_dim)
        elif self._c.observe == 'point_cloud':
            encoder = models.PointCloudEncoderGlobal(3, emb, sizes=self._c.pn_layers,
                                                     dropout=self._c.pn_dropout, features_from_layers=())
            decoder = models.PointCloudDecoder(emb, layers=self._c.pn_layers, pn_number=self._c.pn_number)

        self.encoder = nn.Sequential(
            encoder,
            nn.Flatten(-2),
            models.TanhLayerNormEmbedding(frames_stack * emb, emb)
        )
        self.decoder = nn.Sequential(
            models.TanhLayerNormEmbedding(emb, frames_stack * emb),
            nn.Unflatten(-1, (frames_stack, emb)),
            decoder
        )

        self._log_alpha = nn.Parameter(torch.tensor(self._c.init_log_alpha))

        self._target_encoder, self._target_actor, self._target_critic, self._target_projection =\
            utils.make_targets(self.encoder, self.actor, self.critic, self.projection)

        self._rl_params = utils.make_param_group(self.critic, self.actor)
        self._ae_params = utils.make_param_group(self.encoder, self.dm, self.projection, self.prediction, self.decoder)

        self.optim = torch.optim.Adam([
            {'params': self._rl_params, 'lr': self._c.rl_lr},
            {'params': self._ae_params, 'lr': self._c.ae_lr, 'weight_decay': self._c.weight_decay},
            {'params': [self._log_alpha], 'lr': self._c.dual_lr}
        ])
        self._target_entropy = -self.act_dim
        self.to(self.device)

    @torch.no_grad()
    def _update_targets(self):
        utils.soft_update(self._target_encoder, self.encoder, self._c.encoder_tau)
        utils.soft_update(self._target_projection, self.projection, self._c.encoder_tau)
        utils.soft_update(self._target_critic, self.critic, self._c.critic_tau)
        utils.soft_update(self._target_actor, self.actor, self._c.actor_tau)

