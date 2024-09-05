import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal
import torch.nn.functional as F

from niql.utils import compute_gae, shuffle_tensors_together


class MLPActor(nn.Module):

    def __init__(self, input_dim, hidden_dim, act_dim=1):
        super().__init__()
        self.noise_scale = 0.1
        self.mu_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            # nn.Linear(hidden_dim, hidden_dim),
            # nn.ReLU(),
            nn.Linear(hidden_dim, act_dim)
        )
        log_std = -0.5 * np.ones(act_dim, dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))

    def _distribution(self, inputs):
        mu = self.mu_net(inputs)
        std = torch.exp(self.log_std)
        dist = Normal(mu, std)
        return dist

    def log_prob(self, inputs, actions):
        pi = self._distribution(inputs)
        logp_a = pi.log_prob(actions)
        return logp_a

    @torch.no_grad()
    def forward(self, inputs):
        pi = self._distribution(inputs)
        actions = pi.sample()

        # Add noise to the sampled action
        noise = torch.normal(mean=0, std=self.noise_scale, size=actions.shape)
        actions += noise

        # Control the range of actions within (0, 1]
        actions = torch.sigmoid(torch.clamp(actions, -1.5, 10))
        return actions


class MLPCritic(nn.Module):

    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            # nn.Linear(hidden_dim, hidden_dim),
            # nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, inputs):
        v = self.model(inputs)
        return v


class PPOWeightAgents:

    def __init__(
            self, n_agents, input_dim, hidden_dim, learning_rate=0.000005, optimiser="adam",
            tr_epochs=5, tr_batch_size=64, target_kl=0.01, clip_ratio=0.1, value_clip_eps=0.2, device="cpu"
    ):
        self.n_agents = n_agents
        self.tr_epochs = tr_epochs
        self.tr_batch_size = tr_batch_size
        self.target_kl = target_kl
        self.clip_ratio = clip_ratio
        self.value_clip_eps = value_clip_eps

        # Create agent models
        self.actors = nn.ModuleDict({
            f"agent_{i}": MLPActor(input_dim, hidden_dim) for i in range(n_agents)
        }).to(device)
        self.critics = nn.ModuleDict({
            f"agent_{i}": MLPCritic(input_dim, hidden_dim) for i in range(n_agents)
        }).to(device)

        if optimiser == "rmsprop":
            from torch.optim import RMSprop
            self.actors_optimiser = RMSprop(params=self.actors.parameters(), lr=learning_rate)
            self.critics_optimiser = RMSprop(params=self.critics.parameters(), lr=learning_rate)

        elif optimiser == "adam":
            from torch.optim import Adam
            self.actors_optimiser = Adam(params=self.actors.parameters(), lr=learning_rate)
            self.critics_optimiser = Adam(params=self.critics.parameters(), lr=learning_rate)

        else:
            raise ValueError("choose one optimiser type from rmsprop(RMSprop) or adam(Adam)")

    def compute_weights(self, inputs):
        """
        Compute the weights for the given samples.

        :param inputs: tensor of shape [B, T, num_agents, input_dim]
        :return: weights ndarray of shape [B, T, num_agents]
        """
        weights = []
        for i in range(self.n_agents):
            agent_inputs = inputs[:, :, i]
            actor = self.actors[f"agent_{i}"]
            wts = actor(agent_inputs)
            weights.append(wts)
        weights = torch.cat(weights, dim=-1)
        return weights

    def __call__(self, *args, **kwargs):
        return self.compute_weights(*args, **kwargs)

    def update(self, inputs, weights, rewards, dones, seq_mask):
        """
        Update the models.

        :param inputs: tensor of shape [B, T, num_agents, input_dim]
        :param weights: the actions, tensor of shape [B, T, num_agents]
        :param rewards: tensor of shape [B, T, num_agents]
        :param dones: flags for indicating terminal states/obs [B, T, num_agents]
        :param seq_mask: sequence mask, tensor of shape [B, T, num_agents]
        """
        # Ensure models are in training mode
        self.actors.train()
        self.critics.train()

        # Compute initial values
        old_log_probs, old_q_vals = self._compute_vals_and_log_probs(inputs, weights)
        old_log_probs = old_log_probs.detach()
        old_q_vals = old_q_vals.detach()

        # Estimate advantages and returns using GAE
        advantages, lambda_returns = compute_gae(
            rewards=rewards,
            values=old_q_vals,
            dones=dones
        )
        advantages = advantages.detach()
        lambda_returns = lambda_returns.detach()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-7)

        # prepare data
        B, T = inputs.shape[:2]
        inputs = inputs.view(B * T, self.n_agents, -1)
        actions = weights.view(B * T, self.n_agents)
        advantages = advantages.view(B * T, self.n_agents)
        lambda_returns = lambda_returns.view(B * T, self.n_agents)
        old_log_probs = old_log_probs.view(B * T, self.n_agents)
        old_q_vals = old_q_vals.view(B * T, self.n_agents)
        seq_mask = seq_mask.view(B * T, self.n_agents)

        # Shuffle all tensors together
        inputs, actions, advantages, lambda_returns, old_log_probs, old_q_vals, seq_mask = shuffle_tensors_together(
            inputs, actions, advantages, lambda_returns, old_log_probs, old_q_vals, seq_mask
        )

        train_actor = True
        actor_losses = []
        critic_losses = []
        critic_grad_norms = []
        actor_grad_norms = []

        # update models
        for epoch in range(self.tr_epochs):
            for batch_ofs in range(0, len(inputs), self.tr_batch_size):
                batch_inputs = inputs[batch_ofs: batch_ofs + self.tr_batch_size]
                batch_actions = actions[batch_ofs: batch_ofs + self.tr_batch_size]
                batch_advantages = advantages[batch_ofs: batch_ofs + self.tr_batch_size]
                batch_returns = lambda_returns[batch_ofs: batch_ofs + self.tr_batch_size]
                batch_old_log_probs = old_log_probs[batch_ofs: batch_ofs + self.tr_batch_size]
                batch_seq_mask = seq_mask[batch_ofs: batch_ofs + self.tr_batch_size]

                # Model outputs
                log_probs, q_vals = self._compute_vals_and_log_probs(
                    batch_inputs.unsqueeze(1),
                    batch_actions.unsqueeze(1),
                )
                log_probs = log_probs.squeeze(1)
                q_vals = q_vals.squeeze(1)

                # Critics update
                self.critics_optimiser.zero_grad()

                # Clipped value function loss
                old_vals = old_q_vals[batch_ofs: batch_ofs + self.tr_batch_size]
                value_clipped = old_vals + torch.clamp(q_vals - old_vals, -self.value_clip_eps, self.value_clip_eps)

                # Unclipped and clipped losses
                critic_loss_unclipped = F.mse_loss(q_vals, batch_returns, reduction="none")
                critic_loss_clipped = F.mse_loss(value_clipped, batch_returns, reduction="none")

                # Choose the maximum of both losses
                critic_loss = torch.max(critic_loss_unclipped, critic_loss_clipped)
                critic_loss = (critic_loss * batch_seq_mask).sum() / batch_seq_mask.sum()

                critic_loss.backward()
                grad_norm_cr = torch.nn.utils.clip_grad_norm_(self.critics.parameters(), 1)
                self.critics_optimiser.step()
                critic_grad_norms.append(grad_norm_cr)
                critic_losses.append(critic_loss.item())

                # Actors update
                if train_actor:
                    self.actors_optimiser.zero_grad()
                    ratio = torch.exp(log_probs - batch_old_log_probs)
                    clip_adv = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * batch_advantages
                    actor_loss = -(torch.min(ratio * batch_advantages, clip_adv)).mean()
                    # actor_loss += (-torch.log(torch.abs(batch_actions))).mean()
                    # actor_loss += 0.5 * (torch.exp(log_probs).sum() - np.prod(list(batch_actions.shape)))
                    actor_loss.backward()
                    grad_norm_ac = torch.nn.utils.clip_grad_norm_(self.actors.parameters(), 1)
                    self.actors_optimiser.step()
                    actor_grad_norms.append(grad_norm_ac)
                    actor_losses.append(actor_loss.item())

                    # Check for actor training termination
                    approx_kl = (batch_old_log_probs - log_probs).mean().item()
                    train_actor = approx_kl <= 1.5 * self.target_kl

        # stats
        return {
            "weights_policy_actor_mean_loss": np.mean(actor_losses),
            "weights_policy_critic_mean_loss": np.mean(critic_losses),
            "weights_actor_grad_mean_grad_norm": np.mean(actor_grad_norms),
            "weights_critic_grad_mean_grad_norm": np.mean(critic_grad_norms),
        }

    def _compute_vals_and_log_probs(self, inputs, weights):
        q_vals = []
        log_probs = []
        for i in range(self.n_agents):
            actor = self.actors[f"agent_{i}"]
            critic = self.critics[f"agent_{i}"]
            agent_inputs = inputs[:, :, i]
            agent_actions = weights[:, :, i]
            q_vals.append(
                critic(agent_inputs)
            )
            log_probs.append(
                actor.log_prob(agent_inputs, agent_actions.unsqueeze(-1))
            )
        q_vals = torch.cat(q_vals, dim=2)
        log_probs = torch.cat(log_probs, dim=2)
        return log_probs, q_vals



