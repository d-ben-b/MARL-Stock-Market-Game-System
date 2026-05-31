import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class PPO:
    def __init__(
        self,
        network,
        learning_rate,
        gamma=0.99,
        gae_lambda=0.95,
        clip_coef=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        epochs=10,
        minibatch_size=256,
    ):
        self.network = network
        self.optimizer = optim.Adam(
            self.network.parameters(), lr=learning_rate, eps=1e-5
        )

        # PPO 超參數
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_coef = clip_coef
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.epochs = epochs
        self.minibatch_size = minibatch_size

        self.device = next(network.parameters()).device

    def update_learning_rate(self, current_update, total_updates, initial_lr):
        """處理學習率衰減"""
        frac = 1.0 - (current_update - 1.0) / total_updates
        lrnow = frac * initial_lr
        self.optimizer.param_groups[0]["lr"] = lrnow

    def compute_gae(self, rewards, values, dones, next_value, next_done):
        """計算廣義優勢估計 (GAE)"""
        num_steps = len(rewards)
        advantages = torch.zeros_like(rewards).to(self.device)
        lastgaelam = 0
        for t in reversed(range(num_steps)):
            if t == num_steps - 1:
                nextnonterminal = 1.0 - next_done
                nextvalues = next_value
            else:
                nextnonterminal = 1.0 - dones[t + 1]
                nextvalues = values[t + 1]
            delta = rewards[t] + self.gamma * nextvalues * nextnonterminal - values[t]
            advantages[t] = lastgaelam = (
                delta + self.gamma * self.gae_lambda * nextnonterminal * lastgaelam
            )
        returns = advantages + values
        return advantages, returns

    def train_step(self, b_obs, b_actions, b_logprobs, b_advantages, b_returns):
        """執行 PPO 的 Epoch 更新"""
        num_steps = len(b_obs)
        b_inds = np.arange(num_steps)

        v_loss_total = 0
        pg_loss_total = 0

        for epoch in range(self.epochs):
            np.random.shuffle(b_inds)
            for start in range(0, num_steps, self.minibatch_size):
                end = start + self.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = self.network.get_action_and_value(
                    b_obs[mb_inds], b_actions.long()[mb_inds]
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                mb_advantages = b_advantages[mb_inds]
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                    mb_advantages.std() + 1e-8
                )

                # PPO 核心：截斷目標函數 (Clipped Surrogate Objective)
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(
                    ratio, 1 - self.clip_coef, 1 + self.clip_coef
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                newvalue = newvalue.view(-1)
                v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()
                entropy_loss = entropy.mean()

                # 總 Loss
                loss = pg_loss - self.ent_coef * entropy_loss + self.vf_coef * v_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)
                self.optimizer.step()

                v_loss_total += v_loss.item()
                pg_loss_total += pg_loss.item()

        updates_count = self.epochs * (num_steps // self.minibatch_size)
        return v_loss_total / updates_count, pg_loss_total / updates_count
