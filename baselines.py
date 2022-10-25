# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
from banditsrl import BanditSRL, TORCH_FLOAT
from typing import Any, Optional
import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch.utils.tensorboard import SummaryWriter


class IGWExplorer(BanditSRL):
    def __init__(
        self, 
        cfg: DictConfig,
        num_actions: int, 
        model: Optional[nn.Module] = None, 
        feature_dim: Optional[int] = None, 
        tb_writer: Optional[SummaryWriter] = None
    ) -> None:
        super().__init__(cfg, num_actions, model, feature_dim, tb_writer)
        self.gamma_exponent = cfg.gamma_exponent
        self.gamma_scale = cfg.gamma_scale
        self.exploration_param = num_actions

    def _post_train_reset(self) -> None:
        pass

    def play_base_action(self, features: np.ndarray) -> int:
        if self.gamma_exponent == "cbrt":
            gamma = self.gamma_scale * np.cbrt(self.t + 1)
        elif self.gamma_exponent == "sqrt":
            gamma = self.gamma_scale * np.sqrt(self.t + 1)
        features_tensor = torch.tensor(features, dtype=TORCH_FLOAT, device=self.cfg.device)
        with torch.no_grad():
            if self.cfg.refit_linear:
                phi = self.model.embedding(features_tensor)
                predicted_rewards = torch.matmul(phi, self.theta).squeeze()
            else:
                predicted_rewards = self.model(features_tensor).squeeze()
        predicted_rewards = predicted_rewards.cpu().numpy()
        gap = predicted_rewards.max() - predicted_rewards
        opt_arms = np.where(gap <= self.cfg.mingap_clip)[0]
        subopt_arms = np.where(gap > self.cfg.mingap_clip)[0]
        prob = np.zeros(self.num_actions)
        prob[subopt_arms] = 1. / (self.exploration_param + gamma * gap[subopt_arms])
        prob[opt_arms] = (1 - prob[subopt_arms].sum()) / len(opt_arms)
        assert np.isclose(prob.sum(), 1)

        if self.cfg.use_tb:
            self.tb_writer.add_scalar('prob_optarms', 1 - prob[subopt_arms].sum(), self.t)
            self.tb_writer.add_scalar('gamma', gamma, self.t)
        action = self.np_random.choice(self.num_actions, 1, p=prob).item()
        return action
        

class GradientUCB(BanditSRL):

    def __init__(
        self, 
        cfg: DictConfig, 
        num_actions: int, 
        model: Optional[nn.Module] = None, 
        feature_dim: Optional[int] = None, 
        tb_writer: Optional[SummaryWriter] = None
    ) -> None:
        super().__init__(cfg, num_actions, model, feature_dim, tb_writer)
        self.num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.Z = self.cfg.ucb_regularizer * torch.ones((self.num_params, ), dtype=TORCH_FLOAT, device=self.cfg.device)

    def _post_train_reset(self) -> None:
        pass

    def play_base_action(self, features: np.ndarray) -> int:
        features_tensor = torch.tensor(features, dtype=TORCH_FLOAT, device=self.cfg.device)
        predicted_rewards = self.model(features_tensor).squeeze()
        gs = []
        ucbs = []
        bonuses = []
        for reward in predicted_rewards:
            self.model.zero_grad()
            reward.backward(retain_graph=True)
            g = torch.cat([p.grad.flatten().detach() for p in self.model.parameters()])
            gs.append(g)
            bonus = g * g / self.Z
            bonus = self.cfg.bonus_scale * torch.sqrt(torch.sum(bonus))
            ucb = (reward + bonus).item()
            bonuses.append(bonus.item())
            ucbs.append(ucb)
        action = np.argmax(ucbs)
        self.Z += gs[action] * gs[action]

        if self.cfg.use_tb:
            self.tb_writer.add_scalar('bonus selected action', bonuses[action], self.t)
        return action
