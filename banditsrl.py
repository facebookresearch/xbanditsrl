import numpy as np
from typing import Any, Optional, Dict, Tuple
from omegaconf import DictConfig
from collections import deque, defaultdict
from prometheus_client import Summary
from tqdm import tqdm
import time
import os
import pickle
import torch
import torch.nn as nn
from torch.nn import functional as F
TORCH_FLOAT = torch.float64
from torch.utils.tensorboard import SummaryWriter
import pdb

class DataBuffer:

    def __init__(self, capacity: int, seed: int) -> None:
        self.buffer = deque(maxlen=capacity)
        self.seed = seed
        self.np_random = np.random.RandomState(self.seed)

    def __len__(self) -> int:
        return len(self.buffer)

    def append(self, experience: Tuple) -> None:
        """Add experience to the buffer.
        """
        self.buffer.append(experience)

    def get_all(self):
        return self.sample(batch_size=len(self.buffer), replace=False)

    def sample(self, batch_size:int, replace:bool=True):
        nelem = len(self.buffer)
        if batch_size > nelem:
            replace = True
        indices = self.np_random.choice(nelem, size=batch_size, replace=replace)
        out = (np.array(el) for el in zip(*(self.buffer[idx] for idx in indices)))
        return out

class BanditSRL:

    def __init__(
        self,
        cfg: DictConfig,
        num_actions: int,
        model: Optional[nn.Module] = None,
        feature_dim: Optional[int] = None,
        tb_writer: Optional[SummaryWriter] = None,
        freeze_model: Optional[bool] = False
    ) -> None:
        # self.env = env
        self.cfg = cfg
        self.num_actions = num_actions
        self.model = model
        self.t = 0
        self.buffer = DataBuffer(capacity=self.cfg.buffer_capacity, seed=self.cfg.seed)
        self.explore_buffer = DataBuffer(capacity=self.cfg.buffer_capacity, seed=self.cfg.seed+1)
        self.update_time = 2
        self.np_random = np.random.RandomState(self.cfg.seed)
        self.tb_writer = tb_writer
        self.freeze_model = freeze_model
        # initialization
        if self.model is not None:
            dim = self.model.embedding_dim
            orig_dim = self.model.input_size
            self.model.to(self.cfg.device)
            self.model.to(TORCH_FLOAT)
            if not self.freeze_model:
                self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)
        else:
            dim = feature_dim
            orig_dim = feature_dim
            assert dim > 0
        self.b_vec = torch.zeros(dim, dtype=TORCH_FLOAT).to(self.cfg.device)
        self.inv_A = torch.eye(dim, dtype=TORCH_FLOAT).to(self.cfg.device) / self.cfg.ucb_regularizer
        self.A = torch.eye(dim, dtype=TORCH_FLOAT).to(self.cfg.device) * self.cfg.ucb_regularizer
        self.theta = torch.zeros(dim, dtype=TORCH_FLOAT).to(self.cfg.device)
        self.A_logdet = torch.logdet(self.A).cpu().item()
        self.feature_dim = dim
        self.features_bound = np.sqrt(orig_dim)
        self.param_bound = np.sqrt(orig_dim)
        if self.cfg.weight_rayleigh > 0:
            self.unit_vector = torch.ones(self.model.embedding_dim, dtype=TORCH_FLOAT).to(self.cfg.device) / np.sqrt(self.model.embedding_dim)
            self.unit_vector.requires_grad = True
            self.unit_vector_optimizer = torch.optim.SGD([self.unit_vector], lr=self.cfg.lr)
        # save initial model
        if self.cfg.save_model_at_train and self.model:
            path = os.path.join(self.cfg.log_path, f"model_state_dict_n{self.t}.pt")
            torch.save(self.model.state_dict(), path)

    def step_time(self):
        self.t += 1

    def play_action(self, features: np.ndarray) -> int:
        assert features.shape[0] == self.num_actions

        # forced exploration
        if self.cfg.forced_exploration_decay == "cbrt":
            self.forced_epsilon = 1. / np.cbrt(self.t + 1)
        elif self.cfg.forced_exploration_decay == "cbrt":
            self.forced_epsilon = 1. / np.sqrt(self.t + 1)
        elif self.cfg.forced_exploration_decay in ["none", "None", None, "zero"]:
            self.forced_epsilon = -1
        else:
            raise NotImplementedError()

        # check GLRT
        glrt_active, min_ratio, beta, greedy_action = self.glrt(features)
        # avoid that the GLRT triggers at the very beginning
        glrt_active = glrt_active and self.cfg.check_glrt and (len(self.explore_buffer)>self.cfg.batch_size)

        if self.cfg.use_tb:
            self.tb_writer.add_scalar('forced exploration', self.forced_epsilon, self.t)
            self.tb_writer.add_scalar('GRLT', int(glrt_active), self.t)

        if glrt_active:
            # play glrt
            self.save_exp_data = 0
            return greedy_action
        elif self.forced_epsilon > 0 and self.np_random.rand() <= self.forced_epsilon:
            # forced random exploration
            self.save_exp_data = 1
            return self.np_random.choice(self.action_space.n, size=1).item()
        else:
            self.save_exp_data = 1
            return self.play_base_action(features=features)

    def glrt(self, features: np.ndarray):
        features_tensor = torch.tensor(features, dtype=TORCH_FLOAT, device=self.cfg.device)
        if self.model is not None:
            with torch.no_grad():
                phi = self.model.embedding(features_tensor)
                dim = self.model.embedding_dim
        else:
            phi = features_tensor
            dim = self.feature_dim
        predicted_rewards = torch.matmul(phi, self.theta)
        opt_arms = torch.where(predicted_rewards > predicted_rewards.max() - self.cfg.mingap_clip)[0]
        subopt_arms = torch.where(predicted_rewards <= predicted_rewards.max() - self.cfg.mingap_clip)[0]
        action = self.np_random.choice(opt_arms.cpu().detach().numpy().flatten()).item()
        # Generalized Likelihood Ratio Test
        val = - 2 * np.log(self.cfg.delta) + dim * np.log(
            1 + 2 * self.t * self.features_bound / (self.cfg.ucb_regularizer * dim))
        # val = self.A_logdet - dim * np.log(self.ucb_regularizer) - 2 * np.log(self.delta)
        beta = self.cfg.noise_std * np.sqrt(val) + self.param_bound * np.sqrt(self.cfg.ucb_regularizer)
        if len(subopt_arms) == 0:
            min_ratio = beta ** 2 + 1
        else:
            prediction_diff = predicted_rewards[subopt_arms] - predicted_rewards[action]
            phi_diff = phi[subopt_arms] - phi[action]
            weighted_norm = (torch.matmul(phi_diff, self.inv_A) * phi_diff).sum(axis=1)
            likelihood_ratio = (prediction_diff) ** 2 / (2 * weighted_norm.clamp_min(1e-10))
            min_ratio = likelihood_ratio.min().cpu().detach().numpy()
        is_active = bool(min_ratio > self.cfg.glrt_scale * beta**2)
        return is_active, min_ratio, beta, action


    def update(self, features: np.ndarray, action: int, reward: float) -> None:
        features_selected_act = features[action]
        self.buffer.append(experience=(features_selected_act, reward))
        if self.save_exp_data:
            self.explore_buffer.append(experience=(features_selected_act, reward))
        
        ####################################
        # Update prediction on the embedding
        # - compute V and b => theta
        ####################################
        feature_tensor = torch.tensor(features_selected_act.reshape(1,-1), dtype=TORCH_FLOAT).to(self.cfg.device)
        if self.model is not None:
            with torch.no_grad():
                phi = self.model.embedding(feature_tensor).squeeze()
        else:
            phi = feature_tensor.squeeze()
        self.A += torch.outer(phi, phi)
        self.b_vec += phi * reward
        self.theta = torch.linalg.solve(self.A, self.b_vec)
        self.inv_A = torch.linalg.inv(self.A)
        self.A_logdet = torch.logdet(self.A).cpu().item()
        self.features_bound = max(self.features_bound, torch.norm(phi, p=2).cpu().item())
        self.param_bound = torch.linalg.norm(self.theta, 2).cpu().item()


        ####################################
        # Retrain representation
        ####################################
        if self.t == self.update_time:
            if self.cfg.update_every > 5:
                self.update_time += self.cfg.update_every
            else:
                self.update_time = int(np.ceil(max(1, self.update_time) * self.cfg.update_every))
            
            if self.t > self.cfg.batch_size:
                return self._train()
        return None

    def _train(self) -> float:
        if self.model is None or self.freeze_model:
            return 0
        
        exp_features, exp_rewards = self.explore_buffer.get_all()
        features, rewards = self.buffer.get_all()
        # if we have less exploratory samples than features, we duplicate them
        idxs = self.np_random.choice(exp_features.shape[0], size=features.shape[0])
        torch_dataset = torch.utils.data.TensorDataset(
            torch.tensor(exp_features[idxs], dtype=TORCH_FLOAT, device=self.cfg.device),
            torch.tensor(exp_rewards[idxs].reshape(-1, 1), dtype=TORCH_FLOAT, device=self.cfg.device),
            torch.tensor(features, dtype=TORCH_FLOAT, device=self.cfg.device),
            torch.tensor(rewards.reshape(-1, 1), dtype=TORCH_FLOAT, device=self.cfg.device)
        )
        loader = torch.utils.data.DataLoader(dataset=torch_dataset, batch_size=self.cfg.batch_size, shuffle=False)
        self.model.train()
        for epoch in range(self.cfg.max_updates):
            epoch_metrics = defaultdict(list)

            for batch_exp_feat, batch_exp_rew, batch_features, batch_rewards in loader:
                metrics = self._train_loss(batch_exp_feat, batch_exp_rew, batch_features, batch_rewards)
                # update epoch metrics
                for key, value in metrics.items():
                    epoch_metrics[key].append(value)
                # self.writer.flush()
        self.model.eval()
        self._post_train_reset()
        # log to tensorboard
        if self.cfg.use_tb:
            for key, value in epoch_metrics.items():
                self.tb_writer.add_scalar('epoch_' + key, np.mean(value), self.t)
        avg_loss = np.mean(epoch_metrics['train_loss'])

        # debug metric
        aux_metrics = {}
        aux_metrics["train_loss"] = avg_loss
        return aux_metrics

    # functions to be implemented
    def _train_loss(
        self, exp_features_tensor, exp_rewards_tensor, features_tensor, rewards_tensor
    ) -> Dict:
        loss = 0
        metrics = {}
        # MSE LOSS
        if not np.isclose(self.cfg.weight_mse,0):
            prediction = self.model(exp_features_tensor)
            mse_loss = F.mse_loss(prediction, exp_rewards_tensor)
            metrics['mse_loss'] = mse_loss.cpu().item()
            loss = loss + self.cfg.weight_mse * mse_loss
        
        # Rayleigh loss
        if not np.isclose(self.cfg.weight_rayleigh, 0):
            phi = self.model.embedding(features_tensor)
            if self.cfg.normalize_features:
                # norm = torch.norm(phi, dim=1, keepdim=False).max() if self.cfg.use_maxnorm else torch.norm(phi, dim=1, keepdim=True)
                # phi = phi / norm
                if self.cfg.use_maxnorm:
                    norm = torch.norm(phi, dim=1, keepdim=False).max()
                    phi = phi / norm
                else:
                    phi = F.normalize(phi, dim=1)
            A = torch.matmul(phi.T, phi) + self.cfg.ucb_regularizer * torch.eye(phi.shape[1], device=self.cfg.device)
            A /= phi.shape[0]
            # compute loss to update the unit vector
            unit_vector_loss = torch.dot(self.unit_vector, torch.matmul(A.detach(), self.unit_vector))
            self.unit_vector_optimizer.zero_grad()
            unit_vector_loss.backward()
            self.unit_vector_optimizer.step()
            self.unit_vector.data = F.normalize(self.unit_vector.data, dim=0)
            # recompute the loss to update embedding
            phi = self.model.embedding(features_tensor)
            if self.cfg.normalize_features:
                if self.cfg.use_maxnorm:
                    norm = torch.norm(phi, dim=1, keepdim=False).max()
                    phi = phi / norm
                else:
                    phi = F.normalize(phi, dim=1)
            A = torch.matmul(phi.T, phi) + self.cfg.ucb_regularizer * torch.eye(phi.shape[1], device=self.cfg.device)
            A /= phi.shape[0]
            rayleigh_loss = - torch.dot(self.unit_vector.detach(), torch.matmul(A, self.unit_vector.detach()))
            metrics['rayleigh_loss'] = rayleigh_loss.cpu().item()
            loss += self.cfg.weight_rayleigh * rayleigh_loss

        # weak loss
        if not np.isclose(self.cfg.weight_weak, 0):
            phi = self.model.embedding(features_tensor)
            if self.cfg.normalize_features:
                if self.cfg.use_maxnorm:
                    norm = torch.norm(phi, dim=1, keepdim=False).max()
                    phi = phi / norm
                else:
                    phi = F.normalize(phi, dim=1)
            A = torch.matmul(phi.T, phi) + self.cfg.ucb_regularizer * torch.eye(phi.shape[1], device=self.cfg.device)
            A /= phi.shape[0]
            with torch.no_grad():
                all_phi = self.model.embedding(exp_features_tensor)
                # all_phi = all_phi / torch.norm(all_phi, dim=1, keepdim=True)
                all_phi = F.normalize(all_phi, dim=1)
            sum_norms = (torch.matmul(all_phi.detach(), A) * all_phi.detach()).sum(axis=1)
            min_feat_loss = - sum_norms.min() #.mean()
            loss += self.cfg.weight_weak * min_feat_loss
            metrics['weak_loss'] = min_feat_loss.cpu().item()
        
        # perform an SGD step
        self.optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 100)
        self.optimizer.step()
        metrics['train_loss'] = loss.cpu().item()
        return metrics

    def _post_train_reset(self) -> None:
        if self.model is None or self.freeze_model:
            return None
        dim = self.model.embedding_dim
        batch_features, batch_rewards = self.buffer.get_all()
        features_tensor = torch.tensor(batch_features, dtype=TORCH_FLOAT, device=self.cfg.device)
        rewards_tensor = torch.tensor(batch_rewards, dtype=TORCH_FLOAT, device=self.cfg.device)
        with torch.no_grad():
            phi = self.model.embedding(features_tensor)
        A = torch.matmul(phi.T, phi) + self.cfg.ucb_regularizer * torch.eye(dim, device=self.cfg.device)
        b_vec = torch.matmul(phi.T, rewards_tensor)
        theta = torch.linalg.solve(A, b_vec)
        assert torch.allclose(torch.matmul(A, theta), b_vec), (A, theta, b_vec)
        self.theta = theta
        self.A = A
        self.b_vec = b_vec
        self.inv_A = torch.linalg.inv(self.A)
        self.A_logdet = torch.logdet(self.A).cpu().item()

        self.features_bound = torch.norm(phi, p=2, dim=1).max().cpu().item()
        self.param_bound = torch.linalg.norm(self.theta, 2).item()
        if self.cfg.use_tb:
            self.tb_writer.add_scalar('features_bound', self.features_bound, self.t)
            self.tb_writer.add_scalar('param_bound', self.param_bound, self.t)

        # prediction = torch.matmul(phi, self.theta)
        # mse_loss = (prediction - rewards_tensor).pow(2).mean()
        # self.writer.add_scalar('mse_linear', mse_loss.item(), self.t)

        if self.cfg.save_model_at_train and self.model:
            path = os.path.join(self.cfg.log_path, f"model_state_dict_n{self.t}.pt")
            torch.save(self.model.state_dict(), path)

    def play_base_action(self, features: np.ndarray) -> int:
        pass


class BSRLLinUcb(BanditSRL):

    def play_base_action(self, features: np.ndarray) -> int:
        features_tensor = torch.tensor(features, dtype=TORCH_FLOAT, device=self.cfg.device)
        if self.model is not None:
            dim = self.model.embedding_dim
            with torch.no_grad():
                phi = self.model.embedding(features_tensor)
        else:
            dim = self.feature_dim
            phi = features_tensor
        
        if self.cfg.adaptive_bonus_linucb == False:
            val = - 2 * np.log(self.cfg.delta) + dim * np.log(1 + 2 * self.t * self.features_bound / (self.cfg.ucb_regularizer * dim))
        else:
            val = self.A_logdet - dim * np.log(self.cfg.ucb_regularizer) - 2 * np.log(self.cfg.delta)
        beta = self.cfg.noise_std * np.sqrt(val) + self.param_bound * np.sqrt(self.cfg.ucb_regularizer)
        # print("Alog:", self.A_logdet)
        # print("beta:", beta)
        bonus = (torch.matmul(phi, self.inv_A) * phi).sum(axis=1)
        bonus = self.cfg.bonus_scale * beta * torch.sqrt(bonus)
        # print("bonus: ", bonus)
        ucb = torch.matmul(phi, self.theta) + bonus
        # print("ucb:", ucb)
        action = torch.argmax(ucb).item()
        assert 0 <= action < self.num_actions, ucb
        return action


class BSRLEpsGreedy(BanditSRL):

    def play_base_action(self, features: np.ndarray) -> int:
        if self.cfg.epsilon_decay == "cbrt":
            self.epsilon = 1. / np.cbrt(self.t + 1)
        elif self.cfg.epsilon_decay == "sqrt":
            self.epsilon = 1. / np.sqrt(self.t + 1)
        elif self.cfg.epsilon_decay == "zero":
            self.epsilon = 0
        else:
            raise NotImplementedError()

        if self.cfg.use_tb:
            self.tb_writer.add_scalar('eps-greedy exploration', self.epsilon, self.t)
        
        if self.np_random.rand() <= self.epsilon:
            return self.np_random.choice(self.num_actions, size=1).item()
        else:
            features_tensor = torch.tensor(features, dtype=TORCH_FLOAT, device=self.cfg.device)
            if self.model is not None:
                with torch.no_grad():
                    phi = self.model.embedding(features_tensor)
            else:
                phi = features_tensor
            predicted_rewards = torch.matmul(phi, self.theta)
            opt_arms = torch.where(predicted_rewards > predicted_rewards.max() - self.cfg.mingap_clip)[0]
            action = self.np_random.choice(opt_arms.cpu().detach().numpy().flatten()).item()
        return action


class BSRLThompsonSampling(BanditSRL):

    def play_base_action(self, features: np.ndarray) -> int:
        features_tensor = torch.tensor(features, dtype=TORCH_FLOAT, device=self.cfg.device)
        if self.model is not None:
            dim = self.model.embedding_dim
            with torch.no_grad():
                phi = self.model.embedding(features_tensor)
        else:
            dim = self.feature_dim
            phi = features_tensor

        if self.cfg.adaptive_bonus_linucb == False:
            val = - 2 * np.log(self.cfg.delta) + dim * np.log(1 + 2 * self.t * self.features_bound / (self.cfg.ucb_regularizer * dim))
        else:
            val = self.A_logdet - dim * np.log(self.cfg.ucb_regularizer) - 2 * np.log(self.cfg.delta)
        beta = self.cfg.noise_std * np.sqrt(val) + self.param_bound * np.sqrt(self.cfg.ucb_regularizer)

        mg = torch.distributions.multivariate_normal.MultivariateNormal(self.theta, covariance_matrix=(self.cfg.bonus_scale * beta)**2 * self.inv_A)
        theta_ts = mg.sample()
        predicted_rewards = torch.matmul(phi, theta_ts)
        opt_arms = torch.where(predicted_rewards > predicted_rewards.max() - self.cfg.mingap_clip)[0]
        action = self.np_random.choice(opt_arms.cpu().detach().numpy().flatten()).item()
        return action


class BSRLIGW(BanditSRL):

    def play_base_action(self, features: np.ndarray) -> int:
        if self.cfg.epsilon_decay == "cbrt":
            gamma = self.cfg.gamma_scale * np.cbrt(self.t + 1)
        elif self.cfg.epsilon_decay == "sqrt":
            gamma = self.cfg.gamma_scale * np.sqrt(self.t + 1)
        features_tensor = torch.tensor(features, dtype=TORCH_FLOAT, device=self.cfg.device)
        if self.model is not None:
            dim = self.model.embedding_dim
            with torch.no_grad():
                phi = self.model.embedding(features_tensor)
        else:
            dim = self.feature_dim
            phi = features_tensor
        predicted_rewards = torch.matmul(phi, self.theta).cpu().numpy()
        gap = predicted_rewards.max() - predicted_rewards
        opt_arms = np.where(gap <= self.cfg.mingap_clip)[0]
        subopt_arms = np.where(gap > self.cfg.mingap_clip)[0]
        prob = np.zeros(self.num_actions)
        prob[subopt_arms] = 1. / (self.num_actions + gamma * gap[subopt_arms])
        prob[opt_arms] = (1 - prob[subopt_arms].sum()) / len(opt_arms)
        assert np.isclose(prob.sum(), 1)

        if self.cfg.use_tb:
            self.tb_writer.add_scalar('prob_optarms', 1 - prob[subopt_arms].sum(), self.t)
            self.tb_writer.add_scalar('gamma', gamma, self.t)
        action = self.np_random.choice(self.num_actions, 1, p=prob).item()
        return action
