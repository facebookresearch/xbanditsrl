# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
from typing import Any
from collections import defaultdict, deque
from tqdm import tqdm
import time
import os
import pickle
import torch.nn as nn
from typing import Tuple, List
import math
from banditsrl import DataBuffer
import torch
from torch.utils.tensorboard import SummaryWriter
import pdb


def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Linear):
            n = m.in_features + m.out_features
            m.weight.data.normal_(0, math.sqrt(4. / n))
            if m.bias is not None:
                m.bias.data.zero_()


class LinearEmbNet(nn.Module):

    def __init__(self, input_size: int, layers_data: List[Tuple]):
        super().__init__()
        self.layers = nn.ModuleList()
        self.input_size = input_size 
        if layers_data:
            for size, activation in layers_data:
                self.layers.append(nn.Linear(input_size, size))
                input_size = size
                if activation is not None:
                    assert isinstance(activation, nn.Module)
                    self.layers.append(activation)
            self.embedding_dim = layers_data[-1][0]
        else:
            self.embedding_dim = input_size
            self.layers = None
        self.fc2 = nn.Linear(self.embedding_dim, 1, bias=False)
        initialize_weights(self)

    def embedding(self, x):
        if self.layers:
            for layer in self.layers:
                x = layer(x)
        return x

    def forward(self, x):
        x = self.embedding(x)
        return self.fc2(x)

class LinearNetwork(LinearEmbNet):

    def __init__(self, input_size: int):
        super().__init__(input_size, None)

class RFFNet(nn.Module):

    def __init__(self, input_size: int, embedding_dim: int):
        super().__init__()
        self.input_size = input_size
        self.embedding_dim = embedding_dim
        self.fc1 = nn.Linear(self.input_size, self.embedding_dim, bias=True)
        self.fc2 = nn.Linear(self.embedding_dim, 1, bias=False)

    def embedding(self, x):
        x = self.fc1(x)
        x = torch.cos(x)
        return x

    def forward(self, x):
        x = self.embedding(x)
        return self.fc2(x)

class Runner:

    def __init__(
        self,
        env: Any,
        algo: Any,
        horizon: int
    ) -> None:
        self.env = env
        self.algo = algo
        self.horizon = horizon

    def run(
        self, throttle: int=100, log_path: str=None, 
        writer: SummaryWriter=None, save_tmp_every: int=50000,
        save_history: bool=False, log_every_t: int=1
    ) -> None:
        save_tmp_every = int(save_tmp_every)
        metrics = defaultdict(list)
        metrics['regret'] = [0]
        metrics['expected_regret'] = [0]
        metrics["optimal_arm"] = [0]
        metrics["time"] = [0]
        regret, exp_regret = 0, 0
        sum_pull_optimal_arm = 0
        self.t = 0
        if save_history:
            history = DataBuffer(capacity=self.horizon)

        tqdm_print = {
            '% opt arm (last 100 steps)': 0.0,
            'train loss': 0.0,
            'expected regret': 0.0
        }
        is_opt_arm_limcap = deque(maxlen=100)
        with tqdm(initial=self.t, total=self.horizon, postfix=tqdm_print) as pbar:
            while self.t < self.horizon:
                # pdb.set_trace()
                start = time.time()
                # get features ( \phi(x,a) )_{a \in A_t}, i.e. matrix (num_actions x dim)
                features = self.env.sample()
                # select action
                action = self.algo.play_action(features=features)
                # observe reward
                reward = self.env.reward(action)
                # update model
                aux_metrics = self.algo.update(features, action, reward)
                stopt = time.time()

                # print()
                # print(f"time: {self.t}")
                # print(f"feat: {features}")
                # print(f"action: {action}")
                # print(f"rew: {reward}")

                if save_history:
                    history.append((features[action], action, reward))

                ##############################################################
                # LOGGING
                # update metrics
                ##############################################################
                if aux_metrics:
                    for key, value in aux_metrics.items():
                        metrics[key].append(value)
                expected_reward = self.env.expected_reward(action)
                best_reward = self.env.best_reward()
                exp_regret += best_reward - expected_reward
                regret += best_reward - reward
                tmp_is_opt = np.abs(expected_reward - best_reward)<1e-6
                is_opt_arm_limcap.append(tmp_is_opt)
                sum_pull_optimal_arm += tmp_is_opt
                p_optimal_arm = np.mean(is_opt_arm_limcap)
                if (self.t+1) % log_every_t == 0:
                    metrics['runtime'].append(stopt - start)
                    metrics['expected_reward'].append(expected_reward)
                    metrics['best_reward'].append(best_reward)
                    metrics['regret'].append(regret)
                    metrics["expected_regret"].append(exp_regret)
                    metrics["optimal_arm"].append(p_optimal_arm/(self.t+1))
                    metrics["time"].append(self.t+1)

                # update tqdm_print
                tqdm_print['expected regret'] = exp_regret
                tqdm_print['% opt arm (last 100 steps)'] = '{:.2%}'.format(p_optimal_arm)
                if aux_metrics:
                    if "train_loss" in aux_metrics.keys():
                        tqdm_print['train loss'] = aux_metrics["train_loss"]
                if self.t % throttle == 0:
                    pbar.set_postfix(tqdm_print)
                    pbar.update(throttle)

                ##############################################################
                # Visualization
                ##############################################################
                if writer and ((self.t+1) % log_every_t == 0):
                    writer.add_scalar("expected regret", tqdm_print['expected regret'], self.t)
                    writer.add_scalar('perc optimal pulls (last 100 steps)', p_optimal_arm, self.t)
                
                ##############################################################
                # step (and tmp logging)
                ##############################################################
                self.t += 1
                self.algo.step_time()
                if self.t % save_tmp_every == 0:
                    with open(os.path.join(log_path, "latest_result.pkl"), 'wb') as f:
                        pickle.dump(metrics, f, protocol=pickle.HIGHEST_PROTOCOL)
                    if save_history:
                        hfeatures, hactions, hrewards = (np.array(el) for el in zip(*(history.buffer[idx] for idx in range(len(history)))))
                        payload = {'features': hfeatures, 'actions': hactions, 'rewards': hrewards}
                        with open(os.path.join(log_path, "history.pt"), 'wb') as f:
                            torch.save(payload, f)

        # convert metrics to numpy.array
        for key, value in metrics.items():
            metrics[key] = np.array(value)
        if save_history:
            hfeatures, hactions, hrewards = (np.array(el) for el in zip(*(history.buffer[idx] for idx in range(len(history)))))
            payload = {'features': hfeatures, 'actions': hactions, 'rewards': hrewards}
            with open(os.path.join(log_path, "history.pt"), 'wb') as f:
                torch.save(payload, f)
        return metrics
