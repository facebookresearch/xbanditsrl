# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.utils import get_original_cwd
import os
from pathlib import Path
import logging
import torch
import torch.nn as nn
import random
import banditsrl
import baselines as xb_baselines
import envs as xb_envs
import core as xb_core
from core import LinearEmbNet, initialize_weights, RFFNet
import json
import pickle
import time
from datetime import timedelta
from torch.utils.tensorboard import SummaryWriter

# A logger for this file
log = logging.getLogger(__name__)

def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

@hydra.main(config_path="configs/xbrl", config_name="config")
def my_app(cfg: DictConfig) -> None:
    try:
        work_dir = Path.cwd()
        original_dir = get_original_cwd()
        log.info(f"Current working directory : {work_dir}")
        log.info(f"Orig working directory    : {original_dir}")
        set_seed_everywhere(cfg.seed)
        device = torch.device(cfg.device)
        cfg.log_path = str(work_dir)

        if cfg.domain.type == "c2b":
            env = xb_envs.make_c2b(
                dataset_name=cfg.domain.dataset_name,
                bandit_model=cfg.domain.bandit_model,
                rew_optimal_labels=cfg.domain.rew_optimal_labels,
                rew_other_labels=cfg.domain.rew_other_labels,
                seed=cfg.seed,
                noise=cfg.domain.noise,
                noise_param=cfg.domain.noise_param, shuffle=cfg.domain.shuffle
            )
            print()
            print("="*20)
            print(env.description())
            mingap = env.min_suboptimality_gap()
            print(f"min gap: {mingap}")
            print("="*20,"\n")
            log.info("="*20)
            log.info(env.description())
            log.info(f"min gap: {mingap}")
            log.info("="*20)
        elif cfg.domain.type == "wheel":
            ncontexts, narms, dim = cfg.domain.ncontexts, cfg.domain.narms, cfg.domain.dim
            features = np.zeros((ncontexts, narms, dim))
            rewards = np.zeros((ncontexts, narms))
            rewards[:, 0] = cfg.domain.mu_1
            rewards[:, 1:5] = cfg.domain.mu_2
            rand_local = np.random.RandomState(cfg.domain.seed_problem)
            # import matplotlib.pyplot as plt
            # fig, ax = plt.subplots()
            # plt.plot([-1,-0.5],[0,0], color="black", alpha=0.5)
            # plt.plot([0.5,1],[0,0], color="black", alpha=0.5)
            # plt.plot([0,0],[-1,-0.5], color="black", alpha=0.5)
            # plt.plot([0,0],[0.5,1], color="black", alpha=0.5)
            # c1 = plt.Circle((0, 0), 1, color='black', fill=False, alpha=0.5)
            # c2 = plt.Circle((0, 0), 0.5, color='black', fill=False, alpha=0.5)
            # ax.add_patch(c1)
            # ax.add_patch(c2)
            for i in range(ncontexts):
                rho = rand_local.rand()
                theta = rand_local.rand() * 2 * np.pi
                x = np.array([np.cos(theta), np.sin(theta)]) * rho
                for j in range(narms):
                    y = np.zeros(narms)
                    y[j] = 1
                    features[i,j,:] = np.concatenate([x,y])
                if np.linalg.norm(x) > cfg.domain.radius:
                    if x[0] >= 0 and x[1] >= 0:
                        rewards[i, 1] = cfg.domain.mu_3
                        # plt.scatter(x[0],x[1],c="red",s=10,alpha=0.5)
                    elif x[0] >= 0 and x[1] < 0:
                        rewards[i, 2] = cfg.domain.mu_3
                        # plt.scatter(x[0],x[1],c="green",s=10,alpha=0.5)
                    elif x[0] < 0 and x[1] >= 0:
                        rewards[i, 3] = cfg.domain.mu_3
                        # plt.scatter(x[0],x[1],c="orange",s=10,alpha=0.5)
                    else:
                        rewards[i, 4] = cfg.domain.mu_3
                        # plt.scatter(x[0],x[1],c="purple",s=10,alpha=0.5)
                else:
                    pass
                    # plt.scatter(x[0],x[1],c="blue",s=10,alpha=0.5)
            # plt.show()
            env = xb_envs.XBFinite(
                feature_matrix=features, 
                reward_matrix=rewards, seed=cfg.seed, 
                noise=cfg.domain.noise_type, noise_param=cfg.domain.noise_param
            )
            print(f"min gap: {env.min_suboptimality_gap()}")
        else:
            raise ValueError(f"Unknown domain type: {cfg.domain.type}")

        if not cfg.algo in ["linucb", "egreedy", "rfflinucb", "rffegreedy"]:
            if cfg.layers not in [None, "none", "None"]:
                hid_dim = cfg.layers
                if isinstance(cfg.layers, str):
                    hid_dim = cfg.layers.split(",")
                    hid_dim = [int(el) for el in hid_dim]
                if not isinstance(hid_dim, list):
                    hid_dim = [hid_dim]
                layers = [(el, nn.ReLU() if cfg.use_relu else nn.Tanh()) for el in hid_dim]
            else:
                layers = None # linear in the features
            net = LinearEmbNet(env.feature_dim, layers).to(device)
            print("Network:\n" + net.__str__() + "\n")
            log.info("Network:\n" + net.__str__() + "\n")
            initialize_weights(net)
        elif cfg.algo.startswith("rff"):
            net = RFFNet(env.feature_dim, cfg.layers).to(device)
            print("Network:\n" + net.__str__() + "\n")
            log.info("Network:\n" + net.__str__() + "\n")
            initialize_weights(net)
            
        
        tbwriter = None
        if cfg.use_tb:
            tbwriter = SummaryWriter(work_dir)

        if cfg.algo == "bsrllinucb":
            algo = banditsrl.BSRLLinUcb(cfg=cfg, num_actions=env.num_actions, model=net, feature_dim=None, tb_writer=tbwriter)
        elif cfg.algo == "bsrlegreedy":
            algo = banditsrl.BSRLEpsGreedy(cfg=cfg, num_actions=env.num_actions, model=net, feature_dim=None, tb_writer=tbwriter)
        elif cfg.algo == "bsrlts":
            algo = banditsrl.BSRLThompsonSampling(cfg=cfg, num_actions=env.num_actions, model=net, feature_dim=None, tb_writer=tbwriter)
        elif cfg.algo == "bsrligw":
            algo = banditsrl.BSRLIGW(cfg=cfg, num_actions=env.num_actions, model=net, feature_dim=None, tb_writer=tbwriter)
        elif cfg.algo == "igwexp":
            algo = xb_baselines.IGWExplorer(cfg=cfg, num_actions=env.num_actions, model=net, feature_dim=None, tb_writer=tbwriter)
        elif cfg.algo == "linucb":
            # no neural network
            algo = banditsrl.BSRLLinUcb(cfg=cfg, num_actions=env.num_actions, model=None, feature_dim=env.feature_dim, tb_writer=tbwriter)
        elif cfg.algo == "egreedy":
            # no neural network
            algo = banditsrl.BSRLEpsGreedy(cfg=cfg, num_actions=env.num_actions, model=None, feature_dim=env.feature_dim, tb_writer=tbwriter)
        elif cfg.algo == "gradientucb":
            algo = xb_baselines.GradientUCB(cfg=cfg, num_actions=env.num_actions, model=net, feature_dim=None, tb_writer=tbwriter)
        elif cfg.algo == "rfflinucb":
            algo = banditsrl.BSRLLinUcb(cfg=cfg, num_actions=env.num_actions, model=net, feature_dim=None, tb_writer=tbwriter, freeze_model=True)
        elif cfg.algo == "rffegreedy":
            algo = banditsrl.BSRLEpsGreedy(cfg=cfg, num_actions=env.num_actions, model=net, feature_dim=None, tb_writer=tbwriter, freeze_model=True)
        else:
            raise ValueError("Unknown algorithm")
        
        print(f"Algorithm: {cfg.algo} (wmse: {cfg.weight_mse}, wweak: {cfg.weight_weak}, wray: {cfg.weight_rayleigh})\n")
        log.info(f"Algorithm: {cfg.algo} (wmse: {cfg.weight_mse}, wweak: {cfg.weight_weak}, wray: {cfg.weight_rayleigh})")
        
        with open(os.path.join(work_dir, "config.json"), 'w') as f:
            json.dump(OmegaConf.to_container(cfg), f, indent=4, sort_keys=True)
        log.info("Configuration has been saved")

        runner = xb_core.Runner(env=env, algo=algo, horizon=cfg.horizon)
        log.info(f"Running: {type(algo).__name__}")
        start_running = time.time()

        result = runner.run(
            log_path=work_dir, writer=tbwriter, 
            save_tmp_every=max(1,int(cfg.horizon/10)), save_history=cfg.save_history,
            log_every_t=cfg.log_every_t
        )
        finished_in = time.time() - start_running
        elapsed_time = str(timedelta(seconds=finished_in))
        print(f"\nFinished in {elapsed_time}")
        log.info(f"Finished in {elapsed_time}")
        result["execution_time"] = finished_in

        with open(os.path.join(work_dir, "result.pkl"), 'wb') as f:
            pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as exc:
        log.error("="*20)
        log.error("EXCEPTION")
        log.error("-"*20)
        log.exception(exc)
        log.error("="*20)
        raise exc

if __name__ == "__main__":
    my_app()
