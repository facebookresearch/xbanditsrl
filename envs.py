# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
from typing import Optional, Any
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler, LabelEncoder
from sklearn.datasets import fetch_openml
import sklearn

class XBFinite:
    N_NONE=0
    N_GAUSSIAN=1
    N_BERNOULLI=2

    def __init__(
        self,
        feature_matrix: np.ndarray,
        reward_matrix: np.ndarray,
        noise: Optional[str]=None,
        noise_param: Optional[Any]=None,
        seed: Optional[int]=0
    ) -> None:
        assert noise in [None, "bernoulli", "gaussian", "none", "None"]
        self.feature_matrix = feature_matrix
        self.reward_matrix = reward_matrix
        self.noise = self.N_NONE
        if noise == "bernoulli":
            self.noise = self.N_BERNOULLI
        elif noise == "gaussian":
            self.noise = self.N_GAUSSIAN
        self.noise_param = noise_param
        self.seed = seed
        self.np_random = np.random.RandomState(seed)
        assert len(self.feature_matrix.shape) == 3
        assert (self.feature_matrix.shape[0] == self.reward_matrix.shape[0]) and (self.feature_matrix.shape[1] == self.reward_matrix.shape[1])
        self.feature_dim = self.feature_matrix.shape[-1]
        self.num_actions = self.feature_matrix.shape[1]
        self.idx = 0

    def sample(self) -> np.ndarray:
        self.idx = self.np_random.choice(self.feature_matrix.shape[0], 1).item()
        return self.feature_matrix[self.idx]
    
    def reward(self, action) -> float:
        reward = self.reward_matrix[self.idx, action]
        if self.noise != self.N_NONE:
            if self.noise == self.N_BERNOULLI:
                assert 0<=reward<=1
                reward = self.np_random.binomial(n=1, p=reward)
            else:
                reward = reward + self.np_random.randn() * self.noise_param  
        return reward

    def best_reward(self) -> float:
        return np.max(self.reward_matrix[self.idx])

    def expected_reward(self, action) -> float:
        return self.reward_matrix[self.idx, action]
    
    def min_suboptimality_gap(self, tol=1e-6):
        min_gap = np.inf
        for ctx in range(self.reward_matrix.shape[0]):
            arr = sorted(self.reward_matrix[ctx])
            for i in range(self.num_actions-1):
                diff = arr[self.num_actions-1] - arr[i]
                if diff < min_gap - tol and diff > tol:
                    min_gap = diff
        return min_gap

class OneHotC2B:
    """ One-Hot Classification to Bandit Environment
    """
    N_NONE=0
    N_GAUSSIAN=1
    N_BERNOULLI=2

    def __init__(
        self,
        X: np.ndarray,
        labels: np.ndarray,
        rew_optimal_labels: Optional[float] = 1,
        rew_other_labels: Optional[float] = 0,
        seed: Optional[int] = 0,
        noise: Optional[str] = None,
        noise_param: Optional[float] = None,
        dataset_name: Optional[str] = "",
        shuffle: Optional[bool] = True
    ) -> None:
        assert noise in [None, "bernoulli", "gaussian", "none", "None"]
        self.X = X #num_context x dim
        # classes are converted into {0, ..., n_classes - 1} values
        # these are the optimal action for each context
        self.labels = OrdinalEncoder(dtype=int).fit_transform(labels.reshape((-1, 1)))
        self.rew_optimal_labels = rew_optimal_labels
        self.rew_other_labels = rew_other_labels
        self.seed = seed
        self.noise = self.N_NONE
        if noise == "bernoulli":
            self.noise = self.N_BERNOULLI
        elif noise == "gaussian":
            self.noise = self.N_GAUSSIAN

        if shuffle:
            self.X, self.labels = sklearn.utils.shuffle(self.X, self.labels, random_state=self.seed)
        
        self.noise_param = noise_param
        self.dataset_name = dataset_name
        self.np_random = np.random.RandomState(seed=self.seed)
        self.num_actions = np.unique(self.labels).shape[0]
        self.idx = 0
        self.__post_init__()

    def __post_init__(self) -> None:
        self.feature_dim = self.X.shape[1] + self.num_actions
        self.eye = np.eye(self.num_actions)

    def _generate_feat_and_rewards(self, idx) -> np.ndarray:
        context = self.X[idx]
        na = self.num_actions 
        tile_p = [na] + [1]*len(context.shape)
        x = np.tile(context, tile_p)
        x_y = np.hstack((x, self.eye))
        rwd = np.ones((na,)) * self.rew_other_labels
        rwd[self.labels[idx]] = self.rew_optimal_labels
        assert x_y.shape == (self.num_actions, self.feature_dim)
        return x_y, rwd
    
    def sample(self) -> np.ndarray:
        self.idx = self.np_random.choice(self.X.shape[0], 1).item()
        # print("idx: ", self.idx)
        xy, _ = self._generate_feat_and_rewards(self.idx)
        return xy

    def reward(self, action: int) -> float:
        assert 0 <= action <= self.num_actions - 1
        reward = self.rew_optimal_labels if self.labels[self.idx] == action else self.rew_other_labels
        if self.noise != self.N_NONE:
            if self.noise == self.N_BERNOULLI:
                reward = self.np_random.binomial(n=1, p=reward)
            else:
                reward = reward + self.np_random.rand() * self.noise_param  
        return reward
    
    def best_reward(self) -> float:
        return self.rew_optimal_labels

    def expected_reward(self, action) -> float:
        reward = self.rew_optimal_labels if self.labels[self.idx] == action else self.rew_other_labels
        return reward
    
    def min_suboptimality_gap(self, tol=1e-6):
        return self.rew_optimal_labels - self.rew_other_labels

    def description(self) -> str:
        desc = f"{self.dataset_name}\n"
        desc += f"n_contexts: {self.X.shape[0]}\n"
        desc += f"context_dim: {self.X.shape[1]}\n"
        desc += f"n_actions: {self.num_actions}\n"
        desc += f"rewards[subopt, optimal]: [{self.rew_other_labels}, {self.rew_optimal_labels}]\n"
        desc += f"noise type: {self.noise} (noise param: {self.noise_param})\n"
        desc += f"seed: {self.seed}\n"
        desc += f"feature dimension: {self.feature_dim}"
        return desc


class ExpandedC2B(OneHotC2B):

    def __post_init__(self) -> None:
        self.feature_dim = self.X.shape[1] * self.num_actions

    def _generate_feat_and_rewards(self, idx) -> np.ndarray:
        context = self.X[idx]
        na = self.num_actions
        act_dim = self.X.shape[1]
        Ft = np.zeros((na, self.feature_dim))
        for a in range(na):
            Ft[a, a * act_dim:a * act_dim + act_dim] = context
        rwd = np.ones((na,)) * self.rew_other_labels
        rwd[self.labels[idx]] = self.rew_optimal_labels
        assert Ft.shape == (self.num_actions, self.feature_dim)
        return Ft, rwd


def make_c2b(
    dataset_name: str,
    bandit_model: str="expanded",
    rew_optimal_labels: float=1,
    rew_other_labels: float=0,
    seed: int=0,
    noise: str=None,
    noise_param:float=None,
    shuffle:bool=True
):
    assert bandit_model in ["expanded", "onehot"]
    assert noise in ["gaussian", "bernoulli", None, "None", "none"]
    # Fetch data
    if dataset_name in ['adult_num', 'adult_onehot']:
        X, y = fetch_openml('adult', version=1, return_X_y=True)
        is_NaN = X.isna()
        row_has_NaN = is_NaN.any(axis=1)
        X = X[~row_has_NaN]
        # y = y[~row_has_NaN]
        y = X["occupation"]
        X = X.drop(["occupation"],axis=1)
        cat_ix = X.select_dtypes(include=['category']).columns
        num_ix = X.select_dtypes(include=['int64', 'float64']).columns
        encoder = LabelEncoder()
        # now apply the transformation to all the columns:
        for col in cat_ix:
            X[col] = encoder.fit_transform(X[col])
        y = encoder.fit_transform(y)
        if dataset_name == 'adult_onehot':
            cat_features = OneHotEncoder(sparse=False).fit_transform(X[cat_ix])
            num_features = StandardScaler().fit_transform(X[num_ix])
            X = np.concatenate((num_features, cat_features), axis=1)
        else:
            X = StandardScaler().fit_transform(X)
    elif dataset_name in ['mushroom_num', 'mushroom_onehot']:
        X, y = fetch_openml('mushroom', version=1, return_X_y=True)
        encoder = LabelEncoder()
        # now apply the transformation to all the columns:
        for col in X.columns:
            X[col] = encoder.fit_transform(X[col])
        # X = X.drop(["veil-type"],axis=1)
        y = encoder.fit_transform(y)
        if dataset_name == 'mushroom_onehot':
            X = OneHotEncoder(sparse=False).fit_transform(X)
        else:
            X = StandardScaler().fit_transform(X)
    elif dataset_name == 'covertype':
        # https://www.openml.org/d/150
        # there are some 0/1 features -> consider just numeric
        X, y = fetch_openml('covertype', version=3, return_X_y=True)
        X = StandardScaler().fit_transform(X)
        y = LabelEncoder().fit_transform(y)
    elif dataset_name == 'shuttle':
        # https://www.openml.org/d/40685
        # all numeric, no missing values
        X, y = fetch_openml('shuttle', version=1, return_X_y=True)
        X = StandardScaler().fit_transform(X)
        y = LabelEncoder().fit_transform(y)
    elif dataset_name == 'magic':
        # https://www.openml.org/d/1120
        # all numeric, no missing values
        X, y = fetch_openml('MagicTelescope', version=1, return_X_y=True)
        X = StandardScaler().fit_transform(X)
        y = LabelEncoder().fit_transform(y)
    else:
        raise RuntimeError('Dataset does not exist')
    if bandit_model == "onehot":
        bandit = OneHotC2B(
            X=X, labels=y, 
            rew_optimal_labels=rew_optimal_labels, rew_other_labels=rew_other_labels,
            seed=seed,
            noise=noise, noise_param=noise_param,
            dataset_name=dataset_name, shuffle=shuffle
        )
    elif bandit_model == "expanded":
        bandit = ExpandedC2B(
            X=X, labels=y, 
            rew_optimal_labels=rew_optimal_labels, rew_other_labels=rew_other_labels,
            seed=seed,
            noise=noise, noise_param=noise_param,
            dataset_name=dataset_name, shuffle=shuffle
        )
    else:
        raise RuntimeError('Bandit model does not exist')
    return bandit
