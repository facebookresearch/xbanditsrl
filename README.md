# Contextual Bandit Spectral Representation Learner (xBanditSRL)

This is a PyTorch implementation of **BanditSRL** from

**Scalable Representation Learning in Linear Contextual Bandits with Constant Regret Guarantees**

by [Andrea Tirinzoni](https://andreatirinzoni.github.io/), [Matteo Papini](https://t3p.github.io/), [Ahmed Touati](https://scholar.google.com/citations?user=D4LT5xAAAAAJ&hl=en), [Alessandro Lazaric](https://scholar.google.com/citations?user=6JZ3R6wAAAAJ&hl=en), [Matteo Pirotta](https://teopir.github.io/)


[[Paper]](https://arxiv.org/abs/2210.13083)

## Requirements
The code can be run either on cpu or on gpu with CUDA 11. Then, the simplest way to install all required dependencies is to create an anaconda environment by running
```sh
conda env create -f conda_env.yml
```
If you don't have a CUDA-compatible GPU, remove the line "nvidia::cudatoolkit=11.3" from `conda_env.yml`.

After the instalation ends you can activate your environment with
```sh
conda activate xbanditsrl
```

## Instructions

To run an algorithm on task you can use the script `runner_exp.py`

```sh
python runner_exp.py algo=bsrllinucb domain=wheel
```

Logs are stored in the output folder. To launch tensorboard run:

```sh
tensorboard --logdir output
```

By default, the code runs on CPU. To switch to GPU, add the command line argument `device=cuda` (or modify configs/xbrl/config.yaml).

### Implemented Agents

| Agent | Name | Paper |
|---|---|---|
|LinUCB| `linucb`| [paper](https://arxiv.org/abs/1003.0146) |
|epsilon-GREEDY| `egreedy`| |
|BanditSRL-LinUCB| `bsrllinucb`| OURS |
|BanditSRL-eps-GREEDY| `bsrlegreedy`| OURS |
|BanditSRL-TS| `bsrlts`| OURS |
|BanditSRL-IGW (inverse gap weighting)| `bsrligw`| OURS |
|NeuralUCB (diagonal)| `gradientucb`| [paper](https://arxiv.org/abs/1911.04462) |
|inverse gap weighting strategy| `igwexp`| [paper1](http://phillong.info/publications/peval.pdf), [paper2](https://arxiv.org/abs/2002.04926), [paper3](https://arxiv.org/abs/2003.12699)  |
|RFF-based algo| `rfflinucb`, `rffegreedy`| |


Neural-LinUCB, Neural-eps-GREEDY, Neural-TS (see [paper](https://arxiv.org/abs/1802.09127)) are special cases of BanditSRL with `check_glrt=False`, `weight_rayleigh=0`, `weight_weak=0`.

### Experiments in the paper

To run the full set of experiments (this can take long time) run the following commands

```sh
./scripts/expall.sh
./run_{dataset}.sh

./scripts/wheelnetexp.sh
./run_netexp_{dataset}.sh
```

where dataset can be `magic`, `statlog`, `covertype`, `mushroom` and `wheel`.


## License
The majority of xBanditSRL is licensed under CC-BY-NC, however portions of the project are available under separate license terms: scikit-learn is licensed under the BSD license; tensorflow is licensed under Apache 2.0.

