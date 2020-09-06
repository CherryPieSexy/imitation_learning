# Imitation Learning

This repo contains simple PyTorch implementation of some Reinforcement Learning algorithms:
- Advantage Actor Critic (A2C) - a synchronous variant of [*A3C*](https://arxiv.org/abs/1602.01783)
- Proximal Policy Optimization (PPO) - one of the most popular RL algorithms [*PPO*](https://arxiv.org/abs/1707.06347), 
                               [*Truly PPO*](https://arxiv.org/abs/1903.07940), 
                               [*Implementation Matters*](https://arxiv.org/abs/2005.12729), 
                               [*A Large-Scale Empirical Study of PPO*](https://arxiv.org/abs/2006.05990)
- On-policy Maximum A Posteriori Policy Optimization (V-MPO) - Algorithm that DeepMind used in their last works [*V-MPO*](https://arxiv.org/abs/1909.12238) (not working yet...)
- Behavior Cloning (BC) - simple technique to clone some expert behaviour into new policy

Each algorithm supports vector/image observation spaces and discrete/continuous action spaces. 

## Why repo is called "Imitation Learning"?
When I started this project and repo, I thought that Imitation Learning would be my main focus, 
and model-free methods would be used only at the beginning to train 'experts'. 
However, it appeared that PPO implementation (and its tricks) took more time than I expected. 
As the result now most of the code is related to PPO, but I am still interested in Imitation Learning and going to add several related algorithms.

Also I found that PPO works remarkably well, better than reported in papers. 
For example, I was able to train Humanoid to run for 
~11.500 reward points in 2 hours on laptop without GPU, using only ~8 million environment transitions.

## Current Functionality

For now this repo contains some model-free on-policy algorithm implementations: A2C, PPO, V-MPO. 
Each algorithm supports discrete (Categorical, GumbelSoftmax) and continuous (Beta, Normal, tanh(Normal)) policy distributions, 
and vector or image observation environments.

As found in paper [*Implementation Matters*](https://arxiv.org/abs/2005.12729), 
PPO algo works mostly because of "code-level" optimizations. Here I implemented some of them:
- [x] Value function clipping
- [x] Observation normalization & clipping
- [x] Reward scaling & clipping (in my experiments normalization works better compared to scaling)
- [x] Orthogonal initialization of neural network weights
- [x] Gradient clipping
- [ ] Learning rate annealing (will be added soon)

In addition, I implemented roll-back loss from [*Truly PPO paper*](https://arxiv.org/abs/1903.07940), which works very well, 
and 'advantage-recompute' option from [*A Large-Scale Empirical Study of PPO paper*](https://arxiv.org/abs/2006.05990). 

For image-observation environments I added special regularization similar to [*CURL paper*](https://arxiv.org/abs/2004.04136), 
but instead of contrastive loss between features from convolutional feature extractor, 
I directly minimize D_KL between policies on augmented and non-augmented images.

As for Imitation Learning algorithms, there is only Behavior Cloning for now, but more will be added soon.

#### Code structure
    .
    ├── algorithms
        ├── agents        # A2C, PPO, V-MPO, BC, any different algo...
        └── ...           # all different algorithm parts: neural networks, probability distributions, etc
    ├── experts           # checkpoints of trained models, just some *.pth files with nn model description and weights inside
    ├── train_scripts
        ├── train_ppo.py  # script for trainging PPO algorithm
        ├── ...           # training scripts for different algorithms
        └── configs       # sh scripts (configs) to train an algorithm.
    ├── trainers          # implementation of trainers for different algo. Trainer is a manager that controls data-collection, model optimization and testing, etc.
    ├── utils             # all other 'support' functions that does not fit in any other folder.

#### Training example
Example of training config of PPO algo on CartPole-v1 env:
```bash

python3 train_scripts/train_ppo.py
--log_dir "logs/CartPole/"                                 # directory where logs will be stored
--env_name CartPole-v1                                     # name of the environment to train on
--action_repeat 1 --train_env_num 4 --test_env_num 4 \     # environment parameters
--hidden_size 64 --device_online cpu --device_train cpu \  # actor-critic hidden size and devices for data-collecting model and training model
--policy Categorical \                                     # policy distribution type
--returns_estimator '1-step' \                             # algo to estimate returns. Choose from '1-step', 'n-step', 'gae'
--learning_rate 1e-3 --gamma 0.99 --entropy 1e-3 \         # training hyperparameters
--clip_grad 0.5 --gae_lambda 0.95 \
--ppo_epsilon 0.1 --rollback_alpha 0.05 \                  # PPO hyperparameters
--ppo_n_epoch 4 --ppo_n_mini_batches 2 \
--n_epoch 10 --n_step_per_epoch 200 \                      # training parameters: number of epoch, epoch size, rollout size
--rollout_len 10 --n_tests_per_epoch 20
```

To see all the available options write ```python train_scripts/train_ppo.py -h``` in the terminal.

Obtained policy: 

![cartpole](gifs/cartpole.gif)

Training results (including training config, tensorboard logs and model checkpoints) will be saved in ```--log_dir``` folder.

#### Testing example
Results of trained policy may be shown with ```train_scripts/test.py``` script. 
This script is able to: 
- just show how policy acts in environment
- measure mean reward and episode len over some number of episodes
- record gif of policy playing episode (will be added soon)
- record demo file with trajectories

Type ```python train_scripts/test.py -h``` in the terminal to see how to use it.

#### Trained environments
GIFs of some of results:

BipedalWalker-v3: mean reward ~333, 0 fails over 1000 episodes (config will be added soon)

![bipedal](./gifs/bipedal.gif)

Humanoid-v3: mean reward ~11.3k, 14 fails over 1000 episodes, [config](train_scripts/configs/ppo_humanoid.sh)

![humanoid](./gifs/humanoid.gif)

CarRacing-v0: mean reward = 894 ± 32, 26 fails over 100 episodes 
(episode is considered failed if reward < 900), 
[config](train_scripts/configs/ppo_carracing.sh) 

![carracing](./gifs/carracing.gif)

## Current issues
PPO training is not stable enough: it focusing on _local_ reward, not in global episode return for some reason. 
It means that Bipedal, Humanoid and Car mostly improve their speed 
and don't care about falling on the ground or driving off the track. 
However, maximum reward is kinda stable - I can obtain ~10-11k max reward on Humanoid-v3 after 2 training epoch.

GPU support is not tested yet

## Further plans
- Test PPO on more environments
- Add logging where it is possible
- Add Motion Imitation [*DeepMimic paper*](https://arxiv.org/abs/1804.02717) algo
- Add self-play trainer with PPO as backbone algo
- Switch to more convenient configs (yaml?)
- Support recurrent policies models and training
- ...
