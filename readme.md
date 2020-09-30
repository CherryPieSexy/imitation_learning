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

## Current Functionality

For now this repo contains some model-free on-policy algorithm implementations: A2C, PPO, V-MPO and BC. 
Each algorithm supports discrete (Categorical, Bernoulli, GumbelSoftmax) and continuous (Beta, Normal, tanh(Normal), RealNVP) policy distributions, 
and vector or image observation environments. Beta and tanh(Normal) works best in my experiments (tested on BipedalWalker and Humanoid environments).

As found in paper [*Implementation Matters*](https://arxiv.org/abs/2005.12729), 
PPO algo works mostly because of "code-level" optimizations. Here I implemented most of them:
- [ ] Value function clipping (works better without it, not supported now)
- [x] Observation normalization & clipping
- [x] Reward scaling & clipping (in my experiments normalization works better compared to scaling)
- [x] Orthogonal initialization of neural network weights
- [x] Gradient clipping
- [ ] Learning rate annealing (will be added)

In addition, I implemented roll-back loss from [*Truly PPO paper*](https://arxiv.org/abs/1903.07940), which works very well, 
and 'advantage-recompute' option from [*A Large-Scale Empirical Study of PPO paper*](https://arxiv.org/abs/2006.05990). 

For image-observation environments I added special regularization similar to [*CURL paper*](https://arxiv.org/abs/2004.04136), 
but instead of contrastive loss between features from convolutional feature extractor, 
I directly minimize D_KL between policies on augmented and non-augmented images.

As for Imitation Learning algorithms, there is only Behavior Cloning for now, but more will be added.

#### Code structure
    .
    ├── algorithms
        ├── agents              # A2C, PPO, V-MPO, BC, any different agent algo...
        └── ...                 # all different algorithm parts: neural networks, probability distributions, etc
    ├── experts                 # checkpoints of trained models, just some *.pth files with nn model description and weights inside
    ├── train_scripts
        ├── train_on_policy.py  # script for trainging an on-policy algorithm
        ├── train_bc.py         # script to train policy with behavior cloning
        ├── test.py             # script for testing trained agent
        └── configs             # folder with .yaml configs to trian agents
    ├── trainers                # implementation of trainers for different algorithms. Trainer is a manager that controls data-collection, model optimization and testing, etc.
    ├── utils                   # all other 'support' functions that does not fit in any other folder.

#### Training example
Each experiment requires yaml config, look at examples here: [folder](train_scripts/configs).

Example of training PPO agent on CartPole-v1 env:
```bash
python train_scripts/train_on_policy.py -c train_scripts/configs/cartpole/ppo.yaml
```

Training results (including training config, tensorboard logs and model checkpoints) will be saved in ```--log_dir``` folder.

Obtained policy: 

![cartpole](gifs/cartpole.gif)

To train on custom environment (with gym interface) add in yaml config env type and name:
```yaml
env_type: 'gym.envs.classic_control.cartpole' # path to file with the environment class or installed python module
env_name: 'CartPoleEnv'  # the environment class name
env_args: {}  # dict with environment arguments, in this case it is empty
```
Similar functionality exists for custom env wrapper and neural network. To learn about available parameters for algorithms/trainers look into their docstrings.

#### Testing example
Results of trained policy may be shown with ```train_scripts/test.py``` script. 
This script is able to: 
- just show how policy acts in environment
- measure mean reward and episode len over some number of episodes
- record gif of policy playing episode (will be added)
- record demo file with trajectories

Type ```python train_scripts/test.py -h``` in the terminal to see how to use it.

#### Behavior Cloning example
Demo file for BC is expected to be .pickle with episodes list inside. 
An episode is a list of is \[observations, actions, rewards\], where observations = \[obs_0, obs_1, ..., obs_T\], 
similar with action and rewards.

- Record demo file from trained policy: 
    ```bash
    python train_scripts/test.py -f logs/cartpole/ppo/ -p checkpoints/epoch_10.pth -n 10 -r -d demo_files/cartpole_demo_10_ep.pickle -t -1
    ```
- Prepare config to train BC: [config](train_scripts/configs/cartpole/bc.yaml)
- Run BC training script: 
    ```bash
    python train_scripts/train_bc.py -c train_scripts/configs/cartpole/bc.yaml
    ```
- ???
- Enjoy policy:
    ```bash
    python train_scripts/test.py -f logs/cartpole/bc/ -p checkpoints/epoch_6.pth -n 10 -r
    ```

#### Trained environments
GIFs of some of results:

BipedalWalker-v3: mean reward ~333, 0 fails over 1000 episodes (config will be added soon)

![bipedal](./gifs/bipedal.gif)

Humanoid-v3: mean reward ~11.3k, 14 fails over 1000 episodes, [config](train_scripts/configs/ppo_humanoid.sh)

![humanoid](./gifs/humanoid.gif)

Experiments with Humanoid done in mujoco v2 
which have integration bug that makes environment easier. For academic purposes it is correct to use version of mujoco for Humanoid is 1.5

CarRacing-v0: mean reward = 894 ± 32, 26 fails over 100 episodes 
(episode is considered failed if reward < 900), 
[config](train_scripts/configs/ppo_carracing.sh) 

![carracing](./gifs/carracing.gif)

## Current issues
V-MPO implementation trains slower than A2C. Probably because of not optimal hyper-parameters sampling, need to investigate.

PPO training is not stable enough: it focusing on _local_ reward, not in global episode return for some reason. 
It means that Bipedal, Humanoid and Car mostly improve their speed 
and don't care about falling on the ground or driving off the track. 
However, maximum reward is kinda stable - I can obtain ~10-11k max reward on Humanoid-v3 after 2 training epoch.

RealNVP converges too fast and exploration becomes too small to find better reward and improve policy further. 
Tuning entropy regularization is difficult: with too low entropy coefficient agent learns and converges same, 
but with too high values agent optimizes only entropy.

Entropy calculation for squeezed distribution (tanh(Normal) and RealNVP) possibly coded wrong, need to be checked. 

## Further plans
- Add logging where it is possible
- Add Motion Imitation [*DeepMimic paper*](https://arxiv.org/abs/1804.02717) algo
- Add self-play trainer with PPO as backbone algo
- Support recurrent policies models and training
- ...
