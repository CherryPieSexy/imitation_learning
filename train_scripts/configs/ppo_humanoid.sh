#!/usr/bin/env bash

export PYTHONPATH=$PYTHONPATH:../../
python3 train_scripts/train_ppo.py \
--log_dir "logs/Humanoid/ppo/8/" \
--env_name Humanoid-v3 \
--normalize_obs --normalize_reward \
--obs_clip 10.0 --reward_clip 10.0 \
--action_repeat 1 --train_env_num 16 --test_env_num 4 \
--hidden_size 64 --device_online cpu --device_train cpu \
--policy 'RealNVP' --returns_estimator 'gae' \
--learning_rate 3e-4 --gamma 0.99 --entropy 0.0 --clip_grad 0.5 \
--gae_lambda 0.9 \
--ppo_epsilon 0.2 --rollback_alpha 0.05 --ppo_n_epoch 10 --ppo_n_mini_batches 8 \
--n_epoch 10 --n_step_per_epoch 1000 --rollout_len 256 --n_tests_per_epoch 20
