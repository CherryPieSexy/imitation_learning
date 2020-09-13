#!/usr/bin/env bash

export PYTHONPATH=$PYTHONPATH:../../
python3 train_scripts/train_ppo.py \
--log_dir "logs/Bipedal/new_4/2/" \
--env_name BipedalWalkerHardcore-v3 \
--die_penalty 80 \
--normalize_obs --normalize_reward \
--obs_clip 10.0 \
--action_repeat 1 --train_env_num 16 --test_env_num 4 \
--hidden_size 128 --device_online cpu --device_train cpu \
--policy 'TanhNormal' --returns_estimator 'gae' \
--learning_rate 3e-4 --gamma 0.99 --entropy 5e-2 --clip_grad 0.5 \
--gae_lambda 0.9 \
--ppo_epsilon 0.1 --rollback_alpha 0.05 --ppo_n_epoch 5 --ppo_n_mini_batches 4 \
--n_epoch 20 --n_step_per_epoch 2_000 --rollout_len 64 --n_tests_per_epoch 100
