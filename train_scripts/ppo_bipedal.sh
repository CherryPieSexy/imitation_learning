#!/usr/bin/env zsh

export PYTHONPATH=$PYTHONPATH:../../
python3 train_scripts/train_ppo.py \
--log_dir "logs/Bipedal/new/10/" \
--env_name BipedalWalkerHardcore-v3 --normalize_reward \
--frame_skip 4 --train_env_num 16 --test_env_num 4 \
--hidden_size 64 --device=cpu \
--policy 'TanhNormal' --returns_estimator 'gae' \
--learning_rate 1e-3 --gamma 0.99 --entropy 1e-3 --clip_grad 0.5 \
--gae_lambda 0.95 \
--ppo_epsilon 0.1 --ppo_n_epoch 4 --ppo_mini_batch 160 \
--n_epoch 20 --n_step_per_epoch 5_000 --rollout_len 20 --n_tests_per_epoch 20
