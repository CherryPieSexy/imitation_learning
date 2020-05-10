#!/usr/bin/env zsh

#export PYTHONPATH=$PYTHONPATH:../../
python3 train_scripts/train_ppo.py \
--log_dir "logs/CartPole/" \
--env_name CartPole-v1 --train_env_num 4 --test_env_num 4 \
--hidden_size 64 --device=cpu \
--policy Categorical --normalize_adv --returns_estimator 'gae' \
--learning_rate 1e-3 --gamma 0.99 --entropy 1e-3 --clip_grad 100500 \
--gae_lambda 0.95 \
--ppo_epsilon 0.1 --ppo_n_epoch 3 --ppo_mini_batch 15 \
--n_epoch 100 --n_step_per_epoch 1_000 --rollout_len 5 --n_tests_per_epoch 20
