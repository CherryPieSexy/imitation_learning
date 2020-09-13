#!/usr/bin/env zsh

export PYTHONPATH=$PYTHONPATH:../../
python3 train_scripts/train_ppo.py \
--log_dir "logs/CarRacing/15/" \
--env_name CarRacing-v0 \
--action_repeat 2 \
--train_env_num 8 --test_env_num 2 \
--hidden_size 0 --device_online=cpu --device_train=cpu \
--policy 'Beta' --returns_estimator 'gae' \
--learning_rate 5e-5 --gamma 0.995 --entropy 0.0 --clip_grad 0.1 \
--gae_lambda 0.9 \
--ppo_epsilon 0.2 --rollback_alpha 0.1 --ppo_n_epoch 10 --ppo_n_mini_batches 8 \
--n_epoch 20 --n_step_per_epoch 50 --rollout_len 128 --n_tests_per_epoch 10 \
--image_aug_alpha 0.1
