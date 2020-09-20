#!/usr/bin/env zsh

export PYTHONPATH=$PYTHONPATH:../../
python3 train_scripts/train_v_mpo.py \
--log_dir "logs/CartPole/vmpo/1/" \
--env_name CartPole-v1 \
--update_period 5 --return_pi \
--action_repeat 1 --train_env_num 4 --test_env_num 4 \
--hidden_size 64 --device_online=cpu --device_train=cpu \
--policy 'Categorical' --returns_estimator '1-step' \
--learning_rate 1e-3 --gamma 0.99 --entropy 0.0 --clip_grad 0.5 \
--gae_lambda 0.9 \
--n_epoch 20 --n_step_per_epoch 200 --rollout_len 5 --n_tests_per_epoch 20
