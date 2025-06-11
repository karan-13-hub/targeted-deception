#make a train script for the kuhn poker game
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
#!/bin/bash

python train.py \
    --run_name bo3-5p\
    --env kuhn_poker\
    --num_players 5 \
    --deck_size 11 \
    --betting_rounds 5 \
    --ante 1 \
    --batch_size 64 \
    --device cuda:0 \
    --env Kuhn-poker \
    --epochs 10000 \
    --logdir ./data/ \
    --buffer_size 10000 \
    --eps 1.0 \
    --gamma 0.99\
    --tau 1e-3\
    --min_eps 0.01\
    --eps_frames 1000 \
    --seed 42 \
    --learning_rate 5e-4 \
    --lr_decay_gamma 0.999 \
    --clip_grad_norm 5.0 \
    --hidden_size 256 \
    --gru_hidden_size 512 \
    --belief_order 3 \
    --num_episodes_warmup 1000 \
    --num_train_eps_per_epoch 1 \
    --num_eval_eps_per_epoch 20 \
    --num_collect_agents 5 \
    --update_collect_agents 100 \
    --eval_after 50 \
    --num_train_iters_per_epoch 1 \
    --log_video 0 \
    --save_model_every 5000 \
    --save_ckpt_every 100 \
    --model_ckpt_fname ./model_ckpt/ \
    --data_ckpt_fname ./data_ckpt/ \