#make a train script for the kuhn poker game
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
#!/bin/bash

python eval_models.py \
    --num_players 2 \
    --deck_size 3 \
    --betting_rounds 2 \
    --ante 1 \
    --batch_size 128 \
    --learning_rate 1e-4 \
    --lr_decay_gamma 0.999 \
    --device cuda:0 \
    --hidden_size 128 \
    --clip_grad_norm 5.0 \
    --models_dir ./trained_models/ \
    --all_against_all \
    # --first_order True \
    # --second_order True \
    # --load_from_checkpoint  \