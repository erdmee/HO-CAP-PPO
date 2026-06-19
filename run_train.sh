#!/bin/bash
set -e

export CUDA_VISIBLE_DEVICES=0

: "${TASK:=Gr_shadow_train}"

# Train WITHOUT --video to keep the renderer off. 
# Monitor via tensorboard reward curves: generate the submission video separately with run_play.sh on a checkpoint.
python scripts/rl_games/train.py \
    --task "$TASK" \
    --num_envs "${NUM_ENVS:-1024}" \
    --headless