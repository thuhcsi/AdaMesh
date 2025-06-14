#!/bin/sh
set -ex

export PYTHONPATH=./
exp_name=MEAD_ft/dddebug
exp_dir=RUN/${exp_name}
mkdir -p ${exp_dir}

accelerate launch --config_file=./config/accelerate_cfg_1p.yaml main/train.py \
    --config=./config/train/lora_mead.yaml \
    save_path ${exp_dir}