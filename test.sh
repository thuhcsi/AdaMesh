#!/bin/sh
set -ex

export PYTHONPATH=./

model_path="./xxx"
test_data_dir="./xxx"
test_save_dir="./xxx"

CUDA_VISIBLE_DEVICES=0 python main/test.py \
    --config=./config/train/s2e_base.yaml \
    model_path $model_path \
    test_data_dir $test_data_dir \
    test_save_dir $test_save_dir
