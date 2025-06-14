#!/usr/bin/env python
import argparse
import os
import random
import time
import logging
import numpy as np
from base import config


def get_parser():
    parser = argparse.ArgumentParser(description=' ')
    parser.add_argument('--config', type=str, default='**.yaml', help='config file')
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16", "fp8"],
        help="Whether to use mixed precision. Choose"
        "between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10."
        "and an Nvidia Ampere GPU.",
    )
    parser.add_argument("--cpu", action="store_true", help="If passed, will train on the CPU.")
    parser.add_argument('opts', help=' ', default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()
    assert args.config is not None
    cfg = config.load_cfg_from_cfg_file(args.config)
    if args.opts is not None:
        cfg = config.merge_cfg_from_list(cfg, args.opts)
    return cfg


def get_logger(save_dir):
    logger_name = "main-logger"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    
    # 创建一个文件处理器并设置保存位置
    file_handler = logging.FileHandler(os.path.join(save_dir, "training.log"))
    file_handler.setLevel(logging.INFO)
    
    # 设置输出格式
    formatter = logging.Formatter("[%(asctime)s %(levelname)s %(filename)s line %(lineno)d %(process)d]=>%(message)s")
    file_handler.setFormatter(formatter)
    
    # 将文件处理器添加到 logger 中
    logger.addHandler(file_handler)

    # 创建一个流处理器，用于在控制台上显示日志
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger


def count_parameters(model):
    param_num = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1024 / 1024
    pram_size = param_num * 4

    return param_num, pram_size

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def check_mkdir(dir_name):
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)


def check_makedirs(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


def main_process(args):
    return not args.multiprocessing_distributed or (
            args.multiprocessing_distributed and args.rank % args.ngpus_per_node == 0)
