#!/usr/bin/env python
import os
import time
import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
from tensorboardX import SummaryWriter

from base.baseTrainer import poly_learning_rate, save_checkpoint
from base.utilities import get_parser, get_logger, AverageMeter, count_parameters
from models import get_model
from metrics.loss import calc_l1_loss
from torch.optim.lr_scheduler import StepLR

from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
import loralib as lora


def main():
    cfg = get_parser()
    # import pdb; pdb.set_trace()
    ddp_scaler = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(cpu=False, mixed_precision="no", kwargs_handlers=[ddp_scaler])

    # ####################### Model ####################### #
    global logger, writer
    logger = get_logger(cfg.save_path)
    writer = SummaryWriter(cfg.save_path)
    model = get_model(cfg)
    if accelerator.is_local_main_process:
        logger.info(cfg)

    if cfg.resume_path is not None:
        if accelerator.is_local_main_process:
            logger.info("Loading checkpoint '{}'".format(cfg.resume_path))
        pretrain_checkpoint = torch.load(cfg.resume_path, map_location='cpu')
        model.load_state_dict(pretrain_checkpoint["state_dict"], strict=True)
        # lora.mark_only_lora_as_trainable(model)
    
    if accelerator.is_local_main_process:
        param_num, param_size = count_parameters(model)
        logger.info(f"Number of parameters: {param_num}M, size: {param_size}MB")

    # ####################### Optimizer ####################### #
    if cfg.use_sgd:
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), 
                                    lr=cfg.base_lr, momentum=cfg.momentum,
                                    weight_decay=cfg.weight_decay)
    else:
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), 
                                      lr=cfg.base_lr)

    if cfg.StepLR:
        scheduler = StepLR(optimizer, step_size=cfg.step_size, gamma=cfg.gamma)
    else:
        scheduler = None

    # ####################### Data Loader ####################### #
    from dataset.data_loader import get_dataloaders
    dataset = get_dataloaders(cfg)
    train_loader = dataset['train']
    if cfg.evaluate:
        val_loader = dataset['valid']
    
    model, optimizer, scheduler, train_loader, val_loader \
    = accelerator.prepare(model, optimizer, scheduler, train_loader, val_loader)

    # model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    # ####################### Train ############################# #
    for epoch in range(cfg.start_epoch, cfg.epochs):
        loss_mid_train, loss_after_train, loss_train = train(train_loader, model, calc_l1_loss, optimizer, epoch, cfg, accelerator)
        epoch_log = epoch + 1
        if cfg.StepLR:
            scheduler.step()
        if accelerator.is_local_main_process:
            logger.info('TRAIN Epoch: {} '
                        'loss_mid_train: {} '
                        'loss_after_train: {} '
                        'loss_train: {} '
                        .format(epoch_log, loss_mid_train, loss_after_train, loss_train)
                        )
            for m, s in zip([loss_mid_train, loss_after_train, loss_train],
                            ["train/loss_mid", "train/loss_after", "train/loss"]):
                writer.add_scalar(s, m, epoch_log)


        if cfg.evaluate and (epoch_log % cfg.eval_freq == 0):
            loss_mid_val, loss_after_val, loss_val = validate(val_loader, model, calc_l1_loss, epoch, cfg, accelerator)
            if accelerator.is_local_main_process:
                logger.info('VAL Epoch: {} '
                            'loss_mid_val: {} '
                            'loss_after_val: {} '
                            'loss_val: {} '
                            .format(epoch_log, loss_mid_val, loss_after_val, loss_val)
                            )
                for m, s in zip([loss_mid_val, loss_after_val, loss_val],
                                ["val/loss_mid", "val/loss_after", "val/loss"]):
                    writer.add_scalar(s, m, epoch_log)


        if (epoch_log % cfg.save_freq == 0) and accelerator.is_local_main_process:
            save_checkpoint(model,
                            sav_path=os.path.join(cfg.save_path, 'model'),
                            filename=f"model_{epoch_log}.pth.tar",
                            )


def train(train_loader, model, loss_fn, optimizer, epoch, cfg, accelerator):
    device = accelerator.device

    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_mid_meter = AverageMeter()
    loss_after_meter = AverageMeter()
    loss_meter = AverageMeter()

    # model.train()
    end = time.time()
    max_iter = cfg.epochs * len(train_loader)
    # for i, data in enumerate(train_loader):
    for i, (speech_feat, motion, output_mask, input_len) in enumerate(train_loader):
        current_iter = epoch * len(train_loader) + i + 1
        data_time.update(time.time() - end)
        speech_feat = speech_feat.to(device)
        motion = motion.to(device)
        output_mask = output_mask.to(device)
        input_len = input_len.to(device)

        predict_mid, predict = model(speech_feat, input_len)
        
        # import pdb; pdb.set_trace()
        # LOSS
        loss_mid, loss_after, loss = loss_fn(predict_mid, predict, motion, output_mask)

        optimizer.zero_grad()
        accelerator.backward(loss)
        # loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()
        for m, x in zip([loss_mid_meter, loss_after_meter, loss_meter],
                        [loss_mid, loss_after, loss]): #info[0] is perplexity
            m.update(x.item(), 1)
        
        # Adjust lr
        if cfg.poly_lr:
            current_lr = poly_learning_rate(cfg.base_lr, current_iter, max_iter, power=cfg.power)
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr
        else:
            current_lr = optimizer.param_groups[0]['lr']

        # calculate remain time
        remain_iter = max_iter - current_iter
        remain_time = remain_iter * batch_time.avg
        t_m, t_s = divmod(remain_time, 60)
        t_h, t_m = divmod(t_m, 60)
        remain_time = '{:02d}:{:02d}:{:02d}'.format(int(t_h), int(t_m), int(t_s))

        if (i + 1) % cfg.print_freq == 0 and accelerator.is_local_main_process:
            logger.info('Epoch: [{}/{}][{}/{}] '
                        'Data: {data_time.val:.3f} ({data_time.avg:.3f}) '
                        'Batch: {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                        'Remain: {remain_time} '
                        'Loss: {loss_meter.val:.4f} '
                        .format(epoch + 1, cfg.epochs, i + 1, len(train_loader),
                                batch_time=batch_time, data_time=data_time,
                                remain_time=remain_time,
                                loss_meter=loss_meter
                                ))
            for m, s in zip([loss_mid_meter, loss_after_meter, loss_meter],
                            ["train_batch/loss_mid", "train_batch/loss_after", "train_batch/loss"]):
                writer.add_scalar(s, m.val, current_iter)
            writer.add_scalar('learning_rate', current_lr, current_iter)

    return loss_mid_meter.avg, loss_after_meter.avg, loss_meter.avg


def validate(val_loader, model, loss_fn, epoch, cfg, accelerator):
    device = accelerator.device

    loss_mid_meter = AverageMeter()
    loss_after_meter = AverageMeter()
    loss_meter = AverageMeter()
    model.eval()

    with torch.no_grad():
        for i, (speech_feat, motion, output_mask, input_len) in enumerate(val_loader):
            speech_feat = speech_feat.to(device)
            motion = motion.to(device)
            output_mask = output_mask.to(device)
            input_len = input_len.to(device)

            predict_mid, predict = model(speech_feat, input_len)

            # LOSS
            loss_mid, loss_after, loss = loss_fn(predict_mid, predict, motion, output_mask)

            # if cfg.distributed:
            #     loss = reduce_tensor(loss, cfg)


            for m, x in zip([loss_mid_meter, loss_after_meter, loss_meter],
                            [loss_mid, loss_after, loss]):
                m.update(x.item(), 1) #batch_size = 1 for validation


    return loss_mid_meter.avg, loss_after_meter.avg, loss_meter.avg


if __name__ == '__main__':
    main()
