"""
07_train_pretrain.py - Pretraining Runner for Point-MAE

This module handles the self-supervised pretraining of Point-MAE:
- Dataset loading (ShapeNet55)
- Model building
- Training loop with Chamfer Distance loss
- Checkpoint saving
- TensorBoard logging
- Multi-GPU distributed training support

Training Configuration:
- Optimizer: AdamW (lr=0.001, weight_decay=0.05)
- Scheduler: Cosine LR with warmup
- Epochs: 300
- Batch size: 128 (total across GPUs)
- Data augmentation: Scale and Translate
"""

import os
import time
import torch
import torch.nn as nn
import numpy as np

from sklearn.svm import LinearSVC
from timm.scheduler import CosineLRScheduler


class AverageMeter:
    """Computes and stores the average and current value."""
    
    def __init__(self, items=None):
        self.items = items
        self.n = 0
        self.reset()
    
    def reset(self):
        if self.items is not None:
            self.vals = [0] * len(self.items)
            self.avgs = [0] * len(self.items)
            self.sums = [0] * len(self.items)
        else:
            self.val = 0
            self.avg = 0
            self.sum = 0
    
    def update(self, vals, n=1):
        if self.items is not None:
            for i, v in enumerate(vals):
                self.vals[i] = v
                self.sums[i] += v * n
            self.n += n
            for i in range(len(self.items)):
                self.avgs[i] = self.sums[i] / self.n
        else:
            self.val = vals
            self.sum += vals * n
            self.n += n
            self.avg = self.sum / self.n
    
    def val(self):
        return self.vals if self.items else self.val
    
    def avg(self):
        return self.avgs if self.items else self.avg


class Acc_Metric:
    """Accuracy metric for tracking best performance."""
    
    def __init__(self, acc=0.):
        if isinstance(acc, dict):
            self.acc = acc['acc']
        else:
            self.acc = acc
    
    def better_than(self, other):
        return self.acc > other.acc
    
    def state_dict(self):
        return {'acc': self.acc}


def worker_init_fn(worker_id):
    """Initialize worker with unique random seed."""
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def build_dataset(args, config):
    """Build dataset and dataloader."""
    from datasets import build_dataset as build_ds
    
    dataset = build_ds(config._base_, config.others)
    shuffle = config.others.subset == 'train'
    
    if args.distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=shuffle)
        dataloader = torch.utils.data.DataLoader(
            dataset, 
            batch_size=config.others.bs,
            num_workers=int(args.num_workers),
            drop_last=config.others.subset == 'train',
            worker_init_fn=worker_init_fn,
            sampler=sampler
        )
    else:
        sampler = None
        dataloader = torch.utils.data.DataLoader(
            dataset, 
            batch_size=config.others.bs,
            shuffle=shuffle,
            drop_last=config.others.subset == 'train',
            num_workers=int(args.num_workers),
            worker_init_fn=worker_init_fn
        )
    
    return sampler, dataloader


def build_model(config):
    """Build model from config."""
    from model import build_model as build_m
    return build_m(config)


def build_optimizer_scheduler(base_model, config):
    """Build optimizer and learning rate scheduler."""
    opti_config = config.optimizer
    
    if opti_config.type == 'AdamW':
        # Separate weight decay for different parameter groups
        def add_weight_decay(model, weight_decay=1e-5, skip_list=()):
            decay = []
            no_decay = []
            for name, param in model.module.named_parameters():
                if not param.requires_grad:
                    continue
                if len(param.shape) == 1 or name.endswith(".bias") or 'token' in name or name in skip_list:
                    no_decay.append(param)
                else:
                    decay.append(param)
            return [
                {'params': no_decay, 'weight_decay': 0.},
                {'params': decay, 'weight_decay': weight_decay}
            ]
        
        param_groups = add_weight_decay(base_model, weight_decay=opti_config.kwargs.weight_decay)
        optimizer = torch.optim.AdamW(param_groups, **opti_config.kwargs)
    elif opti_config.type == 'Adam':
        optimizer = torch.optim.Adam(base_model.parameters(), **opti_config.kwargs)
    elif opti_config.type == 'SGD':
        optimizer = torch.optim.SGD(base_model.parameters(), nesterov=True, **opti_config.kwargs)
    else:
        raise NotImplementedError(f"Optimizer {opti_config.type} not implemented")
    
    # Scheduler
    sche_config = config.scheduler
    if sche_config.type == 'CosLR':
        scheduler = CosineLRScheduler(
            optimizer,
            t_initial=sche_config.kwargs.epochs,
            t_mul=1,
            lr_min=1e-6,
            decay_rate=0.1,
            warmup_lr_init=1e-6,
            warmup_t=sche_config.kwargs.initial_epochs,
            cycle_limit=1,
            t_in_epochs=True
        )
    elif sche_config.type == 'StepLR':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, **sche_config.kwargs)
    else:
        raise NotImplementedError(f"Scheduler {sche_config.type} not implemented")
    
    return optimizer, scheduler


def save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, prefix, args, logger=None):
    """Save model checkpoint."""
    if args.local_rank == 0:
        state = {
            'base_model': base_model.module.state_dict() if args.distributed else base_model.module.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'metrics': metrics.state_dict() if metrics is not None else {},
            'best_metrics': best_metrics.state_dict() if best_metrics is not None else {},
        }
        save_path = os.path.join(args.experiment_path, f'{prefix}.pth')
        torch.save(state, save_path)
        if logger:
            logger.info(f"Save checkpoint at {save_path}")


def resume_model(base_model, args, logger=None):
    """Resume model from checkpoint."""
    ckpt_path = os.path.join(args.experiment_path, 'ckpt-last.pth')
    if not os.path.exists(ckpt_path):
        if logger:
            logger.info(f'[RESUME] No checkpoint at {ckpt_path}')
        return 0, 0
    
    if logger:
        logger.info(f'[RESUME] Loading from {ckpt_path}')
    
    map_location = {'cuda:%d' % 0: 'cuda:%d' % args.local_rank}
    state_dict = torch.load(ckpt_path, map_location=map_location)
    
    base_ckpt = {k.replace("module.", ""): v for k, v in state_dict['base_model'].items()}
    base_model.load_state_dict(base_ckpt, strict=True)
    
    start_epoch = state_dict['epoch'] + 1
    best_metrics = state_dict['best_metrics']
    if not isinstance(best_metrics, dict):
        best_metrics = best_metrics.state_dict()
    
    if logger:
        logger.info(f'[RESUME] Resume from epoch {start_epoch - 1}')
    
    return start_epoch, best_metrics


def resume_optimizer(optimizer, args, logger=None):
    """Resume optimizer state."""
    ckpt_path = os.path.join(args.experiment_path, 'ckpt-last.pth')
    if not os.path.exists(ckpt_path):
        return
    
    state_dict = torch.load(ckpt_path, map_location='cpu')
    optimizer.load_state_dict(state_dict['optimizer'])
    if logger:
        logger.info(f'[RESUME] Loaded optimizer from {ckpt_path}')


def reduce_tensor(tensor, args):
    """Reduce tensor across all GPUs."""
    rt = tensor.clone()
    torch.distributed.all_reduce(rt, op=torch.distributed.ReduceOp.SUM)
    rt /= args.world_size
    return rt


def run_net(args, config, train_writer=None, val_writer=None):
    """
    Main pretraining loop.
    
    Args:
        args: Command line arguments
        config: Training configuration
        train_writer: TensorBoard writer for training
        val_writer: TensorBoard writer for validation
    """
    import logging
    logger = logging.getLogger(args.log_name)
    
    # Import transforms
    from transforms import PointcloudScaleAndTranslate
    train_transforms = PointcloudScaleAndTranslate()
    
    # Build datasets
    (train_sampler, train_dataloader), (_, test_dataloader) = \
        build_dataset(args, config.dataset.train), build_dataset(args, config.dataset.val)
    
    # Build model
    base_model = build_model(config.model)
    if args.use_gpu:
        base_model.to(args.local_rank)
    
    # Initialize metrics
    start_epoch = 0
    best_metrics = Acc_Metric(0.)
    metrics = Acc_Metric(0.)
    
    # Resume if needed
    if args.resume:
        start_epoch, best_metric = resume_model(base_model, args, logger)
        best_metrics = Acc_Metric(best_metric)
    elif args.start_ckpts is not None:
        state_dict = torch.load(args.start_ckpts, map_location='cpu')
        base_ckpt = {k.replace("module.", ""): v for k, v in state_dict['base_model'].items()}
        base_model.load_state_dict(base_ckpt, strict=True)
        logger.info(f'Loaded checkpoint from {args.start_ckpts}')
    
    # Distributed training
    if args.distributed:
        if args.sync_bn:
            base_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(base_model)
            logger.info('Using Synchronized BatchNorm')
        base_model = nn.parallel.DistributedDataParallel(
            base_model, 
            device_ids=[args.local_rank % torch.cuda.device_count()],
            find_unused_parameters=True
        )
        logger.info('Using Distributed Data Parallel')
    else:
        logger.info('Using Data Parallel')
        base_model = nn.DataParallel(base_model).cuda()
    
    # Build optimizer and scheduler
    optimizer, scheduler = build_optimizer_scheduler(base_model, config)
    
    if args.resume:
        resume_optimizer(optimizer, args, logger)
    
    # Training loop
    base_model.zero_grad()
    for epoch in range(start_epoch, config.max_epoch + 1):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        
        base_model.train()
        
        epoch_start_time = time.time()
        batch_start_time = time.time()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter(['Loss'])
        
        num_iter = 0
        n_batches = len(train_dataloader)
        
        for idx, (taxonomy_ids, model_ids, data) in enumerate(train_dataloader):
            num_iter += 1
            n_itr = epoch * n_batches + idx
            
            data_time.update(time.time() - batch_start_time)
            
            npoints = config.dataset.train.others.npoints
            dataset_name = config.dataset.train._base_.NAME
            
            if dataset_name == 'ShapeNet':
                points = data.cuda()
            elif dataset_name == 'ModelNet':
                points = data[0].cuda()
                from model import fps
                points = fps(points, npoints)
            else:
                raise NotImplementedError(f'Dataset {dataset_name} not supported')
            
            assert points.size(1) == npoints
            
            # Apply transforms
            points = train_transforms(points)
            
            # Forward pass
            loss = base_model(points)
            
            # Handle multi-GPU
            try:
                loss.backward()
            except:
                loss = loss.mean()
                loss.backward()
            
            # Optimizer step
            if num_iter == config.step_per_update:
                num_iter = 0
                optimizer.step()
                base_model.zero_grad()
            
            # Update metrics
            if args.distributed:
                loss = reduce_tensor(loss, args)
                losses.update([loss.item() * 1000])
            else:
                losses.update([loss.item() * 1000])
            
            if args.distributed:
                torch.cuda.synchronize()
            
            # TensorBoard logging
            if train_writer is not None:
                train_writer.add_scalar('Loss/Batch/Loss', loss.item(), n_itr)
                train_writer.add_scalar('Loss/Batch/LR', optimizer.param_groups[0]['lr'], n_itr)
            
            batch_time.update(time.time() - batch_start_time)
            batch_start_time = time.time()
            
            # Print progress
            if idx % 20 == 0:
                logger.info(
                    f'[Epoch {epoch}/{config.max_epoch}][Batch {idx + 1}/{n_batches}] '
                    f'BatchTime = {batch_time.vals[0] if hasattr(batch_time, "vals") else batch_time.val:.3f}s '
                    f'DataTime = {data_time.vals[0] if hasattr(data_time, "vals") else data_time.val:.3f}s '
                    f'Loss = {losses.avgs[0]:.4f} '
                    f'lr = {optimizer.param_groups[0]["lr"]:.6f}'
                )
        
        # Step scheduler
        if isinstance(scheduler, list):
            for item in scheduler:
                item.step(epoch)
        else:
            scheduler.step(epoch)
        
        epoch_end_time = time.time()
        
        # Epoch logging
        if train_writer is not None:
            train_writer.add_scalar('Loss/Epoch/Loss', losses.avgs[0], epoch)
        
        logger.info(
            f'[Training] EPOCH: {epoch} '
            f'EpochTime = {epoch_end_time - epoch_start_time:.3f}s '
            f'Loss = {losses.avgs[0]:.4f} '
            f'lr = {optimizer.param_groups[0]["lr"]:.6f}'
        )
        
        # Save checkpoints
        save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, 'ckpt-last', args, logger)
        
        if epoch % 25 == 0 and epoch >= 250:
            save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, f'ckpt-epoch-{epoch:03d}', args, logger)
    
    # Close writers
    if train_writer is not None:
        train_writer.close()
    if val_writer is not None:
        val_writer.close()


def test_net():
    """Placeholder for test function."""
    pass


if __name__ == '__main__':
    print("Pretraining runner module loaded successfully!")
    print("Use 09_main.py to start training.")
