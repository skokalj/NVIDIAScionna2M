"""
08_train_finetune.py - Finetuning Runner for Point-MAE

This module handles finetuning Point-MAE for classification:
- Dataset loading (ModelNet40, ScanObjectNN)
- Load pretrained weights
- Training loop with Cross-Entropy loss
- Validation and testing
- Vote-based evaluation
- Multi-GPU distributed training support

Training Configuration:
- Optimizer: AdamW (lr=0.0005, weight_decay=0.05)
- Scheduler: Cosine LR with warmup
- Epochs: 300
- Batch size: 32
- Data augmentation: Scale and Translate
"""

import os
import time
import torch
import torch.nn as nn
import numpy as np

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


class Acc_Metric:
    """Accuracy metric for tracking best performance."""
    
    def __init__(self, acc=0.):
        if isinstance(acc, dict):
            self.acc = acc['acc']
        elif hasattr(acc, 'acc'):
            self.acc = acc.acc
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
            'base_model': base_model.module.state_dict() if hasattr(base_model, 'module') else base_model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'metrics': metrics.state_dict() if metrics is not None else {},
            'best_metrics': best_metrics.state_dict() if best_metrics is not None else {},
        }
        save_path = os.path.join(args.experiment_path, f'{prefix}.pth')
        torch.save(state, save_path)
        if logger:
            logger.info(f"Save checkpoint at {save_path}")


def load_model(base_model, ckpt_path, logger=None):
    """Load model weights from checkpoint."""
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f'No checkpoint at {ckpt_path}')
    
    if logger:
        logger.info(f'Loading weights from {ckpt_path}')
    
    state_dict = torch.load(ckpt_path, map_location='cpu')
    
    if state_dict.get('model') is not None:
        base_ckpt = {k.replace("module.", ""): v for k, v in state_dict['model'].items()}
    elif state_dict.get('base_model') is not None:
        base_ckpt = {k.replace("module.", ""): v for k, v in state_dict['base_model'].items()}
    else:
        raise RuntimeError('Checkpoint format not recognized')
    
    base_model.load_state_dict(base_ckpt, strict=True)
    
    epoch = state_dict.get('epoch', -1)
    metrics = state_dict.get('metrics', 'No Metrics')
    
    if logger:
        logger.info(f'Loaded checkpoint from epoch {epoch}')


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


def gather_tensor(tensor, args):
    """Gather tensors from all GPUs."""
    tensor_list = [torch.zeros_like(tensor) for _ in range(args.world_size)]
    torch.distributed.all_gather(tensor_list, tensor)
    return torch.cat(tensor_list, dim=0)


def validate(base_model, test_dataloader, epoch, val_writer, args, config, logger=None):
    """Validate model on test set."""
    base_model.eval()
    
    test_pred = []
    test_label = []
    npoints = config.npoints
    
    with torch.no_grad():
        for idx, (taxonomy_ids, model_ids, data) in enumerate(test_dataloader):
            points = data[0].cuda()
            label = data[1].cuda()
            
            from model import fps
            points = fps(points, npoints)
            
            logits = base_model(points)
            target = label.view(-1)
            
            pred = logits.argmax(-1).view(-1)
            
            test_pred.append(pred.detach())
            test_label.append(target.detach())
    
    test_pred = torch.cat(test_pred, dim=0)
    test_label = torch.cat(test_label, dim=0)
    
    if args.distributed:
        test_pred = gather_tensor(test_pred, args)
        test_label = gather_tensor(test_label, args)
    
    acc = (test_pred == test_label).sum() / float(test_label.size(0)) * 100.
    
    if logger:
        logger.info(f'[Validation] EPOCH: {epoch}  acc = {acc:.4f}')
    
    if args.distributed:
        torch.cuda.synchronize()
    
    if val_writer is not None:
        val_writer.add_scalar('Metric/ACC', acc, epoch)
    
    return Acc_Metric(acc)


def validate_vote(base_model, test_dataloader, epoch, val_writer, args, config, logger=None, times=10):
    """Validate with voting (multiple augmented predictions)."""
    if logger:
        logger.info(f'[VALIDATION_VOTE] epoch {epoch}')
    
    base_model.eval()
    
    from 04_transforms import PointcloudScaleAndTranslate
    test_transforms = PointcloudScaleAndTranslate()
    
    test_pred = []
    test_label = []
    npoints = config.npoints
    
    # Import pointnet2 for FPS
    try:
        from pointnet2_ops import pointnet2_utils
        HAS_POINTNET2 = True
    except:
        HAS_POINTNET2 = False
    
    with torch.no_grad():
        for idx, (taxonomy_ids, model_ids, data) in enumerate(test_dataloader):
            points_raw = data[0].cuda()
            label = data[1].cuda()
            
            if npoints == 1024:
                point_all = 1200
            elif npoints == 4096:
                point_all = 4800
            elif npoints == 8192:
                point_all = 8192
            else:
                point_all = int(npoints * 1.2)
            
            if points_raw.size(1) < point_all:
                point_all = points_raw.size(1)
            
            if HAS_POINTNET2:
                fps_idx_raw = pointnet2_utils.furthest_point_sample(points_raw, point_all)
            else:
                from model import fps
                fps_idx_raw = None
            
            local_pred = []
            
            for kk in range(times):
                if fps_idx_raw is not None:
                    fps_idx = fps_idx_raw[:, np.random.choice(point_all, npoints, False)]
                    points = pointnet2_utils.gather_operation(
                        points_raw.transpose(1, 2).contiguous(), fps_idx
                    ).transpose(1, 2).contiguous()
                else:
                    from model import fps
                    points = fps(points_raw, npoints)
                
                points = test_transforms(points)
                
                logits = base_model(points)
                target = label.view(-1)
                
                local_pred.append(logits.detach().unsqueeze(0))
            
            pred = torch.cat(local_pred, dim=0).mean(0)
            _, pred_choice = torch.max(pred, -1)
            
            test_pred.append(pred_choice)
            test_label.append(target.detach())
    
    test_pred = torch.cat(test_pred, dim=0)
    test_label = torch.cat(test_label, dim=0)
    
    if args.distributed:
        test_pred = gather_tensor(test_pred, args)
        test_label = gather_tensor(test_label, args)
    
    acc = (test_pred == test_label).sum() / float(test_label.size(0)) * 100.
    
    if logger:
        logger.info(f'[Validation_vote] EPOCH: {epoch}  acc_vote = {acc:.4f}')
    
    if args.distributed:
        torch.cuda.synchronize()
    
    if val_writer is not None:
        val_writer.add_scalar('Metric/ACC_vote', acc, epoch)
    
    return Acc_Metric(acc)


def run_net(args, config, train_writer=None, val_writer=None):
    """
    Main finetuning loop.
    
    Args:
        args: Command line arguments
        config: Training configuration
        train_writer: TensorBoard writer for training
        val_writer: TensorBoard writer for validation
    """
    import logging
    logger = logging.getLogger(args.log_name)
    
    from 04_transforms import PointcloudScaleAndTranslate
    train_transforms = PointcloudScaleAndTranslate()
    
    # Build datasets
    (train_sampler, train_dataloader), (_, test_dataloader) = \
        build_dataset(args, config.dataset.train), build_dataset(args, config.dataset.val)
    
    # Build model
    base_model = build_model(config.model)
    
    # Initialize metrics
    start_epoch = 0
    best_metrics = Acc_Metric(0.)
    best_metrics_vote = Acc_Metric(0.)
    metrics = Acc_Metric(0.)
    
    # Resume or load pretrained
    if args.resume:
        start_epoch, best_metric = resume_model(base_model, args, logger)
        best_metrics = Acc_Metric(best_metrics)
    else:
        if args.ckpts is not None:
            base_model.load_model_from_ckpt(args.ckpts)
        else:
            logger.info('Training from scratch')
    
    if args.use_gpu:
        base_model.to(args.local_rank)
    
    # Distributed training
    if args.distributed:
        if args.sync_bn:
            base_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(base_model)
            logger.info('Using Synchronized BatchNorm')
        base_model = nn.parallel.DistributedDataParallel(
            base_model, 
            device_ids=[args.local_rank % torch.cuda.device_count()]
        )
        logger.info('Using Distributed Data Parallel')
    else:
        logger.info('Using Data Parallel')
        base_model = nn.DataParallel(base_model).cuda()
    
    # Build optimizer and scheduler
    optimizer, scheduler = build_optimizer_scheduler(base_model, config)
    
    if args.resume:
        resume_optimizer(optimizer, args, logger)
    
    # Import pointnet2 for FPS
    try:
        from pointnet2_ops import pointnet2_utils
        HAS_POINTNET2 = True
    except:
        HAS_POINTNET2 = False
    
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
        losses = AverageMeter(['loss', 'acc'])
        
        num_iter = 0
        n_batches = len(train_dataloader)
        npoints = config.npoints
        
        for idx, (taxonomy_ids, model_ids, data) in enumerate(train_dataloader):
            num_iter += 1
            n_itr = epoch * n_batches + idx
            
            data_time.update(time.time() - batch_start_time)
            
            points = data[0].cuda()
            label = data[1].cuda()
            
            # Sample points
            if npoints == 1024:
                point_all = 1200
            elif npoints == 2048:
                point_all = 2400
            elif npoints == 4096:
                point_all = 4800
            elif npoints == 8192:
                point_all = 8192
            else:
                point_all = int(npoints * 1.2)
            
            if points.size(1) < point_all:
                point_all = points.size(1)
            
            if HAS_POINTNET2:
                fps_idx = pointnet2_utils.furthest_point_sample(points, point_all)
                fps_idx = fps_idx[:, np.random.choice(point_all, npoints, False)]
                points = pointnet2_utils.gather_operation(
                    points.transpose(1, 2).contiguous(), fps_idx
                ).transpose(1, 2).contiguous()
            else:
                from model import fps
                points = fps(points, npoints)
            
            # Apply transforms
            points = train_transforms(points)
            
            # Forward pass
            ret = base_model(points)
            
            loss, acc = base_model.module.get_loss_acc(ret, label)
            
            loss.backward()
            
            # Optimizer step
            if num_iter == config.step_per_update:
                if config.get('grad_norm_clip') is not None:
                    torch.nn.utils.clip_grad_norm_(base_model.parameters(), config.grad_norm_clip, norm_type=2)
                num_iter = 0
                optimizer.step()
                base_model.zero_grad()
            
            # Update metrics
            if args.distributed:
                loss = reduce_tensor(loss, args)
                acc = reduce_tensor(acc, args)
                losses.update([loss.item(), acc.item()])
            else:
                losses.update([loss.item(), acc.item()])
            
            if args.distributed:
                torch.cuda.synchronize()
            
            # TensorBoard logging
            if train_writer is not None:
                train_writer.add_scalar('Loss/Batch/Loss', loss.item(), n_itr)
                train_writer.add_scalar('Loss/Batch/TrainAcc', acc.item(), n_itr)
                train_writer.add_scalar('Loss/Batch/LR', optimizer.param_groups[0]['lr'], n_itr)
            
            batch_time.update(time.time() - batch_start_time)
            batch_start_time = time.time()
        
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
            f'Loss = {losses.avgs[0]:.4f} Acc = {losses.avgs[1]:.2f} '
            f'lr = {optimizer.param_groups[0]["lr"]:.6f}'
        )
        
        # Validation
        if epoch % args.val_freq == 0 and epoch != 0:
            metrics = validate(base_model, test_dataloader, epoch, val_writer, args, config, logger)
            
            better = metrics.better_than(best_metrics)
            
            if better:
                best_metrics = metrics
                save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, 'ckpt-best', args, logger)
                logger.info("-" * 80)
            
            # Vote validation for high accuracy
            if args.vote:
                if metrics.acc > 92.1 or (better and metrics.acc > 91):
                    metrics_vote = validate_vote(base_model, test_dataloader, epoch, val_writer, args, config, logger)
                    if metrics_vote.better_than(best_metrics_vote):
                        best_metrics_vote = metrics_vote
                        logger.info("*" * 80)
                        save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics_vote, 'ckpt-best_vote', args, logger)
        
        # Save last checkpoint
        save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, 'ckpt-last', args, logger)
    
    # Close writers
    if train_writer is not None:
        train_writer.close()
    if val_writer is not None:
        val_writer.close()


def test_net(args, config):
    """Test model on test set."""
    import logging
    logger = logging.getLogger(args.log_name)
    
    logger.info('Tester start...')
    
    _, test_dataloader = build_dataset(args, config.dataset.test)
    
    base_model = build_model(config.model)
    load_model(base_model, args.ckpts, logger)
    
    if args.use_gpu:
        base_model.to(args.local_rank)
    
    if args.distributed:
        raise NotImplementedError("Distributed testing not implemented")
    
    test(base_model, test_dataloader, args, config, logger)


def test(base_model, test_dataloader, args, config, logger=None):
    """Run test evaluation."""
    base_model.eval()
    
    test_pred = []
    test_label = []
    npoints = config.npoints
    
    with torch.no_grad():
        for idx, (taxonomy_ids, model_ids, data) in enumerate(test_dataloader):
            points = data[0].cuda()
            label = data[1].cuda()
            
            from model import fps
            points = fps(points, npoints)
            
            logits = base_model(points)
            target = label.view(-1)
            
            pred = logits.argmax(-1).view(-1)
            
            test_pred.append(pred.detach())
            test_label.append(target.detach())
    
    test_pred = torch.cat(test_pred, dim=0)
    test_label = torch.cat(test_label, dim=0)
    
    acc = (test_pred == test_label).sum() / float(test_label.size(0)) * 100.
    
    if logger:
        logger.info(f'[TEST] acc = {acc:.4f}')
    
    # Vote testing
    if logger:
        logger.info('[TEST_VOTE]')
    
    best_acc = 0.
    for t in range(1, 10):
        this_acc = test_vote(base_model, test_dataloader, 1, None, args, config, logger, times=10)
        if best_acc < this_acc:
            best_acc = this_acc
        if logger:
            logger.info(f'[TEST_VOTE_time {t}] acc = {this_acc:.4f}, best acc = {best_acc:.4f}')
    
    if logger:
        logger.info(f'[TEST_VOTE] acc = {best_acc:.4f}')


def test_vote(base_model, test_dataloader, epoch, val_writer, args, config, logger=None, times=10):
    """Test with voting."""
    base_model.eval()
    
    from 04_transforms import PointcloudScaleAndTranslate
    test_transforms = PointcloudScaleAndTranslate()
    
    test_pred = []
    test_label = []
    npoints = config.npoints
    
    try:
        from pointnet2_ops import pointnet2_utils
        HAS_POINTNET2 = True
    except:
        HAS_POINTNET2 = False
    
    with torch.no_grad():
        for idx, (taxonomy_ids, model_ids, data) in enumerate(test_dataloader):
            points_raw = data[0].cuda()
            label = data[1].cuda()
            
            if npoints == 1024:
                point_all = 1200
            elif npoints == 4096:
                point_all = 4800
            elif npoints == 8192:
                point_all = 8192
            else:
                point_all = int(npoints * 1.2)
            
            if points_raw.size(1) < point_all:
                point_all = points_raw.size(1)
            
            if HAS_POINTNET2:
                fps_idx_raw = pointnet2_utils.furthest_point_sample(points_raw, point_all)
            else:
                fps_idx_raw = None
            
            local_pred = []
            
            for kk in range(times):
                if fps_idx_raw is not None:
                    fps_idx = fps_idx_raw[:, np.random.choice(point_all, npoints, False)]
                    points = pointnet2_utils.gather_operation(
                        points_raw.transpose(1, 2).contiguous(), fps_idx
                    ).transpose(1, 2).contiguous()
                else:
                    from model import fps
                    points = fps(points_raw, npoints)
                
                points = test_transforms(points)
                
                logits = base_model(points)
                target = label.view(-1)
                
                local_pred.append(logits.detach().unsqueeze(0))
            
            pred = torch.cat(local_pred, dim=0).mean(0)
            _, pred_choice = torch.max(pred, -1)
            
            test_pred.append(pred_choice)
            test_label.append(target.detach())
    
    test_pred = torch.cat(test_pred, dim=0)
    test_label = torch.cat(test_label, dim=0)
    
    acc = (test_pred == test_label).sum() / float(test_label.size(0)) * 100.
    
    if val_writer is not None:
        val_writer.add_scalar('Metric/ACC_vote', acc, epoch)
    
    return acc


if __name__ == '__main__':
    print("Finetuning runner module loaded successfully!")
    print("Use 09_main.py to start training.")
