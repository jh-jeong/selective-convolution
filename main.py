from __future__ import division

import sys
import json
import os
import time
import math

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim

import models
from datasets import get_dataset
from utils import Logger
from utils import AverageMeter
from utils import save_checkpoint


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cudnn.benchmark = True


def update_learning_rate(optimizer, epoch, args, cur_batch, num_batches):
    lr_init = args.get('lr_init', 0.1)
    num_epochs = args['num_epochs']

    T_total = num_epochs * num_batches
    T_cur = (epoch % num_epochs) * num_batches + cur_batch
    lr = 0.5 * lr_init * (1 + math.cos(math.pi * T_cur / T_total))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def error_k(output, target, ks=(1,)):
    """Computes the precision@k for the specified values of k"""
    max_k = max(ks)
    batch_size = target.size(0)

    _, pred = output.topk(max_k, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    results = []
    for k in ks:
        correct_k = correct[:k].view(-1).float().sum(0)
        results.append(100.0 - correct_k.mul_(100.0 / batch_size))
    return results


def train(epoch, model, criterion, optimizer, dataloader, logger, args):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    error_top1 = AverageMeter()
    error_top5 = AverageMeter()

    # Switch to train mode
    model.train()
    num_batches = len(dataloader)
    check = time.time()
    for n, (images, labels) in enumerate(dataloader):
        images, labels = images.to(device), labels.to(device)
        data_time.update(time.time() - check)

        lr = update_learning_rate(optimizer, epoch, args, n, num_batches)

        check = time.time()
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Compute gradient and do SGD step
        model.zero_grad()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Measure elapsed time
        batch_time.update(time.time() - check)

        # Measure accuracy and record loss
        top1, top5 = error_k(outputs.data, labels, ks=(1, 5))
        batch_size = images.size(0)
        losses.update(loss.item(), batch_size)
        error_top1.update(top1.item(), batch_size)
        error_top5.update(top5.item(), batch_size)

        if n % 10 == 0:
            logger.log('[Epoch %3d; %3d] [Time %.3f] [Data %.3f] [Loss %f] [LR %.3f]' %
                       (epoch, n, batch_time.value, data_time.value, losses.value, lr))
        check = time.time()

    logger.log('[DONE] [Time %.3f] [Data %.3f] [Loss %f] [Train@1 %.3f] [Train@5 %.3f]' %
               (batch_time.average, data_time.average, losses.average,
                error_top1.average, error_top5.average))
    logger.scalar_summary('loss', losses.average, epoch)
    logger.scalar_summary('train_1', error_top1.average, epoch)
    logger.scalar_summary('batch_time', batch_time.average, epoch)


def test(epoch, model, criterion, dataloader, logger=None):
    batch_time = AverageMeter()
    losses = AverageMeter()
    error_top1 = AverageMeter()
    error_top5 = AverageMeter()

    # Switch to eval mode
    model.eval()
    with torch.no_grad():
        for n, (images, labels) in enumerate(dataloader):
            images, labels = images.to(device), labels.to(device)

            check = time.time()
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Measure elapsed time
            batch_time.update(time.time() - check)

            # Measure accuracy and record loss
            top1, top5 = error_k(outputs.data, labels, ks=(1, 5))

            batch_size = images.size(0)
            losses.update(loss.item(), batch_size)
            error_top1.update(top1.item(), batch_size)
            error_top5.update(top5.item(), batch_size)

            if n % 10 == 0:
                if logger:
                    logger.log('[Test %3d] [Time %.3f] [Loss %f] [Test@1 %.3f] [Test@5 %.3f]' %
                               (n, batch_time.value, losses.value, error_top1.value, error_top5.value))
                else:
                    print('[Test %3d] [Time %.3f] [Loss %f] [Test@1 %.3f] [Test@5 %.3f]' %
                          (n, batch_time.value, losses.value, error_top1.value, error_top5.value))

    if logger:
        logger.log(' * [Error@1 %.3f] [Error@5 %.3f] [Loss %.3f]' %
                   (error_top1.average, error_top5.average, losses.average))
        logger.scalar_summary('error_1', error_top1.average, epoch)
        logger.scalar_summary('error_5', error_top5.average, epoch)
        logger.scalar_summary('loss_test', losses.average, epoch)

    return error_top1.average


def main(args, fn):
    logger = Logger(fn)

    hparams = args['model_hparams']
    if args['dataset'] in ['cifar10', 'fmnist']:
        hparams['n_classes'] = 10
    elif args['dataset'] == 'cifar100':
        hparams['n_classes'] = 100
    elif args['dataset'] == 'tinyimg':
        hparams['n_classes'] = 200
    elif args['dataset'] == 'imagenet':
        hparams['n_classes'] = 1000
    logger.log(args)
    hparams['dataset'] = args['dataset']

    model = models.__dict__[args['model']](hparams)
    logger.log(model)

    if torch.cuda.is_available():
        n_gpus = torch.cuda.device_count()
        if n_gpus > 1:
            logger.log('Multi-GPU mode: using %d GPUs for training.' % n_gpus)
            model = nn.DataParallel(model).cuda()
        else:
            logger.log('Single-GPU mode.')
            model = model.cuda()
    else:
        n_gpus = 0

    # Configure parameters to optimize
    pg_normal = []
    pg_small = []

    for p in model.parameters():
        if not p.requires_grad:
            continue
        elif hasattr(p, 'wd_small') and p.wd_small:
            pg_small.append(p)
        else:
            pg_normal.append(p)

    params = [
        {'params': pg_normal, 'weight_decay': 1e-4},
        {'params': pg_small, 'weight_decay': 1e-5}
    ]

    # Define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(params,
                                lr=args.get('lr_init', 0.1),
                                momentum=args.get('momentum', 0.9),
                                nesterov=True)

    train_set, test_set = get_dataset(args['dataset'])
    n_workers = max(8*n_gpus, 4)
    train_loader = torch.utils.data.DataLoader(train_set,
                                               shuffle=True,
                                               pin_memory=True,
                                               batch_size=args['batch_size'],
                                               num_workers=n_workers)
    test_loader = torch.utils.data.DataLoader(test_set,
                                              shuffle=False,
                                              pin_memory=True,
                                              batch_size=args['batch_size'],
                                              num_workers=n_workers)

    best = 100.0
    for epoch in range(args['num_epochs']):
        train(epoch, model, criterion, optimizer, train_loader, logger, args)
        error = test(epoch, model, criterion, test_loader, logger)

        # Perform dealloc/realloc for SelectiveConv2d modules
        for m in model.modules():
            if type(m).__name__ in ['SelectiveConv2d']:
                if epoch < 0.5 * args['num_epochs']:
                    m.dealloc()
                    m.realloc()

        if isinstance(model, nn.DataParallel):
            save_states = model.module.state_dict()
        else:
            save_states = model.state_dict()

        is_best = (best > error)
        if is_best:
            best = error
        save_checkpoint(epoch, args, best,
                        save_states, optimizer.state_dict(),
                        logger.logdir, is_best)
        logger.scalar_summary('best', best, epoch)
        logger.log('[Epoch %3d] [Test %5.2f] [Best %5.2f]' % (epoch, error, best))


if __name__ == '__main__':
    config_path = sys.argv[1]
    with open(config_path) as file:
        config = file.read()
        print(config)
        args = json.loads(config)
    config_fn = os.path.split(config_path)[-1].split('.')[0]

    main(args, config_fn)

