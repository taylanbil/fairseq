#!/usr/bin/env python3 -u
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.
"""
Train a new model on one or across multiple GPUs.
"""

import collections
import math
import os
import random
from datetime import datetime

import torch
import torch_xla
import torch_xla.distributed.data_parallel as dp
import torch_xla.utils.utils as xu
import torch_xla.core.xla_model as xm

from fairseq import checkpoint_utils, distributed_utils, options, progress_bar, tasks, utils
from fairseq.data import iterators
from fairseq.trainer import Trainer
from fairseq.meters import AverageMeter, StopwatchMeter


def initialize_loader_for_epoch(args, epoch_itr):
    # Update parameters every N batches
    if epoch_itr.epoch <= len(args.update_freq):
        update_freq = args.update_freq[epoch_itr.epoch - 1]
    else:
        update_freq = args.update_freq[-1]

    # Initialize data iterator
    itr = epoch_itr.next_epoch_itr(
          fix_batches_to_gpus=False, shuffle=(epoch_itr.epoch >= args.curriculum))
    itr = iterators.GroupedIterator(itr, update_freq)
    progress = progress_bar.build_progress_bar(
          args, itr, epoch_itr.epoch, prefix='training', no_progress_bar='simple')
    return progress


def print_model_criterion(model, criterion, args):
      print(model)
      print('| model {}, criterion {}'.format(args.arch,
                                              criterion.__class__.__name__))
      print('| num. model params: {} (num. trained: {})'.format(
          sum(p.numel() for p in model.parameters()),
          sum(p.numel() for p in model.parameters() if p.requires_grad),
      ))


def main(args, init_distributed=False):
    utils.import_user_module(args)

    assert args.max_tokens is not None or args.max_sentences is not None, \
        'Must specify batch size either with --max-tokens or --max-sentences'

    # Initialize CUDA and distributed training
    if torch.cuda.is_available() and not args.cpu:
        torch.cuda.set_device(args.device_id)
    torch.manual_seed(args.seed)
    if init_distributed:
        args.distributed_rank = distributed_utils.distributed_init(args)

    # Print args
    print(args)

    # Setup task, e.g., translation, language modeling, etc.
    task = tasks.setup_task(args)

    # Load valid dataset (we load training data below, based on the latest checkpoint)
    for valid_sub_split in args.valid_subset.split(','):
        task.load_dataset(valid_sub_split, combine=False, epoch=0)

    # Build model and criterion
    model = task.build_model(args)
    criterion = task.build_criterion(args)
    print_model_criterion(model, criterion, args)

    # Build trainer
    trainer = Trainer(args, task, model, criterion)
    print('| training on {} GPUs'.format(args.distributed_world_size))
    print('| max tokens per GPU = {} and max sentences per GPU = {}'.format(
        args.max_tokens,
        args.max_sentences,
    ))

    # Load the latest checkpoint if one is available and restore the
    # corresponding train iterator
    extra_state, epoch_itr = checkpoint_utils.load_checkpoint(args, trainer)

    # Train until the learning rate gets too small
    max_epoch = args.max_epoch or math.inf
    max_update = args.max_update or math.inf
    lr = trainer.get_lr()
    train_meter = StopwatchMeter()
    train_meter.start()
    valid_losses = [None]
    valid_subsets = args.valid_subset.split(',')
    while lr > args.min_lr and epoch_itr.epoch < max_epoch and trainer.get_num_updates() < max_update:
        # train for one epoch
        train(args, trainer, task, epoch_itr)

        if not args.disable_validation and epoch_itr.epoch % args.validate_interval == 0:
            valid_losses = validate(args, trainer, task, epoch_itr, valid_subsets)
        else:
            valid_losses = [None]

        # only use first validation loss to update the learning rate
        lr = trainer.lr_step(epoch_itr.epoch, valid_losses[0])

        # save checkpoint
        if epoch_itr.epoch % args.save_interval == 0:
            checkpoint_utils.save_checkpoint(args, trainer, epoch_itr, valid_losses[0])

        if ':' in getattr(args, 'data', ''):
            # sharded data: get train iterator for next epoch
            epoch_itr = trainer.get_train_iterator(epoch_itr.epoch)
    train_meter.stop()
    print('| done training in {:.1f} seconds'.format(train_meter.sum))


def train(args, trainer, task, epoch_itr):
    """Train the model for one epoch."""
    progress = initialize_loader_for_epoch(args, epoch_itr)
    extra_meters = collections.defaultdict(lambda: AverageMeter())
    valid_subsets = args.valid_subset.split(',')
    max_update = args.max_update or math.inf
    for i, samples in enumerate(progress, start=epoch_itr.iterations_in_epoch):
        log_output = trainer.train_step(samples)
        if log_output is None:
            continue

        # log mid-epoch stats
        stats = get_training_stats(trainer)
        for k, v in log_output.items():
            if k in ['loss', 'nll_loss', 'ntokens', 'nsentences', 'sample_size']:
                continue  # these are already logged above
            if 'loss' in k:
                extra_meters[k].update(v, log_output['sample_size'])
            else:
                extra_meters[k].update(v)
            stats[k] = extra_meters[k].avg
        progress.log(stats, tag='train', step=stats['num_updates'])

        # ignore the first mini-batch in words-per-second calculation
        if i == 0:
            trainer.get_meter('wps').reset()

        num_updates = trainer.get_num_updates()
        if (
            not args.disable_validation
            and args.save_interval_updates > 0
            and num_updates % args.save_interval_updates == 0
            and num_updates > 0
        ):
            valid_losses = validate(args, trainer, task, epoch_itr, valid_subsets)
            checkpoint_utils.save_checkpoint(args, trainer, epoch_itr, valid_losses[0])

        if num_updates >= max_update:
            break

    # log end-of-epoch stats
    stats = get_training_stats(trainer)
    for k, meter in extra_meters.items():
        stats[k] = meter.avg
    progress.print(stats, tag='train', step=stats['num_updates'])

    # reset training meters
    for k in [
        'train_loss', 'train_nll_loss', 'wps', 'ups', 'wpb', 'bsz', 'gnorm', 'clip',
    ]:
        meter = trainer.get_meter(k)
        if meter is not None:
            meter.reset()


def get_training_stats(trainer):
    stats = collections.OrderedDict()
    stats['loss'] = trainer.get_meter('train_loss')
    if trainer.get_meter('train_nll_loss').count > 0:
        nll_loss = trainer.get_meter('train_nll_loss')
        stats['nll_loss'] = nll_loss
    else:
        nll_loss = trainer.get_meter('train_loss')
    stats['ppl'] = utils.get_perplexity(nll_loss.avg)
    stats['wps'] = trainer.get_meter('wps')
    stats['ups'] = trainer.get_meter('ups')
    stats['wpb'] = trainer.get_meter('wpb')
    stats['bsz'] = trainer.get_meter('bsz')
    stats['num_updates'] = trainer.get_num_updates()
    stats['lr'] = trainer.get_lr()
    stats['gnorm'] = trainer.get_meter('gnorm')
    stats['clip'] = trainer.get_meter('clip')
    stats['oom'] = trainer.get_meter('oom')
    if trainer.get_meter('loss_scale') is not None:
        stats['loss_scale'] = trainer.get_meter('loss_scale')
    stats['wall'] = round(trainer.get_meter('wall').elapsed_time)
    stats['train_wall'] = trainer.get_meter('train_wall')
    return stats


def validate(args, trainer, task, epoch_itr, subsets):
    """Evaluate the model on the validation set(s) and return the losses."""
    valid_losses = []
    for subset in subsets:
        # Initialize data iterator
        itr = task.get_batch_iterator(
            dataset=task.dataset(subset),
            max_tokens=args.max_tokens_valid,
            max_sentences=args.max_sentences_valid,
            max_positions=utils.resolve_max_positions(
                task.max_positions(),
                trainer.get_model().max_positions(),
            ),
            ignore_invalid_inputs=args.skip_invalid_size_inputs_valid_test,
            required_batch_size_multiple=args.required_batch_size_multiple,
            seed=args.seed,
            num_shards=args.distributed_world_size,
            shard_id=args.distributed_rank,
            num_workers=args.num_workers,
        ).next_epoch_itr(shuffle=False)
        progress = progress_bar.build_progress_bar(
            args, itr, epoch_itr.epoch,
            prefix='valid on \'{}\' subset'.format(subset),
            no_progress_bar='simple'
        )

        # reset validation loss meters
        for k in ['valid_loss', 'valid_nll_loss']:
            meter = trainer.get_meter(k)
            if meter is not None:
                meter.reset()
        extra_meters = collections.defaultdict(lambda: AverageMeter())

        for sample in progress:
            log_output = trainer.valid_step(sample)

            for k, v in log_output.items():
                if k in ['loss', 'nll_loss', 'ntokens', 'nsentences', 'sample_size']:
                    continue
                extra_meters[k].update(v)

        # log validation stats
        stats = get_valid_stats(trainer)
        for k, meter in extra_meters.items():
            stats[k] = meter.avg
        progress.print(stats, tag=subset, step=trainer.get_num_updates())

        valid_losses.append(stats[args.best_checkpoint_metric].avg)
    return valid_losses


def get_valid_stats(trainer):
    stats = collections.OrderedDict()
    stats['loss'] = trainer.get_meter('valid_loss')
    if trainer.get_meter('valid_nll_loss').count > 0:
        nll_loss = trainer.get_meter('valid_nll_loss')
        stats['nll_loss'] = nll_loss
    else:
        nll_loss = stats['loss']
    stats['ppl'] = utils.get_perplexity(nll_loss.avg)
    stats['num_updates'] = trainer.get_num_updates()
    if hasattr(checkpoint_utils.save_checkpoint, 'best'):
        stats['best_loss'] = min(
            checkpoint_utils.save_checkpoint.best, stats['loss'].avg)
    return stats


def distributed_main(i, args, start_rank=0):
    args.device_id = i
    if args.distributed_rank is None:  # torch.multiprocessing.spawn
        args.distributed_rank = start_rank + i
    main(args, init_distributed=True)

def parse_input_shapes(input_shapes_arg):
    input_shapes = (
        shape.replace('*', 'x').split('x') for shape in input_shapes_arg)
    input_shapes = [list(map(int, shape)) for shape in input_shapes]
    if len(input_shapes) == 1:
        return input_shapes
    input_shapes.sort(key=lambda shape: shape[1])
    errmsg = (
        'Invalid --input_shapes. Batch sizes (dimension 1) need to increase as '
        'num_tokens (dimension 2) decrease. e.g. 16x128 32x64 64x32'
    )
    assert all(
         shape1[0] > shape2[0]
         for shape1, shape2 in zip(input_shapes, input_shapes[1:])), errmsg
    return input_shapes


def main_tpu(args):

    def now():
        return datetime.now().strftime('%H:%M:%S')

    def log_step(step_type, device, step, log_output=None, tracker=None):
        msg = '{}/ {}, device {}, step {}'.format(
            step_type, now(), device, step
        )
        if tracker:
            rates = tracker.rate(), tracker.global_rate()
            msg += ', Rate={:.2f}, GlobalRate={:.2f}'.format(*rates)
        if log_output:
            msg += ', loss={:.4f}, nll_loss={:.4f}'.format(
                log_output['loss'].item(), log_output['nll_loss'].item()
            )
        return msg

    def prepare_task(args, devices):
        # Setup task, e.g., translation, language modeling, etc.
        task = tasks.setup_task(args)

        # Load valid dataset (we load training data below, based on the latest checkpoint)
        for valid_sub_split in args.valid_subset.split(','):
            task.load_dataset(valid_sub_split, combine=True, epoch=0)

        # Build models and criteria to print some metadata
        model_parallel = dp.DataParallel(
            lambda: task.build_model(args), device_ids=devices)
        model, criterion = task.build_model(args), task.build_criterion(args)
        print_model_criterion(model, criterion, args)
        del model, criterion

        # Build trainers
        trainers = {
            device: Trainer(args, task, model, task.build_criterion(args), xla=True)
            for device, model in zip(model_parallel.devices, model_parallel.models)
        }
        lr = trainers[devices[0]].get_lr()

        # Load the latest checkpoint if one is available and restore the
        # corresponding train iterator
        extra_state, epoch_itr = checkpoint_utils.load_checkpoint(
            args, trainers[devices[0]])
        progress = initialize_loader_for_epoch(args, epoch_itr)
        # FIXME
        checkpoint_utils.save_checkpoint(args, trainers[devices[0]], epoch_itr, None)

        import pdb
        pdb.set_trace()
        if extra_state is not None:
            # checkpoint detected, load saved model weights to all devices
            xu.eprint(
                'checkpoint detected, device 0 meters need to be '
                're-loaded to device'
            )
            checkpoint_utils.load_checkpoint_tpu(args, trainers, devices[0])
        valid_subsets = args.valid_subset.split(',')
        return task, trainers, model_parallel, epoch_itr, lr, valid_subsets

    def train_loop_fn(model, loader, device, context):
        trainer = trainers[str(device)]
        stats, log_output = None, None
        tracker = xm.RateTracker()
        for i, samples in loader:
            if i and not (i % args.log_steps):
                print(
                    log_step(
                        'training', device, i,
                        log_output=log_output, tracker=tracker),
                    flush=True,
                )
            log_output = trainer.train_step(samples)
            xm.optimizer_step(trainer.optimizer)
            tracker.add(sum(sample['nsentences'] for sample in samples))
        return tracker

    def valid_loop_fn(model, loader, device, context):
        trainer = trainers[str(device)]
        # reset validation loss meters
        for k in ['valid_loss', 'valid_nll_loss']:
            meter = trainer.get_meter(k)
            if meter is not None:
                meter.reset()
        extra_meters = collections.defaultdict(lambda: AverageMeter())
        for i, sample in loader:
            if not (i % args.log_steps):
                print(log_step('validation', device, i, tracker=None))
            log_output = trainer.valid_step(sample)
            for k, v in log_output.items():
                if k in ['loss', 'nll_loss', 'ntokens', 'nsentences', 'sample_size']:
                    continue
                extra_meters[k].update(v)
        stats = get_valid_stats(trainer)
        for k, meter in extra_meters.items():
            stats[k] = meter.avg
        return stats

    def validate_subset(args, trainers, task, epoch_itr, subset):
        print('Validating the subset "{}"'.format(subset))
        # Initialize data iterator
        itr = task.get_batch_iterator(
            dataset=task.dataset(subset),
            max_tokens=args.max_tokens,
            max_sentences=args.max_sentences_valid,
            max_positions=utils.resolve_max_positions(
                task.max_positions(),
                list(trainers.values())[0].get_model().max_positions(),
            ),
            ignore_invalid_inputs=args.skip_invalid_size_inputs_valid_test,
            required_batch_size_multiple=args.required_batch_size_multiple,
            seed=args.seed,
            num_workers=args.num_workers
        ).next_epoch_itr(shuffle=False)
        progress = progress_bar.build_progress_bar(
            args, itr, epoch_itr.epoch,
            prefix='valid on \'{}\' subset'.format(subset),
            no_progress_bar='simple'
        )
        stats_per_device = model_parallel(valid_loop_fn, progress)
        valid_losses = [stats['loss'].avg for stats in stats_per_device]
        print('validation stats on subset "{}" - {}'.format(subset, now()))
        for stats in stats_per_device:
            progress.print(
                stats, tag=subset, step=trainers['xla:1'].get_num_updates()
            )
        return valid_losses

    def validate_subsets(args, trainers, task, epoch_itr, subsets):
        valid_losses = {
            subset: validate_subset(args, trainers, task, epoch_itr, subset)
            for subset in subsets
        }
        return valid_losses

    def keep_training(lr, epoch_itr, trainers):
        # Train until the learning rate gets too small
        max_epoch = args.max_epoch or math.inf
        max_update = args.max_update or math.inf
        lr = min(trainer.get_lr() for trainer in trainers.values())
        n_updates = max(trainer.get_num_updates() for trainer in trainers.values())
        return ((lr > args.min_lr) and (epoch_itr.epoch < max_epoch) and
            (n_updates < max_update))

    xu.eprint('Args')
    for key, val in args.__dict__.items():
        xu.eprint('\t{} {}'.format(key, val))

    devices = xm.get_xla_supported_devices(max_devices=args.num_cores)
    task, trainers, model_parallel, epoch_itr, lr, valid_subsets = prepare_task(
        args, devices)

    train_meter = StopwatchMeter()
    train_meter.start()
    while keep_training(lr, epoch_itr, trainers):
        # TRAINING
        print('Epoch {} begin {}'.format(epoch_itr.epoch + 1, now()))
        progress = initialize_loader_for_epoch(args, epoch_itr)
        trackers = model_parallel(train_loop_fn, progress)
        print('Epoch {} Training stats:'.format(epoch_itr.epoch))
        for device, trainer in trainers.items():
            stats = get_training_stats(trainer)
            print('device {}'.format(device))
            progress.print(stats, tag=device)
        print('Epoch {} Tracker Rates:'.format(epoch_itr.epoch))
        for tracker in trackers:
            rates = tracker.rate(), tracker.global_rate()
            print('\tRate={:.2f}, GlobalRate={:.2f}'.format(*rates))
        print('Epoch {} end {}'.format(epoch_itr.epoch, now()))
        if args.metrics_debug:
            print(torch_xla._XLAC._xla_metrics_report())

        # VALIDATION
        if not args.disable_validation and epoch_itr.epoch % args.validate_interval == 0:
            valid_losses = validate_subsets(
                args, trainers, task, epoch_itr, valid_subsets
            )

            # only use average first validation loss from the first device
            # to update the learning rate
            vloss = valid_losses[valid_subsets[0]][0].item()
            print('old learning rate: {}'.format(lr))
            lr = trainers[devices[0]].lr_step(epoch_itr.epoch, vloss)
            print('new learning rate: {}'.format(lr))
        else:
            vloss = None

        # save checkpoint
        if epoch_itr.epoch % args.save_interval == 0:
            checkpoint_utils.save_checkpoint(
                args, trainers[devices[0]], epoch_itr, vloss)

        if args.metrics_debug:
            print(torch_xla._XLAC._xla_metrics_report())

    train_meter.stop()
    print('| done training in {:.1f} seconds'.format(train_meter.sum))


def cli_main_gpu(args):
    if args.distributed_init_method is None:
        distributed_utils.infer_init_method(args)

    if args.distributed_init_method is not None:
        # distributed training
        if torch.cuda.device_count() > 1 and not args.distributed_no_spawn:
            start_rank = args.distributed_rank
            args.distributed_rank = None  # assign automatically
            torch.multiprocessing.spawn(
                fn=distributed_main,
                args=(args, start_rank),
                nprocs=torch.cuda.device_count(),
            )
        else:
            distributed_main(args.device_id, args)
    elif args.distributed_world_size > 1:
        # fallback for single node with multiple GPUs
        assert args.distributed_world_size <= torch.cuda.device_count()
        port = random.randint(10000, 20000)
        args.distributed_init_method = 'tcp://localhost:{port}'.format(port=port)
        args.distributed_rank = None  # set based on device id
        if max(args.update_freq) > 1 and args.ddp_backend != 'no_c10d':
            print('| NOTE: you may get better performance with: --ddp-backend=no_c10d')
        torch.multiprocessing.spawn(
            fn=distributed_main,
            args=(args, ),
            nprocs=args.distributed_world_size,
        )
    else:
        # single GPU training
        main(args)


def get_args():
    parser = options.get_training_parser()
    # TPU: need to control certain flags here.
    # e.g. parallelization needs to be suppressed and deferred to torch_xla flags
    # e.g. input tensor shapes need to be controlled via --input_shapes
    parser.add_argument(
        '--input_shapes',
        nargs='*',
        default=None,
        help=(
            'This is used to specify batches and pad lengths. Ex: '
            '`--input_shapes 256x32 512x16` will produce batches w/ 256 '
            'sentences padded to length 32, or 512 sentences padded to length '
            '16. Including too many input shapes will cause graph recompiles and'
            ' degrade performance. On the other extreme, including 1 shape may '
            'waste a ton of flops, since batches may contain a lot of pad '
            'indices on average. Note that the max pad length in this arg will '
            'be used as `--max-source-positions`'))
    parser.add_argument('--log_steps', type=int, default=20)
    parser.add_argument('--num_cores', type=int, default=8)
    parser.add_argument('--metrics_debug', action='store_true')
    parser.add_argument('--use_gpu', action='store_true')
    args = options.parse_args_and_arch(parser)
    return args


def adjust_args_tpu(args):
    if args.fp16:
        raise RuntimeError(
            '--fp16 was provided, this is controlled by env var XLA_USE_BF16')
    if args.distributed_world_size > 1:
        xu.eprint('suppressing "distributed_world_size"')
        args.distributed_world_size = 1
    if args.distributed_init_method is not None:
        xu.eprint('suppressing "distributed_init_method"')
        args.distributed_init_method = None
    if args.input_shapes is None:
        raise RuntimeError(
            'Please specify batches and pad lengths using '
            '--input_shapes. Ex: `--input_shapes 256x32 512x16` .'
            'Please refer to the description of the --input_shape'
            ' arg in --help'
        )
    gpu_input_shape_args = ['max_sentences', 'max_sentences_valid', 'max_tokens']
    nonnull_gpu_input_shape_args = [
        arg for arg in gpu_input_shape_args if getattr(args, arg) is not None
    ]
    if nonnull_gpu_input_shape_args:
      errmsg = (
          'On TPUs, please control input shapes '
          'using `--input_shapes`. Any non-null arg in {} will trigger'
          ' this error.'
      ).format(gpu_input_shape_args)
      raise RuntimeError(errmsg)

    args.input_shapes = parse_input_shapes(args.input_shapes)
    # XXX (taylanbil): do we ever have more than 2 dimensions in fairseq?
    args.max_source_positions = args.input_shapes[-1][1]
    return args


def cli_main():
    args = get_args()
    if args.use_gpu:
        cli_main_gpu(args)
    else:
        args = adjust_args_tpu(args)
        # From here on out we are in TPU context
        if xu.getenv_as('XLA_USE_BF16', bool, False):
            xu.eprint(
                'WARNING: bfloat16 is enabled. Note that fairseq meters such as '
                'loss will accumulate the numerator, and increment the denominator. '
                'Due to lack of precision in higher numbers in bfloat16, these '
                'meters will report invalid values after a while.')

        main_tpu(args)


if __name__ == '__main__':
    cli_main()
