import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import random_split
# from torch.utils.tensorboard import SummaryWriter

import timm

from timm.models.layers import trunc_normal_
from timm.data.mixup import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy

import os

from torchvision import datasets, transforms
from engine import ViTDataSet

import util.lr_decay as lrd
import util.misc as misc
from util.pos_embed import interpolate_pos_embed
from util.misc import NativeScalerWithGradNormCount as NativeScaler

import models_YaTC

from engine import train_one_epoch, evaluate


def get_args_parser():
    parser = argparse.ArgumentParser('YaTC fine-tuning for traffic classification', add_help=False)
    # 64
    parser.add_argument('--batch_size', default=256, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=30, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--model', default='TraFormer_YaTC', type=str, metavar='MODEL',
                        help='Name of model to train')

    parser.add_argument('--input_size', default=40, type=int,
                        help='images input size')

    parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')

    # Optimizer parameters
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=2e-5, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=2e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--layer_decay', type=float, default=0.75,
                        help='layer-wise lr decay from ELECTRA/BEiT')

    parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')
#20
    parser.add_argument('--warmup_epochs', type=int, default=20, metavar='N',
                        help='epochs to warmup LR')

    # Augmentation parameters
    parser.add_argument('--color_jitter', type=float, default=None, metavar='PCT',
                        help='Color jitter factor (enabled only when not using Auto/RandAug)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.1,
                        help='Label smoothing (default: 0.1)')

    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')

    # * Mixup params
    parser.add_argument('--mixup', type=float, default=0,
                        help='mixup alpha, mixup enabled if > 0.')
    parser.add_argument('--cutmix', type=float, default=0,
                        help='cutmix alpha, cutmix enabled if > 0.')
    parser.add_argument('--cutmix_minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup_prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup_switch_prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup_mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')
    parser.add_argument('--finetune', default='./output_dir/pretrained_model.pth',
                        help='finetune from checkpoint')
    parser.add_argument('--output_dir', default='./checkpoints/finetune_model.pth',
                        help='finetune from checkpoint')
    parser.add_argument('--train_path', type=str, default='datasets/finetune_dataset_ctu_train.tsv',
                        help='dataset path')
    parser.add_argument('--val_path',type=str, default='datasets/finetune_dataset_ctu_val.tsv',
                        help='dataset path')
    parser.add_argument('--test_path', type=str, default='datasets/finetune_dataset_ctu_test.tsv',
                        help='dataset path')
    parser.add_argument('--nb_classes', default=8, type=int,
                        help='number of the classification types')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true',
                        help='Perform evaluation only')
    parser.add_argument('--dist_eval', action='store_true', default=False,
                        help='Enabling distributed evaluation (recommended during training for faster monitor')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    return parser


def main(args, seed):

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True
    mean = [0.5]
    std = [0.5]

    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    dataset_train = ViTDataSet(args.train_path, transform, is_train=True)
    dataset_val = ViTDataSet(args.val_path, transform)
    dataset_test = ViTDataSet(args.test_path, transform)

    log_writer = None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, shuffle=True,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, shuffle=False,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, shuffle=False,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )

    model = models_YaTC.__dict__[args.model](
        num_classes=args.nb_classes,
        drop_path_rate=args.drop_path,
    )

    if args.finetune and not args.eval:
        checkpoint = torch.load(args.finetune, map_location='cpu')
        print("Load pre-trained checkpoint from: %s" % args.finetune)
        checkpoint_model = checkpoint['model']
        state_dict = model.state_dict()
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        # interpolate position embedding
        interpolate_pos_embed(model, checkpoint_model)

        # load pre-trained model
        msg = model.load_state_dict(checkpoint_model, strict=False)
        print(msg)

        # manually initialize fc layer
        trunc_normal_(model.head.weight, std=2e-5)

    model.to(device)
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params (M): %.2f' % (n_parameters / 1.e6))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()

    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    # build optimizer with layer-wise lr decay (lrd)
    param_groups = lrd.param_groups_lrd(model, args.weight_decay,
                                        no_weight_decay_list=model.no_weight_decay(),
                                        layer_decay=args.layer_decay
                                        )
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr)
    loss_scaler = NativeScaler()

    criterion = torch.nn.CrossEntropyLoss()

    if args.eval:
        test_stats = evaluate(data_loader_val, model, device)
        print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
        exit(0)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_accuracy = 0.0
    max_f1 = 0.0

    for epoch in range(args.start_epoch, args.epochs):

        train_stats = train_one_epoch(
            model, criterion, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            args.clip_grad,
            log_writer=log_writer,
            args=args
        )

        test_stats = evaluate(data_loader_val, model, device)
        if test_stats['acc1'] > max_accuracy:
            max_accuracy = test_stats['acc1']
            print(f'Get best acc1 with {max_accuracy}, save to {args.output_dir}')
            torch.save(model.state_dict(), args.output_dir)

        print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.4f}")
        # max_accuracy = max(max_accuracy, test_stats["acc1"])
        print(f"F1 of the network on the {len(dataset_val)} test images: {test_stats['macro_f1']:.4f}")
        max_f1 = max(max_f1, test_stats["macro_f1"])
        print(f'Max Accuracy: {max_accuracy:.4f}')
        print(f'Max F1: {max_f1:.4f}')


    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

    best_model = models_YaTC.__dict__[args.model](
        num_classes=args.nb_classes,
        drop_path_rate=args.drop_path,
    )
    best_model.load_state_dict(torch.load(args.output_dir), strict=False)
    best_model = best_model.to(device)
    test_stats = evaluate(data_loader_test, best_model, device)

    print(f"Accuracy of the network on the {len(dataset_test)} test images: {test_stats['acc1']:.4f}")
    print(f"Precision of the network on the {len(dataset_test)} test images: {test_stats['macro_pre']:.4f}")
    print(f"Recall of the network on the {len(dataset_test)} test images: {test_stats['macro_rec']:.4f}")
    print(f"F1 of the network on the {len(dataset_test)} test images: {test_stats['macro_f1']:.4f}")
    return test_stats['acc1'], test_stats['macro_pre'], test_stats['macro_rec'], test_stats['macro_f1']


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    accuracy_list = []
    precision_list = []
    recall_list = []
    f1_list = []
    for seed in range(3):
        accuracy, precision, recall, f1 = main(args, seed)
        accuracy_list.append(accuracy)
        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)

    print(f"Accuracy: {np.mean(accuracy_list):.4f}+-{np.std(accuracy_list):.4f}")
    print(f"Precision: {np.mean(precision_list):.4f}+-{np.std(precision_list):.4f}")
    print(f"Recall: {np.mean(recall_list):.4f}+-{np.std(recall_list):.4f}")
    print(f"F1-score: {np.mean(f1_list):.4f}+-{np.std(f1_list):.4f}")
