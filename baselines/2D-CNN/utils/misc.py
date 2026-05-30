import random
import os
import numpy as np
import torch

from torchvision import transforms
from timm.data import create_transform


def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def build_transform(is_train, input_size):
    if is_train:
        transform = create_transform(
            input_size=input_size,
            is_training=True,
            color_jitter=0.4,
            mean=[0.5],
            std=[0.5],
            auto_augment='rand-m9-mstd0.5-inc1',
            interpolation='bicubic',
            re_prob=0.25,
            re_mode='pixel',
            re_count=1,
        )

        transform.transforms[0] = transforms.RandomCrop(input_size, padding=4)
        return transform

    t = [transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
    return transforms.Compose(t)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n

    @property
    def avg(self):
        return self.sum / self.count

    def __str__(self):
        fmtstr = '{} {' + self.fmt + '} ({' + self.fmt + '})'
        return fmtstr.format(self.name, self.val, self.avg)



