import argparse

import torch
import torch.nn as nn
from dgl.dataloading import GraphDataLoader

from tqdm import tqdm
from dataloader import MixTrafficFlowDataset4DGL
from model import MixTemporalGNN
from optim import GradualWarmupScheduler
from utils import show_time, set_seed, get_device, mix_collate_fn
from config import *

torch.autograd.set_detect_anomaly(True)


def train(opt):
    model = MixTemporalGNN(num_classes=config.NUM_CLASSES, K=opt.K, embedding_size=config.EMBEDDING_SIZE, h_feats=config.H_FEATS,
                           dropout=config.DROPOUT, downstream_dropout=config.DOWNSTREAM_DROPOUT)
    dataset = MixTrafficFlowDataset4DGL(header_path=config.HEADER_TEST_GRAPH_DATA,
                                        payload_path=config.TEST_GRAPH_DATA, is_train=False)
    print(f'dataset length is {len(dataset)}.')
    dataloader = GraphDataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True, collate_fn=mix_collate_fn,
                                 num_workers=num_workers)
    model = model.to(device)
    model.train()
    num_steps = len(dataloader) * config.MAX_EPOCH
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LR, weight_decay=config.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=num_steps - int(num_steps * config.WARM_UP), eta_min=config.LR_MIN)
    warmup_scheduler = GradualWarmupScheduler(optimizer, warmup_iter=int(num_steps * config.WARM_UP), after_scheduler=scheduler)
    warmup_scheduler.step()  # Warm up starts from lr = 0
    criterion = nn.CrossEntropyLoss(label_smoothing=config.LABEL_SMOOTHING)

    for epoch in range(config.MAX_EPOCH):
        num_correct = 0
        num_tests = 0
        loss_all = []
        batch_iter = tqdm(dataloader, desc=f'Processing epoch {epoch:02d}')
        for batch_id, (header_data, payload_data, labels) in enumerate(batch_iter):
            header_data = header_data.to(device, non_blocking=True)
            payload_data = payload_data.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            pred = model(header_data, payload_data, labels)
            loss = criterion(pred, labels)
            batch_iter.set_postfix({f'loss': f'{loss.item(): 6.3f}'})
            loss_all.append(float(loss))
            num_correct += (pred.argmax(1) == labels).sum().item()
            num_tests += len(labels)
            loss /= config.GRADIENT_ACCUMULATION
            loss.backward()
            if ((batch_id + 1) % config.GRADIENT_ACCUMULATION == 0) or (batch_id + 1 == len(dataloader)):
                optimizer.step()
                optimizer.zero_grad()
            warmup_scheduler.step()
            if epoch % 1 == 0:
                print('\n{} In epoch {}, lr: {:.5f}, loss: {:.4f}, acc: {:.4f}'.format(show_time(), epoch, optimizer.param_groups[0]['lr'], float(loss), num_correct / num_tests))

        torch.save(model.state_dict(), config.MIX_MODEL_CHECKPOINT.replace('.pth', f'_epoch{epoch:02d}.pth'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, help="dataset", default='kitsune')
    parser.add_argument("--cuda", type=str, help="cuda", default='cuda')
    parser.add_argument("--num_workers", type=int, help="num workers", default=-1)
    parser.add_argument("--K", type=int, default=3)
    opt = parser.parse_args()

    if opt.dataset == 'iscx-vpn':
        config = ISCXVPNConfig()
    elif opt.dataset == 'iscx-nonvpn':
        config = ISCXNonVPNConfig()
    elif opt.dataset == 'iscx-tor':
        config = ISCXTorConfig()
    elif opt.dataset == 'iscx-nontor':
        config = ISCXNonTorConfig()
    elif opt.dataset == 'ustc':
        config = USTCConfig()
    elif opt.dataset == 'med':
        config = MedConfig()
    elif opt.dataset == 'mqtt':
        config = MQTTConfig()
    elif opt.dataset == 'kitsune':
        config = KitsuneConfig()
    elif opt.dataset == 'ctu':
        config = CTUConfig()
    else:
        raise Exception('Dataset Error')

    device = torch.device('cuda')
    num_workers = opt.num_workers if opt.num_workers >= 1 else config.NUM_WORKERS
    set_seed()
    train(opt)
