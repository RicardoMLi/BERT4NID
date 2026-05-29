import argparse

import numpy as np
import torch
from dgl.dataloading import GraphDataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from utils import set_seed, get_device, mix_collate_fn
from dataloader import MixTrafficFlowDataset4DGL
from model import MixTemporalGNN
from config import *


torch.autograd.set_detect_anomaly(True)


def test(opt):
    model = MixTemporalGNN(num_classes=config.NUM_CLASSES, K=opt.K, embedding_size=config.EMBEDDING_SIZE, h_feats=config.H_FEATS,
                           dropout=config.DROPOUT, downstream_dropout=config.DOWNSTREAM_DROPOUT).to(device)
    model.load_state_dict(torch.load(config.MIX_MODEL_CHECKPOINT.replace('.pth', '_epoch19.pth'), map_location={'cuda:0': 'cuda:' + str(opt.cuda),
                                                                                'cuda:1': 'cuda:' + str(opt.cuda),
                                                                                'cuda:2': 'cuda:' + str(opt.cuda),
                                                                                'cuda:3': 'cuda:' + str(opt.cuda)}))
    model.eval()
    dataset = MixTrafficFlowDataset4DGL(header_path=config.HEADER_TEST_GRAPH_DATA,
                                        payload_path=config.TEST_GRAPH_DATA, is_train=False)
    dataloader = GraphDataLoader(dataset, batch_size=32, shuffle=False, collate_fn=mix_collate_fn,
                                 num_workers=config.NUM_WORKERS, pin_memory=False)

    label_preds = []
    label_ids = []
    with torch.no_grad():
        for header_data, payload_data, labels in dataloader:
            header_data = header_data.to(device, non_blocking=True)
            payload_data = payload_data.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            pred = model(header_data, payload_data, labels)
            pred_label = pred.argmax(1).detach().cpu().numpy()
            label_preds.extend(pred_label)
            label_ids.extend(labels.detach().cpu().numpy())

    accuracy = accuracy_score(label_ids, label_preds)
    precision = precision_score(label_ids, label_preds, average='weighted')
    recall = recall_score(label_ids, label_preds, average='weighted')
    f1 = f1_score(label_ids, label_preds, average='weighted')
    print(f'Accuracy: {accuracy:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1: {f1:.4f}')

    return accuracy, precision, recall, f1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, help="dataset", required=True)
    parser.add_argument("--cuda", type=str, default="1")
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

    device = get_device(index=opt.cuda)
    accuracy_list, precision_list, recall_list, f1_list = [], [], [], []
    for seed in range(3):
        set_seed(seed)
        accuracy, precision, recall, f1 = test(opt)
        accuracy_list.append(accuracy)
        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)

    print(f'Accuracy: {np.mean(accuracy_list):.4f}+-{np.std(accuracy_list):.4f}')
    print(f'Precision: {np.mean(precision_list):.4f}+-{np.std(precision_list):.4f}')
    print(f'Recall: {np.mean(recall_list):.4f}+-{np.std(recall_list):.4f}')
    print(f'F1: {np.mean(f1_list):.4f}+-{np.std(f1_list):.4f}')
