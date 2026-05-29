import os
import argparse
import random

import torch
import dgl
import multiprocessing
import numpy as np

from concurrent.futures import as_completed, ProcessPoolExecutor
from utils import show_time, construct_graph, split_flow_Tor_nonoverlapping, split_flow_ISCX, split_flow_ISCX_m
from config import *


def fetch_npzs(packet_queue, npz_file, label):
    print('{} {} Process Starting'.format(show_time(), npz_file))
    packet_queue.put((npz_file, label))


def construct_dataset_from_bytes_ISCX(packet_queue):
    payload_data_list = []
    header_data_list = []
    data_label = []
    while True:
        item = packet_queue.get()
        if item is None:  # 检测到结束标志`
            break
        else:
            # npz_file为保存每个flow的header_list, payload_list, payload_length_list等的路径
            npz_file, label = item
            if not npz_file.endswith('.npz'):
                continue

            payload_list, header_list = split_flow_ISCX_m(npz_file, allow_empty=False, pad_trunc=True, config=config)
            if not payload_list or not header_list:
                continue

            # [ [[flow1_p1_payload], [flow1_p2_payload],..., [flow1_pn_payload]], [[flow2_p1_payload], [flow2_p2_payload],..., [flow2_pn_payload]],..., [[flown_p1_payload], [flown_p2_payload],..., [flown_pn_payload]] ]
            payload_data_list.extend(payload_list)
            header_data_list.extend(header_list)
            data_label.append(label)

    test_size = int(len(payload_data_list) * 0.1)
    all_indices = list(range(len(payload_data_list)))
    test_indices = random.sample(all_indices, test_size)
    train_indices = [index for index in all_indices if index not in test_indices]
    payload_data_list = np.array(payload_data_list)
    data_label = np.array(data_label)
    header_data_list = np.array(header_data_list)
    train_label, test_label = data_label[train_indices], data_label[test_indices]
    payload_train, payload_test = payload_data_list[train_indices], payload_data_list[test_indices]
    header_train, header_test = header_data_list[train_indices], header_data_list[test_indices]
    print(f'payload_train length is {len(payload_train)}, header_train length is {len(header_train)}, label length is {len(train_label)}')

    # 将数据保存到npz文件中，其中键为data，值为相应数据，键为label，值为相应标签
    np.savez_compressed(config.TRAIN_DATA, data=payload_train, label=train_label)
    np.savez_compressed(config.TEST_DATA, data=payload_test, label=test_label)
    np.savez_compressed(config.HEADER_TRAIN_DATA, data=header_train, label=train_label)
    np.savez_compressed(config.HEADER_TEST_DATA, data=header_test, label=test_label)


def construct_dataset_from_bytes(dir_path_dict):
    # 使用 Manager().Queue() 替换 multiprocessing.Queue()
    with multiprocessing.Manager() as manager:
        packet_queue = manager.Queue(maxsize=100)

        # 创建生产者进程
        with ProcessPoolExecutor(max_workers=12) as executor:
            futures = []
            for label, npz_dir in dir_path_dict.items():
                npz_paths = os.listdir(npz_dir)
                # npz_paths = np.random.choice(npz_paths, size=7000) if len(npz_paths) > 7000 else npz_paths
                for npz_path in npz_paths:
                    future = executor.submit(
                        fetch_npzs,
                        packet_queue,
                        os.path.join(npz_dir, npz_path),
                        label
                    )
                    futures.append(future)

            # 创建消费者进程
            consumer_process = multiprocessing.Process(
                target=construct_dataset_from_bytes_ISCX,
                args=(packet_queue, )
            )
            consumer_process.start()

            # 等待生产者进程结束
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    print(f"Failed to read pcap file: {e}")

            # 通知消费者进程结束
            packet_queue.put(None)

        # 等待消费者进程结束
        consumer_process.join()


def construct_graph_format_data(file_path, save_path, type, w_size=5, pmi=1):
    file = np.load(file_path, allow_pickle=True)
    gs = []
    if type == 'payload':
        # data: [[flow1_p1_payload], [flow1_p2_payload], [flow2_p1_payload],..., [flown_p1_payload],..., [flown_pn_payload]]
        data = file['data'].reshape(-1, config.BYTE_PAD_TRUNC_LENGTH)
    else:
        data = file['data'].reshape(-1, config.HEADER_BYTE_PAD_TRUNC_LENGTH)
    label = file['label']
    for ind, p in enumerate(data):
        gs.append(construct_graph(bytes=p, w_size=w_size, k=pmi))
        if ind % 500 == 0:
            print('{} {} Graphs Constructed'.format(show_time(), ind))

    dgl.save_graphs(save_path, gs, {"glabel": torch.LongTensor(label)})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, help="dataset", required=True)
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

    construct_dataset_from_bytes(dir_path_dict=config.SEP_NPZ_FILE)
    construct_graph_format_data(file_path=config.TRAIN_DATA, save_path=config.TRAIN_GRAPH_DATA, type='payload')
    construct_graph_format_data(file_path=config.TEST_DATA, save_path=config.TEST_GRAPH_DATA, type='payload')
    construct_graph_format_data(file_path=config.HEADER_TRAIN_DATA, save_path=config.HEADER_TRAIN_GRAPH_DATA, type='header')
    construct_graph_format_data(file_path=config.HEADER_TEST_DATA, save_path=config.HEADER_TEST_GRAPH_DATA, type='header')