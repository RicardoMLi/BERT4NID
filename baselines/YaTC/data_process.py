import random
import os
import torch
import argparse
import time
import binascii
import multiprocessing

import numpy as np
import pandas as pd

from pathlib import Path
from constants import *
from scapy.all import rdpcap
from scapy.layers.inet import IP, TCP, UDP
from scapy.layers.l2 import Ether
from sklearn.model_selection import train_test_split
from concurrent.futures import as_completed, ProcessPoolExecutor


class PcapDict(dict):
    def __init__(self, root_dir: str, num_samples_per_class: int, mapper: dict):
        super().__init__()
        self.root_dir = Path(root_dir)
        self.num_samples_per_class = num_samples_per_class
        self.mapper = mapper
        self._load_data()

    def _load_data(self):
        # Iterate through all subdirectories in the root directory
        for label_dir in self.root_dir.iterdir():
            # Skip if it's not a directory
            if not label_dir.is_dir():
                continue

            # folder name like Skype.pcap
            label = label_dir.name
            label = self.mapper[str(label).replace('.pcap', '')]
            # Iterate through all pcap files in the subdirectory
            pcap_files = [
                pcap_file
                for pcap_file in label_dir.iterdir()
                if pcap_file.name.endswith(".pcap")
            ]

            # Get the top large pcap files
            pcap_files = np.random.choice(pcap_files, size=self.num_samples_per_class) if len(pcap_files) > self.num_samples_per_class else pcap_files
            # Store the label and the list of pcap files
            self[label] = pcap_files

    def __repr__(self):
        return f"PcapDict(root_dir={self.root_dir})"


def set_seed(seed=7):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def read_and_fetch_packets(packet_queue, pcap_path, label):
    print(f"Reading from file: {pcap_path}.")
    packets = rdpcap(str(pcap_path))
    packet_queue.put((packets, label))


def remove_ether_header(packet):
    if Ether in packet:
        return packet[Ether].payload

    return packet


def mask_ip(packet):
    if IP in packet:
        packet[IP].src = "0.0.0.0"
        packet[IP].dst = "0.0.0.0"

    return packet


def mask_udp(packet):
    if UDP in packet:
        packet[UDP].sport = 0
        packet[UDP].dport = 0

    return packet


def mask_tcp(packet):
    if TCP in packet:
        packet[TCP].sport = 0
        packet[TCP].dport = 0

    return packet


def transform_packet(packet):
    if IP not in packet:
        return None

    packet = remove_ether_header(packet)
    packet = mask_ip(packet)
    packet = mask_udp(packet)
    packet = mask_tcp(packet)

    return packet


def read_MFR_bytes(packets, max_packet_num):
    data = []
    for packet in packets:
        packet = transform_packet(packet)
        if packet is None:
            print(f"We just deal with ip packets.")
            continue

        header = (binascii.hexlify(bytes(packet['IP']))).decode()
        try:
            payload = (binascii.hexlify(bytes(packet['Raw']))).decode()
            header = header.replace(payload, '')
        except:
            payload = ''
        if len(header) > 160:
            header = header[:160]
        elif len(header) < 160:
            header += '0' * (160 - len(header))
        if len(payload) > 480:
            payload = payload[:480]
        elif len(payload) < 480:
            payload += '0' * (480 - len(payload))
        data.append((header, payload))
        if len(data) >= max_packet_num:
            break
    if len(data) < max_packet_num:
        for i in range(max_packet_num-len(data)):
            data.append(('0'*160, '0'*480))
    final_data = ''
    for h, p in data:
        final_data += h
        final_data += p
    return final_data


def transform_flow(packet_queue, output_path, max_packet_num):
    xs = []
    labels = []
    total_packet = 0
    while True:
        item = packet_queue.get()
        if item is None:  # 检测到结束标志`
            break
        else:
            packets, label = item
            num_packet = 0
            x = read_MFR_bytes(packets, max_packet_num)

            if x == '':
                continue

            x = np.array([int(x[i:i + 2], 16) for i in range(0, len(x), 2)])
            x = ','.join(list(map(str, x)))
            xs.append(x)
            labels.append(label)
            total_packet += num_packet

            if len(xs) > 50000:
                print(f'flow length is more than 50000, write to {output_path}')
                table = pd.DataFrame({'x': xs, 'labels': labels})
                table.to_csv(output_path, mode='a', index=False, header=None if os.path.exists(output_path) else True, sep='\t')

                xs = []
                labels = []

    # Save the rest data as csv file
    table = pd.DataFrame({'x': xs, 'labels': labels})
    table.to_csv(output_path, mode='a', index=False, header=False, sep='\t')

    print("Dataset processed and saved.")
    print(f"Total packet number is {total_packet}")


def flow2dataset(args):
    data_dir_path = Path(args.split_session_folder)

    pcap_dict = PcapDict(str(data_dir_path), args.num_samples_per_class, mapper=CTU_13_MAPPER)

    # 使用 Manager().Queue() 替换 multiprocessing.Queue()
    with multiprocessing.Manager() as manager:
        packet_queue = manager.Queue(maxsize=100)

        # 创建生产者进程
        with ProcessPoolExecutor(max_workers=12) as executor:
            futures = []
            for label, label_path in pcap_dict.items():
                for pcap_path in label_path:
                    future = executor.submit(
                        read_and_fetch_packets,
                        packet_queue,
                        pcap_path,
                        label
                    )
                    futures.append(future)

            # 创建消费者进程
            consumer_process = multiprocessing.Process(
                target=transform_flow,
                args=(packet_queue, args.output_path, args.max_packet_num)
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


def split_dataset(tsv_file, seed, test_size=0.15):
    table = pd.read_csv(tsv_file, sep='\t')
    train_set, test_set = train_test_split(table, test_size=test_size, random_state=seed)
    train_set, val_set = train_test_split(train_set, test_size=test_size, random_state=seed)

    train_file = tsv_file.replace('.tsv', '_train.tsv')
    test_file = tsv_file.replace('.tsv', '_test.tsv')
    val_file = tsv_file.replace('.tsv', '_val.tsv')

    train_set.to_csv(train_file, index=False, sep='\t')
    test_set.to_csv(test_file, index=False, sep='\t')
    val_set.to_csv(val_file, index=False, sep='\t')
    os.remove(tsv_file)

# def MFR_generator(flows_pcap_path, output_path):
#     flows = glob.glob(flows_pcap_path + "/*/*/*.pcap")
#     makedir(output_path)
#     makedir(output_path + "/train")
#     makedir(output_path + "/test")
#     classes = glob.glob(flows_pcap_path + "/*/*")
#     for cla in tqdm(classes):
#         makedir(cla.replace(flows_pcap_path, output_path))
#     for flow in tqdm(flows):
#         content = read_MFR_bytes(flow)
#         content = np.array([int(content[i:i + 2], 16) for i in range(0, len(content), 2)])
#         fh = np.reshape(content, (40, 40))
#         fh = np.uint8(fh)
#         im = Image.fromarray(fh)
#         im.save(flow.replace('.pcap', '.png').replace(flows_pcap_path, output_path))


if __name__ == "__main__":
    t1 = time.time()

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--seed", type=int, default=3407,
                        help="The random seed.")
    parser.add_argument("--split_session_folder", type=str, required=True,
                        help="Folder to store the split pcap session")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Folder to store the bert finetune dataset.")
    parser.add_argument("--num_samples_per_class", type=int, default=9999,
                        help="The number of samples per class to save training time.")
    parser.add_argument("--max_packet_num", type=int, default=5,
                        help="The max packet number of a flow.")
    parser.add_argument("--test_size", type=float, default=0.15,
                        help="The test ratio in train_test_split.")

    args = parser.parse_args()

    set_seed(args.seed)
    flow2dataset(args)
    split_dataset(args.output_path, args.seed, test_size=args.test_size)

    t2 = time.time()
    print('run time: %s sec' % time.strftime("%H:%M:%S", time.gmtime(t2 - t1)))
