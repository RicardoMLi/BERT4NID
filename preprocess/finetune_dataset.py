import time
import sys
import argparse
import os
import binascii
import multiprocessing
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pathlib import Path
from sklearn.model_selection import train_test_split
from utils.process_packet import transform_packet
from utils.constants import *
from uer.utils.seed import set_seed
from utils.preprocess_utils import PcapDict, read_and_fetch_packets
from vocab.main import bigram_generation, onegram_generation
from concurrent.futures import as_completed, ProcessPoolExecutor


def pad_packet(packets, packet_len=128):
    flow_data_string = ''
    for packet in packets:
        packet = transform_packet(packet)
        if packet is None:
            print(f"We just deal with ip packets.")
            continue

        packet = binascii.hexlify(bytes(packet)).decode()
        flow_data_string += bigram_generation(packet, packet_len=packet_len)

    # 所有packet全为None
    if flow_data_string == '':
        print("All packets return None in this flow.")

    return flow_data_string


def pad_flow(packets, max_packet_num=6, packet_len=128):
    packet_count = 0
    flow_data_string = ''
    for packet in packets:
        packet = transform_packet(packet)
        if packet is None:
            print(f"We just deal with ip packets.")
            continue

        packet = binascii.hexlify(bytes(packet)).decode()
        if packet_count == max_packet_num:
            break
        else:
            flow_data_string += bigram_generation(packet, packet_len=packet_len)

        packet_count += 1

    # 所有packet全为None
    if flow_data_string == '':
        print("All packets return None in this flow.")

    return flow_data_string, packet_count


def transform_flow(packet_queue, output_path, max_packet_num, level, packet_len):
    pd.set_option('max_colwidth', 7168)
    xs = []
    labels = []
    total_packet = 0
    while True:
        item = packet_queue.get()
        if item is None:  # 检测到结束标志
            break
        else:
            packets, label = item
            # flow level or packet level
            num_packet = 0
            if level == 'flow':
                x, num_packet = pad_flow(packets, max_packet_num, packet_len)
            else:
                x = pad_packet(packets, packet_len)

            if x == '':
                continue

            xs.append(x)
            labels.append(label)
            total_packet += num_packet

    # Save the rest data as csv file
    table = pd.DataFrame({'label': labels, 'text_a': xs})
    table.to_csv(output_path, mode='w', index=False, header=True, sep='\t')

    print("Dataset processed and saved.")
    print(f"Total packet number is {total_packet}")


def flow2dataset(args):
    data_dir_path = Path(args.split_session_folder)

    pcap_dict = PcapDict(str(data_dir_path), args.num_samples_per_class, mapper=args.mapper)

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
                args=(packet_queue, args.output_path, args.max_packet_num, args.level, args.packet_len)
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
    pd.set_option('max_colwidth', 7168)
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


if __name__ == "__main__":
    t1 = time.time()

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--seed", type=int, default=3407,
                        help="The random seed.")
    parser.add_argument("--dataset", type=str, required=True, 
                        help="The dataset name, e.g., UTSC, Kitsune, CTU-13, MQTT-IoT, MedBIoT.")
    parser.add_argument("--split_session_folder", type=str, required=True,
                        help="Folder to store the split pcap session")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Folder to store the bert finetune dataset.")
    parser.add_argument("--num_samples_per_class", type=int, default=7000,
                        help="The number of samples per class to save training time.")
    parser.add_argument("--max_packet_num", type=int, default=3,
                        help="The max packet number of a flow.")
    parser.add_argument("--test_size", type=float, default=0.15,
                        help="The test ratio in train_test_split.")
    parser.add_argument("--level", type=str, default='flow',
                        help="Flow level or packet level intrusion detection.")
    parser.add_argument("--packet_len", type=int, default=128,
                        help="The length of each packet.")

    args = parser.parse_args()

    if args.dataset == "ustc":
        args.mapper = UTSC_MAPPER
    elif args.dataset == "kitsune":
        args.mapper = Kitsune_Mapper
    elif args.dataset == "ctu":
        args.mapper = CTU_13_MAPPER
    elif args.dataset == "mqtt":
        args.mapper = MQTT_IoT_MAPPER
    elif args.dataset == "med":
        args.mapper = MedBIoT_MAPPER
    elif args.dataset == "edge":
        args.mapper = Edge_IIoT_MAPPER
    else:
        raise ValueError("Unsupported dataset. Please choose from: ustc, kitsune, ctu, mqtt, med, edge.")
    
    set_seed(args.seed)
    flow2dataset(args)
    split_dataset(args.output_path, args.seed, test_size=args.test_size)

    t2 = time.time()
    print('run time: %s sec' % time.strftime("%H:%M:%S", time.gmtime(t2 - t1)))
