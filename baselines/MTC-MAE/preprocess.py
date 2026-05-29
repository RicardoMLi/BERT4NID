import time
import argparse
import os
import binascii
import multiprocessing
import pandas as pd

from pathlib import Path
from sklearn.model_selection import train_test_split
from utils.process_packet import transform_packet
from utils.constants import *
from utils.preprocess_utils import PcapDict, PretrainPcapDict, read_and_fetch_packets, set_seed
from concurrent.futures import as_completed, ProcessPoolExecutor


def pad_flow(packets, max_flow_length=784):
    packet_count = 0
    x = []
    for packet in packets:
        packet = transform_packet(packet)
        if packet is None:
            print(f"We just deal with ip packets.")
            continue

        packet = binascii.hexlify(bytes(packet)).decode()
        packet = [int(packet[i:i + 2], 16) for i in range(0, len(packet), 2)]
        x += packet

        if len(x) >= max_flow_length:
            x = x[:max_flow_length]
            break

        packet_count += 1

    # 所有packet全为None
    if x is []:
        print("All packets return None in this flow.")
        return None

    if len(x) < max_flow_length:
        pad_length = max_flow_length - len(x)
        x += [0] * pad_length

    return x,  packet_count


def transform_flow(packet_queue, output_path, max_flow_length):
    xs = []
    labels = []
    total_packet = 0
    while True:
        item = packet_queue.get()
        if item is None:  # 检测到结束标志`
            break
        else:
            packets, label = item
            x, num_packet = pad_flow(packets, max_flow_length)

            if x is None:
                continue

            x = ','.join(list(map(str, x)))
            xs.append(x)
            labels.append(label)
            total_packet += num_packet

            if len(xs) > 50000:
                print(f'flow length is more than 50000, write to {output_path}')
                table = pd.DataFrame({'x': xs, 'labels': labels})
                table.to_csv(output_path, mode='a', index=False, header=False if os.path.exists(output_path) else True, sep='\t')

                xs = []
                labels = []

    # Save the rest data as csv file
    table = pd.DataFrame({'x': xs, 'labels': labels})
    table.to_csv(output_path, mode='a', index=False, header=False, sep='\t')

    print("Dataset processed and saved.")
    print(f"Total packet number is {total_packet}")


def flow2dataset(args):
    data_dir_path = Path(args.split_session_folder)

    if args.pretrain:
        pcap_dict = PretrainPcapDict(str(data_dir_path))
    else:
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
                args=(packet_queue, args.output_path,  args.max_flow_length)
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


if __name__ == "__main__":
    t1 = time.time()

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--seed", type=int, default=3407,
                        help="The random seed.")
    parser.add_argument("--split_session_folder", type=str, required=True,
                        help="Folder to store the split pcap session")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Folder to store the bert finetune dataset.")
    parser.add_argument("--num_samples_per_class", type=int, default=7000,
                        help="The number of samples per class to save training time.")
    parser.add_argument("--test_size", type=float, default=0.15,
                        help="The test ratio in train_test_split.")
    parser.add_argument("--max_flow_length", type=int, default=784,
                        help="The length of each flow.")
    parser.add_argument("--pretrain", type=bool, default=False,
                        help="The length of each flow.")

    args = parser.parse_args()

    set_seed(args.seed)
    flow2dataset(args)
    if not args.pretrain:
        split_dataset(args.output_path, args.seed, test_size=args.test_size)

    t2 = time.time()
    print('run time: %s sec' % time.strftime("%H:%M:%S", time.gmtime(t2 - t1)))
