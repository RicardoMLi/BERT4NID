import argparse
import os
import multiprocessing

import numpy as np
from scapy.all import hexdump, PcapReader
from pathlib import Path
from concurrent.futures import as_completed, ProcessPoolExecutor

from utils import show_time, get_bytes_from_raw, get_bytes_from_raw_new, transform_packet
from config import *


def read_and_fetch_packets(packet_queue, npz_save_path, file_path, flow_length):
    print('{} {} Process Starting'.format(show_time(), file_path))
    p_header_list = []
    p_payload_list = []
    # payload_length = []
    # pkt_length = []
    # src_ip = []
    # dst_ip = []
    # src_port = []
    # dst_port = []
    # time = []
    # protocol = []
    # flag = []
    # mss = []
    file_path = Path(file_path)
    count = 0
    with PcapReader(str(file_path)) as packets:
    # packets = rdpcap(str(file_path))
        for pkt in packets:
            pkt = transform_packet(pkt)
            p_header = get_bytes_from_raw_new(pkt)
            p_payload = []
            if pkt.haslayer("Raw"):
                _, p_payload = get_bytes_from_raw(hexdump(pkt["Raw"].load, dump=True))
            
            p_header_list.append(p_header)
            p_payload_list.append(p_payload)
            count += 1
            if count == flow_length:  # 截断
                break

        # payload_length.append(len(p_payload))
        # pkt_length.append(len(p_header) + len(p_payload))
        # src_ip.append(pkt.src)
        # dst_ip.append(pkt.dst)
        # src_port.append(pkt.sport)
        # dst_port.append(pkt.dport)
        # time.append(pkt.time)
        # protocol.append(pkt.proto)
        # flag.append(pkt['TCP'].flags)
        # mss_default = 0
        # for k, v in pkt['TCP'].options:
        #     if k == 'MSS':
        #         mss_default = v
        # mss.append(mss_default)

    packet_queue.put((file_path.name, npz_save_path, p_header_list, p_payload_list))  # , payload_length, pkt_length, src_ip, dst_ip, src_port, dst_port, time, protocol, flag, mss


def save_npz(packet_queue):
    while True:
        item = packet_queue.get()
        if item is None:  # 检测到结束标志`
            break
        else:
            # , payload_length, pkt_length, src_ip, dst_ip, src_port, dst_port, time, protocol, flag, mss
            file_path, npz_save_path, p_header_list, p_payload_list = item
            if len(p_header_list) != len(p_payload_list):
                print("Error")
                raise Exception('Error')

            if not p_payload_list:
                print('All packets in this flow contain no payload.')
                continue

            save_file = file_path[:-4] + 'npz'
            # 将各个类别中的flow的header_list, payload_list, payload_length_list等保存到本地npz文件
            # 格式为[[p1_header], [p2_header], ..., [pn_header]]
            # npz文件中键为header的值为header_list
            np.savez_compressed(os.path.join(npz_save_path, save_file),
                                header=np.array(p_header_list, dtype=object),
                                payload=np.array(p_payload_list, dtype=object))
                                # payload_length=np.array(payload_length, dtype=object),
                                # pkt_length=np.array(pkt_length, dtype=object),
                                # src_ip=np.array(src_ip, dtype=object),
                                # dst_ip=np.array(dst_ip, dtype=object),
                                # src_port=np.array(src_port, dtype=object),
                                # dst_port=np.array(dst_port, dtype=object),
                                # time=np.array(time, dtype=object),
                                # protocol=np.array(protocol, dtype=object),
                                # flag=np.array(flag, dtype=object),
                                # mss=np.array(mss, dtype=object))


def pcap2npy4ISCX(dir_path_dict, save_path_dict, max_files, flow_length):
    with multiprocessing.Manager() as manager:
        packet_queue = manager.Queue(maxsize=100)

        # 创建生产者进程
        with ProcessPoolExecutor(max_workers=10) as executor:
            futures = []
            for label, dir_path in dir_path_dict.items():
                pcap_paths = os.listdir(dir_path)
                pcap_paths = np.random.choice(pcap_paths, max_files) if len(pcap_paths) > max_files else pcap_paths
                for pcap_path in pcap_paths:
                    if not pcap_path.endswith('.pcap'):
                        continue

                    npz_save_path = save_path_dict[label]
                    future = executor.submit(
                        read_and_fetch_packets,
                        packet_queue,
                        npz_save_path,
                        os.path.join(dir_path, pcap_path),
                        flow_length
                    )
                    futures.append(future)

            # 创建消费者进程
            consumer_process = multiprocessing.Process(
                target=save_npz,
                args=(packet_queue,)
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

    pcap2npy4ISCX(dir_path_dict=config.DIR_PATH_DICT, save_path_dict=config.SEP_NPZ_FILE,max_files=config.MAX_SEG_PER_CLASS, flow_length=config.FLOW_PAD_TRUNC_LENGTH)

