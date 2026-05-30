import time
import os
import binascii
import multiprocessing
import pyarrow as pa
import pyarrow.parquet as pq

from functools import partial
from pathlib import Path
from scapy.all import rdpcap
from preprocess.pipeline import CopyStage, Pipeline
from preprocess.process_packet import transform_packet
from concurrent.futures import as_completed, ProcessPoolExecutor
from utils.arguments import PreprocessArguments, StageArguments
from utils.file_utils import str2path
from utils.constants import *
from utils.misc import set_seed
from utils.logging_utils import get_logger

logger = get_logger(__name__)


class PcapDict(dict):
    def __init__(self, root_dir: str, num_samples_per_class: int):
        super().__init__()
        self.root_dir = Path(root_dir)
        self.num_samples_per_class = num_samples_per_class
        self._load_data()

    def _load_data(self):
        # Iterate through all subdirectories in the root directory
        for label_dir in self.root_dir.iterdir():
            # Skip if it's not a directory
            if not label_dir.is_dir():
                continue

            # folder name like Skype.pcap
            label = label_dir.name
            # for key, value in CIC_IOT_MAPPER.items():
            #     if str(label).replace('.pcap', '') in value:
            #         label = key
            label = UTSC_MAPPER[str(label).replace('.pcap', '')]
            # Iterate through all pcap files in the subdirectory
            pcap_files = [
                pcap_file
                for pcap_file in label_dir.iterdir()
                if pcap_file.name.endswith(".pcap")
            ]
            pcap_files.sort(key=lambda x: os.path.getsize(x))
            # Get the top large pcap files
            pcap_files = pcap_files[-self.num_samples_per_class:] if len(pcap_files) > self.num_samples_per_class else pcap_files
            # Store the label and the list of pcap files
            self[label] = pcap_files

    def __repr__(self):
        return f"PcapDict(root_dir={self.root_dir})"

def tshark_extract_pipeline(pre_args):
    src_root, dst_root, output_root = str2path(
        pre_args.dataset_src_root_dir,
        pre_args.dataset_dst_root_dir,
        pre_args.output_dir,
    )

    assert src_root.is_dir()
    dst_root.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)

    stage_args = partial(
        StageArguments,
        output_dir=output_root,
        src_file=output_root.joinpath("src.txt"),
        dst_file=output_root.joinpath("dst.txt"),
        num_workers=pre_args.num_workers
    )

    pipeline_args = [
        (
            CopyStage,
            # 将pcap文件按五元组切分成bidirectional flow
            stage_args(
                name="split_sessions",
                category="CopyStage",
                src_folder=src_root,
                dst_folder=dst_root.joinpath(pre_args.split_session_folder),
                file2folder=True,
                cmd="bash preprocess/split_sessions.sh {1} {2} "
                    f"{pre_args.min_packet_num} "
                    f"{pre_args.min_file_size} "
                    f"session "
                    f"{pre_args.splitcap_path} ",
            )
        ),
        # (
        #     CopyStage,
        #     # 将bidirectional flow按照time_window细分为burst
        #     stage_args(
        #         name="trim_sessions",
        #         category="CopyStage",
        #         src_folder=dst_root.joinpath(pre_args.split_session_folder),
        #         dst_folder=dst_root.joinpath(pre_args.trim_time_folder),
        #         cmd="bash preprocess/trim.sh {1} {2} "
        #             f"{pre_args.time_window} "
        #     )
        # ),
    ]

    pipeline = Pipeline(pipeline_args)
    pipeline.run()

def read_and_fetch_packets(packet_queue, pcap_path, label):
    print(f"Reading from file: {pcap_path}.")
    packets = rdpcap(str(pcap_path))
    packet_queue.put((packets, label))


def trim_pad_burst(packets, max_pcap_size):
    x = []
    num_packet_omit = len(packets)
    for packet in packets:
        packet = transform_packet(packet)
        if packet is None:
            num_packet_omit -= 1
            print(f"tcp or udp payload is none.")
            continue

        packet = binascii.hexlify(bytes(packet)).decode()
        # 将pcap文件从字节流转换为图片格式, bytes -> [0, 255]
        packet = [int(packet[i:i + 2], 16) for i in range(0, len(packet), 2)]
        x += packet
        if len(x) >= max_pcap_size:
            x = x[:max_pcap_size]
            break

    # 所有packet全为None
    if num_packet_omit == 0:
        print("All packets return None in this burst.")
        return None

    if len(x) < max_pcap_size:
        pad_length = max_pcap_size - len(x)
        x += [0] * pad_length

    return x


def transform_burst(packet_queue, output_path, max_file_size):
    xs = []
    labels = []
    while True:
        item = packet_queue.get()
        if item is None:  # 检测到结束标志`
            break
        else:
            packets, label = item
            # 将pcap文件trim或者pad到784字节
            x = trim_pad_burst(packets, max_file_size)
            if x is None:
                continue

            xs.append(','.join(map(str, x)))
            labels.append(label)

    # Create a Dataset from the list of dictionaries
    print(f"deal with busrt completed, dataset contains {len(xs)} sample.")
    table = pa.table([xs, labels], names=['x', 'labels'])

    # Save transformed Dataset as Arrow file
    dataset_path = os.path.join(output_path, "data_iot_7.parquet")
    pq.write_table(table, dataset_path)

    print("Dataset processed and saved.")


def burst2dataset(args):
    data_dir_path = Path(os.path.join(args.dataset_dst_root_dir, args.trim_time_folder))

    pcap_dict = PcapDict(str(data_dir_path), args.num_samples_per_class)

    # 使用 Manager().Queue() 替换 multiprocessing.Queue()
    with multiprocessing.Manager() as manager:
        packet_queue = manager.Queue(maxsize=100)

        # 创建生产者进程
        with ProcessPoolExecutor(max_workers=args.njobs) as executor:
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
                target=transform_burst,
                args=(packet_queue, args.output_dir, args.max_file_size)
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


if __name__ == "__main__":
    t1 = time.time()

    pre_args = PreprocessArguments()
    set_seed(pre_args.seed)
    tshark_extract_pipeline(pre_args)
    burst2dataset(pre_args)

    t2 = time.time()
    print('run time: %s sec' % time.strftime("%H:%M:%S", time.gmtime(t2 - t1)))
