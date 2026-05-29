import numpy as np

from pathlib import Path
from scapy.all import rdpcap


def read_and_fetch_packets(packet_queue, pcap_path, label):
    print(f"Reading from file: {pcap_path}.")
    packets = rdpcap(str(pcap_path))
    packet_queue.put((packets, label))


class PcapDict(dict):
    def __init__(self, root_dir: str, num_samples_per_class: int, mapper: dict):
        super().__init__()
        self.root_dir = Path(root_dir)
        self.num_samples_per_class = num_samples_per_class
        self.mapper = mapper
        self._load_data()

    def _load_data(self):
        # Iterate through all subdirectories in the root directory
        # label = 0
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

            # pcap_files.sort(key=lambda x: os.path.getsize(x))
            # Get the top large pcap files
            pcap_files = np.random.choice(pcap_files, size=self.num_samples_per_class) if len(pcap_files) > self.num_samples_per_class else pcap_files
            # Store the label and the list of pcap files
            if label in self:
                self[label] = np.append(self[label], pcap_files)
            else:
                self[label] = pcap_files
            # label += 1

    def __repr__(self):
        return f"PcapDict(root_dir={self.root_dir})"
