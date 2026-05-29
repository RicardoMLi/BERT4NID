import dgl
import random

import numpy as np
from dgl.data import DGLDataset
from config import Config


config = Config()


class MixTrafficFlowDataset4DGL(DGLDataset):
    def __init__(self, payload_path, header_path, is_train=True):
        self.payload_path = payload_path
        self.header_path = header_path
        self.is_train = is_train
        super(MixTrafficFlowDataset4DGL, self).__init__(name="MixTrafficFlowDataset4DGL")

    def process(self):
        self.payload_data, self.label = dgl.load_graphs(self.payload_path)
        self.header_data, self.label = dgl.load_graphs(self.header_path)
        self.label = self.label["glabel"].numpy()
        # if self.is_train:
        #     train_size = int(len(self.label) * 0.01)
        #     train_indices = random.sample(list(range(len(self.label))), train_size)
        #     tmp_payload, tmp_header, tmp_label = [], [], np.ones(train_size, dtype=int)
        #     for index, selected_index in enumerate(train_indices):
        #         start_ind = config.FLOW_PAD_TRUNC_LENGTH * selected_index
        #         end_ind = start_ind + config.FLOW_PAD_TRUNC_LENGTH
        #         tmp_payload += self.payload_data[start_ind: end_ind]
        #         tmp_header += self.header_data[start_ind: end_ind]
        #         tmp_label[index] = self.label[selected_index]

        #     self.header_data = tmp_header
        #     self.payload_data = tmp_payload
        #     self.label = tmp_label

        assert len(self.payload_data) == len(self.header_data), "Error {} != {}".format(len(self.payload_data), len(self.header_data))

    def __getitem__(self, index):
        start_ind = config.FLOW_PAD_TRUNC_LENGTH * index
        end_ind = start_ind + config.FLOW_PAD_TRUNC_LENGTH
        header = self.header_data[start_ind: end_ind]
        payload = self.payload_data[start_ind: end_ind]
        label = self.label[index]
        return header, payload, label

    def __len__(self):
        return len(self.label)


if __name__ == '__main__':
    pass
