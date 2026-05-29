import random
import numpy as np
import pandas as pd

from PIL import Image
from torch.utils.data import Dataset


class ViTDataSet(Dataset):
    def __init__(self, tsv_path, transform, is_train=False):
        super(ViTDataSet, self).__init__()

        self.data = pd.read_csv(tsv_path, sep='\t')
        self.xs = self.data['x'].values
        self.labels = self.data['labels'].values
        # if is_train:
        #     train_size = int(len(self.xs) * 0.01)
        #     all_indices = list(range(len(self.xs)))
        #     train_indices = random.sample(all_indices, train_size)
        #     self.xs = self.xs[train_indices]
        #     self.labels = self.labels[train_indices]
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        x = np.reshape(list(map(int, self.xs[index].split(','))), (28, 28))
        x = Image.fromarray(np.uint8(x))
        y = self.labels[index]

        return self.transform(x), y


