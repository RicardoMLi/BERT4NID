import random

from datasets.data import tr_samples, tr_labels, te_samples, te_labels


class Dataset:
    def __init__(self, training: bool = True):
        if training:
            self.samples = tr_samples.to('cuda')
            self.labels = tr_labels.to('cuda')
        else:
            self.samples = te_samples.to('cuda')
            self.labels = te_labels.to('cuda')
        # if training:
        #     train_size = int(len(tr_samples) * 0.01)
        #     all_indices = list(range(len(tr_samples)))
        #     train_indices = random.sample(all_indices, train_size)
        #     self.samples = tr_samples[train_indices].to('cuda')
        #     self.labels = tr_labels[train_indices].to('cuda')
        # else:
        #     self.samples = te_samples.to('cuda')
        #     self.labels = te_labels.to('cuda')

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx: int):
        return self.samples[idx], self.labels[idx]
