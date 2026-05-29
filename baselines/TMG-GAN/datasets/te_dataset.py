from datasets.c_dataset import Dataset


class TeDataset(Dataset):
    def __init__(self):
        super().__init__(training=False)
