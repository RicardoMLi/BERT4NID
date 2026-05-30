import torch
import torchmetrics
import random
import os
import torch.nn as nn
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch.nn.functional as F

from PIL import Image
from torchvision import transforms
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torch.utils.data import DataLoader, Dataset


def set_seed(seed=7):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


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


class BurstDataModule(pl.LightningDataModule):
    def __init__(self, train_path, test_path, val_path, train_transform, test_transform):
        super(BurstDataModule, self).__init__()

        self.train_path = train_path
        self.test_path = test_path
        self.val_path = val_path
        self.train_transform = train_transform
        self.test_transform = test_transform

    def prepare_data(self):
        # load dataset
        self.train_ds = ViTDataSet(self.train_path, self.train_transform, is_train=True)
        self.test_ds = ViTDataSet(self.test_path, self.test_transform)
        self.val_ds = ViTDataSet(self.val_path, self.test_transform)

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=512,
            pin_memory=True,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=512,
            pin_memory=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=512,
            pin_memory=True
        )


class CNN(pl.LightningModule):
    def __init__(self, num_classes):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, padding=2)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.relu = nn.ReLU(inplace=True)

        self.flatten = nn.Flatten()
        self.linear = nn.Linear(64*7*7, 1024)
        self.dropout = nn.Dropout(0.5)
        self.output = nn.Linear(1024, num_classes)

        self.accuracy_score = torchmetrics.Accuracy('multiclass', num_classes=num_classes, average='weighted')
        self.f1_score = torchmetrics.F1Score('multiclass', num_classes=num_classes, average='weighted')
        self.precision_score = torchmetrics.Precision('multiclass', num_classes=num_classes, average='weighted')
        self.recall_score = torchmetrics.Recall('multiclass', num_classes=num_classes, average='weighted')

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.dropout(self.linear(x))

        return self.output(x)

    def _common_step(self, batch, batch_idx):
        x, y = batch[0], batch[1]
        out = self.forward(x)
        loss = F.cross_entropy(out, y)

        return loss, out, y

    def training_step(self, batch, batch_idx):
        loss, out, y = self._common_step(batch, batch_idx)
        self.log("training loss", loss.item())

        return loss

    def update_metric(self, y_true, y_pred):
        self.accuracy_score.update(y_true, y_pred)
        self.precision_score.update(y_true, y_pred)
        self.recall_score.update(y_true, y_pred)
        self.f1_score.update(y_true, y_pred)

    def compute_metric(self):
        result = {
            'accuracy': self.accuracy_score.compute(),
            'precision': self.precision_score.compute(),
            'recall': self.recall_score.compute(),
            'f1': self.f1_score.compute()
        }

        return result

    def reset_metric(self):
        self.accuracy_score.reset()
        self.precision_score.reset()
        self.recall_score.reset()
        self.f1_score.reset()

    def validation_step(self, batch, batch_idx):
        loss, out, y = self._common_step(batch, batch_idx)
        predict = torch.argmax(out, dim=-1)
        self.update_metric(y, predict)
        self.log("val_loss", loss.item())

        return loss

    def on_validation_epoch_end(self):
        result = self.compute_metric()
        print(f"\nEpoch [{self.current_epoch}] - Accuracy: {result['accuracy']:.4f}, Precision: {result['precision']:.4f}, "
              f"Recall: {result['recall']:.4f}, F1: {result['f1']:.4f}")

    def on_test_start(self):
        self.reset_metric()

    def test_step(self, batch, batch_idx):
        loss, out, y = self._common_step(batch, batch_idx)
        predict = torch.argmax(out, dim=-1)
        self.update_metric(y, predict)
        self.log("test_loss", loss.item())

        return loss

    def on_test_end(self):
        result = self.compute_metric()
        print(f"\nTesting - Accuracy: {result['accuracy']:.4f}, Precision: {result['precision']:.4f}, "
              f"Recall: {result['recall']:.4f}, F1: {result['f1']:.4f}")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-4)

        return {'optimizer': optimizer}


if __name__ == "__main__":
    set_seed(888)
    torch.set_float32_matmul_precision(precision='high')
    train_path = r'./datasets/finetune_dataset_med_train.tsv'
    test_path = r'./datasets/finetune_dataset_med_test.tsv'
    val_path = r'./datasets/finetune_dataset_med_val.tsv'
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    dm = BurstDataModule(train_path, test_path, val_path, transform, transform)
    model = CNN(num_classes=4)

    early_stopping = EarlyStopping('val_loss', patience=5)
    checkpoint_callback = ModelCheckpoint(dirpath='checkpoints', save_top_k=1, monitor="val_loss",
                                          save_weights_only=True)

    trainer = pl.Trainer(
        accelerator='cuda' if torch.cuda.is_available() else 'cpu',
        min_epochs=1,
        precision=16,
        max_epochs=50,
        callbacks=[checkpoint_callback, early_stopping]
    )

    trainer.fit(model, dm)
    trainer.validate(model, dm)
    trainer.test(model, dm)
