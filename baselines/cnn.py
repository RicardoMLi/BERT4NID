import torch
import time
import os
import random
import torch.nn as nn
import pandas as pd
import numpy as np

from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
from torchmetrics import Accuracy, Precision, Recall, F1Score
from torch.utils.data import Dataset, DataLoader, random_split

class CNN(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv1d(input_dim, 16, kernel_size=3, stride=1, padding=2)
        self.bn1 = nn.BatchNorm1d(16)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=2)
        self.bn2 = nn.BatchNorm1d(32)
        self.conv3 = nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=2)
        self.bn3 = nn.BatchNorm1d(64)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.relu = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(64, 32)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(32, num_classes)
    
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = torch.squeeze(self.pool(x), dim=2)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x


class IDSDataset(Dataset):
    def __init__(self, base_path):
        super().__init__()

        self.data = pd.DataFrame()
        for csv_path in os.listdir(base_path):
            if csv_path.endswith('.csv'):
                data = pd.read_csv(os.path.join(base_path, csv_path))
                self.data = pd.concat([data, self.data], axis=0)

        self.data = self.data.sample(frac=1)
        self.labels = self.data.pop('label').values
        self.data = preprocess_ustc(self.data).astype(np.float32)
        self.input_dims = self.data.shape[1]
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return np.expand_dims(self.data[idx], axis=1), self.labels[idx]


def preprocess_ustc(data):
    scaler = MinMaxScaler()
    data = scaler.fit_transform(data)
    return data

def min_max_norm(df, name):
    x = df[name].values.reshape(-1, 1)
    min_max_scaler = MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    df[name] = x_scaled

def preprocess_edge(df):
    df.drop('src_ip', axis=1, inplace=True)
    df.drop('dst_ip', axis=1, inplace=True)
    df.drop('src_port', axis=1, inplace=True)
    df.drop('dst_port', axis=1, inplace=True)
    df.drop('timestamp', axis=1, inplace=True)

    norm_cols = df.columns.values
    for feature_id in norm_cols:
        min_max_norm(df, feature_id)

    return df
    
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train(epoch, train_loader, device, num_classes, input_dims):
    model = CNN(input_dim=input_dims, num_classes=num_classes).to(device)
    cross_entropy_loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    epoch_iter = tqdm(range(epoch))
    best_acc = 0.0
    model.train()
    for epoch in epoch_iter:
        loss_list = []
        num_correct = 0
        num_tests = 0
        for batch_id, (data, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            data = data.to(device)
            labels = labels.to(device)
            logits = model(data)
            num_correct += (logits.argmax(1) == labels).sum().item()
            num_tests += len(labels)
            loss = cross_entropy_loss(logits, labels)
            loss_list.append(loss.item())
            loss.backward()
            optimizer.step()

        acc = num_correct / num_tests
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), 'cnn.pt')

        epoch_iter.set_description(f"Epoch {epoch + 1} | train_loss: {np.mean(loss_list):.4f} | train_acc: {acc:.4f} | best_acc: {best_acc:.4f}")

    print('Training finished!')


def test(model, eval_loader, device, num_classes):
    model.eval()

    accuracy = Accuracy(task='multiclass', num_classes=num_classes, average='weighted').to(device)
    recall = Recall(task='multiclass', num_classes=num_classes, average='weighted').to(device)
    precision = Precision(task='multiclass', num_classes=num_classes, average='weighted').to(device)
    f1_score = F1Score(task='multiclass', num_classes=num_classes, average='weighted').to(device)
    total = 0
    with torch.no_grad():
        for data, label in eval_loader:
            data = data.to(device)
            total += data.size(0)
            label = label.to(device)
            logits = model(data)
            pred = logits.argmax(dim=1)
            accuracy(pred, label)
            recall(pred, label)
            precision(pred, label)
            f1_score(pred, label)
            
    acc = accuracy.compute().item()
    pre = precision.compute().item()
    rec = recall.compute().item()
    f1 = f1_score.compute().item()

    print(f"#Test_accuracy: {acc:.4f}")
    print(f"#Test_precision: {pre:.4f}")
    print(f"#Test_recall: {rec:.4f}")
    print(f"#Test_f1: {f1:.4f}")

    return acc, pre, rec, f1, total


base_path = r'F:\datasets\med_processed'
num_classes = 4
epoch = 20
dataset = IDSDataset(base_path)
input_dims = dataset.input_dims   # 115 for kitsune
train_size = int(len(dataset) * 0.8)  # 80% 用于训练
test_size = len(dataset) - train_size  # 余下的部分用于测试
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# 使用 DataLoader 载入训练集和测试集
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

acc_list = []
pre_list = []
rec_list = []
f1_list = []
for seed in [1, 2, 3]:
    print(f'Using seed: {seed} for training and testing.')
    set_seed(seed)
    train(epoch, train_loader, device, num_classes, input_dims)
    model = CNN(input_dim=input_dims, num_classes=num_classes)
    model.load_state_dict(torch.load('cnn.pt'))
    model = model.to(device)
    acc, pre, rec, f1, total = test(model, test_loader, device, num_classes)
    acc_list.append(acc)
    pre_list.append(pre)
    rec_list.append(rec)
    f1_list.append(f1)

print(f'Average Accuracy: {np.mean(acc_list):.4f} +- {np.std(acc_list):.4f}')
print(f'Average Precision: {np.mean(pre_list):.4f} +- {np.std(pre_list):.4f}')
print(f'Average Recall: {np.mean(rec_list):.4f} +- {np.std(rec_list):.4f}')
print(f'Average F1: {np.mean(f1_list):.4f} +- {np.std(f1_list):.4f}')

