import datasets
import torch
import numpy as np
import torch.nn.functional as F

from pathlib import Path
from torch.utils.data import DataLoader
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from ml.utils import load_traffic_classification_cnn_model, set_seed
from ml.dataset import dataset_collate_function
from torchmetrics import Accuracy, Precision, Recall, F1Score

acc_list = []
pre_list = []
rec_list = []
f1_list = []
num_classes = 9
test_path = Path(r'train_test_data/kitsune/test.parquet')
dataset_dict = datasets.load_dataset(str(test_path.absolute()), cache_dir='/data3/wzy/hyy/lzy')
dataset = dataset_dict[list(dataset_dict.keys())[0]]
dataloader = DataLoader(
    dataset,
    batch_size=4096,
    num_workers=12,
    collate_fn=dataset_collate_function,
)

for seed in [1, 2, 3]:
    set_seed(seed)
    model_path = r'checkpoints/model.pt'
    model = load_traffic_classification_cnn_model(model_path, gpu=True)
    model.eval()
    accuracy = Accuracy(task='multiclass', num_classes=num_classes, average='weighted').to(model.device)
    recall = Recall(task='multiclass', num_classes=num_classes, average='weighted').to(model.device)
    precision = Precision(task='multiclass', num_classes=num_classes, average='weighted').to(model.device)
    f1_score = F1Score(task='multiclass', num_classes=num_classes, average='weighted').to(model.device)
    # y_true = []
    # y_predict = []
    with torch.no_grad():
        for batch in dataloader:
            x = batch["feature"].float().to(model.device)
            y = batch["label"].long()
            # y_true.extend(y.numpy())
            y = y.to(model.device)
            y_hat = torch.argmax(F.log_softmax(model(x), dim=1), dim=1)
            # y_predict.extend(y_hat.cpu().numpy())
            accuracy(y_hat, y)
            recall(y_hat, y)
            precision(y_hat, y)
            f1_score(y_hat, y)

    # accuracy = accuracy_score(y_true, y_predict)
    # precision = precision_score(y_true, y_predict, average='weighted', zero_division=1)
    # recall = recall_score(y_true, y_predict, average='weighted')
    # f1 = f1_score(y_true, y_predict, average='weighted')
    acc = accuracy.compute().item()
    pre = precision.compute().item()
    rec = recall.compute().item()
    f1 = f1_score.compute().item()

    acc_list.append(acc)
    pre_list.append(pre)
    rec_list.append(rec)
    f1_list.append(f1)

    print(f'Accuracy: {acc:.4f}')
    print(f'Precision: {pre:.4f}')
    print(f'Recall: {rec:.4f}')
    print(f'F1 Score: {f1:.4f}')

print(f'Average Accuracy: {np.mean(acc_list):.4f} +- {np.std(acc_list):.4f}')
print(f'Average Precision: {np.mean(pre_list):.4f} +- {np.std(pre_list):.4f}')
print(f'Average Recall: {np.mean(rec_list):.4f} +- {np.std(rec_list):.4f}')
print(f'Average F1: {np.mean(f1_list):.4f} +- {np.std(f1_list):.4f}')