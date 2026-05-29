import os
import time
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from tqdm import tqdm
import torch.utils.data as Data
from modules.model_finetune import Classifier, MTMAE_finetuneModel
from models.model_MTMAE import encoder
from pathlib import Path
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
from utils.dataloader_pretrain import ViTDataSet
from torch.utils.data import random_split
from utils.preprocess_utils import set_seed


def fpr(y_true, y_pred):
    classes = np.unique(y_true)
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    FP_total = 0
    TN_total = 0

    for i in range(len(classes)):
        FP = cm[:, i].sum() - cm[i, i]
        TN = cm.sum() - cm[i, :].sum() -cm[:, i].sum() + cm[i, i]

        FP_total += FP
        TN_total += TN
    
    return FP_total / TN_total if TN_total > 0 else 0.

def finetune(classifierHead):
    Encoder = encoder(img_size=28, patch_size=4, in_chans=1, embed_dim=128,
                      encoder_depth=8, num_heads=8, mlp_ratio=3.)
    # load the pretrained weights
    weights_dict = Path(r"./weights/encoder_299.pth")
    Encoder.load_state_dict(torch.load(weights_dict, map_location=torch.device('cpu')))
    print(f"Load weights from {weights_dict} successfully.")
    model = MTMAE_finetuneModel(Encoder, classifierHead)
    model.to(device)

    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=warmup_epochs, T_mult=1, eta_min=0)
    # ---------------------------------------------------------------------------------------------------------------------#

    accuracy_max = 0
    # finetune
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        train_Subset_data_loader = tqdm(data_loader, desc=f'Processing epoch {epoch:02d}')
        for batch, data in enumerate(train_Subset_data_loader):
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            train_Subset_data_loader.set_postfix({f'loss': f'{loss.item(): 6.3f}'})
            loss.backward()
            optimizer.step()
            total_loss += loss.detach().item()
        scheduler.step()

        # val
        model.eval()
        predicted_labels = []
        true_labels = []
        
        with torch.no_grad():
            for batch, data in enumerate(val_loader):
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                predicted_labels.extend(predicted.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())
        accuracy = accuracy_score(true_labels, predicted_labels)
        if accuracy > accuracy_max:
            accuracy_max = accuracy
            torch.save(model.state_dict(), os.path.join(save_dir_model, "Amodel_temp_max.pth"))

        precision = precision_score(true_labels, predicted_labels, average='weighted', zero_division=1)
        recall = recall_score(true_labels, predicted_labels, average='weighted')
        f1 = f1_score(true_labels, predicted_labels, average='weighted')
        fpr_score = fpr(true_labels, predicted_labels)

        print(f"\nEpoch {epoch + 1}/{epochs}:Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}, FPR: {fpr_score:.4f}")

def test(classifierHead):
    print("Testing")
    predicted_labels = []
    true_labels = []
    model_test = encoder(img_size=28, patch_size=4, in_chans=1, embed_dim=128,
                      encoder_depth=8, num_heads=8, mlp_ratio=3.)

    model = MTMAE_finetuneModel(model_test, classifierHead)
    model.load_state_dict(torch.load(os.path.join(save_dir_model, "Amodel_temp_max.pth")), strict=False)
    model = model.to(device)
    # begin = time.time()
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            total += len(labels)
            images, labels = images.cuda(), labels.cuda()
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            predicted_labels.extend(predicted.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
    # 评估模型
    accuracy = accuracy_score(true_labels, predicted_labels)
    fpr_score = fpr(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels, average='weighted')
    recall = recall_score(true_labels, predicted_labels, average='weighted')
    f1 = f1_score(true_labels, predicted_labels, average='weighted')

    print(f'Accuracy: {accuracy:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1 Score: {f1:.4f}')
    print(f'FPR: {fpr_score:.4f}')

    return accuracy, precision, recall, f1

    # end = time.time()
    # print(f'infer time (ms) per sample: {(end - begin) / total * 1000}')


if __name__ == '__main__':
    num_classes = 4
    batch_size = 256
    epochs = 5 # ###################
    lr = 5e-6
    weight_decay = 0.05
    warmup_epochs = 1  # ###################
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_dir_run = "./runs"
    save_dir_model = "./weights"
    train_path = r'./datasets/finetune_dataset_med_train.tsv'
    val_path = r'./datasets/finetune_dataset_med_val.tsv'
    test_path = r'./datasets/finetune_dataset_med_test.tsv'

    mean = [0.5]
    std = [0.5]

    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    
    acc_list = []
    pre_list = []
    rec_list = []
    f1_list = []
    for seed in [1, 2, 3]:
        set_seed(seed)
        train_subset = ViTDataSet(train_path, transform, is_train=True)
        # train_subset, _ = random_split(train_subset, [19095, len(train_subset) - 19095])
        val_subset = ViTDataSet(val_path, transform)
        test_set = ViTDataSet(test_path, transform)

        data_loader = Data.DataLoader(dataset=train_subset, batch_size=batch_size, shuffle=True)
        val_loader = Data.DataLoader(dataset=val_subset, batch_size=batch_size, shuffle=False)
        test_loader = Data.DataLoader(dataset=test_set, batch_size=256, shuffle=False)
        # ---------------------------------------------------------------------------------------------------------------------#

        classifierHead = Classifier(latent_dim=128, cls_dim=num_classes)  # 修改cls-dim
        set_seed(seed)
        finetune(classifierHead)
        acc, pre, rec, f1 = test(classifierHead)
        acc_list.append(acc)
        pre_list.append(pre)
        rec_list.append(rec)
        f1_list.append(f1)
    
    print(f'Average Accuracy: {np.mean(acc_list):.4f} +- {np.std(acc_list):.4f}')
    print(f'Average Precision: {np.mean(pre_list):.4f} +- {np.std(pre_list):.4f}')
    print(f'Average Recall: {np.mean(rec_list):.4f} +- {np.std(rec_list):.4f}')
    print(f'Average F1: {np.mean(f1_list):.4f} +- {np.std(f1_list):.4f}')

