import torch
import time
import numpy as np
from torch.nn.functional import cross_entropy
from torch.optim import Adam
from sklearn import metrics
from torch.utils.data import DataLoader


from config import classifier_config
from models.cd_model import CDModel
from logger import Logger


class Classifier:
    def __init__(self, name: str, feature_num: int, label_num: int):
        self.name = f'{name}_classifier'
        self.feature_num = feature_num
        self.label_num = label_num
        self.model = CDModel(feature_num, label_num).to('cuda')
        self.logger = Logger(name)
        self.confusion_matrix: np.ndarray = None
        self.metrics = {
            'Precision': 0.0,
            'Recall': 0.0,
            'F1': 0.0,
            'Accuracy': 0.0,
        }

    def fit(self, dataset):
        self.model.train()
        self.logger.info('Started training')
        self.logger.debug(f'Using device: cuda')
        optimizer = Adam(
            params=self.model.parameters(),
            lr=classifier_config.lr,
        )
        dl = DataLoader(dataset, classifier_config.batch_size, shuffle=True)
        for e in range(classifier_config.epochs):
            for idx, (samples, labels) in enumerate(dl):
                print(f'\repoch {e + 1} / {classifier_config.epochs}: {(idx + 1) / len(dl): .2%}', end='')
                self.model.zero_grad()
                prediction = self.model(samples)[1]
                loss = cross_entropy(
                    input=prediction,
                    target=labels,
                )
                loss.backward()
                optimizer.step()
        self.model.eval()
        self.logger.info('Finished training')

    def predict(self, x: torch.Tensor, use_prob: bool = False) -> torch.Tensor:
        with torch.no_grad():
            prob = self.model(x)[1]
        if use_prob:
            return prob.squeeze(dim=1).detach()
        else:
            return torch.argmax(prob, dim=1)

    def test(self, dataset):
        self.model = self.model.cpu()
        predicted_labels = self.predict(dataset.samples.cpu())
        real_labels = dataset.labels.cpu()
        self.confusion_matrix = metrics.confusion_matrix(
            y_true=real_labels,
            y_pred=predicted_labels,
            labels=[i for i in range(self.label_num)]
        )
        self.metrics['Precision'] = metrics.precision_score(
            y_true=real_labels,
            y_pred=predicted_labels,
            average='weighted',
            zero_division=0,
        )
        self.metrics['Recall'] = metrics.recall_score(
            y_true=real_labels,
            y_pred=predicted_labels,
            average='weighted',
            zero_division=0,
        )
        self.metrics['F1'] = metrics.f1_score(
            y_true=real_labels,
            y_pred=predicted_labels,
            average='weighted',
            zero_division=0,
        )
        self.metrics['Accuracy'] = metrics.accuracy_score(
            y_true=real_labels,
            y_pred=predicted_labels,
        )
        self.model = self.model.to('cuda')

    def binary_test(self, dataset):
        self.model = self.model.cpu()
        predicted_labels = self.predict(dataset.samples.cpu())
        real_labels = dataset.labels.cpu()
        for idx, item in enumerate(predicted_labels):
            if item > 0:
                predicted_labels[idx] = 1
        for idx, item in enumerate(real_labels):
            if item > 0:
                real_labels[idx] = 1
        self.confusion_matrix = metrics.confusion_matrix(
            y_true=real_labels,
            y_pred=predicted_labels,
        )
        self.metrics['Precision'] = metrics.precision_score(
            y_true=real_labels,
            y_pred=predicted_labels,
            average='macro',
            zero_division=0,
        )
        self.metrics['Recall'] = metrics.recall_score(
            y_true=real_labels,
            y_pred=predicted_labels,
            average='macro',
            zero_division=0,
        )
        self.metrics['F1'] = metrics.f1_score(
            y_true=real_labels,
            y_pred=predicted_labels,
            average='macro',
            zero_division=0,
        )
        self.metrics['Accuracy'] = metrics.accuracy_score(
            y_true=real_labels,
            y_pred=predicted_labels,
        )
        self.model = self.model.to('cuda')
