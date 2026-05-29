from pyexpat import model
import random
import argparse
import numpy as np
import torch
import tqdm
import os
import sys
import torch.nn as nn
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from uer.layers import str2embedding
from sklearn.manifold import TSNE
from utils.constants import *
from uer.encoders import str2encoder
from uer.utils.constants import CLS_TOKEN, SEP_TOKEN
from uer.utils import str2optimizer, str2scheduler, str2tokenizer
from uer.utils.config import load_hyperparam
from uer.utils.seed import set_seed
from uer.model_saver import save_model
from uer.opts import finetune_opts
from sklearn.metrics import precision_score, recall_score, f1_score

class Classifier(nn.Module):
    def __init__(self, args):
        super(Classifier, self).__init__()
        self.embedding = str2embedding[args.embedding](args, len(args.tokenizer.vocab))
        self.encoder = str2encoder[args.encoder](args)
        self.labels_num = args.labels_num
        self.pooling = args.pooling
        self.soft_targets = args.soft_targets
        self.soft_alpha = args.soft_alpha
        self.output_layer_1 = nn.Linear(args.hidden_size, args.hidden_size)
        self.output_layer_2 = nn.Linear(args.hidden_size, self.labels_num)

    def forward_features(self, src, tgt, seg, soft_tgt=None):
        # Embedding.
        emb = self.embedding(src, seg)
        # Encoder.
        output = self.encoder(emb, seg)
        # Target.
        if self.pooling == "mean":
            output = torch.mean(output, dim=1)
        elif self.pooling == "max":
            output = torch.max(output, dim=1)[0]
        elif self.pooling == "last":
            output = output[:, -1, :]
        else:
            output = output[:, 0, :]

        output = torch.tanh(self.output_layer_1(output))

        return output

    def forward(self, src, tgt, seg, soft_tgt=None):
        """
        Args:
            src: [batch_size x seq_length]
            tgt: [batch_size]
            seg: [batch_size x seq_length]
        """
        # Embedding.
        emb = self.embedding(src, seg)
        # Encoder.
        output = self.encoder(emb, seg)
        # Target.
        if self.pooling == "mean":
            output = torch.mean(output, dim=1)
        elif self.pooling == "max":
            output = torch.max(output, dim=1)[0]
        elif self.pooling == "last":
            output = output[:, -1, :]
        else:
            output = output[:, 0, :]

        output_1 = torch.tanh(self.output_layer_1(output))
        logits = self.output_layer_2(output_1)
        if tgt is not None:
            if self.soft_targets and soft_tgt is not None:
                loss = self.soft_alpha * nn.MSELoss()(logits, soft_tgt) + \
                       (1 - self.soft_alpha) * nn.NLLLoss()(nn.LogSoftmax(dim=-1)(logits), tgt.view(-1))
            else:
                loss = nn.NLLLoss()(nn.LogSoftmax(dim=-1)(logits), tgt.view(-1))
            return loss, logits
        else:
            return None, logits


def count_labels_num(path):
    labels_set, columns = set(), {}
    with open(path, mode="r", encoding="utf-8") as f:
        for line_id, line in enumerate(f):
            if line_id == 0:
                for i, column_name in enumerate(line.strip().replace('"', '').split("\t")):
                    columns[column_name] = i
                continue
            line = line.strip().replace('"', '').split("\t")
            label = int(line[columns["label"]])
            labels_set.add(label)
    return len(labels_set)


def load_or_initialize_parameters(args, model):
    if args.pretrained_model_path is not None:
        # Initialize with pretrained model.
        model.load_state_dict(torch.load(args.pretrained_model_path, map_location={'cuda:1': 'cuda:0', 'cuda:2': 'cuda:0',
                                                                                   'cuda:3': 'cuda:0'}), strict=False)
    else:
        # Initialize with normal distribution.
        for n, p in list(model.named_parameters()):
            if "gamma" not in n and "beta" not in n:
                p.data.normal_(0, 0.02)


def build_optimizer(args, model):
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
                {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.01},
                {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.0}
    ]
    if args.optimizer in ["adamw"]:
        optimizer = str2optimizer[args.optimizer](optimizer_grouped_parameters, lr=args.learning_rate, correct_bias=False)
    else:
        optimizer = str2optimizer[args.optimizer](optimizer_grouped_parameters, lr=args.learning_rate,
                                                  scale_parameter=False, relative_step=False)
    if args.scheduler in ["constant"]:
        scheduler = str2scheduler[args.scheduler](optimizer)
    elif args.scheduler in ["constant_with_warmup"]:
        scheduler = str2scheduler[args.scheduler](optimizer, args.train_steps*args.warmup)
    else:
        scheduler = str2scheduler[args.scheduler](optimizer, args.train_steps*args.warmup, args.train_steps)
    return optimizer, scheduler


def batch_loader(batch_size, src, tgt, seg, soft_tgt=None):
    instances_num = src.size()[0]
    for i in range(instances_num // batch_size):
        src_batch = src[i * batch_size: (i + 1) * batch_size, :]
        tgt_batch = tgt[i * batch_size: (i + 1) * batch_size]
        seg_batch = seg[i * batch_size: (i + 1) * batch_size, :]
        if soft_tgt is not None:
            soft_tgt_batch = soft_tgt[i * batch_size : (i + 1) * batch_size, :]
            yield src_batch, tgt_batch, seg_batch, soft_tgt_batch
        else:
            yield src_batch, tgt_batch, seg_batch, None
    if instances_num > instances_num // batch_size * batch_size:
        src_batch = src[instances_num // batch_size * batch_size:, :]
        tgt_batch = tgt[instances_num // batch_size * batch_size:]
        seg_batch = seg[instances_num // batch_size * batch_size:, :]
        if soft_tgt is not None:
            soft_tgt_batch = soft_tgt[instances_num // batch_size * batch_size:, :]
            yield src_batch, tgt_batch, seg_batch, soft_tgt_batch
        else:
            yield src_batch, tgt_batch, seg_batch, None


def read_dataset(args, path):
    dataset, columns = [], {}
    with open(path, mode="r", encoding="utf-8") as f:
        for line_id, line in enumerate(f):
            if line_id == 0:
                for i, column_name in enumerate(line.strip().replace('"', '').split("\t")):
                    columns[column_name] = i
                continue
            line = line[:-1].replace('"', '').split("\t")
            tgt = int(line[columns["label"]])
            if args.soft_targets and "logits" in columns.keys():
                soft_tgt = [float(value) for value in line[columns["logits"]].split(" ")]
            if "text_b" not in columns:  # Sentence classification.
                text_a = line[columns["text_a"]]
                src = args.tokenizer.convert_tokens_to_ids([CLS_TOKEN] + args.tokenizer.tokenize(text_a))
                seg = [1] * len(src)
            else:  # Sentence-pair classification.
                text_a, text_b = line[columns["text_a"]], line[columns["text_b"]]
                src_a = args.tokenizer.convert_tokens_to_ids([CLS_TOKEN] + args.tokenizer.tokenize(text_a) + [SEP_TOKEN])
                src_b = args.tokenizer.convert_tokens_to_ids(args.tokenizer.tokenize(text_b) + [SEP_TOKEN])
                src = src_a + src_b
                seg = [1] * len(src_a) + [2] * len(src_b)

            if len(src) > args.seq_length:
                src = src[: args.seq_length]
                seg = seg[: args.seq_length]
            while len(src) < args.seq_length:
                src.append(0)
                seg.append(0)
            if args.soft_targets and "logits" in columns.keys():
                dataset.append((src, tgt, seg, soft_tgt))
            else:
                dataset.append((src, tgt, seg))

    return dataset


def train_model(args, model, optimizer, scheduler, src_batch, tgt_batch, seg_batch, soft_tgt_batch=None):
    model.zero_grad()

    src_batch = src_batch.to(args.device)
    tgt_batch = tgt_batch.to(args.device)
    seg_batch = seg_batch.to(args.device)
    if soft_tgt_batch is not None:
        soft_tgt_batch = soft_tgt_batch.to(args.device)

    loss, _ = model(src_batch, tgt_batch, seg_batch, soft_tgt_batch)
    if torch.cuda.device_count() > 1:
        loss = torch.mean(loss)

    if args.fp16:
        with args.amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
    else:
        loss.backward()

    optimizer.step()
    scheduler.step()

    return loss


def evaluate(args, dataset, print_confusion_matrix=False, plot_tsne=False):
    src = torch.LongTensor([sample[0] for sample in dataset])
    tgt = torch.LongTensor([sample[1] for sample in dataset])
    seg = torch.LongTensor([sample[2] for sample in dataset])

    batch_size = args.batch_size

    correct = 0
    # Confusion matrix.
    confusion = torch.zeros(args.labels_num, args.labels_num, dtype=torch.long)
    true_labels = []
    predict_labels = []
    srcs = []
    # features = []
    args.model.eval()

    for i, (src_batch, tgt_batch, seg_batch, _) in enumerate(batch_loader(batch_size, src, tgt, seg)):
        if plot_tsne:
            srcs.extend(src_batch.numpy())
        src_batch = src_batch.to(args.device)
        tgt_batch = tgt_batch.to(args.device)
        seg_batch = seg_batch.to(args.device)
        with torch.no_grad():
            _, logits = args.model(src_batch, tgt_batch, seg_batch)
            # feature = args.model.module.forward_features(src_batch, tgt_batch, seg_batch)
        # features.extend(feature.cpu().numpy())
        true_labels.extend(tgt_batch.cpu().numpy())
        pred = torch.argmax(nn.Softmax(dim=1)(logits), dim=1)
        predict_labels.extend(pred.cpu().numpy())
        gold = tgt_batch
        for j in range(pred.size()[0]):
            confusion[pred[j], gold[j]] += 1
        correct += torch.sum(pred == gold).item()

    # if plot_tsne:
    #     # 随机挑选1000个样本采用t-sne可视化
    #     colors = ['red', 'blue', 'black', 'green', 'yellow', 'grey', 'orange', 'purple', 'gold', 'olivedrab', 'fuchsia',
    #               'deeppink', 'peru', 'tan', 'burlywood', 'lime', 'navy', 'crimson', 'lightgreen', 'cyan']
    #     inverse_mapper = {v: k for k, v in UTSC_MAPPER.items()}
    #     sample_index = np.random.choice(len(srcs), size=700, replace=False)
    #     src_tsne = [srcs[index] for index in sample_index]
    #     feature_tsne = [features[index] for index in sample_index]
    #     label_tsne = [true_labels[index] for index in sample_index]
    #     embedded_src = TSNE(n_components=3, init='pca').fit_transform(src_tsne)
    #     embedded_feature = TSNE(n_components=3, init='pca').fit_transform(feature_tsne)

    #     plt.figure(figsize=(7, 7))
    #     ax = plt.axes(projection='3d')
    #     x_min, x_max = np.min(embedded_src, 0), np.max(embedded_src, 0)
    #     embedded_src = embedded_src / (x_max - x_min)
    #     # 将数据对应坐标输入到figure中，不同标签取不同的颜色
    #     for label, color in zip(np.arange(args.labels_num), colors):
    #         subset = embedded_src[np.array(label_tsne) == label]
    #         ax.scatter(subset[:, 0], subset[:, 1], subset[:, 2], label=f'Class {inverse_mapper[label]}', c=color)

    #     ax.set_title('Original data on USTC-TFC')
    #     ax.set_xlabel('t-SNE Component 1')
    #     ax.set_ylabel('t-SNE Component 2')
    #     ax.set_zlabel('t-SNE Component 3')
    #     plt.legend()
    #     plt.savefig('original.svg', dpi=500, format="svg")
    #     plt.close()

    #     plt.figure(figsize=(7, 7))
    #     ax = plt.axes(projection='3d')
    #     x_min, x_max = np.min(embedded_feature, 0), np.max(embedded_feature, 0)
    #     embedded_feature = embedded_feature / (x_max - x_min)
    #     for label, color in zip(np.arange(args.labels_num), colors):
    #         subset = embedded_feature[np.array(label_tsne) == label]
    #         ax.scatter(subset[:, 0], subset[:, 1], subset[:, 2], label=f'Class {inverse_mapper[label]}', c=color)

    #     ax.set_title('Latent features on USTC-TFC')
    #     ax.set_xlabel('t-SNE Component 1')
    #     ax.set_ylabel('t-SNE Component 2')
    #     ax.set_zlabel('t-SNE Component 3')
    #     plt.legend(loc='upper left')
    #     plt.savefig('feature.svg', dpi=500, format="svg")
    #     plt.close()

    # if print_confusion_matrix:
    #     print("Confusion matrix:")
    #     print(confusion)
    #     cf_array = confusion.numpy()
    #     with open("./confusion_matrix",'w') as f:
    #         for cf_a in cf_array:
    #             f.write(str(cf_a)+'\n')
    #     print("Report precision, recall, and f1:")
    #     eps = 1e-9
    #     for i in range(confusion.size()[0]):
    #         p = confusion[i, i].item() / (confusion[i, :].sum().item() + eps)
    #         r = confusion[i, i].item() / (confusion[:, i].sum().item() + eps)
    #         if (p + r) == 0:
    #             f1 = 0
    #         else:
    #             f1 = 2 * p * r / (p + r)
    #         print("Label {}: {:.3f}, {:.3f}, {:.3f}".format(i, p, r, f1))

    precision = precision_score(true_labels, predict_labels, average="weighted")
    recall = recall_score(true_labels, predict_labels, average="weighted")
    f1 = f1_score(true_labels, predict_labels, average="weighted")
    print("\nAcc. (Correct/Total): {:.4f} ({}/{}) ".format(correct / len(dataset), correct, len(dataset)))
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}")

    return correct / len(dataset), precision, recall, f1


def main(seed):
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    finetune_opts(parser)

    parser.add_argument("--pooling", choices=["mean", "max", "first", "last"], default="first",
                        help="Pooling type.")

    parser.add_argument("--tokenizer", choices=["bert", "char", "space"], default="bert",
                        help="Specify the tokenizer."
                             "Original Google BERT uses bert tokenizer on Chinese corpus."
                             "Char tokenizer segments sentences into characters."
                             "Space tokenizer segments sentences into words according to space."
                             )

    parser.add_argument("--soft_targets", action='store_true',
                        help="Train model with logits.")
    parser.add_argument("--soft_alpha", type=float, default=0.5,
                        help="Weight of the soft targets loss.")

    args = parser.parse_args()

    # Load the hyperparameters from the config file.
    args = load_hyperparam(args)

    set_seed(seed)

    # Count the number of labels.
    args.labels_num = count_labels_num(args.train_path)
    print("labels_num is ", args.labels_num)

    # Build tokenizer.
    args.tokenizer = str2tokenizer[args.tokenizer](args)

    # Build classification model.
    model = Classifier(args)

    # Load or initialize parameters.
    load_or_initialize_parameters(args, model)

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(args.device)
    # Training phase.
    trainset = read_dataset(args, args.train_path)
    random.shuffle(trainset)
    instances_num = len(trainset)
    batch_size = args.batch_size

    src = torch.LongTensor([example[0] for example in trainset])
    tgt = torch.LongTensor([example[1] for example in trainset])
    seg = torch.LongTensor([example[2] for example in trainset])

    if args.soft_targets:
        soft_tgt = torch.FloatTensor([example[3] for example in trainset])
    else:
        soft_tgt = None

    args.train_steps = int(instances_num * args.epochs_num / batch_size) + 1

    print("Batch size: ", batch_size)
    print("The number of training instances:", instances_num)

    optimizer, scheduler = build_optimizer(args, model)

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)
        args.amp = amp

    if torch.cuda.device_count() > 1:
        print("{} GPUs are available. Let's use them.".format(torch.cuda.device_count()))
        model = torch.nn.DataParallel(model)

    args.model = model

    total_loss, result, best_result = 0.0, 0.0, 0.0

    print("Start training.")

    for epoch in tqdm.tqdm(range(1, args.epochs_num + 1)):
        model.train()
        for i, (src_batch, tgt_batch, seg_batch, soft_tgt_batch) in enumerate(batch_loader(batch_size, src, tgt, seg, soft_tgt)):
            loss = train_model(args, model, optimizer, scheduler, src_batch, tgt_batch, seg_batch, soft_tgt_batch)
            total_loss += loss.item()
            if (i + 1) % args.report_steps == 0:
                print("\nEpoch id: {}, Training steps: {}, Avg loss: {:.3f}".format(epoch, i + 1, total_loss / args.report_steps))
                total_loss = 0.0

        result = evaluate(args, read_dataset(args, args.val_path))
        if result[0] > best_result:
            best_result = result[0]
            save_model(model, args.output_model_path)

    # Evaluation phase.
    print("Test set evaluation.")
    if torch.cuda.device_count() > 1:
        model.module.load_state_dict(torch.load(args.output_model_path))
    else:
        model.load_state_dict(torch.load(args.output_model_path))

    return evaluate(args, read_dataset(args, args.test_path))


if __name__ == "__main__":
    accuracy_list = []
    precision_list = []
    recall_list = []
    f1_list = []
    for seed in range(3):
        print(f"Begins {seed + 1} training, random seed is {seed}")
        accuracy, precision, recall, f1 = main(seed)
        accuracy_list.append(accuracy)
        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)

    print(f"Accuracy: {np.mean(accuracy_list)}+-{np.std(accuracy_list)}")
    print(f"Precision: {np.mean(precision_list)}+-{np.std(precision_list)}")
    print(f"Recall: {np.mean(recall_list)}+-{np.std(recall_list)}")
    print(f"F1-score: {np.mean(f1_list)}+-{np.std(f1_list)}")
