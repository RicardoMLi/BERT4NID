import torch
import os
import time
import pandas as pd
import numpy as np

from sklearn import preprocessing
from sklearn.model_selection import train_test_split


def min_max_norm(df, name):
    x = df[name].values.reshape(-1, 1)
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    df[name] = x_scaled


def data_preprocess(df):
    
    for feature_id in range(0, len(df.columns.values) - 1):
        min_max_norm(df, df.columns[feature_id])

    df_Y = df.pop('label').values
    return df, df_Y

def data_preprocess_ctu(df):
    df_Y = df.pop('label').values
    for label in ['src_ip', 'dst_ip', 'src_port', 'dst_port', 'timestamp']:
        df.drop(label, axis=1, inplace=True)
    
    for i in range(1, len(df.columns.values)):
        min_max_norm(df, df.columns.values[i])

    df = pd.get_dummies(df, columns=['protocol'])
    return df, df_Y


def data_preprocess_edge(df):
    df.drop('src_ip', axis=1, inplace=True)
    df.drop('dst_ip', axis=1, inplace=True)
    df.drop('src_port', axis=1, inplace=True)
    df.drop('dst_port', axis=1, inplace=True)
    df.drop('timestamp', axis=1, inplace=True)

    norm_cols = df.columns.values[1:-2]
    for feature_id in norm_cols:
        min_max_norm(df, feature_id)

    return df


base_csv_path = r'F:\入侵检测数据集\Kitsune_processed'
df = pd.DataFrame()
for csv_path in os.listdir(base_csv_path):
    data = pd.read_csv(os.path.join(base_csv_path, csv_path))
    df = pd.concat([data, df], axis=0)

# df = pd.read_csv(r'E:\projects\datasets\data_ctu.csv')
data, Y = data_preprocess(df)
# data = data.sample(frac=1)
X = data.values.astype(np.float32)
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
tr_samples, tr_labels = [], []
for index in range(len(x_train)):
    tr_samples.append(torch.from_numpy(x_train[index]))
    tr_labels.append(y_train[index])

tr_samples = torch.stack(tr_samples)
tr_labels = torch.tensor(tr_labels)


te_samples, te_labels = [], []
for index in range(len(x_test)):
    te_samples.append(torch.from_numpy(x_test[index]))
    te_labels.append(y_test[index])

te_samples = torch.stack(te_samples)
te_labels = torch.tensor(te_labels)

feature_num = len(tr_samples[0])
label_num = max(tr_labels).item() + 1
print(f"feature_num is {feature_num}, label_num is {label_num}")

