import os
import random
import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

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

    norm_cols = df.columns.values[:-1]
    for feature_id in norm_cols:
        min_max_norm(df, feature_id)
    
    return df

base_path = r'F:\入侵检测数据集\kitsune_processed'
dataframe = pd.DataFrame()
for csv_path in os.listdir(base_path):
    if csv_path.endswith('.csv'):
        data = pd.read_csv(os.path.join(base_path, csv_path))
        dataframe = pd.concat([data, dataframe], axis=0)

dataframe = dataframe.sample(frac=1)
labels = dataframe.pop('label').values
dataframe = preprocess_ustc(dataframe).astype(np.float32)
x_train, x_test, y_train, y_test = train_test_split(dataframe, labels, test_size=0.15)

acc_list = []
pre_list = []
rec_list = []
f1_list = []
for seed in [1, 2, 3]:
    set_seed(seed)
    # model = KNeighborsClassifier()
    # model = DecisionTreeClassifier()
    model = RandomForestClassifier()
    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    acc = accuracy_score(y_test, pred)
    pre = precision_score(y_test, pred, average='weighted')
    rec = recall_score(y_test, pred, average='weighted')
    f1 = f1_score(y_test, pred, average='weighted')
    acc_list.append(acc)
    pre_list.append(pre)
    rec_list.append(rec)
    f1_list.append(f1)

print(f'Average Accuracy: {np.mean(acc_list):.4f} +- {np.std(acc_list):.4f}')
print(f'Average Precision: {np.mean(pre_list):.4f} +- {np.std(pre_list):.4f}')
print(f'Average Recall: {np.mean(rec_list):.4f} +- {np.std(rec_list):.4f}')
print(f'Average F1: {np.mean(f1_list):.4f} +- {np.std(f1_list):.4f}')
