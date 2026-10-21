"""Hyperparameter search for Kaggle house price prediction."""
import hashlib
import os
import tarfile
import zipfile
import requests
import numpy as np
import pandas as pd
import torch
from torch import nn
import sys

# Force UTF-8 encoding for stdout
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# ── Replicate notebook utility functions ──────────────────────────────
DATA_HUB = dict()
DATA_URL = 'http://d2l-data.s3-accelerate.amazonaws.com/'

def download(name, cache_dir=os.path.join('..', 'data')):
    assert name in DATA_HUB, f"{name} not in DATA_HUB"
    url, sha1_hash = DATA_HUB[name]
    os.makedirs(cache_dir, exist_ok=True)
    fname = os.path.join(cache_dir, url.split('/')[-1])
    if os.path.exists(fname):
        sha1 = hashlib.sha1()
        with open(fname, 'rb') as f:
            while True:
                data = f.read(1048576)
                if not data:
                    break
                sha1.update(data)
        if sha1.hexdigest() == sha1_hash:
            return fname
    print(f'Downloading from {url} ...')
    r = requests.get(url, stream=True, verify=True)
    with open(fname, 'wb') as f:
        f.write(r.content)
    return fname

# Register datasets
DATA_HUB['kaggle_house_train'] = (
    DATA_URL + 'kaggle_house_pred_train.csv',
    '585e9cc93e70b39160e7921475f9bcd7d31219ce')
DATA_HUB['kaggle_house_test'] = (
    DATA_URL + 'kaggle_house_pred_test.csv',
    'fa19780a7b011d9b009e8bff8e99922a8ee2eb90')

# ── Load & preprocess data ────────────────────────────────────────────
print("Loading data...")
train_data = pd.read_csv(download('kaggle_house_train'))
test_data = pd.read_csv(download('kaggle_house_test'))
print(f"Train: {train_data.shape}, Test: {test_data.shape}")

all_features = pd.concat((train_data.iloc[:, 1:-1], test_data.iloc[:, 1:]))

numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index
all_features[numeric_features] = all_features[numeric_features].apply(
    lambda x: (x - x.mean()) / (x.std()))
all_features[numeric_features] = all_features[numeric_features].fillna(0)

all_features = pd.get_dummies(all_features, dummy_na=True)

n_train = train_data.shape[0]
train_features = torch.tensor(
    all_features[:n_train].values.astype(float), dtype=torch.float32)
test_features = torch.tensor(
    all_features[n_train:].values.astype(float), dtype=torch.float32)
train_labels = torch.tensor(
    train_data.SalePrice.values.astype(float).reshape(-1, 1), dtype=torch.float32)

in_features = train_features.shape[1]
print(f"Features: {in_features}, Train samples: {n_train}")

# ── Core functions ────────────────────────────────────────────────────
loss = nn.MSELoss()

def get_net(hidden_layers, dropout):
    layers = []
    prev = in_features
    for h in hidden_layers:
        layers.append(nn.Linear(prev, h))
        layers.append(nn.ReLU())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        prev = h
    layers.append(nn.Linear(prev, 1))
    return nn.Sequential(*layers)

def log_rmse(net, features, labels):
    with torch.no_grad():
        clipped_preds = torch.clamp(net(features), 1, float('inf'))
        rmse = torch.sqrt(loss(torch.log(clipped_preds), torch.log(labels)))
    return rmse.item()

def train(net, train_features, train_labels, test_features, test_labels,
          num_epochs, learning_rate, weight_decay, batch_size):
    train_ls, test_ls = [], []
    train_iter = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(train_features, train_labels),
        batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate,
                                 weight_decay=weight_decay)
    for epoch in range(num_epochs):
        for X, y in train_iter:
            optimizer.zero_grad()
            l = loss(net(X), y)
            l.backward()
            optimizer.step()
        train_ls.append(log_rmse(net, train_features, train_labels))
        if test_labels is not None:
            test_ls.append(log_rmse(net, test_features, test_labels))
    return train_ls, test_ls

def get_k_fold_data(k, i, X, y):
    assert k > 1
    fold_size = X.shape[0] // k
    X_train, y_train = None, None
    for j in range(k):
        idx = slice(j * fold_size, (j + 1) * fold_size)
        X_part, y_part = X[idx, :], y[idx]
        if j == i:
            X_valid, y_valid = X_part, y_part
        elif X_train is None:
            X_train, y_train = X_part, y_part
        else:
            X_train = torch.cat([X_train, X_part], 0)
            y_train = torch.cat([y_train, y_part], 0)
    return X_train, y_train, X_valid, y_valid

def k_fold(k, X_train, y_train, num_epochs, learning_rate, weight_decay,
           batch_size, hidden_layers, dropout):
    valid_ls_list = []
    for i in range(k):
        data = get_k_fold_data(k, i, X_train, y_train)
        net = get_net(hidden_layers, dropout)
        train_ls, valid_ls = train(net, *data, num_epochs, learning_rate,
                                   weight_decay, batch_size)
        valid_ls_list.append(valid_ls[-1])
    return sum(valid_ls_list) / len(valid_ls_list)

# ── Search space ──────────────────────────────────────────────────────
k = 5

configs = [
    # (hidden_layers, dropout, lr, weight_decay, num_epochs, batch_size)
    # ── Single layer baseline ──
    ([],       0.0,  0.01,  0,    200,  64),
    ([],       0.0,  0.05,  0,    200,  64),
    ([],       0.0,  0.005, 0,    200,  64),
    # ── Small MLP ──
    ([128, 32], 0.3, 0.01,  5,    300,  64),
    ([128, 32], 0.3, 0.005, 5,    300,  64),
    ([128, 32], 0.3, 0.01,  10,   300,  64),
    ([128, 32], 0.3, 0.005, 10,   500,  64),
    ([64, 16],  0.3, 0.01,  5,    300,  64),
    ([64, 16],  0.3, 0.005, 5,    500,  64),
    # ── Medium MLP ──
    ([256, 64], 0.3, 0.01,  5,    300,  64),
    ([256, 64], 0.3, 0.005, 10,   500,  64),
    ([256, 64], 0.4, 0.005, 10,   500,  64),
    ([256, 128, 64], 0.3, 0.005, 10, 500, 64),
    ([256, 128, 64], 0.3, 0.001, 10, 500, 64),
    ([256, 128, 64], 0.4, 0.005, 10, 500, 128),
    ([256, 128, 64], 0.4, 0.001, 10, 700, 64),
    # ── Single hidden layer ──
    ([128],     0.3, 0.01,  5,    300,  64),
    ([128],     0.3, 0.005, 5,    500,  64),
    ([256],     0.3, 0.01,  5,    300,  64),
    ([256],     0.3, 0.005, 5,    500,  64),
    # ── Low lr fine-tuning ──
    ([256, 64], 0.3, 0.001, 5,    500,  64),
    ([256, 64], 0.3, 0.001, 10,   700,  64),
    ([256, 128, 64], 0.4, 0.003, 10, 500, 64),
]

print(f"\n{'='*80}")
print(f"Total configs to test: {len(configs)}")
print(f"{'='*80}\n")

best_valid = float('inf')
best_config = None
results = []

for idx, (hidden, do, lr_val, wd, epochs, bs) in enumerate(configs):
    desc = f"hidden={hidden}, do={do}, lr={lr_val}, wd={wd}, ep={epochs}, bs={bs}"
    try:
        valid = k_fold(k, train_features, train_labels, epochs, lr_val, wd,
                       bs, hidden, do)
        results.append((valid, desc, hidden, do, lr_val, wd, epochs, bs))
        marker = "  << NEW BEST" if valid < best_valid else ""
        print(f"[{idx+1:2d}/{len(configs)}] valid={valid:.6f}  | {desc}{marker}")
        sys.stdout.flush()
        if valid < best_valid:
            best_valid = valid
            best_config = (hidden, do, lr_val, wd, epochs, bs)
    except Exception as e:
        print(f"[{idx+1:2d}/{len(configs)}] ERROR: {e} | {desc}")
        sys.stdout.flush()

print(f"\n{'='*80}")
print(f"BEST VALID LOSS: {best_valid:.6f}")
print(f"Config: hidden={best_config[0]}, dropout={best_config[1]}, lr={best_config[2]}, "
      f"weight_decay={best_config[3]}, epochs={best_config[4]}, batch_size={best_config[5]}")
print(f"{'='*80}")

# ── Top-5 results ──
results.sort(key=lambda x: x[0])
print("\nTop-5 Results:")
for i, (v, d, *_) in enumerate(results[:5]):
    print(f"  {i+1}. valid={v:.6f}  | {d}")
print()
