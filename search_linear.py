"""Single-layer linear model hyperparameter search for Kaggle house price."""
import hashlib, os, tarfile, zipfile, requests, sys, io
import numpy as np, pandas as pd
import torch
from torch import nn

if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

DATA_HUB = dict()
DATA_URL = 'http://d2l-data.s3-accelerate.amazonaws.com/'

def download(name, cache_dir=os.path.join('..', 'data')):
    url, sha1_hash = DATA_HUB[name]
    os.makedirs(cache_dir, exist_ok=True)
    fname = os.path.join(cache_dir, url.split('/')[-1])
    if os.path.exists(fname):
        sha1 = hashlib.sha1()
        with open(fname, 'rb') as f:
            while True:
                data = f.read(1048576)
                if not data: break
                sha1.update(data)
        if sha1.hexdigest() == sha1_hash:
            return fname
    print(f'Downloading {url}...')
    r = requests.get(url, stream=True, verify=True)
    with open(fname, 'wb') as f:
        f.write(r.content)
    return fname

DATA_HUB['kaggle_house_train'] = (DATA_URL + 'kaggle_house_pred_train.csv', '585e9cc93e70b39160e7921475f9bcd7d31219ce')
DATA_HUB['kaggle_house_test'] = (DATA_URL + 'kaggle_house_pred_test.csv', 'fa19780a7b011d9b009e8bff8e99922a8ee2eb90')

print("Loading data...")
train_data = pd.read_csv(download('kaggle_house_train'))
test_data = pd.read_csv(download('kaggle_house_test'))
all_features = pd.concat((train_data.iloc[:, 1:-1], test_data.iloc[:, 1:]))
numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index
all_features[numeric_features] = all_features[numeric_features].apply(lambda x: (x - x.mean()) / (x.std()))
all_features[numeric_features] = all_features[numeric_features].fillna(0)
all_features = pd.get_dummies(all_features, dummy_na=True)
n_train = train_data.shape[0]
train_features = torch.tensor(all_features[:n_train].values.astype(float), dtype=torch.float32)
test_features = torch.tensor(all_features[n_train:].values.astype(float), dtype=torch.float32)
train_labels = torch.tensor(train_data.SalePrice.values.astype(float).reshape(-1, 1), dtype=torch.float32)
in_features = train_features.shape[1]
print(f"Features: {in_features}, Train: {n_train}")

# ── Single-layer model (fixed) ──
loss = nn.MSELoss()

def get_net():
    return nn.Sequential(nn.Linear(in_features, 1))

def log_rmse(net, features, labels):
    with torch.no_grad():
        clipped_preds = torch.clamp(net(features), 1, float('inf'))
        return torch.sqrt(loss(torch.log(clipped_preds), torch.log(labels))).item()

def train(net, train_features, train_labels, test_features, test_labels,
          num_epochs, learning_rate, weight_decay, batch_size):
    train_ls, test_ls = [], []
    train_iter = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(train_features, train_labels),
        batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=weight_decay)
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

def k_fold(k, X_train, y_train, num_epochs, learning_rate, weight_decay, batch_size):
    vl = []
    for i in range(k):
        data = get_k_fold_data(k, i, X_train, y_train)
        net = get_net()
        _, valid_ls = train(net, *data, num_epochs, learning_rate, weight_decay, batch_size)
        vl.append(valid_ls[-1])
    return sum(vl) / len(vl)

# ── Search grid ──
k = 5
lrs = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 3.0, 5.0]
wds = [0, 0.1, 1, 5, 10, 50]
epochs_opts = [100, 300, 500]
batch_sizes = [64, 128]

# Limit grid to reasonable size
import itertools
configs = list(itertools.product(lrs, wds, epochs_opts, batch_sizes))
print(f"\nTotal configs: {len(configs)}")

best_valid = float('inf')
best_config = None
results = []

for idx, (lr_val, wd, epochs, bs) in enumerate(configs):
    try:
        valid = k_fold(k, train_features, train_labels, epochs, lr_val, wd, bs)
        results.append((valid, lr_val, wd, epochs, bs))
        marker = "  << BEST" if valid < best_valid else ""
        print(f"[{idx+1:4d}/{len(configs)}] valid={valid:.6f} | lr={lr_val}, wd={wd}, ep={epochs}, bs={bs}{marker}")
        sys.stdout.flush()
        if valid < best_valid:
            best_valid = valid
            best_config = (lr_val, wd, epochs, bs)
    except Exception as e:
        print(f"[{idx+1:4d}/{len(configs)}] ERROR: {e} | lr={lr_val}, wd={wd}, ep={epochs}, bs={bs}")
        sys.stdout.flush()

print(f"\n{'='*60}")
print(f"BEST: valid={best_valid:.6f}")
print(f"lr={best_config[0]}, wd={best_config[1]}, epochs={best_config[2]}, bs={best_config[3]}")
print(f"{'='*60}")

results.sort(key=lambda x: x[0])
print("\nTop-10:")
for i, (v, lr_val, wd, ep, bs) in enumerate(results[:10]):
    print(f"  {i+1}. valid={v:.6f} | lr={lr_val}, wd={wd}, ep={ep}, bs={bs}")
