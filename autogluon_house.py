"""AutoGluon for Kaggle house price - medium quality, faster."""
import hashlib, os, sys, io
import numpy as np, pandas as pd

if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from autogluon.tabular import TabularDataset, TabularPredictor

# ── Download ──
DATA_HUB = dict()
DATA_URL = 'http://d2l-data.s3-accelerate.amazonaws.com/'

def download(name, cache_dir=os.path.join('..', 'data')):
    import requests
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

# ── Load & preprocess ──
print("Loading data...")
train_data = pd.read_csv(download('kaggle_house_train'))
test_data = pd.read_csv(download('kaggle_house_test'))

train_features_raw = train_data.iloc[:, 1:-1]
train_labels = train_data['SalePrice']
test_features_raw = test_data.iloc[:, 1:]

all_features = pd.concat((train_features_raw, test_features_raw), axis=0)
numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index
all_features[numeric_features] = all_features[numeric_features].apply(
    lambda x: (x - x.mean()) / (x.std()))
all_features[numeric_features] = all_features[numeric_features].fillna(0)
all_features = pd.get_dummies(all_features, dummy_na=True)

n_train = train_data.shape[0]
train_df = all_features[:n_train].copy()
train_df['SalePrice'] = train_labels.values
test_df = all_features[n_train:].copy()
print(f"Train: {train_df.shape}, Test: {test_df.shape}")

# ── 5-fold CV ──
k = 5
fold_size = n_train // k
cv_scores = []

print(f"\n{'='*60}")
print("5-Fold CV (log RMSE)")
print(f"{'='*60}")

for i in range(k):
    print(f"\n--- Fold {i+1}/{k} ---", flush=True)

    idx_start = i * fold_size
    idx_end = (i + 1) * fold_size
    valid_idx = list(range(idx_start, idx_end))
    train_idx = [j for j in range(n_train) if j not in valid_idx]

    train_fold = train_df.iloc[train_idx].copy()
    valid_fold = train_df.iloc[valid_idx].copy()

    # Log-transform target
    train_fold['SalePrice'] = np.log(train_fold['SalePrice'])

    predictor = TabularPredictor(
        label='SalePrice',
        eval_metric='root_mean_squared_error',
        problem_type='regression',
        verbosity=0,
    ).fit(
        train_data=TabularDataset(train_fold),
        presets='medium_quality',
        time_limit=120,
    )

    preds_log = predictor.predict(TabularDataset(valid_fold.drop(columns=['SalePrice'])))
    valid_labels_log = np.log(valid_fold['SalePrice'].values)
    log_rmse = np.sqrt(np.mean((preds_log - valid_labels_log) ** 2))
    cv_scores.append(log_rmse)
    print(f"Fold {i+1} log_rmse={log_rmse:.6f}", flush=True)

mean_cv = np.mean(cv_scores)
std_cv = np.std(cv_scores)
print(f"\nCV: {mean_cv:.6f} +/- {std_cv:.6f}")

# ── Leaderboard (last fold) ──
print(f"\n--- Model Leaderboard (last fold) ---")
lb = predictor.leaderboard(silent=True)
print(lb.to_string())

# ── Final train ──
print(f"\n{'='*60}")
print("Final model on all data...")
print(f"{'='*60}")

train_all = train_df.copy()
train_all['SalePrice'] = np.log(train_df['SalePrice'])

final_predictor = TabularPredictor(
    label='SalePrice',
    eval_metric='root_mean_squared_error',
    problem_type='regression',
    verbosity=1,
).fit(
    train_data=TabularDataset(train_all),
    presets='medium_quality',
    time_limit=300,
)

preds_log = final_predictor.predict(TabularDataset(test_df))
preds = np.exp(preds_log)

submission = pd.DataFrame({'Id': test_data['Id'], 'SalePrice': preds})
submission.to_csv('submission_autogluon.csv', index=False)
print(f"Saved submission_autogluon.csv")
print(f"\nFinal Leaderboard:")
print(final_predictor.leaderboard(silent=True).to_string())
