import numpy as np
import os
import json
import time
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score

PROCESSED_PATH = os.path.expanduser(
    "~/Desktop/sleeping-staging/preprocessing/processed")
GRID_PATH = os.path.expanduser(
    "~/Desktop/sleeping-staging/results/metrics/grid_search")
os.makedirs(GRID_PATH, exist_ok=True)

if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')
print(f"디바이스: {device}")


print("데이터 로딩 중...")
X_train = np.load(os.path.join(PROCESSED_PATH, 'X_train_sc.npy'))
y_train = np.load(os.path.join(PROCESSED_PATH, 'y_train_sc.npy'))
X_val   = np.load(os.path.join(PROCESSED_PATH, 'X_val_sc.npy'))
y_val   = np.load(os.path.join(PROCESSED_PATH, 'y_val_sc.npy'))

class SleepDataset(Dataset):
    def __init__(self, X, y):
        X = X.copy().astype(np.float32)
        for i in range(X.shape[1]):
            mean = X[:, i, :].mean()
            std  = X[:, i, :].std() + 1e-8
            X[:, i, :] = (X[:, i, :] - mean) / std
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
    def __len__(self): return len(self.y)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

class_counts  = np.bincount(y_train)
class_weights = 1.0 / class_counts
class_weights = class_weights / class_weights.sum() * len(class_counts)
class_weights_tensor = torch.FloatTensor(class_weights).to(device)

class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dropout):
        super().__init__()
        pad = (kernel_size - 1) // 2
        self.conv1    = nn.Conv1d(in_ch, out_ch, kernel_size, padding=pad)
        self.bn1      = nn.BatchNorm1d(out_ch)
        self.conv2    = nn.Conv1d(out_ch, out_ch, kernel_size, padding=pad)
        self.bn2      = nn.BatchNorm1d(out_ch)
        self.relu     = nn.ReLU()
        self.drop     = nn.Dropout(dropout)
        self.shortcut = nn.Conv1d(in_ch, out_ch, 1) \
                        if in_ch != out_ch else nn.Identity()
    def forward(self, x):
        res = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.drop(out)
        out = self.bn2(self.conv2(out))
        if out.shape[2] != res.shape[2]:
            res = res[:, :, :out.shape[2]]
        return self.relu(out + res)

class CNN1D(nn.Module):
    def __init__(self, n_channels=2, n_classes=5,
                 base_filters=32, dropout=0.3):
        super().__init__()
        self.blocks = nn.Sequential(
            ResidualBlock(n_channels,     base_filters,   50, dropout),
            nn.MaxPool1d(8, 8),
            ResidualBlock(base_filters,   base_filters*2, 10, dropout),
            nn.MaxPool1d(4, 4),
            ResidualBlock(base_filters*2, base_filters*4, 10, dropout),
            nn.MaxPool1d(4, 4),
        )
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(base_filters*4, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, n_classes)
        )
    def forward(self, x):
        return self.classifier(self.gap(self.blocks(x)))

def train_and_evaluate(model, train_loader, val_loader,
                       lr, epochs=30, patience=7):
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    optimizer = optim.AdamW(model.parameters(),
                            lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=epochs, eta_min=1e-6)
    best_f1    = 0
    no_improve = 0

    for epoch in range(1, epochs + 1):
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X_batch), y_batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                preds = model(X_batch.to(device)).argmax(1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(y_batch.numpy())
        f1 = f1_score(all_labels, all_preds,
                      average='macro', zero_division=0)

        if f1 > best_f1:
            best_f1 = f1
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    return best_f1


param_space = {
    'lr':           [1e-2, 5e-3, 1e-3, 5e-4, 3e-4, 1e-4, 5e-5],
    'dropout':      [0.1, 0.2, 0.3, 0.4, 0.5],
    'base_filters': [16, 32, 64],
    'batch_size':   [32, 64, 128],
}

N_SEARCH = 25
random.seed(42)

results     = []
best_f1     = 0
best_params = {}

print(f"\n1D-CNN Random Search 시작 (N={N_SEARCH})")
print("="*50)

for idx in range(N_SEARCH):
    params = {k: random.choice(v) for k, v in param_space.items()}
    print(f"\n[{idx+1}/{N_SEARCH}] {params}")

    train_loader = DataLoader(
        SleepDataset(X_train, y_train),
        batch_size=params['batch_size'],
        shuffle=True, num_workers=0)
    val_loader = DataLoader(
        SleepDataset(X_val, y_val),
        batch_size=params['batch_size'],
        shuffle=False, num_workers=0)

    model = CNN1D(
        n_channels=2, n_classes=5,
        base_filters=params['base_filters'],
        dropout=params['dropout']
    ).to(device)

    start = time.time()
    f1 = train_and_evaluate(
        model, train_loader, val_loader,
        lr=params['lr'], epochs=20, patience=5)
    elapsed = time.time() - start

    print(f"  Val F1: {f1:.4f} | Time: {elapsed:.1f}s")

    results.append({
        'params': params,
        'val_f1': float(f1),
        'time_sec': float(elapsed)
    })

    if f1 > best_f1:
        best_f1     = f1
        best_params = params
        print(f"  ★ New Best!")

print(f"\n{'='*50}")
print(f"1D-CNN Best Val F1: {best_f1:.4f}")
print(f"Best Params: {best_params}")

output = {
    'model': 'CNN1D',
    'n_search': N_SEARCH,
    'best_val_f1': float(best_f1),
    'best_params': best_params,
    'all_results': sorted(results,
                          key=lambda x: x['val_f1'],
                          reverse=True)
}
with open(os.path.join(GRID_PATH, 'cnn1d_search.json'), 'w') as f:
    json.dump(output, f, indent=4)
print("저장 완료: cnn1d_search.json")