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

class SleepTransformer(nn.Module):
    def __init__(self, n_channels=2, n_classes=5,
                 d_model=128, n_heads=8, n_layers=2,
                 dim_feedforward=256, dropout=0.3):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(n_channels, 32, 50, padding=24),
            nn.BatchNorm1d(32), nn.ReLU(), nn.Dropout(dropout),
            nn.MaxPool1d(8, 8),
            nn.Conv1d(32, 64, 10, padding=4),
            nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(dropout),
            nn.MaxPool1d(4, 4),
            nn.Conv1d(64, d_model, 10, padding=4),
            nn.BatchNorm1d(d_model), nn.ReLU(), nn.Dropout(dropout),
            nn.MaxPool1d(4, 4),
        )
        self.pos_encoding = nn.Embedding(100, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 64), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(64, n_classes))

    def forward(self, x):
        out = self.cnn(x).permute(0, 2, 1)
        pos = torch.arange(out.shape[1], device=x.device)
        out = out + self.pos_encoding(pos)
        out = self.transformer(out).mean(dim=1)
        return self.classifier(out)

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
    'lr':              [1e-2, 5e-3, 1e-3, 5e-4, 3e-4, 1e-4, 5e-5],
    'dropout':         [0.1, 0.2, 0.3, 0.4, 0.5],
    'd_model':         [64, 128, 256],
    'n_heads':         [4, 8],
    'n_layers':        [1, 2, 3, 4],
    'batch_size':      [16, 32, 64, 128],
}

N_SEARCH = 25
random.seed(42)

results     = []
best_f1     = 0
best_params = {}

print(f"\nTransformer Random Search 시작 (N={N_SEARCH})")
print("="*50)

tried = 0
while tried < N_SEARCH:
    params = {k: random.choice(v) for k, v in param_space.items()}

    if params['d_model'] % params['n_heads'] != 0:
        continue

    tried += 1
    print(f"\n[{tried}/{N_SEARCH}] {params}")

    train_loader = DataLoader(
        SleepDataset(X_train, y_train),
        batch_size=params['batch_size'],
        shuffle=True, num_workers=0)
    val_loader = DataLoader(
        SleepDataset(X_val, y_val),
        batch_size=params['batch_size'],
        shuffle=False, num_workers=0)

    model = SleepTransformer(
        n_channels=2, n_classes=5,
        d_model=params['d_model'],
        n_heads=params['n_heads'],
        n_layers=params['n_layers'],
        dim_feedforward=params['d_model'] * 2,
        dropout=params['dropout']
    ).to(device)

    start = time.time()
    f1 = train_and_evaluate(
        model, train_loader, val_loader,
        lr=params['lr'], epochs=30, patience=7)
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
print(f"Transformer Best Val F1: {best_f1:.4f}")
print(f"Best Params: {best_params}")

output = {
    'model': 'Transformer',
    'n_search': N_SEARCH,
    'best_val_f1': float(best_f1),
    'best_params': best_params,
    'all_results': sorted(results,
                          key=lambda x: x['val_f1'],
                          reverse=True)
}
with open(os.path.join(GRID_PATH, 'transformer_search.json'), 'w') as f:
    json.dump(output, f, indent=4)
print("저장 완료: transformer_search.json")