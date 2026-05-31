import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os, json, time
from sklearn.metrics import (accuracy_score, f1_score,
                             cohen_kappa_score, classification_report)

CHANNEL_CONFIG = 'eeg_eog_emg'
N_CHANNELS     = 4
LR           = 1e-3
DROPOUT      = 0.2
BASE_FILTERS = 64
BATCH_SIZE   = 64
EPOCHS       = 50
PATIENCE     = 10

PROCESSED_PATH = os.path.expanduser(
    "~/Desktop/sleeping-staging/preprocessing/processed")
RESULTS_PATH = os.path.expanduser(
    "~/Desktop/sleeping-staging/results/metrics/channel_exp")
os.makedirs(RESULTS_PATH, exist_ok=True)

if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')
print(f"디바이스: {device}")

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
                 base_filters=64, dropout=0.2):
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

def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        loss = criterion(model(X_batch), y_batch)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * len(y_batch)
        correct    += (model(X_batch).argmax(1) == y_batch).sum().item()
        total      += len(y_batch)
    return total_loss / total, correct / total

def evaluate(model, loader):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            preds = model(X_batch.to(device)).argmax(1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y_batch.numpy())
    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)
    acc   = accuracy_score(all_labels, all_preds)
    f1    = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    kappa = cohen_kappa_score(all_labels, all_preds)
    return acc, f1, kappa, all_preds, all_labels

if __name__ == '__main__':
    print(f"데이터 로딩 중... [{CHANNEL_CONFIG}]")
    X_train = np.load(os.path.join(PROCESSED_PATH,
                      f'X_train_sc_{CHANNEL_CONFIG}.npy'))
    y_train = np.load(os.path.join(PROCESSED_PATH,
                      f'y_train_sc_{CHANNEL_CONFIG}.npy'))
    X_val   = np.load(os.path.join(PROCESSED_PATH,
                      f'X_val_sc_{CHANNEL_CONFIG}.npy'))
    y_val   = np.load(os.path.join(PROCESSED_PATH,
                      f'y_val_sc_{CHANNEL_CONFIG}.npy'))
    X_test  = np.load(os.path.join(PROCESSED_PATH,
                      f'X_test_sc_{CHANNEL_CONFIG}.npy'))
    y_test  = np.load(os.path.join(PROCESSED_PATH,
                      f'y_test_sc_{CHANNEL_CONFIG}.npy'))

    train_loader = DataLoader(SleepDataset(X_train, y_train),
                              batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(SleepDataset(X_val,   y_val),
                              batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader  = DataLoader(SleepDataset(X_test,  y_test),
                              batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    class_counts  = np.bincount(y_train)
    class_weights = 1.0 / class_counts
    class_weights = class_weights / class_weights.sum() * len(class_counts)
    class_weights_tensor = torch.FloatTensor(class_weights).to(device)

    model     = CNN1D(n_channels=N_CHANNELS, n_classes=5,
                      base_filters=BASE_FILTERS,
                      dropout=DROPOUT).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    optimizer = optim.AdamW(model.parameters(),
                            lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=EPOCHS, eta_min=1e-6)

    best_val_f1 = 0
    best_epoch  = 0
    history     = []
    no_improve  = 0

    print(f"\n학습 시작 (epochs={EPOCHS}, patience={PATIENCE})")
    start = time.time()

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion)
        val_acc, val_f1, val_kappa, _, _ = evaluate(model, val_loader)
        scheduler.step()

        current_lr = optimizer.param_groups[0]['lr']
        history.append({
            'epoch': epoch,
            'train_loss': float(train_loss),
            'train_acc':  float(train_acc),
            'val_acc':    float(val_acc),
            'val_f1':     float(val_f1),
            'val_kappa':  float(val_kappa),
        })

        print(f"Epoch {epoch:02d}/{EPOCHS} | "
              f"Loss: {train_loss:.4f} | "
              f"Train Acc: {train_acc:.4f} | "
              f"Val Acc: {val_acc:.4f} | "
              f"Val F1: {val_f1:.4f} | "
              f"Kappa: {val_kappa:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch  = epoch
            no_improve  = 0
            torch.save(model.state_dict(),
                       os.path.join(RESULTS_PATH,
                       f'best_cnn1d_{CHANNEL_CONFIG}.pt'))
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"Early stopping at epoch {epoch}")
                break

    train_time = time.time() - start
    print(f"\n학습 완료! {train_time:.1f}초")
    print(f"Best epoch: {best_epoch}, Best Val F1: {best_val_f1:.4f}")

    model.load_state_dict(torch.load(
        os.path.join(RESULTS_PATH, f'best_cnn1d_{CHANNEL_CONFIG}.pt'),
        map_location=device))
    test_acc, test_f1, test_kappa, test_preds, test_labels = evaluate(
        model, test_loader)

    print(f"\n=== Test 결과 ===")
    print(f"Accuracy     : {test_acc:.4f}")
    print(f"F1 (macro)   : {test_f1:.4f}")
    print(f"Cohen's Kappa: {test_kappa:.4f}")
    print(classification_report(test_labels, test_preds,
          target_names=['W','N1','N2','N3','REM'], zero_division=0))

    results = {
        'model': 'CNN1D',
        'channel_config': CHANNEL_CONFIG,
        'dataset': 'cassette',
        'n_channels': N_CHANNELS,
        'params': {
            'lr': LR, 'dropout': DROPOUT,
            'base_filters': BASE_FILTERS, 'batch_size': BATCH_SIZE
        },
        'best_epoch': best_epoch,
        'best_val_f1': float(best_val_f1),
        'train_time_sec': float(train_time),
        'test_metrics': {
            'accuracy': float(test_acc),
            'f1_macro': float(test_f1),
            'kappa':    float(test_kappa)
        },
        'history': history
    }
    with open(os.path.join(RESULTS_PATH,
              f'cnn1d_{CHANNEL_CONFIG}.json'), 'w') as f:
        json.dump(results, f, indent=4)
    print(f"저장 완료: cnn1d_{CHANNEL_CONFIG}.json")