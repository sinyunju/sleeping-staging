import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os, json, time
from sklearn.metrics import (accuracy_score, f1_score,
                             cohen_kappa_score, classification_report)

PROCESSED_PATH = os.path.expanduser(
    "~/Desktop/sleeping-staging/preprocessing/processed")
RESULTS_PATH = os.path.expanduser(
    "~/Desktop/sleeping-staging/results/metrics/data_exp")
os.makedirs(RESULTS_PATH, exist_ok=True)

N_CHANNELS      = 2
LR              = 1e-4
DROPOUT         = 0.1
D_MODEL         = 256
N_HEADS         = 8
N_LAYERS        = 3
DIM_FEEDFORWARD = 512
BATCH_SIZE      = 16
EPOCHS          = 50
PATIENCE        = 10
WARMUP_EPOCHS   = 5

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

class SleepTransformer(nn.Module):
    def __init__(self, n_channels=2, n_classes=5,
                 d_model=256, n_heads=8, n_layers=3,
                 dim_feedforward=512, dropout=0.1):
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
    print("데이터 로딩 중... [combined]")
    X_train = np.load(os.path.join(PROCESSED_PATH, 'X_train_comb.npy'))
    y_train = np.load(os.path.join(PROCESSED_PATH, 'y_train_comb.npy'))
    X_val   = np.load(os.path.join(PROCESSED_PATH, 'X_val_comb.npy'))
    y_val   = np.load(os.path.join(PROCESSED_PATH, 'y_val_comb.npy'))
    X_test  = np.load(os.path.join(PROCESSED_PATH, 'X_test_comb.npy'))
    y_test  = np.load(os.path.join(PROCESSED_PATH, 'y_test_comb.npy'))
    print(f"Train shape: {X_train.shape}")

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

    model     = SleepTransformer(
                    n_channels=N_CHANNELS, n_classes=5,
                    d_model=D_MODEL, n_heads=N_HEADS,
                    n_layers=N_LAYERS,
                    dim_feedforward=DIM_FEEDFORWARD,
                    dropout=DROPOUT).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

    def lr_lambda(epoch):
        if epoch < WARMUP_EPOCHS:
            return (epoch + 1) / WARMUP_EPOCHS
        progress = (epoch - WARMUP_EPOCHS) / (EPOCHS - WARMUP_EPOCHS)
        return 0.5 * (1 + np.cos(np.pi * progress))
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best_val_f1 = 0
    best_epoch  = 0
    history     = []
    no_improve  = 0

    print(f"\n학습 시작 (epochs={EPOCHS}, patience={PATIENCE}, warmup={WARMUP_EPOCHS})")
    start = time.time()

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion)
        val_acc, val_f1, val_kappa, _, _ = evaluate(model, val_loader)
        scheduler.step()

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
                       os.path.join(RESULTS_PATH, 'best_transformer_combined.pt'))
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"Early stopping at epoch {epoch}")
                break

    train_time = time.time() - start
    print(f"\n학습 완료! {train_time:.1f}초")
    print(f"Best epoch: {best_epoch}, Best Val F1: {best_val_f1:.4f}")

    model.load_state_dict(torch.load(
        os.path.join(RESULTS_PATH, 'best_transformer_combined.pt'),
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
        'model': 'Transformer',
        'channel_config': 'eeg',
        'dataset': 'combined',
        'n_channels': N_CHANNELS,
        'params': {
            'lr': LR, 'dropout': DROPOUT,
            'd_model': D_MODEL, 'n_heads': N_HEADS,
            'n_layers': N_LAYERS, 'batch_size': BATCH_SIZE,
            'warmup_epochs': WARMUP_EPOCHS
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
    with open(os.path.join(RESULTS_PATH, 'transformer_combined.json'), 'w') as f:
        json.dump(results, f, indent=4)
    print("저장 완료: transformer_combined.json")