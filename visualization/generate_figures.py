import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix, f1_score
from scipy import signal as scipy_signal
import os


BASE       = os.path.expanduser("~/Desktop/sleeping-staging")
PROC       = os.path.join(BASE, "preprocessing/processed")
METRICS    = os.path.join(BASE, "results/metrics")
FIG_DIR    = os.path.join(BASE, "results/figures")
os.makedirs(FIG_DIR, exist_ok=True)

LABELS     = ['W', 'N1', 'N2', 'N3', 'REM']
COLORS     = ['#E24B4A', '#EF9F27', '#1D9E75', '#378ADD', '#7F77DD']
MODEL_COLORS = {'RF': '#888780', '1D-CNN': '#378ADD',
                'CNN+LSTM': '#1D9E75', 'Transformer': '#7F77DD'}


if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')


class SleepDataset(Dataset):
    def __init__(self, X, y, normalize=True):
        X = X.copy().astype(np.float32)
        if normalize:
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
            nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, n_classes)
        )
    def forward(self, x):
        return self.classifier(self.gap(self.blocks(x)))

class CNN_LSTM(nn.Module):
    def __init__(self, n_channels=2, n_classes=5,
                 lstm_hidden=256, lstm_layers=3, dropout=0.2):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(n_channels, 32, 50, padding=24),
            nn.BatchNorm1d(32), nn.ReLU(), nn.Dropout(dropout),
            nn.MaxPool1d(8, 8),
            nn.Conv1d(32, 64, 10, padding=4),
            nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(dropout),
            nn.MaxPool1d(4, 4),
            nn.Conv1d(64, 128, 10, padding=4),
            nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(dropout),
            nn.MaxPool1d(4, 4),
        )
        self.lstm = nn.LSTM(128, lstm_hidden, lstm_layers,
                            batch_first=True,
                            dropout=dropout if lstm_layers > 1 else 0)
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden, 64), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(64, n_classes))
    def forward(self, x):
        out = self.cnn(x).permute(0, 2, 1)
        out, _ = self.lstm(out)
        return self.classifier(out[:, -1, :])

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

def get_predictions(model, X, y, batch_size=128):
    model.eval()
    loader = DataLoader(SleepDataset(X, y),
                        batch_size=batch_size, shuffle=False)
    all_preds, all_labels = [], []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            preds = model(X_batch.to(device)).argmax(1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y_batch.numpy())
    return np.array(all_preds), np.array(all_labels)


def plot_confusion_matrices():
    print("Figure 1: Confusion Matrix 생성 중...")

    X_test = np.load(os.path.join(PROC, 'X_test_sc.npy'))
    y_test = np.load(os.path.join(PROC, 'y_test_sc.npy'))
    X_test_eeg_eog_emg = np.load(
        os.path.join(PROC, 'X_test_sc_eeg_eog_emg.npy'))
    y_test_eeg_eog_emg = np.load(
        os.path.join(PROC, 'y_test_sc_eeg_eog_emg.npy'))

    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    fig.suptitle('Confusion Matrix by Model (Test Set, EEG 2ch, Cassette)',
                 fontsize=14, fontweight='bold', y=1.02)

   
    from sklearn.ensemble import RandomForestClassifier
    from scipy import signal as scipy_signal

    def extract_psd(X):
        bands = {'delta':(0.5,4),'theta':(4,8),
                 'alpha':(8,12),'sigma':(12,15),'beta':(15,30)}
        features = []
        for i in range(len(X)):
            sf = []
            for ch in range(X.shape[1]):
                sig = X[i, ch, :]
                freqs, psd = scipy_signal.welch(sig, fs=100, nperseg=256)
                bp = {}
                for bn, (lo, hi) in bands.items():
                    mask = (freqs >= lo) & (freqs < hi)
                    power = np.trapezoid(psd[mask], freqs[mask])
                    bp[bn] = power
                    sf.append(power)
                total = sum(bp.values()) + 1e-10
                sf.append(bp['delta']/total)
                sf.append(bp['theta']/total)
                sf.append(bp['alpha']/total)
                sf.append(bp['delta']/(bp['beta']+1e-10))
                sf.append(total)
            features.append(sf)
        return np.array(features)

    X_train = np.load(os.path.join(PROC, 'X_train_sc.npy'))
    y_train = np.load(os.path.join(PROC, 'y_train_sc.npy'))
    X_train_psd = extract_psd(X_train)
    X_test_psd  = extract_psd(X_test)
    rf = RandomForestClassifier(n_estimators=200, n_jobs=-1,
                                random_state=42, class_weight='balanced')
    rf.fit(X_train_psd, y_train)
    rf_preds = rf.predict(X_test_psd)

    configs = [
        ('RF',          rf_preds,      y_test,             '#888780'),
        ('1D-CNN',      None,          y_test,             '#378ADD'),
        ('CNN+LSTM',    None,          y_test,             '#1D9E75'),
        ('Transformer', None,          y_test,             '#7F77DD'),
    ]

  
    models_info = [
        ('1D-CNN',
         CNN1D(n_channels=2, n_classes=5, base_filters=64, dropout=0.2),
         os.path.join(METRICS, 'basic_exp/best_cnn1d_final.pt'),
         X_test, y_test),
        ('CNN+LSTM',
         CNN_LSTM(n_channels=2, n_classes=5,
                  lstm_hidden=256, lstm_layers=3, dropout=0.2),
         os.path.join(METRICS, 'basic_exp/best_cnn_lstm_final.pt'),
         X_test, y_test),
        ('Transformer',
         SleepTransformer(n_channels=2, n_classes=5, d_model=256,
                          n_heads=8, n_layers=3, dropout=0.1),
         os.path.join(METRICS, 'basic_exp/best_transformer_final.pt'),
         X_test, y_test),
    ]

    dl_preds = {}
    for name, model, pt_path, X, y in models_info:
        model.load_state_dict(torch.load(pt_path, map_location=device))
        model.to(device)
        preds, _ = get_predictions(model, X, y)
        dl_preds[name] = preds

    for ax, (name, preds, y_true, color) in zip(axes, configs):
        if preds is None:
            preds = dl_preds[name]
        cm = confusion_matrix(y_true, preds)
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

        im = ax.imshow(cm_norm, cmap='Blues', vmin=0, vmax=1)
        ax.set_xticks(range(5))
        ax.set_yticks(range(5))
        ax.set_xticklabels(LABELS, fontsize=11)
        ax.set_yticklabels(LABELS, fontsize=11)
        ax.set_xlabel('Predicted', fontsize=11)
        ax.set_ylabel('True', fontsize=11)
        ax.set_title(name, fontsize=13, fontweight='bold', color=color)

        for i in range(5):
            for j in range(5):
                val = cm_norm[i, j]
                text_color = 'white' if val > 0.5 else 'black'
                ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                        fontsize=10, color=text_color, fontweight='bold')

    plt.colorbar(im, ax=axes[-1], fraction=0.046, pad=0.04)
    plt.tight_layout()
    path = os.path.join(FIG_DIR, 'fig1_confusion_matrix.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  저장: {path}")


def plot_learning_curves():
    print("Figure 2: Learning Curve 생성 중...")

    json_files = {
        'RF':          None,  
        '1D-CNN':      os.path.join(METRICS, 'basic_exp/cnn1d_final.json'),
        'CNN+LSTM':    os.path.join(METRICS, 'basic_exp/cnn_lstm_final.json'),
        'Transformer': os.path.join(METRICS, 'basic_exp/transformer_final.json'),
    }

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle('Learning Curves (Val F1 by Epoch)',
                 fontsize=14, fontweight='bold')

    dl_models = ['1D-CNN', 'CNN+LSTM', 'Transformer']
    colors = ['#378ADD', '#1D9E75', '#7F77DD']

    for ax, name, color in zip(axes, dl_models, colors):
        with open(json_files[name]) as f:
            data = json.load(f)
        history = data['history']
        epochs    = [h['epoch']     for h in history]
        train_acc = [h['train_acc'] for h in history]
        val_f1    = [h['val_f1']    for h in history]
        best_ep   = data['best_epoch']
        best_f1   = data['best_val_f1']

        ax.plot(epochs, train_acc, color=color, alpha=0.4,
                linewidth=1.5, linestyle='--', label='Train Acc')
        ax.plot(epochs, val_f1, color=color,
                linewidth=2, label='Val F1')
        ax.axvline(x=best_ep, color='red', linestyle='--',
                   alpha=0.7, linewidth=1.2)
        ax.annotate(f'Best\nEpoch {best_ep}\nF1={best_f1:.3f}',
                    xy=(best_ep, best_f1),
                    xytext=(best_ep + 1.5, best_f1 - 0.08),
                    fontsize=9, color='red',
                    arrowprops=dict(arrowstyle='->', color='red', lw=1))
        ax.set_title(name, fontsize=13, fontweight='bold', color=color)
        ax.set_xlabel('Epoch', fontsize=11)
        ax.set_ylabel('Score', fontsize=11)
        ax.set_ylim(0.3, 1.0)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.spines[['top', 'right']].set_visible(False)

    plt.tight_layout()
    path = os.path.join(FIG_DIR, 'fig2_learning_curves.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  저장: {path}")


def plot_feature_importance():
    print("Figure 3: Feature Importance 생성 중...")

    channel_names = ['Fpz-Cz', 'Pz-Oz']
    band_names = ['delta', 'theta', 'alpha', 'sigma', 'beta',
                  'delta_ratio', 'theta_ratio', 'alpha_ratio',
                  'delta_beta_ratio', 'total_power']
    feature_names = [f"{ch}_{band}"
                     for ch in channel_names for band in band_names]


    from sklearn.ensemble import RandomForestClassifier
    from scipy import signal as scipy_signal

    def extract_psd(X):
        bands = {'delta':(0.5,4),'theta':(4,8),
                 'alpha':(8,12),'sigma':(12,15),'beta':(15,30)}
        features = []
        for i in range(len(X)):
            sf = []
            for ch in range(X.shape[1]):
                sig = X[i, ch, :]
                freqs, psd = scipy_signal.welch(sig, fs=100, nperseg=256)
                bp = {}
                for bn, (lo, hi) in bands.items():
                    mask = (freqs >= lo) & (freqs < hi)
                    power = np.trapezoid(psd[mask], freqs[mask])
                    bp[bn] = power
                    sf.append(power)
                total = sum(bp.values()) + 1e-10
                sf.append(bp['delta']/total)
                sf.append(bp['theta']/total)
                sf.append(bp['alpha']/total)
                sf.append(bp['delta']/(bp['beta']+1e-10))
                sf.append(total)
            features.append(sf)
        return np.array(features)

    X_train = np.load(os.path.join(PROC, 'X_train_sc.npy'))
    y_train = np.load(os.path.join(PROC, 'y_train_sc.npy'))
    X_train_psd = extract_psd(X_train)

    rf = RandomForestClassifier(n_estimators=200, n_jobs=-1,
                                random_state=42, class_weight='balanced')
    rf.fit(X_train_psd, y_train)
    importances = rf.feature_importances_
    idx = np.argsort(importances)[::-1]

 
    top_n = 15
    top_idx   = idx[:top_n]
    top_imp   = importances[top_idx]
    top_names = [feature_names[i] for i in top_idx]


    bar_colors = ['#378ADD' if 'Fpz' in n else '#1D9E75'
                  for n in top_names]

    fig, ax = plt.subplots(figsize=(10, 7))
    bars = ax.barh(range(top_n), top_imp[::-1],
                   color=bar_colors[::-1], alpha=0.85, edgecolor='white')
    ax.set_yticks(range(top_n))
    ax.set_yticklabels([top_names[top_n-1-i] for i in range(top_n)],
                       fontsize=11)
    ax.set_xlabel('Feature Importance', fontsize=12)
    ax.set_title('Random Forest Feature Importance (Top 15)',
                 fontsize=13, fontweight='bold')

    for bar, val in zip(bars, top_imp[::-1]):
        ax.text(val + 0.002, bar.get_y() + bar.get_height()/2,
                f'{val:.4f}', va='center', fontsize=9)

    patch1 = mpatches.Patch(color='#378ADD', label='EEG Fpz-Cz')
    patch2 = mpatches.Patch(color='#1D9E75', label='EEG Pz-Oz')
    ax.legend(handles=[patch1, patch2], fontsize=11,
              loc='lower right')
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    path = os.path.join(FIG_DIR, 'fig3_feature_importance.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  저장: {path}")


def plot_classwise_f1():
    print("Figure 4: 클래스별 F1 비교 생성 중...")

    X_test = np.load(os.path.join(PROC, 'X_test_sc.npy'))
    y_test = np.load(os.path.join(PROC, 'y_test_sc.npy'))

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import f1_score
    from scipy import signal as scipy_signal

    def extract_psd(X):
        bands = {'delta':(0.5,4),'theta':(4,8),
                 'alpha':(8,12),'sigma':(12,15),'beta':(15,30)}
        features = []
        for i in range(len(X)):
            sf = []
            for ch in range(X.shape[1]):
                sig = X[i, ch, :]
                freqs, psd = scipy_signal.welch(sig, fs=100, nperseg=256)
                bp = {}
                for bn, (lo, hi) in bands.items():
                    mask = (freqs >= lo) & (freqs < hi)
                    power = np.trapezoid(psd[mask], freqs[mask])
                    bp[bn] = power
                    sf.append(power)
                total = sum(bp.values()) + 1e-10
                sf.append(bp['delta']/total)
                sf.append(bp['theta']/total)
                sf.append(bp['alpha']/total)
                sf.append(bp['delta']/(bp['beta']+1e-10))
                sf.append(total)
            features.append(sf)
        return np.array(features)

    X_train = np.load(os.path.join(PROC, 'X_train_sc.npy'))
    y_train = np.load(os.path.join(PROC, 'y_train_sc.npy'))
    rf = RandomForestClassifier(n_estimators=200, n_jobs=-1,
                                random_state=42, class_weight='balanced')
    rf.fit(extract_psd(X_train), y_train)
    rf_preds = rf.predict(extract_psd(X_test))

    models_info = [
        ('1D-CNN',
         CNN1D(n_channels=2, n_classes=5, base_filters=64, dropout=0.2),
         os.path.join(METRICS, 'basic_exp/best_cnn1d_final.pt')),
        ('CNN+LSTM',
         CNN_LSTM(n_channels=2, n_classes=5,
                  lstm_hidden=256, lstm_layers=3, dropout=0.2),
         os.path.join(METRICS, 'basic_exp/best_cnn_lstm_final.pt')),
        ('Transformer',
         SleepTransformer(n_channels=2, n_classes=5, d_model=256,
                          n_heads=8, n_layers=3, dropout=0.1),
         os.path.join(METRICS, 'basic_exp/best_transformer_final.pt')),
    ]

    all_f1s = {'RF': f1_score(y_test, rf_preds,
                              average=None, zero_division=0)}
    for name, model, pt_path in models_info:
        model.load_state_dict(torch.load(pt_path, map_location=device))
        model.to(device)
        preds, labels = get_predictions(model, X_test, y_test)
        all_f1s[name] = f1_score(labels, preds,
                                 average=None, zero_division=0)

    x = np.arange(5)
    width = 0.2
    model_names = ['RF', '1D-CNN', 'CNN+LSTM', 'Transformer']
    mc = ['#888780', '#378ADD', '#1D9E75', '#7F77DD']

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, (name, color) in enumerate(zip(model_names, mc)):
        offset = (i - 1.5) * width
        bars = ax.bar(x + offset, all_f1s[name], width,
                      label=name, color=color, alpha=0.85,
                      edgecolor='white')
        for bar, val in zip(bars, all_f1s[name]):
            ax.text(bar.get_x() + bar.get_width()/2, val + 0.01,
                    f'{val:.2f}', ha='center', va='bottom',
                    fontsize=8, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(LABELS, fontsize=13)
    ax.set_ylabel('F1-score', fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.set_title('Class-wise F1-score by Model\n(EEG 2ch, Cassette)',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')
    ax.spines[['top', 'right']].set_visible(False)

    plt.tight_layout()
    path = os.path.join(FIG_DIR, 'fig4_classwise_f1.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  저장: {path}")

def plot_channel_effect():
    print("Figure 5: 채널 추가 효과 생성 중...")

    data = {
        'RF':          [0.5771, 0.6254, 0.6313],
        '1D-CNN':      [0.7409, 0.7434, 0.7570],
        'CNN+LSTM':    [0.7138, 0.6963, 0.7111],
        'Transformer': [0.7351, 0.7429, 0.7198],
    }
    channels = ['EEG\n(2ch)', 'EEG+EOG\n(3ch)', 'EEG+EOG+EMG\n(4ch)']
    model_names  = ['RF', '1D-CNN', 'CNN+LSTM', 'Transformer']
    mc = ['#888780', '#378ADD', '#1D9E75', '#7F77DD']
    markers = ['o', 's', '^', 'D']

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    
    ax = axes[0]
    for name, color, marker in zip(model_names, mc, markers):
        ax.plot([0, 1, 2], data[name], color=color, marker=marker,
                linewidth=2.5, markersize=9, label=name, alpha=0.9)
        for xi, yi in enumerate(data[name]):
            ax.annotate(f'{yi:.3f}',
                        xy=(xi, yi), xytext=(0, 10),
                        textcoords='offset points',
                        ha='center', fontsize=8.5, color=color,
                        fontweight='bold')

    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(channels, fontsize=11)
    ax.set_ylabel('Macro F1-score', fontsize=12)
    ax.set_ylim(0.5, 0.82)
    ax.set_title('Channel Addition Effect\n(Macro F1, Cassette)',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.spines[['top', 'right']].set_visible(False)

    
    rem_data = {
        'RF':          [0.28, 0.46, 0.47],
        '1D-CNN':      [0.65, 0.71, 0.73],
        'CNN+LSTM':    [0.62, 0.60, 0.63],
        'Transformer': [0.64, 0.67, 0.66],
    }
    ax2 = axes[1]
    for name, color, marker in zip(model_names, mc, markers):
        ax2.plot([0, 1, 2], rem_data[name], color=color,
                 marker=marker, linewidth=2.5, markersize=9,
                 label=name, alpha=0.9)
        for xi, yi in enumerate(rem_data[name]):
            ax2.annotate(f'{yi:.2f}',
                         xy=(xi, yi), xytext=(0, 10),
                         textcoords='offset points',
                         ha='center', fontsize=8.5, color=color,
                         fontweight='bold')

    ax2.set_xticks([0, 1, 2])
    ax2.set_xticklabels(channels, fontsize=11)
    ax2.set_ylabel('REM F1-score', fontsize=12)
    ax2.set_ylim(0.2, 0.85)
    ax2.set_title('EOG/EMG Effect on REM Classification\n(REM F1, Cassette)',
                  fontsize=13, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.spines[['top', 'right']].set_visible(False)

    plt.tight_layout()
    path = os.path.join(FIG_DIR, 'fig5_channel_effect.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  저장: {path}")


def plot_data_effect():
    print("Figure 6: 데이터 실험 효과 생성 중...")

    data = {
        'RF':          [0.5771, 0.5806, 0.6557],
        '1D-CNN':      [0.7409, 0.6840, 0.7335],
        'CNN+LSTM':    [0.7138, 0.5432, 0.6703],
        'Transformer': [0.7351, 0.6605, 0.6892],
    }
    datasets = ['Cassette', 'Combined', 'Combined\n+snorm']
    model_names = ['RF', '1D-CNN', 'CNN+LSTM', 'Transformer']
    mc = ['#888780', '#378ADD', '#1D9E75', '#7F77DD']
    markers = ['o', 's', '^', 'D']

    fig, ax = plt.subplots(figsize=(10, 6))
    for name, color, marker in zip(model_names, mc, markers):
        ax.plot([0, 1, 2], data[name], color=color, marker=marker,
                linewidth=2.5, markersize=9, label=name, alpha=0.9)
        for xi, yi in enumerate(data[name]):
            offset = 10 if xi != 1 else -18
            ax.annotate(f'{yi:.3f}',
                        xy=(xi, yi), xytext=(0, offset),
                        textcoords='offset points',
                        ha='center', fontsize=9, color=color,
                        fontweight='bold')
    ax.axvspan(0.5, 1.5, alpha=0.06, color='red',
               label='Domain mismatch')
    ax.axvspan(1.5, 2.5, alpha=0.06, color='green',
               label='Subject-wise norm')

    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(datasets, fontsize=12)
    ax.set_ylabel('Macro F1-score', fontsize=12)
    ax.set_ylim(0.45, 0.80)
    ax.set_title('Data Experiment: Domain Mismatch & Subject-wise Normalization\n(EEG 2ch)',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=10, loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.spines[['top', 'right']].set_visible(False)

    plt.tight_layout()
    path = os.path.join(FIG_DIR, 'fig6_data_effect.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  저장: {path}")

def plot_frequency_band_analysis():
    print("Figure 7: 주파수 대역별 구분력 분석 생성 중...")

    from scipy import signal as scipy_signal

    X_train = np.load(os.path.join(PROC, 'X_train_sc.npy'))
    y_train = np.load(os.path.join(PROC, 'y_train_sc.npy'))

    bands = {
        'Delta\n(0.5-4Hz)':  (0.5, 4),
        'Theta\n(4-8Hz)':    (4, 8),
        'Alpha\n(8-12Hz)':   (8, 12),
        'Sigma\n(12-15Hz)':  (12, 15),
        'Beta\n(15-30Hz)':   (15, 30),
    }

    stage_powers = {stage: {band: [] for band in bands}
                   for stage in range(5)}

    print("  PSD 계산 중 (시간이 좀 걸려요)...")
    for i in range(len(X_train)):
        sig = X_train[i, 0, :] 
        freqs, psd = scipy_signal.welch(sig, fs=100, nperseg=256)
        for band_name, (lo, hi) in bands.items():
            mask = (freqs >= lo) & (freqs < hi)
            power = np.trapezoid(psd[mask], freqs[mask])
            stage_powers[y_train[i]][band_name].append(power)

    mean_powers = np.zeros((5, 5))  
    band_names_list = list(bands.keys())
    for stage in range(5):
        for j, band_name in enumerate(band_names_list):
            vals = stage_powers[stage][band_name]
            mean_powers[stage, j] = np.log10(np.mean(vals) + 1e-20)

    norm_powers = mean_powers.copy()
    for j in range(5):
        col = norm_powers[:, j]
        norm_powers[:, j] = (col - col.min()) / (col.max() - col.min() + 1e-10)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    ax = axes[0]
    im = ax.imshow(norm_powers, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')
    ax.set_xticks(range(5))
    ax.set_xticklabels(band_names_list, fontsize=10)
    ax.set_yticks(range(5))
    ax.set_yticklabels(LABELS, fontsize=12)
    ax.set_title('Frequency Band Power by Sleep Stage\n(EEG Fpz-Cz, Normalized)',
                 fontsize=12, fontweight='bold')
    for i in range(5):
        for j in range(5):
            val = norm_powers[i, j]
            text_color = 'black' if 0.3 < val < 0.8 else 'white'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                    fontsize=11, fontweight='bold', color=text_color)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                 label='Normalized Power (0=low, 1=high)')

    ax2 = axes[1]
    x = np.arange(5)
    width = 0.16
    for i, (label, color) in enumerate(zip(LABELS, COLORS)):
        offset = (i - 2) * width
        ax2.bar(x + offset, norm_powers[i], width,
                label=label, color=color, alpha=0.85, edgecolor='white')

    ax2.set_xticks(x)
    ax2.set_xticklabels(band_names_list, fontsize=10)
    ax2.set_ylabel('Normalized Power', fontsize=12)
    ax2.set_title('Discriminative Power of Frequency Bands\nper Sleep Stage',
                  fontsize=12, fontweight='bold')
    ax2.legend(fontsize=10, loc='upper right')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.spines[['top', 'right']].set_visible(False)

    plt.tight_layout()
    path = os.path.join(FIG_DIR, 'fig7_frequency_band_analysis.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  저장: {path}")

def plot_training_stability():
    print("Figure 8: 학습 안정성 비교 생성 중...")

    configs = [
        ('1D-CNN\n(Cassette)',
         os.path.join(METRICS, 'basic_exp/cnn1d_final.json'),
         '#378ADD', '-'),
        ('1D-CNN\n(Combined)',
         os.path.join(METRICS, 'data_exp/cnn1d_combined.json'),
         '#378ADD', '--'),
        ('CNN+LSTM\n(Cassette)',
         os.path.join(METRICS, 'basic_exp/cnn_lstm_final.json'),
         '#1D9E75', '-'),
        ('CNN+LSTM\n(Combined)',
         os.path.join(METRICS, 'data_exp/cnn_lstm_combined.json'),
         '#1D9E75', '--'),
        ('Transformer\n(Cassette)',
         os.path.join(METRICS, 'basic_exp/transformer_final.json'),
         '#7F77DD', '-'),
        ('Transformer\n(Combined)',
         os.path.join(METRICS, 'data_exp/transformer_combined.json'),
         '#7F77DD', '--'),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Training Stability: Cassette vs Combined Dataset',
                 fontsize=14, fontweight='bold')

    model_pairs = [
        ('1D-CNN',      '#378ADD', configs[0], configs[1]),
        ('CNN+LSTM',    '#1D9E75', configs[2], configs[3]),
        ('Transformer', '#7F77DD', configs[4], configs[5]),
    ]

    for ax, (name, color, cfg_sc, cfg_comb) in zip(axes, model_pairs):
        for cfg_name, json_path, c, ls in [
            (cfg_sc[0],   cfg_sc[1],   color, '-'),
            (cfg_comb[0], cfg_comb[1], color, '--'),
        ]:
            with open(json_path) as f:
                data = json.load(f)
            history = data['history']
            epochs = [h['epoch']  for h in history]
            val_f1 = [h['val_f1'] for h in history]
            best_ep = data['best_epoch']

            label = 'Cassette' if '(Cassette)' in cfg_name else 'Combined'
            ax.plot(epochs, val_f1, color=c, linestyle=ls,
                    linewidth=2, label=label, alpha=0.9)
            ax.axvline(x=best_ep, color=c, linestyle=ls,
                       alpha=0.5, linewidth=1)

        ax.set_title(name, fontsize=13, fontweight='bold', color=color)
        ax.set_xlabel('Epoch', fontsize=11)
        ax.set_ylabel('Val F1-score', fontsize=11)
        ax.set_ylim(0.3, 0.75)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.spines[['top', 'right']].set_visible(False)

    plt.tight_layout()
    path = os.path.join(FIG_DIR, 'fig8_training_stability.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  저장: {path}")

def plot_model_architectures():
    print("Figure 9: Model Architecture 생성 중...")

    BLUE   = '#85B7EB'
    TEAL   = '#5DCAA5'
    AMBER  = '#FAC775'
    PURPLE = '#AFA9EC'
    GRAY   = '#D3D1C7'
    DARK   = '#2C2C2A'
    MID    = '#5F5E5A'

    def draw_box(ax, x, y, w, h, color, title, subtitle=None, fontsize=9):
        from matplotlib.patches import FancyBboxPatch
        box = FancyBboxPatch((x, y), w, h,
                             boxstyle="round,pad=0.02",
                             facecolor=color, edgecolor='white',
                             linewidth=1.2, zorder=3)
        ax.add_patch(box)
        if subtitle:
            ax.text(x+w/2, y+h*0.62, title, ha='center', va='center',
                    fontsize=fontsize, fontweight='bold', color=DARK, zorder=4)
            ax.text(x+w/2, y+h*0.28, subtitle, ha='center', va='center',
                    fontsize=fontsize-1.5, color=MID, zorder=4)
        else:
            ax.text(x+w/2, y+h/2, title, ha='center', va='center',
                    fontsize=fontsize, fontweight='bold', color=DARK, zorder=4)

    def draw_arrow(ax, x1, y1, x2, y2):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color=MID,
                                    lw=1.2, connectionstyle='arc3,rad=0'))

    fig, axes = plt.subplots(1, 4, figsize=(22, 10))
    fig.patch.set_facecolor('white')
    titles = ['(a) Random Forest', '(b) 1D-CNN', '(c) CNN+LSTM', '(d) Transformer']

    ax = axes[0]; ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis('off')
    ax.set_title(titles[0], fontsize=11, fontweight='bold', color=DARK, pad=10)
    layers_rf = [
        (GRAY,   'Raw EEG signal',      '2ch × 3,000 samples'),
        (TEAL,   'PSD extraction',      'Welch method per channel'),
        (GRAY,   'Feature vector',      '16~32 dim'),
        (TEAL,   'Random Forest',       'n_estimators=200, balanced'),
        (PURPLE, 'W / N1 / N2 / N3 / REM', None),
    ]
    ys_rf = [0.82, 0.64, 0.46, 0.28, 0.10]
    for (c,t,s), y in zip(layers_rf, ys_rf):
        draw_box(ax, 0.05, y, 0.90, 0.14, c, t, s)
    for i in range(len(ys_rf)-1):
        draw_arrow(ax, 0.50, ys_rf[i], 0.50, ys_rf[i+1]+0.14)

    ax = axes[1]; ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis('off')
    ax.set_title(titles[1], fontsize=11, fontweight='bold', color=DARK, pad=10)
    layers_cnn = [
        (GRAY,   'Raw signal',          'n_ch × 3,000 samples'),
        (BLUE,   'Residual Block 1',    'Conv(k=50) + BN + ReLU + MaxPool(8)'),
        (BLUE,   'Residual Block 2',    'Conv(k=10) + BN + ReLU + MaxPool(4)'),
        (BLUE,   'Residual Block 3',    'Conv(k=10) + BN + ReLU + MaxPool(4)'),
        (TEAL,   'GAP → FC classifier', 'Linear(256→64→5)'),
        (PURPLE, 'W / N1 / N2 / N3 / REM', None),
    ]
    ys_cnn = [0.84, 0.67, 0.50, 0.33, 0.16, 0.01]
    for (c,t,s), y in zip(layers_cnn, ys_cnn):
        draw_box(ax, 0.05, y, 0.90, 0.13, c, t, s)
    for i in range(len(ys_cnn)-1):
        draw_arrow(ax, 0.50, ys_cnn[i], 0.50, ys_cnn[i+1]+0.13)
    ax.annotate('', xy=(0.95, 0.63), xytext=(0.95, 0.80),
                arrowprops=dict(arrowstyle='->', color='#888780', lw=0.8,
                                linestyle='dashed', connectionstyle='arc3,rad=0'))
    ax.text(0.97, 0.72, 'shortcut', fontsize=6.5, color=MID, rotation=90, va='center')

    ax = axes[2]; ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis('off')
    ax.set_title(titles[2], fontsize=11, fontweight='bold', color=DARK, pad=10)
    layers_lstm = [
        (GRAY,   'Raw signal',          'n_ch × 3,000 samples'),
        (BLUE,   'CNN encoder (3 blocks)', '→ (B, 128, 23)'),
        (GRAY,   'Permute',             '(B, 128, 23) → (B, 23, 128)'),
        (AMBER,  '3-layer LSTM',        'hidden=256, sequential (23 timesteps)'),
        (GRAY,   'Last timestep → FC',  'Linear(256→64→5)'),
        (PURPLE, 'W / N1 / N2 / N3 / REM', None),
    ]
    ys_lstm = [0.84, 0.67, 0.50, 0.33, 0.16, 0.01]
    for (c,t,s), y in zip(layers_lstm, ys_lstm):
        draw_box(ax, 0.05, y, 0.90, 0.13, c, t, s)
    for i in range(len(ys_lstm)-1):
        draw_arrow(ax, 0.50, ys_lstm[i], 0.50, ys_lstm[i+1]+0.13)

    ax = axes[3]; ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis('off')
    ax.set_title(titles[3], fontsize=11, fontweight='bold', color=DARK, pad=10)
    layers_trans = [
        (GRAY,   'Raw signal',          'n_ch × 3,000 samples'),
        (BLUE,   'CNN encoder (3 blocks)', '→ (B, 256, 23)'),
        (GRAY,   'Permute + Pos. Enc.', '(B, 23, 256)'),
        (AMBER,  'Transformer Enc. ×3', 'Multi-Head Attn(8 heads) + FFN(512)'),
        (TEAL,   'Global Avg Pool → FC', 'Average 23 positions → Linear(256→5)'),
        (PURPLE, 'W / N1 / N2 / N3 / REM', None),
    ]
    ys_trans = [0.84, 0.67, 0.50, 0.33, 0.16, 0.01]
    for (c,t,s), y in zip(layers_trans, ys_trans):
        draw_box(ax, 0.05, y, 0.90, 0.13, c, t, s)
    for i in range(len(ys_trans)-1):
        draw_arrow(ax, 0.50, ys_trans[i], 0.50, ys_trans[i+1]+0.13)

    import matplotlib.patches as mpatches
    legend_patches = [
        mpatches.Patch(color=BLUE,   label='CNN / Conv layers'),
        mpatches.Patch(color=AMBER,  label='Sequential model (LSTM / Transformer)'),
        mpatches.Patch(color=TEAL,   label='Feature extraction / Aggregation'),
        mpatches.Patch(color=GRAY,   label='Input / Intermediate'),
        mpatches.Patch(color=PURPLE, label='Output (5 sleep stages)'),
    ]
    fig.legend(handles=legend_patches, loc='lower center', ncol=5,
               fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.01))
    plt.suptitle('Model Architectures for Sleep Stage Classification',
                 fontsize=13, fontweight='bold', y=1.01, color=DARK)
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    path = os.path.join(FIG_DIR, 'fig9_model_architectures.png')
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  저장: {path}")

if __name__ == '__main__':
    print("=" * 55)
    print("시각화 생성 시작")
    print(f"저장 경로: {FIG_DIR}")
    print("=" * 55)

    plot_learning_curves()       
    plot_channel_effect()         
    plot_data_effect()            
    plot_training_stability()     
    plot_feature_importance()     
    plot_frequency_band_analysis()
    plot_classwise_f1()           
    plot_confusion_matrices()    
    plot_model_architectures() 

    print("\n" + "=" * 55)
    print("모든 figure 생성 완료!")
    print(f"저장 경로: {FIG_DIR}")
    print("=" * 55)