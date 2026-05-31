import numpy as np
import os
import json
import time
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score, cohen_kappa_score
from scipy import signal as scipy_signal

PROCESSED_PATH = os.path.expanduser(
    "~/Desktop/sleeping-staging/preprocessing/processed")
GRID_PATH = os.path.expanduser(
    "~/Desktop/sleeping-staging/results/metrics/grid_search")
os.makedirs(GRID_PATH, exist_ok=True)


print("데이터 로딩 중...")
X_train = np.load(os.path.join(PROCESSED_PATH, 'X_train_sc.npy'))
y_train = np.load(os.path.join(PROCESSED_PATH, 'y_train_sc.npy'))
X_val   = np.load(os.path.join(PROCESSED_PATH, 'X_val_sc.npy'))
y_val   = np.load(os.path.join(PROCESSED_PATH, 'y_val_sc.npy'))

def extract_psd_features(X):
    bands = {
        'delta': (0.5, 4), 'theta': (4, 8),
        'alpha': (8, 12),  'sigma': (12, 15), 'beta': (15, 30)
    }
    features = []
    for i in range(len(X)):
        if i % 3000 == 0:
            print(f"  {i}/{len(X)} 처리 중...")
        sample_features = []
        for ch in range(X.shape[1]):
            sig = X[i, ch, :]
            freqs, psd = scipy_signal.welch(sig, fs=100, nperseg=256)
            band_powers = {}
            for band_name, (low, high) in bands.items():
                mask = (freqs >= low) & (freqs < high)
                power = np.trapezoid(psd[mask], freqs[mask])
                band_powers[band_name] = power
                sample_features.append(power)
            total = sum(band_powers.values()) + 1e-10
            sample_features.append(band_powers['delta'] / total)
            sample_features.append(band_powers['theta'] / total)
            sample_features.append(band_powers['alpha'] / total)
            sample_features.append(
                band_powers['delta'] / (band_powers['beta'] + 1e-10))
            sample_features.append(total)
        features.append(sample_features)
    return np.array(features)

print("\nPSD feature 추출 중...")
X_train_psd = extract_psd_features(X_train)
X_val_psd   = extract_psd_features(X_val)
print(f"Feature shape: {X_train_psd.shape}")


param_space = {
    'n_estimators': [50, 100, 200, 300, 500],
    'max_depth':    [None, 5, 10, 15, 20, 30],
    'max_features': ['sqrt', 'log2', 0.3, 0.5],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf':  [1, 2, 4],
}


N_SEARCH = 25
random.seed(42)

results     = []
best_f1     = 0
best_params = {}

print(f"\nRF Random Search 시작 (N={N_SEARCH})")
print("="*50)

for idx in range(N_SEARCH):
    params = {k: random.choice(v) for k, v in param_space.items()}
    print(f"\n[{idx+1}/{N_SEARCH}] {params}")

    start = time.time()
    rf = RandomForestClassifier(
        **params,
        class_weight='balanced',
        n_jobs=-1,
        random_state=42
    )
    rf.fit(X_train_psd, y_train)
    elapsed = time.time() - start

    y_pred = rf.predict(X_val_psd)
    f1     = f1_score(y_val, y_pred, average='macro', zero_division=0)
    acc    = accuracy_score(y_val, y_pred)
    kappa  = cohen_kappa_score(y_val, y_pred)

    print(f"  Val F1: {f1:.4f} | Acc: {acc:.4f} | "
          f"Kappa: {kappa:.4f} | Time: {elapsed:.1f}s")

    results.append({
        'params': {k: str(v) for k, v in params.items()},
        'val_f1':    float(f1),
        'val_acc':   float(acc),
        'val_kappa': float(kappa),
        'time_sec':  float(elapsed)
    })

    if f1 > best_f1:
        best_f1     = f1
        best_params = params
        print(f"  ★ New Best!")

print(f"\n{'='*50}")
print(f"RF Best Val F1: {best_f1:.4f}")
print(f"Best Params: {best_params}")

output = {
    'model': 'RandomForest',
    'n_search': N_SEARCH,
    'best_val_f1': float(best_f1),
    'best_params': {k: str(v) for k, v in best_params.items()},
    'all_results': sorted(results,
                          key=lambda x: x['val_f1'],
                          reverse=True)
}
with open(os.path.join(GRID_PATH, 'rf_search.json'), 'w') as f:
    json.dump(output, f, indent=4)
print("저장 완료: rf_search.json")