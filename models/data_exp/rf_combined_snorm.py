import numpy as np
import os, json, time
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, f1_score,
                             cohen_kappa_score, classification_report)
from scipy import signal as scipy_signal

PROCESSED_PATH = os.path.expanduser(
    "~/Desktop/sleeping-staging/preprocessing/processed")
RESULTS_PATH = os.path.expanduser(
    "~/Desktop/sleeping-staging/results/metrics/data_exp_snorm")
os.makedirs(RESULTS_PATH, exist_ok=True)

def extract_psd_features(X):
    bands = {
        'delta': (0.5,4), 'theta': (4,8),
        'alpha': (8,12),  'sigma': (12,15), 'beta': (15,30)
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

print("데이터 로딩 중... [combined + subject norm]")
X_train = np.load(os.path.join(PROCESSED_PATH, 'X_train_comb_snorm.npy'))
y_train = np.load(os.path.join(PROCESSED_PATH, 'y_train_comb_snorm.npy'))
X_val   = np.load(os.path.join(PROCESSED_PATH, 'X_val_comb_snorm.npy'))
y_val   = np.load(os.path.join(PROCESSED_PATH, 'y_val_comb_snorm.npy'))
X_test  = np.load(os.path.join(PROCESSED_PATH, 'X_test_comb_snorm.npy'))
y_test  = np.load(os.path.join(PROCESSED_PATH, 'y_test_comb_snorm.npy'))
print(f"Train shape: {X_train.shape}")

print("\nPSD feature 추출 중...")
X_train_psd = extract_psd_features(X_train)
X_val_psd   = extract_psd_features(X_val)
X_test_psd  = extract_psd_features(X_test)
print(f"Feature shape: {X_train_psd.shape}")

print("\nRandom Forest 학습 중...")
start = time.time()
rf = RandomForestClassifier(
    n_estimators=200, max_depth=None,
    n_jobs=-1, random_state=42, class_weight='balanced')
rf.fit(X_train_psd, y_train)
train_time = time.time() - start
print(f"학습 완료! {train_time:.1f}초")

def evaluate(model, X, y, split_name):
    y_pred = model.predict(X)
    acc   = accuracy_score(y, y_pred)
    f1    = f1_score(y, y_pred, average='macro', zero_division=0)
    kappa = cohen_kappa_score(y, y_pred)
    print(f"\n=== {split_name} ===")
    print(f"Accuracy     : {acc:.4f}")
    print(f"F1 (macro)   : {f1:.4f}")
    print(f"Cohen's Kappa: {kappa:.4f}")
    print(classification_report(y, y_pred,
          target_names=['W','N1','N2','N3','REM'], zero_division=0))
    return {'accuracy': float(acc), 'f1_macro': float(f1),
            'kappa': float(kappa)}

val_metrics  = evaluate(rf, X_val_psd,  y_val,  'Validation')
test_metrics = evaluate(rf, X_test_psd, y_test, 'Test')

results = {
    'model': 'RandomForest',
    'channel_config': 'eeg',
    'dataset': 'combined',
    'normalization': 'subject_wise',
    'n_channels': 2,
    'train_time_sec': float(train_time),
    'val_metrics': val_metrics,
    'test_metrics': test_metrics
}
with open(os.path.join(RESULTS_PATH, 'rf_combined_snorm.json'), 'w') as f:
    json.dump(results, f, indent=4)
print("\n저장 완료: rf_combined_snorm.json")