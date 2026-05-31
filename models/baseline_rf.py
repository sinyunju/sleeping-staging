import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score, classification_report
from scipy import signal as scipy_signal
import json
import time


PROCESSED_PATH = os.path.expanduser("~/Desktop/sleeping-staging/preprocessing/processed")
RESULTS_PATH = os.path.expanduser("~/Desktop/sleeping-staging/results/metrics")
os.makedirs(RESULTS_PATH, exist_ok=True)


print("데이터 로딩 중...")
X_train = np.load(os.path.join(PROCESSED_PATH, 'X_train_sc.npy'))
y_train = np.load(os.path.join(PROCESSED_PATH, 'y_train_sc.npy'))
X_val   = np.load(os.path.join(PROCESSED_PATH, 'X_val_sc.npy'))
y_val   = np.load(os.path.join(PROCESSED_PATH, 'y_val_sc.npy'))
X_test  = np.load(os.path.join(PROCESSED_PATH, 'X_test_sc.npy'))
y_test  = np.load(os.path.join(PROCESSED_PATH, 'y_test_sc.npy'))
print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")


def extract_psd_features(X):

    n_samples, n_channels, _ = X.shape
    features = []

    bands = {
        'delta' : (0.5,  4),
        'theta' : (4,    8),
        'alpha' : (8,   12),
        'sigma' : (12,  15),  
        'beta'  : (15,  30),
    }

    for i in range(n_samples):
        if i % 1000 == 0:
            print(f"  {i}/{n_samples} 처리 중...")
        sample_features = []
        for ch in range(n_channels):
            sig = X[i, ch, :]
            freqs, psd = scipy_signal.welch(sig, fs=100, nperseg=256)

            band_powers = {}
            for band_name, (low, high) in bands.items():
                mask = (freqs >= low) & (freqs < high)
                power = np.trapezoid(psd[mask], freqs[mask])
                band_powers[band_name] = power
                sample_features.append(power)

         
            total_power = sum(band_powers.values()) + 1e-10
            sample_features.append(band_powers['delta'] / total_power)         
            sample_features.append(band_powers['theta'] / total_power)         
            sample_features.append(band_powers['alpha'] / total_power)         
            sample_features.append(band_powers['delta'] / (band_powers['beta'] + 1e-10))  
            sample_features.append(total_power)                             

        features.append(sample_features)

    return np.array(features)


print("\nPSD feature 추출 중...")
start = time.time()
X_train_psd = extract_psd_features(X_train)
X_val_psd   = extract_psd_features(X_val)
X_test_psd  = extract_psd_features(X_test)
print(f"완료! 소요 시간: {time.time()-start:.1f}초")
print(f"Feature shape: {X_train_psd.shape}")  


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
          target_names=['W', 'N1', 'N2', 'N3', 'REM'],
          zero_division=0))

    return {'accuracy': float(acc), 'f1_macro': float(f1), 'kappa': float(kappa)}


print("\nRandom Forest 학습 중...")
start = time.time()
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    n_jobs=-1,
    random_state=42,
    class_weight='balanced'
)
rf.fit(X_train_psd, y_train)
train_time = time.time() - start
print(f"학습 완료! 소요 시간: {train_time:.1f}초")


val_metrics  = evaluate(rf, X_val_psd,  y_val,  'Validation')
test_metrics = evaluate(rf, X_test_psd, y_test, 'Test')


channel_names = ['Fpz-Cz', 'Pz-Oz']
band_names = ['delta', 'theta', 'alpha', 'sigma', 'beta',
              'delta_ratio', 'theta_ratio', 'alpha_ratio',
              'delta_beta_ratio', 'total_power']
feature_names = [f"{ch}_{band}" for ch in channel_names for band in band_names]

importances = rf.feature_importances_
top_idx = np.argsort(importances)[::-1][:10]
print("\n=== Top 10 Feature Importance ===")
for idx in top_idx:
    print(f"{feature_names[idx]:<30}: {importances[idx]:.4f}")


results = {
    'model': 'RandomForest_PSD',
    'dataset': 'cassette',
    'n_estimators': 200,
    'class_weight': 'balanced',
    'n_features': X_train_psd.shape[1],
    'train_time_sec': float(train_time),
    'val_metrics': val_metrics,
    'test_metrics': test_metrics
}
with open(os.path.join(RESULTS_PATH, 'rf_results.json'), 'w') as f:
    json.dump(results, f, indent=4)
print("\n결과 저장 완료: rf_results.json")