import mne
import numpy as np
import os
from collections import defaultdict

DATA_PATH = os.path.expanduser("~/Downloads/sleep-edf-database-expanded-1.0.0")
CASSETTE_PATH = os.path.join(DATA_PATH, "sleep-cassette")
TELEMETRY_PATH = os.path.join(DATA_PATH, "sleep-telemetry")
PROCESSED_PATH = os.path.expanduser("~/Desktop/sleeping-staging/preprocessing/processed")
os.makedirs(PROCESSED_PATH, exist_ok=True)

LABEL_MAPPING = {
    'Sleep stage W': 0,
    'Sleep stage 1': 1,
    'Sleep stage 2': 2,
    'Sleep stage 3': 3,
    'Sleep stage 4': 3,
    'Sleep stage R': 4,
}

CHANNELS = ['EEG Fpz-Cz', 'EEG Pz-Oz']

def process_one_subject(psg_path, hyp_path, channels=CHANNELS):
    raw = mne.io.read_raw_edf(psg_path, preload=True, verbose=False)
    available = [ch for ch in channels if ch in raw.ch_names]
    raw.pick(available)
    raw.filter(0.5, 30, verbose=False)
    annotations = mne.read_annotations(hyp_path)
    raw.set_annotations(annotations, verbose=False)
    events, event_id = mne.events_from_annotations(
        raw, event_id=LABEL_MAPPING, verbose=False
    )
    if len(events) == 0:
        return None, None
    epochs = mne.Epochs(
        raw, events, event_id=event_id,
        tmin=0, tmax=29.99,
        baseline=None, verbose=False,
        preload=True
    )
    X = epochs.get_data()
    y = epochs.events[:, 2]
    return X, y

def process_dataset(data_path, label='cassette'):
    psg_files = sorted([f for f in os.listdir(data_path) if f.endswith("PSG.edf")])
    hyp_files = sorted([f for f in os.listdir(data_path) if f.endswith("Hypnogram.edf")])
    
    all_X, all_y = [], []
    subject_ids = [] 
    
    for i, (psg_file, hyp_file) in enumerate(zip(psg_files, hyp_files)):
        print(f"[{label}] Processing {i+1}/{len(psg_files)}: {psg_file}")
        X, y = process_one_subject(
            os.path.join(data_path, psg_file),
            os.path.join(data_path, hyp_file)
        )
        if X is not None:
            all_X.append(X)
            all_y.append(y)
            subject_ids.extend([i] * len(y))
    
    X_all = np.concatenate(all_X, axis=0)
    y_all = np.concatenate(all_y, axis=0)
    subject_ids = np.array(subject_ids)
    
    return X_all, y_all, subject_ids

print("=== Processing Sleep-Cassette ===")
X_sc, y_sc, subj_sc = process_dataset(CASSETTE_PATH, label='cassette')
print(f"\nCassette 완료!")
print(f"X shape: {X_sc.shape}")
print(f"y shape: {y_sc.shape}")
print(f"레이블 분포: {np.unique(y_sc, return_counts=True)}")

np.save(os.path.join(PROCESSED_PATH, 'X_cassette.npy'), X_sc)
np.save(os.path.join(PROCESSED_PATH, 'y_cassette.npy'), y_sc)
np.save(os.path.join(PROCESSED_PATH, 'subj_cassette.npy'), subj_sc)
print("Cassette 저장 완료!")

print("=== Processing Sleep-Telemetry ===")
X_st, y_st, subj_st = process_dataset(TELEMETRY_PATH, label='telemetry')
print(f"\nTelemetry 완료!")
print(f"X shape: {X_st.shape}")
print(f"y shape: {y_st.shape}")
print(f"레이블 분포: {np.unique(y_st, return_counts=True)}")

np.save(os.path.join(PROCESSED_PATH, 'X_telemetry.npy'), X_st)
np.save(os.path.join(PROCESSED_PATH, 'y_telemetry.npy'), y_st)
np.save(os.path.join(PROCESSED_PATH, 'subj_telemetry.npy'), subj_st)
print("Telemetry 저장 완료!")

X_combined = np.concatenate([X_sc, X_st], axis=0)
y_combined = np.concatenate([y_sc, y_st], axis=0)

subj_st_offset = subj_st + 153
subj_combined = np.concatenate([subj_sc, subj_st_offset], axis=0)

np.save(os.path.join(PROCESSED_PATH, 'X_combined.npy'), X_combined)
np.save(os.path.join(PROCESSED_PATH, 'y_combined.npy'), y_combined)
np.save(os.path.join(PROCESSED_PATH, 'subj_combined.npy'), subj_combined)
print(f"\nCombined 저장 완료!")
print(f"X shape: {X_combined.shape}")
print(f"y shape: {y_combined.shape}")
print(f"레이블 분포: {np.unique(y_combined, return_counts=True)}")

from sklearn.model_selection import GroupShuffleSplit
import numpy as np
import os

PROCESSED_PATH = os.path.expanduser("~/Desktop/sleeping-staging/preprocessing/processed")

def split_dataset(X, y, subject_ids, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_state=42):
    unique_subjects = np.unique(subject_ids)
    n_subjects = len(unique_subjects)
    
    np.random.seed(random_state)
    shuffled = np.random.permutation(unique_subjects)
    
    n_train = int(n_subjects * train_ratio)
    n_val   = int(n_subjects * val_ratio)
    
    train_subj = shuffled[:n_train]
    val_subj   = shuffled[n_train:n_train+n_val]
    test_subj  = shuffled[n_train+n_val:]
    
    train_idx = np.isin(subject_ids, train_subj)
    val_idx   = np.isin(subject_ids, val_subj)
    test_idx  = np.isin(subject_ids, test_subj)
    
    return (
        X[train_idx], y[train_idx],
        X[val_idx],   y[val_idx],
        X[test_idx],  y[test_idx],
        len(train_subj), len(val_subj), len(test_subj)
    )

X_sc = np.load(os.path.join(PROCESSED_PATH, 'X_cassette.npy'))
y_sc = np.load(os.path.join(PROCESSED_PATH, 'y_cassette.npy'))
subj_sc = np.load(os.path.join(PROCESSED_PATH, 'subj_cassette.npy'))

X_train, y_train, X_val, y_val, X_test, y_test, n_tr, n_val, n_te = split_dataset(X_sc, y_sc, subj_sc)

print("=== Cassette Split 결과 ===")
print(f"Train: {X_train.shape} ({n_tr}명)")
print(f"Val  : {X_val.shape} ({n_val}명)")
print(f"Test : {X_test.shape} ({n_te}명)")
print(f"\nTrain 레이블 분포: {np.unique(y_train, return_counts=True)}")
print(f"Val   레이블 분포: {np.unique(y_val,   return_counts=True)}")
print(f"Test  레이블 분포: {np.unique(y_test,  return_counts=True)}")

np.save(os.path.join(PROCESSED_PATH, 'X_train_sc.npy'), X_train)
np.save(os.path.join(PROCESSED_PATH, 'y_train_sc.npy'), y_train)
np.save(os.path.join(PROCESSED_PATH, 'X_val_sc.npy'),   X_val)
np.save(os.path.join(PROCESSED_PATH, 'y_val_sc.npy'),   y_val)
np.save(os.path.join(PROCESSED_PATH, 'X_test_sc.npy'),  X_test)
np.save(os.path.join(PROCESSED_PATH, 'y_test_sc.npy'),  y_test)
print("\nCassette split 저장 완료!")

X_comb = np.load(os.path.join(PROCESSED_PATH, 'X_combined.npy'))
y_comb = np.load(os.path.join(PROCESSED_PATH, 'y_combined.npy'))
subj_comb = np.load(os.path.join(PROCESSED_PATH, 'subj_combined.npy'))

X_train_c, y_train_c, X_val_c, y_val_c, X_test_c, y_test_c, n_tr_c, n_val_c, n_te_c = split_dataset(X_comb, y_comb, subj_comb)

print("\n=== Combined Split 결과 ===")
print(f"Train: {X_train_c.shape} ({n_tr_c}명)")
print(f"Val  : {X_val_c.shape} ({n_val_c}명)")
print(f"Test : {X_test_c.shape} ({n_te_c}명)")

np.save(os.path.join(PROCESSED_PATH, 'X_train_comb.npy'), X_train_c)
np.save(os.path.join(PROCESSED_PATH, 'y_train_comb.npy'), y_train_c)
np.save(os.path.join(PROCESSED_PATH, 'X_val_comb.npy'),   X_val_c)
np.save(os.path.join(PROCESSED_PATH, 'y_val_comb.npy'),   y_val_c)
np.save(os.path.join(PROCESSED_PATH, 'X_test_comb.npy'),  X_test_c)
np.save(os.path.join(PROCESSED_PATH, 'y_test_comb.npy'),  y_test_c)
print("Combined split 저장 완료!")