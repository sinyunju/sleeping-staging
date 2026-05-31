import mne
import numpy as np
import os

DATA_PATH      = os.path.expanduser(
    "~/Downloads/sleep-edf-database-expanded-1.0.0")
CASSETTE_PATH  = os.path.join(DATA_PATH, "sleep-cassette")
TELEMETRY_PATH = os.path.join(DATA_PATH, "sleep-telemetry")
PROCESSED_PATH = os.path.expanduser(
    "~/Desktop/sleeping-staging/preprocessing/processed")
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
    if len(available) < len(channels):
        return None, None
    raw.pick(available)
    raw.filter(0.5, 30, verbose=False)
    annotations = mne.read_annotations(hyp_path)
    raw.set_annotations(annotations, verbose=False)
    events, event_id = mne.events_from_annotations(
        raw, event_id=LABEL_MAPPING, verbose=False)
    if len(events) == 0:
        return None, None
    epochs = mne.Epochs(
        raw, events, event_id=event_id,
        tmin=0, tmax=29.99,
        baseline=None, verbose=False, preload=True)
    X = epochs.get_data()  # (n_epochs, n_channels, 3000)
    y = epochs.events[:, 2]


    for ch in range(X.shape[1]):
        mean = X[:, ch, :].mean()
        std  = X[:, ch, :].std() + 1e-8
        X[:, ch, :] = (X[:, ch, :] - mean) / std
    return X, y

def process_dataset(data_path, label=''):
    psg_files = sorted([f for f in os.listdir(data_path)
                        if f.endswith("PSG.edf")])
    hyp_files = sorted([f for f in os.listdir(data_path)
                        if f.endswith("Hypnogram.edf")])
    all_X, all_y, subject_ids = [], [], []

    for i, (psg_file, hyp_file) in enumerate(zip(psg_files, hyp_files)):
        print(f"  [{label}] {i+1}/{len(psg_files)}: {psg_file}")
        X, y = process_one_subject(
            os.path.join(data_path, psg_file),
            os.path.join(data_path, hyp_file))
        if X is not None:
            all_X.append(X)
            all_y.append(y)
            subject_ids.extend([i] * len(y))

    return (np.concatenate(all_X, axis=0),
            np.concatenate(all_y, axis=0),
            np.array(subject_ids))

def split_dataset(X, y, subject_ids,
                  train_ratio=0.7, val_ratio=0.15,
                  random_state=42):
    unique_subjects = np.unique(subject_ids)
    n_subjects = len(unique_subjects)
    np.random.seed(random_state)
    shuffled = np.random.permutation(unique_subjects)
    n_train  = int(n_subjects * train_ratio)
    n_val    = int(n_subjects * val_ratio)
    train_subj = shuffled[:n_train]
    val_subj   = shuffled[n_train:n_train+n_val]
    test_subj  = shuffled[n_train+n_val:]
    train_idx  = np.isin(subject_ids, train_subj)
    val_idx    = np.isin(subject_ids, val_subj)
    test_idx   = np.isin(subject_ids, test_subj)
    return (X[train_idx], y[train_idx],
            X[val_idx],   y[val_idx],
            X[test_idx],  y[test_idx])

print("=== Cassette 처리 중 ===")
X_sc, y_sc, subj_sc = process_dataset(CASSETTE_PATH, 'cassette')
print(f"Cassette shape: {X_sc.shape}")

X_tr, y_tr, X_val, y_val, X_te, y_te = split_dataset(
    X_sc, y_sc, subj_sc)

np.save(os.path.join(PROCESSED_PATH, 'X_train_sc_snorm.npy'), X_tr)
np.save(os.path.join(PROCESSED_PATH, 'y_train_sc_snorm.npy'), y_tr)
np.save(os.path.join(PROCESSED_PATH, 'X_val_sc_snorm.npy'),   X_val)
np.save(os.path.join(PROCESSED_PATH, 'y_val_sc_snorm.npy'),   y_val)
np.save(os.path.join(PROCESSED_PATH, 'X_test_sc_snorm.npy'),  X_te)
np.save(os.path.join(PROCESSED_PATH, 'y_test_sc_snorm.npy'),  y_te)
print("Cassette snorm 저장 완료!")

print("\n=== Telemetry 처리 중 ===")
X_st, y_st, subj_st = process_dataset(TELEMETRY_PATH, 'telemetry')

subj_st_offset = subj_st + 153
X_comb    = np.concatenate([X_sc, X_st], axis=0)
y_comb    = np.concatenate([y_sc, y_st], axis=0)
subj_comb = np.concatenate([subj_sc, subj_st_offset], axis=0)
print(f"Combined shape: {X_comb.shape}")

X_tr_c, y_tr_c, X_val_c, y_val_c, X_te_c, y_te_c = split_dataset(
    X_comb, y_comb, subj_comb)

np.save(os.path.join(PROCESSED_PATH, 'X_train_comb_snorm.npy'), X_tr_c)
np.save(os.path.join(PROCESSED_PATH, 'y_train_comb_snorm.npy'), y_tr_c)
np.save(os.path.join(PROCESSED_PATH, 'X_val_comb_snorm.npy'),   X_val_c)
np.save(os.path.join(PROCESSED_PATH, 'y_val_comb_snorm.npy'),   y_val_c)
np.save(os.path.join(PROCESSED_PATH, 'X_test_comb_snorm.npy'),  X_te_c)
np.save(os.path.join(PROCESSED_PATH, 'y_test_comb_snorm.npy'),  y_te_c)
print("Combined snorm 저장 완료!")

print("\n모든 피험자별 정규화 전처리 완료!")