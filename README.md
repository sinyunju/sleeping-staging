Automatic Sleep Stage Classification Using PSG Signals

> [2026-1] 기계학습 Term Project

---

## 프로젝트 개요

Sleep-EDF Expanded 데이터셋을 활용하여 PSG(Polysomnography) 신호에서 수면 단계(Wake / N1 / N2 / N3 / REM)를 자동 분류하는 기계학습 모델을 개발하고, 3단계 체계적 실험을 통해 채널 구성과 데이터 도메인 특성이 모델 성능에 미치는 영향을 분석하였다.

### 실험 구성

| 단계 | 고정 조건 | 변화 조건 |
|------|-----------|-----------|
| 1단계 (기본 실험) | EEG 2ch + Cassette | 4개 모델 비교 |
| 2단계 (채널 실험) | Cassette 고정 | EEG → EEG+EOG → EEG+EOG+EMG |
| 3단계 (데이터 실험) | EEG 2ch 고정 | Cassette → Combined → Combined+snorm |

### 주요 결과

- **Best 성능**: 1D-CNN + EEG+EOG+EMG + Cassette → Macro F1 **0.7570**, Cohen's κ **0.6805**
- 채널 추가 효과는 모델 구조에 종속적: RF·1D-CNN 향상, CNN+LSTM 오히려 하락
- Combined 단순 결합 시 도메인 불일치로 성능 저하 → 피험자별 정규화(snorm)로 부분 회복

---

## 프로젝트 구조
```
sleeping-staging/
├── eda/
│   ├── eda.ipynb                  # 탐색적 데이터 분석
│   └── figures/                   # EDA 결과 이미지
├── preprocessing/
│   ├── preprocess.py              # 기본 전처리 (EEG 2ch, Cassette)
│   ├── preprocess_channels.py     # 채널 실험용 전처리 (EOG, EMG 추가)
│   ├── preprocess_subject_norm.py # 피험자별 정규화 (Combined+snorm)
│   └── processed/                 # 전처리 결과 .npy 파일 (gitignore)
├── models/
│   ├── baseline_rf.py             # 초기 RF 실험
│   ├── cnn1d.py                   # 초기 1D-CNN 실험
│   ├── cnn_lstm.py                # 초기 CNN+LSTM 실험
│   ├── transformer.py             # 초기 Transformer 실험
│   ├── random_search/             # 하이퍼파라미터 탐색 (N=25)
│   ├── basic_exp/                 # 1단계: EEG 2ch + Cassette 최종 모델
│   ├── channel_exp/               # 2단계: 채널 실험
│   └── data_exp/                  # 3단계: 데이터 실험
├── results/
│   ├── figures/                   # 시각화 결과 이미지
│   └── metrics/                   # 실험 결과 JSON + 모델 가중치 (.pt, gitignore)
├── visualization/
│   └── generate_figures.py        # 모든 결과 시각화 생성
├── .gitignore
├── README.md
└── requirements.txt
```
---

## 설치 및 환경 설정

### 요구 사항

- Python 3.10 이상
- macOS (Apple Silicon MPS) / Linux (CUDA) / CPU 환경 모두 지원

### 설치

```bash
# 1. 저장소 클론
git clone https://github.com/sinyunju/sleeping-staging.git
cd sleeping-staging

# 2. 가상환경 생성 및 활성화
python -m venv venv
source venv/bin/activate        # macOS/Linux

# 3. 패키지 설치
pip install -r requirements.txt

# 4. PyTorch 별도 설치 (환경에 맞게 선택)
# Apple Silicon (MPS):
pip install torch torchvision
# CUDA (GPU):
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
# CPU only:
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

---

## 데이터 준비

Sleep-EDF Expanded 데이터셋은 PhysioNet에서 직접 다운로드해야 한다.
(ODC Open Database License, 학술 연구 목적 사용 가능)

```bash
# PhysioNet 웹사이트에서 직접 다운로드
# https://physionet.org/content/sleep-edfx/1.0.0/
# 로그인 후 Download 버튼 클릭 (약 5.5GB)
```

다운로드 후 아래 경로에 위치시킨다.
```
sleeping-staging/data/sleep-edf-database-expanded-1.0.0/
├── sleep-cassette/   # SC4001E0-PSG.edf 등 306개 파일
└── sleep-telemetry/  # ST7011J0-PSG.edf 등 88개 파일
```

> **주의**: 데이터는 용량(약 5.5GB) 문제로 GitHub에 포함되지 않는다.
> 전처리 코드의 기본 데이터 경로는 `~/Downloads/sleep-edf-database-expanded-1.0.0`이다.
> 다른 경로에 저장한 경우 `preprocessing/preprocess.py` 상단의 `DATA_PATH`를 수정한다.

---

## 실험 재현 방법

### Step 1. 전처리

```bash
# 1단계·2단계용: EEG 2ch + Cassette
python preprocessing/preprocess.py

# 2단계용: EOG, EMG 채널 추가
python preprocessing/preprocess_channels.py

# 3단계용: Combined + 피험자별 정규화
python preprocessing/preprocess_subject_norm.py
```

### Step 2. 하이퍼파라미터 탐색 (선택)

```bash
python models/random_search/rf_search.py
python models/random_search/cnn1d_search.py
python models/random_search/cnn_lstm_search.py
python models/random_search/transformer_search.py
```

### Step 3. 1단계: 기본 실험

```bash
python models/basic_exp/rf_final.py
python models/basic_exp/cnn1d_final.py
python models/basic_exp/cnn_lstm_final.py
python models/basic_exp/transformer_final.py
```

### Step 4. 2단계: 채널 실험

```bash
python models/channel_exp/rf_eeg_eog.py
python models/channel_exp/cnn1d_eeg_eog.py
python models/channel_exp/cnn_lstm_eeg_eog.py
python models/channel_exp/transformer_eeg_eog.py

python models/channel_exp/rf_eeg_eog_emg.py
python models/channel_exp/cnn1d_eeg_eog_emg.py
python models/channel_exp/cnn_lstm_eeg_eog_emg.py
python models/channel_exp/transformer_eeg_eog_emg.py
```

### Step 5. 3단계: 데이터 실험

```bash
python models/data_exp/rf_combined.py
python models/data_exp/cnn1d_combined.py
python models/data_exp/cnn_lstm_combined.py
python models/data_exp/transformer_combined.py

python models/data_exp/rf_combined_snorm.py
python models/data_exp/cnn1d_combined_snorm.py
python models/data_exp/cnn_lstm_combined_snorm.py
python models/data_exp/transformer_combined_snorm.py
```

### Step 6. 시각화

```bash
python visualization/generate_figures.py
```

---

## 주요 하이퍼파라미터

| 모델 | lr | dropout | batch_size | 기타 |
|------|----|---------|------------|------|
| RF | — | — | — | n_estimators=200, class_weight=balanced |
| 1D-CNN | 1e-3 | 0.2 | 64 | base_filters=64 |
| CNN+LSTM | 3e-4 | 0.2 | 32 | lstm_hidden=256, lstm_layers=3 |
| Transformer | 1e-4 | 0.1 | 16 | d_model=256, n_heads=8, n_layers=3, warmup=5 |

---

