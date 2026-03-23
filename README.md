# ECG Signal Quality Classifier V3 — Clinical Grade

A progressive benchmark of five machine-learning models for binary classification of 12-lead ECG signal quality (clean vs. degraded), evaluated on the [PTB-XL](https://physionet.org/content/ptb-xl/1.0.3/) dataset.

## Motivation

Poor signal quality in electrocardiograms leads to misdiagnosis and wasted clinical time. Automated quality screening can flag degraded recordings before they reach a cardiologist. This project builds and compares five classifiers of increasing complexity, from a simple logistic regression to a CNN-Transformer ensemble with uncertainty quantification so practitioners can choose the right accuracy/complexity trade-off for their deployment environment.

## Models

| # | Model | Input | Key Idea |
|---|-------|-------|----------|
| 1 | **Logistic Regression A** | Basic SQI features | L1-regularized, 15 selected features |
| 2 | **Logistic Regression B** | Enhanced SQI features | L1-regularized, 20 selected features including HRV & spectral entropy |
| 3 | **MLP** (DL1) | Basic SQI features | Simple feedforward network with class-weighted BCE loss |
| 4 | **CNN** (DL2) | Raw 12-lead waveforms | 1-D ResNet-style convolutional network |
| 5 | **CNN-Transformer Ensemble** (DL3) | Raw 12-lead waveforms | Multi-scale CNN + Transformer encoder, 5-member ensemble, focal loss, epistemic & aleatoric uncertainty, conformal prediction |

All models share the same stratified train / validation / test split (PTB-XL `strat_fold`), the same threshold-optimization procedure (Youden, F2, cost-sensitive), and the same evaluation metrics so results are directly comparable.

## Features & Signal Processing

**Preprocessing pipeline** — applied per-lead before feature extraction:
1. Baseline removal via median filter
2. 60 Hz notch filter (Q = 30)
3. Low-pass Butterworth filter (40 Hz, 3rd order)

**Basic SQI features** (per-lead, then aggregated across 12 leads as mean ± std): signal statistics (mean, std, RMS, peak-to-peak, skew, kurtosis), Hjorth parameters (mobility, complexity), spectral power bands (total, QRS-band, baseline, high-frequency), power-fraction ratios, and composite indices (pSQI, kSQI, sSQI, basSQI).

**Enhanced SQI features** add: inter-detector agreement (iSQI) between Pan-Tompkins and Hamilton R-peak detectors, HRV statistics (RR mean, std, RMSSD, CV), spectral entropy, inter-lead correlation statistics, and limb/precordial SQI group means.

Feature selection uses variance thresholding followed by mutual-information ranking.

## Dataset

**PTB-XL** — 21,799 twelve-lead ECG recordings (10 s, 500 Hz) with clinician annotations. Degradation labels are derived from four noise-annotation columns (`baseline_drift`, `static_noise`, `burst_noise`, `electrodes_problems`): any non-zero annotation marks a recording as *degraded*.

> Wagner, P., et al. "PTB-XL, a large publicly available electrocardiography dataset." *Scientific Data* 7, 154 (2020).

## Evaluation

Every model is assessed on the held-out test fold with:

- **ROC-AUC** and **AUPRC** (area under the precision-recall curve)
- **Sensitivity / Specificity / PPV / NPV** at the Youden-optimal threshold
- **Expected Calibration Error (ECE)**
- Per-model diagnostic dashboards (ROC & PR curves, calibration plot, confusion matrix, threshold analysis)

The CNN-Transformer ensemble additionally reports epistemic uncertainty (inter-member disagreement), aleatoric uncertainty, and conformal-prediction coverage.

A final comparison table ranks all five models side-by-side.

## Interpretability

Gradient-based saliency analysis is included for both deep-learning models that operate on raw waveforms:

- **Vanilla gradients** and **Gradient × Input** maps for the CNN and CNN-Transformer
- **Ensemble-averaged saliency** for the CNN-Transformer (mean across 5 members)
- **Lead-importance comparison** — per-lead mean |grad × input| to reveal which ECG leads each architecture relies on most

## Project Structure

```
├── ECG_Classifier_V3.ipynb   # Full pipeline: data → features → training → evaluation
├── output_clinical/           # Cached models, features, and result dashboards (generated)
│   └── results/               # Saved figures (ROC curves, dashboards, saliency maps)
└── README.md
```

## Requirements

```
python >= 3.9
numpy
pandas
scipy
scikit-learn
matplotlib
seaborn
wfdb
torch
joblib
```

Install with:

```bash
pip install numpy pandas scipy scikit-learn matplotlib seaborn wfdb torch joblib
```

## Quick Start

1. **Download PTB-XL** from [PhysioNet](https://physionet.org/content/ptb-xl/1.0.3/) and extract it.

2. **Set the data path** — open the notebook and update `DATA_PATH` near the top of the *Data and Feature Loading* section to point at your extracted PTB-XL directory.

3. **Run the notebook end-to-end.** Models and features are automatically cached to `output_clinical/` so subsequent runs skip expensive computation.

```bash
jupyter notebook ECG_Classifier_V3.ipynb
```

GPU/MPS acceleration is detected automatically for the CNN and CNN-Transformer models; the pipeline falls back to CPU if neither is available.

## Caching

Every trained model and extracted feature set is persisted to disk with a deterministic parameter hash. Changing a hyperparameter (e.g. learning rate, number of ensemble members) produces a new cache key, so you can experiment without overwriting previous results.

## License

This project is provided for research and educational purposes. The PTB-XL dataset is available under the [Open Data Commons Attribution License (ODC-BY v1.0)](https://physionet.org/content/ptb-xl/1.0.3/).
