# Time-Series Forecasting Pipeline

A comprehensive time-series forecasting project for BPJS Ketenagakerjaan data, comparing LSTM deep learning models with H2O AutoML ensemble methods.

## 📋 Project Overview

This project implements a complete forecasting pipeline for BPJS Ketenagakerjaan financial data, including:
- **Data ingestion and preprocessing** with transparent missing value handling
- **LSTM neural networks** using TensorFlow/Keras
- **H2O AutoML** for automated machine learning
- **Comprehensive evaluation** with multiple metrics
- **Visualization** of predictions and model comparisons

## 📊 Dataset Description

| Dataset | Description | Years | Targets |
|---------|-------------|-------|---------|
| **Klaim** | Claims/disbursement amounts | 2014-2025 | JHT, JKK, JKM, JP, JKP, BPJS |
| **Iuran** | Contribution/premium amounts | 2015-2025 | JHT, JKK, JKM, JP, JKP, BPJS |

### Target Variables
- **JHT**: Jaminan Hari Tua (Old Age Security)
- **JKK**: Jaminan Kecelakaan Kerja (Work Accident Insurance)
- **JKM**: Jaminan Kematian (Death Insurance)
- **JP**: Jaminan Pensiun (Pension)
- **JKP**: Jaminan Kehilangan Pekerjaan (Unemployment Insurance)
- **BPJS**: Aggregate total

## ⚠️ Missing Value Treatment

The pipeline handles structural and legacy missing values transparently:

| Target | Years Missing | Reason | Treatment |
|--------|---------------|--------|-----------|
| JP (Klaim) | 2014 | Legacy data unavailable | Exclude year |
| JKP | 2014-2020 | Program not yet launched | Model from 2021 only |

All treatments are logged to `data/processed/missing_report.csv`.

## 🚀 How to Run

### Prerequisites
- R 4.5+
- Python 3.10+ (for TensorFlow)
- Java 8+ (for H2O)

### Installation
```bash
# Clone repository
git clone <repository-url>
cd project-timeseries-forecast

# Install R dependencies
Rscript -e "install.packages(c('keras3', 'tensorflow', 'h2o', 'ggplot2', 'patchwork', 'jsonlite'))"

# Install TensorFlow (if needed)
Rscript -e "keras3::install_keras()"
```

### Run Complete Pipeline
```r
# In R or RStudio
setwd("project-timeseries-forecast")
source("main.R")
```

### Run Individual Components
```r
# Just LSTM
source("experiments/lstm/scripts/00_setup_keras.R")
source("experiments/lstm/scripts/01_train_lstm.R")

# Just H2O AutoML
source("experiments/h2o_automl/scripts/01_train_h2o.R")
```

## 📁 Output Files

```
results/
├── predictions/
│   ├── all_predictions.csv          # Combined predictions
│   ├── predictions_lstm_*.csv       # Per-model predictions
│   └── predictions_h2o_*.csv
├── metrics/
│   ├── metrics_detail.csv           # All metrics
│   ├── metrics_summary.csv          # Ranked summary
│   └── compare_models.csv           # Model comparison
├── plots/
│   ├── actual_vs_pred_*.png         # Forecast plots
│   ├── error_*.png                  # Error analysis
│   └── comparison_rmse.png          # Model comparison
└── run_config_*.json                # Run configuration
```

## 📏 Evaluation Design

### Metrics
| Metric | Formula | Interpretation |
|--------|---------|----------------|
| MAE | mean(\|actual - pred\|) | Average absolute error |
| RMSE | √mean((actual - pred)²) | Penalizes large errors |
| sMAPE | 2×\|error\|/(|a|+|p|) × 100 | Symmetric percentage error |
| R² | 1 - SS_res/SS_tot | Explained variance |

### Validation Strategy
- **Holdout**: Last 2 years (2024-2025) reserved for testing
- **LSTM**: Sequence-based with lookback window
- **H2O**: 3-fold cross-validation on training data

## ⚙️ Configuration

Edit `R/config.R` to customize:
```r
RANDOM_SEED <- 42
HOLDOUT_YEARS <- c(2024, 2025)
LSTM_LOOKBACK <- 3
LSTM_UNITS <- 16
H2O_MAX_RUNTIME_SECS <- 120
```

## 📦 Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| keras3 | 3.x | LSTM modeling |
| tensorflow | 2.20+ | Deep learning backend |
| h2o | 3.x | AutoML |
| ggplot2 | 3.x | Visualization |
| patchwork | 1.x | Plot composition |
| jsonlite | 1.x | Config export |

## 🔄 Reproducibility

- Random seed set in `config.R` (default: 42)
- TensorFlow seed set via `tf$random$set_seed()`
- H2O seed passed to `h2o.automl()`
- All run parameters logged to `run_config_*.json`

## ⚡ Limitations

1. **Small sample size**: Only 11-12 data points per target
2. **JKP targets**: Only 5 data points (2021-2025) - models may be unreliable
3. **No exogenous variables**: Pure univariate forecasting
4. **Annual granularity**: Cannot capture seasonal patterns

## 📝 License

This project is for academic/thesis purposes.

---
*Generated: 2026-01-28*
