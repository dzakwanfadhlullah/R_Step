# Time-Series Forecasting Comparison: LSTM vs H2O AutoML

## 📋 Project Overview
Proyek perbandingan akademis untuk thesis/skripsi yang membandingkan performa model **LSTM (Deep Learning)** dengan **H2O AutoML** pada data time-series tahunan BPJS Ketenagakerjaan.

## 📊 Datasets
- **Klaim ARIMA.csv** - Data klaim BPJS (2014-2025, 12 titik data)
- **Iuran ARIMA.csv** - Data iuran BPJS (2015-2025, 11 titik data)

### Target Columns
| Kolom | Deskripsi |
|-------|-----------|
| JHT | Jaminan Hari Tua |
| JKK | Jaminan Kecelakaan Kerja |
| JKM | Jaminan Kematian |
| JP | Jaminan Pensiun |
| JKP | Jaminan Kehilangan Pekerjaan |
| BPJS | Total BPJS |

## 🚀 How to Run
```bash
git clone <repo-url>
cd project-timeseries-forecast
```

```r
# Restore dependencies
renv::restore()

# Run full pipeline
source("main.R")
```


## 📁 Project Structure
```
project-timeseries-forecast/
├── data/
│   ├── raw/              # CSV asli
│   └── processed/        # Data bersih & laporan
├── R/                    # Script utilitas bersama
├── experiments/
│   ├── lstm/             # Model LSTM
│   └── h2o_automl/       # Model H2O AutoML
├── results/
│   ├── predictions/      # File prediksi
│   ├── metrics/          # Evaluasi metrik
│   └── plots/            # Visualisasi
├── main.R                # Script utama
└── README.md
```

## Dependencies
- R version: 4.5.2 (2025-10-31 ucrt)
- Core packages (pinned in renv.lock):
  - readr: 2.1.6
  - dplyr: 1.1.4
  - tidyr: 1.3.2
  - data.table: 1.18.0
  - zoo: 1.8-15
  - tseries: 0.10-59
  - forecast: 9.0.0
  - keras: 2.16.0
  - tensorflow: 2.20.0
  - h2o: 3.44.0.3
  - ggplot2: 4.0.1
  - scales: 1.4.0
  - patchwork: 1.3.2
  - jsonlite: 2.0.0
  - knitr: 1.51

## ⚠️ Known Limitations

- Data sangat kecil (11-12 titik per series)
- Missing values struktural pada JKP (pre-2021)
- Missing JP 2014 pada dataset Klaim

## 📝 License
Academic use only.
