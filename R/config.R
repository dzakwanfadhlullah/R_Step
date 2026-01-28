RANDOM_SEED <- 42

DATA_RAW_PATH <- file.path("data", "raw")
DATA_PROCESSED_PATH <- file.path("data", "processed")
RESULTS_PATH <- "results"

TARGET_COLUMNS <- c("JHT", "JKK", "JKM", "JP", "JKP", "BPJS")
DATASETS <- c("klaim", "iuran")

LSTM_LOOKBACK <- 3
LSTM_UNITS <- 16
H2O_MAX_RUNTIME_SECS <- 120

EVAL_MODE <- "rolling"
HOLDOUT_YEARS <- c(2024, 2025)
ROLLING_INITIAL_WINDOW <- 5
