RANDOM_SEED <- 42

DATA_RAW_PATH <- file.path("data", "raw")
DATA_PROCESSED_PATH <- file.path("data", "processed")
RESULTS_PATH <- "results"

TARGET_COLUMNS <- c("JHT", "JKK", "JKM", "JP", "JKP", "BPJS")
DATASETS <- c("klaim", "iuran")

LSTM_LOOKBACK <- 3
LSTM_UNITS <- 16
# H2O Settings
H2O_MAX_RUNTIME_SECS <- 30 # Reduced for faster execution on small data
H2O_NFOLDS <- 3            # Cross-validation folds
H2O_MAX_MODELS <- 15       # Limit models to prevent overhead on small datasets

# Forecasting Mode Configuration
FORECAST_MODE <- TRUE                    # TRUE = forecast future, FALSE = validate
FORECAST_YEARS <- c(2026, 2027, 2028, 2029, 2030)  # Years to forecast
HOLDOUT_YEARS <- c()                     # Empty = use ALL data for training
ROLLING_INITIAL_WINDOW <- 5

# ==============================================================================
# TREATMENT_RULES - Missing Value Handling Configuration
# ==============================================================================
# These rules determine how missing values are treated during preprocessing
# and modeling. All treatments are logged to missing_report.csv for transparency.
#
# STRICT_MODE: 
#   TRUE  -> Exclude years with missing values (no imputation)
#   FALSE -> Allow imputation methods
#
# ALLOW_INTERPOLATION:
#   TRUE  -> Use interpolation for legacy missing (JP 2014)
#   FALSE -> Exclude year instead of interpolating
#
# JKP_MIN_POINTS_WARNING:
#   Minimum data points threshold. If available points < this value,
#   emit a warning about model reliability.
# ==============================================================================

TREATMENT_RULES <- list(
  STRICT_MODE = TRUE,
  ALLOW_INTERPOLATION = FALSE,
  JKP_MIN_POINTS_WARNING = 5
)

# ==============================================================================
# MISSING VALUE DOCUMENTATION
# ==============================================================================
# 
# Known Missing Values in Dataset:
#
# 1. Klaim - JP (2014)
#    - Reason: legacy (data era lama tidak tersedia)
#    - Treatment: exclude_year (dengan STRICT_MODE=TRUE)
#    - Impact: JP Klaim dimodelkan dari 2015-2025 (11 titik)
#
# 2. Klaim - JKP (2014-2020)
#    - Reason: structural (program JKP belum ada)
#    - Treatment: exclude_years (model hanya 2021-2025)
#    - Impact: CRITICAL - hanya 5 titik data, model mungkin unreliable
#
# 3. Iuran - JKP (2015-2020)
#    - Reason: structural (program JKP belum ada)
#    - Treatment: exclude_years (model hanya 2021-2025)
#    - Impact: CRITICAL - hanya 5 titik data, model mungkin unreliable
#
# ==============================================================================
