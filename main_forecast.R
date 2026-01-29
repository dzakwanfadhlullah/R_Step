# ==============================================================================
# main_forecast.R - Forecasting Pipeline (2026-2030)
# ==============================================================================
# This is the main entry point for running the FORECASTING pipeline.
# Unlike main.R (validation mode), this script:
#   - Trains on ALL available data (2014-2025)
#   - Generates forecasts for 2026-2030
#   - Produces full timeline plots similar to Excel reference
#
# Usage:
#   source("main_forecast.R")
# ==============================================================================

message("\n", strrep("=", 70))
message(" BPJS KETENAGAKERJAAN - FORECASTING PIPELINE 2026-2030")
message(" Started: ", format(Sys.time(), "%Y-%m-%d %H:%M:%S"))
message(strrep("=", 70), "\n")

# Record start time
PIPELINE_START <- Sys.time()

# ==============================================================================
# STEP 1: SOURCE ALL SCRIPTS
# ==============================================================================
message("\nStep 1: Loading dependencies and sourcing scripts...")

source("R/config.R")
source("R/utils_io.R")
source("R/preprocess.R")
source("R/plotting_forecast.R")

# Pre-load Environment Setup
message("Step 1.1: Initializing AI Environments...")
source("experiments/lstm/scripts/00_setup_keras.R")
source("experiments/h2o_automl/scripts/00_setup_h2o.R")

# Load forecast scripts
source("experiments/lstm/scripts/03_forecast_lstm.R")
source("experiments/h2o_automl/scripts/03_forecast_h2o.R")

message("✓ All scripts loaded\n")

# ==============================================================================
# STEP 2: INGEST AND PREPROCESS DATA
# ==============================================================================
message("Step 2: Ingesting and preprocessing data...")

datasets_clean <- list()

for (dataset_name in DATASETS) {
  tryCatch({
    raw_files <- list.files(
      DATA_RAW_PATH, 
      pattern = sprintf("(?i)%s.*\\.csv$", dataset_name),
      full.names = TRUE
    )
    
    if (length(raw_files) == 0) {
      warning(sprintf("No raw data file found for dataset: %s", dataset_name))
      next
    }
    
    raw_data <- read_raw_csv(raw_files[1])
    clean_data <- clean_dataset(raw_data, dataset_name)
    datasets_clean[[dataset_name]] <- clean_data
    
    message(sprintf("  ✓ %s: loaded and cleaned", dataset_name))
  }, error = function(e) {
    warning(sprintf("Error processing %s: %s", dataset_name, e$message))
  })
}

message("✓ Data preprocessing complete\n")

# ==============================================================================
# STEP 3: RUN FORECASTING PIPELINE
# ==============================================================================
message("Step 3: Running forecasting pipeline...")
message(sprintf("  Forecast horizon: %s", paste(FORECAST_YEARS, collapse = ", ")))

all_forecasts <- list()
success_count <- 0
skip_count <- 0

for (dataset_name in names(datasets_clean)) {
  message(sprintf("\n=== Processing %s ===", toupper(dataset_name)))
  
  clean_data <- datasets_clean[[dataset_name]]
  all_forecasts[[dataset_name]] <- list()
  
  for (target_col in TARGET_COLUMNS) {
    # Check if target exists
    if (is.null(clean_data$by_target[[target_col]])) {
      message(sprintf("  ⚠️ Skipped %s: no data", target_col))
      skip_count <- skip_count + 1
      next
    }
    
    n_points <- nrow(clean_data$by_target[[target_col]])
    
    # Skip if too few points
    if (n_points < 4) {
      message(sprintf("  ⚠️ Skipped %s: only %d points (need >= 4)", target_col, n_points))
      skip_count <- skip_count + 1
      next
    }
    
    all_forecasts[[dataset_name]][[target_col]] <- list()
    
    # Run LSTM forecast
    tryCatch({
      lstm_result <- run_forecast_pipeline_lstm(
        cleaned_data = clean_data,
        dataset_name = dataset_name,
        target_col = target_col,
        forecast_years = FORECAST_YEARS,
        epochs = 100,
        verbose = 0
      )
      all_forecasts[[dataset_name]][[target_col]]$lstm <- lstm_result
      if (!is.null(lstm_result)) success_count <- success_count + 1
    }, error = function(e) {
      warning(sprintf("LSTM error %s/%s: %s", dataset_name, target_col, e$message))
    })
    
    # Run H2O forecast
    tryCatch({
      h2o_result <- run_forecast_pipeline_h2o(
        cleaned_data = clean_data,
        dataset_name = dataset_name,
        target_col = target_col,
        forecast_years = FORECAST_YEARS
      )
      all_forecasts[[dataset_name]][[target_col]]$h2o <- h2o_result
      if (!is.null(h2o_result)) success_count <- success_count + 1
    }, error = function(e) {
      warning(sprintf("H2O error %s/%s: %s", dataset_name, target_col, e$message))
    })
  }
}

message("\n✓ Forecasting complete\n")

# ==============================================================================
# STEP 4: AGGREGATE ALL FORECASTS
# ==============================================================================
message("Step 4: Aggregating forecasts...")

# Combine all forecasts into single CSV
all_forecast_rows <- list()

for (ds in names(all_forecasts)) {
  for (tgt in names(all_forecasts[[ds]])) {
    tgt_data <- all_forecasts[[ds]][[tgt]]
    
    if (!is.null(tgt_data$lstm)) {
      all_forecast_rows[[length(all_forecast_rows) + 1]] <- tgt_data$lstm$combined
    }
    if (!is.null(tgt_data$h2o)) {
      all_forecast_rows[[length(all_forecast_rows) + 1]] <- tgt_data$h2o$combined
    }
  }
}

if (length(all_forecast_rows) > 0) {
  all_forecasts_df <- do.call(rbind, all_forecast_rows)
  
  forecast_dir <- file.path("results", "forecasts")
  dir.create(forecast_dir, recursive = TRUE, showWarnings = FALSE)
  
  all_path <- file.path(forecast_dir, "all_forecasts.csv")
  write.csv(all_forecasts_df, all_path, row.names = FALSE)
  message(sprintf("✓ All forecasts saved: %s", all_path))
}

# ==============================================================================
# STEP 5: GENERATE PLOTS
# ==============================================================================
message("\nStep 5: Generating forecast plots...")

generate_all_forecast_plots(all_forecasts, output_dir = "results/plots")

# ==============================================================================
# STEP 6: SAVE RUN CONFIGURATION
# ==============================================================================
message("\nStep 6: Saving run configuration...")

run_config <- list(
  run_id = format(Sys.time(), "%Y%m%d_%H%M%S"),
  timestamp = as.character(Sys.time()),
  mode = "forecasting",
  forecast_years = FORECAST_YEARS,
  system = list(
    r_version = R.version.string,
    platform = R.version$platform,
    os = Sys.info()["sysname"]
  ),
  results = list(
    success_count = success_count,
    skip_count = skip_count
  )
)

config_path <- file.path("results", sprintf("run_config_forecast_%s.json", run_config$run_id))
write(jsonlite::toJSON(run_config, pretty = TRUE, auto_unbox = TRUE), config_path)
message(sprintf("✓ Config saved: %s", config_path))

# ==============================================================================
# STEP 7: SHUTDOWN H2O
# ==============================================================================
message("\nStep 7: Cleaning up...")
tryCatch({
  h2o.shutdown(prompt = FALSE)
  message("✓ H2O cluster shutdown")
}, error = function(e) {
  message("  (H2O already stopped)")
})

# ==============================================================================
# SUMMARY
# ==============================================================================
PIPELINE_END <- Sys.time()
duration <- difftime(PIPELINE_END, PIPELINE_START, units = "secs")

message("\n", strrep("=", 70))
message(" PIPELINE SUMMARY")
message(strrep("=", 70))
message(sprintf("\nDuration: %.1f seconds", as.numeric(duration)))
message(sprintf("\nForecast Years: %s", paste(FORECAST_YEARS, collapse = ", ")))
message(sprintf("\nModel Training:"))
message(sprintf("  ✓ Successful: %d", success_count))
message(sprintf("  ⚠️ Skipped:    %d", skip_count))
message(sprintf("\nOutput Files:"))
message(sprintf("  - Forecasts: results/forecasts/"))
message(sprintf("  - Plots:     results/plots/"))
message(sprintf("  - Config:    %s", config_path))
message(sprintf("\nCompleted: %s", format(Sys.time(), "%Y-%m-%d %H:%M:%S")))
message("\n", strrep("=", 70))
message(" PIPELINE COMPLETE")
message(strrep("=", 70), "\n")
