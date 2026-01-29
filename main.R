# ==============================================================================
# main.R - Time-Series Forecasting Pipeline Orchestrator
# ==============================================================================
# This is the main entry point for running the complete forecasting pipeline.
#
# Pipeline Steps:
#   1. Load all dependencies and source R scripts
#   2. Load configuration
#   3. Ingest and preprocess data
#   4. Generate missing value report
#   5. Feature engineering
#   6. Train models (LSTM + H2O AutoML) per dataset and target
#   7. Generate plots
#   8. Generate comparison tables
#   9. Save run configuration
#   10. Print summary
#
# Usage:
#   source("main.R")
#   # Or from command line:
#   Rscript main.R
# ==============================================================================

message("\n", strrep("=", 70))
message(" TIME-SERIES FORECASTING PIPELINE")
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
source("R/split.R")
source("R/features_timeseries.R")
source("R/metrics.R")
source("R/run_config.R")
source("R/plotting.R")

# Pre-load Environment Setup (One-time)
message("Step 1.1: Initializing AI Environments (Keras & H2O)...")
source("experiments/lstm/scripts/00_setup_keras.R")
source("experiments/h2o_automl/scripts/00_setup_h2o.R")

# Pre-load Model Training Scripts
source("experiments/lstm/scripts/01_train_lstm.R")
source("experiments/lstm/scripts/02_predict_lstm.R")
source("experiments/h2o_automl/scripts/01_train_h2o.R")

message("✓ All scripts and environments loaded\n")

# ==============================================================================
# STEP 2: INITIALIZE TRACKING VARIABLES
# ==============================================================================
message("Step 2: Initializing pipeline...")

# Track results
all_predictions <- list()
all_metrics <- list()
skipped_targets <- list()
failed_targets <- list()

# Track counts
success_count <- 0
skip_count <- 0
fail_count <- 0

message("✓ Pipeline initialized\n")

# ==============================================================================
# STEP 3: INGEST AND PREPROCESS DATA
# ==============================================================================
message("Step 3: Ingesting and preprocessing data...")

datasets_clean <- list()

for (dataset_name in DATASETS) {
  tryCatch({
    # Find data file
    raw_files <- list.files(
      DATA_RAW_PATH, 
      pattern = sprintf("(?i)%s.*\\.csv$", dataset_name),
      full.names = TRUE
    )
    
    if (length(raw_files) == 0) {
      warning(sprintf("No raw data file found for dataset: %s", dataset_name))
      next
    }
    
    # Read and clean
    raw_data <- read_raw_csv(raw_files[1])
    clean_data <- clean_dataset(raw_data, dataset_name)
    datasets_clean[[dataset_name]] <- clean_data
    
    # Save cleaned data
    clean_path <- file.path(DATA_PROCESSED_PATH, sprintf("%s_clean.csv", dataset_name))
    save_clean_dataset(clean_data, dataset_name)
    
    message(sprintf("  ✓ %s: %d rows, %d targets processed", 
                    dataset_name, nrow(raw_data), length(clean_data$by_target)))
    
  }, error = function(e) {
    warning(sprintf("Failed to process dataset %s: %s", dataset_name, e$message))
    fail_count <<- fail_count + 1
  })
}

message("✓ Data preprocessing complete\n")

# ==============================================================================
# STEP 4: GENERATE MISSING VALUE REPORT
# ==============================================================================
message("Step 4: Generating missing value report...")

tryCatch({
  # Report is generated during preprocessing, just verify it exists
  report_path <- file.path(DATA_PROCESSED_PATH, "missing_report.csv")
  if (file.exists(report_path)) {
    report <- read.csv(report_path)
    n_missing <- sum(report$is_missing, na.rm = TRUE)
    message(sprintf("  ✓ Missing report: %d missing values documented", n_missing))
  } else {
    message("  ⚠️ Missing report not found, generating...")
    # Generate will happen in clean_dataset
  }
}, error = function(e) {
  warning(sprintf("Missing report error: %s", e$message))
})

message("✓ Missing value documentation complete\n")

# ==============================================================================
# STEP 5: MODEL TRAINING LOOP
# ==============================================================================
message("Step 5: Training models...")

# Minimum data points for training
MIN_DATA_POINTS <- 6
LSTM_ENABLED <- TRUE
H2O_ENABLED <- TRUE

for (dataset_name in names(datasets_clean)) {
  clean_data <- datasets_clean[[dataset_name]]
  
  for (target_col in names(clean_data$by_target)) {
    target_df <- clean_data$by_target[[target_col]]
    n_points <- nrow(target_df)
    
    task_id <- sprintf("%s/%s", dataset_name, target_col)
    message(sprintf("\n--- Processing: %s (%d data points) ---", task_id, n_points))
    
    # Skip if insufficient data
    if (n_points < MIN_DATA_POINTS) {
      msg <- sprintf("Skipped: only %d points available (min: %d)", n_points, MIN_DATA_POINTS)
      message(sprintf("  ⚠️ %s", msg))
      skipped_targets[[task_id]] <- msg
      skip_count <- skip_count + 1
      next
    }
    
    # ----- LSTM Training -----
    if (LSTM_ENABLED) {
      tryCatch({
        message("  Training LSTM...")
        
        # Train model (Function already loaded)
        lstm_result <- train_lstm_for_target(
          cleaned_data = clean_data,
          target_col = target_col,
          lookback = LSTM_LOOKBACK,
          holdout_years = HOLDOUT_YEARS,
          epochs = 100,
          batch_size = 1,
          verbose = 0
        )
        
        if (!is.null(lstm_result)) {
          # Save model (Function already loaded)
          save_lstm_model(lstm_result$model, dataset_name, target_col)
          save_training_log(lstm_result$history, dataset_name, target_col)
          
          # Save predictions (use test predictions for main result)
          preds_df <- lstm_result$predictions$test
          preds_df$dataset <- dataset_name
          preds_df$target <- target_col
          preds_df$model <- "lstm"
          all_predictions[[paste0("lstm_", task_id)]] <- preds_df
          
          # Calculate metrics
          metrics <- calculate_all_metrics(preds_df$actual, preds_df$predicted)
          metrics$dataset <- dataset_name
          metrics$target <- target_col
          metrics$model <- "lstm"
          all_metrics[[paste0("lstm_", task_id)]] <- metrics
          
          message("  ✓ LSTM training complete")
          success_count <- success_count + 1
        } else {
          # Mark as failed if result is NULL but no error caught
          failed_targets[[paste0("lstm_", task_id)]] <- "Result was NULL"
          fail_count <- fail_count + 1
        }
        
      }, error = function(e) {
        msg <- sprintf("LSTM failed: %s", e$message)
        message(sprintf("  ✗ %s", msg))
        failed_targets[[paste0("lstm_", task_id)]] <- msg
        fail_count <<- fail_count + 1
        
        # Log error
        log_dir <- "experiments/lstm/logs"
        dir.create(log_dir, recursive = TRUE, showWarnings = FALSE)
        cat(sprintf("[%s] %s: %s\n", Sys.time(), task_id, msg),
            file = file.path(log_dir, "errors.txt"), append = TRUE)
      })
    }
    
    # ----- H2O AutoML Training -----
    if (H2O_ENABLED) {
      tryCatch({
        message("  Training H2O AutoML...")
        
        # Train model (Function and connection already loaded)
        h2o_result <- run_h2o_pipeline(
          cleaned_data = clean_data,
          dataset_name = dataset_name,
          target_col = target_col,
          max_runtime_secs = H2O_MAX_RUNTIME_SECS
        )
        
        if (!is.null(h2o_result)) {
          # Predictions already saved by pipeline
          preds_df <- h2o_result$predictions
          preds_df$dataset <- dataset_name
          preds_df$target <- target_col
          preds_df$model <- "h2o"
          all_predictions[[paste0("h2o_", task_id)]] <- preds_df
          
          # Calculate metrics
          metrics <- calculate_all_metrics(preds_df$actual, preds_df$predicted)
          metrics$dataset <- dataset_name
          metrics$target <- target_col
          metrics$model <- "h2o"
          all_metrics[[paste0("h2o_", task_id)]] <- metrics
          
          message("  ✓ H2O AutoML training complete")
          success_count <- success_count + 1
        }
        
      }, error = function(e) {
        msg <- sprintf("H2O failed: %s", e$message)
        message(sprintf("  ✗ %s", msg))
        failed_targets[[paste0("h2o_", task_id)]] <- msg
        fail_count <<- fail_count + 1
        
        # Log error
        log_dir <- "experiments/h2o_automl/outputs"
        dir.create(log_dir, recursive = TRUE, showWarnings = FALSE)
        cat(sprintf("[%s] %s: %s\n", Sys.time(), task_id, msg),
            file = file.path(log_dir, "errors.txt"), append = TRUE)
      })
    }
  }
}

message("\n✓ Model training phase complete\n")

# ==============================================================================
# STEP 6: AGGREGATE AND SAVE RESULTS
# ==============================================================================
message("Step 6: Aggregating results...")

# Combine all predictions
if (length(all_predictions) > 0) {
  combined_preds <- do.call(rbind, lapply(all_predictions, function(x) {
    # Ensure consistent columns
    x$timestamp <- Sys.time()
    x[, c("dataset", "target", "model", "year", "actual", "predicted", "timestamp")]
  }))
  
  # Save combined predictions
  pred_path <- file.path(RESULTS_PATH, "predictions", "all_predictions.csv")
  dir.create(dirname(pred_path), recursive = TRUE, showWarnings = FALSE)
  write.csv(combined_preds, pred_path, row.names = FALSE)
  message(sprintf("  ✓ All predictions saved: %s", pred_path))
}

# Combine all metrics
if (length(all_metrics) > 0) {
  metrics_df <- do.call(rbind, lapply(all_metrics, function(x) {
    as.data.frame(x, stringsAsFactors = FALSE)
  }))
  
  # Save detailed metrics
  metrics_path <- file.path(RESULTS_PATH, "metrics", "metrics_detail.csv")
  dir.create(dirname(metrics_path), recursive = TRUE, showWarnings = FALSE)
  write.csv(metrics_df, metrics_path, row.names = FALSE)
  message(sprintf("  ✓ Detailed metrics saved: %s", metrics_path))
  
  # Create summary with rankings
  metrics_summary <- create_metrics_summary(metrics_df, rank_by = "rmse")
  save_metrics_summary(metrics_summary, "metrics_summary.csv")
  
  # Create comparison table
  if (length(unique(metrics_df$model)) > 1) {
    comparison <- compare_models(metrics_df, metric = "rmse")
    comp_path <- file.path(RESULTS_PATH, "compare_models.csv")
    write.csv(comparison, comp_path, row.names = FALSE)
    message(sprintf("  ✓ Model comparison saved: %s", comp_path))
  }
}

message("✓ Results aggregation complete\n")

# ==============================================================================
# STEP 7: GENERATE PLOTS
# ==============================================================================
message("Step 7: Generating plots...")

if (length(all_predictions) > 0) {
  tryCatch({
    combined_preds <- do.call(rbind, all_predictions)
    plot_results <- generate_all_plots(
      preds_all = combined_preds,
      metrics_all = if (exists("metrics_df")) metrics_df else NULL,
      output_dir = file.path(RESULTS_PATH, "plots"),
      dpi = 300
    )
  }, error = function(e) {
    warning(sprintf("Plot generation failed: %s", e$message))
  })
} else {
  message("  ⚠️ No predictions available for plotting")
}

message("✓ Plot generation complete\n")

# ==============================================================================
# STEP 8: SAVE RUN CONFIGURATION
# ==============================================================================
message("Step 8: Saving run configuration...")

run_config <- create_full_run_log(
  model_results = list(
    success_count = success_count,
    skip_count = skip_count,
    fail_count = fail_count
  ),
  skipped_targets = skipped_targets
)

message("✓ Run configuration saved\n")

# ==============================================================================
# STEP 9: SHUTDOWN H2O (if running)
# ==============================================================================
if (H2O_ENABLED) {
  tryCatch({
    if (requireNamespace("h2o", quietly = TRUE)) {
      h2o::h2o.shutdown(prompt = FALSE)
      message("✓ H2O cluster shutdown\n")
    }
  }, error = function(e) {
    # Ignore shutdown errors
  })
}

# ==============================================================================
# STEP 10: PRINT FINAL SUMMARY
# ==============================================================================
PIPELINE_END <- Sys.time()
duration <- as.numeric(difftime(PIPELINE_END, PIPELINE_START, units = "secs"))

message("\n", strrep("=", 70))
message(" PIPELINE SUMMARY")
message(strrep("=", 70))
message(sprintf("
Duration: %.1f seconds

Model Training Results:
  ✓ Successful: %d
  ⚠️ Skipped:    %d
  ✗ Failed:     %d

Skipped Targets:
%s

Failed Targets:
%s

Output Files:
  - Predictions: results/predictions/
  - Metrics:     results/metrics/
  - Plots:       results/plots/
  - Config:      results/run_config_*.json

Completed: %s
",
  duration,
  success_count,
  skip_count,
  fail_count,
  if (length(skipped_targets) > 0) {
    paste("  -", names(skipped_targets), ":", unlist(skipped_targets), collapse = "\n")
  } else { "  (none)" },
  if (length(failed_targets) > 0) {
    # Combine names and values into a clearer string
    fail_msgs <- vapply(seq_along(failed_targets), function(i) {
      sprintf("  - %s: %s", names(failed_targets)[i], failed_targets[[i]])
    }, character(1))
    paste(fail_msgs, collapse = "\n")
  } else { "  (none)" },
  format(PIPELINE_END, "%Y-%m-%d %H:%M:%S")
))

message(strrep("=", 70))
message(" PIPELINE COMPLETE")
message(strrep("=", 70), "\n")
