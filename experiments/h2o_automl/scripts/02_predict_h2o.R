# ==============================================================================
# 02_predict_h2o.R - H2O AutoML Prediction and Model Export Script
# ==============================================================================
# This script handles:
#   - Loading saved H2O models
#   - Making predictions on new data
#   - Exporting models (MOJO format for production)
#   - Calculating evaluation metrics
#   - Proper H2O shutdown
#
# Usage:
#   source("experiments/h2o_automl/scripts/02_predict_h2o.R")
# ==============================================================================

# Set working directory to project root
if (!file.exists("R/config.R")) {
  setwd("project-timeseries-forecast")
}

library(h2o)

# Source project files and ensure H2O is running
source("R/config.R")
source("experiments/h2o_automl/scripts/00_setup_h2o.R")

# ==============================================================================
# MODEL LOADING FUNCTIONS
# ==============================================================================

# ------------------------------------------------------------------------------
# load_h2o_model: Load a saved H2O model
# ------------------------------------------------------------------------------
load_h2o_model <- function(dataset_name, target_col) {
  models_dir <- file.path("experiments", "h2o_automl", "models")
  target_dir <- file.path(models_dir, sprintf("%s_%s", dataset_name, target_col))
  
  if (!dir.exists(target_dir)) {
    stop(sprintf("Model directory not found: %s", target_dir), call. = FALSE)
  }
  
  # H2O saves model with its own ID, find the file
  model_files <- list.files(target_dir, pattern = "^[^.]+$", full.names = TRUE)
  
  if (length(model_files) == 0) {
    stop(sprintf("No model found in: %s", target_dir), call. = FALSE)
  }
  
  model_path <- model_files[1]  # Take first model
  model <- h2o.loadModel(model_path)
  
  message(sprintf("✓ Model loaded: %s", model@model_id))
  
  model
}

# ==============================================================================
# PREDICTION FUNCTIONS
# ==============================================================================

# ------------------------------------------------------------------------------
# predict_with_h2o: Make predictions using loaded model
# ------------------------------------------------------------------------------
predict_with_h2o <- function(model, new_data_df) {
  # Upload to H2O
  new_h2o <- as.h2o(new_data_df)
  
  # Predict
  predictions <- h2o.predict(model, new_h2o)
  preds_df <- as.data.frame(predictions)
  
  preds_df$predict
}

# ------------------------------------------------------------------------------
# evaluate_h2o_predictions: Calculate metrics for H2O predictions
# ------------------------------------------------------------------------------
evaluate_h2o_predictions <- function(actual, predicted) {
  n <- length(actual)
  
  mae <- mean(abs(actual - predicted))
  mse <- mean((actual - predicted)^2)
  rmse <- sqrt(mse)
  
  mape <- if (any(actual == 0)) {
    NA_real_
  } else {
    mean(abs((actual - predicted) / actual)) * 100
  }
  
  smape <- mean(2 * abs(actual - predicted) / (abs(actual) + abs(predicted) + 1e-8)) * 100
  
  # R-squared
  ss_res <- sum((actual - predicted)^2)
  ss_tot <- sum((actual - mean(actual))^2)
  r_squared <- if (ss_tot == 0) NA_real_ else 1 - (ss_res / ss_tot)
  
  list(
    n = n,
    mae = mae,
    mse = mse,
    rmse = rmse,
    mape = mape,
    smape = smape,
    r_squared = r_squared
  )
}

# ==============================================================================
# MODEL EXPORT FUNCTIONS
# ==============================================================================

# ------------------------------------------------------------------------------
# export_h2o_mojo: Export model as MOJO for production
# ------------------------------------------------------------------------------
export_h2o_mojo <- function(model, dataset_name, target_col) {
  mojo_dir <- file.path("experiments", "h2o_automl", "models", "mojo")
  dir.create(mojo_dir, recursive = TRUE, showWarnings = FALSE)
  
  mojo_path <- file.path(
    mojo_dir,
    sprintf("%s_%s.zip", dataset_name, target_col)
  )
  
  tryCatch({
    h2o.save_mojo(model, path = mojo_dir, force = TRUE)
    message(sprintf("✓ MOJO exported: %s", mojo_path))
    invisible(mojo_path)
  }, error = function(e) {
    warning(sprintf("MOJO export failed: %s", e$message))
    invisible(NULL)
  })
}

# ==============================================================================
# METRICS SAVING
# ==============================================================================

# ------------------------------------------------------------------------------
# save_h2o_metrics: Save evaluation metrics to CSV
# ------------------------------------------------------------------------------
save_h2o_metrics <- function(metrics_list, dataset_name, target_col, model_id = "h2o_automl") {
  metrics_dir <- file.path("results", "metrics")
  dir.create(metrics_dir, recursive = TRUE, showWarnings = FALSE)
  
  metrics_df <- data.frame(
    dataset = dataset_name,
    target = target_col,
    model = model_id,
    n = metrics_list$n,
    mae = metrics_list$mae,
    mse = metrics_list$mse,
    rmse = metrics_list$rmse,
    mape = metrics_list$mape,
    smape = metrics_list$smape,
    r_squared = metrics_list$r_squared,
    timestamp = Sys.time()
  )
  
  metrics_path <- file.path(
    metrics_dir,
    sprintf("metrics_h2o_%s_%s.csv", dataset_name, target_col)
  )
  
  write.csv(metrics_df, metrics_path, row.names = FALSE)
  message(sprintf("✓ Metrics saved: %s", metrics_path))
  
  invisible(metrics_path)
}

# ==============================================================================
# COMPARISON FUNCTIONS
# ==============================================================================

# ------------------------------------------------------------------------------
# compare_model_results: Compare H2O vs LSTM results
# ------------------------------------------------------------------------------
compare_model_results <- function(dataset_name, target_col) {
  pred_dir <- file.path("results", "predictions")
  
  # Load predictions
  lstm_path <- file.path(pred_dir, sprintf("predictions_lstm_%s_%s.csv", dataset_name, target_col))
  h2o_path <- file.path(pred_dir, sprintf("predictions_h2o_%s_%s.csv", dataset_name, target_col))
  
  comparison <- list()
  
  if (file.exists(lstm_path)) {
    lstm_df <- read.csv(lstm_path)
    lstm_metrics <- evaluate_h2o_predictions(lstm_df$actual, lstm_df$predicted)
    comparison$lstm <- lstm_metrics
  }
  
  if (file.exists(h2o_path)) {
    h2o_df <- read.csv(h2o_path)
    h2o_metrics <- evaluate_h2o_predictions(h2o_df$actual, h2o_df$predicted)
    comparison$h2o <- h2o_metrics
  }
  
  if (length(comparison) == 0) {
    message("No prediction files found for comparison.")
    return(NULL)
  }
  
  # Create comparison table
  comparison_df <- data.frame(
    model = names(comparison),
    rmse = sapply(comparison, function(x) x$rmse),
    mae = sapply(comparison, function(x) x$mae),
    smape = sapply(comparison, function(x) x$smape)
  )
  
  comparison_df
}

# ==============================================================================
# SHUTDOWN FUNCTION
# ==============================================================================

# ------------------------------------------------------------------------------
# shutdown_h2o_cluster: Properly shutdown H2O
# ------------------------------------------------------------------------------
shutdown_h2o_cluster <- function(prompt = FALSE) {
  tryCatch({
    h2o.shutdown(prompt = prompt)
    message("✓ H2O cluster shutdown complete")
  }, error = function(e) {
    message("H2O cluster was not running or already shut down")
  })
}

# ==============================================================================
# FULL EVALUATION PIPELINE
# ==============================================================================

# ------------------------------------------------------------------------------
# evaluate_h2o_model: Full evaluation for a trained model
# ------------------------------------------------------------------------------
evaluate_h2o_model <- function(result, dataset_name, target_col) {
  if (is.null(result)) {
    return(NULL)
  }
  
  # Calculate metrics
  metrics <- evaluate_h2o_predictions(
    result$predictions$actual,
    result$predictions$predicted
  )
  
  # Save metrics
  save_h2o_metrics(
    metrics_list = metrics,
    dataset_name = dataset_name,
    target_col = target_col,
    model_id = result$metadata$best_model_id
  )
  
  # Print summary
  message(sprintf("\n### Evaluation Summary: %s/%s ###", dataset_name, target_col))
  message(sprintf("  Best model: %s", result$metadata$best_model_id))
  message(sprintf("  Test samples: %d", metrics$n))
  message(sprintf("  RMSE: %.4f", metrics$rmse))
  message(sprintf("  MAE: %.4f", metrics$mae))
  message(sprintf("  SMAPE: %.2f%%", metrics$smape))
  
  metrics
}

# ==============================================================================
# EXAMPLE USAGE
# ==============================================================================
# 
# # Load data and train
# source("R/utils_io.R")
# source("R/preprocess.R")
# klaim <- read_raw_csv("data/raw/Klaim ARIMA.csv")
# klaim_clean <- clean_dataset(klaim, "klaim")
# 
# source("experiments/h2o_automl/scripts/01_train_h2o.R")
# result <- run_h2o_pipeline(klaim_clean, "klaim", "JHT")
# 
# # Evaluate
# metrics <- evaluate_h2o_model(result, "klaim", "JHT")
# 
# # Compare with LSTM
# comparison <- compare_model_results("klaim", "JHT")
# print(comparison)
# 
# # Shutdown when done
# shutdown_h2o_cluster()
