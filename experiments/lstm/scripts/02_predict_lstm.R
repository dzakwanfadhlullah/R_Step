# ==============================================================================
# 02_predict_lstm.R - LSTM Prediction and Model Saving Script
# ==============================================================================
# This script handles:
#   - Loading trained models
#   - Making predictions on test data
#   - Inverse scaling predictions to original scale
#   - Saving models, predictions, and training logs
#
# Usage:
#   source("experiments/lstm/scripts/02_predict_lstm.R")
# ==============================================================================

# Load dependencies
library(keras3)
library(tensorflow)

# Set working directory to project root
if (!file.exists("R/config.R")) {
  setwd("project-timeseries-forecast")
}

# Source project files
source("R/config.R")
source("R/utils_io.R")
source("R/preprocess.R")
source("R/split.R")

# ==============================================================================
# MODEL SAVING FUNCTIONS
# ==============================================================================

# ------------------------------------------------------------------------------
# save_lstm_model: Save trained model to H5 file
# ------------------------------------------------------------------------------
save_lstm_model <- function(model, dataset_name, target_col) {
  # Create models directory
  models_dir <- file.path("experiments", "lstm", "models")
  dir.create(models_dir, recursive = TRUE, showWarnings = FALSE)
  
  # Generate filename
  model_path <- file.path(
    models_dir, 
    sprintf("lstm_%s_%s.keras", dataset_name, target_col)
  )
  
  # Save model
  save_model(model, model_path)
  message(sprintf("✓ Model saved: %s", model_path))
  
  invisible(model_path)
}

# ------------------------------------------------------------------------------
# load_lstm_model: Load saved model from file
# ------------------------------------------------------------------------------
load_lstm_model <- function(dataset_name, target_col) {
  models_dir <- file.path("experiments", "lstm", "models")
  model_path <- file.path(
    models_dir, 
    sprintf("lstm_%s_%s.keras", dataset_name, target_col)
  )
  
  if (!file.exists(model_path)) {
    stop(sprintf("Model not found: %s", model_path), call. = FALSE)
  }
  
  model <- load_model(model_path)
  message(sprintf("✓ Model loaded: %s", model_path))
  
  model
}

# ==============================================================================
# TRAINING LOG FUNCTIONS
# ==============================================================================

# ------------------------------------------------------------------------------
# save_training_log: Save training history to CSV
# ------------------------------------------------------------------------------
save_training_log <- function(history, dataset_name, target_col) {
  # Create logs directory
  logs_dir <- file.path("experiments", "lstm", "logs")
  dir.create(logs_dir, recursive = TRUE, showWarnings = FALSE)
  
  # Extract metrics from history
  metrics <- history$metrics
  n_epochs <- length(metrics$loss)
  
  log_df <- data.frame(
    epoch = 1:n_epochs,
    loss = metrics$loss,
    mae = if (!is.null(metrics$mae)) metrics$mae else NA_real_,
    val_loss = if (!is.null(metrics$val_loss)) metrics$val_loss else NA_real_,
    val_mae = if (!is.null(metrics$val_mae)) metrics$val_mae else NA_real_
  )
  
  # Generate filename
  log_path <- file.path(
    logs_dir, 
    sprintf("training_log_%s_%s.csv", dataset_name, target_col)
  )
  
  # Save log
  write.csv(log_df, log_path, row.names = FALSE)
  message(sprintf("✓ Training log saved: %s (%d epochs)", log_path, n_epochs))
  
  invisible(log_path)
}

# ==============================================================================
# PREDICTION FUNCTIONS
# ==============================================================================

# ------------------------------------------------------------------------------
# predict_and_inverse: Make predictions and inverse scale to original values
# ------------------------------------------------------------------------------
predict_and_inverse <- function(model, X_data, scaler_params) {
  # Make predictions (normalized scale)
  pred_normalized <- predict(model, X_data)
  
  # Inverse transform to original scale
  # For minmax: original = normalized * (max - min) + min
  if (scaler_params$method == "minmax") {
    range_val <- scaler_params$max - scaler_params$min
    pred_original <- as.vector(pred_normalized) * range_val + scaler_params$min
  } else if (scaler_params$method == "zscore") {
    pred_original <- as.vector(pred_normalized) * scaler_params$sd + scaler_params$mean
  } else {
    stop("Unknown scaler method", call. = FALSE)
  }
  
  pred_original
}

# ------------------------------------------------------------------------------
# save_predictions: Save predictions to standard format CSV
# ------------------------------------------------------------------------------
save_predictions <- function(predictions_df, dataset_name, target_col, model_type = "lstm") {
  # Create predictions directory
  pred_dir <- file.path("results", "predictions")
  dir.create(pred_dir, recursive = TRUE, showWarnings = FALSE)
  
  # Add metadata columns
  predictions_df$dataset <- dataset_name
  predictions_df$target <- target_col
  predictions_df$model <- model_type
  predictions_df$timestamp <- Sys.time()
  
  # Reorder columns
  predictions_df <- predictions_df[, c(
    "dataset", "target", "model", "year", "actual", "predicted", "timestamp"
  )]
  
  # Generate filename
  pred_path <- file.path(
    pred_dir, 
    sprintf("predictions_%s_%s_%s.csv", model_type, dataset_name, target_col)
  )
  
  # Save predictions
  write.csv(predictions_df, pred_path, row.names = FALSE)
  message(sprintf("✓ Predictions saved: %s", pred_path))
  
  invisible(pred_path)
}

# ==============================================================================
# FULL TRAINING AND SAVING PIPELINE
# ==============================================================================

# ------------------------------------------------------------------------------
# run_lstm_pipeline: Complete pipeline for one target
# ------------------------------------------------------------------------------
run_lstm_pipeline <- function(cleaned_data,
                               dataset_name,
                               target_col,
                               lookback = LSTM_LOOKBACK,
                               holdout_years = HOLDOUT_YEARS,
                               epochs = 100,
                               batch_size = 1,
                               verbose = 1) {
  
  message(sprintf("\n" %+% strrep("=", 60)))
  message(sprintf(" LSTM Pipeline: %s / %s", dataset_name, target_col))
  message(strrep("=", 60))
  
  # Source training functions
  source("experiments/lstm/scripts/01_train_lstm.R")
  
  # Train model
  result <- train_lstm_for_target(
    cleaned_data = cleaned_data,
    target_col = target_col,
    lookback = lookback,
    holdout_years = holdout_years,
    epochs = epochs,
    batch_size = batch_size,
    verbose = verbose
  )
  
  if (is.null(result)) {
    warning(sprintf("Training failed for %s/%s", dataset_name, target_col))
    return(NULL)
  }
  
  # Save model
  save_lstm_model(result$model, dataset_name, target_col)
  
  # Save training log
  save_training_log(result$history, dataset_name, target_col)
  
  # Save predictions (test set)
  save_predictions(result$predictions$test, dataset_name, target_col, "lstm")
  
  message(sprintf("\n✅ Pipeline complete for %s/%s", dataset_name, target_col))
  
  result
}

# ------------------------------------------------------------------------------
# run_all_lstm_pipelines: Run pipeline for all targets in a dataset
# ------------------------------------------------------------------------------
run_all_lstm_pipelines <- function(cleaned_data,
                                    dataset_name,
                                    targets = NULL,
                                    skip_critical = FALSE,
                                    ...) {
  
  if (is.null(targets)) {
    targets <- names(cleaned_data$by_target)
  }
  
  results <- list()
  
  for (target in targets) {
    # Check if target has critical warning (< 6 data points)
    n_points <- nrow(cleaned_data$by_target[[target]])
    
    if (n_points < 6 && skip_critical) {
      message(sprintf(
        "\n⚠️ Skipping %s: only %d data points (critical threshold)",
        target, n_points
      ))
      next
    }
    
    tryCatch({
      results[[target]] <- run_lstm_pipeline(
        cleaned_data = cleaned_data,
        dataset_name = dataset_name,
        target_col = target,
        ...
      )
    }, error = function(e) {
      warning(sprintf("Error training %s/%s: %s", dataset_name, target, e$message))
      results[[target]] <<- NULL
    })
  }
  
  results
}

# ==============================================================================
# EVALUATION METRICS
# ==============================================================================

# ------------------------------------------------------------------------------
# calculate_metrics: Calculate forecast accuracy metrics
# ------------------------------------------------------------------------------
calculate_metrics <- function(actual, predicted) {
  n <- length(actual)
  
  # Mean Absolute Error
  mae <- mean(abs(actual - predicted))
  
  # Mean Squared Error
  mse <- mean((actual - predicted)^2)
  
  # Root Mean Squared Error
  rmse <- sqrt(mse)
  
  # Mean Absolute Percentage Error (handle zeros)
  mape <- if (any(actual == 0)) {
    NA_real_
  } else {
    mean(abs((actual - predicted) / actual)) * 100
  }
  
  # Symmetric MAPE (more robust)
  smape <- mean(2 * abs(actual - predicted) / (abs(actual) + abs(predicted) + 1e-8)) * 100
  
  list(
    n = n,
    mae = mae,
    mse = mse,
    rmse = rmse,
    mape = mape,
    smape = smape
  )
}

# ------------------------------------------------------------------------------
# save_metrics: Save metrics to CSV
# ------------------------------------------------------------------------------
save_metrics <- function(metrics_list, dataset_name, target_col, model_type = "lstm") {
  # Create metrics directory
  metrics_dir <- file.path("results", "metrics")
  dir.create(metrics_dir, recursive = TRUE, showWarnings = FALSE)
  
  # Convert to data frame
  metrics_df <- data.frame(
    dataset = dataset_name,
    target = target_col,
    model = model_type,
    n = metrics_list$n,
    mae = metrics_list$mae,
    mse = metrics_list$mse,
    rmse = metrics_list$rmse,
    mape = metrics_list$mape,
    smape = metrics_list$smape,
    timestamp = Sys.time()
  )
  
  # Generate filename
  metrics_path <- file.path(
    metrics_dir, 
    sprintf("metrics_%s_%s_%s.csv", model_type, dataset_name, target_col)
  )
  
  # Save metrics
  write.csv(metrics_df, metrics_path, row.names = FALSE)
  message(sprintf("✓ Metrics saved: %s", metrics_path))
  
  invisible(metrics_path)
}

# ==============================================================================
# STRING CONCATENATION HELPER
# ==============================================================================
`%+%` <- function(a, b) paste0(a, b)

# ==============================================================================
# EXAMPLE USAGE (uncomment to run)
# ==============================================================================
# 
# # Load and clean data
# klaim <- read_raw_csv("data/raw/Klaim ARIMA.csv")
# klaim_clean <- clean_dataset(klaim, "klaim")
# 
# # Run pipeline for single target
# result <- run_lstm_pipeline(
#   cleaned_data = klaim_clean,
#   dataset_name = "klaim",
#   target_col = "JHT",
#   epochs = 50,
#   verbose = 0
# )
# 
# # Calculate and save metrics
# metrics <- calculate_metrics(
#   result$predictions$test$actual,
#   result$predictions$test$predicted
# )
# save_metrics(metrics, "klaim", "JHT")
