# ==============================================================================
# metrics.R - Standard Evaluation Metrics for Forecasting
# ==============================================================================
# This file contains all evaluation metric functions used for comparing
# model performance across LSTM and H2O AutoML experiments.
#
# Usage:
#   source("R/metrics.R")
# ==============================================================================

# ==============================================================================
# INDIVIDUAL METRIC FUNCTIONS
# ==============================================================================

# ------------------------------------------------------------------------------
# metric_mae: Mean Absolute Error
# ------------------------------------------------------------------------------
# MAE = mean(|actual - predicted|)
# Lower is better. Same unit as the target variable.
#
# Args:
#   actual    - Vector of actual values
#   predicted - Vector of predicted values
#
# Returns:
#   Numeric MAE value, or NA if input is invalid
# ------------------------------------------------------------------------------
metric_mae <- function(actual, predicted) {
  if (length(actual) != length(predicted)) {
    warning("Length mismatch between actual and predicted")
    return(NA_real_)
  }
  
  if (all(is.na(actual)) || all(is.na(predicted))) {
    return(NA_real_)
  }
  
  # Remove NA pairs
  valid_idx <- !is.na(actual) & !is.na(predicted)
  if (sum(valid_idx) == 0) return(NA_real_)
  
  mean(abs(actual[valid_idx] - predicted[valid_idx]))
}

# ------------------------------------------------------------------------------
# metric_mse: Mean Squared Error
# ------------------------------------------------------------------------------
# MSE = mean((actual - predicted)^2)
# Lower is better. Squared unit of target variable.
# ------------------------------------------------------------------------------
metric_mse <- function(actual, predicted) {
  if (length(actual) != length(predicted)) {
    warning("Length mismatch between actual and predicted")
    return(NA_real_)
  }
  
  if (all(is.na(actual)) || all(is.na(predicted))) {
    return(NA_real_)
  }
  
  valid_idx <- !is.na(actual) & !is.na(predicted)
  if (sum(valid_idx) == 0) return(NA_real_)
  
  mean((actual[valid_idx] - predicted[valid_idx])^2)
}

# ------------------------------------------------------------------------------
# metric_rmse: Root Mean Squared Error
# ------------------------------------------------------------------------------
# RMSE = sqrt(MSE)
# Lower is better. Same unit as the target variable.
# More sensitive to large errors than MAE.
# ------------------------------------------------------------------------------
metric_rmse <- function(actual, predicted) {
  mse <- metric_mse(actual, predicted)
  if (is.na(mse)) return(NA_real_)
  sqrt(mse)
}

# ------------------------------------------------------------------------------
# metric_mape: Mean Absolute Percentage Error
# ------------------------------------------------------------------------------
# MAPE = mean(|actual - predicted| / |actual|) * 100
# CAUTION: Undefined when actual = 0. Asymmetric error.
#
# Args:
#   actual    - Vector of actual values
#   predicted - Vector of predicted values
#   na_zero   - If TRUE, exclude zero values from calculation (default TRUE)
#
# Returns:
#   Numeric MAPE value (in percentage), or NA if invalid
# ------------------------------------------------------------------------------
metric_mape <- function(actual, predicted, na_zero = TRUE) {
  if (length(actual) != length(predicted)) {
    warning("Length mismatch between actual and predicted")
    return(NA_real_)
  }
  
  if (all(is.na(actual)) || all(is.na(predicted))) {
    return(NA_real_)
  }
  
  valid_idx <- !is.na(actual) & !is.na(predicted)
  
  if (na_zero) {
    # Exclude zero actual values to avoid division by zero
    valid_idx <- valid_idx & (actual != 0)
  }
  
  if (sum(valid_idx) == 0) {
    warning("No valid observations for MAPE calculation (all actuals are 0 or NA)")
    return(NA_real_)
  }
  
  mean(abs((actual[valid_idx] - predicted[valid_idx]) / actual[valid_idx])) * 100
}

# ------------------------------------------------------------------------------
# metric_smape: Symmetric Mean Absolute Percentage Error
# ------------------------------------------------------------------------------
# sMAPE = mean(2 * |actual - pred| / (|actual| + |pred|)) * 100
# Symmetric version of MAPE, bounded between 0% and 200%.
# More robust than MAPE for near-zero values.
#
# Args:
#   actual    - Vector of actual values
#   predicted - Vector of predicted values
#   epsilon   - Small value to avoid division by zero (default 1e-8)
#
# Returns:
#   Numeric sMAPE value (in percentage), or NA if invalid
# ------------------------------------------------------------------------------
metric_smape <- function(actual, predicted, epsilon = 1e-8) {
  if (length(actual) != length(predicted)) {
    warning("Length mismatch between actual and predicted")
    return(NA_real_)
  }
  
  if (all(is.na(actual)) || all(is.na(predicted))) {
    return(NA_real_)
  }
  
  valid_idx <- !is.na(actual) & !is.na(predicted)
  if (sum(valid_idx) == 0) return(NA_real_)
  
  a <- actual[valid_idx]
  p <- predicted[valid_idx]
  
  # Calculate sMAPE with epsilon to avoid division by zero
  denominator <- abs(a) + abs(p) + epsilon
  smape_values <- 2 * abs(a - p) / denominator
  
  mean(smape_values) * 100
}

# ------------------------------------------------------------------------------
# metric_r_squared: Coefficient of Determination (R²)
# ------------------------------------------------------------------------------
# R² = 1 - (SS_res / SS_tot)
# Range: (-∞, 1]. 1 = perfect fit, 0 = baseline (mean), negative = worse than mean
# ------------------------------------------------------------------------------
metric_r_squared <- function(actual, predicted) {
  if (length(actual) != length(predicted)) {
    warning("Length mismatch between actual and predicted")
    return(NA_real_)
  }
  
  valid_idx <- !is.na(actual) & !is.na(predicted)
  if (sum(valid_idx) < 2) return(NA_real_)
  
  a <- actual[valid_idx]
  p <- predicted[valid_idx]
  
  ss_res <- sum((a - p)^2)
  ss_tot <- sum((a - mean(a))^2)
  
  if (ss_tot == 0) return(NA_real_)
  
  1 - (ss_res / ss_tot)
}

# ==============================================================================
# COMBINED METRICS FUNCTION
# ==============================================================================

# ------------------------------------------------------------------------------
# calculate_all_metrics: Calculate all standard metrics at once
# ------------------------------------------------------------------------------
# Args:
#   actual    - Vector of actual values
#   predicted - Vector of predicted values
#
# Returns:
#   Named list with all metrics
# ------------------------------------------------------------------------------
calculate_all_metrics <- function(actual, predicted) {
  n_valid <- sum(!is.na(actual) & !is.na(predicted))
  
  list(
    n = n_valid,
    mae = metric_mae(actual, predicted),
    mse = metric_mse(actual, predicted),
    rmse = metric_rmse(actual, predicted),
    mape = metric_mape(actual, predicted),
    smape = metric_smape(actual, predicted),
    r_squared = metric_r_squared(actual, predicted)
  )
}

# ==============================================================================
# EVALUATION FUNCTIONS
# ==============================================================================

# ------------------------------------------------------------------------------
# evaluate_predictions: Evaluate predictions from a data frame
# ------------------------------------------------------------------------------
# Expects a data frame with columns: actual, predicted
# And optionally: dataset, target, model, fold
#
# Args:
#   preds_df   - Data frame with predictions
#   group_cols - Columns to group by (default: c("dataset", "target", "model"))
#
# Returns:
#   Data frame with metrics for each group
# ------------------------------------------------------------------------------
evaluate_predictions <- function(preds_df, 
                                  group_cols = c("dataset", "target", "model")) {
  
  # Check required columns
  required <- c("actual", "predicted")
  missing_cols <- setdiff(required, names(preds_df))
  if (length(missing_cols) > 0) {
    stop(sprintf("Missing required columns: %s", paste(missing_cols, collapse = ", ")))
  }
  
  # Filter to existing group columns
  group_cols <- intersect(group_cols, names(preds_df))
  
  if (length(group_cols) == 0) {
    # No grouping, calculate overall metrics
    metrics <- calculate_all_metrics(preds_df$actual, preds_df$predicted)
    result <- as.data.frame(metrics)
    return(result)
  }
  
  # Create group key
  preds_df$group_key <- apply(preds_df[, group_cols, drop = FALSE], 1, paste, collapse = "|")
  
  # Calculate metrics per group
  groups <- unique(preds_df$group_key)
  results <- list()
  
  for (g in groups) {
    mask <- preds_df$group_key == g
    subset_df <- preds_df[mask, ]
    
    metrics <- calculate_all_metrics(subset_df$actual, subset_df$predicted)
    
    # Parse group key back to columns
    key_parts <- strsplit(g, "\\|")[[1]]
    
    # Build row as list first, then convert to data frame
    row_list <- list()
    for (i in seq_along(group_cols)) {
      row_list[[group_cols[i]]] <- key_parts[i]
    }
    
    row_list$n <- metrics$n
    row_list$mae <- metrics$mae
    row_list$mse <- metrics$mse
    row_list$rmse <- metrics$rmse
    row_list$mape <- metrics$mape
    row_list$smape <- metrics$smape
    row_list$r_squared <- metrics$r_squared
    
    results[[length(results) + 1]] <- as.data.frame(row_list, stringsAsFactors = FALSE)
  }
  
  # Combine results
  do.call(rbind, results)
}

# ------------------------------------------------------------------------------
# evaluate_rolling_folds: Evaluate metrics across rolling CV folds
# ------------------------------------------------------------------------------
evaluate_rolling_folds <- function(preds_df) {
  if (!"fold" %in% names(preds_df)) {
    stop("preds_df must have 'fold' column for rolling evaluation")
  }
  
  # Calculate per-fold metrics
  fold_metrics <- evaluate_predictions(
    preds_df, 
    group_cols = c("dataset", "target", "model", "fold")
  )
  
  # Calculate average across folds
  group_cols <- c("dataset", "target", "model")
  group_cols <- intersect(group_cols, names(fold_metrics))
  
  if (length(group_cols) == 0) {
    # Single group, average all folds
    avg_metrics <- data.frame(
      n_folds = nrow(fold_metrics),
      avg_mae = mean(fold_metrics$mae, na.rm = TRUE),
      avg_rmse = mean(fold_metrics$rmse, na.rm = TRUE),
      avg_smape = mean(fold_metrics$smape, na.rm = TRUE),
      std_rmse = sd(fold_metrics$rmse, na.rm = TRUE)
    )
    return(avg_metrics)
  }
  
  # Average per group
  fold_metrics$group_key <- apply(fold_metrics[, group_cols, drop = FALSE], 1, paste, collapse = "|")
  
  groups <- unique(fold_metrics$group_key)
  results <- list()
  
  for (g in groups) {
    mask <- fold_metrics$group_key == g
    subset_df <- fold_metrics[mask, ]
    
    key_parts <- strsplit(g, "\\|")[[1]]
    
    row <- data.frame(stringsAsFactors = FALSE)
    for (i in seq_along(group_cols)) {
      row[[group_cols[i]]] <- key_parts[i]
    }
    
    row$n_folds <- nrow(subset_df)
    row$avg_mae <- mean(subset_df$mae, na.rm = TRUE)
    row$avg_rmse <- mean(subset_df$rmse, na.rm = TRUE)
    row$avg_smape <- mean(subset_df$smape, na.rm = TRUE)
    row$std_rmse <- sd(subset_df$rmse, na.rm = TRUE)
    
    results[[length(results) + 1]] <- row
  }
  
  do.call(rbind, results)
}

# ==============================================================================
# AGGREGATION AND RANKING
# ==============================================================================

# ------------------------------------------------------------------------------
# create_metrics_summary: Create summary with rankings
# ------------------------------------------------------------------------------
create_metrics_summary <- function(metrics_df, rank_by = "rmse") {
  # Add rank column per target
  if (!"target" %in% names(metrics_df)) {
    metrics_df$rank <- rank(metrics_df[[rank_by]], na.last = "keep")
    return(metrics_df)
  }
  
  # Rank within each target
  targets <- unique(metrics_df$target)
  
  for (tgt in targets) {
    mask <- metrics_df$target == tgt
    metrics_df$rank[mask] <- rank(metrics_df[[rank_by]][mask], na.last = "keep")
  }
  
  # Sort by target and rank
  metrics_df <- metrics_df[order(metrics_df$target, metrics_df$rank), ]
  
  metrics_df
}

# ------------------------------------------------------------------------------
# compare_models: Create comparison pivot table
# ------------------------------------------------------------------------------
compare_models <- function(metrics_df, metric = "rmse") {
  if (!all(c("target", "model") %in% names(metrics_df))) {
    stop("metrics_df must have 'target' and 'model' columns")
  }
  
  if (!metric %in% names(metrics_df)) {
    stop(sprintf("Metric '%s' not found in metrics_df", metric))
  }
  
  # Create pivot: rows = target, columns = model
  targets <- unique(metrics_df$target)
  models <- unique(metrics_df$model)
  
  pivot <- data.frame(target = targets)
  
  for (model in models) {
    col_values <- numeric(length(targets))
    for (i in seq_along(targets)) {
      mask <- metrics_df$target == targets[i] & metrics_df$model == model
      if (any(mask)) {
        col_values[i] <- metrics_df[[metric]][mask][1]
      } else {
        col_values[i] <- NA_real_
      }
    }
    pivot[[model]] <- col_values
  }
  
  # Add best model column
  model_cols <- setdiff(names(pivot), "target")
  pivot$best_model <- apply(pivot[, model_cols, drop = FALSE], 1, function(row) {
    if (all(is.na(row))) return(NA_character_)
    model_cols[which.min(row)]
  })
  
  pivot
}

# ------------------------------------------------------------------------------
# save_metrics_summary: Export summary to CSV
# ------------------------------------------------------------------------------
save_metrics_summary <- function(metrics_df, filename = "metrics_summary.csv") {
  metrics_dir <- file.path("results", "metrics")
  dir.create(metrics_dir, recursive = TRUE, showWarnings = FALSE)
  
  filepath <- file.path(metrics_dir, filename)
  write.csv(metrics_df, filepath, row.names = FALSE)
  message(sprintf("✓ Metrics summary saved: %s", filepath))
  
  invisible(filepath)
}

# ==============================================================================
# TEST FUNCTIONS
# ==============================================================================

test_metrics <- function() {
  # Test data
  actual <- c(100, 120, 140, 160, 180)
  predicted <- c(105, 118, 145, 155, 185)
  
  # Test individual metrics
  mae <- metric_mae(actual, predicted)
  stopifnot(!is.na(mae))
  stopifnot(mae > 0)
  
  rmse <- metric_rmse(actual, predicted)
  stopifnot(!is.na(rmse))
  stopifnot(rmse >= mae)  # RMSE >= MAE always
  
  smape <- metric_smape(actual, predicted)
  stopifnot(!is.na(smape))
  stopifnot(smape >= 0 && smape <= 200)
  
  r2 <- metric_r_squared(actual, predicted)
  stopifnot(!is.na(r2))
  
  # Test combined
  all_metrics <- calculate_all_metrics(actual, predicted)
  stopifnot(length(all_metrics) == 7)
  
  message("test_metrics: PASSED")
  invisible(TRUE)
}

test_evaluate_predictions <- function() {
  preds_df <- data.frame(
    dataset = rep("klaim", 4),
    target = rep("JHT", 4),
    model = c("lstm", "lstm", "h2o", "h2o"),
    actual = c(100, 120, 100, 120),
    predicted = c(105, 115, 102, 118)
  )
  
  result <- evaluate_predictions(preds_df)
  
  stopifnot(nrow(result) == 2)  # Two models
  stopifnot(all(c("mae", "rmse") %in% names(result)))
  
  message("test_evaluate_predictions: PASSED")
  invisible(TRUE)
}
