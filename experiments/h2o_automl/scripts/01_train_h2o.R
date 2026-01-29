# ==============================================================================
# 01_train_h2o.R - H2O AutoML Training Script
# ==============================================================================
# This script handles:
#   - Converting time-series data to supervised learning format
#   - Training H2O AutoML models
#   - Saving leaderboard and best model
#
# Prerequisites:
#   - Run 00_setup_h2o.R first
#   - Data preprocessed with lag/rolling features
#
# Usage:
#   source("experiments/h2o_automl/scripts/01_train_h2o.R")
# ==============================================================================

# Set working directory to project root
if (!file.exists("R/config.R")) {
  setwd("project-timeseries-forecast")
}

# Load dependencies
library(h2o)

# Source project files
source("R/config.R")
source("R/utils_io.R")
source("R/preprocess.R")
source("R/split.R")
source("R/features_timeseries.R")

# Ensure H2O is running (handled by main.R)
# source("experiments/h2o_automl/scripts/00_setup_h2o.R")

# ==============================================================================
# DATA PREPARATION FOR H2O
# ==============================================================================

# ------------------------------------------------------------------------------
# prepare_h2o_data: Convert time-series to supervised learning format
# ------------------------------------------------------------------------------
# Creates a tabular dataset with lag features suitable for H2O.
# IMPORTANT: Only uses past values as features (no future leakage!)
#
# Args:
#   target_df   - Data frame with Tahun and target column
#   target_col  - Name of target column
#   lags        - Lag values to create (default: c(1, 2))
#   windows     - Rolling window sizes (default: c(2))
#   holdout_years - Years for test set
#
# Returns:
#   List with:
#     - train_df: Training data frame
#     - test_df: Test data frame
#     - feature_names: Names of predictor columns
#     - target_name: Name of target column (always "target")
# ------------------------------------------------------------------------------
prepare_h2o_data <- function(target_df,
                              target_col,
                              lags = c(1, 2),
                              windows = c(2),
                              holdout_years = HOLDOUT_YEARS) {
  
  message(sprintf("\n=== Preparing H2O data for %s ===", target_col))
  
  # Step 1: Build features using existing function
  n_points <- nrow(target_df)
  cfg <- recommend_feature_config(n_points)
  
  # Use recommended config if not specified
  if (missing(lags)) lags <- cfg$lags
  if (missing(windows)) windows <- cfg$windows
  
  feature_result <- build_features_for_target(
    df = target_df,
    target_col = target_col,
    lags = lags,
    windows = windows
  )
  
  # Get complete rows only (remove NAs from lag creation)
  features_df <- get_complete_rows(feature_result)
  feature_names <- feature_result$feature_names
  
  message(sprintf(
    "  Created %d features: %s",
    length(feature_names),
    paste(feature_names, collapse = ", ")
  ))
  
  # Step 2: Split into train/test based on year
  train_mask <- features_df$Tahun < min(holdout_years)
  test_mask <- features_df$Tahun %in% holdout_years
  
  train_df <- features_df[train_mask, ]
  test_df <- features_df[test_mask, ]
  
  message(sprintf(
    "  Train: %d rows (years %s)",
    nrow(train_df),
    paste(train_df$Tahun, collapse = ",")
  ))
  message(sprintf(
    "  Test: %d rows (years %s)",
    nrow(test_df),
    paste(test_df$Tahun, collapse = ",")
  ))
  
  # Validation
  if (nrow(train_df) < 3) {
    warning("Very few training samples! Model may be unreliable.")
  }
  
  list(
    train_df = train_df,
    test_df = test_df,
    feature_names = feature_names,
    target_name = "target",  # build_features_for_target uses "target"
    years_train = train_df$Tahun,
    years_test = test_df$Tahun
  )
}

# ==============================================================================
# H2O AUTOML TRAINING
# ==============================================================================

# ------------------------------------------------------------------------------
# train_h2o_automl: Train H2O AutoML for a single target
# ------------------------------------------------------------------------------
train_h2o_automl <- function(cleaned_data,
                              dataset_name,
                              target_col,
                              max_runtime_secs = H2O_MAX_RUNTIME_SECS,
                              nfolds = 3,
                              seed = RANDOM_SEED,
                              holdout_years = HOLDOUT_YEARS) {
  
  message(sprintf("\n" %+% strrep("=", 60)))
  message(sprintf(" H2O AutoML: %s / %s", dataset_name, target_col))
  message(strrep("=", 60))
  
  # Get target data
  target_df <- cleaned_data$by_target[[target_col]]
  
  if (is.null(target_df) || nrow(target_df) == 0) {
    warning(sprintf("No data available for target: %s", target_col))
    return(NULL)
  }
  
  # Prepare data
  data <- prepare_h2o_data(
    target_df = target_df,
    target_col = target_col,
    holdout_years = holdout_years
  )
  
  # Check minimum samples
  if (nrow(data$train_df) < 3) {
    warning(sprintf(
      "Target %s: insufficient training data (%d rows). Skipping.",
      target_col, nrow(data$train_df)
    ))
    return(NULL)
  }
  
  # Upload to H2O
  message("\n  Uploading data to H2O...")
  train_h2o <- as.h2o(data$train_df)
  test_h2o <- as.h2o(data$test_df)
  
  # Define predictors and response
  x <- data$feature_names
  y <- data$target_name
  
  message(sprintf("  Predictors: %s", paste(x, collapse = ", ")))
  message(sprintf("  Response: %s", y))
  
  # Run AutoML
  message(sprintf(
    "\n  Running AutoML (max %d seconds, %d-fold CV)...",
    max_runtime_secs, nfolds
  ))
  
  # Dynamic nfolds: H2O needs at least 2*nfolds rows for CV
  # If we have < 10 rows, CV is risky or impossible
  actual_nfolds <- if (nrow(data$train_df) < 10) 0 else nfolds
  if (actual_nfolds != nfolds) {
    message(sprintf("  ℹ Dataset too small (%d rows), disabling CV (nfolds=0)", nrow(data$train_df)))
  }
  
  # Get max models from config if available
  max_models <- if (exists("H2O_MAX_MODELS")) H2O_MAX_MODELS else NULL
  
  aml <- h2o.automl(
    x = x,
    y = y,
    training_frame = train_h2o,
    max_runtime_secs = max_runtime_secs,
    max_models = max_models,
    seed = seed,
    nfolds = actual_nfolds,
    exclude_algos = c("DeepLearning", "StackedEnsemble"), # Skip heavy models
    keep_cross_validation_predictions = (actual_nfolds > 0),
    sort_metric = "RMSE",
    stopping_metric = "RMSE",
    stopping_tolerance = 0.001,
    stopping_rounds = 3
  )
  
  # Get leaderboard
  lb <- h2o.get_leaderboard(aml, extra_columns = "ALL")
  lb_df <- as.data.frame(lb)
  
  message(sprintf("\n  AutoML complete. Trained %d models.", nrow(lb_df)))
  
  # Best model
  best_model <- aml@leader
  
  if (is.null(best_model)) {
    warning(sprintf("AutoML found no valid models for %s. Leaderboard is empty.", target_col))
    return(NULL)
  }
  
  best_model_id <- as.character(best_model@model_id)
  message(sprintf("  Best model: %s", best_model_id))
  
  # Make predictions on test set
  message("  Making predictions on test set...")
  predictions <- h2o.predict(best_model, test_h2o)
  preds_df <- as.data.frame(predictions)
  
  # Build results data frame
  results_df <- data.frame(
    year = data$years_test,
    actual = data$test_df$target,
    predicted = as.numeric(preds_df[[1]])
  )
  
  message("  Predictions complete.")
  
  list(
    automl = aml,
    best_model = best_model,
    leaderboard = lb_df,
    predictions = results_df,
    train_h2o = train_h2o,
    test_h2o = test_h2o,
    feature_names = data$feature_names,
    metadata = list(
      dataset = dataset_name,
      target = target_col,
      n_train = nrow(data$train_df),
      n_test = nrow(data$test_df),
      n_models = nrow(lb_df),
      best_model_id = best_model_id
    )
  )
}

# ==============================================================================
# SAVE FUNCTIONS
# ==============================================================================

# ------------------------------------------------------------------------------
# save_h2o_leaderboard: Save leaderboard to CSV
# ------------------------------------------------------------------------------
save_h2o_leaderboard <- function(leaderboard_df, dataset_name, target_col) {
  outputs_dir <- file.path("experiments", "h2o_automl", "outputs")
  dir.create(outputs_dir, recursive = TRUE, showWarnings = FALSE)
  
  lb_path <- file.path(
    outputs_dir,
    sprintf("leaderboard_%s_%s.csv", dataset_name, target_col)
  )
  
  write.csv(leaderboard_df, lb_path, row.names = FALSE)
  message(sprintf("✓ Leaderboard saved: %s", lb_path))
  
  invisible(lb_path)
}

# ------------------------------------------------------------------------------
# save_h2o_model: Save best model
# ------------------------------------------------------------------------------
save_h2o_model <- function(model, dataset_name, target_col) {
  models_dir <- file.path("experiments", "h2o_automl", "models")
  dir.create(models_dir, recursive = TRUE, showWarnings = FALSE)
  
  # H2O saves with its own naming, so we create a subdirectory
  target_dir <- file.path(models_dir, sprintf("%s_%s", dataset_name, target_col))
  dir.create(target_dir, recursive = TRUE, showWarnings = FALSE)
  
  model_path <- h2o.saveModel(model, path = target_dir, force = TRUE)
  message(sprintf("✓ Model saved: %s", model_path))
  
  invisible(model_path)
}

# ------------------------------------------------------------------------------
# save_h2o_predictions: Save predictions to CSV
# ------------------------------------------------------------------------------
save_h2o_predictions <- function(predictions_df, dataset_name, target_col) {
  pred_dir <- file.path("results", "predictions")
  dir.create(pred_dir, recursive = TRUE, showWarnings = FALSE)
  
  # Add metadata
  predictions_df$dataset <- dataset_name
  predictions_df$target <- target_col
  predictions_df$model <- "h2o_automl"
  predictions_df$timestamp <- Sys.time()
  
  # Reorder columns
  predictions_df <- predictions_df[, c(
    "dataset", "target", "model", "year", "actual", "predicted", "timestamp"
  )]
  
  pred_path <- file.path(
    pred_dir,
    sprintf("predictions_h2o_%s_%s.csv", dataset_name, target_col)
  )
  
  write.csv(predictions_df, pred_path, row.names = FALSE)
  message(sprintf("✓ Predictions saved: %s", pred_path))
  
  invisible(pred_path)
}

# ==============================================================================
# FULL PIPELINE
# ==============================================================================

# ------------------------------------------------------------------------------
# run_h2o_pipeline: Complete H2O AutoML pipeline for one target
# ------------------------------------------------------------------------------
run_h2o_pipeline <- function(cleaned_data,
                              dataset_name,
                              target_col,
                              max_runtime_secs = H2O_MAX_RUNTIME_SECS,
                              ...) {
  
  # Train model
  result <- train_h2o_automl(
    cleaned_data = cleaned_data,
    dataset_name = dataset_name,
    target_col = target_col,
    max_runtime_secs = max_runtime_secs,
    ...
  )
  
  if (is.null(result)) {
    return(NULL)
  }
  
  # Save outputs
  save_h2o_leaderboard(result$leaderboard, dataset_name, target_col)
  save_h2o_model(result$best_model, dataset_name, target_col)
  save_h2o_predictions(result$predictions, dataset_name, target_col)
  
  message(sprintf("\n✅ H2O Pipeline complete for %s/%s", dataset_name, target_col))
  
  result
}

# ------------------------------------------------------------------------------
# run_all_h2o_pipelines: Run pipeline for all targets
# ------------------------------------------------------------------------------
run_all_h2o_pipelines <- function(cleaned_data,
                                   dataset_name,
                                   targets = NULL,
                                   skip_critical = FALSE,
                                   ...) {
  
  if (is.null(targets)) {
    targets <- names(cleaned_data$by_target)
  }
  
  results <- list()
  
  for (target in targets) {
    n_points <- nrow(cleaned_data$by_target[[target]])
    
    if (n_points < 6 && skip_critical) {
      message(sprintf(
        "\n⚠️ Skipping %s: only %d data points (critical threshold)",
        target, n_points
      ))
      next
    }
    
    tryCatch({
      results[[target]] <- run_h2o_pipeline(
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
# STRING HELPER
# ==============================================================================
`%+%` <- function(a, b) paste0(a, b)

# ==============================================================================
# TEST FUNCTION
# ==============================================================================

test_h2o_data_prep <- function() {
  # Create mock data
  test_df <- data.frame(
    Tahun = 2018:2025,
    JHT = c(100, 120, 140, 160, 180, 200, 220, 240)
  )
  
  result <- prepare_h2o_data(
    target_df = test_df,
    target_col = "JHT",
    lags = c(1, 2),
    windows = c(),
    holdout_years = c(2024, 2025)
  )
  
  stopifnot(nrow(result$train_df) > 0)
  stopifnot(nrow(result$test_df) == 2)
  stopifnot("lag_1" %in% result$feature_names)
  
  message("test_h2o_data_prep: PASSED")
  invisible(TRUE)
}
