# ==============================================================================
# 01_train_lstm.R - LSTM Model Training Script
# ==============================================================================
# This script trains LSTM models for time-series forecasting.
#
# Prerequisites:
#   - TensorFlow/Keras installed (run 00_setup_keras.R first)
#   - Data preprocessed and features created
#
# Usage:
#   source("experiments/lstm/scripts/01_train_lstm.R")
# ==============================================================================

# Load dependencies
library(keras3)
library(tensorflow)

# Set working directory to project root
if (!file.exists("R/config.R")) {
  setwd("project-timeseries-forecast")
}

# Source project files (handled by main.R)
# source("R/config.R")
# source("R/utils_io.R")
# source("R/preprocess.R")
# source("R/split.R")
# source("R/features_timeseries.R")

# Set seeds for reproducibility
set.seed(RANDOM_SEED)
tensorflow::tf$random$set_seed(as.integer(RANDOM_SEED))

# ==============================================================================
# SEQUENCE CREATION FUNCTIONS
# ==============================================================================

# ------------------------------------------------------------------------------
# create_sequences: Convert time-series to LSTM-ready sequences
# ------------------------------------------------------------------------------
# Converts a 1D series into supervised learning format for LSTM.
#
# Args:
#   series   - Numeric vector (should be normalized)
#   lookback - Number of past timesteps to use as input
#
# Returns:
#   List with:
#     - X: 3D array (samples, timesteps, features) for LSTM input
#     - y: 1D array of target values
#     - indices: Original indices (useful for train/test split tracking)
#
# Example with lookback=3:
#   series = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
#   X[0] = [0.1, 0.2, 0.3] -> y[0] = 0.4
#   X[1] = [0.2, 0.3, 0.4] -> y[1] = 0.5
#   X[2] = [0.3, 0.4, 0.5] -> y[2] = 0.6
# ------------------------------------------------------------------------------
create_sequences <- function(series, lookback) {
  n <- length(series)
  
  if (n <= lookback) {
    stop(sprintf(
      "Series length (%d) must be greater than lookback (%d)",
      n, lookback
    ), call. = FALSE)
  }
  
  # Number of samples we can create
  n_samples <- n - lookback
  
  # Initialize arrays
  X <- array(NA_real_, dim = c(n_samples, lookback, 1))
  y <- numeric(n_samples)
  indices <- integer(n_samples)
  
  for (i in 1:n_samples) {
    # Input: previous 'lookback' values
    X[i, , 1] <- series[i:(i + lookback - 1)]
    # Target: next value after the lookback window
    y[i] <- series[i + lookback]
    # Original index of target (1-indexed)
    indices[i] <- i + lookback
  }
  
  message(sprintf(
    "Created %d sequences with lookback=%d from %d data points",
    n_samples, lookback, n
  ))
  
  list(
    X = X,
    y = y,
    indices = indices,
    lookback = lookback
  )
}

# ------------------------------------------------------------------------------
# prepare_lstm_data: Full data preparation pipeline for LSTM
# ------------------------------------------------------------------------------
# Handles normalization, sequence creation, and train/test split.
#
# Args:
#   target_df   - Data frame with Tahun and target column
#   target_col  - Name of target column
#   lookback    - LSTM lookback window (default from config)
#   holdout_years - Years to use for testing
#   normalize_method - "minmax" or "zscore" (default "minmax")
#
# Returns:
#   List with:
#     - X_train, y_train: Training data
#     - X_test, y_test: Test data
#     - scaler_params: Normalization parameters (for inverse transform)
#     - train_years, test_years: Years in each set
#     - metadata: Additional info
# ------------------------------------------------------------------------------
prepare_lstm_data <- function(target_df, 
                              target_col,
                              lookback = LSTM_LOOKBACK,
                              holdout_years = HOLDOUT_YEARS,
                              normalize_method = "minmax") {
  
  message(sprintf("\n=== Preparing LSTM data for %s ===", target_col))
  
  # Step 1: Split data FIRST (before any preprocessing)
  split_result <- create_holdout_split(target_df, holdout_years)
  train_df <- split_result$train
  test_df <- split_result$test
  
  train_series <- train_df[[target_col]]
  test_series <- test_df[[target_col]]
  
  # Step 2: Normalize using TRAINING data parameters only
  # This is critical to prevent data leakage!
  message("  Normalizing data (params from training set only)...")
  
  train_norm_result <- normalize_series(train_series, method = normalize_method)
  scaler_params <- train_norm_result$params
  train_normalized <- train_norm_result$normalized
  
  # Apply same normalization to test data using TRAINING params
  test_normalized <- normalize_series(
    test_series, 
    method = normalize_method, 
    params = scaler_params
  )$normalized
  
  message(sprintf(
    "  Scaler params: min=%.2e, max=%.2e",
    scaler_params$min, scaler_params$max
  ))
  
  # Step 3: Combine normalized data in temporal order for sequence creation
  # We need the full normalized series to create sequences that span train/test
  full_normalized <- c(train_normalized, test_normalized)
  full_years <- c(train_df$Tahun, test_df$Tahun)
  
  n_train <- length(train_normalized)
  n_total <- length(full_normalized)
  
  # Step 4: Create sequences from FULL normalized data
  seq_result <- create_sequences(full_normalized, lookback)
  
  # Step 5: Split sequences into train/test based on target year
  # The target of sequence i corresponds to full_years[indices[i]]
  target_years <- full_years[seq_result$indices]
  
  # Train sequences: target year is in training period
  train_seq_mask <- target_years < min(holdout_years)
  # Test sequences: target year is in holdout period
  test_seq_mask <- target_years %in% holdout_years
  
  X_train <- seq_result$X[train_seq_mask, , , drop = FALSE]
  y_train <- seq_result$y[train_seq_mask]
  train_target_years <- target_years[train_seq_mask]
  
  X_test <- seq_result$X[test_seq_mask, , , drop = FALSE]
  y_test <- seq_result$y[test_seq_mask]
  test_target_years <- target_years[test_seq_mask]
  
  # Validation
  if (length(y_train) == 0) {
    warning("No training sequences created! Data may be too small.")
  }
  if (length(y_test) == 0) {
    warning("No test sequences created! Check holdout_years setting.")
  }
  
  message(sprintf(
    "  Train sequences: %d, Test sequences: %d",
    length(y_train), length(y_test)
  ))
  message(sprintf(
    "  Train years: %s, Test years: %s",
    paste(train_target_years, collapse = ","),
    paste(test_target_years, collapse = ",")
  ))
  
  list(
    X_train = X_train,
    y_train = y_train,
    X_test = X_test,
    y_test = y_test,
    scaler_params = scaler_params,
    train_target_years = train_target_years,
    test_target_years = test_target_years,
    lookback = lookback,
    metadata = list(
      target_col = target_col,
      n_train_sequences = length(y_train),
      n_test_sequences = length(y_test),
      normalize_method = normalize_method,
      holdout_years = holdout_years
    )
  )
}

# ==============================================================================
# LSTM MODEL BUILDING
# ==============================================================================

# ------------------------------------------------------------------------------
# build_lstm_model: Create LSTM model architecture
# ------------------------------------------------------------------------------
build_lstm_model <- function(lookback, 
                             units = LSTM_UNITS, 
                             learning_rate = 0.001) {
  
  model <- keras_model_sequential() %>%
    # LSTM layer
    layer_lstm(
      units = units,
      input_shape = c(lookback, 1),
      return_sequences = FALSE
    ) %>%
    # Dropout for regularization (important for small data)
    layer_dropout(rate = 0.2) %>%
    # Dense output layer
    layer_dense(units = 1)
  
  # Compile model
  model %>% compile(
    loss = "mse",
    optimizer = optimizer_adam(learning_rate = learning_rate),
    metrics = c("mae")
  )
  
  message(sprintf(
    "Built LSTM model: lookback=%d, units=%d, params=%d",
    lookback, units, count_params(model)
  ))
  
  model
}

# ==============================================================================
# TRAINING FUNCTIONS
# ==============================================================================

# ------------------------------------------------------------------------------
# train_lstm_for_target: Train LSTM model for a single target
# ------------------------------------------------------------------------------
train_lstm_for_target <- function(cleaned_data,
                                   target_col,
                                   lookback = LSTM_LOOKBACK,
                                   holdout_years = HOLDOUT_YEARS,
                                   epochs = 100,
                                   batch_size = 1,
                                   verbose = 1) {
  
  message(sprintf("\n########## Training LSTM for %s ##########", target_col))
  
  # Get target data
  target_df <- cleaned_data$by_target[[target_col]]
  
  if (is.null(target_df) || nrow(target_df) == 0) {
    warning(sprintf("No data available for target: %s", target_col))
    return(NULL)
  }
  
  # Check if we have enough data
  n_points <- nrow(target_df)
  min_required <- lookback + length(holdout_years) + 1
  
  if (n_points < min_required) {
    warning(sprintf(
      "Target %s: insufficient data (%d points, need %d). Skipping.",
      target_col, n_points, min_required
    ))
    return(NULL)
  }
  
  # Prepare data
  data <- prepare_lstm_data(
    target_df = target_df,
    target_col = target_col,
    lookback = lookback,
    holdout_years = holdout_years
  )
  
  # Check sequences
  if (data$metadata$n_train_sequences < 3) {
    warning(sprintf(
      "Target %s: only %d training sequences. Model may be unreliable.",
      target_col, data$metadata$n_train_sequences
    ))
  }
  
  # Build model
  model <- build_lstm_model(lookback = lookback)
  
  # Early stopping callback
  early_stop <- callback_early_stopping(
    monitor = "loss",
    patience = 20,
    restore_best_weights = TRUE
  )
  
  # Train model
  message(sprintf("Training for max %d epochs...", epochs))
  
  history <- model %>% fit(
    x = data$X_train,
    y = data$y_train,
    epochs = epochs,
    batch_size = batch_size,
    validation_split = 0,  # No validation split for very small data
    callbacks = list(early_stop),
    verbose = verbose
  )
  
  # Make predictions
  message("Making predictions...")
  
  train_pred_normalized <- predict(model, data$X_train)
  test_pred_normalized <- predict(model, data$X_test)
  
  # Inverse transform predictions to original scale
  # Wrap single target params in a list and name it with target_col
  scaler_list <- list()
  scaler_list[[target_col]] <- data$scaler_params
  
  train_pred <- denormalize_target_data(
    as.vector(train_pred_normalized),
    target_col,
    scaler_list
  )
  test_pred <- denormalize_target_data(
    as.vector(test_pred_normalized),
    target_col,
    scaler_list
  )
  
  # Inverse transform actuals
  train_actual <- inverse_normalize(data$y_train, data$scaler_params)
  test_actual <- inverse_normalize(data$y_test, data$scaler_params)
  
  message(sprintf("Training complete. Final loss: %.6f", min(history$metrics$loss)))
  
  list(
    model = model,
    history = history,
    predictions = list(
      train = data.frame(
        year = data$train_target_years,
        actual = train_actual,
        predicted = train_pred
      ),
      test = data.frame(
        year = data$test_target_years,
        actual = test_actual,
        predicted = test_pred
      )
    ),
    scaler_params = data$scaler_params,
    metadata = data$metadata
  )
}

# ==============================================================================
# TEST FUNCTIONS
# ==============================================================================

test_create_sequences <- function() {
  series <- c(0.1, 0.2, 0.3, 0.4, 0.5, 0.6)
  result <- create_sequences(series, lookback = 3)
  
  # Should create 3 sequences
  stopifnot(dim(result$X)[1] == 3)
  stopifnot(length(result$y) == 3)
  
  # Check first sequence
  stopifnot(all(result$X[1, , 1] == c(0.1, 0.2, 0.3)))
  stopifnot(result$y[1] == 0.4)
  
  # Check last sequence  
  stopifnot(all(result$X[3, , 1] == c(0.3, 0.4, 0.5)))
  stopifnot(result$y[3] == 0.6)
  
  message("test_create_sequences: PASSED")
  invisible(TRUE)
}

# Run tests if executed directly
if (interactive()) {
  message("Running tests...")
  test_create_sequences()
  message("\nTests passed! Ready to train models.")
}
