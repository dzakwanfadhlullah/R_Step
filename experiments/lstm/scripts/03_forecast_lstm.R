# ==============================================================================
# 03_forecast_lstm.R - LSTM Future Forecasting Script
# ==============================================================================
# This script handles forecasting FUTURE years (2026-2030) using LSTM:
#   - Train on ALL available data (no holdout)
#   - Generate forecasts for FORECAST_YEARS
#   - Use recursive prediction (predict one year, use as input for next)
#
# Usage:
#   source("experiments/lstm/scripts/03_forecast_lstm.R")
# ==============================================================================

# ==============================================================================
# RECURSIVE FORECAST FUNCTION
# ==============================================================================

# ------------------------------------------------------------------------------
# forecast_future_years: Generate forecasts for multiple future years
# ------------------------------------------------------------------------------
# This uses a recursive approach:
#   1. Train model on all historical data
#   2. Use last 'lookback' values to predict year N+1
#   3. Append prediction to history
#   4. Repeat for each forecast year
# ------------------------------------------------------------------------------
forecast_future_lstm <- function(model,
                                   historical_values,
                                   scaler_params,
                                   forecast_years,
                                   lookback) {
  
  # Initialize with historical values (normalized)
  min_val <- scaler_params$min
  max_val <- scaler_params$max
  range_val <- max_val - min_val
  
  # Normalize historical values
  hist_normalized <- (historical_values - min_val) / range_val
  
  # Store forecasts
  forecasts <- numeric(length(forecast_years))
  
  # Current sequence (last 'lookback' values)
  current_seq <- tail(hist_normalized, lookback)
  
  for (i in seq_along(forecast_years)) {
    # Prepare input: reshape to (1, lookback, 1)
    X_input <- array(current_seq, dim = c(1, lookback, 1))
    
    # Predict next value (normalized)
    pred_norm <- predict(model, X_input, verbose = 0)
    pred_norm <- as.vector(pred_norm)[1]
    
    # Inverse transform to original scale
    pred_original <- pred_norm * range_val + min_val
    forecasts[i] <- pred_original
    
    # Update sequence for next iteration
    current_seq <- c(current_seq[-1], pred_norm)
  }
  
  data.frame(
    year = forecast_years,
    predicted = forecasts
  )
}

# ------------------------------------------------------------------------------
# train_and_forecast_lstm: Full training + forecasting pipeline
# ------------------------------------------------------------------------------
train_and_forecast_lstm <- function(cleaned_data,
                                      target_col,
                                      lookback = LSTM_LOOKBACK,
                                      forecast_years = FORECAST_YEARS,
                                      epochs = 100,
                                      batch_size = 1,
                                      verbose = 0) {
  
  # Get target data
  target_data <- cleaned_data$by_target[[target_col]]
  
  if (is.null(target_data) || nrow(target_data) < lookback + 1) {
    warning(sprintf("Insufficient data for %s", target_col))
    return(NULL)
  }
  
  # Get values
  years <- target_data$Tahun
  values <- target_data[[target_col]]
  
  # Normalize (min-max scaling)
  min_val <- min(values)
  max_val <- max(values)
  range_val <- max_val - min_val
  
  scaler_params <- list(
    method = "minmax",
    min = min_val,
    max = max_val
  )
  
  # Handle constant data
  if (range_val == 0) {
    warning(sprintf("Constant values for %s", target_col))
    return(list(
      forecasts = data.frame(year = forecast_years, predicted = rep(values[1], length(forecast_years))),
      historical = data.frame(year = years, actual = values),
      scaler_params = scaler_params
    ))
  }
  
  values_norm <- (values - min_val) / range_val
  
  # Create sequences for training on ALL data
  n <- length(values_norm)
  n_samples <- n - lookback
  
  X <- array(NA_real_, dim = c(n_samples, lookback, 1))
  y <- numeric(n_samples)
  
  for (i in 1:n_samples) {
    X[i, , 1] <- values_norm[i:(i + lookback - 1)]
    y[i] <- values_norm[i + lookback]
  }
  
  message(sprintf("  Training LSTM on %d samples (all data)...", n_samples))
  
  # Build model
  model <- keras_model_sequential() %>%
    layer_lstm(
      units = LSTM_UNITS,
      input_shape = c(lookback, 1),
      return_sequences = FALSE
    ) %>%
    layer_dense(units = 1)
  
  model %>% compile(
    optimizer = optimizer_adam(learning_rate = 0.01),
    loss = 'mse',
    metrics = c('mae')
  )
  
  # Train model (use same data for validation since we're using all data)
  history <- model %>% fit(
    x = X,
    y = y,
    epochs = epochs,
    batch_size = batch_size,
    validation_split = 0.2,
    verbose = verbose,
    callbacks = list(
      callback_early_stopping(
        monitor = "val_loss",
        patience = 15,
        restore_best_weights = TRUE
      )
    )
  )
  
  # Generate forecasts for future years
  forecasts <- forecast_future_lstm(
    model = model,
    historical_values = values,
    scaler_params = scaler_params,
    forecast_years = forecast_years,
    lookback = lookback
  )
  
  message(sprintf("  ✓ Forecast generated for years: %s", 
                  paste(forecast_years, collapse = ", ")))
  
  list(
    model = model,
    history = history,
    forecasts = forecasts,
    historical = data.frame(year = years, actual = values),
    scaler_params = scaler_params
  )
}

# ------------------------------------------------------------------------------
# run_forecast_pipeline: Run forecasting for one dataset/target
# ------------------------------------------------------------------------------
run_forecast_pipeline_lstm <- function(cleaned_data,
                                         dataset_name,
                                         target_col,
                                         lookback = LSTM_LOOKBACK,
                                         forecast_years = FORECAST_YEARS,
                                         epochs = 100,
                                         verbose = 0) {
  
  message(sprintf("\n  [LSTM] Forecasting %s/%s...", dataset_name, target_col))
  
  result <- train_and_forecast_lstm(
    cleaned_data = cleaned_data,
    target_col = target_col,
    lookback = lookback,
    forecast_years = forecast_years,
    epochs = epochs,
    verbose = verbose
  )
  
  if (is.null(result)) {
    return(NULL)
  }
  
  # Save model
  models_dir <- file.path("experiments", "lstm", "models")
  dir.create(models_dir, recursive = TRUE, showWarnings = FALSE)
  model_path <- file.path(models_dir, sprintf("lstm_%s_%s.keras", dataset_name, target_col))
  save_model(result$model, model_path)
  message(sprintf("  ✓ Model saved: %s", model_path))
  
  # Save forecasts
  forecast_dir <- file.path("results", "forecasts")
  dir.create(forecast_dir, recursive = TRUE, showWarnings = FALSE)
  
  forecast_df <- result$forecasts
  forecast_df$dataset <- dataset_name
  forecast_df$target <- target_col
  forecast_df$model <- "lstm"
  forecast_df$timestamp <- Sys.time()
  
  # Add historical data for plotting
  hist_df <- result$historical
  hist_df$dataset <- dataset_name
  hist_df$target <- target_col
  hist_df$model <- "lstm"
  hist_df$predicted <- NA  # No prediction for historical years
  
  # Combine
  combined_df <- rbind(
    hist_df[, c("year", "dataset", "target", "model", "actual", "predicted")],
    data.frame(
      year = forecast_df$year,
      dataset = dataset_name,
      target = target_col,
      model = "lstm",
      actual = NA,  # No actual for forecast years
      predicted = forecast_df$predicted
    )
  )
  
  forecast_path <- file.path(
    forecast_dir, 
    sprintf("forecast_lstm_%s_%s.csv", dataset_name, target_col)
  )
  write.csv(combined_df, forecast_path, row.names = FALSE)
  message(sprintf("  ✓ Forecast saved: %s", forecast_path))
  
  list(
    forecasts = forecast_df,
    historical = result$historical,
    combined = combined_df,
    model = result$model,
    scaler_params = result$scaler_params
  )
}
