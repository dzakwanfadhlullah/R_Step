# ==============================================================================
# 03_forecast_h2o.R - H2O AutoML Future Forecasting Script
# ==============================================================================
# This script handles forecasting FUTURE years (2026-2030) using H2O AutoML:
#   - Train on ALL available data (no holdout)
#   - Generate forecasts for FORECAST_YEARS
#   - Use recursive prediction for GBM/RF models
#
# Usage:
#   source("experiments/h2o_automl/scripts/03_forecast_h2o.R")
# ==============================================================================

library(h2o)

# ==============================================================================
# H2O FORECAST FUNCTIONS
# ==============================================================================

# ------------------------------------------------------------------------------
# forecast_future_h2o: Generate forecasts for future years using H2O model
# ------------------------------------------------------------------------------
forecast_future_h2o <- function(model,
                                 historical_df,
                                 target_col,
                                 forecast_years) {
  
  # Get last known values for lag features
  n <- nrow(historical_df)
  values <- historical_df[[target_col]]
  
  forecasts <- numeric(length(forecast_years))
  
  # Initialize with historical lag values
  lag_1 <- values[n]
  lag_2 <- values[n - 1]
  
  for (i in seq_along(forecast_years)) {
    # Create features for prediction
    rolling_mean <- mean(c(lag_1, lag_2))
    
    new_data <- data.frame(
      lag_1 = lag_1,
      lag_2 = lag_2,
      rolling_mean_2 = rolling_mean
    )
    
    # Convert to H2O frame
    new_h2o <- as.h2o(new_data)
    
    # Predict
    pred <- h2o.predict(model, new_h2o)
    pred_val <- as.numeric(pred[1, 1])
    
    forecasts[i] <- pred_val
    
    # Update lags for next iteration
    lag_2 <- lag_1
    lag_1 <- pred_val
    
    # Clean up
    h2o.rm(new_h2o)
  }
  
  data.frame(
    year = forecast_years,
    predicted = forecasts
  )
}

# ------------------------------------------------------------------------------
# train_and_forecast_h2o: Full training + forecasting pipeline
# ------------------------------------------------------------------------------
train_and_forecast_h2o <- function(cleaned_data,
                                     target_col,
                                     forecast_years = FORECAST_YEARS,
                                     max_runtime_secs = H2O_MAX_RUNTIME_SECS,
                                     max_models = H2O_MAX_MODELS,
                                     seed = RANDOM_SEED) {
  
  # Get target data
  target_data <- cleaned_data$by_target[[target_col]]
  
  if (is.null(target_data) || nrow(target_data) < 4) {
    warning(sprintf("Insufficient data for %s (need >= 4 rows)", target_col))
    return(NULL)
  }
  
  # Create lag features for ALL data
  values <- target_data[[target_col]]
  years <- target_data$Tahun
  n <- length(values)
  
  # Features: lag_1, lag_2, rolling_mean_2
  train_df <- data.frame(
    year = years[3:n],
    value = values[3:n],
    lag_1 = values[2:(n-1)],
    lag_2 = values[1:(n-2)],
    rolling_mean_2 = (values[2:(n-1)] + values[1:(n-2)]) / 2
  )
  
  message(sprintf("  Training H2O on %d samples (all data)...", nrow(train_df)))
  
  # Convert to H2O frame
  train_h2o <- as.h2o(train_df)
  
  # Column names
  x <- c("lag_1", "lag_2", "rolling_mean_2")
  y <- "value"
  
  # Train AutoML
  aml <- h2o.automl(
    x = x,
    y = y,
    training_frame = train_h2o,
    max_runtime_secs = max_runtime_secs,
    max_models = max_models,
    seed = seed,
    nfolds = 0,  # No CV for small data
    exclude_algos = c("DeepLearning", "StackedEnsemble"),
    sort_metric = "RMSE",
    stopping_metric = "RMSE",
    stopping_tolerance = 0.001,
    stopping_rounds = 3
  )
  
  message(sprintf("  ✓ AutoML complete. Best model: %s", aml@leader@model_id))
  
  # Generate forecasts
  forecasts <- forecast_future_h2o(
    model = aml@leader,
    historical_df = target_data,
    target_col = target_col,
    forecast_years = forecast_years
  )
  
  message(sprintf("  ✓ Forecast generated for years: %s", 
                  paste(forecast_years, collapse = ", ")))
  
  # Clean up H2O frames
  h2o.rm(train_h2o)
  
  list(
    model = aml@leader,
    model_id = aml@leader@model_id,
    leaderboard = as.data.frame(aml@leaderboard),
    forecasts = forecasts,
    historical = data.frame(year = years, actual = values)
  )
}

# ------------------------------------------------------------------------------
# run_forecast_pipeline_h2o: Run forecasting for one dataset/target
# ------------------------------------------------------------------------------
run_forecast_pipeline_h2o <- function(cleaned_data,
                                        dataset_name,
                                        target_col,
                                        forecast_years = FORECAST_YEARS,
                                        max_runtime_secs = H2O_MAX_RUNTIME_SECS,
                                        max_models = H2O_MAX_MODELS,
                                        seed = RANDOM_SEED) {
  
  message(sprintf("\n  [H2O] Forecasting %s/%s...", dataset_name, target_col))
  
  result <- train_and_forecast_h2o(
    cleaned_data = cleaned_data,
    target_col = target_col,
    forecast_years = forecast_years,
    max_runtime_secs = max_runtime_secs,
    max_models = max_models,
    seed = seed
  )
  
  if (is.null(result)) {
    return(NULL)
  }
  
  # Save model
  models_dir <- file.path("experiments", "h2o_automl", "models", paste0(dataset_name, "_", target_col))
  dir.create(models_dir, recursive = TRUE, showWarnings = FALSE)
  h2o.saveModel(result$model, path = models_dir, force = TRUE)
  message(sprintf("  ✓ Model saved: %s", models_dir))
  
  # Save leaderboard
  lb_path <- file.path("experiments", "h2o_automl", "outputs", 
                       sprintf("leaderboard_%s_%s.csv", dataset_name, target_col))
  dir.create(dirname(lb_path), recursive = TRUE, showWarnings = FALSE)
  write.csv(result$leaderboard, lb_path, row.names = FALSE)
  
  # Save forecasts
  forecast_dir <- file.path("results", "forecasts")
  dir.create(forecast_dir, recursive = TRUE, showWarnings = FALSE)
  
  forecast_df <- result$forecasts
  forecast_df$dataset <- dataset_name
  forecast_df$target <- target_col
  forecast_df$model <- "h2o"
  
  # Add historical data
  hist_df <- result$historical
  hist_df$dataset <- dataset_name
  hist_df$target <- target_col
  hist_df$model <- "h2o"
  hist_df$predicted <- NA
  
  # Combine
  combined_df <- rbind(
    hist_df[, c("year", "dataset", "target", "model", "actual", "predicted")],
    data.frame(
      year = forecast_df$year,
      dataset = dataset_name,
      target = target_col,
      model = "h2o",
      actual = NA,
      predicted = forecast_df$predicted
    )
  )
  
  forecast_path <- file.path(
    forecast_dir, 
    sprintf("forecast_h2o_%s_%s.csv", dataset_name, target_col)
  )
  write.csv(combined_df, forecast_path, row.names = FALSE)
  message(sprintf("  ✓ Forecast saved: %s", forecast_path))
  
  list(
    forecasts = forecast_df,
    historical = result$historical,
    combined = combined_df,
    model_id = result$model_id,
    leaderboard = result$leaderboard
  )
}
