# ==============================================================================
# features_timeseries.R - Time-Series Feature Engineering
# ==============================================================================
# 
# DESIGN RATIONALE (IMPORTANT - READ THIS):
# ------------------------------------------
# This dataset is EXTREMELY SMALL (11-12 data points per series).
# Creating too many features will lead to SEVERE OVERFITTING.
#
# Feature budget strategy:
#   - Maximum 2-3 lag features (lag_1, lag_2, optionally lag_3)
#   - Maximum 1-2 rolling features (rolling_mean_2, optionally rolling_mean_3)
#   - Each feature "costs" at least 1 data point (due to NA at start)
#
# With 12 data points and lag_2 + rolling_mean_2:
#   - Loss: 2 rows (first 2 rows become NA)
#   - Remaining: 10 usable rows
#   - For LSTM with lookback=3: 10-3+1 = 8 training sequences max
#
# DO NOT add more features without considering the data budget!
#
# ==============================================================================

# ------------------------------------------------------------------------------
# create_lag_features: Create lagged versions of a series
# ------------------------------------------------------------------------------
# Creates lag columns where lag_k contains the value from k periods ago.
#
# Args:
#   series  - Numeric vector of values
#   lags    - Vector of lag values to create (default: c(1, 2))
#   prefix  - Column name prefix (default: "lag")
#
# Returns:
#   Data frame with columns: value, lag_1, lag_2, etc.
#   First k rows will have NA in lag_k column (no prior data)
#
# Example:
#   series = [100, 110, 120, 130]
#   Result:
#     value  lag_1  lag_2
#     100    NA     NA
#     110    100    NA
#     120    110    100
#     130    120    110
# ------------------------------------------------------------------------------
create_lag_features <- function(series, lags = c(1, 2), prefix = "lag") {
  n <- length(series)
  
  # Initialize result with original values
  result <- data.frame(value = series)
  
  # Create each lag
  for (lag in lags) {
    if (lag >= n) {
      warning(sprintf(
        "Lag %d >= series length %d, column will be all NA",
        lag, n
      ))
    }
    
    # Shift values by lag positions
    lagged <- c(rep(NA_real_, lag), series[1:(n - lag)])
    col_name <- paste0(prefix, "_", lag)
    result[[col_name]] <- lagged
  }
  
  result
}

# ------------------------------------------------------------------------------
# create_rolling_features: Create rolling window statistics
# ------------------------------------------------------------------------------
# Creates rolling mean (and optionally other stats) over a window.
#
# Args:
#   series   - Numeric vector of values
#   windows  - Vector of window sizes (default: c(2, 3))
#   stats    - Vector of statistics to compute (default: "mean")
#              Supported: "mean", "sd", "min", "max"
#   align    - Alignment of window (default: "right" = trailing window)
#
# Returns:
#   Data frame with rolling statistic columns
#   First (window-1) rows will have NA (not enough prior data)
#
# Example:
#   series = [100, 200, 300, 400]
#   windows = c(2)
#   Result:
#     rolling_mean_2
#     NA             (not enough data)
#     150            (mean of 100, 200)
#     250            (mean of 200, 300)
#     350            (mean of 300, 400)
# ------------------------------------------------------------------------------
create_rolling_features <- function(series, windows = c(2), stats = "mean", align = "right") {
  n <- length(series)
  result <- data.frame(row.names = 1:n)
  
  for (window in windows) {
    if (window > n) {
      warning(sprintf(
        "Window %d > series length %d, column will be all NA",
        window, n
      ))
    }
    
    for (stat in stats) {
      # Compute rolling statistic
      rolled <- numeric(n)
      
      for (i in 1:n) {
        if (i < window) {
          # Not enough prior data
          rolled[i] <- NA_real_
        } else {
          # Get window slice (trailing)
          window_slice <- series[(i - window + 1):i]
          
          rolled[i] <- switch(
            stat,
            "mean" = mean(window_slice, na.rm = FALSE),
            "sd"   = sd(window_slice, na.rm = FALSE),
            "min"  = min(window_slice, na.rm = FALSE),
            "max"  = max(window_slice, na.rm = FALSE),
            NA_real_
          )
        }
      }
      
      col_name <- sprintf("rolling_%s_%d", stat, window)
      result[[col_name]] <- rolled
    }
  }
  
  result
}

# ------------------------------------------------------------------------------
# build_features_for_target: Complete feature pipeline for one target series
# ------------------------------------------------------------------------------
# Builds all features for a target column, ready for modeling.
#
# Args:
#   df          - Data frame with Tahun and target column
#   target_col  - Name of target column
#   lags        - Lag values (default from config or c(1, 2))
#   windows     - Rolling window sizes (default c(2) or empty for very small data)
#
# Returns:
#   List with:
#     - features_df: Data frame with Tahun, target, and all features
#     - feature_names: Names of feature columns (for model input)
#     - rows_dropped: Number of rows with NA (unusable for training)
#     - usable_rows: Number of rows available for training
# ------------------------------------------------------------------------------
build_features_for_target <- function(df, target_col, lags = c(1, 2), windows = c(2)) {
  if (!target_col %in% names(df)) {
    stop(sprintf("Target column '%s' not found in data frame", target_col), call. = FALSE)
  }
  
  if (!"Tahun" %in% names(df)) {
    stop("Data frame must have 'Tahun' column", call. = FALSE)
  }
  
  series <- df[[target_col]]
  n <- length(series)
  
  # Start with Tahun and target
  result <- data.frame(
    Tahun = df$Tahun,
    target = series
  )
  
  feature_names <- character(0)
  
  # Add lag features
  if (length(lags) > 0) {
    lag_df <- create_lag_features(series, lags = lags)
    lag_cols <- setdiff(names(lag_df), "value")
    result <- cbind(result, lag_df[, lag_cols, drop = FALSE])
    feature_names <- c(feature_names, lag_cols)
  }
  
  # Add rolling features (only if we have enough data)
  # Skip rolling features if data is too small
  if (length(windows) > 0 && n >= max(windows) + 2) {
    roll_df <- create_rolling_features(series, windows = windows, stats = "mean")
    roll_cols <- names(roll_df)
    result <- cbind(result, roll_df)
    feature_names <- c(feature_names, roll_cols)
  } else if (length(windows) > 0) {
    message(sprintf(
      "  Skipping rolling features for %s: data too small (%d points)",
      target_col, n
    ))
  }
  
  # Calculate usable rows (non-NA in all feature columns)
  if (length(feature_names) > 0) {
    complete_mask <- complete.cases(result[, c("target", feature_names)])
  } else {
    complete_mask <- !is.na(result$target)
  }
  
  rows_dropped <- sum(!complete_mask)
  usable_rows <- sum(complete_mask)
  
  # Warning if too few usable rows
  if (usable_rows < 6) {
    warning(sprintf(
      "Target '%s': only %d usable rows after feature creation (dropped %d). Model may be unreliable.",
      target_col, usable_rows, rows_dropped
    ))
  }
  
  # Warning if > 30% rows dropped
  drop_pct <- rows_dropped / n * 100
  if (drop_pct > 30) {
    warning(sprintf(
      "Target '%s': %.1f%% of data dropped due to feature NA (%d/%d rows). Consider reducing lag/window sizes.",
      target_col, drop_pct, rows_dropped, n
    ))
  }
  
  message(sprintf(
    "  %s: %d features, %d/%d usable rows (dropped %d = %.1f%%)",
    target_col, length(feature_names), usable_rows, n, rows_dropped, drop_pct
  ))
  
  list(
    features_df = result,
    feature_names = feature_names,
    rows_dropped = rows_dropped,
    usable_rows = usable_rows,
    complete_mask = complete_mask
  )
}

# ------------------------------------------------------------------------------
# get_complete_rows: Filter to only rows with complete feature data
# ------------------------------------------------------------------------------
get_complete_rows <- function(feature_result) {
  feature_result$features_df[feature_result$complete_mask, ]
}

# ------------------------------------------------------------------------------
# recommend_feature_config: Recommend feature configuration based on data size
# ------------------------------------------------------------------------------
recommend_feature_config <- function(n_data_points) {
  if (n_data_points < 5) {
    # Too small for any meaningful features
    list(lags = c(1), windows = c(), message = "CRITICAL: Data too small, minimal features only")
  } else if (n_data_points < 8) {
    # Very small - minimal features
    list(lags = c(1), windows = c(), message = "WARNING: Small data, using lag_1 only")
  } else if (n_data_points < 12) {
    # Small but workable
    list(lags = c(1, 2), windows = c(), message = "Small data, using lag_1 and lag_2")
  } else {
    # Reasonable size
    list(lags = c(1, 2), windows = c(2), message = "Standard features: lag_1, lag_2, rolling_mean_2")
  }
}

# ------------------------------------------------------------------------------
# Test functions
# ------------------------------------------------------------------------------
test_lag_features <- function() {
  series <- c(100, 110, 120, 130)
  result <- create_lag_features(series, lags = c(1, 2))
  
  # Check structure
  stopifnot(ncol(result) == 3)  # value, lag_1, lag_2
  stopifnot(all(names(result) == c("value", "lag_1", "lag_2")))
  
  # Check values
  stopifnot(all(result$value == c(100, 110, 120, 130)))
  stopifnot(is.na(result$lag_1[1]))
  stopifnot(result$lag_1[2] == 100)
  stopifnot(result$lag_1[4] == 120)
  stopifnot(is.na(result$lag_2[1]) && is.na(result$lag_2[2]))
  stopifnot(result$lag_2[3] == 100)
  
  message("test_lag_features: PASSED")
  invisible(TRUE)
}

test_rolling_features <- function() {
  series <- c(100, 200, 300, 400)
  result <- create_rolling_features(series, windows = c(2), stats = "mean")
  
  # Check structure
  stopifnot(ncol(result) == 1)
  stopifnot(names(result) == "rolling_mean_2")
  
  # Check values
  stopifnot(is.na(result$rolling_mean_2[1]))
  stopifnot(result$rolling_mean_2[2] == 150)  # mean(100, 200)
  stopifnot(result$rolling_mean_2[3] == 250)  # mean(200, 300)
  stopifnot(result$rolling_mean_2[4] == 350)  # mean(300, 400)
  
  message("test_rolling_features: PASSED")
  invisible(TRUE)
}

test_build_features <- function() {
  test_df <- data.frame(
    Tahun = 2018:2023,
    JHT = c(100, 150, 200, 250, 300, 350)
  )
  
  result <- build_features_for_target(test_df, "JHT", lags = c(1, 2), windows = c(2))
  
  # Check usable rows (first 2 rows have NA due to lag_2)
  stopifnot(result$rows_dropped == 2)
  stopifnot(result$usable_rows == 4)
  stopifnot(length(result$feature_names) == 3)  # lag_1, lag_2, rolling_mean_2
  
  message("test_build_features: PASSED")
  invisible(TRUE)
}
