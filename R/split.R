# ==============================================================================
# split.R - Time-Respecting Data Splitting Functions
# ==============================================================================
# Functions:
#   - create_holdout_split(): Single train/test split based on holdout years
#   - create_rolling_splits(): Walk-forward / rolling-origin validation splits
#   - validate_split(): Ensure no temporal leakage
# ==============================================================================

# ------------------------------------------------------------------------------
# create_holdout_split: Create single train/test split based on holdout years
# ------------------------------------------------------------------------------
# Args:
#   df            - Data frame with Tahun column (sorted)
#   holdout_years - Vector of years to use as test set (e.g., c(2024, 2025))
#
# Returns:
#   List with:
#     - train: Data frame with training data (years < min(holdout_years))
#     - test: Data frame with test data (years in holdout_years)
#     - info: Split metadata
# ------------------------------------------------------------------------------
create_holdout_split <- function(df, holdout_years) {
  if (!"Tahun" %in% names(df)) {
    stop("Data frame must have 'Tahun' column", call. = FALSE)
  }
  
  # Ensure sorted
  df <- df[order(df$Tahun), ]
  
  # Split threshold
  split_year <- min(holdout_years)
  
  # Create train/test
  train_mask <- df$Tahun < split_year
  test_mask <- df$Tahun %in% holdout_years
  
  train_df <- df[train_mask, ]
  test_df <- df[test_mask, ]
  
  # Validate no temporal leakage
  if (nrow(train_df) > 0 && nrow(test_df) > 0) {
    if (max(train_df$Tahun) >= min(test_df$Tahun)) {
      stop("Temporal leakage detected: train max year >= test min year", call. = FALSE)
    }
  }
  
  # Info
  info <- list(
    split_type = "holdout",
    holdout_years = holdout_years,
    train_years = train_df$Tahun,
    test_years = test_df$Tahun,
    n_train = nrow(train_df),
    n_test = nrow(test_df)
  )
  
  message(sprintf(
    "Holdout split: train=%d (%d-%d), test=%d (%s)",
    info$n_train,
    if (info$n_train > 0) min(info$train_years) else NA,
    if (info$n_train > 0) max(info$train_years) else NA,
    info$n_test,
    paste(holdout_years, collapse = ",")
  ))
  
  list(
    train = train_df,
    test = test_df,
    info = info
  )
}

# ------------------------------------------------------------------------------
# create_rolling_splits: Create walk-forward / rolling-origin splits
# ------------------------------------------------------------------------------
# Walk-forward validation for time-series:
#   Fold 1: train = years[1:initial_window], test = years[initial_window + 1]
#   Fold 2: train = years[1:(initial_window+1)], test = years[initial_window + 2]
#   ... and so on until end of data
#
# Args:
#   df             - Data frame with Tahun column (sorted)
#   initial_window - Number of years to use for initial training
#   horizon        - Number of steps ahead to forecast (typically 1)
#
# Returns:
#   List of folds, each containing:
#     - fold_id: Integer fold number
#     - train: Training data frame
#     - test: Test data frame
#     - info: Fold metadata
# ------------------------------------------------------------------------------
create_rolling_splits <- function(df, initial_window, horizon = 1) {
  if (!"Tahun" %in% names(df)) {
    stop("Data frame must have 'Tahun' column", call. = FALSE)
  }
  
  # Ensure sorted
  df <- df[order(df$Tahun), ]
  n <- nrow(df)
  
  # Validate we have enough data
  min_required <- initial_window + horizon
  if (n < min_required) {
    warning(sprintf(
      "Not enough data for rolling split: have %d, need at least %d (initial=%d + horizon=%d)",
      n, min_required, initial_window, horizon
    ))
    return(list())
  }
  
  # Generate folds
  folds <- list()
  fold_id <- 1
  
  for (train_end in initial_window:(n - horizon)) {
    test_start <- train_end + 1
    test_end <- min(train_end + horizon, n)
    
    train_df <- df[1:train_end, ]
    test_df <- df[test_start:test_end, ]
    
    # Validate no temporal leakage
    if (max(train_df$Tahun) >= min(test_df$Tahun)) {
      stop(sprintf(
        "Temporal leakage in fold %d: train max=%d >= test min=%d",
        fold_id, max(train_df$Tahun), min(test_df$Tahun)
      ), call. = FALSE)
    }
    
    fold_info <- list(
      fold_id = fold_id,
      split_type = "rolling",
      train_years = train_df$Tahun,
      test_years = test_df$Tahun,
      n_train = nrow(train_df),
      n_test = nrow(test_df),
      horizon = horizon
    )
    
    folds[[fold_id]] <- list(
      fold_id = fold_id,
      train = train_df,
      test = test_df,
      info = fold_info
    )
    
    fold_id <- fold_id + 1
  }
  
  message(sprintf(
    "Rolling splits: %d folds created (initial_window=%d, horizon=%d)",
    length(folds), initial_window, horizon
  ))
  
  folds
}

# ------------------------------------------------------------------------------
# get_split_summary: Get summary of all folds
# ------------------------------------------------------------------------------
get_split_summary <- function(folds) {
  if (length(folds) == 0) {
    return(data.frame())
  }
  
  summary_list <- lapply(folds, function(fold) {
    data.frame(
      fold_id = fold$fold_id,
      n_train = fold$info$n_train,
      n_test = fold$info$n_test,
      train_start = min(fold$info$train_years),
      train_end = max(fold$info$train_years),
      test_year = paste(fold$info$test_years, collapse = ","),
      stringsAsFactors = FALSE
    )
  })
  
  do.call(rbind, summary_list)
}

# ------------------------------------------------------------------------------
# validate_no_leakage: Validate that train and test have no temporal overlap
# ------------------------------------------------------------------------------
validate_no_leakage <- function(train_df, test_df) {
  if (nrow(train_df) == 0 || nrow(test_df) == 0) {
    return(TRUE)
  }
  
  train_max <- max(train_df$Tahun)
  test_min <- min(test_df$Tahun)
  
  if (train_max >= test_min) {
    stop(sprintf(
      "TEMPORAL LEAKAGE DETECTED! Train max year (%d) >= Test min year (%d)",
      train_max, test_min
    ), call. = FALSE)
  }
  
  invisible(TRUE)
}

# ------------------------------------------------------------------------------
# save_split_indices: Save split configuration for reproducibility
# ------------------------------------------------------------------------------
save_split_indices <- function(klaim_splits, iuran_splits, output_path = NULL) {
  if (is.null(output_path)) {
    output_path <- file.path("data", "processed", "split_indices.rds")
  }
  
  # Create directory if needed
  dir.create(dirname(output_path), recursive = TRUE, showWarnings = FALSE)
  
  # Prepare data structure (exclude actual data, just keep indices/years)
  save_data <- list(
    timestamp = Sys.time(),
    klaim = lapply(klaim_splits, function(fold) {
      list(
        fold_id = fold$fold_id,
        train_years = fold$info$train_years,
        test_years = fold$info$test_years
      )
    }),
    iuran = lapply(iuran_splits, function(fold) {
      list(
        fold_id = fold$fold_id,
        train_years = fold$info$train_years,
        test_years = fold$info$test_years
      )
    })
  )
  
  saveRDS(save_data, output_path)
  message(sprintf("Split indices saved to: %s", output_path))
  
  invisible(output_path)
}

# ------------------------------------------------------------------------------
# validate_processed_data: Validate integrity of processed datasets
# ------------------------------------------------------------------------------
validate_processed_data <- function(klaim_clean, iuran_clean) {
  message("\n=== Validating Processed Data ===\n")
  
  errors <- character(0)
  warnings <- character(0)
  
  # 1. Check row counts
  if (nrow(klaim_clean$full) != 12) {
    errors <- c(errors, sprintf("Klaim: expected 12 rows, got %d", nrow(klaim_clean$full)))
  }
  if (nrow(iuran_clean$full) != 11) {
    errors <- c(errors, sprintf("Iuran: expected 11 rows, got %d", nrow(iuran_clean$full)))
  }
  
  # 2. Check for unexpected NA in non-JP/JKP columns
  non_special_cols <- c("JHT", "JKK", "JKM", "BPJS")
  
  for (col in non_special_cols) {
    klaim_na <- sum(is.na(klaim_clean$full[[col]]))
    iuran_na <- sum(is.na(iuran_clean$full[[col]]))
    
    if (klaim_na > 0) {
      errors <- c(errors, sprintf("Klaim %s: unexpected %d NA values", col, klaim_na))
    }
    if (iuran_na > 0) {
      errors <- c(errors, sprintf("Iuran %s: unexpected %d NA values", col, iuran_na))
    }
  }
  
  # 3. Check expected NA patterns for JP and JKP
  # Klaim JP: 1 NA (2014)
  klaim_jp_na <- sum(is.na(klaim_clean$full$JP))
  if (klaim_jp_na != 1) {
    warnings <- c(warnings, sprintf("Klaim JP: expected 1 NA, got %d", klaim_jp_na))
  }
  
  # Klaim JKP: 7 NA (2014-2020)
  klaim_jkp_na <- sum(is.na(klaim_clean$full$JKP))
  if (klaim_jkp_na != 7) {
    warnings <- c(warnings, sprintf("Klaim JKP: expected 7 NA, got %d", klaim_jkp_na))
  }
  
  # Iuran JKP: 6 NA (2015-2020)
  iuran_jkp_na <- sum(is.na(iuran_clean$full$JKP))
  if (iuran_jkp_na != 6) {
    warnings <- c(warnings, sprintf("Iuran JKP: expected 6 NA, got %d", iuran_jkp_na))
  }
  
  # 4. Check data types
  for (col in c("JHT", "JKK", "JKM", "JP", "JKP", "BPJS")) {
    if (!is.numeric(klaim_clean$full[[col]])) {
      errors <- c(errors, sprintf("Klaim %s: not numeric", col))
    }
    if (!is.numeric(iuran_clean$full[[col]])) {
      errors <- c(errors, sprintf("Iuran %s: not numeric", col))
    }
  }
  
  # Report
  if (length(errors) > 0) {
    message("❌ ERRORS:")
    for (e in errors) message(sprintf("   - %s", e))
  }
  
  if (length(warnings) > 0) {
    message("⚠️  WARNINGS:")
    for (w in warnings) message(sprintf("   - %s", w))
  }
  
  if (length(errors) == 0 && length(warnings) == 0) {
    message("✅ All validations passed!")
  }
  
  message(sprintf(
    "\n📊 Summary: Processed %d rows for Klaim, %d rows for Iuran",
    nrow(klaim_clean$full), nrow(iuran_clean$full)
  ))
  
  list(
    valid = length(errors) == 0,
    errors = errors,
    warnings = warnings
  )
}

# ------------------------------------------------------------------------------
# Test functions
# ------------------------------------------------------------------------------
test_holdout_split <- function() {
  test_df <- data.frame(
    Tahun = 2014:2020,
    value = c(100, 200, 300, 400, 500, 600, 700)
  )
  
  result <- create_holdout_split(test_df, holdout_years = c(2019, 2020))
  
  stopifnot(nrow(result$train) == 5)  # 2014-2018
  stopifnot(nrow(result$test) == 2)   # 2019-2020
  stopifnot(max(result$train$Tahun) == 2018)
  stopifnot(min(result$test$Tahun) == 2019)
  
  message("test_holdout_split: PASSED")
  invisible(TRUE)
}

test_rolling_splits <- function() {
  test_df <- data.frame(
    Tahun = 2014:2020,
    value = c(100, 200, 300, 400, 500, 600, 700)
  )
  
  folds <- create_rolling_splits(test_df, initial_window = 3, horizon = 1)
  
  # With 7 data points, initial=3, horizon=1:
  # Fold 1: train 2014-2016, test 2017
  # Fold 2: train 2014-2017, test 2018
  # Fold 3: train 2014-2018, test 2019
  # Fold 4: train 2014-2019, test 2020
  
  stopifnot(length(folds) == 4)
  stopifnot(folds[[1]]$info$n_train == 3)
  stopifnot(folds[[1]]$info$n_test == 1)
  stopifnot(max(folds[[1]]$train$Tahun) == 2016)
  stopifnot(folds[[1]]$test$Tahun == 2017)
  
  message("test_rolling_splits: PASSED")
  invisible(TRUE)
}
