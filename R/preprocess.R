# ==============================================================================
# preprocess.R - Preprocessing & Missing Value Analysis
# ==============================================================================
# Functions:
#   - detect_missing(): Detect and map all missing values
#   - classify_missing(): Add reason categories and data availability info
#   - generate_missing_report(): Create comprehensive missing report
# ==============================================================================

# ------------------------------------------------------------------------------
# detect_missing: Detect NA values per target column and year
# ------------------------------------------------------------------------------
# Args:
#   df          - Data frame with Tahun and target columns
#   dataset_name - "klaim" or "iuran"
#
# Returns:
#   Data frame with columns: dataset, target, year, value, is_missing
# ------------------------------------------------------------------------------
detect_missing <- function(df, dataset_name) {
  # Get target columns (exclude Tahun)
  target_cols <- setdiff(names(df), "Tahun")
  
  # Pre-allocate vectors for efficiency
  n_rows <- nrow(df)
  n_targets <- length(target_cols)
  total_records <- n_rows * n_targets
  
  datasets <- character(total_records)
  targets <- character(total_records)
  years <- integer(total_records)
  values <- numeric(total_records)
  is_missing <- logical(total_records)
  
  idx <- 1
  for (target in target_cols) {
    for (i in seq_len(n_rows)) {
      datasets[idx] <- dataset_name
      targets[idx] <- target
      years[idx] <- df$Tahun[i]
      val <- df[[target]][i]
      values[idx] <- if (is.na(val)) NA_real_ else as.numeric(val)
      is_missing[idx] <- is.na(val)
      idx <- idx + 1
    }
  }
  
  # Create data frame from vectors (avoids list column issue)
  missing_df <- data.frame(
    dataset = datasets,
    target = targets,
    year = years,
    value = values,
    is_missing = is_missing,
    stringsAsFactors = FALSE
  )
  
  message(sprintf(
    "[%s] Detected %d missing values out of %d total cells",
    dataset_name,
    sum(missing_df$is_missing),
    nrow(missing_df)
  ))
  
  missing_df
}

# ------------------------------------------------------------------------------
# classify_missing: Add reason_category and data_points_available
# ------------------------------------------------------------------------------
# Reason categories:
#   - "structural" : Program did not exist (JKP before 2021)
#   - "legacy"     : Historical data unavailable (JP 2014 in Klaim)
#   - "none"       : No missing value
#
# Args:
#   missing_df - Output from detect_missing()
#
# Returns:
#   Data frame with additional columns: reason_category, data_points_available
# ------------------------------------------------------------------------------
classify_missing <- function(missing_df) {
  # Initialize new columns
  missing_df$reason_category <- "none"
  
  # Apply classification rules
  for (i in seq_len(nrow(missing_df))) {
    if (!missing_df$is_missing[i]) {
      next
    }
    
    dataset <- missing_df$dataset[i]
    target <- missing_df$target[i]
    year <- missing_df$year[i]
    
    # Rule 1: JKP before 2021 is structural (program didn't exist)
    if (target == "JKP" && year < 2021) {
      missing_df$reason_category[i] <- "structural"
      next
    }
    
    # Rule 2: JP in 2014 for Klaim is legacy (data not recorded)
    if (dataset == "klaim" && target == "JP" && year == 2014) {
      missing_df$reason_category[i] <- "legacy"
      next
    }
    
    # Rule 3: Any other missing is considered random
    missing_df$reason_category[i] <- "random"
  }
  
  # Calculate data_points_available per dataset-target combination
  missing_df$data_points_available <- NA_integer_
  
  # Get unique combinations using paste (avoid data.frame subsetting issues)
  combo_keys <- unique(paste(missing_df$dataset, missing_df$target, sep = "|"))
  for (key in combo_keys) {
    parts <- strsplit(key, "|", fixed = TRUE)[[1]]
    ds <- parts[1]
    tg <- parts[2]
    
    mask <- missing_df$dataset == ds & missing_df$target == tg
    available <- sum(!missing_df$is_missing[mask])
    missing_df$data_points_available[mask] <- available
  }
  
  # Log summary
  structural_count <- sum(missing_df$reason_category == "structural")
  legacy_count <- sum(missing_df$reason_category == "legacy")
  random_count <- sum(missing_df$reason_category == "random")
  
  message(sprintf(
    "Classification: structural=%d, legacy=%d, random=%d",
    structural_count, legacy_count, random_count
  ))
  
  missing_df
}

# ------------------------------------------------------------------------------
# get_missing_summary: Get summary of data availability per target
# ------------------------------------------------------------------------------
get_missing_summary <- function(classified_df) {
  # Get unique combinations using paste (avoid data.frame subsetting issues)
  combo_keys <- unique(paste(classified_df$dataset, classified_df$target, sep = "|"))
  
  summary_list <- list()
  for (key in combo_keys) {
    parts <- strsplit(key, "|", fixed = TRUE)[[1]]
    ds <- parts[1]
    tg <- parts[2]
    
    mask <- classified_df$dataset == ds & classified_df$target == tg
    subset_df <- classified_df[mask, ]
    
    total_years <- nrow(subset_df)
    available <- sum(!subset_df$is_missing)
    missing <- sum(subset_df$is_missing)
    
    # Get year range for available data
    available_years <- subset_df$year[!subset_df$is_missing]
    if (length(available_years) > 0) {
      year_range <- paste(min(available_years), max(available_years), sep = "-")
    } else {
      year_range <- "N/A"
    }
    
    # Determine warning level
    warning_level <- if (available < 6) {
      "CRITICAL"
    } else if (available < 8) {
      "WARNING"
    } else {
      "OK"
    }
    
    summary_list[[length(summary_list) + 1]] <- data.frame(
      dataset = ds,
      target = tg,
      total_years = total_years,
      available_points = available,
      missing_points = missing,
      available_year_range = year_range,
      warning_level = warning_level,
      stringsAsFactors = FALSE
    )
  }
  
  summary_df <- do.call(rbind, summary_list)
  rownames(summary_df) <- NULL
  
  # Print warnings
  critical <- summary_df[summary_df$warning_level == "CRITICAL", ]
  if (nrow(critical) > 0) {
    message("\n⚠️  CRITICAL: Following targets have < 6 data points:")
    for (k in seq_len(nrow(critical))) {
      message(sprintf(
        "   - %s/%s: only %d points (%s)",
        critical$dataset[k],
        critical$target[k],
        critical$available_points[k],
        critical$available_year_range[k]
      ))
    }
  }
  
  summary_df
}

# ------------------------------------------------------------------------------
# analyze_missing_values: Complete missing value analysis pipeline
# ------------------------------------------------------------------------------
# Main function that runs the full analysis
#
# Args:
#   klaim_df - Klaim dataset (cleaned)
#   iuran_df - Iuran dataset (cleaned)
#
# Returns:
#   List with: detailed (full missing map), summary (per target summary)
# ------------------------------------------------------------------------------
analyze_missing_values <- function(klaim_df, iuran_df) {
  message("\n=== Missing Value Analysis ===\n")
  
  # Step 1: Detect missing values
  message("Step 1: Detecting missing values...")
  klaim_missing <- detect_missing(klaim_df, "klaim")
  iuran_missing <- detect_missing(iuran_df, "iuran")
  
  # Combine both datasets
  all_missing <- rbind(klaim_missing, iuran_missing)
  
  # Step 2: Classify missing values
  message("\nStep 2: Classifying missing values...")
  classified <- classify_missing(all_missing)
  
  # Step 3: Generate summary
  message("\nStep 3: Generating summary...")
  summary_df <- get_missing_summary(classified)
  
  # Step 4: Generate and save missing report
  message("\nStep 4: Generating missing report...")
  report <- generate_missing_report(classified)
  
  # Return results
  list(
    detailed = classified,
    summary = summary_df,
    report = report
  )
}

# ------------------------------------------------------------------------------
# generate_missing_report: Create and save missing report CSV
# ------------------------------------------------------------------------------
# Adds chosen_treatment based on reason_category and treatment_rules
#
# Args:
#   classified_df - Output from classify_missing()
#   treatment_rules - List with STRICT_MODE, ALLOW_INTERPOLATION settings
#
# Returns:
#   Data frame with chosen_treatment column added
# ------------------------------------------------------------------------------
generate_missing_report <- function(classified_df, treatment_rules = NULL) {
  # Default treatment rules if not provided
  if (is.null(treatment_rules)) {
    treatment_rules <- list(
      STRICT_MODE = TRUE,
      ALLOW_INTERPOLATION = FALSE,
      JKP_MIN_POINTS_WARNING = 5
    )
  }
  
  # Filter to only missing values for main report
  missing_only <- classified_df[classified_df$is_missing, ]
  
  # Add chosen_treatment based on reason_category and rules
  missing_only$chosen_treatment <- vapply(
    seq_len(nrow(missing_only)),
    function(i) {
      reason <- missing_only$reason_category[i]
      target <- missing_only$target[i]
      
      if (reason == "structural") {
        # JKP pre-2021: model only from valid years
        return("exclude_years")
      } else if (reason == "legacy") {
        # JP 2014 in Klaim
        if (treatment_rules$ALLOW_INTERPOLATION) {
          return("interpolate_nearest")
        } else {
          return("exclude_year")
        }
      } else if (reason == "random") {
        if (treatment_rules$STRICT_MODE) {
          return("exclude_year")
        } else {
          return("interpolate_linear")
        }
      } else {
        return("no_action")
      }
    },
    character(1)
  )
  
  # Select columns for report
  report_df <- missing_only[, c(
    "dataset", "target", "year", "is_missing", 
    "reason_category", "data_points_available", "chosen_treatment"
  )]
  
  # Save to CSV
  output_dir <- file.path("data", "processed")
  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
  report_path <- file.path(output_dir, "missing_report.csv")
  write.csv(report_df, report_path, row.names = FALSE)
  
  # Print summary
  n_missing <- nrow(report_df)
  n_combinations <- length(unique(paste(report_df$dataset, report_df$target)))
  
  message(sprintf(
    "\n📋 %d missing values detected across %d target-year combinations",
    n_missing, n_combinations
  ))
  message(sprintf("📄 Missing report saved to: %s", report_path))
  
  # Print treatment breakdown
  treatment_table <- table(report_df$chosen_treatment)
  message("\n📊 Treatment breakdown:")
  for (treatment in names(treatment_table)) {
    message(sprintf("   - %s: %d", treatment, treatment_table[treatment]))
  }
  
  invisible(report_df)
}

# ------------------------------------------------------------------------------
# Test function
# ------------------------------------------------------------------------------
test_missing_detection <- function() {
  # Create sample data similar to Klaim
  test_df <- data.frame(
    Tahun = 2014:2016,
    JHT = c(100, 200, 300),
    JP = c(NA, 50, 60),      # Missing in 2014
    JKP = c(NA, NA, NA)      # All missing (structural)
  )
  
  result <- detect_missing(test_df, "test")
  
  # Check expectations
  stopifnot(nrow(result) == 9)  # 3 years * 3 targets
  stopifnot(sum(result$is_missing) == 4)  # 1 JP + 3 JKP
  
  classified <- classify_missing(result)
  stopifnot(sum(classified$reason_category == "structural") == 3)  # JKP
  
  message("test_missing_detection: PASSED")
  invisible(TRUE)
}

# ==============================================================================
# PHASE 4: CLEANING & NORMALIZATION FUNCTIONS
# ==============================================================================

# ------------------------------------------------------------------------------
# clean_dataset: Clean and prepare dataset for modeling
# ------------------------------------------------------------------------------
# Applies sorting, validation, and missing value handling based on treatment rules
#
# Args:
#   df             - Raw data frame from read_raw_csv()
#   dataset_name   - "klaim" or "iuran"
#   treatment_rules - List with STRICT_MODE, ALLOW_INTERPOLATION settings
#
# Returns:
#   List with:
#     - full: Complete data with NA preserved (for analysis)
#     - by_target: List of data frames filtered per target (NA excluded)
#     - metadata: Treatment info and statistics
# ------------------------------------------------------------------------------
clean_dataset <- function(df, dataset_name, treatment_rules = NULL) {
  # Default treatment rules
  if (is.null(treatment_rules)) {
    treatment_rules <- list(
      STRICT_MODE = TRUE,
      ALLOW_INTERPOLATION = FALSE
    )
  }
  
  message(sprintf("\n=== Cleaning dataset: %s ===", dataset_name))
  
  # Step 1: Sort by Tahun ascending
  df <- df[order(df$Tahun), ]
  rownames(df) <- NULL
  message("  [1] Sorted by Tahun ascending")
  
  # Step 2: Validate no duplicate years
  if (any(duplicated(df$Tahun))) {
    dup_years <- df$Tahun[duplicated(df$Tahun)]
    stop(sprintf(
      "Dataset '%s' has duplicate years: %s",
      dataset_name, paste(dup_years, collapse = ", ")
    ), call. = FALSE)
  }
  message("  [2] No duplicate years found")
  
  # Step 3: Get target columns
  target_cols <- setdiff(names(df), "Tahun")
  
  # Step 4: Create filtered versions per target (STRICT_MODE)
  by_target <- list()
  metadata <- list(
    dataset = dataset_name,
    treatment_rules = treatment_rules,
    targets = list()
  )
  
  for (target in target_cols) {
    # Get non-NA indices for this target
    valid_mask <- !is.na(df[[target]])
    target_df <- df[valid_mask, c("Tahun", target)]
    
    # Record metadata
    metadata$targets[[target]] <- list(
      total_years = nrow(df),
      available_years = sum(valid_mask),
      excluded_years = df$Tahun[!valid_mask],
      year_range = if (any(valid_mask)) {
        c(min(target_df$Tahun), max(target_df$Tahun))
      } else {
        c(NA, NA)
      }
    )
    
    by_target[[target]] <- target_df
    
    message(sprintf(
      "  [3] %s: %d/%d years available (%s)",
      target,
      sum(valid_mask),
      nrow(df),
      if (any(!valid_mask)) paste("excluded:", paste(df$Tahun[!valid_mask], collapse = ",")) else "none excluded"
    ))
  }
  
  message(sprintf("  Dataset '%s' cleaned successfully", dataset_name))
  
  # Return structure
  list(
    full = df,
    by_target = by_target,
    metadata = metadata
  )
}

# ------------------------------------------------------------------------------
# normalize_series: Normalize a numeric series
# ------------------------------------------------------------------------------
# Methods:
#   "minmax" - Scale to [0, 1] range: (x - min) / (max - min)
#   "zscore" - Standardize: (x - mean) / sd
#
# Args:
#   series  - Numeric vector to normalize
#   method  - "minmax" or "zscore"
#   params  - Optional pre-computed parameters (for applying to test data)
#
# Returns:
#   List with:
#     - normalized: Transformed values
#     - params: Parameters used (for inverse transform)
# ------------------------------------------------------------------------------
normalize_series <- function(series, method = "minmax", params = NULL) {
  # Handle edge cases
  if (length(series) == 0) {
    return(list(normalized = numeric(0), params = list()))
  }
  
  # Remove NA for parameter calculation
  valid_values <- series[!is.na(series)]
  
  if (length(valid_values) == 0) {
    return(list(normalized = rep(NA_real_, length(series)), params = list()))
  }
  
  if (method == "minmax") {
    # Compute parameters from data if not provided
    if (is.null(params)) {
      params <- list(
        method = "minmax",
        min = min(valid_values),
        max = max(valid_values)
      )
    }
    
    # Handle constant series (min == max)
    range_val <- params$max - params$min
    if (range_val == 0) {
      normalized <- rep(0.5, length(series))
      normalized[is.na(series)] <- NA_real_
    } else {
      normalized <- (series - params$min) / range_val
    }
    
  } else if (method == "zscore") {
    # Compute parameters from data if not provided
    if (is.null(params)) {
      params <- list(
        method = "zscore",
        mean = mean(valid_values),
        sd = sd(valid_values)
      )
    }
    
    # Handle constant series (sd == 0)
    if (params$sd == 0 || is.na(params$sd)) {
      normalized <- rep(0, length(series))
      normalized[is.na(series)] <- NA_real_
    } else {
      normalized <- (series - params$mean) / params$sd
    }
    
  } else {
    stop(sprintf("Unknown normalization method: %s", method), call. = FALSE)
  }
  
  list(
    normalized = normalized,
    params = params
  )
}

# ------------------------------------------------------------------------------
# inverse_normalize: Inverse transform normalized values back to original scale
# ------------------------------------------------------------------------------
inverse_normalize <- function(normalized_series, params) {
  if (length(normalized_series) == 0) {
    return(numeric(0))
  }
  
  method <- params$method
  
  if (method == "minmax") {
    range_val <- params$max - params$min
    if (range_val == 0) {
      # Original was constant
      original <- rep(params$min, length(normalized_series))
    } else {
      original <- normalized_series * range_val + params$min
    }
    
  } else if (method == "zscore") {
    if (params$sd == 0 || is.na(params$sd)) {
      original <- rep(params$mean, length(normalized_series))
    } else {
      original <- normalized_series * params$sd + params$mean
    }
    
  } else {
    stop(sprintf("Unknown normalization method: %s", method), call. = FALSE)
  }
  
  original
}

# ------------------------------------------------------------------------------
# create_scaler_params: Create scaler parameters for all targets in a dataset
# ------------------------------------------------------------------------------
# IMPORTANT: Should be computed ONLY from training data
#
# Args:
#   cleaned_data - Output from clean_dataset() (uses by_target list)
#   method       - "minmax" or "zscore"
#
# Returns:
#   Named list of scaler params per target: scaler_params$JHT, scaler_params$JKK, etc.
# ------------------------------------------------------------------------------
create_scaler_params <- function(cleaned_data, method = "minmax") {
  scaler_params <- list()
  
  for (target in names(cleaned_data$by_target)) {
    target_df <- cleaned_data$by_target[[target]]
    series <- target_df[[target]]
    
    result <- normalize_series(series, method = method)
    scaler_params[[target]] <- result$params
  }
  
  scaler_params
}

# ------------------------------------------------------------------------------
# normalize_target_data: Normalize a target series using pre-computed params
# ------------------------------------------------------------------------------
normalize_target_data <- function(series, target_name, scaler_params) {
  if (!target_name %in% names(scaler_params)) {
    stop(sprintf("No scaler params found for target: %s", target_name), call. = FALSE)
  }
  
  params <- scaler_params[[target_name]]
  normalize_series(series, method = params$method, params = params)$normalized
}

# ------------------------------------------------------------------------------
# denormalize_target_data: Inverse normalize using pre-computed params
# ------------------------------------------------------------------------------
denormalize_target_data <- function(normalized_series, target_name, scaler_params) {
  if (!target_name %in% names(scaler_params)) {
    stop(sprintf("No scaler params found for target: %s", target_name), call. = FALSE)
  }
  
  params <- scaler_params[[target_name]]
  inverse_normalize(normalized_series, params)
}

# ------------------------------------------------------------------------------
# save_clean_dataset: Save cleaned dataset to CSV
# ------------------------------------------------------------------------------
save_clean_dataset <- function(cleaned_data, dataset_name) {
  output_dir <- file.path("data", "processed")
  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
  
  # Save full dataset
  full_path <- file.path(output_dir, paste0(dataset_name, "_clean.csv"))
  write.csv(cleaned_data$full, full_path, row.names = FALSE)
  message(sprintf("Saved: %s", full_path))
  
  invisible(full_path)
}

# ------------------------------------------------------------------------------
# Test functions
# ------------------------------------------------------------------------------
test_normalize_series <- function() {
  # Test minmax
  series <- c(10, 20, 30, 40, 50)
  result <- normalize_series(series, method = "minmax")
  stopifnot(all.equal(result$normalized, c(0, 0.25, 0.5, 0.75, 1)))
  stopifnot(result$params$min == 10)
  stopifnot(result$params$max == 50)
  
  # Test inverse
  original <- inverse_normalize(result$normalized, result$params)
  stopifnot(all.equal(original, series))
  
  # Test zscore
  result_z <- normalize_series(series, method = "zscore")
  stopifnot(abs(mean(result_z$normalized)) < 1e-10)  # mean should be ~0
  stopifnot(abs(sd(result_z$normalized) - 1) < 1e-10)  # sd should be ~1
  
  # Test with NA
  series_na <- c(10, NA, 30, 40, 50)
  result_na <- normalize_series(series_na, method = "minmax")
  stopifnot(is.na(result_na$normalized[2]))
  
  # Test constant series
  const_series <- c(5, 5, 5, 5)
  result_const <- normalize_series(const_series, method = "minmax")
  stopifnot(all(result_const$normalized == 0.5))
  
  message("test_normalize_series: PASSED")
  invisible(TRUE)
}

test_clean_dataset <- function() {
  # Create test data
  test_df <- data.frame(
    Tahun = c(2016, 2014, 2015),  # Out of order
    JHT = c(300, 100, 200),
    JP = c(30, NA, 20)
  )
  
  result <- clean_dataset(test_df, "test")
  
  # Check sorting
  stopifnot(all(result$full$Tahun == c(2014, 2015, 2016)))
  
  # Check by_target filtering
  stopifnot(nrow(result$by_target$JHT) == 3)  # All rows
  stopifnot(nrow(result$by_target$JP) == 2)   # Excludes NA row
  stopifnot(all(result$by_target$JP$Tahun == c(2015, 2016)))
  
  # Check metadata
  stopifnot(result$metadata$targets$JP$available_years == 2)
  stopifnot(result$metadata$targets$JP$excluded_years == 2014)
  
  message("test_clean_dataset: PASSED")
  invisible(TRUE)
}
