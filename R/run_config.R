# ==============================================================================
# run_config.R - Run Configuration and Logging
# ==============================================================================
# This file handles generating and saving run configuration for 
# reproducibility and experiment tracking.
#
# Usage:
#   source("R/run_config.R")
#   config <- generate_run_config()
#   save_run_config(config)
# ==============================================================================

# Load required package
if (!requireNamespace("jsonlite", quietly = TRUE)) {
  install.packages("jsonlite", repos = "https://cloud.r-project.org")
}
library(jsonlite)

# Source config if not already loaded
if (!exists("RANDOM_SEED")) {
  source("R/config.R")
}

# ==============================================================================
# RUN CONFIGURATION GENERATION
# ==============================================================================

# ------------------------------------------------------------------------------
# generate_run_config: Capture all configuration for a run
# ------------------------------------------------------------------------------
# Captures:
#   - Timestamp
#   - Session info (R version, packages)
#   - All parameters from config.R
#   - Missing treatment applied
#   - Any targets skipped
#
# Args:
#   skipped_targets - Optional list of skipped targets with reasons
#   additional_info - Optional list of additional info to include
#
# Returns:
#   List with complete run configuration
# ------------------------------------------------------------------------------
generate_run_config <- function(skipped_targets = NULL, 
                                 additional_info = NULL) {
  
  # Capture timestamp
  timestamp <- Sys.time()
  
  # Capture session info
  session <- sessionInfo()
  
  # Build config object
  config <- list(
    # Metadata
    run_id = format(timestamp, "%Y%m%d_%H%M%S"),
    timestamp = as.character(timestamp),
    timezone = Sys.timezone(),
    
    # System info
    system = list(
      r_version = paste(session$R.version$major, session$R.version$minor, sep = "."),
      platform = session$platform,
      os = session$running,
      locale = Sys.getlocale()
    ),
    
    # Packages
    packages = list(
      base = names(session$basePkgs),
      attached = sapply(session$otherPkgs, function(x) x$Version),
      loaded = sapply(session$loadedOnly, function(x) x$Version)
    ),
    
    # Project configuration from config.R
    config = list(
      random_seed = RANDOM_SEED,
      data_paths = list(
        raw = DATA_RAW_PATH,
        processed = DATA_PROCESSED_PATH,
        results = RESULTS_PATH
      ),
      target_columns = TARGET_COLUMNS,
      datasets = DATASETS,
      evaluation = list(
        mode = EVAL_MODE,
        holdout_years = HOLDOUT_YEARS,
        rolling_initial_window = ROLLING_INITIAL_WINDOW
      ),
      models = list(
        lstm = list(
          lookback = LSTM_LOOKBACK,
          units = LSTM_UNITS
        ),
        h2o = list(
          max_runtime_secs = H2O_MAX_RUNTIME_SECS
        )
      ),
      treatment_rules = TREATMENT_RULES
    ),
    
    # Skipped targets
    skipped_targets = skipped_targets,
    
    # Additional info
    additional = additional_info
  )
  
  config
}

# ------------------------------------------------------------------------------
# save_run_config: Save run configuration to JSON
# ------------------------------------------------------------------------------
save_run_config <- function(config, filename = NULL) {
  results_dir <- file.path("results")
  dir.create(results_dir, recursive = TRUE, showWarnings = FALSE)
  
  if (is.null(filename)) {
    filename <- sprintf("run_config_%s.json", config$run_id)
  }
  
  filepath <- file.path(results_dir, filename)
  
  # Convert to JSON with pretty printing
  json_content <- toJSON(config, pretty = TRUE, auto_unbox = TRUE)
  writeLines(json_content, filepath)
  
  message(sprintf("✓ Run config saved: %s", filepath))
  
  invisible(filepath)
}

# ------------------------------------------------------------------------------
# load_run_config: Load run configuration from JSON
# ------------------------------------------------------------------------------
load_run_config <- function(filepath) {
  if (!file.exists(filepath)) {
    stop(sprintf("Config file not found: %s", filepath), call. = FALSE)
  }
  
  config <- fromJSON(filepath, simplifyVector = FALSE)
  config
}

# ==============================================================================
# EXPERIMENT LOGGING
# ==============================================================================

# ------------------------------------------------------------------------------
# log_experiment_start: Create log entry for experiment start
# ------------------------------------------------------------------------------
log_experiment_start <- function(experiment_name, description = NULL) {
  log_entry <- list(
    event = "experiment_start",
    experiment = experiment_name,
    description = description,
    timestamp = as.character(Sys.time())
  )
  
  message(sprintf("\n=== Experiment Started: %s ===", experiment_name))
  message(sprintf("Timestamp: %s", log_entry$timestamp))
  
  invisible(log_entry)
}

# ------------------------------------------------------------------------------
# log_experiment_end: Create log entry for experiment end
# ------------------------------------------------------------------------------
log_experiment_end <- function(experiment_name, 
                                results_summary = NULL, 
                                start_time = NULL) {
  end_time <- Sys.time()
  
  log_entry <- list(
    event = "experiment_end",
    experiment = experiment_name,
    timestamp = as.character(end_time),
    duration_seconds = if (!is.null(start_time)) {
      as.numeric(difftime(end_time, start_time, units = "secs"))
    } else {
      NULL
    },
    results_summary = results_summary
  )
  
  message(sprintf("\n=== Experiment Completed: %s ===", experiment_name))
  if (!is.null(log_entry$duration_seconds)) {
    message(sprintf("Duration: %.1f seconds", log_entry$duration_seconds))
  }
  
  invisible(log_entry)
}

# ==============================================================================
# MISSING VALUES DOCUMENTATION
# ==============================================================================

# ------------------------------------------------------------------------------
# get_missing_treatment_summary: Get summary of missing value treatments
# ------------------------------------------------------------------------------
get_missing_treatment_summary <- function() {
  report_path <- file.path(DATA_PROCESSED_PATH, "missing_report.csv")
  
  if (!file.exists(report_path)) {
    message("Missing report not found at: ", report_path)
    return(NULL)
  }
  
  report <- read.csv(report_path, stringsAsFactors = FALSE)
  
  # Summarize by treatment
  treatment_summary <- aggregate(
    is_missing ~ dataset + target + reason_category + chosen_treatment,
    data = report[report$is_missing == TRUE, ],
    FUN = length
  )
  names(treatment_summary)[names(treatment_summary) == "is_missing"] <- "count"
  
  treatment_summary
}

# ==============================================================================
# FULL RUN LOGGING PIPELINE
# ==============================================================================

# ------------------------------------------------------------------------------
# create_full_run_log: Generate complete run log with all info
# ------------------------------------------------------------------------------
create_full_run_log <- function(model_results = NULL, 
                                 skipped_targets = NULL) {
  
  # Get missing treatment summary
  missing_summary <- get_missing_treatment_summary()
  
  # Build skipped targets info
  skipped_info <- NULL
  if (!is.null(skipped_targets)) {
    skipped_info <- lapply(names(skipped_targets), function(name) {
      list(
        target = name,
        reason = skipped_targets[[name]]
      )
    })
  }
  
  # Generate full config
  config <- generate_run_config(
    skipped_targets = skipped_info,
    additional_info = list(
      missing_treatment_summary = missing_summary,
      model_results = model_results
    )
  )
  
  # Save config
  save_run_config(config)
  
  invisible(config)
}

# ==============================================================================
# TEST FUNCTIONS
# ==============================================================================

test_run_config <- function() {
  config <- generate_run_config()
  
  stopifnot(!is.null(config$run_id))
  stopifnot(!is.null(config$timestamp))
  stopifnot(!is.null(config$config$random_seed))
  stopifnot(config$config$random_seed == RANDOM_SEED)
  
  message("test_run_config: PASSED")
  invisible(TRUE)
}
