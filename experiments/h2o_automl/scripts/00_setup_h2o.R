# ==============================================================================
# 00_setup_h2o.R - H2O Environment Setup and Initialization
# ==============================================================================
# This script sets up H2O for AutoML experiments.
#
# Prerequisites:
#   - Java 8+ installed and in PATH
#   - R packages: h2o
#
# Usage:
#   source("experiments/h2o_automl/scripts/00_setup_h2o.R")
# ==============================================================================

message("=== H2O AutoML Setup ===\n")

# Step 1: Install H2O if needed
if (!requireNamespace("h2o", quietly = TRUE)) {
  message("Installing h2o package...")
  install.packages("h2o", repos = "https://cloud.r-project.org")
}

# Step 2: Load H2O
library(h2o)

# Step 3: Initialize H2O cluster
message("\nStep 3: Initializing H2O cluster...")
message("This may take a moment on first run...\n")

# Check if H2O is already running
tryCatch({
  h2o.clusterInfo()
  message("H2O cluster already running, reusing existing cluster.")
}, error = function(e) {
  # Start new H2O instance
  h2o.init(
    nthreads = -1,           # Use all available cores
    max_mem_size = "2G"      # Allocate 2GB RAM
  )
})

# Disable progress bars for cleaner output
h2o.no_progress()

# Step 4: Load config for RANDOM_SEED
if (file.exists("R/config.R")) {
  source("R/config.R")
  message(sprintf("✓ Using RANDOM_SEED = %d", RANDOM_SEED))
} else {
  RANDOM_SEED <- 42
  message(sprintf("✓ Using default RANDOM_SEED = %d", RANDOM_SEED))
}

# Step 5: Display cluster info
message("\n### H2O Cluster Information ###\n")
cluster_info <- h2o.clusterInfo()

# Step 6: Summary
message("\n=== H2O Setup Complete ===")
# Access list safely (old versions might return atomic vector or different structure)
h2o_ver <- tryCatch(as.character(h2o.getVersion()), error = function(e) "Unknown")
message(sprintf("
Environment Summary:
  - H2O Version: %s
  - Random Seed: %d (pass to h2o.automl)
  - Status: Ready for AutoML training
",
  h2o_ver,
  RANDOM_SEED
))

message("Next step: Run 01_train_h2o.R to train models")

# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================

# ------------------------------------------------------------------------------
# check_h2o_running: Check if H2O is running
# ------------------------------------------------------------------------------
check_h2o_running <- function() {
  tryCatch({
    h2o.clusterIsUp()
  }, error = function(e) {
    FALSE
  })
}

# ------------------------------------------------------------------------------
# ensure_h2o_running: Start H2O if not running
# ------------------------------------------------------------------------------
ensure_h2o_running <- function(max_mem = "2G") {
  if (!check_h2o_running()) {
    message("Starting H2O cluster...")
    h2o.init(nthreads = -1, max_mem_size = max_mem)
    h2o.no_progress()
  }
}
