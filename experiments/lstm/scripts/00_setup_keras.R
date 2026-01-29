# ==============================================================================
# 00_setup_keras.R - Keras/TensorFlow Environment Setup
# ==============================================================================
# IMPORTANT: Run this script ONCE to set up the deep learning environment.
# This may take several minutes to download and install TensorFlow.
#
# Prerequisites:
#   - Python 3.8+ installed
#   - R packages: keras, tensorflow, reticulate
#
# Usage:
#   source("experiments/lstm/scripts/00_setup_keras.R")
# ==============================================================================

message("=== Keras/TensorFlow Setup (Keras3 Edition) ===\n")

# Step 1: Install R packages if needed
install_if_missing <- function(pkg) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    message(sprintf("Installing %s...", pkg))
    install.packages(pkg, repos = "https://cloud.r-project.org")
  }
}

install_if_missing("reticulate")
install_if_missing("keras3")
install_if_missing("tensorflow")

# Step 2: Load packages
library(reticulate)
library(keras3)
library(tensorflow)

# Step 3: Check and Setup Python Environment
setup_python_env <- function() {
  message("\nStep 3: Setting up Python and TensorFlow...")
  
  # Try to find existing python with tensorflow
  py_path <- Sys.which("python")
  if (py_path != "") {
    message(sprintf("Found system python: %s", py_path))
    reticulate::use_python(py_path, required = TRUE)
    
    # Try to import tensorflow
    res <- tryCatch({
      tf <- import("tensorflow")
      message(sprintf("TensorFlow v%s already available in system python.", tf$`__version__`))
      return(TRUE)
    }, error = function(e) {
      message("TensorFlow not found in system python. Attempting pip install...")
      system2(py_path, c("-m", "pip", "install", "tensorflow"))
      return(FALSE)
    })
    
    if (res) return(TRUE)
    
    # Check again after pip install
    tryCatch({
      tf <- import("tensorflow")
      message(sprintf("TensorFlow v%s installed successfully via pip.", tf$`__version__`))
      return(TRUE)
    }, error = function(e) {
      message("Pip install failed or tensorflow still not importable.")
    })
  }
  
  # If all else fails, try the standard R way
  message("Attempting keras3::install_keras()...")
  tryCatch({
    keras3::install_keras()
    return(TRUE)
  }, error = function(e) {
    message(sprintf("install_keras failed: %s", e$message))
    return(FALSE)
  })
}

success <- setup_python_env()

if (!success) {
  stop("CRITICAL: Failed to setup Keras/TensorFlow environment.", call. = FALSE)
}

# Step 4: Verify installation
message("\n### Verification ###\n")

# Test TensorFlow
tf <- tensorflow::tf
tf_version <- tf$`__version__`
message(sprintf("✓ TensorFlow version: %s", tf_version))

# Test basic operation
test_tensor <- tf$constant("Hello TensorFlow")
message(sprintf("✓ Test tensor: %s", as.character(test_tensor$numpy())))

# Test Keras
keras_version <- keras3::keras$`__version__`
message(sprintf("✓ Keras version: %s", keras_version))

# Check GPU availability
gpus <- tf$config$list_physical_devices("GPU")
if (length(gpus) > 0) {
  message(sprintf("✓ GPU available: %d device(s)", length(gpus)))
} else {
  message("✓ Running on CPU (no GPU detected)")
}

# Step 5: Set up reproducibility
message("\n### Setting up reproducibility ###\n")

# Load config
if (file.exists("R/config.R")) {
  source("R/config.R")
  message(sprintf("✓ Using RANDOM_SEED = %d", RANDOM_SEED))
} else {
  RANDOM_SEED <- 42
  message(sprintf("✓ Using default RANDOM_SEED = %d", RANDOM_SEED))
}

# Set R seed
set.seed(RANDOM_SEED)
message("✓ R random seed set")

# Set TensorFlow seed
tf$random$set_seed(as.integer(RANDOM_SEED))
message("✓ TensorFlow random seed set")

# Set Python random seed via reticulate
py_run_string(sprintf("
import random
import numpy as np
random.seed(%d)
np.random.seed(%d)
", RANDOM_SEED, RANDOM_SEED))
message("✓ Python random seeds set")

# Step 6: Summary
message("\n=== Setup Complete ===")
message(sprintf("
Environment Summary:
  - TensorFlow: v%s
  - Keras: v%s
  - Device: %s
  - Random Seed: %d
  - Status: Ready for LSTM training
", 
  tf_version, 
  keras_version,
  ifelse(length(gpus) > 0, "GPU", "CPU"),
  RANDOM_SEED
))

message("Next step: Run 01_train_lstm.R to train models")
