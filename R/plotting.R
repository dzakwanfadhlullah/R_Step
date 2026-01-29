# ==============================================================================
# plotting.R - Visualization Functions for Forecasting Results
# ==============================================================================
# This file contains all plotting functions for visualizing model predictions,
# errors, and model comparisons.
#
# Dependencies: ggplot2, scales, patchwork
#
# Usage:
#   source("R/plotting.R")
# ==============================================================================

# Install dependencies if needed
install_plot_deps <- function() {
  pkgs <- c("ggplot2", "scales", "patchwork")
  for (pkg in pkgs) {
    if (!requireNamespace(pkg, quietly = TRUE)) {
      install.packages(pkg, repos = "https://cloud.r-project.org")
    }
  }
}

# Load dependencies
install_plot_deps()
library(ggplot2)
library(scales)
library(patchwork)

# ==============================================================================
# THEME AND COLORS
# ==============================================================================

# Academic clean theme
theme_forecast <- function(base_size = 12) {
  theme_minimal(base_size = base_size) +
    theme(
      plot.title = element_text(face = "bold", hjust = 0.5, size = base_size + 2),
      plot.subtitle = element_text(hjust = 0.5, color = "gray40"),
      axis.title = element_text(face = "bold"),
      axis.text = element_text(color = "gray30"),
      legend.position = "bottom",
      legend.title = element_text(face = "bold"),
      panel.grid.minor = element_blank(),
      panel.grid.major = element_line(color = "gray90"),
      plot.background = element_rect(fill = "white", color = NA),
      panel.background = element_rect(fill = "white", color = NA)
    )
}

# Color palette
COLORS <- list(
  actual = "#2C3E50",      # Dark blue-gray
  predicted = "#E74C3C",   # Red
  train_shade = "#E8F6F3", # Light teal
  test_shade = "#FDEBD0",  # Light orange
  error_pos = "#27AE60",   # Green
  error_neg = "#C0392B",   # Dark red
  lstm = "#3498DB",        # Blue
  h2o = "#9B59B6"          # Purple
)

# ==============================================================================
# ACTUAL VS PREDICTED PLOTS
# ==============================================================================

# ------------------------------------------------------------------------------
# plot_actual_vs_pred: Plot actual vs predicted values over time
# ------------------------------------------------------------------------------
# Args:
#   preds_df    - Data frame with columns: year, actual, predicted
#   model       - Model name (e.g., "lstm", "h2o")
#   dataset     - Dataset name (e.g., "klaim", "iuran")
#   target      - Target variable name (e.g., "JHT")
#   holdout_years - Years in test set (for shading)
#   save_path   - Optional path to save the plot
#   dpi         - Resolution for saving (default 300)
#
# Returns:
#   ggplot object
# ------------------------------------------------------------------------------
plot_actual_vs_pred <- function(preds_df,
                                 model,
                                 dataset,
                                 target,
                                 holdout_years = NULL,
                                 save_path = NULL,
                                 dpi = 300) {
  
  # Validate input
  required_cols <- c("year", "actual", "predicted")
  if (!all(required_cols %in% names(preds_df))) {
    stop("preds_df must have columns: year, actual, predicted")
  }
  
  # Prepare data
  preds_df <- preds_df[order(preds_df$year), ]
  
  # Determine train/test split for shading
  if (is.null(holdout_years)) {
    # Try to detect from data
    source("R/config.R", local = TRUE)
    if (exists("HOLDOUT_YEARS")) {
      holdout_years <- HOLDOUT_YEARS
    }
  }
  
  # Calculate y-axis limits with padding
  y_min <- min(c(preds_df$actual, preds_df$predicted), na.rm = TRUE)
  y_max <- max(c(preds_df$actual, preds_df$predicted), na.rm = TRUE)
  y_range <- y_max - y_min
  y_limits <- c(y_min - 0.1 * y_range, y_max + 0.1 * y_range)
  
  # Build plot
  p <- ggplot(preds_df, aes(x = year))
  
  # Add shading for train/test periods
  if (!is.null(holdout_years) && length(holdout_years) > 0) {
    test_start <- min(holdout_years) - 0.5
    test_end <- max(holdout_years) + 0.5
    train_end <- test_start
    train_start <- min(preds_df$year) - 0.5
    
    p <- p +
      annotate("rect", 
               xmin = train_start, xmax = train_end,
               ymin = -Inf, ymax = Inf,
               fill = COLORS$train_shade, alpha = 0.5) +
      annotate("rect",
               xmin = test_start, xmax = test_end,
               ymin = -Inf, ymax = Inf,
               fill = COLORS$test_shade, alpha = 0.5)
  }
  
  # Add lines
  p <- p +
    geom_line(aes(y = actual, color = "Actual"), 
              size = 1.2, na.rm = TRUE) +
    geom_point(aes(y = actual, color = "Actual"), 
               size = 3, na.rm = TRUE) +
    geom_line(aes(y = predicted, color = "Predicted"), 
              size = 1.2, linetype = "dashed", na.rm = TRUE) +
    geom_point(aes(y = predicted, color = "Predicted"), 
               size = 3, shape = 17, na.rm = TRUE)
  
  # Styling
  p <- p +
    scale_color_manual(
      name = "",
      values = c("Actual" = COLORS$actual, "Predicted" = COLORS$predicted)
    ) +
    scale_x_continuous(breaks = preds_df$year) +
    scale_y_continuous(labels = comma_format()) +
    coord_cartesian(ylim = y_limits) +
    labs(
      title = sprintf("%s Forecast: %s - %s", 
                      toupper(model), toupper(dataset), target),
      subtitle = if (!is.null(holdout_years)) {
        sprintf("Training: -%d | Test: %s", 
                min(holdout_years) - 1,
                paste(holdout_years, collapse = ", "))
      } else { NULL },
      x = "Year",
      y = "Value"
    ) +
    theme_forecast()
  
  # Save if path provided
  if (!is.null(save_path)) {
    dir.create(dirname(save_path), recursive = TRUE, showWarnings = FALSE)
    ggsave(save_path, p, width = 10, height = 6, dpi = dpi, bg = "white")
    message(sprintf("✓ Plot saved: %s", save_path))
  }
  
  p
}

# ==============================================================================
# ERROR ANALYSIS PLOTS
# ==============================================================================

# ------------------------------------------------------------------------------
# plot_errors: Error analysis with time series and distribution
# ------------------------------------------------------------------------------
plot_errors <- function(preds_df,
                         model,
                         dataset,
                         target,
                         save_path = NULL,
                         dpi = 300) {
  
  # Calculate errors
  preds_df$error <- preds_df$actual - preds_df$predicted
  preds_df$error_pct <- (preds_df$error / preds_df$actual) * 100
  
  # Check for systematic bias
  mean_error <- mean(preds_df$error, na.rm = TRUE)
  all_positive <- all(preds_df$error > 0, na.rm = TRUE)
  all_negative <- all(preds_df$error < 0, na.rm = TRUE)
  
  bias_label <- if (all_positive) {
    "⚠️ Systematic under-prediction"
  } else if (all_negative) {
    "⚠️ Systematic over-prediction"
  } else {
    sprintf("Mean error: %.2f", mean_error)
  }
  
  # Plot 1: Error time series
  p1 <- ggplot(preds_df, aes(x = year, y = error)) +
    geom_hline(yintercept = 0, linetype = "dashed", color = "gray50") +
    geom_col(aes(fill = error > 0), width = 0.7) +
    geom_text(aes(label = round(error, 0)), vjust = -0.5, size = 3) +
    scale_fill_manual(
      values = c("TRUE" = COLORS$error_pos, "FALSE" = COLORS$error_neg),
      guide = "none"
    ) +
    scale_x_continuous(breaks = preds_df$year) +
    scale_y_continuous(labels = comma_format()) +
    labs(
      title = "Prediction Errors Over Time",
      subtitle = bias_label,
      x = "Year",
      y = "Error (Actual - Predicted)"
    ) +
    theme_forecast()
  
  # Plot 2: Error distribution
  if (nrow(preds_df) >= 3) {
    p2 <- ggplot(preds_df, aes(x = error)) +
      geom_histogram(aes(y = after_stat(density)), 
                     bins = min(10, nrow(preds_df)),
                     fill = COLORS$actual, alpha = 0.7) +
      geom_vline(xintercept = 0, linetype = "dashed", color = "red") +
      geom_vline(xintercept = mean_error, linetype = "solid", color = "blue") +
      labs(
        title = "Error Distribution",
        subtitle = sprintf("Mean: %.2f | SD: %.2f", 
                           mean_error, sd(preds_df$error, na.rm = TRUE)),
        x = "Error",
        y = "Density"
      ) +
      theme_forecast()
  } else {
    # Not enough data for histogram
    p2 <- ggplot() +
      annotate("text", x = 0.5, y = 0.5, 
               label = "Not enough data\nfor distribution plot",
               size = 5, color = "gray50") +
      theme_void() +
      labs(title = "Error Distribution")
  }
  
  # Combine plots
  combined <- p1 / p2 +
    plot_annotation(
      title = sprintf("Error Analysis: %s - %s - %s", 
                      toupper(model), toupper(dataset), target),
      theme = theme(plot.title = element_text(face = "bold", hjust = 0.5))
    )
  
  # Save if path provided
  if (!is.null(save_path)) {
    dir.create(dirname(save_path), recursive = TRUE, showWarnings = FALSE)
    ggsave(save_path, combined, width = 10, height = 10, dpi = dpi, bg = "white")
    message(sprintf("✓ Plot saved: %s", save_path))
  }
  
  combined
}

# ==============================================================================
# COMPARISON PLOTS
# ==============================================================================

# ------------------------------------------------------------------------------
# plot_model_comparison: Bar chart comparing models by metric
# ------------------------------------------------------------------------------
plot_model_comparison <- function(metrics_df,
                                   metric = "rmse",
                                   save_path = NULL,
                                   dpi = 300) {
  
  if (!all(c("target", "model", metric) %in% names(metrics_df))) {
    stop(sprintf("metrics_df must have columns: target, model, %s", metric))
  }
  
  # Build plot
  p <- ggplot(metrics_df, aes(x = target, y = .data[[metric]], fill = model)) +
    geom_col(position = position_dodge(width = 0.8), width = 0.7) +
    geom_text(aes(label = round(.data[[metric]], 2)),
              position = position_dodge(width = 0.8),
              vjust = -0.5, size = 3) +
    scale_fill_manual(
      name = "Model",
      values = c("lstm" = COLORS$lstm, "h2o" = COLORS$h2o, 
                 "h2o_automl" = COLORS$h2o)
    ) +
    scale_y_continuous(expand = expansion(mult = c(0, 0.15))) +
    labs(
      title = sprintf("Model Comparison: %s", toupper(metric)),
      x = "Target Variable",
      y = toupper(metric)
    ) +
    theme_forecast() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))
  
  # Save if path provided
  if (!is.null(save_path)) {
    dir.create(dirname(save_path), recursive = TRUE, showWarnings = FALSE)
    ggsave(save_path, p, width = 10, height = 6, dpi = dpi, bg = "white")
    message(sprintf("✓ Plot saved: %s", save_path))
  }
  
  p
}

# ------------------------------------------------------------------------------
# plot_leaderboard: Horizontal bar chart of H2O leaderboard
# ------------------------------------------------------------------------------
plot_leaderboard <- function(leaderboard_df,
                              metric = "rmse",
                              top_n = 10,
                              save_path = NULL,
                              dpi = 300) {
  
  # Select top N models
  if (nrow(leaderboard_df) > top_n) {
    leaderboard_df <- head(leaderboard_df, top_n)
  }
  
  # Get metric column (H2O uses different naming)
  metric_col <- if (metric %in% names(leaderboard_df)) {
    metric
  } else if (toupper(metric) %in% names(leaderboard_df)) {
    toupper(metric)
  } else {
    # Try to find similar column
    metric_cols <- grep(metric, names(leaderboard_df), ignore.case = TRUE, value = TRUE)
    if (length(metric_cols) > 0) metric_cols[1] else names(leaderboard_df)[2]
  }
  
  # Shorten model IDs for display
  leaderboard_df$model_short <- sapply(
    leaderboard_df$model_id,
    function(x) {
      parts <- strsplit(x, "_")[[1]]
      if (length(parts) >= 2) paste(parts[1:2], collapse = "_") else x
    }
  )
  
  # Build plot
  p <- ggplot(leaderboard_df, 
              aes(x = reorder(model_short, -.data[[metric_col]]), 
                  y = .data[[metric_col]])) +
    geom_col(fill = COLORS$h2o, width = 0.7) +
    geom_text(aes(label = round(.data[[metric_col]], 4)), 
              hjust = -0.1, size = 3) +
    coord_flip() +
    scale_y_continuous(expand = expansion(mult = c(0, 0.2))) +
    labs(
      title = sprintf("H2O AutoML Leaderboard (Top %d)", nrow(leaderboard_df)),
      subtitle = sprintf("Ranked by %s", toupper(metric)),
      x = "Model",
      y = toupper(metric)
    ) +
    theme_forecast() +
    theme(axis.text.y = element_text(size = 9))
  
  # Save if path provided
  if (!is.null(save_path)) {
    dir.create(dirname(save_path), recursive = TRUE, showWarnings = FALSE)
    ggsave(save_path, p, width = 10, height = 8, dpi = dpi, bg = "white")
    message(sprintf("✓ Plot saved: %s", save_path))
  }
  
  p
}

# ==============================================================================
# GENERATE ALL PLOTS
# ==============================================================================

# ------------------------------------------------------------------------------
# generate_all_plots: Generate all plots for a complete run
# ------------------------------------------------------------------------------
generate_all_plots <- function(preds_all,
                                metrics_all = NULL,
                                output_dir = "results/plots",
                                dpi = 300) {
  
  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
  
  plot_count <- 0
  empty_count <- 0
  
  # Get unique combinations
  if (!is.data.frame(preds_all)) {
    message("preds_all should be a data frame with all predictions")
    return(NULL)
  }
  
  required_cols <- c("dataset", "target", "model", "year", "actual", "predicted")
  if (!all(required_cols %in% names(preds_all))) {
    stop(sprintf("preds_all must have columns: %s", paste(required_cols, collapse = ", ")))
  }
  
  # Get unique combinations
  combinations <- unique(preds_all[, c("dataset", "target", "model")])
  
  message(sprintf("\n=== Generating plots for %d combinations ===\n", nrow(combinations)))
  
  for (i in seq_len(nrow(combinations))) {
    ds <- combinations$dataset[i]
    tgt <- combinations$target[i]
    mdl <- combinations$model[i]
    
    # Filter data
    mask <- preds_all$dataset == ds & 
            preds_all$target == tgt & 
            preds_all$model == mdl
    subset_df <- preds_all[mask, ]
    
    if (nrow(subset_df) == 0) {
      empty_count <- empty_count + 1
      next
    }
    
    # Generate actual vs predicted plot
    avp_path <- file.path(
      output_dir, 
      sprintf("actual_vs_pred_%s_%s_%s.png", mdl, ds, tgt)
    )
    tryCatch({
      plot_actual_vs_pred(subset_df, mdl, ds, tgt, save_path = avp_path, dpi = dpi)
      plot_count <- plot_count + 1
    }, error = function(e) {
      warning(sprintf("Failed to generate actual_vs_pred for %s/%s/%s: %s", 
                      mdl, ds, tgt, e$message))
    })
    
    # Generate error plot
    err_path <- file.path(
      output_dir,
      sprintf("error_%s_%s_%s.png", mdl, ds, tgt)
    )
    tryCatch({
      plot_errors(subset_df, mdl, ds, tgt, save_path = err_path, dpi = dpi)
      plot_count <- plot_count + 1
    }, error = function(e) {
      warning(sprintf("Failed to generate error plot for %s/%s/%s: %s",
                      mdl, ds, tgt, e$message))
    })
  }
  
  # Generate comparison plot if metrics provided
  if (!is.null(metrics_all) && nrow(metrics_all) > 0) {
    comp_path <- file.path(output_dir, "comparison_rmse.png")
    tryCatch({
      plot_model_comparison(metrics_all, "rmse", save_path = comp_path, dpi = dpi)
      plot_count <- plot_count + 1
    }, error = function(e) {
      warning(sprintf("Failed to generate comparison plot: %s", e$message))
    })
  }
  
  # Summary
  message(sprintf("\n=== Plot Generation Complete ==="))
  message(sprintf("  Generated: %d plots", plot_count))
  message(sprintf("  Empty/skipped: %d", empty_count))
  message(sprintf("  Output directory: %s", output_dir))
  
  invisible(list(
    generated = plot_count,
    skipped = empty_count,
    output_dir = output_dir
  ))
}

# ==============================================================================
# TEST FUNCTIONS
# ==============================================================================

test_plot_actual_vs_pred <- function() {
  test_df <- data.frame(
    year = 2020:2025,
    actual = c(100, 120, 140, 160, 180, 200),
    predicted = c(105, 118, 145, 155, 185, 195)
  )
  
  p <- plot_actual_vs_pred(test_df, "lstm", "klaim", "JHT", 
                            holdout_years = c(2024, 2025))
  
  stopifnot(inherits(p, "ggplot"))
  message("test_plot_actual_vs_pred: PASSED")
  invisible(TRUE)
}

test_plot_errors <- function() {
  test_df <- data.frame(
    year = 2020:2025,
    actual = c(100, 120, 140, 160, 180, 200),
    predicted = c(105, 118, 145, 155, 185, 195)
  )
  
  p <- plot_errors(test_df, "lstm", "klaim", "JHT")
  
  stopifnot(inherits(p, "patchwork"))
  message("test_plot_errors: PASSED")
  invisible(TRUE)
}
