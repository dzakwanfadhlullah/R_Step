# ==============================================================================
# plotting_forecast.R - Forecast Visualization Functions
# ==============================================================================
# This file contains plotting functions specifically for forecasting output:
#   - Combined timeline plots (historical + forecast)
#   - Per-dataset plots (all targets in one chart)
#   - Per-target plots (Iuran + Klaim in one chart)
#
# Dependencies: ggplot2, scales
# ==============================================================================

library(ggplot2)
library(scales)

# ==============================================================================
# COLOR PALETTES
# ==============================================================================

# Colors for targets (matches Excel reference)
TARGET_COLORS <- c(
  "JHT" = "#4472C4",   # Blue
  "JKK" = "#ED7D31",   # Orange
  "JKM" = "#A5A5A5",   # Gray
  "JP"  = "#FFC000",   # Yellow
  "JKP" = "#5B9BD5",   # Light Blue
  "BPJS" = "#70AD47"   # Green
)

# Colors for datasets
DATASET_COLORS <- c(
  "iuran" = "#4472C4",  # Blue
  "klaim" = "#A5A5A5"   # Gray
)

# ==============================================================================
# THEME
# ==============================================================================

theme_forecast_clean <- function(base_size = 11) {
  theme_minimal(base_size = base_size) +
    theme(
      plot.title = element_text(face = "bold", hjust = 0.5, size = base_size + 4),
      axis.title = element_text(face = "bold"),
      axis.text.x = element_text(angle = 45, hjust = 1),
      legend.position = "bottom",
      legend.title = element_blank(),
      panel.grid.minor = element_blank(),
      plot.background = element_rect(fill = "white", color = NA)
    )
}

# ==============================================================================
# PLOT: PER DATASET (All targets in one chart)
# ==============================================================================

#' Plot all targets for a single dataset
#' @param forecast_data Combined data with columns: year, target, actual, predicted
#' @param dataset_name Name of dataset ("iuran" or "klaim")
#' @param title_prefix Optional title prefix
#' @param save_path Path to save the plot
plot_dataset_all_targets <- function(forecast_data, 
                                       dataset_name,
                                       title_prefix = "",
                                       save_path = NULL) {
  
  # Prepare data: combine actual and predicted into single value column
  plot_data <- forecast_data
  plot_data$value <- ifelse(is.na(plot_data$actual), 
                             plot_data$predicted, 
                             plot_data$actual)
  plot_data$type <- ifelse(is.na(plot_data$actual), "Forecast", "Actual")
  
  # Create year labels with "F" suffix for forecast
  plot_data$year_label <- ifelse(
    plot_data$type == "Forecast",
    paste0(plot_data$year, " F"),
    as.character(plot_data$year)
  )
  
  # Get unique years in order
  all_years <- sort(unique(plot_data$year))
  year_labels <- sapply(all_years, function(y) {
    if (y %in% plot_data$year[plot_data$type == "Forecast"]) {
      paste0(y, " F")
    } else {
      as.character(y)
    }
  })
  
  # Build plot
  p <- ggplot(plot_data, aes(x = year, y = value, color = target)) +
    geom_line(size = 1.2) +
    geom_point(size = 2) +
    scale_color_manual(values = TARGET_COLORS) +
    scale_x_continuous(breaks = all_years, labels = year_labels) +
    scale_y_continuous(labels = label_comma()) +
    labs(
      title = paste0(title_prefix, toupper(dataset_name)),
      x = "",
      y = "Nilai (Rp)"
    ) +
    theme_forecast_clean()
  
  # Save if path provided  
  if (!is.null(save_path)) {
    dir.create(dirname(save_path), recursive = TRUE, showWarnings = FALSE)
    ggsave(save_path, p, width = 12, height = 7, dpi = 300, bg = "white")
    message(sprintf("✓ Plot saved: %s", save_path))
  }
  
  p
}

# ==============================================================================
# PLOT: PER TARGET (Iuran + Klaim in one chart)
# ==============================================================================

#' Plot single target with both Iuran and Klaim
#' @param iuran_data Forecast data for iuran
#' @param klaim_data Forecast data for klaim
#' @param target_name Name of target (e.g., "JHT")
#' @param save_path Path to save the plot
plot_target_both_datasets <- function(iuran_data, 
                                        klaim_data, 
                                        target_name,
                                        save_path = NULL) {
  
  # Prepare iuran data
  if (!is.null(iuran_data) && nrow(iuran_data) > 0) {
    iuran_plot <- iuran_data
    iuran_plot$value <- ifelse(is.na(iuran_plot$actual), 
                                iuran_plot$predicted, 
                                iuran_plot$actual)
    iuran_plot$dataset <- "IURAN"
  } else {
    iuran_plot <- NULL
  }
  
  # Prepare klaim data
  if (!is.null(klaim_data) && nrow(klaim_data) > 0) {
    klaim_plot <- klaim_data
    klaim_plot$value <- ifelse(is.na(klaim_plot$actual), 
                                klaim_plot$predicted, 
                                klaim_plot$actual)
    klaim_plot$dataset <- "KLAIM"
  } else {
    klaim_plot <- NULL
  }
  
  # Combine
  plot_data <- rbind(iuran_plot, klaim_plot)
  
  if (is.null(plot_data) || nrow(plot_data) == 0) {
    warning(sprintf("No data for target %s", target_name))
    return(NULL)
  }
  
  # Determine forecast years
  forecast_years <- plot_data$year[is.na(plot_data$actual)]
  all_years <- sort(unique(plot_data$year))
  year_labels <- sapply(all_years, function(y) {
    if (y %in% forecast_years) paste0(y, " F") else as.character(y)
  })
  
  # Build plot
  p <- ggplot(plot_data, aes(x = year, y = value, color = dataset)) +
    geom_line(size = 1.2) +
    geom_point(size = 2) +
    scale_color_manual(values = c("IURAN" = "#4472C4", "KLAIM" = "#A5A5A5")) +
    scale_x_continuous(breaks = all_years, labels = year_labels) +
    scale_y_continuous(labels = label_comma()) +
    labs(
      title = target_name,
      x = "",
      y = "Nilai (Rp)"
    ) +
    theme_forecast_clean()
  
  # Save if path provided
  if (!is.null(save_path)) {
    dir.create(dirname(save_path), recursive = TRUE, showWarnings = FALSE)
    ggsave(save_path, p, width = 10, height = 6, dpi = 300, bg = "white")
    message(sprintf("✓ Plot saved: %s", save_path))
  }
  
  p
}

# ==============================================================================
# PLOT: MODEL COMPARISON (LSTM vs H2O)
# ==============================================================================

#' Plot comparison of LSTM vs H2O forecasts for single target
#' @param lstm_data LSTM forecast data
#' @param h2o_data H2O forecast data
#' @param dataset_name Dataset name
#' @param target_name Target name
#' @param save_path Path to save
plot_model_comparison_forecast <- function(lstm_data,
                                             h2o_data,
                                             dataset_name,
                                             target_name,
                                             save_path = NULL) {
  
  # Prepare LSTM data
  lstm_plot <- lstm_data
  lstm_plot$value <- ifelse(is.na(lstm_plot$actual), 
                             lstm_plot$predicted, 
                             lstm_plot$actual)
  lstm_plot$source <- ifelse(is.na(lstm_plot$actual), "LSTM Forecast", "Actual")
  
  # Prepare H2O data - only forecast part
  h2o_forecast <- h2o_data[!is.na(h2o_data$predicted) & is.na(h2o_data$actual), ]
  h2o_forecast$value <- h2o_forecast$predicted
  h2o_forecast$source <- "H2O Forecast"
  
  # Combine: Actual line + LSTM forecast + H2O forecast
  actual_data <- lstm_plot[!is.na(lstm_plot$actual), ]
  actual_data$source <- "Actual"
  
  lstm_forecast <- lstm_plot[is.na(lstm_plot$actual), ]
  
  plot_data <- rbind(
    actual_data[, c("year", "value", "source")],
    lstm_forecast[, c("year", "value", "source")],
    h2o_forecast[, c("year", "value", "source")]
  )
  
  # Year labels
  forecast_years <- unique(c(lstm_forecast$year, h2o_forecast$year))
  all_years <- sort(unique(plot_data$year))
  year_labels <- sapply(all_years, function(y) {
    if (y %in% forecast_years) paste0(y, " F") else as.character(y)
  })
  
  p <- ggplot(plot_data, aes(x = year, y = value, color = source, linetype = source)) +
    geom_line(size = 1.2) +
    geom_point(size = 2) +
    scale_color_manual(values = c(
      "Actual" = "#2C3E50",
      "LSTM Forecast" = "#3498DB",
      "H2O Forecast" = "#9B59B6"
    )) +
    scale_linetype_manual(values = c(
      "Actual" = "solid",
      "LSTM Forecast" = "dashed",
      "H2O Forecast" = "dotted"
    )) +
    scale_x_continuous(breaks = all_years, labels = year_labels) +
    scale_y_continuous(labels = label_comma()) +
    labs(
      title = sprintf("%s - %s: LSTM vs H2O", toupper(dataset_name), target_name),
      x = "",
      y = "Nilai (Rp)"
    ) +
    theme_forecast_clean()
  
  if (!is.null(save_path)) {
    dir.create(dirname(save_path), recursive = TRUE, showWarnings = FALSE)
    ggsave(save_path, p, width = 10, height = 6, dpi = 300, bg = "white")
    message(sprintf("✓ Plot saved: %s", save_path))
  }
  
  p
}

# ==============================================================================
# GENERATE ALL FORECAST PLOTS
# ==============================================================================

#' Generate all forecast plots from combined forecast data
#' @param all_forecasts List with forecasts by dataset/target/model
#' @param output_dir Output directory for plots
generate_all_forecast_plots <- function(all_forecasts, output_dir = "results/plots") {
  
  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
  plot_count <- 0
  
  # Extract datasets
  datasets <- names(all_forecasts)
  
  for (ds in datasets) {
    ds_data <- all_forecasts[[ds]]
    
    # 1. Per-dataset plot (all targets)
    all_targets_data <- do.call(rbind, lapply(names(ds_data), function(tgt) {
      tgt_data <- ds_data[[tgt]]
      if (is.null(tgt_data)) return(NULL)
      # Use LSTM data for combined plot
      if (!is.null(tgt_data$lstm)) {
        df <- tgt_data$lstm$combined
        df$target <- tgt
        return(df)
      }
      return(NULL)
    }))
    
    if (!is.null(all_targets_data) && nrow(all_targets_data) > 0) {
      plot_path <- file.path(output_dir, sprintf("forecast_per_dataset_%s.png", ds))
      plot_dataset_all_targets(all_targets_data, ds, save_path = plot_path)
      plot_count <- plot_count + 1
    }
  }
  
  # 2. Per-target plots (Iuran + Klaim)
  if (length(datasets) >= 2 && "iuran" %in% datasets && "klaim" %in% datasets) {
    iuran_data <- all_forecasts$iuran
    klaim_data <- all_forecasts$klaim
    
    all_targets <- unique(c(names(iuran_data), names(klaim_data)))
    
    for (tgt in all_targets) {
      iuran_tgt <- if (!is.null(iuran_data[[tgt]]$lstm)) iuran_data[[tgt]]$lstm$combined else NULL
      klaim_tgt <- if (!is.null(klaim_data[[tgt]]$lstm)) klaim_data[[tgt]]$lstm$combined else NULL
      
      plot_path <- file.path(output_dir, sprintf("forecast_per_target_%s.png", tgt))
      tryCatch({
        plot_target_both_datasets(iuran_tgt, klaim_tgt, tgt, save_path = plot_path)
        plot_count <- plot_count + 1
      }, error = function(e) {
        warning(sprintf("Failed to plot %s: %s", tgt, e$message))
      })
    }
  }
  
  # 3. Model comparison plots (LSTM vs H2O)
  for (ds in datasets) {
    ds_data <- all_forecasts[[ds]]
    
    for (tgt in names(ds_data)) {
      tgt_data <- ds_data[[tgt]]
      
      if (!is.null(tgt_data$lstm) && !is.null(tgt_data$h2o)) {
        plot_path <- file.path(output_dir, sprintf("forecast_comparison_%s_%s.png", ds, tgt))
        tryCatch({
          plot_model_comparison_forecast(
            lstm_data = tgt_data$lstm$combined,
            h2o_data = tgt_data$h2o$combined,
            dataset_name = ds,
            target_name = tgt,
            save_path = plot_path
          )
          plot_count <- plot_count + 1
        }, error = function(e) {
          warning(sprintf("Failed comparison plot %s/%s: %s", ds, tgt, e$message))
        })
      }
    }
  }
  
  message(sprintf("\n✓ Generated %d forecast plots", plot_count))
  invisible(plot_count)
}
