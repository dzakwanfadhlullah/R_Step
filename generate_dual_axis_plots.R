# ==============================================================================
# generate_dual_axis_plots.R - Dual Y-Axis Forecast Comparison Charts
# ==============================================================================
# Creates combined Iuran + Klaim charts with dual Y-axis for each target
# showing Actual, LSTM Forecast, and H2O Forecast
#
# Usage:
#   source("generate_dual_axis_plots.R")
# ==============================================================================

library(ggplot2)
library(scales)

# Load forecast data
all_forecasts <- read.csv("results/forecasts/all_forecasts.csv")

# Define targets
targets <- c("JHT", "JKK", "JKM", "JP", "JKP", "BPJS")

# Color palette
colors <- list(
  iuran_actual = "#1f4e79",      # Dark blue
  iuran_lstm = "#5b9bd5",        # Light blue (dashed)
  iuran_h2o = "#7030a0",         # Purple (dotted)
  klaim_actual = "#595959",      # Dark gray
  klaim_lstm = "#00b0f0",        # Cyan (dashed)
  klaim_h2o = "#c00000"          # Dark red (dotted)
)

# Create output directory
dir.create("results/plots", recursive = TRUE, showWarnings = FALSE)

# Function to create dual-axis plot for one target
create_dual_axis_plot <- function(target_name) {
  
  # Filter data for this target
  iuran_lstm <- all_forecasts[all_forecasts$dataset == "iuran" & 
                               all_forecasts$target == target_name & 
                               all_forecasts$model == "lstm", ]
  iuran_h2o <- all_forecasts[all_forecasts$dataset == "iuran" & 
                              all_forecasts$target == target_name & 
                              all_forecasts$model == "h2o", ]
  klaim_lstm <- all_forecasts[all_forecasts$dataset == "klaim" & 
                               all_forecasts$target == target_name & 
                               all_forecasts$model == "lstm", ]
  klaim_h2o <- all_forecasts[all_forecasts$dataset == "klaim" & 
                              all_forecasts$target == target_name & 
                              all_forecasts$model == "h2o", ]
  
  # Check if we have data
  if (nrow(iuran_lstm) == 0 || nrow(klaim_lstm) == 0) {
    message(sprintf("  Skipping %s: insufficient data", target_name))
    return(NULL)
  }
  
  # Prepare iuran data
  iuran_lstm$value <- ifelse(is.na(iuran_lstm$actual), iuran_lstm$predicted, iuran_lstm$actual)
  iuran_lstm$type <- ifelse(is.na(iuran_lstm$actual), "forecast", "actual")
  
  iuran_h2o$value <- ifelse(is.na(iuran_h2o$actual), iuran_h2o$predicted, iuran_h2o$actual)
  iuran_h2o$type <- ifelse(is.na(iuran_h2o$actual), "forecast", "actual")
  
  # Prepare klaim data
  klaim_lstm$value <- ifelse(is.na(klaim_lstm$actual), klaim_lstm$predicted, klaim_lstm$actual)
  klaim_lstm$type <- ifelse(is.na(klaim_lstm$actual), "forecast", "actual")
  
  klaim_h2o$value <- ifelse(is.na(klaim_h2o$actual), klaim_h2o$predicted, klaim_h2o$actual)
  klaim_h2o$type <- ifelse(is.na(klaim_h2o$actual), "forecast", "actual")
  
  # Get ranges for scaling
  iuran_max <- max(c(iuran_lstm$value, iuran_h2o$value), na.rm = TRUE)
  klaim_max <- max(c(klaim_lstm$value, klaim_h2o$value), na.rm = TRUE)
  
  # Calculate scale factor for secondary axis
  scale_factor <- iuran_max / klaim_max
  
  # Scale klaim values to match iuran axis
  klaim_lstm$value_scaled <- klaim_lstm$value * scale_factor
  klaim_h2o$value_scaled <- klaim_h2o$value * scale_factor
  
  # Year labels
  all_years <- sort(unique(c(iuran_lstm$year, klaim_lstm$year)))
  year_labels <- sapply(all_years, function(y) {
    if (y >= 2026) paste0(y, " F") else as.character(y)
  })
  
  # Build plot
  p <- ggplot() +
    # Iuran Actual
    geom_line(data = iuran_lstm[iuran_lstm$type == "actual", ],
              aes(x = year, y = value, color = "Iuran Actual"),
              size = 1.2) +
    geom_point(data = iuran_lstm[iuran_lstm$type == "actual", ],
               aes(x = year, y = value, color = "Iuran Actual"),
               size = 2) +
    # Iuran LSTM Forecast
    geom_line(data = iuran_lstm[iuran_lstm$type == "forecast", ],
              aes(x = year, y = value, color = "Iuran LSTM Forecast"),
              size = 1.2, linetype = "dashed") +
    geom_point(data = iuran_lstm[iuran_lstm$type == "forecast", ],
               aes(x = year, y = value, color = "Iuran LSTM Forecast"),
               size = 2) +
    # Iuran H2O Forecast
    geom_line(data = iuran_h2o[iuran_h2o$type == "forecast", ],
              aes(x = year, y = value, color = "Iuran H2O Forecast"),
              size = 1.2, linetype = "dotted") +
    geom_point(data = iuran_h2o[iuran_h2o$type == "forecast", ],
               aes(x = year, y = value, color = "Iuran H2O Forecast"),
               size = 2) +
    # Klaim Actual (scaled)
    geom_line(data = klaim_lstm[klaim_lstm$type == "actual", ],
              aes(x = year, y = value_scaled, color = "Klaim Actual"),
              size = 1.2) +
    geom_point(data = klaim_lstm[klaim_lstm$type == "actual", ],
               aes(x = year, y = value_scaled, color = "Klaim Actual"),
               size = 2) +
    # Klaim LSTM Forecast (scaled)
    geom_line(data = klaim_lstm[klaim_lstm$type == "forecast", ],
              aes(x = year, y = value_scaled, color = "Klaim LSTM Forecast"),
              size = 1.2, linetype = "dashed") +
    geom_point(data = klaim_lstm[klaim_lstm$type == "forecast", ],
               aes(x = year, y = value_scaled, color = "Klaim LSTM Forecast"),
               size = 2) +
    # Klaim H2O Forecast (scaled)
    geom_line(data = klaim_h2o[klaim_h2o$type == "forecast", ],
              aes(x = year, y = value_scaled, color = "Klaim H2O Forecast"),
              size = 1.2, linetype = "dotted") +
    geom_point(data = klaim_h2o[klaim_h2o$type == "forecast", ],
               aes(x = year, y = value_scaled, color = "Klaim H2O Forecast"),
               size = 2) +
    # Colors
    scale_color_manual(
      name = "",
      values = c(
        "Iuran Actual" = colors$iuran_actual,
        "Iuran LSTM Forecast" = colors$iuran_lstm,
        "Iuran H2O Forecast" = colors$iuran_h2o,
        "Klaim Actual" = colors$klaim_actual,
        "Klaim LSTM Forecast" = colors$klaim_lstm,
        "Klaim H2O Forecast" = colors$klaim_h2o
      ),
      breaks = c("Iuran Actual", "Iuran LSTM Forecast", "Iuran H2O Forecast",
                 "Klaim Actual", "Klaim LSTM Forecast", "Klaim H2O Forecast")
    ) +
    # Dual Y-axis
    scale_y_continuous(
      name = "Iuran (Rp)",
      labels = label_comma(scale = 1e-12, suffix = " T"),
      sec.axis = sec_axis(
        ~ . / scale_factor,
        name = "Klaim (Rp)",
        labels = label_comma(scale = 1e-12, suffix = " T")
      )
    ) +
    scale_x_continuous(breaks = all_years, labels = year_labels) +
    labs(
      title = sprintf("%s - IURAN vs KLAIM: LSTM vs H2O Forecast", target_name),
      x = "Tahun"
    ) +
    theme_minimal(base_size = 11) +
    theme(
      plot.title = element_text(face = "bold", hjust = 0.5, size = 14, color = "#1f4e79"),
      axis.title.y.left = element_text(color = colors$iuran_actual, face = "bold"),
      axis.text.y.left = element_text(color = colors$iuran_actual),
      axis.title.y.right = element_text(color = colors$klaim_actual, face = "bold"),
      axis.text.y.right = element_text(color = colors$klaim_actual),
      axis.text.x = element_text(angle = 45, hjust = 1),
      legend.position = "bottom",
      legend.box = "horizontal",
      panel.grid.minor = element_blank(),
      plot.background = element_rect(fill = "white", color = NA)
    ) +
    guides(color = guide_legend(nrow = 2, byrow = TRUE))
  
  # Save
  save_path <- sprintf("results/plots/forecast_comparison_%s.png", target_name)
  ggsave(save_path, p, width = 12, height = 7, dpi = 300, bg = "white")
  message(sprintf("✓ Saved: %s", save_path))
  
  p
}

# Generate plots for all targets
message("\n=== Generating Dual-Axis Comparison Plots ===\n")
for (target in targets) {
  tryCatch({
    create_dual_axis_plot(target)
  }, error = function(e) {
    message(sprintf("  Error for %s: %s", target, e$message))
  })
}

message("\n=== Done! ===")
