parse_indonesian_number <- function(values) {
  if (is.null(values)) {
    return(numeric())
  }

  if (is.numeric(values)) {
    return(values)
  }

  values_chr <- trimws(as.character(values))
  values_chr[values_chr == ""] <- NA_character_

  cleaned_values <- gsub(",", "", values_chr, fixed = TRUE)
  parsed_values <- suppressWarnings(as.numeric(cleaned_values))

  invalid_mask <- !is.na(values_chr) & is.na(parsed_values)
  if (any(invalid_mask)) {
    warning(
      sprintf(
        "Invalid numeric values detected: %s",
        paste(unique(values_chr[invalid_mask]), collapse = ", ")
      ),
      call. = FALSE
    )
  }

  parsed_values
}

test_parse_indonesian_number <- function() {
  stopifnot(identical(parse_indonesian_number(100), 100))
  stopifnot(identical(parse_indonesian_number(""), as.numeric(NA)))
  stopifnot(identical(
    parse_indonesian_number("12,894,359,909,168.00"),
    12894359909168
  ))
  stopifnot(identical(
    suppressWarnings(parse_indonesian_number("invalid")),
    as.numeric(NA)
  ))
  stopifnot(identical(
    parse_indonesian_number(c("1,000.00", "2,500.50", "")),
    c(1000, 2500.5, NA_real_)
  ))

  invisible(TRUE)
}

read_raw_csv <- function(filepath) {
  if (!file.exists(filepath)) {
    stop(sprintf("File not found: %s", filepath), call. = FALSE)
  }

  # Try readr first, fallback to base R if not available
  if (requireNamespace("readr", quietly = TRUE)) {
    raw_data <- tryCatch(
      readr::read_delim(
        filepath,
        delim = ";",
        col_types = readr::cols(.default = readr::col_character()),
        locale = readr::locale(encoding = "UTF-8"),
        na = c("", "NA"),
        show_col_types = FALSE,
        progress = FALSE
      ),
      error = function(read_error) {
        stop(
          sprintf("Failed to read '%s': %s", filepath, read_error$message),
          call. = FALSE
        )
      }
    )
  } else {
    # Fallback to base R
    raw_data <- tryCatch(
      read.csv(
        filepath,
        sep = ";",
        stringsAsFactors = FALSE,
        na.strings = c("", "NA"),
        fileEncoding = "UTF-8",
        check.names = FALSE
      ),
      error = function(read_error) {
        stop(
          sprintf("Failed to read '%s': %s", filepath, read_error$message),
          call. = FALSE
        )
      }
    )
    # Convert all columns to character for consistent processing
    raw_data[] <- lapply(raw_data, as.character)
  }

  if (!"Tahun" %in% names(raw_data)) {
    warning("Column 'Tahun' not found; filling with NA.", call. = FALSE)
    raw_data$Tahun <- NA_character_
  }

  year_strings <- raw_data$Tahun
  year_values <- suppressWarnings(as.integer(year_strings))
  invalid_years <- !is.na(year_strings) & is.na(year_values)
  if (any(invalid_years)) {
    warning(
      sprintf(
        "Invalid Tahun values detected: %s",
        paste(unique(year_strings[invalid_years]), collapse = ", ")
      ),
      call. = FALSE
    )
  }

  raw_data$Tahun <- year_values

  target_columns <- setdiff(names(raw_data), "Tahun")
  if (length(target_columns) > 0) {
    raw_data[target_columns] <- lapply(raw_data[target_columns], parse_indonesian_number)
  }

  message(sprintf(
    "Read %d rows and %d columns from %s",
    nrow(raw_data),
    ncol(raw_data),
    filepath
  ))

  raw_data
}

validate_raw_data <- function(data_frame, dataset_name) {
  required_columns <- c("Tahun", "JHT", "JKK", "JKM", "JP", "JKP", "BPJS")
  missing_columns <- setdiff(required_columns, names(data_frame))
  if (length(missing_columns) > 0) {
    stop(
      sprintf(
        "Dataset '%s' is missing required columns: %s",
        dataset_name,
        paste(missing_columns, collapse = ", ")
      ),
      call. = FALSE
    )
  }

  if (!is.numeric(data_frame$Tahun)) {
    stop(
      sprintf("Dataset '%s' column 'Tahun' is not numeric.", dataset_name),
      call. = FALSE
    )
  }

  year_values <- data_frame$Tahun
  if (any(is.na(year_values))) {
    stop(
      sprintf("Dataset '%s' has missing Tahun values.", dataset_name),
      call. = FALSE
    )
  }

  if (length(year_values) > 1 && any(diff(year_values) <= 0)) {
    stop(
      sprintf("Dataset '%s' Tahun is not strictly ascending.", dataset_name),
      call. = FALSE
    )
  }

  target_columns <- setdiff(required_columns, "Tahun")
  non_numeric_targets <- target_columns[!vapply(
    data_frame[target_columns],
    is.numeric,
    logical(1)
  )]
  if (length(non_numeric_targets) > 0) {
    stop(
      sprintf(
        "Dataset '%s' non-numeric target columns: %s",
        dataset_name,
        paste(non_numeric_targets, collapse = ", ")
      ),
      call. = FALSE
    )
  }

  min_year <- min(year_values)
  max_year <- max(year_values)
  expected_range <- switch(
    tolower(dataset_name),
    klaim = c(2014L, 2025L),
    iuran = c(2015L, 2025L),
    NULL
  )
  if (is.null(expected_range)) {
    warning(
      sprintf("Unknown dataset name '%s'; skipping year range check.", dataset_name),
      call. = FALSE
    )
  } else if (min_year != expected_range[1] || max_year != expected_range[2]) {
    stop(
      sprintf(
        "Dataset '%s' year range %d-%d does not match expected %d-%d.",
        dataset_name,
        min_year,
        max_year,
        expected_range[1],
        expected_range[2]
      ),
      call. = FALSE
    )
  }

  safe_min <- function(values) {
    if (all(is.na(values))) {
      return(NA_real_)
    }
    min(values, na.rm = TRUE)
  }

  safe_max <- function(values) {
    if (all(is.na(values))) {
      return(NA_real_)
    }
    max(values, na.rm = TRUE)
  }

  safe_mean <- function(values) {
    if (all(is.na(values))) {
      return(NA_real_)
    }
    mean(values, na.rm = TRUE)
  }

  summary_stats <- data.frame(
    column = required_columns,
    min = vapply(data_frame[required_columns], safe_min, numeric(1)),
    max = vapply(data_frame[required_columns], safe_max, numeric(1)),
    mean = vapply(data_frame[required_columns], safe_mean, numeric(1)),
    row.names = NULL
  )

  output_dir <- file.path("data", "processed")
  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
  summary_path <- file.path(output_dir, paste0(dataset_name, "_raw_summary.csv"))
  write.csv(summary_stats, summary_path, row.names = FALSE)

  message(sprintf(
    "Validation passed for %s: %d rows, %d columns.",
    dataset_name,
    nrow(data_frame),
    ncol(data_frame)
  ))
  message(sprintf("Year range: %d-%d.", min_year, max_year))
  message(sprintf("Summary stats saved to %s", summary_path))

  invisible(summary_stats)
}
