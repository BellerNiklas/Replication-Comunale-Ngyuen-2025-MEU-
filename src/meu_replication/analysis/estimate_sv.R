#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(arrow)
  library(coda)
  library(jsonlite)
  library(stochvol)
})

read_config <- function() {
  args <- commandArgs(trailingOnly = TRUE)
  if (length(args) < 1) {
    stop("Expected the pytask-r JSON config path as the last CLI argument.")
  }
  read_json(args[[length(args)]], simplifyVector = TRUE)
}

apply_offset_if_needed <- function(values, offset) {
  adjusted <- as.numeric(values)
  if (any(!is.finite(adjusted))) {
    stop("Residual series contains missing or non-finite values.")
  }
  offset_applied <- 0.0
  if (min(log(adjusted^2)) == -Inf) {
    adjusted <- adjusted + offset
    offset_applied <- offset
  }
  list(
    values = adjusted,
    offset_applied = offset_applied,
    adjusted_for_zero = offset_applied > 0.0
  )
}

run_single_chain <- function(values, config, seed = NULL) {
  if (!is.null(seed)) {
    set.seed(as.integer(seed))
  }

  prepared <- apply_offset_if_needed(values, config$sv_offset)
  draws <- svsample(
    prepared$values,
    draws = as.integer(config$sv_draws),
    burnin = as.integer(config$sv_burnin),
    quiet = TRUE,
    thinpara = as.integer(config$sv_thin_para),
    thinlatent = as.integer(config$sv_thin_latent)
  )
  para_matrix <- as.matrix(draws$para)[, c("mu", "phi", "sigma"), drop = FALSE]
  latent_matrix <- as.matrix(draws$latent)
  para_means <- colMeans(para_matrix)
  latent_means <- colMeans(latent_matrix)
  geweke_z <- tryCatch(
    as.numeric(geweke.diag(draws$para)$z),
    error = function(err) {
      tryCatch(
        as.numeric(geweke.diag(coda::mcmc(para_matrix))$z),
        error = function(inner_err) rep(NaN, ncol(para_matrix))
      )
    }
  )
  list(
    para_means = para_means,
    para_matrix = para_matrix,
    latent_means = latent_means,
    latent_matrix = latent_matrix,
    geweke_z = geweke_z,
    offset_applied = prepared$offset_applied,
    adjusted_for_zero = prepared$adjusted_for_zero,
    n_param_draws = nrow(para_matrix),
    n_latent_draws = nrow(latent_matrix)
  )
}

relative_difference <- function(value_a, value_b, eps = 1e-6) {
  abs(value_a - value_b) / max(abs(value_b), eps)
}

resolve_panel_series <- function(forecast_errors, depends_on) {
  if (!("date" %in% names(forecast_errors))) {
    stop("Forecast-error parquet must contain a date column.")
  }

  available_series <- names(forecast_errors)[names(forecast_errors) != "date"]
  if (anyDuplicated(available_series)) {
    stop("Forecast-error parquet contains duplicated series columns.")
  }

  if (!is.null(depends_on$series_order)) {
    series_order <- as.data.frame(read_parquet(depends_on$series_order))
    if (anyDuplicated(series_order$series_id)) {
      stop("series_order.parquet contains duplicated series_id values.")
    }
    ordered <- series_order[order(series_order$series_position), , drop = FALSE]
    series_names <- as.character(ordered$series_id)
    positions <- as.integer(ordered$series_position)

    if (length(series_names) != length(available_series) ||
        !setequal(series_names, available_series)) {
      stop("series_order.parquet did not align one-to-one with the Y forecast-error columns.")
    }

    return(list(
      series_names = series_names,
      positions = positions
    ))
  }

  list(
    series_names = available_series,
    positions = as.integer(seq_along(available_series) - 1L)
  )
}

empty_panel_outputs <- function(date_values) {
  list(
    params_frame = data.frame(
      series_position = integer(),
      series_key = character(),
      mu = numeric(),
      phi = numeric(),
      sigma = numeric(),
      offset_applied = numeric(),
      adjusted_for_zero = logical(),
      stringsAsFactors = FALSE
    ),
    diagnostics_frame = data.frame(
      series_type = character(),
      series_position = integer(),
      series_key = character(),
      geweke_mu = numeric(),
      geweke_phi = numeric(),
      geweke_sigma = numeric(),
      acceptance_phi = numeric(),
      offset_applied = numeric(),
      adjusted_for_zero = logical(),
      n_param_draws = integer(),
      n_latent_draws = integer(),
      stringsAsFactors = FALSE
    ),
    latent_frame = data.frame(date = date_values, stringsAsFactors = FALSE)
  )
}

write_panel_outputs <- function(config) {
  forecast_errors <- as.data.frame(read_parquet(config$depends_on$forecast_errors))
  date_values <- as.character(forecast_errors$date)
  panel_series <- resolve_panel_series(forecast_errors, config$depends_on)
  series_names <- panel_series$series_names
  positions <- panel_series$positions

  if (length(series_names) == 0L) {
    empty_outputs <- empty_panel_outputs(date_values)
    write_parquet(empty_outputs$params_frame, config$produces$params)
    write_parquet(empty_outputs$latent_frame, config$produces$latent)
    write_parquet(empty_outputs$diagnostics_frame, config$produces$diagnostics)
    return(invisible(NULL))
  }

  params_rows <- vector("list", length(series_names))
  diagnostics_rows <- vector("list", length(series_names))
  latent_columns <- vector("list", length(series_names))
  names(latent_columns) <- series_names

  if (anyNA(forecast_errors[series_names])) {
    stop("Forecast-error parquet contains missing values in sampled residual columns.")
  }

  set.seed(as.integer(config$seed))
  for (index in seq_along(series_names)) {
    series_name <- series_names[[index]]
    result <- run_single_chain(
      forecast_errors[[series_name]],
      config
    )

    params_rows[[index]] <- data.frame(
      series_position = as.integer(positions[[index]]),
      series_key = as.character(series_name),
      mu = as.numeric(result$para_means[[1]]),
      phi = as.numeric(result$para_means[[2]]),
      sigma = as.numeric(result$para_means[[3]]),
      offset_applied = as.numeric(result$offset_applied),
      adjusted_for_zero = as.logical(result$adjusted_for_zero),
      stringsAsFactors = FALSE
    )

    diagnostics_rows[[index]] <- data.frame(
      series_type = as.character(config$series_type),
      series_position = as.integer(positions[[index]]),
      series_key = as.character(series_name),
      geweke_mu = as.numeric(result$geweke_z[[1]]),
      geweke_phi = as.numeric(result$geweke_z[[2]]),
      geweke_sigma = as.numeric(result$geweke_z[[3]]),
      acceptance_phi = NA_real_,
      offset_applied = as.numeric(result$offset_applied),
      adjusted_for_zero = as.logical(result$adjusted_for_zero),
      n_param_draws = as.integer(result$n_param_draws),
      n_latent_draws = as.integer(result$n_latent_draws),
      stringsAsFactors = FALSE
    )

    latent_columns[[series_name]] <- as.numeric(result$latent_means)
  }

  params_frame <- do.call(rbind, params_rows)
  diagnostics_frame <- do.call(rbind, diagnostics_rows)
  latent_frame <- data.frame(date = date_values, stringsAsFactors = FALSE)
  for (series_name in series_names) {
    latent_frame[[series_name]] <- latent_columns[[series_name]]
  }

  write_parquet(params_frame, config$produces$params)
  write_parquet(latent_frame, config$produces$latent)
  write_parquet(diagnostics_frame, config$produces$diagnostics)
}

as_job_frame <- function(records) {
  if (is.null(records)) {
    return(data.frame())
  }
  frame <- as.data.frame(records, stringsAsFactors = FALSE)
  if (nrow(frame) == 0L) {
    return(frame)
  }
  frame$series_position <- as.integer(frame$series_position)
  frame$chain_id <- as.integer(frame$chain_id)
  frame$seed <- as.integer(frame$seed)
  frame
}

write_validation_outputs <- function(config) {
  jobs <- as_job_frame(config$jobs)
  if (nrow(jobs) == 0L) {
    write_parquet(data.frame(), config$produces$draws)
    return(invisible(NULL))
  }

  forecast_errors_y <- as.data.frame(read_parquet(config$depends_on$forecast_errors_y))
  forecast_errors_f <- as.data.frame(read_parquet(config$depends_on$forecast_errors_f))
  draw_rows <- vector("list", nrow(jobs))

  for (index in seq_len(nrow(jobs))) {
    job <- jobs[index, , drop = FALSE]
    series_type <- as.character(job$series_type[[1]])
    series_key <- as.character(job$series_key[[1]])
    source_frame <- if (series_type == "y") forecast_errors_y else forecast_errors_f
    if (!(series_key %in% names(source_frame))) {
      stop(sprintf(
        "Validation job requested missing %s series column: %s",
        series_type,
        series_key
      ))
    }
    residuals <- if (series_type == "y") {
      forecast_errors_y[[series_key]]
    } else {
      forecast_errors_f[[series_key]]
    }

    result <- run_single_chain(
      residuals,
      config,
      seed = as.integer(job$seed[[1]])
    )

    draw_rows[[index]] <- data.frame(
      series_type = series_type,
      series_position = as.integer(job$series_position[[1]]),
      series_key = series_key,
      chain_id = as.integer(job$chain_id[[1]]),
      draw_index = seq_len(result$n_param_draws) - 1L,
      mu_draw = as.numeric(result$para_matrix[, 1]),
      phi_draw = as.numeric(result$para_matrix[, 2]),
      sigma_draw = as.numeric(result$para_matrix[, 3]),
      stringsAsFactors = FALSE
    )
  }

  write_parquet(do.call(rbind, draw_rows), config$produces$draws)
}

as_stability_frame <- function(records) {
  if (is.null(records)) {
    return(data.frame())
  }
  frame <- as.data.frame(records, stringsAsFactors = FALSE)
  if (nrow(frame) == 0L) {
    return(frame)
  }
  frame$series_position <- as.integer(frame$series_position)
  frame$seed <- as.integer(frame$seed)
  frame
}

write_stability_outputs <- function(config) {
  series_frame <- as_stability_frame(config$series)
  if (nrow(series_frame) == 0L) {
    write_parquet(data.frame(), config$produces$stability)
    return(invisible(NULL))
  }

  short_config <- list(
    sv_burnin = as.integer(config$short_sv_burnin),
    sv_draws = as.integer(config$short_sv_draws),
    sv_offset = as.numeric(config$sv_offset),
    sv_thin_latent = as.integer(config$short_sv_thin_latent),
    sv_thin_para = as.integer(config$short_sv_thin_para)
  )
  long_config <- list(
    sv_burnin = as.integer(config$long_sv_burnin),
    sv_draws = as.integer(config$long_sv_draws),
    sv_offset = as.numeric(config$sv_offset),
    sv_thin_latent = as.integer(config$long_sv_thin_latent),
    sv_thin_para = as.integer(config$long_sv_thin_para)
  )

  forecast_errors_y <- as.data.frame(read_parquet(config$depends_on$forecast_errors_y))
  rows <- vector("list", nrow(series_frame))

  for (index in seq_len(nrow(series_frame))) {
    record <- series_frame[index, , drop = FALSE]
    series_id <- as.character(record$series_id[[1]])
    seed <- as.integer(record$seed[[1]])
    if (!(series_id %in% names(forecast_errors_y))) {
      stop(sprintf("Stability job requested missing y series column: %s", series_id))
    }
    residuals <- forecast_errors_y[[series_id]]

    short_result <- run_single_chain(residuals, short_config, seed = seed)
    long_result <- run_single_chain(residuals, long_config, seed = seed)

    rows[[index]] <- data.frame(
      series_position = as.integer(record$series_position[[1]]),
      series_id = series_id,
      phi_relative_diff = relative_difference(
        short_result$para_means[[2]],
        long_result$para_means[[2]]
      ),
      sigma_relative_diff = relative_difference(
        short_result$para_means[[3]],
        long_result$para_means[[3]]
      ),
      stringsAsFactors = FALSE
    )
  }

  write_parquet(do.call(rbind, rows), config$produces$stability)
}

config <- read_config()

if (identical(config$task_kind, "panel")) {
  write_panel_outputs(config)
} else if (identical(config$task_kind, "validation")) {
  write_validation_outputs(config)
} else if (identical(config$task_kind, "stability")) {
  write_stability_outputs(config)
} else {
  stop(sprintf("Unsupported task_kind: %s", as.character(config$task_kind)))
}
