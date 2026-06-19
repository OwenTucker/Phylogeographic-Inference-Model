suppressPackageStartupMessages({
  library(outbreaker2)
  library(ape)
  library(igraph)
})

#DEFUALTS

GEN_SHAPE    <- 4       # generation time Gamma shape
GEN_SCALE    <- 2       # generation time Gamma scale 
INIT_MU      <- 0.0018  # starting value for mu estimation (subs/site/day) NOTE: this default is tuned for the synthetic/simulation
SAMPLE_EVERY <- 50      
DEFAULT_PI        <- 0.10
DEFAULT_N_ITER    <- 50000
DEFAULT_MAX_KAPPA <- 10
DEFAULT_SEED      <- 44
DEFAULT_MOVE_T    <- FALSE

parse_args <- function() {
  raw  <- commandArgs(trailingOnly = TRUE)

  if (length(raw) == 0) {
    cat("Usage:\n")
    cat("  Rscript outbreaker_run.R --dir  <sim_dir>    [options]\n")
    cat("  Rscript outbreaker_run.R --all  <parent_dir> [options]\n\n")
    cat("Options:\n")
    cat("  --pi      <float>  reporting probability (default: 0.10)\n")
    cat("  --iter    <int>    MCMC iterations    (default: 50000)\n")
    cat("  --kappa   <int>    max kappa  (default: 10)\n")
    cat("  --seed    <int>    RNG seed  (default: 44)\n")
    cat("  --move_t  <bool>   estimate t_inf  (default: FALSE)\n")
    quit(status = 1)
  }

  p <- list(
    dir     = NULL,
    all     = NULL,
    pi      = DEFAULT_PI,
    iter    = DEFAULT_N_ITER,
    kappa   = DEFAULT_MAX_KAPPA,
    seed    = DEFAULT_SEED,
    move_t  = DEFAULT_MOVE_T,
    label   = "",
    fix_mu  = FALSE,
    no_seq  = FALSE,
    init_mu = NULL
  )

  i <- 1
  while (i <= length(raw)) {
    key <- sub("^--", "", raw[i])
    if (key %in% c("dir", "all", "pi", "iter", "kappa", "seed", "move_t", "label", "fix_mu", "no_seq", "init_mu")) {
      if (i + 1 > length(raw)) stop("Flag --", key, " requires a value")
      val <- raw[i + 1]
      if (key == "pi")  val <- as.numeric(val)
      if (key == "iter")  val <- as.integer(val)
      if (key == "kappa")val <- as.integer(val)
      if (key == "seed")   val <- as.integer(val)
      if (key == "move_t") val <- as.logical(val)
      if (key == "fix_mu") val <- as.logical(val)
      if (key == "no_seq") val <- as.logical(val)
      if (key == "init_mu") val <- as.numeric(val)
      p[[key]] <- val
      i <- i + 2
    } else {
      warning("Unknown flag: --", key, " (ignored)")
      i <- i + 1
    }
  }

  if (is.null(p$dir) && is.null(p$all))
    stop("Must supply either --dir or --all")
  if (!is.null(p$dir) && !is.null(p$all))
    stop("Supply only one of --dir or --all, not both")

  if (is.null(p[["label"]]))   p[["label"]]   <- ""
  if (is.null(p[["fix_mu"]]))  p[["fix_mu"]]  <- FALSE
  if (is.null(p[["no_seq"]]))  p[["no_seq"]]  <- FALSE
  if (is.null(p[["init_mu"]])) p[["init_mu"]] <- INIT_MU

  p
}

run_one <- function(tree_dir, pi_val, n_iter, max_kappa, seed, move_t, label = "", fix_mu = FALSE, no_seq = FALSE, init_mu_val = INIT_MU) {

  input_folder <- if (nchar(label) > 0) paste0("outbreaker_input_", label) else "outbreaker_input"
  input_dir  <- file.path(tree_dir, input_folder)
  fasta_path <- file.path(input_dir, "subset.fasta")
  dates_path <- file.path(input_dir, "dates.txt")

  # Validate
  if (!dir.exists(input_dir))    stop("outbreaker_input/ not found — run preprocess_outbreaker.py first")
  if (!file.exists(fasta_path))  stop("subset.fasta not found in outbreaker_input/")
  if (!file.exists(dates_path))  stop("dates.txt not found in outbreaker_input/")

  cat("  OUTBREAKER2 RUN\n")
  cat(sprintf("  Directory : %s\n", tree_dir))
  cat(sprintf("  pi=%.3f  iter=%d  max_kappa=%d  seed=%d  move_t=%s\n",
              pi_val, n_iter, max_kappa, seed, move_t))

  cat("--- [1] Loading data ---\n")

  sequences <- read.dna(fasta_path, format = "fasta")
  dates     <- scan(dates_path, what = integer(), quiet = TRUE)

  n_seq   <- nrow(sequences)
  seq_len <- ncol(sequences)

  cat(sprintf("Sequences:       %d\n", n_seq))
  cat(sprintf("Sites:           %d\n", seq_len))
  cat(sprintf("Date range:      %d to %d  (span: %d days)\n",
              min(dates), max(dates), max(dates) - min(dates)))
  cat(sprintf("Duplicate dates: %d / %d\n", sum(duplicated(dates)), n_seq))

  if (length(dates) != n_seq)
    stop("Date count (", length(dates), ") != sequence count (", n_seq, ")")

  # Shift so minimum date = 1 (outbreaker2 requires t_inf >= 0, i.e. dates-1 >= 0)
  if (min(dates) < 1) {
    shift <- 1 - min(dates)
    dates <- dates + shift
    cat(sprintf("Dates shifted by +%d (range now: %d to %d)\n",
                shift, min(dates), max(dates)))
  }

  original_ids        <- rownames(sequences)
  rownames(sequences) <- as.character(seq_len(n_seq))
  cat("\n--- [2] Generation time density ---\n")

  date_span <- max(dates) - min(dates)
  w_support <- max(150, date_span + 50)

  w_dens <- dgamma(1:w_support, shape = GEN_SHAPE, scale = GEN_SCALE)
  f_dens <- dgamma(1:w_support, shape = GEN_SHAPE, scale = GEN_SCALE)

  cat(sprintf("Gamma(%g, %g): mean = %g days\n", GEN_SHAPE, GEN_SCALE, GEN_SHAPE * GEN_SCALE))
  cat(sprintf("w_dens support: 1:%d  (date span = %d)\n", w_support, date_span))
  cat(sprintf("w_dens sum:     %.6f\n", sum(w_dens)))

  if (sum(w_dens) < 0.95)
    warning("w_dens sums to < 0.95 — consider extending w_support")

  

  cat("\n--- [3] Building outbreaker_data ---\n")

  data <- outbreaker_data(dna = sequences, dates = dates,
                          w_dens = w_dens, f_dens = f_dens)
  cat("outbreaker_data: OK\n")


  cat("\n--- [4] Creating config ---\n")

  effective_mu      <- if (no_seq) 1e-10 else init_mu_val
  effective_move_mu <- if (no_seq) FALSE  else !fix_mu

  config <- create_config(
    n_iter       = n_iter,
    sample_every = SAMPLE_EVERY,
    init_tree    = "star",
    move_mu      = effective_move_mu,
    init_mu      = effective_mu,
    move_pi      = FALSE,
    init_pi      = pi_val,
    move_eps     = FALSE,
    move_lambda  = FALSE,
    move_kappa   = TRUE,
    init_kappa   = 1,
    max_kappa    = max_kappa,
    move_alpha   = TRUE,
    move_t_inf   = move_t,
    init_t_inf   = dates - 1,
    pb           = TRUE
  )

  cat(sprintf("n_iter:          %d  (samples: %d)\n", n_iter, n_iter / SAMPLE_EVERY))
  cat(sprintf("init_mu:         %.6f\n", init_mu_val))
  cat(sprintf("pi (fixed):      %.4f\n", pi_val))
  cat(sprintf("max_kappa:       %d\n", max_kappa))
  cat(sprintf("move_t_inf:      %s\n", move_t))
  cat(sprintf("fix_mu:          %s  (mu=%s)\n", fix_mu, ifelse(fix_mu, sprintf("%.6f (fixed)", init_mu_val), "estimated")))
  cat(sprintf("no_seq:          %s%s\n", no_seq, ifelse(no_seq, "  (dates-only, mu=1e-10 fixed)", "")))


  cat("\n--- [5] Running outbreaker2 ---\n\n")

  set.seed(seed)
  res <- outbreaker(data = data, config = config)

  cat("\nRun complete.\n")


  cat("\n--- [6] Diagnostics ---\n")

  n_samples   <- nrow(res)
  n_inf_post  <- sum(is.infinite(res$post)  & res$post  < 0, na.rm = TRUE)
  n_inf_like  <- sum(is.infinite(res$like)  & res$like  < 0, na.rm = TRUE)
  n_inf_prior <- sum(is.infinite(res$prior) & res$prior < 0, na.rm = TRUE)

  cat(sprintf("Samples:         %d\n", n_samples))
  cat(sprintf("Posterior range: %.2f  to  %.2f\n",
              min(res$post, na.rm=TRUE), max(res$post, na.rm=TRUE)))
  cat(sprintf("-Inf posterior:  %d / %d\n", n_inf_post,  n_samples))
  cat(sprintf("-Inf likelihood: %d / %d\n", n_inf_like,  n_samples))
  cat(sprintf("-Inf prior:      %d / %d\n", n_inf_prior, n_samples))

  if (n_inf_post > 0)
    warning(sprintf("%d -Inf posteriors detected", n_inf_post))

  if ("mu" %in% names(res)) {
    cat(sprintf("\nmu posterior:    mean=%.6f  median=%.6f  sd=%.6f\n",
                mean(res$mu, na.rm=TRUE),
                median(res$mu, na.rm=TRUE),
                sd(res$mu, na.rm=TRUE)))
  }


  cat("\n--- [7] Extracting MAP tree ---\n")

  burn_n     <- floor(0.2 * n_samples)
  res_burned <- res[burn_n:n_samples, ]
  cat(sprintf("Burn-in:         %d samples (20%%)\n", burn_n))
  cat(sprintf("Post-burn-in:    %d samples\n", nrow(res_burned)))

  map_idx    <- which.max(res_burned$post)
  kappa_cols <- grep("^kappa_", names(res_burned))
  map_alpha  <- unlist(res_burned[map_idx, grep("^alpha_", names(res_burned))])
  map_kappa  <- unlist(res_burned[map_idx, kappa_cols])
  map_tinf   <- unlist(res_burned[map_idx, grep("^t_inf_",  names(res_burned))])

  cat(sprintf("MAP iteration:   %d\n", res_burned$step[map_idx]))
  cat(sprintf("MAP posterior:   %.4f\n", res_burned$post[map_idx]))
  cat("\nKappa distribution at MAP:\n")
  print(table(map_kappa))
  cat(sprintf("Mean kappa:      %.3f\n", mean(map_kappa, na.rm = TRUE)))

  cat("\n--- [8] Saving outputs ---\n")

  saveRDS(res, file.path(input_dir, "outbreaker_results.rds"))

  write.csv(
    data.frame(child = seq_along(map_alpha), parent = map_alpha),
    file.path(input_dir, "map_tree.csv"),  row.names = FALSE
  )
  write.csv(
    data.frame(child = seq_along(map_kappa), kappa = map_kappa),
    file.path(input_dir, "map_kappa.csv"), row.names = FALSE
  )
  write.csv(
    data.frame(child = seq_along(map_tinf), t_inf = map_tinf),
    file.path(input_dir, "map_tinf.csv"),  row.names = FALSE
  )
  write.csv(
    data.frame(simple_id = seq_len(n_seq), original_id = original_ids, date = dates),
    file.path(input_dir, "id_mapping.csv"), row.names = FALSE
  )

  for (f in c("outbreaker_results.rds","map_tree.csv","map_kappa.csv","map_tinf.csv","id_mapping.csv"))
    cat(sprintf("  outbreaker_input/%s\n", f))

  cat("\n--- [9] Trace plots ---\n")

  pdf(file.path(input_dir, "traces.pdf"), width = 10, height = 8)
  par(mfrow = c(2, 2))

  plot(res_burned$step, res_burned$post,
       type = "l", col = "steelblue",
       xlab = "Iteration", ylab = "Log posterior",
       main = sprintf("%s — Posterior", basename(tree_dir)))

  plot(res_burned$step, res_burned$like,
       type = "l", col = "darkgreen",
       xlab = "Iteration", ylab = "Log likelihood",
       main = "Likelihood trace")

  if ("mu" %in% names(res_burned)) {
    plot(res_burned$step, res_burned$mu,
         type = "l", col = "firebrick",
         xlab = "Iteration", ylab = "mu (subs/site/day)",
         main = "Mutation rate (estimated)")
    abline(h = init_mu_val, lty = 2, col = "grey50")
    legend("topright", legend = sprintf("init = %.5f", init_mu_val),
           lty = 2, col = "grey50", bty = "n")
  } else {
    plot.new(); title("mu not in output")
  }

  kappa_mat  <- as.matrix(res_burned[, kappa_cols])
  mean_kappa <- rowMeans(kappa_mat, na.rm = TRUE)
  plot(res_burned$step, mean_kappa,
       type = "l", col = "purple",
       xlab = "Iteration", ylab = "Mean kappa",
       main = "Mean kappa trace")

  dev.off()
  cat("  outbreaker_input/traces.pdf\n")

  edges <- data.frame(from = map_alpha, to = seq_along(map_alpha))
  edges <- edges[!is.na(edges$from) & edges$from != edges$to, ]
  if (nrow(edges) > 0) {
    g <- graph_from_data_frame(edges, directed = TRUE)
    pdf(file.path(input_dir, "map_tree.pdf"), width = 12, height = 12)
    plot(g,
         vertex.size     = 3,
         vertex.label    = NA,
         edge.arrow.size = 0.2,
         layout          = layout_with_fr(g),
         main            = sprintf("MAP Tree — %s", basename(tree_dir)))
    dev.off()
    cat("  outbreaker_input/map_tree.pdf\n")
  }

  invisible(TRUE)
}


args <- parse_args()

sim_dirs <- if (!is.null(args$dir)) {
  args$dir
} else {
  parent <- args$all
  if (!dir.exists(parent)) stop("Parent directory not found: ", parent)
  all_dirs <- list.dirs(parent, recursive = FALSE, full.names = TRUE)
  # Skip dirs without the label-aware outbreaker_input folder
  input_folder <- if (nchar(args$label) > 0) paste0("outbreaker_input_", args$label) else "outbreaker_input"
  has_input <- sapply(all_dirs, function(d) dir.exists(file.path(d, input_folder)))
  skipped   <- sum(!has_input)
  if (skipped > 0)
    cat(sprintf("[INFO] Skipping %d dirs with no outbreaker_input/ folder\n", skipped))
  sort(all_dirs[has_input])
}

n_total <- length(sim_dirs)
cat(sprintf("Processing %d simulation director%s\n\n",
            n_total, ifelse(n_total == 1, "y", "ies")))

results <- list()
for (sim_dir in sim_dirs) {
  outcome <- tryCatch(
    {
      run_one(sim_dir,
              pi_val      = args$pi,
              n_iter      = args$iter,
              max_kappa   = args$kappa,
              seed        = args$seed,
              move_t      = args$move_t,
              label       = args$label,
              fix_mu      = args$fix_mu,
              no_seq      = args$no_seq,
              init_mu_val = args$init_mu)
      "OK"
    },
    error = function(e) {
      cat(sprintf("\n  [ERROR] %s\n  %s\n\n", basename(sim_dir), conditionMessage(e)))
      conditionMessage(e)
    }
  )
  results[[basename(sim_dir)]] <- outcome
}

if (n_total > 1) {
  ok_n   <- sum(sapply(results, identical, "OK"))
  fail_n <- n_total - ok_n
  cat("============================================================\n")
  cat(sprintf("  BATCH COMPLETE:  %d / %d succeeded\n", ok_n, n_total))
  if (fail_n > 0) {
    cat(sprintf("  FAILED (%d):\n", fail_n))
    failed <- names(results)[!sapply(results, identical, "OK")]
    for (nm in failed) cat(sprintf("    - %s: %s\n", nm, results[[nm]]))
  }
  cat("============================================================\n")
}