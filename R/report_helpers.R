# Helpers for the NBA player fingerprint Quarto report.

processed_path <- function(filename, processed_dir = "data/processed") {
  file.path(processed_dir, filename)
}

read_processed_csv <- function(filename, processed_dir = "data/processed") {
  path <- processed_path(filename, processed_dir)
  if (!file.exists(path)) {
    stop(
      "Missing processed file: ", path,
      "\nRun the Python export command before rendering the report.",
      call. = FALSE
    )
  }
  read.csv(path, stringsAsFactors = FALSE, check.names = FALSE)
}

load_player_fingerprint_outputs <- function(season = "2023-24", processed_dir = "data/processed") {
  season_key <- gsub("-", "_", season)
  list(
    features = read_processed_csv(paste0("player_season_features_", season_key, ".csv"), processed_dir),
    fingerprints = read_processed_csv(paste0("player_fingerprints_", season_key, ".csv"), processed_dir),
    neighbors = read_processed_csv(paste0("player_neighbors_", season_key, ".csv"), processed_dir),
    position_scores = read_processed_csv(paste0("player_position_scores_", season_key, ".csv"), processed_dir),
    archetype_scores = read_processed_csv(paste0("player_archetype_scores_", season_key, ".csv"), processed_dir),
    archetype_explanations = read_processed_csv(paste0("player_archetype_explanations_", season_key, ".csv"), processed_dir),
    player_summary = read_processed_csv(paste0("player_summary_", season_key, ".csv"), processed_dir)
  )
}

top_archetype_matches <- function(archetype_scores, n = 15) {
  top <- archetype_scores[archetype_scores$archetype_rank == 1, ]
  top <- top[order(-top$cosine_similarity), ]
  head(top, n)
}

player_archetype_profile <- function(archetype_scores, player_name) {
  rows <- archetype_scores[archetype_scores$player_name == player_name, ]
  rows[order(rows$archetype_rank), ]
}

player_neighbors <- function(neighbors, player_name, n = 5) {
  rows <- neighbors[neighbors$player_name == player_name, ]
  rows <- rows[order(rows$neighbor_rank), ]
  head(rows, n)
}

plot_archetype_profile <- function(archetype_scores, player_name) {
  rows <- player_archetype_profile(archetype_scores, player_name)
  if (nrow(rows) == 0) {
    warning("No archetype scores found for ", player_name)
    return(invisible(NULL))
  }

  rows <- rows[order(rows$cosine_similarity), ]
  old_mar <- par("mar")
  on.exit(par(mar = old_mar), add = TRUE)
  par(mar = c(5, 11, 3, 2))
  barplot(
    rows$cosine_similarity,
    names.arg = rows$archetype,
    horiz = TRUE,
    las = 1,
    xlim = c(0, 1),
    col = "#4C78A8",
    border = NA,
    main = paste(player_name, "archetype similarity"),
    xlab = "Cosine similarity"
  )
}

plot_top_archetype_counts <- function(archetype_scores) {
  top <- archetype_scores[archetype_scores$archetype_rank == 1, ]
  counts <- sort(table(top$archetype), decreasing = TRUE)
  old_mar <- par("mar")
  on.exit(par(mar = old_mar), add = TRUE)
  par(mar = c(8, 5, 3, 1))
  barplot(
    counts,
    las = 2,
    col = "#59A14F",
    border = NA,
    main = "Top archetype assignment counts",
    ylab = "Players"
  )
}
