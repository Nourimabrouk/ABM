# =================================================================
# BASIC RSIENA EXAMPLE FOR ABM INTEGRATION
# Demonstrates core RSiena workflow for agent-based model validation
# =================================================================

# Load required libraries
library(RSiena)
library(igraph)
library(tidyverse)
library(ggraph)

# Set up
set.seed(42)  # For reproducibility

#' Create Example Longitudinal Network Data
#'
#' Generates synthetic network data that mimics common ABM patterns
#'
#' @param n_actors Number of actors in the network
#' @param n_periods Number of time periods
#' @param base_density Base network density
#' @return Array of adjacency matrices
create_example_networks <- function(n_actors = 20, n_periods = 3, base_density = 0.15) {

  cat("Creating example longitudinal network data...\n")
  cat("Actors:", n_actors, "| Periods:", n_periods, "\n")

  networks <- array(0, dim = c(n_actors, n_actors, n_periods))

  for (t in 1:n_periods) {
    # Increasing density over time (common in friendship networks)
    current_density <- base_density + (t - 1) * 0.05

    # Random baseline network
    prob_matrix <- matrix(current_density, n_actors, n_actors)
    diag(prob_matrix) <- 0

    networks[,,t] <- (matrix(runif(n_actors^2), n_actors, n_actors) < prob_matrix) * 1

    # Add structural effects if not the first period
    if (t > 1) {
      prev_network <- networks[,, t-1]

      # Transitivity effect: friends of friends become friends
      for (i in 1:(n_actors-1)) {
        for (j in (i+1):n_actors) {
          if (networks[i, j, t] == 0) {  # Not yet connected
            # Count common neighbors
            common_neighbors <- sum(prev_network[i,] * prev_network[j,])

            if (common_neighbors > 0) {
              # Probability increases with common neighbors
              trans_prob <- min(0.4, common_neighbors * 0.15)
              if (runif(1) < trans_prob) {
                networks[i, j, t] <- 1
                networks[j, i, t] <- 1  # Symmetric friendship
              }
            }
          }
        }
      }

      # Reciprocity effect: mutual friendships are more stable
      asymmetric_ties <- which((prev_network == 1) & (t(prev_network) == 0), arr.ind = TRUE)
      for (tie in seq_len(nrow(asymmetric_ties))) {
        i <- asymmetric_ties[tie, 1]
        j <- asymmetric_ties[tie, 2]

        if (runif(1) < 0.3) {  # 30% chance of reciprocation
          networks[j, i, t] <- 1
        }
      }
    }

    # Remove self-loops
    diag(networks[,,t]) <- 0

    cat("Period", t, "- Edges:", sum(networks[,,t]), "- Density:",
        round(sum(networks[,,t]) / (n_actors * (n_actors - 1)), 3), "\n")
  }

  return(networks)
}

#' Create Actor Attributes for RSiena Analysis
#'
#' @param n_actors Number of actors
#' @param n_periods Number of time periods
#' @return List of attribute matrices
create_actor_attributes <- function(n_actors, n_periods) {

  cat("Creating actor attributes...\n")

  # Constant attributes (don't change over time)
  gender <- sample(c(0, 1), n_actors, replace = TRUE)  # 0 = female, 1 = male
  ses <- rnorm(n_actors, mean = 0, sd = 1)  # Socioeconomic status

  # Time-varying attributes
  # Academic performance that can change slightly over time
  performance <- matrix(NA, n_periods, n_actors)
  performance[1,] <- rnorm(n_actors, mean = 0, sd = 1)

  for (t in 2:n_periods) {
    # Performance is correlated with previous period but can change
    performance[t,] <- 0.8 * performance[t-1,] + rnorm(n_actors, 0, 0.6)
  }

  list(
    gender = gender,
    ses = ses,
    performance = performance
  )
}

#' Run Complete RSiena Analysis
#'
#' @param networks Array of network adjacency matrices
#' @param attributes List of actor attributes
#' @return List with RSiena results
run_rsiena_analysis <- function(networks, attributes) {

  cat("\n=== Running RSiena Analysis ===\n")

  # 1. Create RSiena data object
  cat("Creating RSiena data object...\n")

  siena_data <- sienaDataCreate(
    friendship = networks,
    gender = coCovar(attributes$gender),
    ses = coCovar(attributes$ses),
    performance = varCovar(attributes$performance)
  )

  print(siena_data)

  # 2. Create effects object
  cat("\nSetting up effects...\n")

  effects <- getEffects(siena_data)

  # Include basic structural effects
  effects <- includeEffects(effects, transTrip)    # Transitivity
  effects <- includeEffects(effects, cycle3)       # 3-cycles
  effects <- includeEffects(effects, reciprocity)  # Reciprocity (if directed)

  # Include attribute effects (homophily)
  effects <- includeEffects(effects, simX, interaction1 = "gender")
  effects <- includeEffects(effects, simX, interaction1 = "ses")

  # Include attribute-network effects
  effects <- includeEffects(effects, egoX, interaction1 = "performance")
  effects <- includeEffects(effects, altX, interaction1 = "performance")

  print(effects)

  # 3. Create algorithm object
  cat("\nSetting algorithm parameters...\n")

  algorithm <- sienaAlgorithmCreate(
    projname = "ABM_RSiena_Example",
    cond = FALSE,
    useStdInits = FALSE,
    nsub = 4,
    n3 = 1000,
    simOnly = FALSE,
    seed = 12345
  )

  # 4. Run estimation
  cat("\nStarting model estimation...\n")
  cat("This may take several minutes...\n")

  results <- tryCatch({
    siena07(algorithm, data = siena_data, effects = effects, verbose = FALSE)
  }, error = function(e) {
    cat("Estimation error:", e$message, "\n")
    return(NULL)
  })

  if (!is.null(results)) {
    cat("\n=== Estimation Results ===\n")
    print(results)

    # Check convergence
    if (results$OK) {
      cat("\n✓ Model converged successfully!\n")
      cat("Maximum convergence ratio:", max(abs(results$tconv)), "\n")
    } else {
      cat("\n⚠ Model did not converge properly\n")
      cat("Maximum convergence ratio:", max(abs(results$tconv)), "\n")
      cat("Consider adjusting algorithm parameters\n")
    }

    return(list(
      data = siena_data,
      effects = effects,
      algorithm = algorithm,
      results = results
    ))
  } else {
    cat("\n❌ Estimation failed\n")
    return(list(
      data = siena_data,
      effects = effects,
      algorithm = algorithm,
      results = NULL
    ))
  }
}

#' Visualize Network Evolution
#'
#' @param networks Array of adjacency matrices
#' @param title Plot title
visualize_network_evolution <- function(networks, title = "Network Evolution") {

  n_periods <- dim(networks)[3]

  # Create plots for each time period
  plots <- list()

  for (t in 1:n_periods) {
    # Convert to igraph
    g <- graph_from_adjacency_matrix(networks[,,t], mode = "undirected")

    # Create plot
    p <- ggraph(g, layout = "stress") +
      geom_edge_link(alpha = 0.6, color = "gray50") +
      geom_node_point(size = 3, color = "steelblue", alpha = 0.8) +
      geom_node_text(aes(label = 1:vcount(g)), size = 2.5, color = "white") +
      theme_void() +
      labs(title = paste("Period", t)) +
      theme(
        plot.title = element_text(hjust = 0.5, size = 12),
        panel.background = element_rect(fill = "white", color = NA)
      )

    plots[[t]] <- p
  }

  # Combine plots
  if (require(patchwork, quietly = TRUE)) {
    combined_plot <- wrap_plots(plots, ncol = n_periods) +
      plot_annotation(
        title = title,
        theme = theme(plot.title = element_text(hjust = 0.5, size = 14, face = "bold"))
      )

    return(combined_plot)
  } else {
    # Return first plot if patchwork not available
    return(plots[[1]])
  }
}

#' Calculate Network Statistics Over Time
#'
#' @param networks Array of adjacency matrices
#' @return Data frame with network statistics
calculate_network_stats <- function(networks) {

  n_periods <- dim(networks)[3]
  stats <- data.frame(
    period = 1:n_periods,
    density = numeric(n_periods),
    transitivity = numeric(n_periods),
    reciprocity = numeric(n_periods),
    avg_degree = numeric(n_periods)
  )

  for (t in 1:n_periods) {
    g <- graph_from_adjacency_matrix(networks[,,t], mode = "directed")

    stats$density[t] <- edge_density(g)
    stats$transitivity[t] <- transitivity(g, type = "global")
    stats$reciprocity[t] <- reciprocity(g)
    stats$avg_degree[t] <- mean(degree(g))
  }

  return(stats)
}

# =================================================================
# MAIN ANALYSIS WORKFLOW
# =================================================================

main_analysis <- function() {
  cat("Starting RSiena Analysis Example\n")
  cat("================================\n\n")

  # 1. Create example data
  networks <- create_example_networks(n_actors = 15, n_periods = 3)
  attributes <- create_actor_attributes(15, 3)

  # 2. Calculate descriptive statistics
  cat("\n=== Descriptive Network Statistics ===\n")
  stats <- calculate_network_stats(networks)
  print(stats)

  # 3. Visualize networks
  cat("\nCreating network visualization...\n")
  if (interactive()) {
    plot <- visualize_network_evolution(networks, "Example Network Evolution")
    print(plot)
  }

  # 4. Run RSiena analysis
  analysis_results <- run_rsiena_analysis(networks, attributes)

  # 5. Interpret results
  if (!is.null(analysis_results$results) && analysis_results$results$OK) {
    cat("\n=== Model Interpretation ===\n")

    # Extract parameter estimates
    theta <- analysis_results$results$theta
    se <- analysis_results$results$se
    tstat <- theta / se

    effects_summary <- data.frame(
      Effect = analysis_results$effects$effectName,
      Estimate = round(theta, 3),
      SE = round(se, 3),
      t_ratio = round(tstat, 3)
    )

    print(effects_summary)

    cat("\nKey Findings:\n")
    cat("- Density parameter:", round(theta[1], 3), "\n")
    cat("- Reciprocity:", round(theta[which(analysis_results$effects$shortName == "recip")[1]], 3), "\n")
    cat("- Transitivity:", round(theta[which(analysis_results$effects$shortName == "transTrip")[1]], 3), "\n")

  } else {
    cat("\n⚠ Model estimation failed or did not converge\n")
    cat("This is common with small networks or insufficient data\n")
    cat("Consider:\n")
    cat("- Increasing network size\n")
    cat("- Adding more time periods\n")
    cat("- Simplifying the model specification\n")
  }

  cat("\nAnalysis complete!\n")

  return(analysis_results)
}

# Run the analysis if script is executed directly
if (interactive() || !exists("sourced")) {
  results <- main_analysis()
}