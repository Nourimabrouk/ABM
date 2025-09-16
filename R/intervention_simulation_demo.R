# ============================================================================
# RSIENA TOLERANCE INTERVENTION: ADVANCED SIMULATION DEMONSTRATION
# ============================================================================
#
# Demo 2: Advanced Intervention Simulation
# Complete intervention experiment from hypothesis to conclusions
# Focus: Testing different targeting strategies for tolerance interventions
#
# Author: ABM Research Project
# Date: 2025
# Purpose: Demonstrate complete intervention research workflow with RSiena
# ============================================================================

# Load required libraries
library(RSiena)
library(network)
library(sna)
library(igraph)
library(ggplot2)
library(dplyr)
library(tidyr)
library(purrr)
library(broom)
library(RColorBrewer)
library(gridExtra)
library(knitr)
library(parallel)

# Set random seed for reproducibility
set.seed(123)

# ============================================================================
# SECTION 1: RESEARCH HYPOTHESIS AND EXPERIMENTAL DESIGN
# ============================================================================

#' Define research hypotheses for tolerance intervention study
#'
#' Establishes theoretical framework and testable predictions
formulate_research_hypotheses <- function() {

  cat("=" * 80, "\n")
  cat("TOLERANCE INTERVENTION RESEARCH HYPOTHESES\n")
  cat("=" * 80, "\n")

  hypotheses <- list(
    H1 = "Central targeting will be more effective than peripheral targeting",
    H2 = "Clustered interventions will show stronger spillover effects",
    H3 = "Random targeting will be least effective overall",
    H4 = "Intervention effects will decay over time without reinforcement",
    H5 = "Network density moderates intervention effectiveness"
  )

  cat("\nFORMAL HYPOTHESES:\n")
  for (i in seq_along(hypotheses)) {
    cat(sprintf("H%d: %s\n", i, hypotheses[[i]]))
  }

  # Define operational measures
  measures <- list(
    primary = "Change in network-wide tolerance mean",
    secondary = c(
      "Tolerance variance reduction",
      "Number of actors reaching high tolerance (>0.8)",
      "Persistence of effects at follow-up",
      "Spillover to non-intervention actors"
    )
  )

  cat("\nOPERATIONAL MEASURES:\n")
  cat("Primary outcome:", measures$primary, "\n")
  cat("Secondary outcomes:\n")
  for (measure in measures$secondary) {
    cat("  •", measure, "\n")
  }

  return(list(hypotheses = hypotheses, measures = measures))
}

# ============================================================================
# SECTION 2: INTERVENTION TARGETING STRATEGIES
# ============================================================================

#' Central targeting: Select highest degree actors
#'
#' Targets most connected individuals for maximum network influence
#' @param network Adjacency matrix
#' @param proportion Proportion of actors to target
#' @return Vector of selected actor IDs
central_targeting <- function(network, proportion = 0.2) {
  degrees <- rowSums(network)
  n_select <- round(nrow(network) * proportion)
  selected <- order(degrees, decreasing = TRUE)[1:n_select]

  cat("Central targeting selected", length(selected), "actors\n")
  cat("Average degree of selected actors:", round(mean(degrees[selected]), 2), "\n")

  return(selected)
}

#' Peripheral targeting: Select lowest degree actors
#'
#' Targets least connected individuals to test diffusion from edges
#' @param network Adjacency matrix
#' @param proportion Proportion of actors to target
#' @return Vector of selected actor IDs
peripheral_targeting <- function(network, proportion = 0.2) {
  degrees <- rowSums(network)
  n_select <- round(nrow(network) * proportion)
  selected <- order(degrees, decreasing = FALSE)[1:n_select]

  cat("Peripheral targeting selected", length(selected), "actors\n")
  cat("Average degree of selected actors:", round(mean(degrees[selected]), 2), "\n")

  return(selected)
}

#' Clustered targeting: Select actors in tight clusters
#'
#' Targets actors within dense network regions for local saturation
#' @param network Adjacency matrix
#' @param proportion Proportion of actors to target
#' @return Vector of selected actor IDs
clustered_targeting <- function(network, proportion = 0.2) {
  # Convert to igraph for clustering
  g <- graph_from_adjacency_matrix(network, mode = "undirected")

  # Find communities
  communities <- cluster_louvain(g)

  # Calculate cluster properties
  cluster_sizes <- table(membership(communities))
  cluster_densities <- sapply(1:max(membership(communities)), function(i) {
    cluster_nodes <- which(membership(communities) == i)
    if (length(cluster_nodes) < 2) return(0)
    cluster_subgraph <- induced_subgraph(g, cluster_nodes)
    edge_density(cluster_subgraph)
  })

  # Select from densest clusters first
  n_select <- round(vcount(g) * proportion)
  selected <- c()

  for (cluster_id in order(cluster_densities, decreasing = TRUE)) {
    cluster_members <- which(membership(communities) == cluster_id)
    n_from_cluster <- min(length(cluster_members), n_select - length(selected))

    if (n_from_cluster > 0) {
      # Within cluster, select highest degree nodes
      cluster_degrees <- degree(g)[cluster_members]
      cluster_selected <- cluster_members[order(cluster_degrees, decreasing = TRUE)[1:n_from_cluster]]
      selected <- c(selected, cluster_selected)
    }

    if (length(selected) >= n_select) break
  }

  selected <- selected[1:n_select]

  cat("Clustered targeting selected", length(selected), "actors\n")
  cat("Number of clusters represented:", length(unique(membership(communities)[selected])), "\n")

  return(selected)
}

#' Random targeting: Random selection for control comparison
#'
#' Randomly selects actors as baseline comparison
#' @param network Adjacency matrix
#' @param proportion Proportion of actors to target
#' @return Vector of selected actor IDs
random_targeting <- function(network, proportion = 0.2) {
  n_select <- round(nrow(network) * proportion)
  selected <- sample(1:nrow(network), n_select)

  cat("Random targeting selected", length(selected), "actors\n")

  return(selected)
}

# ============================================================================
# SECTION 3: INTERVENTION IMPLEMENTATION
# ============================================================================

#' Apply tolerance intervention to selected actors
#'
#' Implements intervention by modifying tolerance levels and tracking effects
#' @param tolerance_matrix Matrix of tolerance levels across waves
#' @param intervention_actors Vector of actor IDs to intervene on
#' @param intervention_wave Wave at which intervention occurs
#' @param effect_size Size of intervention effect
#' @param decay_rate Rate at which intervention effects decay
#' @return Modified tolerance matrix with intervention effects
apply_tolerance_intervention <- function(tolerance_matrix, intervention_actors,
                                       intervention_wave = 2, effect_size = 0.3,
                                       decay_rate = 0.1) {

  cat("Applying tolerance intervention...\n")
  cat("- Target actors:", length(intervention_actors), "\n")
  cat("- Intervention wave:", intervention_wave, "\n")
  cat("- Effect size:", effect_size, "\n")

  n_waves <- ncol(tolerance_matrix)
  modified_tolerance <- tolerance_matrix

  # Apply intervention effect
  for (wave in intervention_wave:n_waves) {
    wave_effect <- effect_size * exp(-decay_rate * (wave - intervention_wave))

    # Boost tolerance for intervention actors
    modified_tolerance[intervention_actors, wave] <-
      pmin(1, modified_tolerance[intervention_actors, wave] + wave_effect)

    cat("Wave", wave, "intervention effect:", round(wave_effect, 3), "\n")
  }

  return(modified_tolerance)
}

#' Generate data with intervention scenarios
#'
#' Creates multiple datasets representing different intervention strategies
#' @param n_actors Number of actors
#' @param n_waves Number of observation waves
#' @param intervention_proportion Proportion of actors to target
#' @return List of datasets for each intervention strategy
generate_intervention_data <- function(n_actors = 60, n_waves = 4,
                                     intervention_proportion = 0.2) {

  cat("Generating intervention experiment data...\n")

  # Generate baseline tolerance and network data
  baseline_data <- generate_baseline_scenario(n_actors, n_waves)

  # Define intervention strategies
  strategies <- c("central", "peripheral", "clustered", "random", "control")

  # Create intervention datasets
  intervention_datasets <- list()

  for (strategy in strategies) {
    cat("\nPreparing", strategy, "intervention scenario...\n")

    # Copy baseline data
    scenario_data <- baseline_data

    if (strategy != "control") {
      # Select intervention targets
      initial_network <- baseline_data$friendship[,,1]

      intervention_actors <- switch(strategy,
        "central" = central_targeting(initial_network, intervention_proportion),
        "peripheral" = peripheral_targeting(initial_network, intervention_proportion),
        "clustered" = clustered_targeting(initial_network, intervention_proportion),
        "random" = random_targeting(initial_network, intervention_proportion)
      )

      # Apply intervention
      scenario_data$tolerance <- apply_tolerance_intervention(
        baseline_data$tolerance,
        intervention_actors,
        intervention_wave = 2,
        effect_size = 0.4,
        decay_rate = 0.15
      )

      scenario_data$intervention_actors <- intervention_actors
    } else {
      scenario_data$intervention_actors <- integer(0)
    }

    scenario_data$strategy <- strategy
    intervention_datasets[[strategy]] <- scenario_data
  }

  cat("\nIntervention datasets generated for", length(strategies), "strategies\n")
  return(intervention_datasets)
}

#' Generate baseline scenario with realistic tolerance dynamics
#'
#' Creates fundamental network and tolerance evolution without intervention
#' @param n_actors Number of actors
#' @param n_waves Number of waves
#' @return Baseline data structure
generate_baseline_scenario <- function(n_actors, n_waves) {

  # Initialize with lower baseline tolerance (more room for improvement)
  initial_tolerance <- pmax(0, pmin(1, rnorm(n_actors, mean = 0.35, sd = 0.25)))

  # Create networks and tolerance evolution
  friendship_networks <- array(0, dim = c(n_actors, n_actors, n_waves))
  tolerance_waves <- matrix(0, nrow = n_actors, ncol = n_waves)
  tolerance_waves[, 1] <- initial_tolerance

  # Generate initial network with moderate density
  for (i in 1:(n_actors-1)) {
    for (j in (i+1):n_actors) {
      # Base probability with tolerance similarity bonus
      tolerance_diff <- abs(initial_tolerance[i] - initial_tolerance[j])
      prob_friendship <- 0.15 * exp(-2 * tolerance_diff)

      if (runif(1) < prob_friendship) {
        friendship_networks[i, j, 1] <- 1
        friendship_networks[j, i, 1] <- 1
      }
    }
  }

  # Evolve network and tolerance over waves
  for (wave in 2:n_waves) {
    prev_tolerance <- tolerance_waves[, wave-1]
    prev_network <- friendship_networks[, , wave-1]

    # Update tolerance with peer influence
    new_tolerance <- prev_tolerance
    for (i in 1:n_actors) {
      friends <- which(prev_network[i, ] == 1)
      if (length(friends) > 0) {
        # Moderate positive influence
        peer_effect <- mean(prev_tolerance[friends]) - prev_tolerance[i]
        influence <- 0.1 * peer_effect
        new_tolerance[i] <- pmax(0, pmin(1, prev_tolerance[i] + influence +
                                       rnorm(1, 0, 0.03)))
      } else {
        new_tolerance[i] <- pmax(0, pmin(1, prev_tolerance[i] + rnorm(1, 0, 0.05)))
      }
    }

    tolerance_waves[, wave] <- new_tolerance

    # Update network structure
    friendship_networks[, , wave] <- prev_network

    # Some network changes based on tolerance evolution
    for (i in 1:(n_actors-1)) {
      for (j in (i+1):n_actors) {
        current_tie <- prev_network[i, j]
        tolerance_diff <- abs(new_tolerance[i] - new_tolerance[j])

        if (current_tie == 1) {
          # Dissolution based on growing tolerance differences
          prob_dissolve <- 0.05 * tolerance_diff^2
          if (runif(1) < prob_dissolve) {
            friendship_networks[i, j, wave] <- 0
            friendship_networks[j, i, wave] <- 0
          }
        } else {
          # Formation based on tolerance similarity
          prob_form <- 0.03 * exp(-3 * tolerance_diff)
          if (runif(1) < prob_form) {
            friendship_networks[i, j, wave] <- 1
            friendship_networks[j, i, wave] <- 1
          }
        }
      }
    }
  }

  return(list(
    friendship = friendship_networks,
    tolerance = tolerance_waves,
    n_actors = n_actors,
    n_waves = n_waves
  ))
}

# ============================================================================
# SECTION 4: STATISTICAL ANALYSIS AND COMPARISON
# ============================================================================

#' Analyze intervention effectiveness across strategies
#'
#' Compares outcomes across different intervention approaches
#' @param intervention_datasets List of datasets from different strategies
#' @return Statistical comparison results
analyze_intervention_effectiveness <- function(intervention_datasets) {

  cat("Analyzing intervention effectiveness...\n")

  # Calculate outcome measures for each strategy
  results_summary <- map_dfr(names(intervention_datasets), function(strategy) {
    data <- intervention_datasets[[strategy]]
    tolerance <- data$tolerance
    n_waves <- data$n_waves

    # Primary outcome: change in mean tolerance
    baseline_mean <- mean(tolerance[, 1])
    final_mean <- mean(tolerance[, n_waves])
    tolerance_change <- final_mean - baseline_mean

    # Secondary outcomes
    baseline_var <- var(tolerance[, 1])
    final_var <- var(tolerance[, n_waves])
    variance_change <- final_var - baseline_var

    high_tolerance_final <- sum(tolerance[, n_waves] > 0.8)
    high_tolerance_baseline <- sum(tolerance[, 1] > 0.8)
    high_tolerance_change <- high_tolerance_final - high_tolerance_baseline

    # Calculate spillover effects (for intervention strategies)
    spillover_effect <- NA
    if (strategy != "control" && length(data$intervention_actors) > 0) {
      non_intervention <- setdiff(1:nrow(tolerance), data$intervention_actors)
      if (length(non_intervention) > 0) {
        baseline_non_intervention <- mean(tolerance[non_intervention, 1])
        final_non_intervention <- mean(tolerance[non_intervention, n_waves])
        spillover_effect <- final_non_intervention - baseline_non_intervention
      }
    }

    # Direct effect on intervention actors
    direct_effect <- NA
    if (strategy != "control" && length(data$intervention_actors) > 0) {
      baseline_intervention <- mean(tolerance[data$intervention_actors, 1])
      final_intervention <- mean(tolerance[data$intervention_actors, n_waves])
      direct_effect <- final_intervention - baseline_intervention
    }

    tibble(
      strategy = strategy,
      tolerance_change = tolerance_change,
      variance_change = variance_change,
      high_tolerance_change = high_tolerance_change,
      spillover_effect = spillover_effect,
      direct_effect = direct_effect,
      final_mean_tolerance = final_mean,
      n_intervention = ifelse(strategy == "control", 0, length(data$intervention_actors))
    )
  })

  # Statistical tests
  cat("\nSTATISTICAL COMPARISONS:\n")
  cat(paste(rep("-", 40), collapse = ""), "\n")

  # Compare each intervention to control
  control_effect <- results_summary$tolerance_change[results_summary$strategy == "control"]

  intervention_strategies <- setdiff(results_summary$strategy, "control")

  statistical_tests <- map_dfr(intervention_strategies, function(strategy) {
    strategy_effect <- results_summary$tolerance_change[results_summary$strategy == strategy]

    # Effect size calculation (Cohen's d approximation)
    pooled_sd <- sqrt((var(intervention_datasets[[strategy]]$tolerance[,1]) +
                      var(intervention_datasets[["control"]]$tolerance[,1])) / 2)
    cohens_d <- (strategy_effect - control_effect) / pooled_sd

    # Simple comparison (in practice, would use proper statistical tests)
    improvement_over_control <- strategy_effect - control_effect

    tibble(
      strategy = strategy,
      effect_vs_control = improvement_over_control,
      cohens_d = cohens_d,
      magnitude = case_when(
        abs(cohens_d) < 0.2 ~ "negligible",
        abs(cohens_d) < 0.5 ~ "small",
        abs(cohens_d) < 0.8 ~ "medium",
        TRUE ~ "large"
      )
    )
  })

  # Print results
  cat("INTERVENTION EFFECTIVENESS SUMMARY:\n")
  print(kable(results_summary, digits = 3))

  cat("\nSTATISTICAL EFFECT SIZES:\n")
  print(kable(statistical_tests, digits = 3))

  # Ranking of strategies
  strategy_ranking <- results_summary %>%
    filter(strategy != "control") %>%
    arrange(desc(tolerance_change)) %>%
    mutate(rank = row_number())

  cat("\nSTRATEGY EFFECTIVENESS RANKING:\n")
  for (i in 1:nrow(strategy_ranking)) {
    cat(sprintf("%d. %s (Δ = %.3f)\n",
                strategy_ranking$rank[i],
                strategy_ranking$strategy[i],
                strategy_ranking$tolerance_change[i]))
  }

  return(list(
    summary = results_summary,
    statistical_tests = statistical_tests,
    ranking = strategy_ranking
  ))
}

# ============================================================================
# SECTION 5: POLICY RECOMMENDATIONS AND IMPLICATIONS
# ============================================================================

#' Generate policy recommendations based on intervention results
#'
#' Translates statistical findings into actionable policy guidance
#' @param analysis_results Results from intervention effectiveness analysis
#' @return Policy recommendations with implementation guidance
generate_policy_recommendations <- function(analysis_results) {

  cat("\n", paste(rep("=", 60), collapse = ""), "\n")
  cat("POLICY RECOMMENDATIONS FOR TOLERANCE INTERVENTIONS\n")
  cat(paste(rep("=", 60), collapse = ""), "\n")

  # Extract key findings
  summary <- analysis_results$summary
  tests <- analysis_results$statistical_tests
  ranking <- analysis_results$ranking

  best_strategy <- ranking$strategy[1]
  best_effect <- ranking$tolerance_change[1]

  # Generate recommendations
  recommendations <- list()

  # Primary recommendation
  recommendations$primary <- sprintf(
    "RECOMMENDED STRATEGY: %s targeting shows the strongest intervention effects (Δ = %.3f)",
    toupper(best_strategy), best_effect
  )

  # Secondary recommendations based on findings
  recommendations$secondary <- c()

  # Spillover effects
  best_spillover <- summary %>%
    filter(!is.na(spillover_effect)) %>%
    arrange(desc(spillover_effect)) %>%
    slice(1)

  if (nrow(best_spillover) > 0) {
    recommendations$secondary <- c(recommendations$secondary,
      sprintf("For maximum spillover effects, consider %s targeting (spillover = %.3f)",
              best_spillover$strategy, best_spillover$spillover_effect))
  }

  # Cost-effectiveness considerations
  cost_effectiveness <- summary %>%
    filter(strategy != "control") %>%
    mutate(effect_per_person = tolerance_change / n_intervention) %>%
    arrange(desc(effect_per_person))

  if (nrow(cost_effectiveness) > 0) {
    most_efficient <- cost_effectiveness$strategy[1]
    recommendations$secondary <- c(recommendations$secondary,
      sprintf("For resource-constrained implementations, %s targeting offers best cost-effectiveness",
              most_efficient))
  }

  # Implementation guidance
  recommendations$implementation <- c(
    "Implement interventions early in network formation periods",
    "Monitor intervention decay and plan reinforcement activities",
    "Consider network density when selecting targeting strategies",
    "Combine direct intervention with spillover optimization",
    "Establish baseline measurements before intervention deployment"
  )

  # Risk factors and limitations
  recommendations$limitations <- c(
    "Effects may vary across different population characteristics",
    "Long-term sustainability requires ongoing support mechanisms",
    "Network structure changes may affect intervention effectiveness",
    "Individual resistance to change should be anticipated",
    "Measurement validity depends on honest tolerance reporting"
  )

  # Print recommendations
  cat("\n1. PRIMARY RECOMMENDATION:\n")
  cat("   ", recommendations$primary, "\n")

  cat("\n2. SECONDARY RECOMMENDATIONS:\n")
  for (rec in recommendations$secondary) {
    cat("   • ", rec, "\n")
  }

  cat("\n3. IMPLEMENTATION GUIDANCE:\n")
  for (guide in recommendations$implementation) {
    cat("   • ", guide, "\n")
  }

  cat("\n4. LIMITATIONS AND CONSIDERATIONS:\n")
  for (limit in recommendations$limitations) {
    cat("   • ", limit, "\n")
  }

  # Evidence strength assessment
  cat("\n5. EVIDENCE STRENGTH:\n")
  large_effects <- sum(tests$magnitude == "large")
  medium_effects <- sum(tests$magnitude == "medium")

  if (large_effects > 0) {
    cat("   ✓ STRONG evidence base with", large_effects, "large effect sizes\n")
  } else if (medium_effects > 0) {
    cat("   ○ MODERATE evidence base with", medium_effects, "medium effect sizes\n")
  } else {
    cat("   ⚠ WEAK evidence base - small effect sizes detected\n")
  }

  return(recommendations)
}

# ============================================================================
# SECTION 6: COMPREHENSIVE VISUALIZATION
# ============================================================================

#' Create publication-ready intervention comparison visualizations
#'
#' Generates comprehensive plots comparing intervention strategies
#' @param intervention_datasets List of intervention datasets
#' @param analysis_results Analysis results
create_intervention_visualizations <- function(intervention_datasets, analysis_results) {

  cat("\nCreating intervention comparison visualizations...\n")

  # Prepare data for plotting
  plot_data <- map_dfr(names(intervention_datasets), function(strategy) {
    data <- intervention_datasets[[strategy]]
    tolerance <- data$tolerance
    n_waves <- ncol(tolerance)

    # Create long format data
    tolerance_long <- expand_grid(
      actor = 1:nrow(tolerance),
      wave = 1:n_waves
    ) %>%
      mutate(
        tolerance = as.vector(tolerance),
        strategy = strategy,
        intervention_status = ifelse(
          strategy != "control" & actor %in% data$intervention_actors,
          "Intervention", "Control"
        )
      )

    return(tolerance_long)
  })

  # Plot 1: Strategy comparison over time
  p1 <- plot_data %>%
    group_by(strategy, wave) %>%
    summarise(
      mean_tolerance = mean(tolerance),
      se_tolerance = sd(tolerance) / sqrt(n()),
      .groups = "drop"
    ) %>%
    ggplot(aes(x = wave, y = mean_tolerance, color = strategy)) +
    geom_line(size = 1.2) +
    geom_point(size = 2) +
    geom_errorbar(aes(ymin = mean_tolerance - se_tolerance,
                     ymax = mean_tolerance + se_tolerance),
                  width = 0.1) +
    scale_color_brewer(palette = "Set1", name = "Strategy") +
    labs(
      title = "Tolerance Evolution by Intervention Strategy",
      x = "Wave",
      y = "Mean Tolerance Level",
      subtitle = "Error bars show standard error"
    ) +
    theme_minimal() +
    theme(legend.position = "bottom")

  # Plot 2: Final outcomes comparison
  final_outcomes <- analysis_results$summary %>%
    filter(strategy != "control") %>%
    select(strategy, tolerance_change, spillover_effect, direct_effect) %>%
    pivot_longer(cols = c(tolerance_change, spillover_effect, direct_effect),
                names_to = "outcome_type",
                values_to = "effect_size") %>%
    filter(!is.na(effect_size))

  p2 <- ggplot(final_outcomes, aes(x = strategy, y = effect_size, fill = outcome_type)) +
    geom_col(position = "dodge", alpha = 0.8) +
    scale_fill_brewer(palette = "Set2",
                     name = "Effect Type",
                     labels = c("Direct Effect", "Spillover Effect", "Overall Change")) +
    labs(
      title = "Intervention Effects by Strategy and Type",
      x = "Intervention Strategy",
      y = "Effect Size (Tolerance Change)"
    ) +
    theme_minimal() +
    theme(legend.position = "bottom")

  # Plot 3: Network visualization for best strategy
  best_strategy <- analysis_results$ranking$strategy[1]
  best_data <- intervention_datasets[[best_strategy]]

  # Create network plots for wave 1 and final wave
  create_network_comparison <- function(data, strategy_name) {
    friendship <- data$friendship
    tolerance <- data$tolerance
    n_waves <- dim(friendship)[3]

    par(mfrow = c(1, 2))

    for (wave in c(1, n_waves)) {
      g <- graph_from_adjacency_matrix(friendship[,,wave], mode = "undirected")

      # Color nodes by tolerance level
      tolerance_colors <- colorRampPalette(c("red", "yellow", "green"))(100)
      tolerance_indices <- pmax(1, pmin(100, round(tolerance[,wave] * 100)))
      V(g)$color <- tolerance_colors[tolerance_indices]

      # Highlight intervention actors
      if ("intervention_actors" %in% names(data) && length(data$intervention_actors) > 0) {
        V(g)$frame.color <- "black"
        V(g)$frame.color[data$intervention_actors] <- "blue"
        V(g)$frame.width <- 1
        V(g)$frame.width[data$intervention_actors] <- 3
      }

      V(g)$size <- 8

      plot(g,
           main = paste(strategy_name, "Strategy - Wave", wave),
           vertex.label = NA,
           edge.color = "gray70",
           edge.width = 1,
           layout = layout_with_fr(g))

      if (wave == 1) {
        legend("topright",
               legend = c("Low Tolerance", "High Tolerance", "Intervention Actor"),
               col = c("red", "green", "white"),
               pch = c(19, 19, 21),
               pt.bg = c("red", "green", "white"),
               pt.cex = c(1, 1, 1.5),
               cex = 0.8)
      }
    }

    par(mfrow = c(1, 1))
  }

  # Display plots
  print(p1)
  print(p2)

  create_network_comparison(best_data, paste("Best Performing:", best_strategy))

  # Plot 4: Effect size comparison
  effect_comparison <- analysis_results$statistical_tests %>%
    ggplot(aes(x = reorder(strategy, effect_vs_control), y = effect_vs_control)) +
    geom_col(aes(fill = magnitude), alpha = 0.8) +
    geom_hline(yintercept = 0, linetype = "dashed") +
    scale_fill_manual(values = c("negligible" = "gray",
                                "small" = "lightblue",
                                "medium" = "orange",
                                "large" = "red"),
                     name = "Effect Magnitude") +
    labs(
      title = "Intervention Effects Relative to Control",
      x = "Intervention Strategy",
      y = "Effect Size vs. Control",
      subtitle = "Positive values indicate improvement over control condition"
    ) +
    theme_minimal() +
    theme(legend.position = "bottom") +
    coord_flip()

  print(effect_comparison)

  cat("Visualization suite complete!\n")
}

# ============================================================================
# SECTION 7: MAIN EXECUTION WORKFLOW
# ============================================================================

#' Run complete intervention simulation study
#'
#' Executes full research workflow from hypothesis to policy recommendations
run_intervention_simulation_demo <- function() {

  cat(paste(rep("=", 80), collapse = ""), "\n")
  cat("RSIENA TOLERANCE INTERVENTION: SIMULATION STUDY\n")
  cat(paste(rep("=", 80), collapse = ""), "\n")

  # Step 1: Formulate hypotheses
  cat("\nSTEP 1: RESEARCH DESIGN\n")
  cat(paste(rep("-", 30), collapse = ""), "\n")
  hypotheses <- formulate_research_hypotheses()

  # Step 2: Generate intervention datasets
  cat("\nSTEP 2: DATA GENERATION\n")
  cat(paste(rep("-", 30), collapse = ""), "\n")
  intervention_data <- generate_intervention_data(
    n_actors = 60,
    n_waves = 4,
    intervention_proportion = 0.25
  )

  # Step 3: Analyze effectiveness
  cat("\nSTEP 3: STATISTICAL ANALYSIS\n")
  cat(paste(rep("-", 30), collapse = ""), "\n")
  analysis_results <- analyze_intervention_effectiveness(intervention_data)

  # Step 4: Generate policy recommendations
  cat("\nSTEP 4: POLICY IMPLICATIONS\n")
  cat(paste(rep("-", 30), collapse = ""), "\n")
  recommendations <- generate_policy_recommendations(analysis_results)

  # Step 5: Create visualizations
  cat("\nSTEP 5: VISUALIZATION\n")
  cat(paste(rep("-", 30), collapse = ""), "\n")
  create_intervention_visualizations(intervention_data, analysis_results)

  # Step 6: Generate research report
  cat("\nSTEP 6: RESEARCH SUMMARY\n")
  cat(paste(rep("-", 30), collapse = ""), "\n")

  cat("\nRESEARCH FINDINGS SUMMARY:\n")
  cat("• Most effective strategy:", analysis_results$ranking$strategy[1], "\n")
  cat("• Effect size range:",
      round(min(analysis_results$summary$tolerance_change[analysis_results$summary$strategy != "control"]), 3),
      "to",
      round(max(analysis_results$summary$tolerance_change[analysis_results$summary$strategy != "control"]), 3), "\n")
  cat("• Strategies with large effects:",
      sum(analysis_results$statistical_tests$magnitude == "large"), "\n")

  cat("\n", paste(rep("=", 80), collapse = ""), "\n")
  cat("INTERVENTION SIMULATION STUDY COMPLETE!\n")
  cat(paste(rep("=", 80), collapse = ""), "\n")

  # Return comprehensive results
  return(list(
    hypotheses = hypotheses,
    data = intervention_data,
    analysis = analysis_results,
    recommendations = recommendations
  ))
}

# ============================================================================
# EXECUTION
# ============================================================================

# Run the complete intervention simulation
if (interactive() || !exists("skip_demo")) {
  simulation_results <- run_intervention_simulation_demo()

  # Save results
  save(simulation_results, file = "intervention_simulation_results.RData")
  cat("\nResults saved to: intervention_simulation_results.RData\n")
}

# ============================================================================
# RESEARCH METHODOLOGY NOTES
# ============================================================================

cat("\n", paste(rep("=", 80), collapse = ""), "\n")
cat("RESEARCH METHODOLOGY: INTERVENTION SIMULATION STUDIES\n")
cat(paste(rep("=", 80), collapse = ""), "\n")

cat("\n1. EXPERIMENTAL DESIGN PRINCIPLES:\n")
cat("   • Multiple comparison groups for robust inference\n")
cat("   • Randomization to control for confounding factors\n")
cat("   • Adequate sample sizes for statistical power\n")
cat("   • Pre-specified outcome measures and analysis plans\n")

cat("\n2. INTERVENTION TARGETING THEORY:\n")
cat("   • Central: Leverage high-degree actors for maximum reach\n")
cat("   • Peripheral: Test diffusion from network edges\n")
cat("   • Clustered: Achieve local saturation effects\n")
cat("   • Random: Provide unbiased comparison baseline\n")

cat("\n3. STATISTICAL CONSIDERATIONS:\n")
cat("   • Multiple testing corrections for family-wise error rates\n")
cat("   • Effect size interpretation beyond statistical significance\n")
cat("   • Confidence intervals for parameter estimates\n")
cat("   • Sensitivity analysis for key assumptions\n")

cat("\n4. POLICY TRANSLATION:\n")
cat("   • Clear actionable recommendations\n")
cat("   • Cost-effectiveness considerations\n")
cat("   • Implementation feasibility assessment\n")
cat("   • Limitation acknowledgment and risk factors\n")

cat("\n5. FUTURE EXTENSIONS:\n")
cat("   • Longitudinal follow-up studies\n")
cat("   • Multi-site replication studies\n")
cat("   • Moderation analysis by population characteristics\n")
cat("   • Economic evaluation frameworks\n")

cat("\nFor technical implementation details, see:\n")
cat("   • custom_effects_demo.R\n")
cat("   • tolerance_basic_demo.R\n")

cat("\n", paste(rep("=", 80), collapse = ""), "\n")