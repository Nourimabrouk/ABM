# ============================================================================
# RSIENA TOLERANCE INTERVENTION RESEARCH: BASIC DEMONSTRATION
# ============================================================================
#
# Demo 1: Basic RSiena Tolerance Model
# Complete workflow: Data → Model → Analysis → Visualization
# Focus: Simple attraction-repulsion influence on tolerance
#
# Author: ABM Research Project
# Date: 2025
# Purpose: Demonstrate fundamental RSiena workflow for tolerance research
# ============================================================================

# Load required libraries
library(RSiena)
library(network)
library(sna)
library(igraph)
library(ggplot2)
library(dplyr)
library(RColorBrewer)
library(gridExtra)

# Set random seed for reproducibility
set.seed(42)

# ============================================================================
# SECTION 1: DATA PREPARATION AND GENERATION
# ============================================================================

#' Generate synthetic tolerance and friendship networks
#'
#' Creates realistic tolerance levels and friendship networks for demonstration
#' @param n_actors Number of actors in the network
#' @param n_waves Number of observation waves
#' @param tolerance_sd Standard deviation for tolerance initialization
#' @return List containing friendship networks and tolerance data
generate_tolerance_data <- function(n_actors = 50, n_waves = 3, tolerance_sd = 0.3) {

  cat("Generating synthetic tolerance intervention data...\n")
  cat("- Actors:", n_actors, "\n")
  cat("- Waves:", n_waves, "\n")

  # Initialize tolerance levels (0 = intolerant, 1 = tolerant)
  # Start with normal distribution around 0.5
  initial_tolerance <- pmax(0, pmin(1, rnorm(n_actors, mean = 0.5, sd = tolerance_sd)))

  # Create friendship networks for each wave
  friendship_networks <- array(0, dim = c(n_actors, n_actors, n_waves))
  tolerance_waves <- matrix(0, nrow = n_actors, ncol = n_waves)

  # Wave 1: Initial state
  tolerance_waves[, 1] <- initial_tolerance

  # Generate initial friendship network based on tolerance similarity
  for (i in 1:(n_actors-1)) {
    for (j in (i+1):n_actors) {
      # Probability of friendship based on tolerance similarity
      tolerance_diff <- abs(tolerance_waves[i, 1] - tolerance_waves[j, 1])
      prob_friendship <- exp(-3 * tolerance_diff) * 0.3  # Similarity attracts

      if (runif(1) < prob_friendship) {
        friendship_networks[i, j, 1] <- 1
        friendship_networks[j, i, 1] <- 1
      }
    }
  }

  # Generate subsequent waves with influence dynamics
  for (wave in 2:n_waves) {
    # Previous wave values
    prev_tolerance <- tolerance_waves[, wave-1]
    prev_network <- friendship_networks[, , wave-1]

    # Calculate influence effects
    new_tolerance <- prev_tolerance

    for (i in 1:n_actors) {
      # Find friends
      friends <- which(prev_network[i, ] == 1)

      if (length(friends) > 0) {
        # Attraction-repulsion influence
        friend_tolerance <- prev_tolerance[friends]

        # Attraction effect: similar values attract
        attraction_effect <- 0
        for (friend_tol in friend_tolerance) {
          similarity <- 1 - abs(prev_tolerance[i] - friend_tol)
          attraction_effect <- attraction_effect + 0.05 * similarity *
                             sign(friend_tol - prev_tolerance[i])
        }

        # Repulsion effect: very different values repel
        repulsion_effect <- 0
        for (friend_tol in friend_tolerance) {
          difference <- abs(prev_tolerance[i] - friend_tol)
          if (difference > 0.7) {  # Threshold for repulsion
            repulsion_effect <- repulsion_effect - 0.03 *
                               sign(friend_tol - prev_tolerance[i])
          }
        }

        # Update tolerance with bounds
        influence <- (attraction_effect + repulsion_effect) / length(friends)
        new_tolerance[i] <- pmax(0, pmin(1, prev_tolerance[i] + influence +
                                        rnorm(1, 0, 0.02)))
      } else {
        # Random walk for isolated actors
        new_tolerance[i] <- pmax(0, pmin(1, prev_tolerance[i] + rnorm(1, 0, 0.05)))
      }
    }

    tolerance_waves[, wave] <- new_tolerance

    # Update friendship network based on new tolerance levels
    friendship_networks[, , wave] <- friendship_networks[, , wave-1]

    # Some friendship changes based on tolerance changes
    for (i in 1:(n_actors-1)) {
      for (j in (i+1):n_actors) {
        current_tie <- friendship_networks[i, j, wave-1]
        tolerance_diff <- abs(new_tolerance[i] - new_tolerance[j])

        if (current_tie == 1) {
          # Dissolution probability increases with tolerance difference
          prob_dissolve <- 0.1 * tolerance_diff^2
          if (runif(1) < prob_dissolve) {
            friendship_networks[i, j, wave] <- 0
            friendship_networks[j, i, wave] <- 0
          }
        } else {
          # Formation probability decreases with tolerance difference
          prob_form <- exp(-2 * tolerance_diff) * 0.05
          if (runif(1) < prob_form) {
            friendship_networks[i, j, wave] <- 1
            friendship_networks[j, i, wave] <- 1
          }
        }
      }
    }
  }

  # Add actor attributes
  actor_attributes <- data.frame(
    id = 1:n_actors,
    age = sample(18:65, n_actors, replace = TRUE),
    education = sample(c("High School", "College", "Graduate"), n_actors,
                      replace = TRUE, prob = c(0.4, 0.4, 0.2)),
    initial_tolerance = initial_tolerance
  )

  cat("Data generation complete!\n")
  cat("- Network density wave 1:", round(sum(friendship_networks[,,1])/(n_actors*(n_actors-1)), 3), "\n")
  cat("- Network density wave", n_waves, ":", round(sum(friendship_networks[,,n_waves])/(n_actors*(n_actors-1)), 3), "\n")
  cat("- Tolerance range wave 1:", round(range(tolerance_waves[,1]), 3), "\n")
  cat("- Tolerance range wave", n_waves, ":", round(range(tolerance_waves[,n_waves]), 3), "\n")

  return(list(
    friendship = friendship_networks,
    tolerance = tolerance_waves,
    attributes = actor_attributes,
    n_actors = n_actors,
    n_waves = n_waves
  ))
}

# ============================================================================
# SECTION 2: RSIENA MODEL SPECIFICATION
# ============================================================================

#' Prepare RSiena data objects
#'
#' Converts raw data into RSiena format for analysis
#' @param data_list Output from generate_tolerance_data()
#' @return RSiena data object ready for analysis
prepare_siena_data <- function(data_list) {

  cat("\nPreparing RSiena data objects...\n")

  # Extract data components
  friendship <- data_list$friendship
  tolerance <- data_list$tolerance
  attributes <- data_list$attributes

  # Create dependent network variable
  friendship_siena <- sienaDependent(friendship, type = "oneMode")

  # Create dependent behavior variable (tolerance)
  tolerance_siena <- sienaDependent(tolerance, type = "behavior")

  # Create individual covariates
  age_covar <- coCovar(attributes$age)
  education_covar <- coCovar(as.numeric(as.factor(attributes$education)))

  # Combine into RSiena data object
  siena_data <- sienaDataCreate(
    friendship = friendship_siena,
    tolerance = tolerance_siena,
    age = age_covar,
    education = education_covar
  )

  cat("RSiena data object created successfully!\n")
  cat("- Network variable: friendship\n")
  cat("- Behavior variable: tolerance\n")
  cat("- Covariates: age, education\n")

  return(siena_data)
}

#' Specify RSiena model with tolerance-specific effects
#'
#' Creates model specification including attraction-repulsion effects
#' @param siena_data RSiena data object
#' @return RSiena effects object
specify_tolerance_model <- function(siena_data) {

  cat("\nSpecifying tolerance influence model...\n")

  # Get basic effects
  basic_effects <- getEffects(siena_data)

  # Network evolution effects
  basic_effects <- includeEffects(basic_effects,
                                 transTrip,      # Transitivity
                                 cycle3,         # 3-cycles
                                 name = "friendship")

  # Behavior-network selection effects
  basic_effects <- includeEffects(basic_effects,
                                 simX,           # Similarity selection
                                 name = "friendship",
                                 interaction1 = "tolerance")

  # Network-behavior influence effects
  basic_effects <- includeEffects(basic_effects,
                                 avAlt,          # Average alter behavior
                                 name = "tolerance",
                                 interaction1 = "friendship")

  # Individual attribute effects
  basic_effects <- includeEffects(basic_effects,
                                 egoX,           # Ego effect
                                 altX,           # Alter effect
                                 name = "friendship",
                                 interaction1 = "age")

  basic_effects <- includeEffects(basic_effects,
                                 effFrom,        # Covariate effect on behavior
                                 name = "tolerance",
                                 interaction1 = "education")

  cat("Model specification complete!\n")
  print(basic_effects[basic_effects$include == TRUE,
                     c("name", "effectName", "type", "include")])

  return(basic_effects)
}

# ============================================================================
# SECTION 3: MODEL ESTIMATION AND DIAGNOSTICS
# ============================================================================

#' Estimate RSiena model with convergence checking
#'
#' Runs estimation algorithm with proper diagnostics
#' @param siena_data RSiena data object
#' @param effects RSiena effects object
#' @param max_iterations Maximum number of iterations
#' @return RSiena results object
estimate_tolerance_model <- function(siena_data, effects, max_iterations = 3000) {

  cat("\nEstimating tolerance influence model...\n")
  cat("This may take several minutes...\n")

  # Set algorithm parameters
  algorithm <- sienaAlgorithmCreate(
    projname = "tolerance_basic",
    nsub = 4,
    n3 = max_iterations,
    MaxDegree = c(friendship = 6),
    seed = 42
  )

  # Estimate model
  results <- siena07(algorithm, data = siena_data, effects = effects,
                    returnDeps = TRUE, verbose = TRUE)

  # Check convergence
  cat("\nConvergence diagnostics:\n")
  convergence_check <- results$tconv.max < 0.25 & results$tconv.max > -0.25

  if (convergence_check) {
    cat("✓ Model converged successfully (t-ratio < 0.25)\n")
  } else {
    cat("⚠ Convergence issues detected (t-ratio >= 0.25)\n")
    cat("Consider running additional iterations\n")
  }

  cat("Maximum t-ratio for convergence:", round(results$tconv.max, 3), "\n")

  return(results)
}

#' Perform goodness of fit testing
#'
#' Tests how well the model reproduces observed network statistics
#' @param results RSiena results object
#' @param siena_data RSiena data object
#' @return Goodness of fit object
assess_model_fit <- function(results, siena_data) {

  cat("\nAssessing model goodness of fit...\n")
  cat("This may take a few minutes...\n")

  # Test fit for key network statistics
  gof_indegree <- sienaGOF(results, siena_data, IndegreeDistribution,
                          verbose = TRUE, join = TRUE)
  gof_outdegree <- sienaGOF(results, siena_data, OutdegreeDistribution,
                           verbose = TRUE, join = TRUE)
  gof_geodesic <- sienaGOF(results, siena_data, GeodesicDistribution,
                          verbose = TRUE, join = TRUE)

  # Print fit summaries
  cat("\nGoodness of fit results:\n")
  cat("Indegree distribution p-value:", round(gof_indegree$pValue, 3), "\n")
  cat("Outdegree distribution p-value:", round(gof_outdegree$pValue, 3), "\n")
  cat("Geodesic distribution p-value:", round(gof_geodesic$pValue, 3), "\n")

  if (gof_indegree$pValue > 0.05 & gof_outdegree$pValue > 0.05) {
    cat("✓ Model fit appears adequate (p > 0.05)\n")
  } else {
    cat("⚠ Some fit issues detected (p < 0.05)\n")
  }

  return(list(
    indegree = gof_indegree,
    outdegree = gof_outdegree,
    geodesic = gof_geodesic
  ))
}

# ============================================================================
# SECTION 4: RESULTS INTERPRETATION AND VISUALIZATION
# ============================================================================

#' Extract and interpret key model parameters
#'
#' Provides substantive interpretation of tolerance influence effects
#' @param results RSiena results object
#' @return Formatted results summary
interpret_tolerance_results <- function(results) {

  cat("\n" + paste(rep("=", 60), collapse = "") + "\n")
  cat("TOLERANCE INFLUENCE MODEL RESULTS\n")
  cat(paste(rep("=", 60), collapse = "") + "\n")

  # Extract parameter estimates
  estimates <- results$theta
  standard_errors <- sqrt(diag(results$covtheta))
  t_ratios <- estimates / standard_errors
  p_values <- 2 * (1 - pnorm(abs(t_ratios)))

  # Create results table
  effect_names <- results$requestedEffects$effectName[results$requestedEffects$include]
  var_names <- results$requestedEffects$name[results$requestedEffects$include]

  results_table <- data.frame(
    Variable = var_names,
    Effect = effect_names,
    Estimate = round(estimates, 3),
    SE = round(standard_errors, 3),
    t_ratio = round(t_ratios, 3),
    p_value = round(p_values, 3),
    Significant = ifelse(p_values < 0.05, "***",
                        ifelse(p_values < 0.10, "*", ""))
  )

  print(results_table)

  # Substantive interpretation
  cat("\nSUBSTANTIVE INTERPRETATION:\n")
  cat(paste(rep("-", 40), collapse = "") + "\n")

  # Find key effects
  influence_effect <- which(grepl("avAlt", effect_names) & grepl("tolerance", var_names))
  selection_effect <- which(grepl("simX", effect_names) & grepl("friendship", var_names))

  if (length(influence_effect) > 0) {
    influence_est <- estimates[influence_effect]
    influence_p <- p_values[influence_effect]

    if (influence_p < 0.05) {
      if (influence_est > 0) {
        cat("✓ POSITIVE TOLERANCE INFLUENCE detected\n")
        cat("  → Friends tend to become more similar in tolerance over time\n")
        cat("  → Effect size:", round(influence_est, 3), "\n")
      } else {
        cat("✓ NEGATIVE TOLERANCE INFLUENCE detected\n")
        cat("  → Friends tend to become more different in tolerance over time\n")
        cat("  → Effect size:", round(influence_est, 3), "\n")
      }
    } else {
      cat("○ No significant tolerance influence detected\n")
    }
  }

  if (length(selection_effect) > 0) {
    selection_est <- estimates[selection_effect]
    selection_p <- p_values[selection_effect]

    if (selection_p < 0.05) {
      if (selection_est > 0) {
        cat("✓ TOLERANCE-BASED SELECTION detected\n")
        cat("  → Actors prefer friends with similar tolerance levels\n")
        cat("  → Effect size:", round(selection_est, 3), "\n")
      } else {
        cat("✓ TOLERANCE-BASED AVOIDANCE detected\n")
        cat("  → Actors avoid friends with similar tolerance levels\n")
        cat("  → Effect size:", round(selection_est, 3), "\n")
      }
    } else {
      cat("○ No significant tolerance-based selection detected\n")
    }
  }

  return(results_table)
}

#' Create comprehensive visualization of results
#'
#' Generates publication-ready plots of networks and tolerance dynamics
#' @param data_list Original data
#' @param results RSiena results object
#' @param gof_results Goodness of fit results
create_tolerance_visualizations <- function(data_list, results, gof_results) {

  cat("\nCreating tolerance intervention visualizations...\n")

  # Extract data
  friendship <- data_list$friendship
  tolerance <- data_list$tolerance
  n_waves <- data_list$n_waves
  n_actors <- data_list$n_actors

  # Color palette for tolerance levels
  tolerance_colors <- colorRampPalette(c("red", "yellow", "green"))(100)

  # Plot 1: Network evolution with tolerance coloring
  par(mfrow = c(2, 2))

  for (wave in c(1, n_waves)) {
    # Create igraph object
    g <- graph_from_adjacency_matrix(friendship[,,wave], mode = "undirected")

    # Set node colors based on tolerance
    tolerance_indices <- pmax(1, pmin(100, round(tolerance[,wave] * 100)))
    V(g)$color <- tolerance_colors[tolerance_indices]
    V(g)$size <- 8

    # Plot network
    plot(g,
         main = paste("Network Wave", wave, "\n(Color = Tolerance Level)"),
         vertex.label = NA,
         edge.color = "gray70",
         edge.width = 1,
         layout = layout_with_fr(g))

    # Add legend
    legend("topright",
           legend = c("Low Tolerance", "High Tolerance"),
           col = c("red", "green"),
           pch = 19,
           cex = 0.8)
  }

  # Plot 3: Tolerance distribution changes
  tolerance_df <- data.frame(
    tolerance = as.vector(tolerance),
    wave = rep(1:n_waves, each = n_actors),
    actor = rep(1:n_actors, n_waves)
  )

  p1 <- ggplot(tolerance_df, aes(x = tolerance, fill = factor(wave))) +
    geom_density(alpha = 0.6) +
    scale_fill_brewer(palette = "Set1", name = "Wave") +
    labs(title = "Tolerance Distribution Evolution",
         x = "Tolerance Level",
         y = "Density") +
    theme_minimal()

  print(p1)

  # Plot 4: Individual tolerance trajectories (sample)
  sample_actors <- sample(1:n_actors, min(10, n_actors))
  trajectory_df <- tolerance_df[tolerance_df$actor %in% sample_actors, ]

  p2 <- ggplot(trajectory_df, aes(x = wave, y = tolerance,
                                 group = actor, color = factor(actor))) +
    geom_line(alpha = 0.7) +
    geom_point(alpha = 0.7) +
    labs(title = "Individual Tolerance Trajectories (Sample)",
         x = "Wave",
         y = "Tolerance Level") +
    theme_minimal() +
    theme(legend.position = "none")

  print(p2)

  # Plot 5: Goodness of fit visualization
  if (!is.null(gof_results)) {
    par(mfrow = c(1, 2))

    plot(gof_results$indegree, main = "Indegree Distribution Fit")
    plot(gof_results$outdegree, main = "Outdegree Distribution Fit")
  }

  # Reset plot parameters
  par(mfrow = c(1, 1))

  cat("Visualizations complete!\n")
}

# ============================================================================
# SECTION 5: MAIN EXECUTION WORKFLOW
# ============================================================================

#' Main function to run complete tolerance analysis
#'
#' Executes full workflow from data generation to results
run_tolerance_basic_demo <- function() {

  cat("=" * 80 + "\n")
  cat("RSIENA TOLERANCE INTERVENTION: BASIC DEMONSTRATION\n")
  cat("=" * 80 + "\n")

  # Step 1: Generate data
  cat("\nSTEP 1: DATA GENERATION\n")
  cat("-" * 30 + "\n")
  tolerance_data <- generate_tolerance_data(n_actors = 50, n_waves = 3)

  # Step 2: Prepare RSiena data
  cat("\nSTEP 2: DATA PREPARATION\n")
  cat("-" * 30 + "\n")
  siena_data <- prepare_siena_data(tolerance_data)

  # Step 3: Specify model
  cat("\nSTEP 3: MODEL SPECIFICATION\n")
  cat("-" * 30 + "\n")
  effects <- specify_tolerance_model(siena_data)

  # Step 4: Estimate model
  cat("\nSTEP 4: MODEL ESTIMATION\n")
  cat("-" * 30 + "\n")
  results <- estimate_tolerance_model(siena_data, effects)

  # Step 5: Assess fit
  cat("\nSTEP 5: GOODNESS OF FIT\n")
  cat("-" * 30 + "\n")
  gof_results <- assess_model_fit(results, siena_data)

  # Step 6: Interpret results
  cat("\nSTEP 6: RESULTS INTERPRETATION\n")
  cat("-" * 30 + "\n")
  results_table <- interpret_tolerance_results(results)

  # Step 7: Create visualizations
  cat("\nSTEP 7: VISUALIZATION\n")
  cat("-" * 30 + "\n")
  create_tolerance_visualizations(tolerance_data, results, gof_results)

  cat("\n" + "=" * 80 + "\n")
  cat("TOLERANCE BASIC DEMONSTRATION COMPLETE!\n")
  cat("=" * 80 + "\n")

  # Return all results for further analysis
  return(list(
    data = tolerance_data,
    siena_data = siena_data,
    effects = effects,
    results = results,
    gof = gof_results,
    summary = results_table
  ))
}

# ============================================================================
# EXECUTION
# ============================================================================

# Run the complete demonstration
if (interactive() || !exists("skip_demo")) {
  demo_results <- run_tolerance_basic_demo()

  # Save results for future use
  save(demo_results, file = "tolerance_basic_demo_results.RData")
  cat("\nResults saved to: tolerance_basic_demo_results.RData\n")
}

# ============================================================================
# EDUCATIONAL NOTES AND TIPS
# ============================================================================

cat("\n" + "=" * 80 + "\n")
cat("EDUCATIONAL NOTES: RSIENA TOLERANCE MODELING\n")
cat("=" * 80 + "\n")

cat("\n1. KEY CONCEPTS:\n")
cat("   • Selection: Who becomes friends with whom based on tolerance\n")
cat("   • Influence: How friends affect each other's tolerance over time\n")
cat("   • Co-evolution: Networks and behaviors change simultaneously\n")

cat("\n2. CRITICAL PARAMETERS:\n")
cat("   • avAlt (average alter): Measures peer influence strength\n")
cat("   • simX (similarity): Measures selection based on similarity\n")
cat("   • Positive values indicate attraction, negative indicate repulsion\n")

cat("\n3. COMMON PITFALLS:\n")
cat("   • Insufficient iterations leading to non-convergence\n")
cat("   • Ignoring goodness-of-fit testing\n")
cat("   • Over-interpreting non-significant effects\n")
cat("   • Confusing selection and influence mechanisms\n")

cat("\n4. BEST PRACTICES:\n")
cat("   • Always check convergence (t-ratios < 0.25)\n")
cat("   • Test multiple model specifications\n")
cat("   • Validate results with goodness-of-fit tests\n")
cat("   • Consider substantive interpretation over statistical significance\n")

cat("\n5. EXTENSIONS:\n")
cat("   • Multi-group models for different populations\n")
cat("   • Time-varying covariates for intervention effects\n")
cat("   • Custom effects for complex theoretical mechanisms\n")

cat("\nFor advanced demonstrations, see:\n")
cat("   • intervention_simulation_demo.R\n")
cat("   • custom_effects_demo.R\n")

cat("\n" + "=" * 80 + "\n")