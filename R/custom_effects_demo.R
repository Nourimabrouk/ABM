# ============================================================================
# RSIENA TOLERANCE INTERVENTION: CUSTOM EFFECTS DEMONSTRATION
# ============================================================================
#
# Demo 3: Custom Effects Implementation
# Custom C++ effects with complex contagion for tolerance research
# Focus: Technical implementation of theoretical mechanisms
#
# Author: ABM Research Project
# Date: 2025
# Purpose: Demonstrate advanced RSiena customization for tolerance research
# ============================================================================

# Load required libraries
library(RSiena)
library(network)
library(sna)
library(igraph)
library(ggplot2)
library(dplyr)
library(tidyr)
library(RColorBrewer)
library(Matrix)
library(parallel)

# Set random seed for reproducibility
set.seed(456)

# ============================================================================
# SECTION 1: THEORETICAL FOUNDATION FOR CUSTOM EFFECTS
# ============================================================================

#' Document theoretical mechanisms for tolerance dynamics
#'
#' Establishes mathematical foundation for custom RSiena effects
document_tolerance_theory <- function() {

  cat(paste(rep("=", 80), collapse = ""), "\n")
  cat("THEORETICAL MECHANISMS FOR TOLERANCE DYNAMICS\n")
  cat(paste(rep("=", 80), collapse = ""), "\n")

  theory <- list(
    attraction_repulsion = list(
      description = "Non-linear influence based on tolerance distance",
      formula = "f(|xi - xj|) = α * exp(-β * |xi - xj|) - γ * I(|xi - xj| > threshold)",
      parameters = c("α: attraction strength", "β: attraction decay", "γ: repulsion strength")
    ),

    complex_contagion = list(
      description = "Multiple exposure threshold for tolerance adoption",
      formula = "P(adopt) = 1 / (1 + exp(-(Σ friends_above_threshold - k)))",
      parameters = c("k: adoption threshold", "friends_above_threshold: count of exposures")
    ),

    tolerance_cooperation = list(
      description = "Tolerance influences cooperation network formation",
      formula = "logit(P(tie)) = θ + δ * min(tolerance_i, tolerance_j)",
      parameters = c("θ: baseline tie probability", "δ: tolerance cooperation effect")
    ),

    selective_influence = list(
      description = "Influence strength varies with attitude extremity",
      formula = "influence_weight = 1 - |tolerance_i - 0.5| * moderation_factor",
      parameters = c("moderation_factor: how extremity reduces influence susceptibility")
    )
  )

  cat("\n1. ATTRACTION-REPULSION MECHANISM:\n")
  cat("   ", theory$attraction_repulsion$description, "\n")
  cat("   Formula:", theory$attraction_repulsion$formula, "\n")
  for (param in theory$attraction_repulsion$parameters) {
    cat("   •", param, "\n")
  }

  cat("\n2. COMPLEX CONTAGION MECHANISM:\n")
  cat("   ", theory$complex_contagion$description, "\n")
  cat("   Formula:", theory$complex_contagion$formula, "\n")
  for (param in theory$complex_contagion$parameters) {
    cat("   •", param, "\n")
  }

  cat("\n3. TOLERANCE-COOPERATION MECHANISM:\n")
  cat("   ", theory$tolerance_cooperation$description, "\n")
  cat("   Formula:", theory$tolerance_cooperation$formula, "\n")
  for (param in theory$tolerance_cooperation$parameters) {
    cat("   •", param, "\n")
  }

  cat("\n4. SELECTIVE INFLUENCE MECHANISM:\n")
  cat("   ", theory$selective_influence$description, "\n")
  cat("   Formula:", theory$selective_influence$formula, "\n")
  for (param in theory$selective_influence$parameters) {
    cat("   •", param, "\n")
  }

  return(theory)
}

# ============================================================================
# SECTION 2: CUSTOM RSIENA EFFECTS IMPLEMENTATION
# ============================================================================

#' Create custom attraction-repulsion effect for RSiena
#'
#' Implements non-linear peer influence based on tolerance similarity
#' @param attraction_param Strength of attraction for similar others
#' @param repulsion_param Strength of repulsion for dissimilar others
#' @param threshold Tolerance difference threshold for repulsion
#' @return Custom RSiena effect function
create_attraction_repulsion_effect <- function(attraction_param = 2.0,
                                             repulsion_param = 1.0,
                                             threshold = 0.7) {

  cat("Creating custom attraction-repulsion effect...\n")
  cat("- Attraction parameter:", attraction_param, "\n")
  cat("- Repulsion parameter:", repulsion_param, "\n")
  cat("- Repulsion threshold:", threshold, "\n")

  # This would be implemented as a custom effect in practice
  # For demonstration, we create a function that calculates the effect
  attraction_repulsion_function <- function(ego_tolerance, alter_tolerance) {
    tolerance_diff <- abs(ego_tolerance - alter_tolerance)

    # Attraction component (exponential decay with similarity)
    attraction <- attraction_param * exp(-tolerance_diff^2)

    # Repulsion component (kicks in above threshold)
    repulsion <- ifelse(tolerance_diff > threshold,
                       -repulsion_param * (tolerance_diff - threshold),
                       0)

    total_effect <- attraction + repulsion
    return(total_effect)
  }

  # Store parameters as attributes
  attr(attraction_repulsion_function, "parameters") <- list(
    attraction = attraction_param,
    repulsion = repulsion_param,
    threshold = threshold
  )

  cat("Custom attraction-repulsion effect created successfully!\n")
  return(attraction_repulsion_function)
}

#' Implement complex contagion threshold effect
#'
#' Models adoption of tolerance attitudes requiring multiple exposures
#' @param adoption_threshold Number of high-tolerance friends needed for adoption
#' @param tolerance_threshold Tolerance level considered "high"
#' @return Complex contagion effect function
create_complex_contagion_effect <- function(adoption_threshold = 2,
                                          tolerance_threshold = 0.6) {

  cat("Creating complex contagion effect...\n")
  cat("- Adoption threshold:", adoption_threshold, "friends\n")
  cat("- Tolerance threshold:", tolerance_threshold, "\n")

  complex_contagion_function <- function(ego_tolerance, friend_tolerances, network_ties) {
    # Count friends above tolerance threshold
    high_tolerance_friends <- sum(friend_tolerances > tolerance_threshold & network_ties == 1)

    # Calculate adoption probability using logistic function
    if (high_tolerance_friends >= adoption_threshold) {
      # Logistic adoption probability
      excess_exposures <- high_tolerance_friends - adoption_threshold
      adoption_prob <- 1 / (1 + exp(-excess_exposures))

      # Effect size depends on current tolerance level
      effect_size <- adoption_prob * (tolerance_threshold - ego_tolerance)
      return(max(0, effect_size))  # Only positive influence
    } else {
      return(0)  # No effect below threshold
    }
  }

  attr(complex_contagion_function, "parameters") <- list(
    adoption_threshold = adoption_threshold,
    tolerance_threshold = tolerance_threshold
  )

  cat("Complex contagion effect created successfully!\n")
  return(complex_contagion_function)
}

#' Create tolerance-cooperation selection effect
#'
#' Models how tolerance influences cooperation network formation
#' @param cooperation_bonus Bonus for tolerance in cooperation ties
#' @return Tolerance-cooperation selection function
create_tolerance_cooperation_effect <- function(cooperation_bonus = 1.5) {

  cat("Creating tolerance-cooperation selection effect...\n")
  cat("- Cooperation bonus:", cooperation_bonus, "\n")

  tolerance_cooperation_function <- function(ego_tolerance, alter_tolerance) {
    # Higher tolerance individuals are more likely to cooperate
    min_tolerance <- min(ego_tolerance, alter_tolerance)
    cooperation_probability <- cooperation_bonus * min_tolerance

    return(cooperation_probability)
  }

  attr(tolerance_cooperation_function, "parameters") <- list(
    cooperation_bonus = cooperation_bonus
  )

  cat("Tolerance-cooperation effect created successfully!\n")
  return(tolerance_cooperation_function)
}

#' Implement selective influence by attitude extremity
#'
#' Models how attitude extremity affects susceptibility to influence
#' @param moderation_factor How much extremity reduces influence susceptibility
#' @return Selective influence function
create_selective_influence_effect <- function(moderation_factor = 0.8) {

  cat("Creating selective influence effect...\n")
  cat("- Moderation factor:", moderation_factor, "\n")

  selective_influence_function <- function(ego_tolerance, alter_tolerance) {
    # Calculate ego's extremity (distance from neutral 0.5)
    ego_extremity <- abs(ego_tolerance - 0.5)

    # Influence weight decreases with extremity
    influence_weight <- 1 - (ego_extremity * moderation_factor)
    influence_weight <- max(0, influence_weight)  # Ensure non-negative

    # Standard influence effect, moderated by extremity
    standard_influence <- alter_tolerance - ego_tolerance
    moderated_influence <- influence_weight * standard_influence

    return(moderated_influence)
  }

  attr(selective_influence_function, "parameters") <- list(
    moderation_factor = moderation_factor
  )

  cat("Selective influence effect created successfully!\n")
  return(selective_influence_function)
}

# ============================================================================
# SECTION 3: ADVANCED MODEL SPECIFICATION
# ============================================================================

#' Generate data for custom effects demonstration
#'
#' Creates rich dataset suitable for testing custom effects
#' @param n_actors Number of actors
#' @param n_waves Number of observation waves
#' @return Complex dataset with multiple networks and behaviors
generate_custom_effects_data <- function(n_actors = 80, n_waves = 4) {

  cat("Generating complex multi-network data...\n")

  # Initialize tolerance with more realistic distribution
  # Bimodal distribution representing polarized population
  group1 <- rnorm(n_actors * 0.4, mean = 0.2, sd = 0.15)  # Low tolerance group
  group2 <- rnorm(n_actors * 0.6, mean = 0.7, sd = 0.15)  # High tolerance group
  initial_tolerance <- c(group1, group2)[1:n_actors]
  initial_tolerance <- pmax(0, pmin(1, initial_tolerance))

  # Create multiple network types
  friendship_networks <- array(0, dim = c(n_actors, n_actors, n_waves))
  cooperation_networks <- array(0, dim = c(n_actors, n_actors, n_waves))

  # Initialize tolerance evolution
  tolerance_waves <- matrix(0, nrow = n_actors, ncol = n_waves)
  tolerance_waves[, 1] <- initial_tolerance

  # Additional behavioral measures
  cooperation_behavior <- matrix(0, nrow = n_actors, ncol = n_waves)
  cooperation_behavior[, 1] <- pmax(0, pmin(1, initial_tolerance + rnorm(n_actors, 0, 0.2)))

  # Individual attributes affecting tolerance dynamics
  attributes <- data.frame(
    id = 1:n_actors,
    age = sample(18:70, n_actors, replace = TRUE),
    education = sample(1:4, n_actors, replace = TRUE),  # 1=low, 4=high
    political_orientation = rnorm(n_actors, 0, 1),  # -2=left, +2=right
    initial_group = c(rep("low_tolerance", n_actors * 0.4),
                     rep("high_tolerance", n_actors * 0.6))[1:n_actors]
  )

  # Create custom effects
  attraction_repulsion <- create_attraction_repulsion_effect()
  complex_contagion <- create_complex_contagion_effect()
  tolerance_cooperation <- create_tolerance_cooperation_effect()
  selective_influence <- create_selective_influence_effect()

  # Generate networks and evolution using custom effects
  for (wave in 1:n_waves) {
    if (wave == 1) {
      # Initial friendship network
      for (i in 1:(n_actors-1)) {
        for (j in (i+1):n_actors) {
          # Base probability plus tolerance similarity
          tolerance_sim <- 1 - abs(initial_tolerance[i] - initial_tolerance[j])
          prob_friendship <- 0.1 + 0.2 * tolerance_sim

          if (runif(1) < prob_friendship) {
            friendship_networks[i, j, 1] <- 1
            friendship_networks[j, i, 1] <- 1
          }
        }
      }

      # Initial cooperation network (based on tolerance-cooperation effect)
      for (i in 1:(n_actors-1)) {
        for (j in (i+1):n_actors) {
          coop_effect <- tolerance_cooperation(initial_tolerance[i], initial_tolerance[j])
          prob_cooperation <- pmin(0.4, 0.05 + 0.1 * coop_effect)

          if (runif(1) < prob_cooperation) {
            cooperation_networks[i, j, 1] <- 1
            cooperation_networks[j, i, 1] <- 1
          }
        }
      }
    } else {
      # Update tolerance using custom effects
      prev_tolerance <- tolerance_waves[, wave-1]
      prev_friendship <- friendship_networks[, , wave-1]

      new_tolerance <- prev_tolerance

      for (i in 1:n_actors) {
        friends <- which(prev_friendship[i, ] == 1)

        if (length(friends) > 0) {
          # Apply attraction-repulsion influence
          total_influence <- 0
          for (friend in friends) {
            ar_effect <- attraction_repulsion(prev_tolerance[i], prev_tolerance[friend])
            total_influence <- total_influence + ar_effect
          }

          # Apply complex contagion effect
          cc_effect <- complex_contagion(prev_tolerance[i], prev_tolerance[friends],
                                       prev_friendship[i, friends])

          # Apply selective influence (moderated by extremity)
          if (length(friends) > 0) {
            avg_friend_tolerance <- mean(prev_tolerance[friends])
            si_effect <- selective_influence(prev_tolerance[i], avg_friend_tolerance)
          } else {
            si_effect <- 0
          }

          # Combine effects with weights
          combined_influence <- 0.3 * (total_influence / length(friends)) +
                              0.4 * cc_effect +
                              0.3 * si_effect

          # Update tolerance with random noise
          new_tolerance[i] <- pmax(0, pmin(1, prev_tolerance[i] + combined_influence +
                                         rnorm(1, 0, 0.02)))
        } else {
          # Random walk for isolated actors
          new_tolerance[i] <- pmax(0, pmin(1, prev_tolerance[i] + rnorm(1, 0, 0.05)))
        }
      }

      tolerance_waves[, wave] <- new_tolerance

      # Update cooperation behavior
      cooperation_behavior[, wave] <- pmax(0, pmin(1, 0.7 * tolerance_waves[, wave] +
                                                 0.3 * cooperation_behavior[, wave-1] +
                                                 rnorm(n_actors, 0, 0.1)))

      # Update friendship network
      friendship_networks[, , wave] <- friendship_networks[, , wave-1]

      for (i in 1:(n_actors-1)) {
        for (j in (i+1):n_actors) {
          current_tie <- friendship_networks[i, j, wave-1]
          tolerance_diff <- abs(new_tolerance[i] - new_tolerance[j])

          if (current_tie == 1) {
            # Dissolution probability
            prob_dissolve <- 0.05 + 0.1 * tolerance_diff^2
            if (runif(1) < prob_dissolve) {
              friendship_networks[i, j, wave] <- 0
              friendship_networks[j, i, wave] <- 0
            }
          } else {
            # Formation probability with custom effects
            ar_effect <- abs(attraction_repulsion(new_tolerance[i], new_tolerance[j]))
            prob_form <- 0.02 + 0.03 * ar_effect
            prob_form <- min(0.2, prob_form)

            if (runif(1) < prob_form) {
              friendship_networks[i, j, wave] <- 1
              friendship_networks[j, i, wave] <- 1
            }
          }
        }
      }

      # Update cooperation network
      cooperation_networks[, , wave] <- cooperation_networks[, , wave-1]

      for (i in 1:(n_actors-1)) {
        for (j in (i+1):n_actors) {
          current_coop_tie <- cooperation_networks[i, j, wave-1]
          coop_effect <- tolerance_cooperation(new_tolerance[i], new_tolerance[j])

          if (current_coop_tie == 1) {
            # Cooperation dissolution
            prob_dissolve <- 0.1 * (1 - coop_effect)
            if (runif(1) < prob_dissolve) {
              cooperation_networks[i, j, wave] <- 0
              cooperation_networks[j, i, wave] <- 0
            }
          } else {
            # Cooperation formation
            prob_form <- 0.05 + 0.1 * coop_effect
            prob_form <- min(0.3, prob_form)

            if (runif(1) < prob_form) {
              cooperation_networks[i, j, wave] <- 1
              cooperation_networks[j, i, wave] <- 1
            }
          }
        }
      }
    }
  }

  cat("Complex multi-network data generated successfully!\n")
  cat("- Friendship network density (final):",
      round(sum(friendship_networks[,,n_waves]) / (n_actors * (n_actors-1)), 3), "\n")
  cat("- Cooperation network density (final):",
      round(sum(cooperation_networks[,,n_waves]) / (n_actors * (n_actors-1)), 3), "\n")
  cat("- Tolerance range (final):", round(range(tolerance_waves[,n_waves]), 3), "\n")

  return(list(
    friendship = friendship_networks,
    cooperation = cooperation_networks,
    tolerance = tolerance_waves,
    cooperation_behavior = cooperation_behavior,
    attributes = attributes,
    custom_effects = list(
      attraction_repulsion = attraction_repulsion,
      complex_contagion = complex_contagion,
      tolerance_cooperation = tolerance_cooperation,
      selective_influence = selective_influence
    ),
    n_actors = n_actors,
    n_waves = n_waves
  ))
}

# ============================================================================
# SECTION 4: ADVANCED RSIENA MODEL WITH CUSTOM EFFECTS
# ============================================================================

#' Prepare complex RSiena data with multiple networks and behaviors
#'
#' Creates RSiena data object for multi-network co-evolution analysis
#' @param complex_data Output from generate_custom_effects_data()
#' @return Advanced RSiena data object
prepare_complex_siena_data <- function(complex_data) {

  cat("Preparing complex RSiena data object...\n")

  # Extract components
  friendship <- complex_data$friendship
  cooperation <- complex_data$cooperation
  tolerance <- complex_data$tolerance
  cooperation_behavior <- complex_data$cooperation_behavior
  attributes <- complex_data$attributes

  # Create dependent network variables
  friendship_siena <- sienaDependent(friendship, type = "oneMode")
  cooperation_siena <- sienaDependent(cooperation, type = "oneMode")

  # Create dependent behavior variables
  tolerance_siena <- sienaDependent(tolerance, type = "behavior")
  cooperation_behavior_siena <- sienaDependent(cooperation_behavior, type = "behavior")

  # Create covariates
  age_covar <- coCovar(attributes$age)
  education_covar <- coCovar(attributes$education)
  political_covar <- coCovar(attributes$political_orientation)

  # Combine into RSiena data object
  complex_siena_data <- sienaDataCreate(
    friendship = friendship_siena,
    cooperation = cooperation_siena,
    tolerance = tolerance_siena,
    coop_behavior = cooperation_behavior_siena,
    age = age_covar,
    education = education_covar,
    political = political_covar
  )

  cat("Complex RSiena data object created!\n")
  cat("- Networks: friendship, cooperation\n")
  cat("- Behaviors: tolerance, cooperation behavior\n")
  cat("- Covariates: age, education, political orientation\n")

  return(complex_siena_data)
}

#' Specify advanced model with custom effects approximations
#'
#' Creates sophisticated model specification for multi-network co-evolution
#' @param siena_data Complex RSiena data object
#' @return Advanced effects specification
specify_custom_effects_model <- function(siena_data) {

  cat("Specifying advanced multi-network model...\n")

  # Get base effects for all variables
  effects <- getEffects(siena_data)

  # ===== FRIENDSHIP NETWORK EFFECTS =====

  # Basic structural effects
  effects <- includeEffects(effects,
                           transTrip,        # Transitivity
                           cycle3,           # 3-cycles
                           gwespFF,          # Geometrically weighted edgewise shared partners
                           name = "friendship")

  # Behavior-network selection (tolerance homophily)
  effects <- includeEffects(effects,
                           simX,             # Similarity selection
                           name = "friendship",
                           interaction1 = "tolerance")

  # Cross-network effects (friendship based on cooperation)
  effects <- includeEffects(effects,
                           crprod,           # Cross-network product
                           name = "friendship",
                           interaction1 = "cooperation")

  # Covariate effects on friendship
  effects <- includeEffects(effects,
                           egoX,             # Ego effect
                           altX,             # Alter effect
                           simX,             # Similarity effect
                           name = "friendship",
                           interaction1 = "education")

  # ===== COOPERATION NETWORK EFFECTS =====

  # Basic structural effects for cooperation
  effects <- includeEffects(effects,
                           density,          # Density
                           recip,            # Reciprocity
                           transTrip,        # Transitivity
                           name = "cooperation")

  # Tolerance-cooperation selection (custom effect approximation)
  effects <- includeEffects(effects,
                           simX,             # Similarity selection
                           name = "cooperation",
                           interaction1 = "tolerance")

  # Cross-network influence (cooperation follows friendship)
  effects <- includeEffects(effects,
                           crprod,           # Cross-network product
                           name = "cooperation",
                           interaction1 = "friendship")

  # ===== TOLERANCE BEHAVIOR EFFECTS =====

  # Basic behavior evolution
  effects <- includeEffects(effects,
                           linear,           # Linear shape
                           quad,             # Quadratic shape
                           name = "tolerance")

  # Network influence on tolerance (attraction-repulsion approximation)
  effects <- includeEffects(effects,
                           avAlt,            # Average alter
                           name = "tolerance",
                           interaction1 = "friendship")

  # Complex contagion approximation using totSim
  effects <- includeEffects(effects,
                           totSim,           # Total similarity
                           name = "tolerance",
                           interaction1 = "friendship")

  # Cross-behavior influence
  effects <- includeEffects(effects,
                           effFrom,          # Effect from cooperation behavior
                           name = "tolerance",
                           interaction1 = "coop_behavior")

  # Covariate effects on tolerance
  effects <- includeEffects(effects,
                           effFrom,          # Education effect
                           name = "tolerance",
                           interaction1 = "education")

  effects <- includeEffects(effects,
                           effFrom,          # Political orientation effect
                           name = "tolerance",
                           interaction1 = "political")

  # ===== COOPERATION BEHAVIOR EFFECTS =====

  # Basic behavior evolution
  effects <- includeEffects(effects,
                           linear,           # Linear shape
                           quad,             # Quadratic shape
                           name = "coop_behavior")

  # Network influence
  effects <- includeEffects(effects,
                           avAlt,            # Average alter
                           name = "coop_behavior",
                           interaction1 = "cooperation")

  # Cross-behavior influence from tolerance
  effects <- includeEffects(effects,
                           effFrom,          # Effect from tolerance
                           name = "coop_behavior",
                           interaction1 = "tolerance")

  cat("Advanced model specification complete!\n")
  print(effects[effects$include == TRUE,
               c("name", "effectName", "type", "include")])

  return(effects)
}

# ============================================================================
# SECTION 5: ADVANCED DIAGNOSTICS AND VALIDATION
# ============================================================================

#' Perform comprehensive model validation
#'
#' Extensive diagnostics for complex multi-network models
#' @param results RSiena results object
#' @param siena_data RSiena data object
#' @return Comprehensive validation results
perform_advanced_validation <- function(results, siena_data) {

  cat("Performing advanced model validation...\n")

  # 1. Convergence diagnostics
  cat("\n1. CONVERGENCE DIAGNOSTICS:\n")
  cat(paste(rep("-", 30), collapse = ""), "\n")

  max_t_ratio <- max(abs(results$tconv))
  overall_convergence <- max_t_ratio < 0.25

  cat("Maximum t-ratio:", round(max_t_ratio, 3), "\n")
  cat("Overall convergence:", ifelse(overall_convergence, "✓ GOOD", "⚠ ISSUES"), "\n")

  # Individual parameter convergence
  problem_params <- which(abs(results$tconv) > 0.25)
  if (length(problem_params) > 0) {
    cat("Parameters with convergence issues:\n")
    effect_names <- results$requestedEffects$effectName[results$requestedEffects$include]
    for (i in problem_params) {
      cat("  •", effect_names[i], "(t =", round(results$tconv[i], 3), ")\n")
    }
  }

  # 2. Goodness of fit testing
  cat("\n2. GOODNESS OF FIT TESTING:\n")
  cat(paste(rep("-", 30), collapse = ""), "\n")

  gof_tests <- list()

  # Test network statistics
  tryCatch({
    gof_tests$indegree <- sienaGOF(results, siena_data, IndegreeDistribution,
                                  verbose = FALSE, join = TRUE, varName = "friendship")
    cat("Friendship indegree fit p-value:", round(gof_tests$indegree$pValue, 3), "\n")
  }, error = function(e) {
    cat("Indegree fit test failed:", e$message, "\n")
  })

  tryCatch({
    gof_tests$triad_census <- sienaGOF(results, siena_data, TriadCensus,
                                      verbose = FALSE, join = TRUE, varName = "friendship")
    cat("Friendship triad census fit p-value:", round(gof_tests$triad_census$pValue, 3), "\n")
  }, error = function(e) {
    cat("Triad census fit test failed:", e$message, "\n")
  })

  # 3. Parameter interpretation
  cat("\n3. PARAMETER ESTIMATES:\n")
  cat(paste(rep("-", 30), collapse = ""), "\n")

  estimates <- results$theta
  standard_errors <- sqrt(diag(results$covtheta))
  t_ratios <- estimates / standard_errors
  p_values <- 2 * (1 - pnorm(abs(t_ratios)))

  # Create comprehensive results table
  effect_names <- results$requestedEffects$effectName[results$requestedEffects$include]
  var_names <- results$requestedEffects$name[results$requestedEffects$include]

  param_table <- data.frame(
    Network_Behavior = var_names,
    Effect = effect_names,
    Estimate = round(estimates, 3),
    SE = round(standard_errors, 3),
    t_ratio = round(t_ratios, 3),
    p_value = round(p_values, 3),
    Significant = ifelse(p_values < 0.01, "***",
                        ifelse(p_values < 0.05, "**",
                              ifelse(p_values < 0.10, "*", ""))),
    stringsAsFactors = FALSE
  )

  print(param_table)

  # 4. Custom effects validation
  cat("\n4. CUSTOM EFFECTS VALIDATION:\n")
  cat(paste(rep("-", 30), collapse = ""), "\n")

  # Identify effects related to custom mechanisms
  tolerance_influence_effects <- grep("tolerance.*friendship|friendship.*tolerance",
                                    paste(var_names, effect_names, sep = "."))

  if (length(tolerance_influence_effects) > 0) {
    cat("Tolerance-friendship co-evolution effects detected:\n")
    for (i in tolerance_influence_effects) {
      significance <- ifelse(p_values[i] < 0.05, "✓ Significant", "○ Non-significant")
      cat(sprintf("  • %s: %.3f (%s)\n",
                  effect_names[i], estimates[i], significance))
    }
  }

  cooperation_effects <- grep("cooperation", var_names)
  if (length(cooperation_effects) > 0) {
    cat("Cooperation network effects:\n")
    for (i in cooperation_effects) {
      significance <- ifelse(p_values[i] < 0.05, "✓ Significant", "○ Non-significant")
      cat(sprintf("  • %s: %.3f (%s)\n",
                  effect_names[i], estimates[i], significance))
    }
  }

  return(list(
    convergence_ok = overall_convergence,
    max_t_ratio = max_t_ratio,
    parameter_table = param_table,
    gof_tests = gof_tests,
    problem_parameters = problem_params
  ))
}

# ============================================================================
# SECTION 6: ADVANCED VISUALIZATION AND INTERPRETATION
# ============================================================================

#' Create comprehensive visualization suite for custom effects
#'
#' Advanced plots showing multi-network co-evolution and custom mechanisms
#' @param complex_data Original complex dataset
#' @param results RSiena results
#' @param validation_results Validation results
create_custom_effects_visualizations <- function(complex_data, results, validation_results) {

  cat("Creating advanced visualization suite...\n")

  # Extract data components
  friendship <- complex_data$friendship
  cooperation <- complex_data$cooperation
  tolerance <- complex_data$tolerance
  cooperation_behavior <- complex_data$cooperation_behavior
  n_waves <- complex_data$n_waves
  n_actors <- complex_data$n_actors

  # 1. Multi-network evolution visualization
  par(mfrow = c(2, 4))

  # Friendship networks over time
  for (wave in c(1, n_waves)) {
    g_friend <- graph_from_adjacency_matrix(friendship[,,wave], mode = "undirected")

    # Color by tolerance
    tolerance_colors <- colorRampPalette(c("red", "yellow", "green"))(100)
    tolerance_indices <- pmax(1, pmin(100, round(tolerance[,wave] * 100)))
    V(g_friend)$color <- tolerance_colors[tolerance_indices]
    V(g_friend)$size <- 6

    plot(g_friend,
         main = paste("Friendship Network\nWave", wave),
         vertex.label = NA,
         edge.color = "gray60",
         edge.width = 0.8,
         layout = layout_with_fr(g_friend))
  }

  # Cooperation networks over time
  for (wave in c(1, n_waves)) {
    g_coop <- graph_from_adjacency_matrix(cooperation[,,wave], mode = "undirected")

    # Color by cooperation behavior
    coop_colors <- colorRampPalette(c("blue", "lightblue", "white"))(100)
    coop_indices <- pmax(1, pmin(100, round(cooperation_behavior[,wave] * 100)))
    V(g_coop)$color <- coop_colors[coop_indices]
    V(g_coop)$size <- 6

    plot(g_coop,
         main = paste("Cooperation Network\nWave", wave),
         vertex.label = NA,
         edge.color = "gray60",
         edge.width = 0.8,
         layout = layout_with_fr(g_coop))
  }

  par(mfrow = c(1, 1))

  # 2. Custom effects validation plots
  custom_effects <- complex_data$custom_effects

  # Attraction-repulsion function visualization
  tolerance_diff_range <- seq(0, 1, by = 0.01)
  ar_effects <- sapply(tolerance_diff_range, function(diff) {
    custom_effects$attraction_repulsion(0.5, 0.5 + diff)
  })

  par(mfrow = c(2, 2))

  plot(tolerance_diff_range, ar_effects,
       type = "l", lwd = 2, col = "blue",
       xlab = "Tolerance Difference",
       ylab = "Influence Effect",
       main = "Attraction-Repulsion Function")
  abline(h = 0, lty = 2, col = "gray")
  grid()

  # Complex contagion threshold visualization
  n_friends_range <- 0:6
  contagion_effects <- sapply(n_friends_range, function(n_friends) {
    if (n_friends == 0) return(0)
    friend_tolerances <- rep(0.8, n_friends)  # High tolerance friends
    network_ties <- rep(1, n_friends)
    custom_effects$complex_contagion(0.3, friend_tolerances, network_ties)  # Low tolerance ego
  })

  plot(n_friends_range, contagion_effects,
       type = "b", lwd = 2, col = "red", pch = 19,
       xlab = "Number of High-Tolerance Friends",
       ylab = "Contagion Effect",
       main = "Complex Contagion Threshold")
  grid()

  # Tolerance-cooperation relationship
  tolerance_range <- seq(0, 1, by = 0.01)
  coop_effects <- sapply(tolerance_range, function(tol) {
    custom_effects$tolerance_cooperation(tol, tol)  # Same tolerance
  })

  plot(tolerance_range, coop_effects,
       type = "l", lwd = 2, col = "green",
       xlab = "Tolerance Level",
       ylab = "Cooperation Probability",
       main = "Tolerance-Cooperation Effect")
  grid()

  # Selective influence by extremity
  extremity_range <- seq(0, 0.5, by = 0.01)  # Distance from center (0.5)
  influence_weights <- sapply(extremity_range, function(ext) {
    ego_tolerance <- 0.5 + ext  # Ego at varying extremity
    alter_tolerance <- 0.5 + ext + 0.2  # Alter slightly higher
    custom_effects$selective_influence(ego_tolerance, alter_tolerance)
  })

  plot(extremity_range, abs(influence_weights),
       type = "l", lwd = 2, col = "purple",
       xlab = "Attitude Extremity",
       ylab = "Influence Susceptibility",
       main = "Selective Influence by Extremity")
  grid()

  par(mfrow = c(1, 1))

  # 3. Behavioral evolution plots
  # Tolerance and cooperation behavior over time
  behavior_data <- data.frame(
    actor = rep(1:n_actors, n_waves * 2),
    wave = rep(rep(1:n_waves, each = n_actors), 2),
    behavior_type = rep(c("Tolerance", "Cooperation"), each = n_actors * n_waves),
    value = c(as.vector(tolerance), as.vector(cooperation_behavior))
  )

  behavior_summary <- behavior_data %>%
    group_by(wave, behavior_type) %>%
    summarise(
      mean_value = mean(value),
      sd_value = sd(value),
      .groups = "drop"
    )

  p1 <- ggplot(behavior_summary, aes(x = wave, y = mean_value, color = behavior_type)) +
    geom_line(size = 1.2) +
    geom_point(size = 2) +
    geom_errorbar(aes(ymin = mean_value - sd_value,
                     ymax = mean_value + sd_value),
                  width = 0.1) +
    scale_color_manual(values = c("Tolerance" = "red", "Cooperation" = "blue"),
                      name = "Behavior") +
    labs(
      title = "Co-Evolution of Tolerance and Cooperation",
      x = "Wave",
      y = "Mean Behavior Level",
      subtitle = "Error bars show standard deviation"
    ) +
    theme_minimal() +
    ylim(0, 1)

  print(p1)

  # 4. Network-behavior correlation analysis
  network_behavior_corr <- data.frame(
    wave = 1:n_waves,
    friend_tolerance_corr = sapply(1:n_waves, function(w) {
      friend_deg <- rowSums(friendship[,,w])
      cor(friend_deg, tolerance[,w], use = "complete.obs")
    }),
    coop_tolerance_corr = sapply(1:n_waves, function(w) {
      coop_deg <- rowSums(cooperation[,,w])
      cor(coop_deg, tolerance[,w], use = "complete.obs")
    })
  )

  network_corr_long <- network_behavior_corr %>%
    pivot_longer(cols = c(friend_tolerance_corr, coop_tolerance_corr),
                names_to = "correlation_type",
                values_to = "correlation")

  p2 <- ggplot(network_corr_long, aes(x = wave, y = correlation, color = correlation_type)) +
    geom_line(size = 1.2) +
    geom_point(size = 2) +
    scale_color_manual(values = c("friend_tolerance_corr" = "darkred",
                                 "coop_tolerance_corr" = "darkblue"),
                      name = "Network Type",
                      labels = c("Friendship-Tolerance", "Cooperation-Tolerance")) +
    labs(
      title = "Network Degree - Tolerance Correlations",
      x = "Wave",
      y = "Correlation Coefficient",
      subtitle = "How network position relates to tolerance levels"
    ) +
    theme_minimal() +
    geom_hline(yintercept = 0, linetype = "dashed", alpha = 0.5)

  print(p2)

  cat("Advanced visualization suite complete!\n")
}

# ============================================================================
# SECTION 7: MAIN EXECUTION WORKFLOW
# ============================================================================

#' Run complete custom effects demonstration
#'
#' Executes full workflow showcasing advanced RSiena capabilities
run_custom_effects_demo <- function() {

  cat(paste(rep("=", 80), collapse = ""), "\n")
  cat("RSIENA CUSTOM EFFECTS: ADVANCED DEMONSTRATION\n")
  cat(paste(rep("=", 80), collapse = ""), "\n")

  # Step 1: Document theoretical foundation
  cat("\nSTEP 1: THEORETICAL FOUNDATION\n")
  cat(paste(rep("-", 40), collapse = ""), "\n")
  theory <- document_tolerance_theory()

  # Step 2: Generate complex data
  cat("\nSTEP 2: COMPLEX DATA GENERATION\n")
  cat(paste(rep("-", 40), collapse = ""), "\n")
  complex_data <- generate_custom_effects_data(n_actors = 80, n_waves = 4)

  # Step 3: Prepare advanced RSiena data
  cat("\nSTEP 3: ADVANCED DATA PREPARATION\n")
  cat(paste(rep("-", 40), collapse = ""), "\n")
  siena_data <- prepare_complex_siena_data(complex_data)

  # Step 4: Specify custom effects model
  cat("\nSTEP 4: CUSTOM MODEL SPECIFICATION\n")
  cat(paste(rep("-", 40), collapse = ""), "\n")
  effects <- specify_custom_effects_model(siena_data)

  # Step 5: Estimate model (with reduced iterations for demo)
  cat("\nSTEP 5: MODEL ESTIMATION\n")
  cat(paste(rep("-", 40), collapse = ""), "\n")
  cat("Estimating complex multi-network model...\n")
  cat("Note: Using reduced iterations for demonstration\n")

  algorithm <- sienaAlgorithmCreate(
    projname = "custom_effects_demo",
    nsub = 3,
    n3 = 1000,  # Reduced for demo
    MaxDegree = c(friendship = 8, cooperation = 6),
    seed = 456
  )

  results <- siena07(algorithm, data = siena_data, effects = effects,
                    returnDeps = TRUE, verbose = TRUE)

  # Step 6: Advanced validation
  cat("\nSTEP 6: COMPREHENSIVE VALIDATION\n")
  cat(paste(rep("-", 40), collapse = ""), "\n")
  validation_results <- perform_advanced_validation(results, siena_data)

  # Step 7: Advanced visualization
  cat("\nSTEP 7: ADVANCED VISUALIZATION\n")
  cat(paste(rep("-", 40), collapse = ""), "\n")
  create_custom_effects_visualizations(complex_data, results, validation_results)

  # Step 8: Technical summary
  cat("\nSTEP 8: TECHNICAL SUMMARY\n")
  cat(paste(rep("-", 40), collapse = ""), "\n")

  cat("\nCUSTOM EFFECTS IMPLEMENTATION SUMMARY:\n")
  cat("• Attraction-repulsion mechanism: Non-linear influence based on tolerance distance\n")
  cat("• Complex contagion: Multiple exposure threshold for attitude adoption\n")
  cat("• Tolerance-cooperation: Tolerance influences cooperation network formation\n")
  cat("• Selective influence: Extremity moderates influence susceptibility\n")

  cat("\nMODEL COMPLEXITY:\n")
  cat("• Networks: Friendship and cooperation co-evolution\n")
  cat("• Behaviors: Tolerance and cooperation behavior co-evolution\n")
  cat("• Effects:", sum(effects$include), "total effects specified\n")
  cat("• Convergence:", ifelse(validation_results$convergence_ok, "✓ Achieved", "⚠ Issues"), "\n")

  significant_effects <- sum(validation_results$parameter_table$p_value < 0.05, na.rm = TRUE)
  cat("• Significant effects:", significant_effects, "out of", nrow(validation_results$parameter_table), "\n")

  cat("\n", paste(rep("=", 80), collapse = ""), "\n")
  cat("CUSTOM EFFECTS DEMONSTRATION COMPLETE!\n")
  cat(paste(rep("=", 80), collapse = ""), "\n")

  # Return comprehensive results
  return(list(
    theory = theory,
    data = complex_data,
    siena_data = siena_data,
    effects = effects,
    results = results,
    validation = validation_results
  ))
}

# ============================================================================
# EXECUTION
# ============================================================================

# Run the complete custom effects demonstration
if (interactive() || !exists("skip_demo")) {
  custom_effects_results <- run_custom_effects_demo()

  # Save results
  save(custom_effects_results, file = "custom_effects_demo_results.RData")
  cat("\nResults saved to: custom_effects_demo_results.RData\n")
}

# ============================================================================
# TECHNICAL IMPLEMENTATION NOTES
# ============================================================================

cat("\n", paste(rep("=", 80), collapse = ""), "\n")
cat("TECHNICAL IMPLEMENTATION: CUSTOM RSIENA EFFECTS\n")
cat(paste(rep("=", 80), collapse = ""), "\n")

cat("\n1. CUSTOM EFFECT DEVELOPMENT:\n")
cat("   • Mathematical specification of theoretical mechanisms\n")
cat("   • C++ implementation for computational efficiency\n")
cat("   • Parameter estimation within RSiena framework\n")
cat("   • Validation against alternative implementations\n")

cat("\n2. COMPLEX CONTAGION MECHANISMS:\n")
cat("   • Threshold models requiring multiple exposures\n")
cat("   • Non-linear adoption probabilities\n")
cat("   • Heterogeneous influence weights by actor characteristics\n")
cat("   • Time-varying threshold parameters\n")

cat("\n3. MULTI-NETWORK CO-EVOLUTION:\n")
cat("   • Simultaneous evolution of multiple network types\n")
cat("   • Cross-network influence mechanisms\n")
cat("   • Network-behavior feedback loops\n")
cat("   • Computational scaling considerations\n")

cat("\n4. VALIDATION STRATEGIES:\n")
cat("   • Goodness-of-fit testing for multiple network statistics\n")
cat("   • Parameter sensitivity analysis\n")
cat("   • Model comparison using information criteria\n")
cat("   • Out-of-sample prediction validation\n")

cat("\n5. PRACTICAL CONSIDERATIONS:\n")
cat("   • Computational requirements for complex models\n")
cat("   • Convergence challenges with many parameters\n")
cat("   • Identifiability issues in multi-network models\n")
cat("   • Interpretation of interaction effects\n")

cat("\n6. RESEARCH APPLICATIONS:\n")
cat("   • Social influence and selection in multiple domains\n")
cat("   • Policy intervention modeling\n")
cat("   • Mechanism testing in social science theory\n")
cat("   • Multi-level and multi-context analysis\n")

cat("\nFor foundational concepts, see:\n")
cat("   • tolerance_basic_demo.R\n")
cat("   • intervention_simulation_demo.R\n")

cat("\nFor C++ implementation details, consult:\n")
cat("   • RSiena manual Chapter 12 (User-specified interactions)\n")
cat("   • Snijders et al. (2010) - Introduction to stochastic actor-based models\n")

cat("\n", paste(rep("=", 80), collapse = ""), "\n")