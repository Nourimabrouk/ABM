# ============================================================================
# RSIENA TOLERANCE RESEARCH: EXAMPLE DATA GENERATOR
# ============================================================================
#
# Purpose: Generate example datasets for immediate testing and exploration
# Author: ABM Research Project
# Date: 2025
# Usage: Create realistic tolerance and network data for RSiena demonstrations
# ============================================================================

# Load required libraries
library(RSiena)
library(network)
library(igraph)
library(Matrix)

# Set random seed for reproducibility
set.seed(789)

# ============================================================================
# SECTION 1: BASIC EXAMPLE DATA
# ============================================================================

#' Generate small example dataset for testing
#'
#' Creates minimal but realistic tolerance network data
#' @param n_actors Number of actors (default: 30 for quick testing)
#' @param n_waves Number of observation waves (default: 3)
#' @return Basic tolerance dataset
generate_basic_example <- function(n_actors = 30, n_waves = 3) {

  cat("Generating basic example dataset...\n")
  cat("- Actors:", n_actors, "\n")
  cat("- Waves:", n_waves, "\n")

  # Initialize tolerance levels
  # Create two groups: lower tolerance (0.2-0.4) and higher tolerance (0.6-0.8)
  group_size <- n_actors %/% 2
  low_tolerance_group <- runif(group_size, 0.15, 0.45)
  high_tolerance_group <- runif(n_actors - group_size, 0.55, 0.85)
  initial_tolerance <- c(low_tolerance_group, high_tolerance_group)

  # Initialize friendship network
  friendship_networks <- array(0, dim = c(n_actors, n_actors, n_waves))
  tolerance_waves <- matrix(0, nrow = n_actors, ncol = n_waves)
  tolerance_waves[, 1] <- initial_tolerance

  # Create initial friendship network with homophily
  for (i in 1:(n_actors-1)) {
    for (j in (i+1):n_actors) {
      # Probability based on tolerance similarity and random baseline
      tolerance_similarity <- 1 - abs(initial_tolerance[i] - initial_tolerance[j])
      prob_friendship <- 0.1 + 0.3 * tolerance_similarity

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
        # Simple peer influence: move toward friends' average
        peer_average <- mean(prev_tolerance[friends])
        influence <- 0.2 * (peer_average - prev_tolerance[i])
        new_tolerance[i] <- pmax(0, pmin(1, prev_tolerance[i] + influence + rnorm(1, 0, 0.05)))
      } else {
        # Random walk for isolated actors
        new_tolerance[i] <- pmax(0, pmin(1, prev_tolerance[i] + rnorm(1, 0, 0.05)))
      }
    }

    tolerance_waves[, wave] <- new_tolerance

    # Update friendship network
    friendship_networks[, , wave] <- prev_network

    # Some friendship changes based on tolerance evolution
    for (i in 1:(n_actors-1)) {
      for (j in (i+1):n_actors) {
        current_tie <- prev_network[i, j]
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
          prob_form <- 0.05 * exp(-2 * tolerance_diff)
          if (runif(1) < prob_form) {
            friendship_networks[i, j, wave] <- 1
            friendship_networks[j, i, wave] <- 1
          }
        }
      }
    }
  }

  # Create actor attributes
  attributes <- data.frame(
    id = 1:n_actors,
    age = sample(18:65, n_actors, replace = TRUE),
    education = sample(c("High School", "College", "Graduate"), n_actors,
                      replace = TRUE, prob = c(0.5, 0.3, 0.2)),
    initial_group = c(rep("Low", group_size), rep("High", n_actors - group_size))
  )

  cat("Basic example dataset created!\n")
  cat("- Initial tolerance range:", round(range(initial_tolerance), 2), "\n")
  cat("- Final tolerance range:", round(range(tolerance_waves[,n_waves]), 2), "\n")
  cat("- Network density (wave 1):", round(sum(friendship_networks[,,1])/(n_actors*(n_actors-1)), 3), "\n")
  cat("- Network density (final):", round(sum(friendship_networks[,,n_waves])/(n_actors*(n_actors-1)), 3), "\n")

  return(list(
    friendship = friendship_networks,
    tolerance = tolerance_waves,
    attributes = attributes,
    n_actors = n_actors,
    n_waves = n_waves,
    description = "Basic example dataset for RSiena tolerance research"
  ))
}

# ============================================================================
# SECTION 2: INTERVENTION EXAMPLE DATA
# ============================================================================

#' Generate intervention example dataset
#'
#' Creates data with clear intervention effect for testing
#' @param n_actors Number of actors
#' @param n_waves Number of waves
#' @param intervention_proportion Proportion of actors to intervene on
#' @return Intervention example dataset
generate_intervention_example <- function(n_actors = 40, n_waves = 4,
                                        intervention_proportion = 0.25) {

  cat("Generating intervention example dataset...\n")

  # Start with basic dataset
  basic_data <- generate_basic_example(n_actors, n_waves)

  # Select intervention targets (highest degree actors in wave 1)
  wave1_network <- basic_data$friendship[,,1]
  degrees <- rowSums(wave1_network)
  n_intervention <- round(n_actors * intervention_proportion)
  intervention_actors <- order(degrees, decreasing = TRUE)[1:n_intervention]

  cat("Selected", length(intervention_actors), "actors for intervention\n")
  cat("Average degree of intervention actors:", round(mean(degrees[intervention_actors]), 2), "\n")

  # Apply intervention effect starting from wave 2
  modified_tolerance <- basic_data$tolerance

  for (wave in 2:n_waves) {
    # Intervention effect decays over time
    effect_size <- 0.4 * exp(-0.2 * (wave - 2))

    # Boost tolerance for intervention actors
    modified_tolerance[intervention_actors, wave] <-
      pmin(1, modified_tolerance[intervention_actors, wave] + effect_size)

    cat("Wave", wave, "intervention effect:", round(effect_size, 3), "\n")
  }

  # Update result
  result <- basic_data
  result$tolerance <- modified_tolerance
  result$intervention_actors <- intervention_actors
  result$description <- "Intervention example dataset with central targeting"

  return(result)
}

# ============================================================================
# SECTION 3: COMPLEX EXAMPLE DATA
# ============================================================================

#' Generate complex multi-network example
#'
#' Creates sophisticated dataset with multiple networks and behaviors
#' @param n_actors Number of actors
#' @param n_waves Number of waves
#' @return Complex example dataset
generate_complex_example <- function(n_actors = 50, n_waves = 3) {

  cat("Generating complex multi-network example dataset...\n")

  # Initialize with realistic tolerance distribution
  initial_tolerance <- pmax(0, pmin(1, rnorm(n_actors, 0.5, 0.3)))

  # Initialize cooperation behavior
  initial_cooperation <- pmax(0, pmin(1, 0.7 * initial_tolerance + rnorm(n_actors, 0, 0.2)))

  # Create networks
  friendship_networks <- array(0, dim = c(n_actors, n_actors, n_waves))
  cooperation_networks <- array(0, dim = c(n_actors, n_actors, n_waves))

  # Initialize behaviors
  tolerance_waves <- matrix(0, nrow = n_actors, ncol = n_waves)
  cooperation_waves <- matrix(0, nrow = n_actors, ncol = n_waves)

  tolerance_waves[, 1] <- initial_tolerance
  cooperation_waves[, 1] <- initial_cooperation

  # Create initial networks
  for (i in 1:(n_actors-1)) {
    for (j in (i+1):n_actors) {
      # Friendship based on tolerance similarity
      tolerance_sim <- 1 - abs(initial_tolerance[i] - initial_tolerance[j])
      prob_friendship <- 0.15 + 0.25 * tolerance_sim

      if (runif(1) < prob_friendship) {
        friendship_networks[i, j, 1] <- 1
        friendship_networks[j, i, 1] <- 1
      }

      # Cooperation based on both tolerance and cooperation behavior
      coop_potential <- min(initial_tolerance[i], initial_tolerance[j]) *
                       (initial_cooperation[i] + initial_cooperation[j]) / 2
      prob_cooperation <- 0.1 + 0.2 * coop_potential

      if (runif(1) < prob_cooperation) {
        cooperation_networks[i, j, 1] <- 1
        cooperation_networks[j, i, 1] <- 1
      }
    }
  }

  # Evolve over waves
  for (wave in 2:n_waves) {
    prev_tolerance <- tolerance_waves[, wave-1]
    prev_cooperation <- cooperation_waves[, wave-1]
    prev_friend_net <- friendship_networks[, , wave-1]
    prev_coop_net <- cooperation_networks[, , wave-1]

    # Update tolerance
    new_tolerance <- prev_tolerance
    for (i in 1:n_actors) {
      friends <- which(prev_friend_net[i, ] == 1)
      if (length(friends) > 0) {
        peer_tolerance <- mean(prev_tolerance[friends])
        influence <- 0.3 * (peer_tolerance - prev_tolerance[i])
        new_tolerance[i] <- pmax(0, pmin(1, prev_tolerance[i] + influence + rnorm(1, 0, 0.03)))
      } else {
        new_tolerance[i] <- pmax(0, pmin(1, prev_tolerance[i] + rnorm(1, 0, 0.05)))
      }
    }

    # Update cooperation behavior
    new_cooperation <- prev_cooperation
    for (i in 1:n_actors) {
      # Cooperation influenced by both networks
      friends <- which(prev_friend_net[i, ] == 1)
      coop_partners <- which(prev_coop_net[i, ] == 1)

      total_influence <- 0
      if (length(friends) > 0) {
        total_influence <- total_influence + 0.2 * (mean(prev_cooperation[friends]) - prev_cooperation[i])
      }
      if (length(coop_partners) > 0) {
        total_influence <- total_influence + 0.3 * (mean(prev_cooperation[coop_partners]) - prev_cooperation[i])
      }

      # Tolerance also influences cooperation
      tolerance_effect <- 0.1 * (new_tolerance[i] - prev_cooperation[i])

      new_cooperation[i] <- pmax(0, pmin(1, prev_cooperation[i] + total_influence +
                                        tolerance_effect + rnorm(1, 0, 0.03)))
    }

    tolerance_waves[, wave] <- new_tolerance
    cooperation_waves[, wave] <- new_cooperation

    # Update networks (simplified evolution)
    friendship_networks[, , wave] <- prev_friend_net
    cooperation_networks[, , wave] <- prev_coop_net

    # Some network changes
    for (i in 1:(n_actors-1)) {
      for (j in (i+1):n_actors) {
        # Friendship changes
        tolerance_diff <- abs(new_tolerance[i] - new_tolerance[j])
        if (prev_friend_net[i, j] == 1) {
          if (runif(1) < 0.1 * tolerance_diff^2) {
            friendship_networks[i, j, wave] <- 0
            friendship_networks[j, i, wave] <- 0
          }
        } else {
          if (runif(1) < 0.05 * exp(-2 * tolerance_diff)) {
            friendship_networks[i, j, wave] <- 1
            friendship_networks[j, i, wave] <- 1
          }
        }

        # Cooperation network changes
        coop_compatibility <- min(new_tolerance[i], new_tolerance[j]) *
                             (new_cooperation[i] + new_cooperation[j]) / 2
        if (prev_coop_net[i, j] == 1) {
          if (runif(1) < 0.15 * (1 - coop_compatibility)) {
            cooperation_networks[i, j, wave] <- 0
            cooperation_networks[j, i, wave] <- 0
          }
        } else {
          if (runif(1) < 0.08 * coop_compatibility) {
            cooperation_networks[i, j, wave] <- 1
            cooperation_networks[j, i, wave] <- 1
          }
        }
      }
    }
  }

  # Create detailed attributes
  attributes <- data.frame(
    id = 1:n_actors,
    age = sample(18:70, n_actors, replace = TRUE),
    education = sample(1:4, n_actors, replace = TRUE),
    income = sample(c("Low", "Medium", "High"), n_actors,
                   replace = TRUE, prob = c(0.4, 0.4, 0.2)),
    political = rnorm(n_actors, 0, 1),
    initial_tolerance_group = cut(initial_tolerance,
                                 breaks = c(0, 0.33, 0.67, 1),
                                 labels = c("Low", "Medium", "High"))
  )

  cat("Complex example dataset created!\n")
  cat("- Friendship density (final):",
      round(sum(friendship_networks[,,n_waves])/(n_actors*(n_actors-1)), 3), "\n")
  cat("- Cooperation density (final):",
      round(sum(cooperation_networks[,,n_waves])/(n_actors*(n_actors-1)), 3), "\n")
  cat("- Tolerance correlation with cooperation (final):",
      round(cor(tolerance_waves[,n_waves], cooperation_waves[,n_waves]), 3), "\n")

  return(list(
    friendship = friendship_networks,
    cooperation = cooperation_networks,
    tolerance = tolerance_waves,
    cooperation_behavior = cooperation_waves,
    attributes = attributes,
    n_actors = n_actors,
    n_waves = n_waves,
    description = "Complex multi-network example with tolerance and cooperation co-evolution"
  ))
}

# ============================================================================
# SECTION 4: DATA EXPORT AND VALIDATION
# ============================================================================

#' Export example dataset to RSiena format
#'
#' Converts example data to proper RSiena data objects
#' @param example_data Output from any generate_*_example function
#' @return RSiena data object ready for analysis
export_to_siena_format <- function(example_data) {

  cat("Converting to RSiena format...\n")

  # Create dependent variables
  friendship_siena <- sienaDependent(example_data$friendship, type = "oneMode")
  tolerance_siena <- sienaDependent(example_data$tolerance, type = "behavior")

  # Create covariates
  age_covar <- coCovar(example_data$attributes$age)

  # Education as factor
  if ("education" %in% names(example_data$attributes)) {
    if (is.character(example_data$attributes$education)) {
      education_numeric <- as.numeric(as.factor(example_data$attributes$education))
    } else {
      education_numeric <- example_data$attributes$education
    }
    education_covar <- coCovar(education_numeric)
  }

  # Create basic RSiena data object
  if (exists("education_covar")) {
    siena_data <- sienaDataCreate(
      friendship = friendship_siena,
      tolerance = tolerance_siena,
      age = age_covar,
      education = education_covar
    )
  } else {
    siena_data <- sienaDataCreate(
      friendship = friendship_siena,
      tolerance = tolerance_siena,
      age = age_covar
    )
  }

  # Add cooperation network if present
  if ("cooperation" %in% names(example_data)) {
    cooperation_siena <- sienaDependent(example_data$cooperation, type = "oneMode")
    cooperation_behavior_siena <- sienaDependent(example_data$cooperation_behavior, type = "behavior")

    # Update siena data object
    siena_data <- sienaDataCreate(
      friendship = friendship_siena,
      cooperation = cooperation_siena,
      tolerance = tolerance_siena,
      coop_behavior = cooperation_behavior_siena,
      age = age_covar,
      education = education_covar
    )
  }

  cat("RSiena data object created successfully!\n")
  return(siena_data)
}

#' Validate example dataset
#'
#' Performs basic checks on generated data quality
#' @param example_data Generated example dataset
validate_example_data <- function(example_data) {

  cat("Validating example dataset...\n")

  n_actors <- example_data$n_actors
  n_waves <- example_data$n_waves

  # Check dimensions
  stopifnot(dim(example_data$friendship) == c(n_actors, n_actors, n_waves))
  stopifnot(dim(example_data$tolerance) == c(n_actors, n_waves))

  # Check value ranges
  stopifnot(all(example_data$friendship %in% c(0, 1)))
  stopifnot(all(example_data$tolerance >= 0 & example_data$tolerance <= 1))

  # Check network symmetry
  for (wave in 1:n_waves) {
    net <- example_data$friendship[,,wave]
    stopifnot(all(net == t(net)))  # Symmetric
    stopifnot(all(diag(net) == 0))  # No self-loops
  }

  # Check attributes
  stopifnot(nrow(example_data$attributes) == n_actors)

  cat("✓ All validation checks passed!\n")
  cat("Dataset is ready for RSiena analysis.\n")

  return(TRUE)
}

# ============================================================================
# SECTION 5: DEMONSTRATION DATA CREATION
# ============================================================================

#' Create all example datasets
#'
#' Generates complete suite of example data for demonstrations
create_all_examples <- function() {

  cat("Creating complete suite of example datasets...\n")

  # Basic example
  cat("\n1. BASIC EXAMPLE:\n")
  basic_data <- generate_basic_example(n_actors = 30, n_waves = 3)
  validate_example_data(basic_data)

  # Intervention example
  cat("\n2. INTERVENTION EXAMPLE:\n")
  intervention_data <- generate_intervention_example(n_actors = 40, n_waves = 4)
  validate_example_data(intervention_data)

  # Complex example
  cat("\n3. COMPLEX EXAMPLE:\n")
  complex_data <- generate_complex_example(n_actors = 50, n_waves = 3)
  validate_example_data(complex_data)

  # Save all examples
  example_datasets <- list(
    basic = basic_data,
    intervention = intervention_data,
    complex = complex_data
  )

  save(example_datasets, file = "rsiena_example_datasets.RData")
  cat("\n✓ All example datasets saved to: rsiena_example_datasets.RData\n")

  # Create RSiena format versions
  cat("\nCreating RSiena format versions...\n")
  siena_datasets <- list(
    basic = export_to_siena_format(basic_data),
    intervention = export_to_siena_format(intervention_data),
    complex = export_to_siena_format(complex_data)
  )

  save(siena_datasets, file = "rsiena_formatted_datasets.RData")
  cat("✓ RSiena formatted datasets saved to: rsiena_formatted_datasets.RData\n")

  return(list(
    raw = example_datasets,
    siena = siena_datasets
  ))
}

# ============================================================================
# SECTION 6: QUICK TEST FUNCTIONS
# ============================================================================

#' Quick test of basic RSiena workflow
#'
#' Runs minimal RSiena analysis on example data for testing
#' @param example_data Example dataset to test
quick_siena_test <- function(example_data = NULL) {

  if (is.null(example_data)) {
    cat("Generating test dataset...\n")
    example_data <- generate_basic_example(n_actors = 25, n_waves = 3)
  }

  cat("Running quick RSiena test...\n")

  # Convert to RSiena format
  siena_data <- export_to_siena_format(example_data)

  # Basic effects
  effects <- getEffects(siena_data)
  effects <- includeEffects(effects, transTrip, name = "friendship")
  effects <- includeEffects(effects, simX, name = "friendship", interaction1 = "tolerance")
  effects <- includeEffects(effects, avAlt, name = "tolerance", interaction1 = "friendship")

  cat("Effects specified:\n")
  print(effects[effects$include == TRUE, c("name", "effectName")])

  # Quick estimation (minimal iterations)
  algorithm <- sienaAlgorithmCreate(
    projname = "quick_test",
    nsub = 2,
    n3 = 200,  # Very short for testing
    seed = 789
  )

  cat("Estimating model (quick test)...\n")
  results <- siena07(algorithm, data = siena_data, effects = effects, verbose = FALSE)

  # Basic results
  cat("\nQuick test results:\n")
  cat("Convergence t-ratio max:", round(max(abs(results$tconv)), 3), "\n")
  cat("Number of parameters:", length(results$theta), "\n")

  if (max(abs(results$tconv)) < 0.5) {
    cat("✓ Basic functionality working!\n")
  } else {
    cat("⚠ Convergence issues (expected with minimal iterations)\n")
  }

  return(results)
}

# ============================================================================
# EXECUTION
# ============================================================================

# Create example datasets when sourced
if (interactive() || !exists("skip_examples")) {
  cat("=" * 60, "\n")
  cat("RSIENA EXAMPLE DATA GENERATOR\n")
  cat("=" * 60, "\n")

  # Create all examples
  all_examples <- create_all_examples()

  # Run quick test
  cat("\n" + "=" * 60 + "\n")
  cat("QUICK FUNCTIONALITY TEST\n")
  cat("=" * 60, "\n")

  test_results <- quick_siena_test(all_examples$raw$basic)

  cat("\n" + "=" * 60 + "\n")
  cat("EXAMPLE DATA GENERATION COMPLETE!\n")
  cat("=" * 60, "\n")

  cat("\nFiles created:\n")
  cat("• rsiena_example_datasets.RData - Raw example datasets\n")
  cat("• rsiena_formatted_datasets.RData - RSiena formatted datasets\n")

  cat("\nUsage:\n")
  cat("load('rsiena_example_datasets.RData')\n")
  cat("basic_data <- example_datasets$basic\n")
  cat("intervention_data <- example_datasets$intervention\n")
  cat("complex_data <- example_datasets$complex\n")
}

# ============================================================================
# USAGE EXAMPLES
# ============================================================================

cat("\n" + "=" * 60 + "\n")
cat("USAGE EXAMPLES\n")
cat("=" * 60, "\n")

cat("\n# Load example datasets\n")
cat("load('rsiena_example_datasets.RData')\n")

cat("\n# Use basic dataset\n")
cat("basic_data <- example_datasets$basic\n")
cat("siena_data <- export_to_siena_format(basic_data)\n")

cat("\n# Generate custom dataset\n")
cat("my_data <- generate_basic_example(n_actors = 40, n_waves = 4)\n")
cat("validate_example_data(my_data)\n")

cat("\n# Quick test\n")
cat("test_results <- quick_siena_test()\n")

cat("\n" + "=" * 60 + "\n")