# ============================================================================
# MASTER DATA GENERATOR: REALISTIC TOLERANCE INTERVENTION SIMULATION
# ============================================================================
#
# Purpose: Generate empirically-grounded simulation data for tolerance
#          intervention research based on German school study design
# Author: ABM Research Project
# Date: 2025
# Usage: Create high-quality, realistic data for comprehensive testing
# ============================================================================

# Load required libraries
library(RSiena)
library(network)
library(igraph)
library(Matrix)
library(MASS)  # For multivariate normal
library(mvtnorm)  # For more multivariate distributions

# Set random seed for reproducibility
set.seed(12345)

# ============================================================================
# SECTION 1: EMPIRICAL CALIBRATION PARAMETERS
# ============================================================================

#' Empirical parameters from German school literature
GERMAN_SCHOOL_PARAMS <- list(
  # Network structure
  network = list(
    density = 0.12,
    transitivity = 0.35,
    ethnic_homophily = 0.45,
    gender_homophily = 0.25,
    reciprocity = 0.6,
    degree_variance = 4.5
  ),

  # Tolerance measures
  tolerance = list(
    mean = 3.2,  # On 1-5 scale
    sd = 0.8,
    reliability = 0.85,
    between_classroom_var = 0.20,
    ethnic_majority_mean = 3.0,
    ethnic_minority_mean = 3.6
  ),

  # Cooperation behavior
  cooperation = list(
    mean = 0.65,  # Proportion scale
    sd = 0.25,
    tolerance_correlation = 0.55,
    network_influence = 0.30
  ),

  # Demographics
  demographics = list(
    minority_proportion = 0.30,
    gender_balance = 0.50,
    age_range = c(15, 18),
    ses_levels = 3
  ),

  # Missing data patterns
  attrition = list(
    wave2_retention = 0.95,
    wave3_retention = 0.90,
    selective_dropout = 0.15  # Higher for low-tolerance students
  )
)

# ============================================================================
# SECTION 2: CLASSROOM NETWORK GENERATOR
# ============================================================================

#' Generate realistic classroom network structure
#'
#' Creates empirically-grounded friendship networks with proper homophily
#' @param n_students Number of students in classroom
#' @param minority_prop Proportion of ethnic minority students
#' @param scenario Classroom scenario ("diverse", "homogeneous", "integrated", "conflicted")
#' @return Network adjacency matrix
generate_classroom_network <- function(n_students = 30, minority_prop = 0.30,
                                     scenario = "diverse") {

  cat("Generating classroom network structure...\n")
  cat("- Students:", n_students, "\n")
  cat("- Minority proportion:", minority_prop, "\n")
  cat("- Scenario:", scenario, "\n")

  # Create demographic attributes
  n_minority <- round(n_students * minority_prop)
  n_majority <- n_students - n_minority

  ethnicity <- c(rep("majority", n_majority), rep("minority", n_minority))
  gender <- sample(c("male", "female"), n_students, replace = TRUE, prob = c(0.5, 0.5))
  age <- sample(15:18, n_students, replace = TRUE, prob = c(0.3, 0.4, 0.25, 0.05))
  ses <- sample(1:3, n_students, replace = TRUE, prob = c(0.4, 0.4, 0.2))

  # Initialize network
  network_matrix <- matrix(0, nrow = n_students, ncol = n_students)

  # Generate friendships with homophily
  for (i in 1:(n_students-1)) {
    for (j in (i+1):n_students) {

      # Base friendship probability
      base_prob <- GERMAN_SCHOOL_PARAMS$network$density * 2  # Adjust for undirected

      # Ethnic homophily effect
      ethnic_same <- ethnicity[i] == ethnicity[j]
      ethnic_bonus <- ifelse(ethnic_same, GERMAN_SCHOOL_PARAMS$network$ethnic_homophily, 0)

      # Gender homophily effect
      gender_same <- gender[i] == gender[j]
      gender_bonus <- ifelse(gender_same, GERMAN_SCHOOL_PARAMS$network$gender_homophily, 0)

      # Age proximity effect
      age_diff <- abs(age[i] - age[j])
      age_bonus <- max(0, 0.2 - 0.1 * age_diff)

      # SES similarity effect
      ses_diff <- abs(ses[i] - ses[j])
      ses_bonus <- max(0, 0.15 - 0.075 * ses_diff)

      # Scenario-specific adjustments
      scenario_modifier <- switch(scenario,
        "diverse" = 1.0,
        "homogeneous" = ifelse(ethnic_same, 1.2, 0.6),
        "integrated" = ifelse(ethnic_same, 0.9, 1.3),
        "conflicted" = ifelse(ethnic_same, 1.4, 0.4)
      )

      # Calculate final probability
      prob_friendship <- (base_prob + ethnic_bonus + gender_bonus +
                         age_bonus + ses_bonus) * scenario_modifier
      prob_friendship <- pmax(0.01, pmin(0.8, prob_friendship))

      # Create friendship
      if (runif(1) < prob_friendship) {
        network_matrix[i, j] <- 1
        network_matrix[j, i] <- 1
      }
    }
  }

  # Ensure transitivity by adding triadic closure
  for (iter in 1:5) {  # Multiple iterations for gradual closure
    for (i in 1:n_students) {
      friends_i <- which(network_matrix[i, ] == 1)
      if (length(friends_i) >= 2) {
        for (j in 1:(length(friends_i)-1)) {
          for (k in (j+1):length(friends_i)) {
            friend_j <- friends_i[j]
            friend_k <- friends_i[k]

            # If j and k are not friends, chance of triadic closure
            if (network_matrix[friend_j, friend_k] == 0) {
              closure_prob <- 0.3  # Moderate triadic closure
              if (runif(1) < closure_prob) {
                network_matrix[friend_j, friend_k] <- 1
                network_matrix[friend_k, friend_j] <- 1
              }
            }
          }
        }
      }
    }
  }

  # Create student attributes data frame
  attributes <- data.frame(
    id = 1:n_students,
    ethnicity = ethnicity,
    gender = gender,
    age = age,
    ses = ses,
    stringsAsFactors = FALSE
  )

  # Calculate network statistics
  actual_density <- sum(network_matrix) / (n_students * (n_students - 1))
  g <- graph_from_adjacency_matrix(network_matrix, mode = "undirected")
  actual_transitivity <- transitivity(g, type = "global")

  cat("Network statistics:\n")
  cat("- Density:", round(actual_density, 3), "\n")
  cat("- Transitivity:", round(actual_transitivity, 3), "\n")
  cat("- Connected components:", components(g)$no, "\n")

  return(list(
    network = network_matrix,
    attributes = attributes,
    scenario = scenario,
    statistics = list(
      density = actual_density,
      transitivity = actual_transitivity,
      components = components(g)$no
    )
  ))
}

# ============================================================================
# SECTION 3: TOLERANCE BEHAVIOR GENERATOR
# ============================================================================

#' Generate realistic tolerance measures
#'
#' Creates tolerance scores with proper distributions and correlations
#' @param attributes Student attributes from network generator
#' @param scenario Classroom scenario affecting tolerance levels
#' @return Tolerance measures and related variables
generate_tolerance_measures <- function(attributes, scenario = "diverse") {

  cat("Generating tolerance behavioral measures...\n")

  n_students <- nrow(attributes)

  # Base tolerance levels by scenario
  scenario_means <- list(
    "diverse" = 3.2,
    "homogeneous" = 2.8,
    "integrated" = 3.8,
    "conflicted" = 2.5
  )

  base_mean <- scenario_means[[scenario]]

  # Ethnic group differences in tolerance
  tolerance_means <- ifelse(
    attributes$ethnicity == "majority",
    base_mean + GERMAN_SCHOOL_PARAMS$tolerance$ethnic_majority_mean - 3.2,
    base_mean + GERMAN_SCHOOL_PARAMS$tolerance$ethnic_minority_mean - 3.2
  )

  # SES effects on tolerance
  ses_effects <- (attributes$ses - 2) * 0.2  # Higher SES -> higher tolerance
  tolerance_means <- tolerance_means + ses_effects

  # Age effects (older students more tolerant)
  age_effects <- (attributes$age - 16) * 0.15
  tolerance_means <- tolerance_means + age_effects

  # Generate tolerance scores with measurement error
  true_tolerance <- pmax(1, pmin(5, rnorm(n_students, tolerance_means,
                                         GERMAN_SCHOOL_PARAMS$tolerance$sd)))

  # Add measurement error (reliability = 0.85)
  measurement_error_var <- var(true_tolerance) * (1 - GERMAN_SCHOOL_PARAMS$tolerance$reliability) /
                          GERMAN_SCHOOL_PARAMS$tolerance$reliability
  measurement_error <- rnorm(n_students, 0, sqrt(measurement_error_var))
  observed_tolerance <- pmax(1, pmin(5, true_tolerance + measurement_error))

  # Generate correlated cooperation behavior
  cooperation_correlation <- GERMAN_SCHOOL_PARAMS$cooperation$tolerance_correlation
  cooperation_residual_sd <- sqrt(1 - cooperation_correlation^2) * 0.25

  cooperation <- (observed_tolerance - 1) / 4  # Scale to 0-1
  cooperation <- cooperation * 0.8 + 0.1  # Adjust range to 0.1-0.9
  cooperation <- cooperation + rnorm(n_students, 0, cooperation_residual_sd)
  cooperation <- pmax(0, pmin(1, cooperation))

  # Generate prejudice as control variable (negatively correlated with tolerance)
  prejudice <- 6 - observed_tolerance + rnorm(n_students, 0, 0.5)
  prejudice <- pmax(1, pmin(5, prejudice))

  # Generate other relevant measures
  openness <- observed_tolerance + rnorm(n_students, 0, 0.4)
  openness <- pmax(1, pmin(5, openness))

  social_distance <- 6 - observed_tolerance + rnorm(n_students, 0, 0.6)
  social_distance <- pmax(1, pmin(5, social_distance))

  cat("Tolerance measures generated:\n")
  cat("- Mean tolerance:", round(mean(observed_tolerance), 2), "\n")
  cat("- Tolerance SD:", round(sd(observed_tolerance), 2), "\n")
  cat("- Tolerance-cooperation correlation:", round(cor(observed_tolerance, cooperation), 2), "\n")
  cat("- Tolerance-prejudice correlation:", round(cor(observed_tolerance, prejudice), 2), "\n")

  return(data.frame(
    id = attributes$id,
    tolerance_true = true_tolerance,
    tolerance = observed_tolerance,
    cooperation = cooperation,
    prejudice = prejudice,
    openness = openness,
    social_distance = social_distance,
    stringsAsFactors = FALSE
  ))
}

# ============================================================================
# SECTION 4: LONGITUDINAL EVOLUTION SIMULATOR
# ============================================================================

#' Simulate network and behavior evolution over waves
#'
#' Creates realistic longitudinal changes with social influence
#' @param initial_network Initial network structure
#' @param initial_attributes Student attributes
#' @param initial_behaviors Initial behavioral measures
#' @param n_waves Number of observation waves
#' @return Longitudinal data structure
simulate_network_evolution <- function(initial_network, initial_attributes,
                                     initial_behaviors, n_waves = 3) {

  cat("Simulating network evolution over", n_waves, "waves...\n")

  n_students <- nrow(initial_attributes)

  # Initialize storage arrays
  networks <- array(0, dim = c(n_students, n_students, n_waves))
  tolerance_waves <- matrix(0, nrow = n_students, ncol = n_waves)
  cooperation_waves <- matrix(0, nrow = n_students, ncol = n_waves)

  # Set initial values
  networks[, , 1] <- initial_network$network
  tolerance_waves[, 1] <- initial_behaviors$tolerance
  cooperation_waves[, 1] <- initial_behaviors$cooperation

  # Simulate evolution
  for (wave in 2:n_waves) {
    cat("Processing wave", wave, "...\n")

    prev_network <- networks[, , wave - 1]
    prev_tolerance <- tolerance_waves[, wave - 1]
    prev_cooperation <- cooperation_waves[, wave - 1]

    # Update tolerance through social influence
    new_tolerance <- prev_tolerance
    for (i in 1:n_students) {
      friends <- which(prev_network[i, ] == 1)
      if (length(friends) > 0) {
        # Peer influence on tolerance
        peer_tolerance_mean <- mean(prev_tolerance[friends])
        influence_strength <- 0.25 * length(friends) / (length(friends) + 2)  # Diminishing returns
        influence_effect <- influence_strength * (peer_tolerance_mean - prev_tolerance[i])

        # Individual stability
        stability_effect <- 0.8 * prev_tolerance[i]

        # Random innovation
        innovation <- rnorm(1, 0, 0.15)

        # Combine effects
        new_tolerance[i] <- 0.2 * stability_effect + influence_effect + innovation
        new_tolerance[i] <- pmax(1, pmin(5, new_tolerance[i]))
      } else {
        # Isolated actors: random walk
        new_tolerance[i] <- pmax(1, pmin(5, prev_tolerance[i] + rnorm(1, 0, 0.2)))
      }
    }

    tolerance_waves[, wave] <- new_tolerance

    # Update cooperation behavior
    new_cooperation <- prev_cooperation
    for (i in 1:n_students) {
      friends <- which(prev_network[i, ] == 1)

      # Direct tolerance effect
      tolerance_effect <- 0.3 * (new_tolerance[i] - 1) / 4  # Scale to 0-1

      # Peer influence on cooperation
      peer_influence <- 0
      if (length(friends) > 0) {
        peer_cooperation_mean <- mean(prev_cooperation[friends])
        peer_influence <- 0.2 * (peer_cooperation_mean - prev_cooperation[i])
      }

      # Individual stability
      stability <- 0.6 * prev_cooperation[i]

      # Random change
      innovation <- rnorm(1, 0, 0.1)

      # Combine effects
      new_cooperation[i] <- stability + tolerance_effect + peer_influence + innovation
      new_cooperation[i] <- pmax(0, pmin(1, new_cooperation[i]))
    }

    cooperation_waves[, wave] <- new_cooperation

    # Update network structure
    new_network <- prev_network

    # Network stability (most ties persist)
    stability_matrix <- matrix(runif(n_students * n_students), n_students, n_students)
    stability_threshold <- 0.9  # 90% of ties persist

    # Dissolution of existing ties
    for (i in 1:(n_students-1)) {
      for (j in (i+1):n_students) {
        if (prev_network[i, j] == 1) {
          # Dissolution based on tolerance divergence
          tolerance_diff <- abs(new_tolerance[i] - new_tolerance[j])
          dissolution_prob <- 0.05 + 0.1 * (tolerance_diff / 4)^2

          if (runif(1) < dissolution_prob) {
            new_network[i, j] <- 0
            new_network[j, i] <- 0
          }
        }
      }
    }

    # Formation of new ties
    for (i in 1:(n_students-1)) {
      for (j in (i+1):n_students) {
        if (prev_network[i, j] == 0) {
          # Formation based on tolerance similarity and mutual friends
          tolerance_similarity <- 1 - abs(new_tolerance[i] - new_tolerance[j]) / 4

          # Mutual friends effect
          mutual_friends <- sum(prev_network[i, ] * prev_network[j, ])
          mutual_friends_effect <- min(0.3, 0.1 * mutual_friends)

          # Ethnic homophily
          ethnic_same <- initial_attributes$ethnicity[i] == initial_attributes$ethnicity[j]
          ethnic_bonus <- ifelse(ethnic_same, 0.1, 0)

          # Formation probability
          formation_prob <- 0.02 + 0.08 * tolerance_similarity +
                           mutual_friends_effect + ethnic_bonus

          if (runif(1) < formation_prob) {
            new_network[i, j] <- 1
            new_network[j, i] <- 1
          }
        }
      }
    }

    networks[, , wave] <- new_network
  }

  # Calculate final statistics
  final_density <- sum(networks[, , n_waves]) / (n_students * (n_students - 1))
  tolerance_change <- mean(tolerance_waves[, n_waves] - tolerance_waves[, 1])
  cooperation_change <- mean(cooperation_waves[, n_waves] - cooperation_waves[, 1])

  cat("Evolution complete:\n")
  cat("- Final network density:", round(final_density, 3), "\n")
  cat("- Average tolerance change:", round(tolerance_change, 3), "\n")
  cat("- Average cooperation change:", round(cooperation_change, 3), "\n")

  return(list(
    networks = networks,
    tolerance = tolerance_waves,
    cooperation = cooperation_waves,
    n_waves = n_waves,
    statistics = list(
      final_density = final_density,
      tolerance_change = tolerance_change,
      cooperation_change = cooperation_change
    )
  ))
}

# ============================================================================
# SECTION 5: INTERVENTION SIMULATOR
# ============================================================================

#' Simulate tolerance intervention with realistic effects
#'
#' Applies intervention targeting strategies with empirical effect sizes
#' @param longitudinal_data Output from simulate_network_evolution
#' @param attributes Student attributes
#' @param targeting Strategy: "central", "peripheral", "random", "clustered"
#' @param target_proportion Proportion of students to target
#' @param effect_magnitude Intervention effect size in SD units
#' @return Data with intervention effects applied
simulate_tolerance_intervention <- function(longitudinal_data, attributes,
                                          targeting = "central",
                                          target_proportion = 0.20,
                                          effect_magnitude = 1.0) {

  cat("Simulating tolerance intervention...\n")
  cat("- Targeting strategy:", targeting, "\n")
  cat("- Target proportion:", target_proportion, "\n")
  cat("- Effect magnitude:", effect_magnitude, "SD units\n")

  n_students <- nrow(attributes)
  n_waves <- longitudinal_data$n_waves
  n_targets <- round(n_students * target_proportion)

  # Select intervention targets based on strategy
  wave1_network <- longitudinal_data$networks[, , 1]
  degrees <- rowSums(wave1_network)

  targets <- switch(targeting,
    "central" = order(degrees, decreasing = TRUE)[1:n_targets],
    "peripheral" = order(degrees, decreasing = FALSE)[1:n_targets],
    "random" = sample(1:n_students, n_targets),
    "clustered" = select_clustered_targets(wave1_network, n_targets),
    stop("Unknown targeting strategy: ", targeting)
  )

  cat("Selected targets with average degree:", round(mean(degrees[targets]), 2), "\n")

  # Apply intervention effects
  modified_tolerance <- longitudinal_data$tolerance
  modified_cooperation <- longitudinal_data$cooperation

  # Base effect size in original scale (1-5 scale, SD ~ 0.8)
  base_effect <- effect_magnitude * 0.8

  for (wave in 2:n_waves) {
    # Effect decays over time
    decay_factor <- exp(-0.3 * (wave - 2))
    current_effect <- base_effect * decay_factor

    # Apply to targets
    modified_tolerance[targets, wave] <-
      pmin(5, modified_tolerance[targets, wave] + current_effect)

    # Spillover to cooperation
    cooperation_spillover <- current_effect * 0.4 / 4  # Scale to cooperation range
    modified_cooperation[targets, wave] <-
      pmin(1, modified_cooperation[targets, wave] + cooperation_spillover)

    # Social influence spreads intervention effects
    if (wave >= 3) {  # Influence starts in wave 3
      influence_network <- longitudinal_data$networks[, , wave - 1]

      for (non_target in setdiff(1:n_students, targets)) {
        target_friends <- intersect(which(influence_network[non_target, ] == 1), targets)
        if (length(target_friends) > 0) {
          # Influence proportional to number of treated friends
          influence_strength <- min(0.5, 0.2 * length(target_friends))
          spillover_effect <- current_effect * influence_strength

          modified_tolerance[non_target, wave] <-
            pmin(5, modified_tolerance[non_target, wave] + spillover_effect)

          cooperation_spillover <- spillover_effect * 0.3 / 4
          modified_cooperation[non_target, wave] <-
            pmin(1, modified_cooperation[non_target, wave] + cooperation_spillover)
        }
      }
    }

    cat("Wave", wave, "effect size:", round(current_effect, 3), "\n")
  }

  # Calculate intervention effectiveness
  pre_intervention <- longitudinal_data$tolerance[, 1]
  post_intervention <- modified_tolerance[, n_waves]

  target_effect <- mean(post_intervention[targets]) - mean(pre_intervention[targets])
  control_change <- mean(post_intervention[-targets]) - mean(pre_intervention[-targets])
  net_effect <- target_effect - control_change

  cat("Intervention effectiveness:\n")
  cat("- Direct effect on targets:", round(target_effect, 3), "\n")
  cat("- Control group change:", round(control_change, 3), "\n")
  cat("- Net intervention effect:", round(net_effect, 3), "\n")

  return(list(
    networks = longitudinal_data$networks,
    tolerance = modified_tolerance,
    cooperation = modified_cooperation,
    intervention_targets = targets,
    targeting_strategy = targeting,
    effect_magnitude = effect_magnitude,
    effectiveness = list(
      direct_effect = target_effect,
      control_change = control_change,
      net_effect = net_effect
    ),
    n_waves = n_waves
  ))
}

#' Select clustered intervention targets
#'
#' Identifies clusters of connected students for intervention
#' @param network Network adjacency matrix
#' @param n_targets Number of targets to select
#' @return Vector of target indices
select_clustered_targets <- function(network, n_targets) {
  g <- graph_from_adjacency_matrix(network, mode = "undirected")

  # Find communities
  communities <- cluster_louvain(g)
  community_sizes <- table(membership(communities))

  # Select largest communities
  large_communities <- which(community_sizes >= n_targets / 2)
  if (length(large_communities) == 0) {
    # Fallback to random selection
    return(sample(nrow(network), n_targets))
  }

  # Select from largest community
  target_community <- large_communities[which.max(community_sizes[large_communities])]
  community_members <- which(membership(communities) == target_community)

  # Select high-degree members from this community
  community_degrees <- rowSums(network[community_members, community_members])
  selected <- community_members[order(community_degrees, decreasing = TRUE)]

  return(selected[1:min(n_targets, length(selected))])
}

# ============================================================================
# SECTION 6: MISSING DATA SIMULATOR
# ============================================================================

#' Apply realistic missing data patterns
#'
#' Introduces missing data following empirical attrition patterns
#' @param intervention_data Complete intervention dataset
#' @param attributes Student attributes
#' @return Data with realistic missing values
apply_missing_data_patterns <- function(intervention_data, attributes) {

  cat("Applying realistic missing data patterns...\n")

  n_students <- nrow(attributes)
  n_waves <- intervention_data$n_waves

  # Create missingness indicators
  missing_network <- array(FALSE, dim = dim(intervention_data$networks))
  missing_tolerance <- matrix(FALSE, nrow = n_students, ncol = n_waves)
  missing_cooperation <- matrix(FALSE, nrow = n_students, ncol = n_waves)

  # Wave 1: No missing data (baseline)
  # Already initialized as FALSE

  # Wave-specific attrition
  retention_rates <- c(1.0,
                      GERMAN_SCHOOL_PARAMS$attrition$wave2_retention,
                      GERMAN_SCHOOL_PARAMS$attrition$wave3_retention)[1:n_waves]

  for (wave in 2:n_waves) {
    # Random attrition
    random_missing <- sample(1:n_students,
                            round(n_students * (1 - retention_rates[wave])))

    # Selective attrition (low-tolerance students more likely to drop out)
    if (wave >= 2) {
      tolerance_w1 <- intervention_data$tolerance[, 1]
      low_tolerance_prob <- pmax(0.05, 0.3 - 0.05 * tolerance_w1)  # Lower tolerance = higher dropout

      selective_missing <- which(runif(n_students) < low_tolerance_prob *
                                GERMAN_SCHOOL_PARAMS$attrition$selective_dropout)

      all_missing <- unique(c(random_missing, selective_missing))
      all_missing <- all_missing[1:min(length(all_missing),
                                      round(n_students * (1 - retention_rates[wave])))]
    } else {
      all_missing <- random_missing
    }

    # Apply missingness
    missing_tolerance[all_missing, wave] <- TRUE
    missing_cooperation[all_missing, wave] <- TRUE

    # Network data missing for missing individuals
    missing_network[all_missing, , wave] <- TRUE
    missing_network[, all_missing, wave] <- TRUE

    cat("Wave", wave, "missing:", length(all_missing), "students\n")
  }

  # Create data with missing values
  tolerance_with_missing <- intervention_data$tolerance
  cooperation_with_missing <- intervention_data$cooperation
  networks_with_missing <- intervention_data$networks

  tolerance_with_missing[missing_tolerance] <- NA
  cooperation_with_missing[missing_cooperation] <- NA
  networks_with_missing[missing_network] <- NA

  # Calculate missing data statistics
  missing_rates <- colMeans(missing_tolerance)

  cat("Missing data summary:\n")
  for (wave in 1:n_waves) {
    cat("- Wave", wave, "missing rate:", round(missing_rates[wave], 3), "\n")
  }

  return(list(
    networks = networks_with_missing,
    tolerance = tolerance_with_missing,
    cooperation = cooperation_with_missing,
    missing_patterns = list(
      tolerance = missing_tolerance,
      cooperation = missing_cooperation,
      network = missing_network
    ),
    missing_rates = missing_rates,
    intervention_targets = intervention_data$intervention_targets,
    targeting_strategy = intervention_data$targeting_strategy,
    effect_magnitude = intervention_data$effect_magnitude,
    effectiveness = intervention_data$effectiveness,
    n_waves = n_waves
  ))
}

# ============================================================================
# SECTION 7: COMPLETE SIMULATION PIPELINE
# ============================================================================

#' Generate complete classroom simulation dataset
#'
#' Master function creating full tolerance intervention simulation
#' @param n_students Number of students (default: 30)
#' @param n_waves Number of observation waves (default: 3)
#' @param minority_prop Proportion of ethnic minority students (default: 0.30)
#' @param scenario Classroom scenario (default: "diverse")
#' @param targeting Intervention targeting strategy (default: "central")
#' @param target_proportion Proportion to target (default: 0.20)
#' @param effect_magnitude Intervention effect size (default: 1.0)
#' @param include_missing Whether to include missing data (default: TRUE)
#' @return Complete simulation dataset
create_simulation_classroom <- function(n_students = 30, n_waves = 3,
                                       minority_prop = 0.30, scenario = "diverse",
                                       targeting = "central", target_proportion = 0.20,
                                       effect_magnitude = 1.0, include_missing = TRUE) {

  cat("="*80, "\n")
  cat("CREATING COMPLETE SIMULATION CLASSROOM\n")
  cat("="*80, "\n")

  # Step 1: Generate network structure
  cat("\n1. GENERATING NETWORK STRUCTURE\n")
  network_data <- generate_classroom_network(n_students, minority_prop, scenario)

  # Step 2: Generate tolerance measures
  cat("\n2. GENERATING TOLERANCE MEASURES\n")
  behavior_data <- generate_tolerance_measures(network_data$attributes, scenario)

  # Step 3: Simulate longitudinal evolution
  cat("\n3. SIMULATING LONGITUDINAL EVOLUTION\n")
  longitudinal_data <- simulate_network_evolution(
    network_data, network_data$attributes, behavior_data, n_waves
  )

  # Step 4: Apply intervention
  cat("\n4. APPLYING INTERVENTION\n")
  intervention_data <- simulate_tolerance_intervention(
    longitudinal_data, network_data$attributes, targeting,
    target_proportion, effect_magnitude
  )

  # Step 5: Apply missing data (if requested)
  if (include_missing) {
    cat("\n5. APPLYING MISSING DATA PATTERNS\n")
    final_data <- apply_missing_data_patterns(intervention_data, network_data$attributes)
  } else {
    final_data <- intervention_data
  }

  # Combine all information
  complete_dataset <- list(
    # Core data
    networks = final_data$networks,
    tolerance = final_data$tolerance,
    cooperation = final_data$cooperation,

    # Student information
    attributes = cbind(network_data$attributes, behavior_data[, -1]),  # Remove duplicate ID

    # Simulation parameters
    parameters = list(
      n_students = n_students,
      n_waves = n_waves,
      minority_prop = minority_prop,
      scenario = scenario,
      targeting = targeting,
      target_proportion = target_proportion,
      effect_magnitude = effect_magnitude,
      include_missing = include_missing
    ),

    # Results and diagnostics
    intervention_targets = final_data$intervention_targets,
    network_statistics = network_data$statistics,
    intervention_effectiveness = final_data$effectiveness,

    # Missing data information
    missing_patterns = if (include_missing) final_data$missing_patterns else NULL,
    missing_rates = if (include_missing) final_data$missing_rates else NULL,

    # Metadata
    creation_date = Sys.time(),
    description = paste("Realistic tolerance intervention simulation:",
                       scenario, "classroom with", targeting, "targeting")
  )

  cat("\n="*80, "\n")
  cat("SIMULATION COMPLETE!\n")
  cat("="*80, "\n")

  # Print summary
  print_simulation_summary(complete_dataset)

  return(complete_dataset)
}

#' Print simulation summary
#'
#' Displays key characteristics of generated simulation data
#' @param simulation_data Complete simulation dataset
print_simulation_summary <- function(simulation_data) {

  cat("\nSIMULATION SUMMARY:\n")
  cat("-"*50, "\n")

  params <- simulation_data$parameters
  cat("Design: ", params$n_students, " students, ", params$n_waves, " waves\n")
  cat("Scenario: ", params$scenario, " classroom\n")
  cat("Intervention: ", params$targeting, " targeting (",
      round(params$target_proportion*100), "% of students)\n")
  cat("Effect size: ", params$effect_magnitude, " SD units\n")

  cat("\nNetwork characteristics:\n")
  cat("- Minority proportion: ", round(params$minority_prop*100), "%\n")
  cat("- Final density: ", round(simulation_data$network_statistics$final_density, 3), "\n")
  cat("- Transitivity: ", round(simulation_data$network_statistics$transitivity, 3), "\n")

  cat("\nBehavioral measures:\n")
  final_wave <- params$n_waves
  cat("- Mean tolerance (final): ", round(mean(simulation_data$tolerance[, final_wave], na.rm = TRUE), 2), "\n")
  cat("- Mean cooperation (final): ", round(mean(simulation_data$cooperation[, final_wave], na.rm = TRUE), 2), "\n")
  cat("- Intervention net effect: ", round(simulation_data$intervention_effectiveness$net_effect, 3), "\n")

  if (params$include_missing) {
    cat("\nMissing data rates:\n")
    for (wave in 1:params$n_waves) {
      cat("- Wave ", wave, ": ", round(simulation_data$missing_rates[wave]*100, 1), "%\n")
    }
  }

  cat("\nDataset ready for RSiena analysis!\n")
}

cat("\n" * 3)
cat("="*80, "\n")
cat("MASTER DATA GENERATOR FOR TOLERANCE INTERVENTION RESEARCH\n")
cat("="*80, "\n")
cat("Empirically-grounded simulation based on German school studies\n")
cat("Ready for comprehensive testing of intervention strategies\n")
cat("="*80, "\n")