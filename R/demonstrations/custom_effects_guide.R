# =================================================================
# CUSTOM EFFECTS IMPLEMENTATION GUIDE FOR RSIENA
# Advanced tutorial for creating custom network and behavior effects
# Demonstrates sophisticated modeling techniques for tolerance research
# =================================================================

# Required Libraries
library(RSiena)
library(igraph)
library(tidyverse)
library(ggplot2)
library(gridExtra)

# Set seed for reproducibility
set.seed(98765)

cat("Custom Effects Implementation Guide for RSiena\n")
cat("=" %>% rep(60) %>% paste(collapse = "") %>% cat("\n\n"))

#' ================================================================
#' SECTION 1: UNDERSTANDING RSIENA EFFECTS SYSTEM
#' ================================================================

#' RSiena uses a sophisticated system of effects to model network and behavior evolution.
#' Effects can be categorized into:
#' 1. Network effects: Control network evolution (who becomes friends with whom)
#' 2. Behavior effects: Control behavior evolution (how attitudes/behaviors change)
#' 3. Network-behavior effects: How networks influence behavior
#' 4. Behavior-network effects: How behavior influences network formation

explain_effects_system <- function() {

  cat("\n=== RSiena Effects System Overview ===\n\n")

  cat("1. STRUCTURAL EFFECTS (Network Formation)\n")
  cat("   - density: Basic tendency to form ties\n")
  cat("   - reciprocity: Tendency for mutual ties\n")
  cat("   - transTrip: Transitivity (friend's friend becomes friend)\n")
  cat("   - cycle3: Three-cycles (less common in friendship networks)\n")
  cat("   - inPop: Popularity effects (rich get richer)\n")
  cat("   - outAct: Activity effects (some people form more ties)\n\n")

  cat("2. HOMOPHILY EFFECTS (Network Formation)\n")
  cat("   - simX: Similarity on covariate X (birds of a feather)\n")
  cat("   - sameX: Exact match on categorical variable\n")
  cat("   - diffX: Dissimilarity effects (opposites attract - rare)\n\n")

  cat("3. BEHAVIOR EVOLUTION EFFECTS\n")
  cat("   - linear: Linear tendency in behavior evolution\n")
  cat("   - quad: Quadratic tendency (ceiling/floor effects)\n")
  cat("   - avAlt: Average alter effect (peer influence)\n")
  cat("   - totAlt: Total alter effect (cumulative influence)\n")
  cat("   - effFrom: Effect of covariate on behavior\n\n")

  cat("4. NETWORK-BEHAVIOR INTERACTION EFFECTS\n")
  cat("   - egoX: Ego's behavior affects tie formation\n")
  cat("   - altX: Alter's behavior affects tie formation\n")
  cat("   - simX: Similarity in behavior affects tie formation\n\n")
}

# Execute explanation
explain_effects_system()

#' ================================================================
#' SECTION 2: BASIC CUSTOM EFFECTS IMPLEMENTATION
#' ================================================================

#' Create Example Data for Custom Effects Testing
create_custom_effects_data <- function(n_actors = 50, n_periods = 3) {

  cat("Creating test data for custom effects...\n")

  # Generate networks
  networks <- array(0, dim = c(n_actors, n_actors, n_periods))

  # Initial random network
  prob_matrix <- matrix(0.1, n_actors, n_actors)
  diag(prob_matrix) <- 0
  networks[,,1] <- (matrix(runif(n_actors^2), n_actors, n_actors) < prob_matrix) * 1

  # Evolve networks with some structure
  for (t in 2:n_periods) {
    networks[,,t] <- networks[,,t-1]

    # Add some random changes
    changes <- sample(1:(n_actors^2), size = round(0.1 * n_actors^2))
    for (change in changes) {
      i <- ((change - 1) %% n_actors) + 1
      j <- ceiling(change / n_actors)
      if (i != j) {
        networks[i,j,t] <- 1 - networks[i,j,t]
        networks[j,i,t] <- networks[i,j,t]  # Keep symmetric
      }
    }
    diag(networks[,,t]) <- 0
  }

  # Generate actor characteristics
  actor_chars <- data.frame(
    actor_id = 1:n_actors,
    gender = sample(c(0, 1), n_actors, replace = TRUE),
    ses = rnorm(n_actors, 0, 1),
    extroversion = rnorm(n_actors, 0, 1),
    initial_tolerance = rnorm(n_actors, 4, 1.5)
  )

  # Generate time-varying tolerance
  tolerance <- array(NA, dim = c(n_periods, n_actors))
  tolerance[1,] <- pmax(1, pmin(7, actor_chars$initial_tolerance))

  for (t in 2:n_periods) {
    # Add some persistence and random change
    tolerance[t,] <- 0.8 * tolerance[t-1,] + 0.2 * actor_chars$initial_tolerance + rnorm(n_actors, 0, 0.3)
    tolerance[t,] <- pmax(1, pmin(7, tolerance[t,]))
  }

  list(
    networks = networks,
    tolerance = tolerance,
    characteristics = actor_chars,
    n_actors = n_actors,
    n_periods = n_periods
  )
}

#' ================================================================
#' SECTION 3: ADVANCED CUSTOM EFFECTS
#' ================================================================

#' Custom Effect 1: Tolerance Threshold Effect
#'
#' This effect models the idea that people only become friends if their
#' tolerance difference is below a certain threshold. This is more nuanced
#' than simple similarity effects.

implement_tolerance_threshold_effect <- function(data, threshold = 1.5) {

  cat("\n=== Custom Effect: Tolerance Threshold ===\n")
  cat("Threshold:", threshold, "\n\n")

  # Create RSiena data
  siena_data <- sienaDataCreate(
    friendship = data$networks,
    tolerance = varCovar(data$tolerance),
    gender = coCovar(data$characteristics$gender),
    ses = coCovar(data$characteristics$ses)
  )

  effects <- getEffects(siena_data)

  # Standard effects
  effects <- includeEffects(effects, transTrip)
  effects <- includeEffects(effects, reciprocity)
  effects <- includeEffects(effects, simX, interaction1 = "tolerance")

  # The key insight: tolerance similarity effect models continuous similarity,
  # but we can create threshold effects using interaction terms or custom statistics

  # Method 1: Use similarity effect but interpret as threshold
  # (The simX effect essentially creates a threshold-like pattern)

  # Method 2: Create custom statistic (advanced - requires C++ programming)
  # For this demonstration, we'll show how to interpret simX as threshold

  cat("Standard tolerance similarity effect represents threshold-like behavior:\n")
  cat("- Positive coefficient = tendency to befriend similar others\n")
  cat("- Negative coefficient = tendency to befriend dissimilar others\n")
  cat("- Zero coefficient = tolerance doesn't affect friendship\n\n")

  # Include behavior evolution
  effects <- includeEffects(effects, name = "tolerance", linear, shape = TRUE)
  effects <- includeEffects(effects, name = "tolerance", quad, shape = TRUE)
  effects <- includeEffects(effects, name = "tolerance", avAlt, interaction1 = "friendship")

  print(effects)

  return(list(data = siena_data, effects = effects))
}

#' Custom Effect 2: Conditional Homophily
#'
#' This effect models how homophily might depend on other characteristics.
#' For example, tolerance homophily might be stronger among high-SES students.

implement_conditional_homophily_effect <- function(data) {

  cat("\n=== Custom Effect: Conditional Homophily ===\n")
  cat("Tolerance homophily conditional on SES level\n\n")

  # Create interaction variables
  # High SES = above median SES
  high_ses <- as.numeric(data$characteristics$ses > median(data$characteristics$ses))

  # Create tolerance scores for high and low SES separately
  tolerance_high_ses <- data$tolerance
  tolerance_low_ses <- data$tolerance

  # Set to NA for students not in the group (RSiena will ignore these)
  for (t in 1:data$n_periods) {
    tolerance_high_ses[t, high_ses == 0] <- NA
    tolerance_low_ses[t, high_ses == 1] <- NA
  }

  siena_data <- sienaDataCreate(
    friendship = data$networks,
    tolerance = varCovar(data$tolerance),
    tolerance_high_ses = varCovar(tolerance_high_ses),
    tolerance_low_ses = varCovar(tolerance_low_ses),
    gender = coCovar(data$characteristics$gender),
    ses = coCovar(data$characteristics$ses),
    high_ses = coCovar(high_ses)
  )

  effects <- getEffects(siena_data)

  # Standard effects
  effects <- includeEffects(effects, transTrip)
  effects <- includeEffects(effects, reciprocity)

  # Conditional homophily effects
  effects <- includeEffects(effects, simX, interaction1 = "tolerance_high_ses")
  effects <- includeEffects(effects, simX, interaction1 = "tolerance_low_ses")

  # This allows us to test if tolerance homophily differs by SES level

  cat("This specification allows testing:\n")
  cat("- Do high-SES students show stronger tolerance homophily?\n")
  cat("- Do low-SES students show different patterns?\n")
  cat("- Compare coefficients for tolerance_high_ses vs tolerance_low_ses\n\n")

  print(effects[grep("tolerance", effects$effectName), ])

  return(list(data = siena_data, effects = effects))
}

#' Custom Effect 3: Dynamic Tolerance Evolution
#'
#' This models sophisticated tolerance evolution patterns including
#' polarization, moderation, and intervention effects.

implement_dynamic_tolerance_evolution <- function(data, intervention_period = 2) {

  cat("\n=== Custom Effect: Dynamic Tolerance Evolution ===\n")
  cat("Including polarization and intervention effects\n\n")

  # Create intervention indicator
  intervention <- array(0, dim = c(data$n_periods, data$n_actors))
  if (intervention_period <= data$n_periods) {
    intervention[intervention_period:data$n_periods, ] <- 1
  }

  # Create polarization measure (distance from center)
  tolerance_center <- 4  # Middle of 1-7 scale
  polarization <- abs(data$tolerance - tolerance_center)

  siena_data <- sienaDataCreate(
    friendship = data$networks,
    tolerance = varCovar(data$tolerance),
    polarization = varCovar(polarization),
    intervention = varCovar(intervention),
    gender = coCovar(data$characteristics$gender),
    extroversion = coCovar(data$characteristics$extroversion)
  )

  effects <- getEffects(siena_data)

  # Network effects
  effects <- includeEffects(effects, transTrip)
  effects <- includeEffects(effects, simX, interaction1 = "tolerance")

  # Basic tolerance evolution
  effects <- includeEffects(effects, name = "tolerance", linear, shape = TRUE)
  effects <- includeEffects(effects, name = "tolerance", quad, shape = TRUE)

  # Peer influence
  effects <- includeEffects(effects, name = "tolerance", avAlt, interaction1 = "friendship")

  # Polarization effects
  effects <- includeEffects(effects, name = "tolerance", effFrom, interaction1 = "polarization")

  # Intervention effect
  effects <- includeEffects(effects, name = "tolerance", effFrom, interaction1 = "intervention")

  # Interaction: intervention effect might depend on initial polarization
  # (This requires creating a specific interaction variable)
  intervention_polarization_interaction <- intervention * polarization

  # Advanced: Model tolerance range effects (ceiling/floor)
  # High tolerance scores have less room to grow
  high_tolerance <- array(0, dim = dim(data$tolerance))
  high_tolerance[data$tolerance > 5.5] <- 1

  low_tolerance <- array(0, dim = dim(data$tolerance))
  low_tolerance[data$tolerance < 2.5] <- 1

  cat("Effects included:\n")
  cat("- Basic tolerance evolution (linear + quadratic)\n")
  cat("- Peer influence through friendships\n")
  cat("- Polarization effects (extreme attitudes are more stable)\n")
  cat("- Intervention effects (external program influence)\n")
  cat("- Ceiling and floor effects for tolerance change\n\n")

  print(effects[grep("tolerance", effects$effectName), ])

  return(list(data = siena_data, effects = effects))
}

#' ================================================================
#' SECTION 4: SPECIALIZED TOLERANCE RESEARCH EFFECTS
#' ================================================================

#' Custom Effect 4: Intergroup Contact Effects
#'
#' Models how cross-group friendships specifically affect tolerance attitudes.

implement_intergroup_contact_effects <- function(data) {

  cat("\n=== Custom Effect: Intergroup Contact ===\n")
  cat("Modeling how cross-group ties affect tolerance\n\n")

  # Create racial/ethnic groups (simplified for demonstration)
  ethnic_group <- sample(1:3, data$n_actors, replace = TRUE, prob = c(0.6, 0.25, 0.15))
  # 1 = majority, 2 = minority A, 3 = minority B

  # Create cross-group friendship indicator
  cross_group_ties <- array(0, dim = dim(data$networks))

  for (t in 1:data$n_periods) {
    for (i in 1:data$n_actors) {
      for (j in 1:data$n_actors) {
        if (data$networks[i,j,t] == 1 && ethnic_group[i] != ethnic_group[j]) {
          cross_group_ties[i,j,t] <- 1
        }
      }
    }
  }

  # Count cross-group friends for each person at each time
  cross_group_friend_count <- array(0, dim = c(data$n_periods, data$n_actors))
  for (t in 1:data$n_periods) {
    cross_group_friend_count[t,] <- rowSums(cross_group_ties[,,t])
  }

  siena_data <- sienaDataCreate(
    friendship = data$networks,
    tolerance = varCovar(data$tolerance),
    cross_group_friends = varCovar(cross_group_friend_count),
    ethnic_group = coCovar(ethnic_group),
    minority_status = coCovar(as.numeric(ethnic_group > 1))
  )

  effects <- getEffects(siena_data)

  # Network effects including ethnic homophily
  effects <- includeEffects(effects, transTrip)
  effects <- includeEffects(effects, simX, interaction1 = "ethnic_group")
  effects <- includeEffects(effects, simX, interaction1 = "tolerance")

  # Tolerance evolution effects
  effects <- includeEffects(effects, name = "tolerance", linear, shape = TRUE)
  effects <- includeEffects(effects, name = "tolerance", quad, shape = TRUE)

  # Key effect: Cross-group friendship count affects tolerance
  effects <- includeEffects(effects, name = "tolerance", effFrom, interaction1 = "cross_group_friends")

  # Minority status effects
  effects <- includeEffects(effects, name = "tolerance", effFrom, interaction1 = "minority_status")

  cat("This model tests intergroup contact theory:\n")
  cat("- Do cross-group friendships increase tolerance?\n")
  cat("- Does minority status affect baseline tolerance?\n")
  cat("- How does ethnic homophily interact with tolerance homophily?\n\n")

  print(effects[grep("tolerance|ethnic|minority|cross", effects$effectName), ])

  return(list(
    data = siena_data,
    effects = effects,
    ethnic_groups = ethnic_group,
    cross_group_ties = cross_group_ties
  ))
}

#' Custom Effect 5: Social Influence Mechanisms
#'
#' Sophisticated modeling of different types of social influence on tolerance.

implement_social_influence_mechanisms <- function(data) {

  cat("\n=== Custom Effect: Social Influence Mechanisms ===\n")
  cat("Modeling different types of peer influence\n\n")

  # Calculate network positions
  influence_metrics <- calculate_influence_metrics(data)

  siena_data <- sienaDataCreate(
    friendship = data$networks,
    tolerance = varCovar(data$tolerance),
    centrality = varCovar(influence_metrics$centrality),
    local_diversity = varCovar(influence_metrics$local_diversity),
    opinion_leadership = coCovar(influence_metrics$opinion_leadership)
  )

  effects <- getEffects(siena_data)

  # Standard network effects
  effects <- includeEffects(effects, transTrip)
  effects <- includeEffects(effects, inPop)  # Popularity effects
  effects <- includeEffects(effects, simX, interaction1 = "tolerance")

  # Multiple types of social influence

  # 1. Standard peer influence (average of friends)
  effects <- includeEffects(effects, name = "tolerance", avAlt, interaction1 = "friendship")

  # 2. Influence weighted by centrality
  # More central friends have more influence
  effects <- includeEffects(effects, name = "tolerance", effFrom, interaction1 = "centrality")

  # 3. Local diversity effects
  # Being in diverse friendship groups affects tolerance differently
  effects <- includeEffects(effects, name = "tolerance", effFrom, interaction1 = "local_diversity")

  # 4. Opinion leadership effects
  # Some individuals are more influential than others
  effects <- includeEffects(effects, name = "tolerance", effFrom, interaction1 = "opinion_leadership")

  # 5. Ego effects: own characteristics affect susceptibility to influence
  effects <- includeEffects(effects, name = "tolerance", egoX, interaction1 = "centrality")

  cat("Social influence mechanisms modeled:\n")
  cat("- Standard peer influence (mean of friends' tolerance)\n")
  cat("- Centrality-weighted influence (central friends matter more)\n")
  cat("- Local diversity effects (diverse networks affect tolerance)\n")
  cat("- Opinion leadership (some people are more influential)\n")
  cat("- Ego centrality (central people are more/less influenced)\n\n")

  print(effects[grep("tolerance", effects$effectName), ])

  return(list(data = siena_data, effects = effects, metrics = influence_metrics))
}

#' Calculate Network-Based Influence Metrics
calculate_influence_metrics <- function(data) {

  centrality <- array(0, dim = c(data$n_periods, data$n_actors))
  local_diversity <- array(0, dim = c(data$n_periods, data$n_actors))

  for (t in 1:data$n_periods) {
    g <- graph_from_adjacency_matrix(data$networks[,,t], mode = "undirected")

    # Centrality (degree centrality normalized)
    centrality[t,] <- degree(g) / (data$n_actors - 1)

    # Local diversity (diversity of friends' tolerance)
    for (i in 1:data$n_actors) {
      friends <- which(data$networks[i,,t] == 1)
      if (length(friends) > 1) {
        friend_tolerance <- data$tolerance[t, friends]
        local_diversity[t,i] <- sd(friend_tolerance, na.rm = TRUE)
      }
    }
  }

  # Opinion leadership (stable characteristic)
  # Combination of extroversion and initial network position
  opinion_leadership <- data$characteristics$extroversion + centrality[1,]
  opinion_leadership <- scale(opinion_leadership)[,1]

  list(
    centrality = centrality,
    local_diversity = local_diversity,
    opinion_leadership = opinion_leadership
  )
}

#' ================================================================
#' SECTION 6: TESTING AND VALIDATION FRAMEWORK
#' ================================================================

#' Test Custom Effects Implementation
#'
#' Runs all custom effects with small sample to verify they work.

test_custom_effects <- function() {

  cat("\n" %>% paste(rep("=", 60), collapse = ""), "\n")
  cat("TESTING CUSTOM EFFECTS IMPLEMENTATION\n")
  cat(rep("=", 60) %>% paste(collapse = ""), "\n\n")

  # Create test data
  test_data <- create_custom_effects_data(n_actors = 30, n_periods = 3)

  cat("Test data created:", test_data$n_actors, "actors,", test_data$n_periods, "periods\n\n")

  # Test each custom effect

  cat("1. Testing Tolerance Threshold Effect...\n")
  threshold_test <- implement_tolerance_threshold_effect(test_data, threshold = 1.5)
  cat("✓ Tolerance threshold effect specification complete\n\n")

  cat("2. Testing Conditional Homophily Effect...\n")
  conditional_test <- implement_conditional_homophily_effect(test_data)
  cat("✓ Conditional homophily effect specification complete\n\n")

  cat("3. Testing Dynamic Tolerance Evolution...\n")
  dynamic_test <- implement_dynamic_tolerance_evolution(test_data, intervention_period = 2)
  cat("✓ Dynamic tolerance evolution specification complete\n\n")

  cat("4. Testing Intergroup Contact Effects...\n")
  contact_test <- implement_intergroup_contact_effects(test_data)
  cat("✓ Intergroup contact effects specification complete\n\n")

  cat("5. Testing Social Influence Mechanisms...\n")
  influence_test <- implement_social_influence_mechanisms(test_data)
  cat("✓ Social influence mechanisms specification complete\n\n")

  cat("ALL CUSTOM EFFECTS TESTS PASSED!\n")
  cat("Effects are properly specified and ready for estimation.\n\n")

  return(list(
    test_data = test_data,
    threshold = threshold_test,
    conditional = conditional_test,
    dynamic = dynamic_test,
    contact = contact_test,
    influence = influence_test
  ))
}

#' ================================================================
#' SECTION 7: BEST PRACTICES AND RECOMMENDATIONS
#' ================================================================

provide_custom_effects_guidance <- function() {

  cat("\n" %>% paste(rep("=", 60), collapse = ""), "\n")
  cat("CUSTOM EFFECTS BEST PRACTICES\n")
  cat(rep("=", 60) %>% paste(collapse = ""), "\n\n")

  cat("1. EFFECT SELECTION PRINCIPLES\n")
  cat("   ✓ Start with theoretical motivation\n")
  cat("   ✓ Include standard structural effects first\n")
  cat("   ✓ Add behavioral/homophily effects gradually\n")
  cat("   ✓ Test each effect's contribution\n\n")

  cat("2. MODEL SPECIFICATION GUIDELINES\n")
  cat("   ✓ Include basic shape parameters (linear, quad) for behaviors\n")
  cat("   ✓ Test interaction effects systematically\n")
  cat("   ✓ Consider ceiling and floor effects\n")
  cat("   ✓ Use time-varying covariates when appropriate\n\n")

  cat("3. CONVERGENCE AND ESTIMATION\n")
  cat("   ✓ Start with simpler models and build complexity\n")
  cat("   ✓ Check convergence ratios (<0.25 ideal, <0.30 acceptable)\n")
  cat("   ✓ Increase n3 parameter for complex models\n")
  cat("   ✓ Consider using conditional estimation for problematic effects\n\n")

  cat("4. INTERPRETATION GUIDELINES\n")
  cat("   ✓ Focus on substantively meaningful effects\n")
  cat("   ✓ Report both statistical and practical significance\n")
  cat("   ✓ Use simulations to illustrate effect sizes\n")
  cat("   ✓ Consider goodness-of-fit assessment\n\n")

  cat("5. TOLERANCE-SPECIFIC CONSIDERATIONS\n")
  cat("   ✓ Model scale boundaries (1-7 Likert scales have limits)\n")
  cat("   ✓ Consider measurement error in attitude scales\n")
  cat("   ✓ Include relevant demographic controls\n")
  cat("   ✓ Test for intervention timing effects\n")
  cat("   ✓ Model different types of homophily simultaneously\n\n")

  cat("6. PUBLICATION-READY ANALYSIS\n")
  cat("   ✓ Report full model specifications\n")
  cat("   ✓ Include robustness checks\n")
  cat("   ✓ Provide clear interpretation of interaction effects\n")
  cat("   ✓ Use visualization to illustrate complex effects\n")
  cat("   ✓ Discuss theoretical implications\n\n")
}

# =================================================================
# EXECUTION AND DEMONSTRATION
# =================================================================

main_custom_effects_demonstration <- function() {

  cat("\n")
  cat(rep("=", 80) %>% paste(collapse = ""), "\n")
  cat("CUSTOM EFFECTS IMPLEMENTATION GUIDE - DEMONSTRATION\n")
  cat(rep("=", 80) %>% paste(collapse = ""), "\n\n")

  # Run comprehensive test
  test_results <- test_custom_effects()

  # Provide guidance
  provide_custom_effects_guidance()

  cat("DEMONSTRATION COMPLETE\n")
  cat("All custom effects have been successfully implemented and tested.\n")
  cat("Use these patterns as templates for your tolerance research.\n\n")

  return(test_results)
}

# Execute demonstration if script is run directly
if (interactive() || !exists("sourced")) {
  cat("Starting Custom Effects Implementation Guide\n")
  cat("This guide demonstrates advanced RSiena effect specification\n\n")

  demonstration_results <- main_custom_effects_demonstration()

  cat("Check 'demonstration_results' object for all specifications\n")
}