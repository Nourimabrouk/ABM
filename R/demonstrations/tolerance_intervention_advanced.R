# =================================================================
# ADVANCED TOLERANCE INTERVENTION ANALYSIS WITH RSIENA
# Complete workflow for modeling tolerance dynamics in social networks
# Demonstrates custom effects, intervention simulation, and publication-ready analysis
# =================================================================

# Required Libraries
library(RSiena)
library(igraph)
library(tidyverse)
library(ggplot2)
library(gridExtra)
library(knitr)
library(kableExtra)
library(corrplot)
library(plotly)
library(viridis)

# Set up for reproducible analysis
set.seed(12345)
options(digits = 4)

#' Advanced Tolerance Intervention Data Generator
#'
#' Creates realistic longitudinal network and tolerance data for intervention studies.
#' Models the complex dynamics of tolerance attitudes in social networks with
#' interventions applied at specific time points.
#'
#' @param n_students Number of students in the network
#' @param n_periods Number of observation periods
#' @param intervention_period When the tolerance intervention occurs (0 = no intervention)
#' @param network_type Type of initial network ("random", "clustered", "school_realistic")
#' @param tolerance_homophily Strength of tolerance-based friendship formation
#' @param intervention_effect_size Effect size of the tolerance intervention
#' @param measurement_error Standard deviation of measurement error in tolerance scales
#' @return List containing networks, tolerance attitudes, covariates, and intervention indicators
create_tolerance_intervention_data <- function(
  n_students = 150,
  n_periods = 4,
  intervention_period = 2,
  network_type = "school_realistic",
  tolerance_homophily = 0.8,
  intervention_effect_size = 0.5,
  measurement_error = 0.3
) {

  cat("Generating Advanced Tolerance Intervention Data\n")
  cat("=" %>% rep(50) %>% paste(collapse = "") %>% cat("\n"))
  cat("Students:", n_students, "| Periods:", n_periods, "\n")
  cat("Intervention at period:", intervention_period, "\n")
  cat("Network type:", network_type, "\n\n")

  # Initialize arrays
  networks <- array(0, dim = c(n_students, n_students, n_periods))
  tolerance_scores <- array(NA, dim = c(n_periods, n_students))

  # Generate stable student characteristics
  student_characteristics <- generate_student_characteristics(n_students)

  # Generate initial tolerance attitudes (influenced by characteristics)
  initial_tolerance <- generate_initial_tolerance(student_characteristics, n_students)
  tolerance_scores[1, ] <- initial_tolerance + rnorm(n_students, 0, measurement_error)

  # Generate initial network based on type
  networks[, , 1] <- generate_initial_network(
    n_students,
    network_type,
    student_characteristics,
    initial_tolerance
  )

  cat("Period 1 - Network density:", round(sum(networks[,,1]) / (n_students * (n_students - 1)), 3), "\n")
  cat("Period 1 - Tolerance mean:", round(mean(tolerance_scores[1,], na.rm = TRUE), 3), "\n")

  # Evolve network and tolerance over time
  for (period in 2:n_periods) {

    # Apply intervention effect if this is the intervention period
    if (period == intervention_period && intervention_period > 0) {
      cat("\n>>> APPLYING TOLERANCE INTERVENTION <<<\n")
      intervention_effect <- apply_tolerance_intervention(
        tolerance_scores[period - 1, ],
        student_characteristics,
        intervention_effect_size
      )
      tolerance_scores[period, ] <- intervention_effect + rnorm(n_students, 0, measurement_error)
    } else {
      # Natural evolution of tolerance attitudes
      tolerance_scores[period, ] <- evolve_tolerance_attitudes(
        tolerance_scores[period - 1, ],
        networks[, , period - 1],
        student_characteristics
      ) + rnorm(n_students, 0, measurement_error)
    }

    # Evolve network based on tolerance homophily and structural effects
    networks[, , period] <- evolve_network(
      networks[, , period - 1],
      tolerance_scores[period, ],
      student_characteristics,
      tolerance_homophily
    )

    cat("Period", period, "- Network density:",
        round(sum(networks[,,period]) / (n_students * (n_students - 1)), 3), "\n")
    cat("Period", period, "- Tolerance mean:",
        round(mean(tolerance_scores[period,], na.rm = TRUE), 3), "\n")
  }

  # Create intervention indicator
  intervention_indicator <- rep(0, n_periods)
  if (intervention_period > 0 && intervention_period <= n_periods) {
    intervention_indicator[intervention_period:n_periods] <- 1
  }

  # Package results
  result <- list(
    networks = networks,
    tolerance = tolerance_scores,
    student_characteristics = student_characteristics,
    intervention_indicator = intervention_indicator,
    intervention_period = intervention_period,
    n_students = n_students,
    n_periods = n_periods,
    metadata = list(
      network_type = network_type,
      tolerance_homophily = tolerance_homophily,
      intervention_effect_size = intervention_effect_size,
      measurement_error = measurement_error
    )
  )

  cat("\nData generation complete!\n\n")
  return(result)
}

#' Generate Realistic Student Characteristics
#'
#' Creates diverse student characteristics that influence both network formation
#' and tolerance attitudes, mimicking real school demographics.
generate_student_characteristics <- function(n_students) {

  # Demographics with realistic correlations
  grade_level <- sample(9:12, n_students, replace = TRUE, prob = c(0.28, 0.26, 0.24, 0.22))

  # Gender with slight correlation to tolerance openness
  gender <- sample(c("Male", "Female", "Other"), n_students, replace = TRUE,
                   prob = c(0.48, 0.48, 0.04))

  # Ethnicity (important for tolerance dynamics)
  ethnicity <- sample(c("White", "Hispanic", "Black", "Asian", "Other"),
                      n_students, replace = TRUE,
                      prob = c(0.45, 0.25, 0.15, 0.10, 0.05))

  # Socioeconomic status (standardized)
  ses <- rnorm(n_students, 0, 1)

  # Academic performance (correlated with SES)
  academic_performance <- 0.4 * scale(ses)[,1] + rnorm(n_students, 0, 0.8)

  # Personality traits affecting tolerance
  openness_to_experience <- rnorm(n_students, 0, 1)
  social_dominance_orientation <- rnorm(n_students, 0, 1)

  # Previous exposure to diversity (influences initial tolerance)
  diversity_exposure <- rbinom(n_students, 1, 0.3) +
                       0.3 * (ethnicity != "White") +
                       0.2 * pnorm(ses)

  data.frame(
    student_id = 1:n_students,
    grade_level = grade_level,
    gender = factor(gender),
    ethnicity = factor(ethnicity),
    ses = scale(ses)[,1],
    academic_performance = scale(academic_performance)[,1],
    openness = scale(openness_to_experience)[,1],
    social_dominance = scale(social_dominance_orientation)[,1],
    diversity_exposure = scale(diversity_exposure)[,1]
  )
}

#' Generate Initial Tolerance Attitudes
#'
#' Creates realistic distribution of tolerance attitudes based on student characteristics.
generate_initial_tolerance <- function(characteristics, n_students) {

  # Tolerance is influenced by multiple factors
  tolerance <- with(characteristics, {
    0.3 * openness +                    # Open people are more tolerant
    -0.25 * social_dominance +          # SDO reduces tolerance
    0.2 * diversity_exposure +          # Exposure increases tolerance
    0.15 * (ethnicity != "White") +     # Minority status increases tolerance
    0.1 * (gender == "Female") +        # Slight gender effect
    rnorm(n_students, 0, 0.5)          # Random component
  })

  # Scale to realistic range (1-7 Likert scale, centered at 4)
  tolerance_scaled <- 4 + 1.5 * scale(tolerance)[,1]

  # Ensure realistic bounds
  pmax(1, pmin(7, tolerance_scaled))
}

#' Generate Initial Network Structure
#'
#' Creates realistic school friendship networks with homophily patterns.
generate_initial_network <- function(n_students, network_type, characteristics, tolerance) {

  network <- matrix(0, n_students, n_students)

  if (network_type == "random") {
    # Simple random network
    prob_matrix <- matrix(0.08, n_students, n_students)
    network <- (matrix(runif(n_students^2), n_students, n_students) < prob_matrix) * 1

  } else if (network_type == "clustered") {
    # Grade-based clustering
    for (grade in unique(characteristics$grade_level)) {
      grade_students <- which(characteristics$grade_level == grade)
      for (i in grade_students) {
        for (j in grade_students) {
          if (i != j && runif(1) < 0.15) {
            network[i, j] <- 1
          }
        }
      }
    }

  } else if (network_type == "school_realistic") {
    # Realistic school network with multiple homophily patterns

    for (i in 1:(n_students-1)) {
      for (j in (i+1):n_students) {

        # Base friendship probability
        base_prob <- 0.02

        # Same grade bonus
        if (characteristics$grade_level[i] == characteristics$grade_level[j]) {
          base_prob <- base_prob + 0.06
        }

        # Gender homophily
        if (characteristics$gender[i] == characteristics$gender[j]) {
          base_prob <- base_prob + 0.03
        }

        # Ethnicity homophily
        if (characteristics$ethnicity[i] == characteristics$ethnicity[j]) {
          base_prob <- base_prob + 0.04
        }

        # SES similarity
        ses_similarity <- 1 - abs(characteristics$ses[i] - characteristics$ses[j]) / 4
        base_prob <- base_prob + 0.02 * ses_similarity

        # Academic performance similarity
        acad_similarity <- 1 - abs(characteristics$academic_performance[i] - characteristics$academic_performance[j]) / 4
        base_prob <- base_prob + 0.02 * acad_similarity

        # Tolerance similarity (weak initial effect)
        tolerance_similarity <- 1 - abs(tolerance[i] - tolerance[j]) / 6
        base_prob <- base_prob + 0.01 * tolerance_similarity

        # Create symmetric friendship
        if (runif(1) < base_prob) {
          network[i, j] <- network[j, i] <- 1
        }
      }
    }
  }

  # Remove self-loops
  diag(network) <- 0

  return(network)
}

#' Apply Tolerance Intervention
#'
#' Models the effect of a tolerance intervention program on student attitudes.
apply_tolerance_intervention <- function(current_tolerance, characteristics, effect_size) {

  # Intervention effectiveness varies by student characteristics
  intervention_susceptibility <- with(characteristics, {
    0.8 +                               # Base susceptibility
    0.3 * openness +                    # Open students more receptive
    -0.2 * social_dominance +           # SDO reduces receptivity
    0.1 * (grade_level - 9) / 3 +       # Older students slightly more receptive
    rnorm(length(openness), 0, 0.2)    # Random variation
  })

  # Ensure positive susceptibility
  intervention_susceptibility <- pmax(0.1, intervention_susceptibility)

  # Apply intervention effect
  new_tolerance <- current_tolerance +
                   effect_size * intervention_susceptibility *
                   (7 - current_tolerance) / 6  # Ceiling effect

  # Ensure bounds
  pmax(1, pmin(7, new_tolerance))
}

#' Evolve Tolerance Attitudes Over Time
#'
#' Models natural evolution of tolerance through peer influence and social learning.
evolve_tolerance_attitudes <- function(previous_tolerance, network, characteristics) {

  n_students <- length(previous_tolerance)
  new_tolerance <- previous_tolerance

  # Peer influence through network
  for (i in 1:n_students) {
    friends <- which(network[i, ] == 1)

    if (length(friends) > 0) {
      # Average tolerance of friends
      friend_tolerance_mean <- mean(previous_tolerance[friends])

      # Influence strength based on characteristics
      influence_receptivity <- 0.1 + 0.05 * characteristics$openness[i] + rnorm(1, 0, 0.02)

      # Update tolerance (towards friends' mean)
      new_tolerance[i] <- previous_tolerance[i] +
                          influence_receptivity * (friend_tolerance_mean - previous_tolerance[i])
    }

    # Add random drift
    new_tolerance[i] <- new_tolerance[i] + rnorm(1, 0, 0.05)
  }

  # Ensure bounds
  pmax(1, pmin(7, new_tolerance))
}

#' Evolve Network Structure
#'
#' Models network evolution with tolerance homophily and structural effects.
evolve_network <- function(previous_network, current_tolerance, characteristics, tolerance_homophily) {

  n_students <- nrow(previous_network)
  new_network <- previous_network

  # Calculate change probabilities for each dyad
  for (i in 1:(n_students-1)) {
    for (j in (i+1):n_students) {

      current_tie <- previous_network[i, j]

      if (current_tie == 0) {
        # Probability of forming new tie

        # Base probability
        form_prob <- 0.01

        # Tolerance similarity effect
        tolerance_similarity <- 1 - abs(current_tolerance[i] - current_tolerance[j]) / 6
        form_prob <- form_prob + tolerance_homophily * 0.03 * tolerance_similarity

        # Other homophily effects (weaker than tolerance)
        if (characteristics$grade_level[i] == characteristics$grade_level[j]) {
          form_prob <- form_prob + 0.02
        }

        # Structural effects: mutual friends
        mutual_friends <- sum(previous_network[i, ] * previous_network[j, ])
        form_prob <- form_prob + 0.005 * mutual_friends

        # Form tie?
        if (runif(1) < form_prob) {
          new_network[i, j] <- new_network[j, i] <- 1
        }

      } else {
        # Probability of dissolving existing tie

        # Base dissolution probability
        dissolve_prob <- 0.05

        # Tolerance dissimilarity increases dissolution
        tolerance_dissimilarity <- abs(current_tolerance[i] - current_tolerance[j]) / 6
        dissolve_prob <- dissolve_prob + tolerance_homophily * 0.02 * tolerance_dissimilarity

        # Dissolve tie?
        if (runif(1) < dissolve_prob) {
          new_network[i, j] <- new_network[j, i] <- 0
        }
      }
    }
  }

  return(new_network)
}

#' Run Comprehensive RSiena Analysis for Tolerance Intervention
#'
#' Conducts full RSiena analysis with custom effects for tolerance dynamics.
run_tolerance_rsiena_analysis <- function(data, include_intervention = TRUE) {

  cat("\n" %>% paste(rep("=", 60), collapse = ""), "\n")
  cat("COMPREHENSIVE RSiena ANALYSIS: TOLERANCE INTERVENTION\n")
  cat(rep("=", 60) %>% paste(collapse = ""), "\n\n")

  # Prepare data for RSiena
  cat("1. Preparing RSiena data objects...\n")

  # Convert networks to RSiena format
  friendship_networks <- data$networks

  # Prepare covariates
  grade_level <- coCovar(data$student_characteristics$grade_level)
  gender_numeric <- as.numeric(data$student_characteristics$gender == "Female")
  gender <- coCovar(gender_numeric)
  ethnicity_minority <- as.numeric(data$student_characteristics$ethnicity != "White")
  minority_status <- coCovar(ethnicity_minority)
  ses <- coCovar(data$student_characteristics$ses)

  # Time-varying tolerance
  tolerance <- varCovar(data$tolerance)

  # Create intervention indicator if requested
  if (include_intervention && data$intervention_period > 0) {
    intervention <- varCovar(matrix(rep(data$intervention_indicator, data$n_students),
                                   data$n_periods, data$n_students))
  }

  # Create RSiena data object
  if (include_intervention && data$intervention_period > 0) {
    siena_data <- sienaDataCreate(
      friendship = friendship_networks,
      tolerance = tolerance,
      grade = grade_level,
      female = gender,
      minority = minority_status,
      ses = ses,
      intervention = intervention
    )
  } else {
    siena_data <- sienaDataCreate(
      friendship = friendship_networks,
      tolerance = tolerance,
      grade = grade_level,
      female = gender,
      minority = minority_status,
      ses = ses
    )
  }

  print(siena_data)

  # 2. Specify model effects
  cat("\n2. Specifying model effects...\n")

  effects <- getEffects(siena_data)

  # Network evolution effects
  effects <- includeEffects(effects, transTrip)      # Transitivity
  effects <- includeEffects(effects, cycle3)         # 3-cycles
  effects <- includeEffects(effects, inPop)          # Popularity (indegree)
  effects <- includeEffects(effects, outAct)         # Activity (outdegree)

  # Homophily effects
  effects <- includeEffects(effects, simX, interaction1 = "grade")
  effects <- includeEffects(effects, simX, interaction1 = "female")
  effects <- includeEffects(effects, simX, interaction1 = "minority")
  effects <- includeEffects(effects, simX, interaction1 = "ses")
  effects <- includeEffects(effects, simX, interaction1 = "tolerance")  # Key effect!

  # Tolerance behavior evolution effects
  effects <- includeEffects(effects, name = "tolerance", linear, shape = TRUE)
  effects <- includeEffects(effects, name = "tolerance", quad, shape = TRUE)
  effects <- includeEffects(effects, name = "tolerance", avAlt, interaction1 = "friendship")  # Peer influence

  # Covariate effects on tolerance
  effects <- includeEffects(effects, name = "tolerance", effFrom, interaction1 = "grade")
  effects <- includeEffects(effects, name = "tolerance", effFrom, interaction1 = "female")
  effects <- includeEffects(effects, name = "tolerance", effFrom, interaction1 = "minority")
  effects <- includeEffects(effects, name = "tolerance", effFrom, interaction1 = "ses")

  # Intervention effect (if included)
  if (include_intervention && data$intervention_period > 0) {
    effects <- includeEffects(effects, name = "tolerance", effFrom, interaction1 = "intervention")
  }

  print(effects)

  # 3. Set algorithm parameters
  cat("\n3. Setting algorithm parameters...\n")

  algorithm <- sienaAlgorithmCreate(
    projname = "ToleranceIntervention",
    cond = FALSE,
    useStdInits = FALSE,
    nsub = 4,
    n3 = 2000,  # Increased for better precision
    simOnly = FALSE,
    seed = 54321,
    diagonalize = 0.2
  )

  # 4. Run the analysis
  cat("\n4. Running RSiena estimation...\n")
  cat("This may take 10-15 minutes for comprehensive analysis...\n\n")

  results <- tryCatch({
    siena07(algorithm, data = siena_data, effects = effects, verbose = TRUE, returnDeps = TRUE)
  }, error = function(e) {
    cat("Estimation error:", e$message, "\n")
    return(NULL)
  })

  # 5. Analyze results
  if (!is.null(results)) {

    cat("\n" %>% paste(rep("=", 50), collapse = ""), "\n")
    cat("ANALYSIS RESULTS\n")
    cat(rep("=", 50) %>% paste(collapse = ""), "\n\n")

    # Check convergence
    convergence_ok <- results$OK
    max_convergence_ratio <- max(abs(results$tconv.max))

    cat("Convergence Status:", ifelse(convergence_ok, "✓ CONVERGED", "⚠ NOT CONVERGED"), "\n")
    cat("Maximum convergence ratio:", round(max_convergence_ratio, 4), "\n")

    if (max_convergence_ratio < 0.25) {
      cat("✓ Excellent convergence (< 0.25)\n")
    } else if (max_convergence_ratio < 0.30) {
      cat("✓ Good convergence (< 0.30)\n")
    } else {
      cat("⚠ Marginal convergence (≥ 0.30) - consider re-estimation\n")
    }

    # Extract and display key results
    theta <- results$theta
    se <- results$se
    t_ratios <- theta / se
    p_values <- 2 * (1 - pnorm(abs(t_ratios)))

    # Create results table
    results_table <- data.frame(
      Effect = effects$effectName,
      Estimate = round(theta, 3),
      SE = round(se, 3),
      t_ratio = round(t_ratios, 3),
      p_value = round(p_values, 3),
      Significant = ifelse(p_values < 0.05, "***",
                          ifelse(p_values < 0.10, "**",
                                ifelse(p_values < 0.15, "*", "")))
    )

    cat("\n=== PARAMETER ESTIMATES ===\n")
    print(results_table)

    # Highlight key findings
    cat("\n=== KEY FINDINGS ===\n")

    # Find key effects
    tolerance_homophily_idx <- grep("tolerance similarity", effects$effectName)
    peer_influence_idx <- grep("tolerance average alter", effects$effectName)
    if (include_intervention) {
      intervention_effect_idx <- grep("tolerance: intervention", effects$effectName)
    }

    if (length(tolerance_homophily_idx) > 0) {
      cat("Tolerance Homophily:", round(theta[tolerance_homophily_idx], 3),
          "(SE =", round(se[tolerance_homophily_idx], 3), ")\n")
      if (p_values[tolerance_homophily_idx] < 0.05) {
        cat("  → Significant tolerance-based friendship selection\n")
      }
    }

    if (length(peer_influence_idx) > 0) {
      cat("Peer Influence on Tolerance:", round(theta[peer_influence_idx], 3),
          "(SE =", round(se[peer_influence_idx], 3), ")\n")
      if (p_values[peer_influence_idx] < 0.05) {
        cat("  → Significant peer influence on tolerance attitudes\n")
      }
    }

    if (include_intervention && length(intervention_effect_idx) > 0) {
      cat("Intervention Effect:", round(theta[intervention_effect_idx], 3),
          "(SE =", round(se[intervention_effect_idx], 3), ")\n")
      if (p_values[intervention_effect_idx] < 0.05) {
        cat("  → Significant intervention effect on tolerance\n")
      }
    }

    return(list(
      data = siena_data,
      effects = effects,
      algorithm = algorithm,
      results = results,
      results_table = results_table,
      convergence_ok = convergence_ok,
      max_convergence_ratio = max_convergence_ratio
    ))

  } else {
    cat("\n❌ Analysis failed\n")
    return(list(
      data = siena_data,
      effects = effects,
      algorithm = algorithm,
      results = NULL
    ))
  }
}

#' Create Publication-Quality Visualizations
#'
#' Generates comprehensive visualizations for the tolerance intervention analysis.
create_tolerance_visualizations <- function(data, analysis_results = NULL) {

  cat("\nCreating publication-quality visualizations...\n")

  # Set theme for all plots
  theme_publication <- theme_minimal() +
    theme(
      text = element_text(size = 12, family = "Arial"),
      axis.title = element_text(size = 14),
      axis.text = element_text(size = 10),
      legend.title = element_text(size = 12),
      legend.text = element_text(size = 10),
      strip.text = element_text(size = 12, face = "bold"),
      plot.title = element_text(size = 16, face = "bold", hjust = 0.5),
      plot.subtitle = element_text(size = 12, hjust = 0.5),
      panel.grid.minor = element_blank()
    )

  # 1. Network Evolution Over Time
  network_stats <- calculate_network_evolution_stats(data)

  p1 <- ggplot(network_stats, aes(x = period)) +
    geom_line(aes(y = density, color = "Density"), size = 1.2) +
    geom_line(aes(y = transitivity, color = "Transitivity"), size = 1.2) +
    geom_point(aes(y = density, color = "Density"), size = 3) +
    geom_point(aes(y = transitivity, color = "Transitivity"), size = 3) +
    geom_vline(xintercept = data$intervention_period, linetype = "dashed",
               color = "red", alpha = 0.7, size = 1) +
    annotate("text", x = data$intervention_period + 0.1, y = 0.9,
             label = "Intervention", angle = 90, color = "red") +
    scale_color_viridis_d(name = "Network Measure") +
    labs(
      title = "Network Structure Evolution",
      subtitle = "Changes in friendship network properties over time",
      x = "Time Period",
      y = "Network Measure"
    ) +
    theme_publication

  # 2. Tolerance Distribution Over Time
  tolerance_long <- data$tolerance %>%
    as.data.frame() %>%
    rownames_to_column("period") %>%
    pivot_longer(-period, names_to = "student", values_to = "tolerance") %>%
    mutate(period = as.numeric(period))

  p2 <- ggplot(tolerance_long, aes(x = period, y = tolerance)) +
    geom_violin(aes(group = period), alpha = 0.7, fill = "lightblue") +
    geom_boxplot(aes(group = period), width = 0.1, alpha = 0.8) +
    stat_summary(fun = mean, geom = "point", size = 3, color = "red") +
    stat_summary(fun = mean, geom = "line", group = 1, color = "red", size = 1.2) +
    geom_vline(xintercept = data$intervention_period, linetype = "dashed",
               color = "red", alpha = 0.7, size = 1) +
    annotate("text", x = data$intervention_period + 0.1, y = 6.5,
             label = "Intervention", angle = 90, color = "red") +
    labs(
      title = "Tolerance Attitude Distribution",
      subtitle = "Distribution and mean tolerance scores over time",
      x = "Time Period",
      y = "Tolerance Score (1-7 scale)"
    ) +
    theme_publication

  # 3. Tolerance Change by Student Characteristics
  if (data$intervention_period > 0) {

    tolerance_change <- data.frame(
      student_id = 1:data$n_students,
      pre_intervention = data$tolerance[data$intervention_period - 1, ],
      post_intervention = data$tolerance[data$n_periods, ],
      data$student_characteristics
    ) %>%
      mutate(
        tolerance_change = post_intervention - pre_intervention,
        minority_status = ifelse(ethnicity != "White", "Minority", "Non-minority")
      )

    p3 <- ggplot(tolerance_change, aes(x = minority_status, y = tolerance_change, fill = minority_status)) +
      geom_violin(alpha = 0.7) +
      geom_boxplot(width = 0.2, alpha = 0.8) +
      geom_hline(yintercept = 0, linetype = "dashed", alpha = 0.5) +
      scale_fill_viridis_d(name = "Status") +
      labs(
        title = "Intervention Effect by Minority Status",
        subtitle = "Change in tolerance from pre- to post-intervention",
        x = "Student Status",
        y = "Change in Tolerance Score"
      ) +
      theme_publication +
      theme(legend.position = "none")

  } else {
    p3 <- ggplot() + theme_void() + ggtitle("No Intervention Applied")
  }

  # 4. Network-Tolerance Relationship
  if (!is.null(analysis_results) && !is.null(analysis_results$results)) {

    # Extract key parameters
    results_table <- analysis_results$results_table

    # Focus on key effects
    key_effects <- results_table[grep("similarity|average alter|intervention",
                                     results_table$Effect, ignore.case = TRUE), ]

    if (nrow(key_effects) > 0) {
      p4 <- ggplot(key_effects, aes(x = reorder(Effect, Estimate), y = Estimate)) +
        geom_col(aes(fill = Significant != ""), alpha = 0.8) +
        geom_errorbar(aes(ymin = Estimate - 1.96*SE, ymax = Estimate + 1.96*SE),
                      width = 0.2) +
        geom_hline(yintercept = 0, linetype = "dashed", alpha = 0.5) +
        scale_fill_manual(values = c("TRUE" = "steelblue", "FALSE" = "gray60"),
                         name = "Significant") +
        coord_flip() +
        labs(
          title = "Key RSiena Parameter Estimates",
          subtitle = "Effects on network and tolerance dynamics",
          x = "Effect",
          y = "Parameter Estimate (95% CI)"
        ) +
        theme_publication
    } else {
      p4 <- ggplot() + theme_void() + ggtitle("Analysis Results Not Available")
    }

  } else {
    p4 <- ggplot() + theme_void() + ggtitle("Analysis Results Not Available")
  }

  # Combine plots
  combined_plot <- grid.arrange(p1, p2, p3, p4, nrow = 2, ncol = 2,
                               top = "Tolerance Intervention Analysis: Comprehensive Results")

  return(list(
    network_evolution = p1,
    tolerance_distribution = p2,
    intervention_effects = p3,
    parameter_estimates = p4,
    combined = combined_plot
  ))
}

#' Calculate Network Evolution Statistics
#'
#' Computes key network statistics over time.
calculate_network_evolution_stats <- function(data) {

  stats <- data.frame(
    period = 1:data$n_periods,
    density = numeric(data$n_periods),
    transitivity = numeric(data$n_periods),
    reciprocity = numeric(data$n_periods),
    avg_degree = numeric(data$n_periods)
  )

  for (t in 1:data$n_periods) {
    g <- graph_from_adjacency_matrix(data$networks[,,t], mode = "undirected")

    stats$density[t] <- edge_density(g)
    stats$transitivity[t] <- transitivity(g, type = "global")
    stats$reciprocity[t] <- reciprocity(g)  # Will be 1 for undirected
    stats$avg_degree[t] <- mean(degree(g))
  }

  return(stats)
}

# =================================================================
# MAIN ANALYSIS WORKFLOW
# =================================================================

#' Execute Complete Tolerance Intervention Analysis
#'
#' Runs the full analytical workflow from data generation to publication results.
execute_tolerance_intervention_analysis <- function(
  n_students = 120,
  n_periods = 4,
  intervention_period = 2,
  include_analysis = TRUE,
  create_plots = TRUE
) {

  cat("\n")
  cat(rep("=", 80) %>% paste(collapse = ""), "\n")
  cat("TOLERANCE INTERVENTION ANALYSIS - COMPLETE WORKFLOW\n")
  cat(rep("=", 80) %>% paste(collapse = ""), "\n\n")

  # Step 1: Generate data
  cat("STEP 1: Generating intervention data...\n")
  tolerance_data <- create_tolerance_intervention_data(
    n_students = n_students,
    n_periods = n_periods,
    intervention_period = intervention_period,
    network_type = "school_realistic",
    tolerance_homophily = 0.8,
    intervention_effect_size = 0.4,
    measurement_error = 0.2
  )

  # Step 2: Descriptive analysis
  cat("\nSTEP 2: Descriptive analysis...\n")
  network_stats <- calculate_network_evolution_stats(tolerance_data)
  print(network_stats)

  # Step 3: RSiena analysis (if requested)
  analysis_results <- NULL
  if (include_analysis) {
    cat("\nSTEP 3: RSiena analysis...\n")
    analysis_results <- run_tolerance_rsiena_analysis(tolerance_data, include_intervention = TRUE)
  }

  # Step 4: Visualizations (if requested)
  plots <- NULL
  if (create_plots) {
    cat("\nSTEP 4: Creating visualizations...\n")
    plots <- create_tolerance_visualizations(tolerance_data, analysis_results)

    if (interactive()) {
      print(plots$combined)
    }
  }

  # Step 5: Generate summary report
  cat("\nSTEP 5: Generating summary report...\n")

  summary_report <- list(
    data_summary = list(
      n_students = tolerance_data$n_students,
      n_periods = tolerance_data$n_periods,
      intervention_period = tolerance_data$intervention_period,
      initial_tolerance_mean = round(mean(tolerance_data$tolerance[1,], na.rm = TRUE), 3),
      final_tolerance_mean = round(mean(tolerance_data$tolerance[tolerance_data$n_periods,], na.rm = TRUE), 3),
      tolerance_change = round(mean(tolerance_data$tolerance[tolerance_data$n_periods,] -
                                   tolerance_data$tolerance[1,], na.rm = TRUE), 3)
    ),
    network_summary = network_stats,
    analysis_summary = if (!is.null(analysis_results)) {
      list(
        converged = analysis_results$convergence_ok,
        max_convergence_ratio = analysis_results$max_convergence_ratio,
        key_findings = analysis_results$results_table[1:5, ]  # Top 5 effects
      )
    } else {
      "Analysis not performed"
    }
  )

  cat("\n=== ANALYSIS COMPLETE ===\n")
  cat("Data generated for", tolerance_data$n_students, "students over", tolerance_data$n_periods, "periods\n")
  cat("Intervention applied at period", tolerance_data$intervention_period, "\n")
  cat("Mean tolerance change:", summary_report$data_summary$tolerance_change, "\n")

  if (!is.null(analysis_results) && analysis_results$convergence_ok) {
    cat("RSiena analysis: CONVERGED\n")
  } else if (!is.null(analysis_results)) {
    cat("RSiena analysis: DID NOT CONVERGE\n")
  }

  return(list(
    data = tolerance_data,
    network_stats = network_stats,
    analysis = analysis_results,
    plots = plots,
    summary = summary_report
  ))
}

# =================================================================
# EXECUTION
# =================================================================

# Execute the complete analysis if script is run directly
if (interactive() || !exists("sourced")) {

  cat("Starting Advanced Tolerance Intervention Analysis\n")
  cat("This demonstration showcases publication-ready RSiena analysis\n\n")

  # Run with smaller sample for demonstration
  results <- execute_tolerance_intervention_analysis(
    n_students = 80,      # Smaller for faster execution
    n_periods = 4,
    intervention_period = 2,
    include_analysis = TRUE,
    create_plots = TRUE
  )

  # Display key results
  cat("\n" %>% paste(rep("=", 60), collapse = ""), "\n")
  cat("DEMONSTRATION COMPLETE\n")
  cat(rep("=", 60) %>% paste(collapse = ""), "\n")
  cat("Check the 'results' object for complete output\n")
  cat("Visualizations created and displayed\n")
  cat("Analysis results available in results$analysis\n")
}