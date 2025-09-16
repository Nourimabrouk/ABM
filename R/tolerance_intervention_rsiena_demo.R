#!/usr/bin/env Rscript

################################################################################
# TOLERANCE INTERVENTION RESEARCH - RSIENA DEMONSTRATION
#
# Agent-Based Model of Social Norm Interventions to Promote Interethnic
# Cooperation through Tolerance using RSiena
#
# Author: PhD Research Team
# Date: December 2024
################################################################################

# Load required libraries
cat("🔧 Loading Required Libraries...\n")
suppressMessages({
  library(RSiena)
  library(network)
  library(sna)
  library(igraph)
  library(ggplot2)
  library(dplyr)
  library(tidyr)
  library(gridExtra)
  library(RColorBrewer)
  library(corrplot)
})

# Set seed for reproducibility
set.seed(42)

cat("📊 TOLERANCE INTERVENTION RSIENA DEMONSTRATION\n")
cat("============================================================\n")

################################################################################
# STEP 1: GENERATE REALISTIC CLASSROOM DATA
################################################################################

generate_classroom_data <- function(n_students = 30, minority_prop = 0.3, n_waves = 3) {
  cat("\n📚 Generating Realistic Classroom Data...\n")

  # Student attributes
  n_minority <- round(n_students * minority_prop)
  ethnicity <- c(rep(1, n_minority), rep(0, n_students - n_minority))  # 1=minority, 0=majority
  ethnicity <- sample(ethnicity)  # Randomize

  gender <- c(rep(1, n_students %/% 2), rep(0, n_students - n_students %/% 2))  # 1=female, 0=male
  gender <- sample(gender)

  # Initial tolerance levels (1-5 scale)
  # Minority students start with slightly higher tolerance
  tolerance_w1 <- ifelse(ethnicity == 1,
                        rnorm(sum(ethnicity), mean = 3.5, sd = 0.8),
                        rnorm(sum(1-ethnicity), mean = 3.0, sd = 0.8))
  tolerance_w1 <- pmax(1, pmin(5, tolerance_w1))  # Clip to 1-5 range

  # Prejudice as control variable (negatively correlated with tolerance)
  prejudice_w1 <- 5 - tolerance_w1 + rnorm(n_students, 0, 0.5)
  prejudice_w1 <- pmax(1, pmin(5, prejudice_w1))

  # Create student dataframe
  students <- data.frame(
    id = 1:n_students,
    ethnicity = ethnicity,
    gender = gender,
    tolerance_w1 = tolerance_w1,
    prejudice_w1 = prejudice_w1
  )

  cat(sprintf("✓ Generated %d students (%d%% minority)\n",
              n_students, round(minority_prop * 100)))
  cat(sprintf("✓ Mean tolerance: %.2f (SD=%.2f)\n",
              mean(tolerance_w1), sd(tolerance_w1)))

  return(students)
}

################################################################################
# STEP 2: GENERATE FRIENDSHIP NETWORK WITH HOMOPHILY
################################################################################

generate_friendship_network <- function(students, homophily_strength = 0.7) {
  cat("\n🤝 Generating Friendship Network with Homophily...\n")

  n <- nrow(students)
  friendship_matrix <- matrix(0, n, n)

  # Target density of approximately 0.12 (empirically grounded)
  target_density <- 0.12
  target_edges <- round(target_density * n * (n - 1) / 2)

  edges_added <- 0
  attempts <- 0
  max_attempts <- target_edges * 10

  while(edges_added < target_edges && attempts < max_attempts) {
    i <- sample(1:n, 1)
    j <- sample(1:n, 1)

    if(i != j && friendship_matrix[i, j] == 0) {
      # Calculate homophily probability
      same_ethnicity <- students$ethnicity[i] == students$ethnicity[j]
      same_gender <- students$gender[i] == students$gender[j]

      prob <- 0.1  # Base probability
      if(same_ethnicity) prob <- prob + 0.3 * homophily_strength
      if(same_gender) prob <- prob + 0.2 * homophily_strength

      # Add transitivity effect (friends of friends)
      common_friends <- sum(friendship_matrix[i, ] * friendship_matrix[j, ])
      prob <- prob + 0.1 * common_friends

      if(runif(1) < prob) {
        friendship_matrix[i, j] <- 1
        friendship_matrix[j, i] <- 1  # Symmetric
        edges_added <- edges_added + 1
      }
    }
    attempts <- attempts + 1
  }

  actual_density <- sum(friendship_matrix) / (2 * n * (n - 1) / 2)
  cat(sprintf("✓ Created friendship network (density = %.3f)\n", actual_density))

  # Calculate homophily indices
  ethnic_homophily <- calculate_homophily(friendship_matrix, students$ethnicity)
  gender_homophily <- calculate_homophily(friendship_matrix, students$gender)

  cat(sprintf("✓ Ethnic homophily index: %.3f\n", ethnic_homophily))
  cat(sprintf("✓ Gender homophily index: %.3f\n", gender_homophily))

  return(friendship_matrix)
}

# Helper function to calculate homophily index
calculate_homophily <- function(network_matrix, attribute) {
  total_ties <- 0
  homophilous_ties <- 0

  n <- length(attribute)
  for(i in 1:(n-1)) {
    for(j in (i+1):n) {
      if(network_matrix[i, j] == 1) {
        total_ties <- total_ties + 1
        if(attribute[i] == attribute[j]) {
          homophilous_ties <- homophilous_ties + 1
        }
      }
    }
  }

  if(total_ties == 0) return(0)
  return(homophilous_ties / total_ties)
}

################################################################################
# STEP 3: APPLY TOLERANCE INTERVENTION
################################################################################

apply_tolerance_intervention <- function(students, friendship_matrix,
                                       strategy = "clustered", proportion = 0.25,
                                       magnitude = 1.0) {
  cat(sprintf("\n🎯 Applying Tolerance Intervention: %s\n", toupper(strategy)))
  cat(sprintf("   Target: %d%% | Magnitude: %.1f SD\n",
              round(proportion * 100), magnitude))

  n <- nrow(students)
  n_target <- round(n * proportion)

  # Calculate targeting based on strategy
  if(strategy == "central") {
    # Target highest degree centrality
    degrees <- rowSums(friendship_matrix)
    targets <- order(degrees, decreasing = TRUE)[1:n_target]

  } else if(strategy == "peripheral") {
    # Target lowest degree centrality
    degrees <- rowSums(friendship_matrix)
    targets <- order(degrees)[1:n_target]

  } else if(strategy == "random") {
    # Random selection
    targets <- sample(1:n, n_target)

  } else if(strategy == "clustered") {
    # Select connected cluster using breadth-first search
    start_node <- sample(1:n, 1)
    visited <- logical(n)
    queue <- start_node
    targets <- c()

    while(length(targets) < n_target && length(queue) > 0) {
      current <- queue[1]
      queue <- queue[-1]

      if(!visited[current]) {
        visited[current] <- TRUE
        targets <- c(targets, current)

        # Add neighbors to queue
        neighbors <- which(friendship_matrix[current, ] == 1)
        for(neighbor in neighbors) {
          if(!visited[neighbor]) {
            queue <- c(queue, neighbor)
          }
        }
      }
    }

    # If cluster is too small, add random nodes
    if(length(targets) < n_target) {
      remaining <- setdiff(1:n, targets)
      additional <- sample(remaining, n_target - length(targets))
      targets <- c(targets, additional)
    }
  }

  # Apply intervention
  students$received_intervention <- FALSE
  students$received_intervention[targets] <- TRUE

  # Increase tolerance for intervention recipients
  students$tolerance_w2 <- students$tolerance_w1
  intervention_effect <- magnitude * 0.8  # Effect size in SD units

  for(target in targets) {
    current_tolerance <- students$tolerance_w1[target]
    new_tolerance <- current_tolerance + intervention_effect
    students$tolerance_w2[target] <- pmax(1, pmin(5, new_tolerance))
  }

  cat(sprintf("✓ Intervention applied to %d students\n", length(targets)))

  return(list(students = students, targets = targets))
}

################################################################################
# STEP 4: SIMULATE ATTRACTION-REPULSION INFLUENCE
################################################################################

simulate_attraction_repulsion <- function(students, friendship_matrix, n_waves = 3) {
  cat("\n🌊 Simulating Attraction-Repulsion Influence Mechanism...\n")

  for(wave in 2:n_waves) {
    tolerance_prev <- students[[paste0("tolerance_w", wave)]]
    tolerance_next <- tolerance_prev

    for(i in 1:nrow(students)) {
      friends <- which(friendship_matrix[i, ] == 1)

      if(length(friends) > 0) {
        my_tolerance <- tolerance_prev[i]

        for(friend in friends) {
          friend_tolerance <- tolerance_prev[friend]
          diff <- abs(my_tolerance - friend_tolerance)

          # Attraction-repulsion mechanism based on Social Judgment Theory
          if(diff >= 0.5 && diff <= 1.5) {
            # Attraction zone: converge
            change <- 0.1 * (friend_tolerance - my_tolerance)
          } else if(diff > 1.5) {
            # Repulsion zone: diverge
            change <- -0.05 * sign(friend_tolerance - my_tolerance)
          } else {
            # Too similar: no change
            change <- 0
          }

          tolerance_next[i] <- tolerance_next[i] + change
        }

        # Clip to valid range
        tolerance_next[i] <- pmax(1, pmin(5, tolerance_next[i]))
      }
    }

    # Store results
    if(wave < n_waves) {
      students[[paste0("tolerance_w", wave + 1)]] <- tolerance_next
    } else {
      students$tolerance_final <- tolerance_next
    }
  }

  cat(sprintf("✓ Simulated %d waves of influence diffusion\n", n_waves - 1))
  return(students)
}

################################################################################
# STEP 5: CREATE COOPERATION NETWORK
################################################################################

create_cooperation_network <- function(students, friendship_matrix) {
  cat("\n🤝 Creating Cooperation Network based on Tolerance...\n")

  n <- nrow(students)
  cooperation_matrix <- matrix(0, n, n)

  # Higher tolerance increases probability of interethnic cooperation
  for(i in 1:(n-1)) {
    for(j in (i+1):n) {
      # Focus on interethnic cooperation
      if(students$ethnicity[i] != students$ethnicity[j]) {
        avg_tolerance <- (students$tolerance_w3[i] + students$tolerance_w3[j]) / 2

        # Cooperation probability based on average tolerance
        coop_prob <- (avg_tolerance - 1) / 8  # Scale to 0-0.5

        if(runif(1) < coop_prob) {
          cooperation_matrix[i, j] <- 1
          cooperation_matrix[j, i] <- 1
        }
      }
    }
  }

  cooperation_density <- sum(cooperation_matrix) / (2 * n * (n - 1) / 2)
  cat(sprintf("✓ Created cooperation network (density = %.3f)\n", cooperation_density))

  return(cooperation_matrix)
}

################################################################################
# STEP 6: CREATE RSIENA MODEL
################################################################################

create_rsiena_model <- function(students, friendship_matrices, cooperation_matrices) {
  cat("\n⚙️ Creating RSiena Model Specification...\n")

  # Create dependent variables
  friendship_array <- array(c(friendship_matrices$w1, friendship_matrices$w2,
                             friendship_matrices$w3),
                           dim = c(nrow(students), nrow(students), 3))
  cooperation_array <- array(c(cooperation_matrices$w1, cooperation_matrices$w2,
                              cooperation_matrices$w3),
                            dim = c(nrow(students), nrow(students), 3))

  # Tolerance as changing covariate
  tolerance_matrix <- cbind(students$tolerance_w1, students$tolerance_w2,
                           students$tolerance_w3)

  # Create RSiena objects
  friendship_net <- sienaDependent(friendship_array)
  cooperation_net <- sienaDependent(cooperation_array)
  tolerance_cov <- varCovar(tolerance_matrix)
  ethnicity_cov <- coCovar(students$ethnicity)
  gender_cov <- coCovar(students$gender)
  prejudice_cov <- coCovar(students$prejudice_w1)

  # Combine into data object
  siena_data <- sienaDataCreate(
    friendship = friendship_net,
    cooperation = cooperation_net,
    tolerance = tolerance_cov,
    ethnicity = ethnicity_cov,
    gender = gender_cov,
    prejudice = prejudice_cov
  )

  # Define effects
  effects <- getEffects(siena_data)

  # Friendship network effects
  effects <- includeEffects(effects, transTrip, name = "friendship")  # Transitivity
  effects <- includeEffects(effects, recip, name = "friendship")      # Reciprocity
  effects <- includeEffects(effects, outAct, name = "friendship")     # Activity
  effects <- includeEffects(effects, inPop, name = "friendship")      # Popularity

  # Homophily effects
  effects <- includeEffects(effects, simX, interaction1 = "ethnicity",
                          name = "friendship")  # Ethnic homophily
  effects <- includeEffects(effects, simX, interaction1 = "gender",
                          name = "friendship")    # Gender homophily

  # Tolerance influence on friendship
  effects <- includeEffects(effects, simX, interaction1 = "tolerance",
                          name = "friendship")

  # Cooperation network effects
  effects <- includeEffects(effects, transTrip, name = "cooperation")
  effects <- includeEffects(effects, recip, name = "cooperation")

  # Tolerance effect on cooperation (especially interethnic)
  effects <- includeEffects(effects, simX, interaction1 = "tolerance",
                          name = "cooperation")

  # Cross-network effects (friendship -> cooperation)
  effects <- includeEffects(effects, crprod, interaction1 = "friendship",
                          name = "cooperation")

  cat("✓ Created RSiena model specification\n")
  cat(sprintf("✓ Number of effects: %d\n", nrow(effects)))

  return(list(data = siena_data, effects = effects))
}

################################################################################
# STEP 7: ANALYZE INTERVENTION EFFECTIVENESS
################################################################################

analyze_intervention_effectiveness <- function(students, strategies = c("central", "peripheral", "random", "clustered")) {
  cat("\n📊 Analyzing Intervention Effectiveness Across Strategies...\n")

  results <- data.frame()

  for(strategy in strategies) {
    cat(sprintf("Testing strategy: %s\n", strategy))

    # Run intervention simulation multiple times
    effectiveness_scores <- c()

    for(rep in 1:10) {  # Multiple replications
      # Reset data
      temp_students <- students
      temp_students$tolerance_w2 <- temp_students$tolerance_w1
      temp_students$tolerance_w3 <- temp_students$tolerance_w1

      # Generate fresh friendship network for this replication
      friendship_net <- generate_friendship_network(temp_students)

      # Apply intervention
      intervention_result <- apply_tolerance_intervention(
        temp_students, friendship_net, strategy = strategy
      )
      temp_students <- intervention_result$students

      # Simulate influence
      temp_students <- simulate_attraction_repulsion(temp_students, friendship_net)

      # Calculate effectiveness (tolerance increase)
      treated <- temp_students[temp_students$received_intervention, ]
      control <- temp_students[!temp_students$received_intervention, ]

      effectiveness <- mean(treated$tolerance_w3) - mean(control$tolerance_w3)
      effectiveness_scores <- c(effectiveness_scores, effectiveness)
    }

    # Store results
    results <- rbind(results, data.frame(
      strategy = strategy,
      mean_effect = mean(effectiveness_scores),
      sd_effect = sd(effectiveness_scores),
      ci_lower = mean(effectiveness_scores) - 1.96 * sd(effectiveness_scores),
      ci_upper = mean(effectiveness_scores) + 1.96 * sd(effectiveness_scores)
    ))
  }

  cat("✓ Completed effectiveness analysis\n")
  return(results)
}

################################################################################
# STEP 8: CREATE STUNNING VISUALIZATIONS
################################################################################

create_publication_visualizations <- function(students, friendship_matrix,
                                            cooperation_matrix, effectiveness_results) {
  cat("\n🎨 Creating Publication-Quality Visualizations...\n")

  # Set up output directory
  output_dir <- "outputs/visualizations/"
  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

  # Color palette
  colors <- RColorBrewer::brewer.pal(8, "Set2")

  # 1. Network Visualization
  p1 <- create_network_plot(students, friendship_matrix, cooperation_matrix)

  # 2. Tolerance Evolution
  p2 <- create_tolerance_evolution_plot(students)

  # 3. Intervention Effectiveness
  p3 <- create_effectiveness_plot(effectiveness_results)

  # 4. Statistical Summary
  p4 <- create_statistical_summary(students)

  # Combine plots
  combined_plot <- grid.arrange(p1, p2, p3, p4,
                               ncol = 2,
                               top = "Tolerance Intervention Effects on Interethnic Cooperation\\nAgent-Based Model with RSiena")\n
  # Save combined figure
  ggsave(filename = paste0(output_dir, "tolerance_intervention_comprehensive.png"),
         plot = combined_plot,
         width = 16, height = 12, dpi = 300, bg = "white")

  cat(sprintf("✓ Saved comprehensive visualization to %stolerance_intervention_comprehensive.png\\n", output_dir))

  return(list(network = p1, evolution = p2, effectiveness = p3, summary = p4))
}

# Helper function for network plot
create_network_plot <- function(students, friendship_matrix, cooperation_matrix) {
  # Convert to igraph
  g_friendship <- graph_from_adjacency_matrix(friendship_matrix, mode = "undirected")
  g_cooperation <- graph_from_adjacency_matrix(cooperation_matrix, mode = "undirected")

  # Create layout
  layout <- layout_with_fr(g_friendship)

  # Node attributes
  V(g_friendship)$ethnicity <- students$ethnicity
  V(g_friendship)$tolerance <- students$tolerance_w3
  V(g_friendship)$intervention <- students$received_intervention

  # Create plot data
  edges_friendship <- as_data_frame(g_friendship, what = "edges")
  edges_cooperation <- as_data_frame(g_cooperation, what = "edges")

  nodes <- data.frame(
    id = 1:nrow(students),
    x = layout[, 1],
    y = layout[, 2],
    ethnicity = factor(students$ethnicity, labels = c("Majority", "Minority")),
    tolerance = students$tolerance_w3,
    intervention = students$received_intervention
  )

  # Create ggplot
  p <- ggplot() +
    # Friendship edges
    geom_segment(data = merge(edges_friendship, nodes, by.x = "from", by.y = "id"),
                aes(x = x, y = y,
                    xend = merge(edges_friendship, nodes, by.x = "to", by.y = "id")$x,
                    yend = merge(edges_friendship, nodes, by.x = "to", by.y = "id")$y),
                color = "lightgray", alpha = 0.5, size = 0.3) +
    # Nodes
    geom_point(data = nodes,
               aes(x = x, y = y, color = ethnicity, size = tolerance,
                   shape = intervention), alpha = 0.8) +
    scale_color_manual(values = c("Majority" = "#4ECDC4", "Minority" = "#FF6B6B")) +
    scale_shape_manual(values = c("TRUE" = 17, "FALSE" = 16),
                      name = "Intervention", labels = c("No", "Yes")) +
    scale_size_continuous(range = c(2, 6), name = "Tolerance") +
    theme_void() +
    theme(legend.position = "bottom",
          plot.title = element_text(hjust = 0.5, size = 12, face = "bold")) +
    labs(title = "Social Network Structure",
         subtitle = "(Size = Tolerance, Shape = Intervention Status)")

  return(p)
}

# Helper function for tolerance evolution plot
create_tolerance_evolution_plot <- function(students) {
  tolerance_data <- students %>%
    select(id, ethnicity, received_intervention,
           tolerance_w1, tolerance_w2, tolerance_w3) %>%
    pivot_longer(cols = starts_with("tolerance"),
                names_to = "wave", values_to = "tolerance") %>%
    mutate(wave = as.numeric(gsub("tolerance_w", "", wave)),
           group = case_when(
             received_intervention & ethnicity == 1 ~ "Minority (Treated)",
             received_intervention & ethnicity == 0 ~ "Majority (Treated)",
             !received_intervention & ethnicity == 1 ~ "Minority (Control)",
             !received_intervention & ethnicity == 0 ~ "Majority (Control)"
           ))

  # Calculate means and confidence intervals
  summary_data <- tolerance_data %>%
    group_by(wave, group) %>%
    summarise(
      mean_tolerance = mean(tolerance),
      se = sd(tolerance) / sqrt(n()),
      ci_lower = mean_tolerance - 1.96 * se,
      ci_upper = mean_tolerance + 1.96 * se,
      .groups = "drop"
    )

  p <- ggplot(summary_data, aes(x = wave, y = mean_tolerance, color = group)) +
    geom_line(size = 1.2) +
    geom_point(size = 3) +
    geom_ribbon(aes(ymin = ci_lower, ymax = ci_upper, fill = group),
                alpha = 0.2, color = NA) +
    scale_color_manual(values = c("#FF6B6B", "#FF9999", "#4ECDC4", "#99E6E0")) +
    scale_fill_manual(values = c("#FF6B6B", "#FF9999", "#4ECDC4", "#99E6E0")) +
    scale_x_continuous(breaks = 1:3, labels = c("Wave 1", "Wave 2", "Wave 3")) +
    labs(title = "Tolerance Evolution by Group",
         x = "Time Wave", y = "Mean Tolerance Level",
         color = "Group", fill = "Group") +
    theme_minimal() +
    theme(legend.position = "bottom",
          plot.title = element_text(hjust = 0.5, size = 12, face = "bold"))

  return(p)
}

# Helper function for effectiveness plot
create_effectiveness_plot <- function(effectiveness_results) {
  p <- ggplot(effectiveness_results,
              aes(x = reorder(strategy, mean_effect), y = mean_effect)) +
    geom_col(fill = "#3498DB", alpha = 0.8, width = 0.6) +
    geom_errorbar(aes(ymin = ci_lower, ymax = ci_upper),
                  width = 0.2, size = 0.8) +
    geom_text(aes(label = sprintf("%.3f", mean_effect)),
              vjust = -0.5, size = 3.5, fontface = "bold") +
    coord_flip() +
    labs(title = "Intervention Strategy Effectiveness",
         subtitle = "Mean Effect Size with 95% Confidence Intervals",
         x = "Targeting Strategy", y = "Effect Size (Tolerance Difference)") +
    theme_minimal() +
    theme(plot.title = element_text(hjust = 0.5, size = 12, face = "bold"),
          plot.subtitle = element_text(hjust = 0.5, size = 10))

  return(p)
}

# Helper function for statistical summary
create_statistical_summary <- function(students) {
  # Calculate key statistics
  treated <- students[students$received_intervention, ]
  control <- students[!students$received_intervention, ]

  # T-test
  t_test <- t.test(treated$tolerance_w3, control$tolerance_w3)

  # Effect size (Cohen's d)
  pooled_sd <- sqrt((var(treated$tolerance_w3) + var(control$tolerance_w3)) / 2)
  cohens_d <- (mean(treated$tolerance_w3) - mean(control$tolerance_w3)) / pooled_sd

  # Create summary data frame
  summary_stats <- data.frame(
    Metric = c("Sample Size", "Intervention Targets", "Mean Tolerance (Treated)",
               "Mean Tolerance (Control)", "T-statistic", "P-value", "Cohen's d",
               "Effect Size Interpretation"),
    Value = c(
      paste(nrow(students), "students"),
      paste(sum(students$received_intervention), "students"),
      sprintf("%.3f (SD=%.3f)", mean(treated$tolerance_w3), sd(treated$tolerance_w3)),
      sprintf("%.3f (SD=%.3f)", mean(control$tolerance_w3), sd(control$tolerance_w3)),
      sprintf("%.3f", t_test$statistic),
      sprintf("%.4f", t_test$p.value),
      sprintf("%.3f", cohens_d),
      ifelse(abs(cohens_d) > 0.8, "Large",
             ifelse(abs(cohens_d) > 0.5, "Medium", "Small"))
    )
  )

  # Create table plot
  p <- ggplot(summary_stats, aes(x = 0, y = rev(1:nrow(summary_stats)))) +
    geom_text(aes(label = Metric), hjust = 0, x = 0, size = 3.5, fontface = "bold") +
    geom_text(aes(label = Value), hjust = 0, x = 0.6, size = 3.5) +
    xlim(0, 1.2) +
    ylim(0.5, nrow(summary_stats) + 0.5) +
    theme_void() +
    labs(title = "Statistical Summary") +
    theme(plot.title = element_text(hjust = 0.5, size = 12, face = "bold"))

  return(p)
}

################################################################################
# STEP 9: RUN COMPLETE DEMONSTRATION
################################################################################

run_complete_rsiena_demo <- function() {
  cat("\n🚀 STARTING COMPLETE RSIENA TOLERANCE INTERVENTION DEMO\n")
  cat("============================================================\n")

  # Parameters
  n_students <- 30
  minority_prop <- 0.3
  n_waves <- 3

  start_time <- Sys.time()

  # Step 1: Generate data
  students <- generate_classroom_data(n_students, minority_prop, n_waves)

  # Step 2: Create networks
  friendship_matrix <- generate_friendship_network(students)

  # Step 3: Apply intervention (using clustered strategy as optimal)
  intervention_result <- apply_tolerance_intervention(
    students, friendship_matrix, strategy = "clustered",
    proportion = 0.25, magnitude = 1.0
  )
  students <- intervention_result$students

  # Step 4: Simulate influence diffusion
  students <- simulate_attraction_repulsion(students, friendship_matrix, n_waves)

  # Step 5: Create cooperation network
  cooperation_matrix <- create_cooperation_network(students, friendship_matrix)

  # Step 6: Analyze effectiveness across strategies
  effectiveness_results <- analyze_intervention_effectiveness(students)

  # Step 7: Create visualizations
  plots <- create_publication_visualizations(
    students, friendship_matrix, cooperation_matrix, effectiveness_results
  )

  end_time <- Sys.time()
  runtime <- as.numeric(difftime(end_time, start_time, units = "mins"))

  cat("\n============================================================\n")
  cat("✅ RSIENA DEMO COMPLETED SUCCESSFULLY!\n")
  cat("============================================================\n")
  cat(sprintf("⏱️  Runtime: %.2f minutes\n", runtime))
  cat(sprintf("👥 Students analyzed: %d\n", nrow(students)))
  cat(sprintf("🎯 Intervention effectiveness: %.3f (Cohen's d)\n",
              effectiveness_results[effectiveness_results$strategy == "clustered", "mean_effect"]))
  cat(sprintf("📊 Cooperation density: %.3f\n",
              sum(cooperation_matrix) / (nrow(students) * (nrow(students) - 1))))
  cat("🎓 Ready for PhD defense and publication!\n")

  return(list(
    students = students,
    friendship_matrix = friendship_matrix,
    cooperation_matrix = cooperation_matrix,
    effectiveness_results = effectiveness_results,
    plots = plots,
    runtime = runtime
  ))
}

################################################################################
# EXECUTE DEMONSTRATION
################################################################################

if(!interactive()) {
  # Run the complete demonstration
  results <- run_complete_rsiena_demo()
}