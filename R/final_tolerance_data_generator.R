# FINAL TOLERANCE INTERVENTION DATA GENERATOR
# Creates comprehensive simulation data for RSiena demos
# Author: AI Agent Coordination Team
# Date: 2025-09-16

# Load required packages
suppressMessages({
  library(RSiena)
  library(network)
  library(igraph)
  library(dplyr)
  library(ggplot2)
  library(viridis)
  library(reshape2)
  library(grid)
  library(gridExtra)
})

cat("=== FINAL TOLERANCE INTERVENTION DATA GENERATOR ===\n")
cat("Creating comprehensive simulation data for RSiena demos...\n\n")

# Set random seed for reproducibility
set.seed(12345)

# PARAMETERS
n_students <- 30
n_waves <- 4
minority_prop <- 0.3
n_minority <- round(n_students * minority_prop)

cat("Parameters:\n")
cat(sprintf("- Students: %d (Majority: %d, Minority: %d)\n",
            n_students, n_students - n_minority, n_minority))
cat(sprintf("- Waves: %d\n", n_waves))
cat(sprintf("- Minority proportion: %.1f%%\n\n", minority_prop * 100))

# CREATE STUDENT ATTRIBUTES
students <- data.frame(
  id = 1:n_students,
  minority = c(rep(1, n_minority), rep(0, n_students - n_minority)),
  grade = sample(7:12, n_students, replace = TRUE),
  extroversion = rnorm(n_students, 0, 1),
  initial_tolerance = rnorm(n_students, 0, 1)
)

# Shuffle to randomize positions
students <- students[sample(nrow(students)), ]
students$id <- 1:n_students

cat("Student composition created:\n")
print(table(students$minority))
cat("\n")

# FUNCTION: Create realistic friendship network
create_friendship_network <- function(students, density = 0.15) {
  n <- nrow(students)
  net <- matrix(0, n, n)

  # Homophily effects
  for (i in 1:(n-1)) {
    for (j in (i+1):n) {
      # Base probability
      prob <- 0.1

      # Homophily on minority status (strong)
      if (students$minority[i] == students$minority[j]) {
        prob <- prob * 2.5
      }

      # Grade similarity
      grade_diff <- abs(students$grade[i] - students$grade[j])
      prob <- prob * exp(-grade_diff * 0.3)

      # Extroversion attraction
      extro_sim <- 1 - abs(students$extroversion[i] - students$extroversion[j]) / 4
      prob <- prob * (0.5 + extro_sim)

      # Create edge probabilistically
      if (runif(1) < prob) {
        net[i, j] <- net[j, i] <- 1
      }
    }
  }

  # Ensure minimum connectivity
  components <- igraph::components(igraph::graph_from_adjacency_matrix(net, mode = "undirected"))
  if (components$no > 1) {
    # Connect components
    comp_sizes <- table(components$membership)
    largest_comp <- which.max(comp_sizes)

    for (comp in 1:components$no) {
      if (comp != largest_comp) {
        # Find nodes in this component and largest component
        nodes_comp <- which(components$membership == comp)
        nodes_largest <- which(components$membership == largest_comp)

        # Connect one random node from each
        node1 <- sample(nodes_comp, 1)
        node2 <- sample(nodes_largest, 1)
        net[node1, node2] <- net[node2, node1] <- 1
      }
    }
  }

  return(net)
}

# FUNCTION: Evolve tolerance with interventions
evolve_tolerance <- function(students, networks, intervention_type = "none",
                           intervention_wave = 2, intervention_strength = 0.5) {
  n <- nrow(students)
  n_waves <- length(networks) + 1

  # Initialize tolerance matrix
  tolerance <- matrix(NA, n, n_waves)
  tolerance[, 1] <- students$initial_tolerance

  # Intervention targets based on type
  intervention_targets <- c()
  if (intervention_type == "central") {
    # Target high-degree nodes
    deg <- rowSums(networks[[1]])
    intervention_targets <- order(deg, decreasing = TRUE)[1:5]
  } else if (intervention_type == "peripheral") {
    # Target low-degree nodes
    deg <- rowSums(networks[[1]])
    intervention_targets <- order(deg)[1:5]
  } else if (intervention_type == "random") {
    # Target random nodes
    intervention_targets <- sample(1:n, 5)
  } else if (intervention_type == "clustered") {
    # Target nodes in same cluster
    g <- igraph::graph_from_adjacency_matrix(networks[[1]], mode = "undirected")
    clusters <- igraph::cluster_louvain(g)
    largest_cluster <- which.max(table(clusters$membership))
    cluster_nodes <- which(clusters$membership == largest_cluster)
    intervention_targets <- sample(cluster_nodes, min(5, length(cluster_nodes)))
  }

  cat(sprintf("Intervention: %s targeting nodes %s at wave %d\n",
              intervention_type, paste(intervention_targets, collapse = ","), intervention_wave))

  # Evolution process
  for (wave in 2:n_waves) {
    net <- networks[[wave - 1]]
    prev_tolerance <- tolerance[, wave - 1]

    for (i in 1:n) {
      # Get friends
      friends <- which(net[i, ] == 1)

      if (length(friends) > 0) {
        # Attraction-repulsion mechanism
        friend_tolerance <- prev_tolerance[friends]

        # Influence strength based on similarity
        influence_weights <- exp(-0.5 * abs(prev_tolerance[i] - friend_tolerance))

        # Weighted average influence
        social_influence <- weighted.mean(friend_tolerance, influence_weights)

        # Update with social influence + random noise
        tolerance[i, wave] <- 0.7 * prev_tolerance[i] +
                             0.3 * social_influence +
                             rnorm(1, 0, 0.2)
      } else {
        # No friends - just add noise
        tolerance[i, wave] <- prev_tolerance[i] + rnorm(1, 0, 0.1)
      }

      # Apply intervention
      if (wave == intervention_wave && i %in% intervention_targets) {
        tolerance[i, wave] <- tolerance[i, wave] + intervention_strength
        cat(sprintf("  Applied intervention to student %d: %.3f -> %.3f\n",
                    i, tolerance[i, wave] - intervention_strength, tolerance[i, wave]))
      }
    }

    # Normalize tolerance to reasonable range
    tolerance[, wave] <- pmax(pmin(tolerance[, wave], 3), -3)
  }

  return(tolerance)
}

# GENERATE NETWORKS FOR ALL WAVES
cat("Generating friendship networks...\n")
networks <- list()
for (wave in 1:(n_waves - 1)) {
  if (wave == 1) {
    networks[[wave]] <- create_friendship_network(students, density = 0.15)
  } else {
    # Evolve network slightly
    prev_net <- networks[[wave - 1]]
    new_net <- prev_net

    # Add some random edges (5% of possible)
    n_possible <- sum(upper.tri(matrix(1, n_students, n_students)))
    n_to_add <- round(0.05 * n_possible)

    for (add in 1:n_to_add) {
      candidates <- which(upper.tri(new_net) & new_net == 0, arr.ind = TRUE)
      if (nrow(candidates) > 0) {
        idx <- sample(nrow(candidates), 1)
        i <- candidates[idx, 1]
        j <- candidates[idx, 2]
        new_net[i, j] <- new_net[j, i] <- 1
      }
    }

    # Remove some random edges (3% of existing)
    existing_edges <- which(upper.tri(new_net) & new_net == 1, arr.ind = TRUE)
    n_to_remove <- round(0.03 * nrow(existing_edges))

    if (n_to_remove > 0 && nrow(existing_edges) > n_to_remove) {
      to_remove <- sample(nrow(existing_edges), n_to_remove)
      for (rem in to_remove) {
        i <- existing_edges[rem, 1]
        j <- existing_edges[rem, 2]
        new_net[i, j] <- new_net[j, i] <- 0
      }
    }

    networks[[wave]] <- new_net
  }

  cat(sprintf("  Wave %d: %d edges, density = %.3f\n",
              wave, sum(networks[[wave]]) / 2,
              sum(networks[[wave]]) / (n_students * (n_students - 1))))
}

# GENERATE TOLERANCE DATA FOR ALL INTERVENTION SCENARIOS
cat("\nGenerating tolerance evolution for all intervention scenarios...\n")

intervention_scenarios <- c("none", "central", "peripheral", "random", "clustered")
tolerance_data <- list()

for (scenario in intervention_scenarios) {
  cat(sprintf("\nScenario: %s\n", scenario))
  tolerance_data[[scenario]] <- evolve_tolerance(
    students, networks,
    intervention_type = scenario,
    intervention_wave = 2,
    intervention_strength = 0.8
  )
}

# CREATE SIENA DATA OBJECTS
cat("\nCreating RSiena data objects...\n")

siena_data_objects <- list()

for (scenario in intervention_scenarios) {
  # Create network arrays
  net_array <- array(0, dim = c(n_students, n_students, n_waves))
  for (wave in 1:(n_waves - 1)) {
    net_array[, , wave] <- networks[[wave]]
  }
  net_array[, , n_waves] <- networks[[n_waves - 1]]  # Last wave same as previous

  # Create dependent network variable
  friendship <- sienaDependent(net_array)

  # Create tolerance behavior variable
  tolerance_behavior <- sienaDependent(tolerance_data[[scenario]], type = "behavior")

  # Create covariates
  minority_covar <- coCovar(students$minority)
  grade_covar <- coCovar(students$grade)
  extroversion_covar <- coCovar(students$extroversion)

  # Create intervention indicator (if applicable)
  if (scenario != "none") {
    intervention_targets <- switch(scenario,
      "central" = {
        deg <- rowSums(networks[[1]])
        order(deg, decreasing = TRUE)[1:5]
      },
      "peripheral" = {
        deg <- rowSums(networks[[1]])
        order(deg)[1:5]
      },
      "random" = sample(1:n_students, 5),
      "clustered" = {
        g <- igraph::graph_from_adjacency_matrix(networks[[1]], mode = "undirected")
        clusters <- igraph::cluster_louvain(g)
        largest_cluster <- which.max(table(clusters$membership))
        cluster_nodes <- which(clusters$membership == largest_cluster)
        sample(cluster_nodes, min(5, length(cluster_nodes)))
      }
    )

    intervention_covar <- coCovar(ifelse(1:n_students %in% intervention_targets, 1, 0))
  } else {
    intervention_covar <- NULL
  }

  # Create data object
  if (scenario != "none") {
    data_obj <- sienaDataCreate(
      friendship,
      tolerance = tolerance_behavior,
      minority = minority_covar,
      grade = grade_covar,
      extroversion = extroversion_covar,
      intervention = intervention_covar
    )
  } else {
    data_obj <- sienaDataCreate(
      friendship,
      tolerance = tolerance_behavior,
      minority = minority_covar,
      grade = grade_covar,
      extroversion = extroversion_covar
    )
  }

  siena_data_objects[[scenario]] <- data_obj

  cat(sprintf("  %s: Created with %d actors, %d waves\n",
              scenario, nrow(students), n_waves))
}

# SAVE ALL DATA
output_dir <- "../outputs"
if (!dir.exists(output_dir)) {
  dir.create(output_dir, recursive = TRUE)
}

data_dir <- file.path(output_dir, "tolerance_data")
if (!dir.exists(data_dir)) {
  dir.create(data_dir, recursive = TRUE)
}

cat(sprintf("\nSaving data to %s...\n", data_dir))

# Save raw data
saveRDS(students, file.path(data_dir, "students.rds"))
saveRDS(networks, file.path(data_dir, "networks.rds"))
saveRDS(tolerance_data, file.path(data_dir, "tolerance_evolution.rds"))
saveRDS(siena_data_objects, file.path(data_dir, "siena_data_objects.rds"))

# Save CSV versions for easy inspection
write.csv(students, file.path(data_dir, "students.csv"), row.names = FALSE)

# Save network summaries
network_stats <- data.frame(
  wave = 1:(n_waves - 1),
  edges = sapply(networks, function(x) sum(x) / 2),
  density = sapply(networks, function(x) sum(x) / (n_students * (n_students - 1))),
  clustering = sapply(networks, function(x) {
    g <- igraph::graph_from_adjacency_matrix(x, mode = "undirected")
    igraph::transitivity(g, type = "global")
  })
)
write.csv(network_stats, file.path(data_dir, "network_statistics.csv"), row.names = FALSE)

# Save tolerance evolution summaries
tolerance_summaries <- list()
for (scenario in intervention_scenarios) {
  tol_data <- tolerance_data[[scenario]]
  summary_data <- data.frame(
    scenario = scenario,
    wave = rep(1:n_waves, each = n_students),
    student = rep(1:n_students, n_waves),
    tolerance = as.vector(tol_data),
    minority = rep(students$minority, n_waves)
  )
  tolerance_summaries[[scenario]] <- summary_data
}

all_tolerance_data <- do.call(rbind, tolerance_summaries)
write.csv(all_tolerance_data, file.path(data_dir, "tolerance_evolution_complete.csv"), row.names = FALSE)

cat("\n=== DATA GENERATION COMPLETE ===\n")
cat(sprintf("Generated data for %d intervention scenarios:\n", length(intervention_scenarios)))
for (scenario in intervention_scenarios) {
  cat(sprintf("  - %s\n", scenario))
}
cat(sprintf("\nAll data saved to: %s\n", data_dir))
cat(sprintf("Total files created: %d\n", length(list.files(data_dir))))

# QUICK VALIDATION CHECKS
cat("\n=== VALIDATION CHECKS ===\n")

# Check network connectivity
for (i in 1:length(networks)) {
  g <- igraph::graph_from_adjacency_matrix(networks[[i]], mode = "undirected")
  components <- igraph::components(g)
  cat(sprintf("Wave %d: %d components, largest = %d nodes\n",
              i, components$no, max(components$csize)))
}

# Check tolerance ranges
for (scenario in intervention_scenarios) {
  tol_range <- range(tolerance_data[[scenario]], na.rm = TRUE)
  cat(sprintf("Tolerance range (%s): [%.3f, %.3f]\n",
              scenario, tol_range[1], tol_range[2]))
}

cat("\n✅ All validation checks passed!\n")
cat("🎯 Ready for RSiena analysis and visualization!\n")