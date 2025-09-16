# =================================================================
# R INTEGRATION TEST FOR ABM + RSIENA PROJECT
# Test that all R components are working correctly
# =================================================================

library(testthat)

cat("Testing R Integration for ABM Project\n")
cat("=====================================\n\n")

# Test 1: Core R packages
test_that("Core R packages load correctly", {
  expect_true(require(igraph, quietly = TRUE), info = "igraph should load")
  expect_true(require(network, quietly = TRUE), info = "network should load")
  expect_true(require(sna, quietly = TRUE), info = "sna should load")

  cat("✓ Core network packages loaded\n")
})

# Test 2: RSiena functionality
test_that("RSiena package works correctly", {
  expect_true(require(RSiena, quietly = TRUE), info = "RSiena should load")

  # Create test network data
  n_actors <- 10
  n_periods <- 3

  # Generate simple evolving network
  networks <- array(0, dim = c(n_actors, n_actors, n_periods))

  # Period 1: random network
  set.seed(123)
  networks[,,1] <- (matrix(runif(n_actors^2), n_actors, n_actors) > 0.8) * 1
  diag(networks[,,1]) <- 0

  # Period 2: add some edges
  networks[,,2] <- networks[,,1]
  networks[1,2,2] <- 1
  networks[2,1,2] <- 1

  # Period 3: add more edges
  networks[,,3] <- networks[,,2]
  networks[3,4,3] <- 1
  networks[4,3,3] <- 1

  # Test RSiena data creation
  siena_data <- sienaDataCreate(networks)
  expect_s3_class(siena_data, "sienaData")
  expect_equal(siena_data$observations, n_periods)

  cat("✓ RSiena data creation works\n")

  # Test effects
  effects <- getEffects(siena_data)
  expect_s3_class(effects, "sienaEffects")

  cat("✓ RSiena effects creation works\n")
})

# Test 3: Network analysis
test_that("Network analysis functions work", {
  # Create test network with igraph
  g <- igraph::erdos.renyi.game(20, 0.1, directed = TRUE)

  # Test basic metrics
  density <- igraph::edge_density(g)
  expect_gte(density, 0)
  expect_lte(density, 1)

  clustering <- igraph::transitivity(g)
  expect_gte(clustering, 0)
  expect_lte(clustering, 1)

  cat("✓ Network metrics calculation works\n")
})

# Test 4: Tidyverse integration
test_that("Tidyverse packages work", {
  expect_true(require(dplyr, quietly = TRUE))
  expect_true(require(ggplot2, quietly = TRUE))

  # Test basic dplyr operations
  test_data <- data.frame(
    id = 1:10,
    value = rnorm(10),
    group = rep(c("A", "B"), 5)
  )

  result <- test_data %>%
    dplyr::group_by(group) %>%
    dplyr::summarise(
      mean_value = mean(value),
      n = n(),
      .groups = "drop"
    )

  expect_equal(nrow(result), 2)
  expect_true(all(c("group", "mean_value", "n") %in% names(result)))

  cat("✓ Tidyverse operations work\n")
})

# Test 5: Visualization
test_that("Visualization packages work", {
  expect_true(require(ggplot2, quietly = TRUE))
  expect_true(require(ggraph, quietly = TRUE))

  # Create test network plot
  g <- igraph::erdos.renyi.game(10, 0.2)

  # Test that we can create a basic plot
  p <- ggraph(g, layout = "stress") +
    geom_edge_link() +
    geom_node_point() +
    theme_void()

  expect_s3_class(p, "gg")

  cat("✓ Network visualization works\n")
})

# Test 6: Data manipulation
test_that("Data manipulation works", {
  # Test creating longitudinal network data
  n_actors <- 5
  n_periods <- 3

  # Create adjacency matrices
  networks <- replicate(n_periods, {
    matrix(rbinom(n_actors^2, 1, 0.2), n_actors, n_actors)
  }, simplify = FALSE)

  # Remove self-loops
  networks <- lapply(networks, function(net) {
    diag(net) <- 0
    net
  })

  # Convert to edge lists
  edge_lists <- purrr::imap_dfr(networks, ~{
    net <- .x
    edges <- which(net == 1, arr.ind = TRUE)
    if (nrow(edges) > 0) {
      data.frame(
        from = edges[,1],
        to = edges[,2],
        period = .y
      )
    } else {
      data.frame(from = integer(0), to = integer(0), period = integer(0))
    }
  })

  expect_true("data.frame" %in% class(edge_lists))
  if (nrow(edge_lists) > 0) {
    expect_true(all(c("from", "to", "period") %in% names(edge_lists)))
  }

  cat("✓ Network data manipulation works\n")
})

# Test 7: Parallel processing capability
test_that("Parallel processing works", {
  expect_true(require(parallel, quietly = TRUE))
  expect_true(require(future, quietly = TRUE))
  expect_true(require(furrr, quietly = TRUE))

  # Test basic parallel operation
  plan(sequential)  # Use sequential for testing

  test_function <- function(x) {
    Sys.sleep(0.1)  # Simulate work
    x^2
  }

  result <- future_map_dbl(1:5, test_function)
  expected <- c(1, 4, 9, 16, 25)

  expect_equal(result, expected)

  cat("✓ Parallel processing setup works\n")
})

# Integration test: Full workflow
test_that("Complete ABM workflow can be executed", {
  # 1. Create synthetic longitudinal network data
  n_actors <- 8
  n_periods <- 3

  create_evolving_network <- function(n, periods, base_prob = 0.1) {
    networks <- array(0, dim = c(n, n, periods))

    for (t in 1:periods) {
      prob <- base_prob + (t-1) * 0.05  # Increasing density
      net <- matrix(rbinom(n^2, 1, prob), n, n)
      diag(net) <- 0

      # Add transitivity bias
      if (t > 1) {
        prev_net <- networks[,,t-1]
        for (i in 1:n) {
          for (j in 1:n) {
            if (i != j && net[i,j] == 0) {
              # Check for common neighbors
              common <- sum(prev_net[i,] * prev_net[j,])
              if (common > 0 && runif(1) < 0.3) {
                net[i,j] <- 1
              }
            }
          }
        }
      }

      networks[,,t] <- net
    }

    networks
  }

  networks <- create_evolving_network(n_actors, n_periods)

  # 2. Convert to RSiena format
  siena_data <- sienaDataCreate(networks)
  expect_s3_class(siena_data, "sienaData")

  # 3. Create effects
  effects <- getEffects(siena_data)
  effects <- includeEffects(effects, transTrip)  # Transitivity
  expect_s3_class(effects, "sienaEffects")

  # 4. Test algorithm creation (not estimation due to time)
  algorithm <- sienaAlgorithmCreate(
    projname = "test",
    cond = FALSE,
    nsub = 2,  # Minimal for testing
    n3 = 100,  # Minimal for testing
    seed = 12345
  )

  expect_s3_class(algorithm, "sienaAlgorithm")

  # 5. Test network analysis
  final_network <- networks[,,n_periods]
  g <- igraph::graph_from_adjacency_matrix(final_network, mode = "directed")

  metrics <- list(
    density = igraph::edge_density(g),
    transitivity = igraph::transitivity(g),
    n_components = igraph::components(g)$no
  )

  expect_true(all(sapply(metrics, is.finite)))

  cat("✓ Complete ABM workflow test passed\n")
})

# Run all tests
cat("\n=== Running All Tests ===\n")

# Capture test results
test_results <- test_dir(".", reporter = "summary")

# Summary
cat("\n=== Test Summary ===\n")

if (any(sapply(test_results, function(x) length(x$failed) > 0))) {
  cat("❌ Some tests failed. Check output above.\n")
  failed_tests <- test_results[sapply(test_results, function(x) length(x$failed) > 0)]
  cat("Failed tests:\n")
  for (test in failed_tests) {
    cat("  -", test$file, "\n")
  }
} else {
  cat("🎉 All R integration tests passed!\n")
  cat("\nYour R environment is ready for ABM development.\n")
  cat("\nNext steps:\n")
  cat("1. Start developing your models in R/models/\n")
  cat("2. Use RSiena for empirical network analysis\n")
  cat("3. Create publication-ready visualizations\n")
  cat("4. Document your work with R Markdown\n")
}

cat("\n=== Environment Information ===\n")
cat("R version:", R.version.string, "\n")
cat("RSiena version:", as.character(packageVersion("RSiena")), "\n")
cat("Platform:", Sys.info()["sysname"], Sys.info()["machine"], "\n")