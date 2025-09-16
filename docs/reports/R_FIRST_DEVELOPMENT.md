# R-First Development Guide for Agent-Based Models

## Overview

This document provides comprehensive instructions for Claude Code agents to work primarily in R for agent-based modeling projects, especially when using the RSiena framework. This approach leverages R's statistical sophistication while maintaining modern AI-assisted development workflows.

## Development Philosophy

### When to Use R vs Python

**Use R primarily for:**
- RSiena network analysis and longitudinal models
- Statistical modeling and hypothesis testing
- Agent-based models requiring sophisticated statistical foundations
- Data visualization with publication-quality outputs (ggplot2)
- Social network analysis with igraph/sna
- Bayesian analysis with brms/Stan
- Reproducible research with R Markdown

**Use Python for:**
- Large-scale data processing and ETL
- Integration with external APIs and services
- Performance-critical simulation components
- Machine learning pipelines
- Web scraping and data collection

### Modern R Development Workflow

1. **Primary Environment**: R with VSCode R extension
2. **AI Assistance**: Claude Code for R development and debugging
3. **Version Control**: Git with proper R/.gitignore
4. **Package Management**: renv for reproducible environments
5. **Documentation**: R Markdown for literate programming
6. **Testing**: testthat for unit tests
7. **Performance**: Rcpp for critical performance sections

## R Environment Setup

### Core R Packages for ABM and RSiena Work

```r
# Essential ABM packages
essential_packages <- c(
  # RSiena ecosystem
  "RSiena",
  "RSienaTest",
  "sienaGOF",

  # Network analysis
  "igraph",
  "network",
  "sna",
  "intergraph",
  "networkDynamic",
  "ndtv",  # Network dynamic visualization

  # Agent-based modeling
  "NetLogoR",  # R implementation inspired by NetLogo
  "RNetLogo",  # Interface to NetLogo
  "SpaDES",    # Spatial discrete event simulation

  # Statistical modeling
  "brms",      # Bayesian regression models
  "rstanarm",  # Applied regression modeling via Stan
  "lme4",      # Linear mixed-effects models
  "nlme",      # Nonlinear mixed-effects models
  "survival",  # Survival analysis

  # Data manipulation and tidyverse
  "tidyverse", # Comprehensive data science packages
  "dplyr",
  "tidyr",
  "purrr",
  "readr",
  "tibble",
  "stringr",
  "forcats",

  # Visualization
  "ggplot2",
  "ggraph",    # Grammar of graphics for networks
  "visNetwork", # Interactive network visualization
  "plotly",    # Interactive plots
  "gganimate", # Animated plots
  "patchwork", # Combining plots
  "ggthemes",  # Additional themes
  "scales",
  "viridis",   # Color palettes

  # Spatial analysis
  "sf",        # Simple features for spatial data
  "sp",        # Spatial data classes
  "raster",    # Raster data
  "leaflet",   # Interactive maps

  # Parallel computing
  "parallel",
  "doParallel",
  "foreach",
  "future",
  "furrr",     # Parallel purrr functions

  # Performance
  "Rcpp",
  "RcppArmadillo",
  "data.table", # Fast data manipulation

  # Reproducibility
  "renv",      # Package management
  "here",      # Path management
  "targets",   # Pipeline management

  # Testing and development
  "testthat",
  "devtools",
  "usethis",
  "pkgdown",

  # Documentation
  "rmarkdown",
  "knitr",
  "bookdown",
  "blogdown",

  # Utilities
  "janitor",   # Data cleaning
  "skimr",     # Data summary
  "broom",     # Tidy model outputs
  "glue",      # String interpolation
  "lubridate", # Date handling
  "jsonlite",  # JSON handling
  "httr",      # HTTP requests
  "rvest"      # Web scraping
)

# Install missing packages
install_if_missing <- function(packages) {
  for (pkg in packages) {
    if (!require(pkg, character.only = TRUE, quietly = TRUE)) {
      cat("Installing", pkg, "...\n")
      install.packages(pkg, dependencies = TRUE)
      library(pkg, character.only = TRUE)
    }
  }
}

install_if_missing(essential_packages)
```

### Project Structure for R-First ABM Development

```
ABM/
├── R/                          # R source code
│   ├── models/                 # ABM model implementations
│   │   ├── social_network_abm.R
│   │   ├── rsiena_models.R
│   │   └── validation_models.R
│   ├── analysis/               # Statistical analysis scripts
│   │   ├── network_analysis.R
│   │   ├── hypothesis_testing.R
│   │   └── sensitivity_analysis.R
│   ├── visualization/          # Plotting and visualization
│   │   ├── network_plots.R
│   │   ├── time_series_plots.R
│   │   └── publication_figures.R
│   └── utils/                  # Utility functions
│       ├── data_processing.R
│       ├── model_helpers.R
│       └── simulation_runners.R
├── data/                       # Data directory
│   ├── raw/                    # Raw empirical data
│   ├── processed/              # Cleaned data
│   └── simulated/              # Model outputs
├── reports/                    # R Markdown reports
│   ├── main_analysis.Rmd
│   ├── model_validation.Rmd
│   └── supplementary_analysis.Rmd
├── tests/                      # Test files
│   └── testthat/
├── renv/                       # renv package management
├── src/                        # Python support code (when needed)
├── notebooks/                  # Jupyter notebooks for Python integration
├── configs/                    # Configuration files
├── outputs/                    # Generated outputs
├── .Rprofile                   # R startup configuration
├── renv.lock                   # Package versions lock file
├── DESCRIPTION                 # Project metadata
└── ABM.Rproj                  # RStudio project file
```

## Claude Code Agent Instructions

### Primary R Development Workflow

When working on ABM projects, Claude Code agents should:

1. **Default to R**: Unless specifically requested otherwise or for tasks where Python is clearly superior
2. **Use RSiena Framework**: For longitudinal network analysis and social influence models
3. **Follow tidyverse conventions**: Use modern R practices with pipe operators and tidy data principles
4. **Write literate code**: Include comprehensive documentation and comments
5. **Create reproducible analysis**: Use R Markdown for reports and analysis
6. **Test thoroughly**: Write unit tests using testthat
7. **Optimize when needed**: Use Rcpp for performance-critical sections

### R Code Style Guidelines

```r
# Use tidyverse style with pipes
network_metrics <- empirical_data %>%
  mutate(
    density = map_dbl(networks, ~graph.density(.x)),
    clustering = map_dbl(networks, ~transitivity(.x)),
    centralization = map_dbl(networks, ~centralization.degree(.x)$centralization)
  ) %>%
  pivot_longer(
    cols = c(density, clustering, centralization),
    names_to = "metric",
    values_to = "value"
  )

# Function documentation with roxygen2 style
#' Estimate RSiena Model with Robust Error Handling
#'
#' @param siena_data A sienaData object created with sienaDataCreate
#' @param effects A sienaEffects object with model specification
#' @param algorithm_settings List of algorithm parameters
#' @return List containing estimation results and diagnostics
#' @export
estimate_rsiena_robust <- function(siena_data, effects, algorithm_settings = NULL) {
  # Implementation
}
```

### RSiena-Specific Development Patterns

```r
# Standard RSiena workflow pattern
create_rsiena_analysis <- function(network_data, actor_attributes = NULL) {

  # 1. Data preparation
  siena_data <- sienaDataCreate(
    networks = network_data,
    attributes = actor_attributes
  )

  # 2. Effects specification
  effects <- getEffects(siena_data)
  effects <- includeEffects(effects, transTrip, cycle3)

  # 3. Algorithm specification
  algorithm <- sienaAlgorithmCreate(
    projname = "analysis",
    cond = FALSE,
    nsub = 4,
    n3 = 1000,
    seed = 12345
  )

  # 4. Estimation with error handling
  results <- tryCatch({
    siena07(algorithm, data = siena_data, effects = effects)
  }, error = function(e) {
    warning("RSiena estimation failed: ", e$message)
    NULL
  })

  # 5. Return structured results
  list(
    data = siena_data,
    effects = effects,
    algorithm = algorithm,
    results = results,
    converged = !is.null(results) && results$OK
  )
}
```

### Integration with Python Components

When Python integration is needed:

```r
# Use reticulate for Python integration
library(reticulate)

# Configure Python environment
use_virtualenv("ABM/.venv")

# Import Python modules
mesa <- import("mesa")
networkx <- import("networkx")

# Convert between R and Python objects
r_to_py_network <- function(igraph_net) {
  # Convert igraph to adjacency matrix
  adj_matrix <- as_adjacency_matrix(igraph_net, sparse = FALSE)

  # Create NetworkX graph in Python
  nx_graph <- networkx$from_numpy_array(r_to_py(adj_matrix))

  return(nx_graph)
}
```

## Development Commands and Shortcuts

### Essential R Commands for ABM Work

```r
# Project setup
usethis::create_project("ABM")
renv::init()

# Package development
devtools::load_all()
devtools::test()
devtools::check()

# Data exploration
skimr::skim(data)
janitor::tabyl(data, variable)

# Network analysis
igraph::graph_from_data_frame(edges, vertices = nodes)
igraph::plot.igraph(network, layout = layout_with_fr)

# RSiena workflow
sienaDataCreate(networks, attributes)
getEffects(siena_data)
siena07(algorithm, data = siena_data, effects = effects)

# Parallel processing
plan(multisession)
future_map(data_list, analysis_function)

# Reproducible reporting
rmarkdown::render("report.Rmd")
targets::tar_make()
```

### Performance Optimization

```r
# Use data.table for large datasets
library(data.table)
dt <- as.data.table(large_dataset)
dt[, metric := calculate_metric(variable), by = group]

# Parallel processing with future
library(future)
library(furrr)

plan(multisession, workers = parallel::detectCores() - 1)
results <- future_map(parameter_combinations, run_simulation)

# Rcpp for critical loops
Rcpp::cppFunction('
  NumericVector fast_calculation(NumericVector x) {
    int n = x.size();
    NumericVector result(n);

    for(int i = 0; i < n; i++) {
      result[i] = complex_calculation(x[i]);
    }

    return result;
  }
')
```

## Testing Strategy for R ABM Projects

```r
# testthat structure
test_that("RSiena data creation works correctly", {
  # Setup test data
  test_networks <- create_test_networks()

  # Test data creation
  siena_data <- sienaDataCreate(test_networks)

  # Assertions
  expect_s3_class(siena_data, "sienaData")
  expect_equal(siena_data$observations, length(test_networks))
})

# Integration tests
test_that("Full RSiena workflow completes", {
  skip_on_cran()  # Long-running test

  results <- create_rsiena_analysis(test_data)
  expect_true(results$converged)
  expect_lt(results$results$tconv.max, 0.1)  # Convergence criterion
})
```

## Publication-Ready Visualization

```r
# Network evolution visualization
library(ggraph)
library(gganimate)

create_network_evolution_plot <- function(networks, layout_type = "stress") {

  # Prepare data for animation
  network_data <- networks %>%
    imap_dfr(~{
      graph_df <- as_data_frame(.x, what = "both")
      graph_df$vertices$time <- .y
      graph_df$edges$time <- .y
      list(vertices = graph_df$vertices, edges = graph_df$edges)
    })

  # Create animated plot
  p <- ggraph(networks[[1]], layout = layout_type) +
    geom_edge_link(alpha = 0.5, color = "gray50") +
    geom_node_point(size = 3, alpha = 0.8) +
    theme_graph() +
    labs(
      title = "Network Evolution Over Time: Period {closest_state}",
      subtitle = "Agent-Based Social Network Model"
    ) +
    transition_states(time) +
    ease_aes("linear")

  return(p)
}

# Publication theme
theme_publication <- function() {
  theme_minimal() +
    theme(
      text = element_text(family = "Arial", size = 12),
      plot.title = element_text(size = 14, face = "bold"),
      axis.text = element_text(size = 10),
      legend.position = "bottom",
      panel.grid.minor = element_blank()
    )
}
```

## Research Reproducibility

### R Markdown Template for ABM Analysis

```r
---
title: "Agent-Based Model Analysis with RSiena"
author: "Research Team"
date: "`r Sys.Date()`"
output:
  html_document:
    toc: true
    code_folding: hide
    theme: flatly
bibliography: references.bib
---

```{r setup, include=FALSE}
knitr::opts_chunk$set(
  echo = TRUE,
  warning = FALSE,
  message = FALSE,
  fig.width = 10,
  fig.height = 6,
  dpi = 300
)

library(targets)
library(here)
source(here("R", "utils", "analysis_functions.R"))
```

# Abstract

Brief description of the agent-based model and research questions.

# Methods

## Model Specification

```{r model-description}
# Describe the ABM model parameters and structure
```

## RSiena Analysis

```{r rsiena-analysis}
# Load and analyze empirical data with RSiena
tar_load(rsiena_results)
print(rsiena_results$results)
```

# Results

```{r main-results, fig.cap="Network evolution comparison"}
# Create comparison plots
create_comparison_plots(abm_results, empirical_results)
```
```

This R-first approach ensures that your agent-based modeling work leverages R's statistical sophistication while maintaining modern development practices with Claude Code assistance.