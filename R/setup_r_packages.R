# =================================================================
# R PACKAGE SETUP FOR ABM + RSIENA PROJECT
# Install all required R packages for agent-based modeling
# =================================================================

# Set CRAN mirror for faster downloads
options(repos = c(CRAN = "https://cran.rstudio.com/"))

cat("Starting R package installation for ABM project...\n")
cat("This may take 10-30 minutes depending on your system.\n\n")

# Function to install packages if not already installed
install_if_missing <- function(packages, from_cran = TRUE) {
  for (pkg in packages) {
    if (!require(pkg, character.only = TRUE, quietly = TRUE)) {
      cat("Installing", pkg, "...\n")
      if (from_cran) {
        install.packages(pkg, dependencies = TRUE)
      } else {
        # For packages not on CRAN
        if (pkg == "RSienaTest") {
          devtools::install_github("stocnet/RSienaTest")
        }
      }

      # Load the package
      if (require(pkg, character.only = TRUE, quietly = TRUE)) {
        cat("✓", pkg, "installed successfully\n")
      } else {
        cat("✗ Failed to install", pkg, "\n")
      }
    } else {
      cat("✓", pkg, "already installed\n")
    }
  }
}

# Core R packages for ABM and RSiena work
cat("=== Installing Core ABM Packages ===\n")
core_packages <- c(
  # RSiena ecosystem
  "RSiena",
  "sienaGOF",

  # Network analysis
  "igraph",
  "network",
  "sna",
  "intergraph",
  "networkDynamic",
  "ndtv",

  # Agent-based modeling
  "NetLogoR",
  "SpaDES"
)

install_if_missing(core_packages)

cat("\n=== Installing Statistical Packages ===\n")
stats_packages <- c(
  # Statistical modeling
  "brms",
  "rstanarm",
  "lme4",
  "nlme",
  "survival",
  "broom",

  # Bayesian analysis
  "bayesplot",
  "rstantools",
  "loo",

  # Additional statistical tools
  "psych",
  "Hmisc",
  "corrplot"
)

install_if_missing(stats_packages)

cat("\n=== Installing Tidyverse and Data Manipulation ===\n")
tidyverse_packages <- c(
  "tidyverse",
  "dplyr",
  "tidyr",
  "purrr",
  "readr",
  "tibble",
  "stringr",
  "forcats",
  "lubridate",
  "janitor",
  "skimr",
  "glue"
)

install_if_missing(tidyverse_packages)

cat("\n=== Installing Visualization Packages ===\n")
viz_packages <- c(
  "ggplot2",
  "ggraph",
  "visNetwork",
  "plotly",
  "gganimate",
  "patchwork",
  "ggthemes",
  "scales",
  "viridis",
  "RColorBrewer",
  "cowplot",
  "gridExtra"
)

install_if_missing(viz_packages)

cat("\n=== Installing Spatial Analysis Packages ===\n")
spatial_packages <- c(
  "sf",
  "sp",
  "raster",
  "leaflet",
  "mapview",
  "tmap"
)

install_if_missing(spatial_packages)

cat("\n=== Installing Performance and Parallel Computing ===\n")
performance_packages <- c(
  "parallel",
  "doParallel",
  "foreach",
  "future",
  "furrr",
  "Rcpp",
  "RcppArmadillo",
  "data.table"
)

install_if_missing(performance_packages)

cat("\n=== Installing Development and Testing ===\n")
dev_packages <- c(
  "devtools",
  "usethis",
  "testthat",
  "roxygen2",
  "pkgdown",
  "remotes"
)

install_if_missing(dev_packages)

cat("\n=== Installing Documentation and Reproducibility ===\n")
repro_packages <- c(
  "rmarkdown",
  "knitr",
  "bookdown",
  "renv",
  "here",
  "targets",
  "tarchetypes"
)

install_if_missing(repro_packages)

cat("\n=== Installing Utilities ===\n")
utility_packages <- c(
  "jsonlite",
  "httr",
  "rvest",
  "xml2",
  "openxlsx",
  "readxl",
  "haven",
  "DBI",
  "RSQLite"
)

install_if_missing(utility_packages)

# Install development versions from GitHub (optional)
cat("\n=== Installing Development Packages (Optional) ===\n")
if (require("devtools", quietly = TRUE)) {
  tryCatch({
    install_if_missing("RSienaTest", from_cran = FALSE)
  }, error = function(e) {
    cat("Could not install RSienaTest from GitHub:", e$message, "\n")
  })
}

# Test RSiena installation specifically
cat("\n=== Testing RSiena Installation ===\n")
if (require("RSiena", quietly = TRUE)) {
  cat("✓ RSiena is installed and working!\n")
  cat("RSiena version:", as.character(packageVersion("RSiena")), "\n")

  # Test basic RSiena functionality
  tryCatch({
    # Create a simple test network
    test_net <- array(c(
      c(0, 1, 0, 0),
      c(1, 0, 1, 1),
      c(0, 1, 0, 1),
      c(0, 1, 1, 0)
    ), dim = c(4, 4, 1))

    # Create RSiena data object
    test_data <- sienaDataCreate(test_net)
    cat("✓ RSiena data creation test passed\n")

  }, error = function(e) {
    cat("⚠ RSiena basic functionality test failed:", e$message, "\n")
  })

} else {
  cat("✗ RSiena installation failed. Please check error messages above.\n")
}

# Summary
cat("\n=== Installation Summary ===\n")
installed_packages <- installed.packages()[,"Package"]
required_packages <- c(core_packages, stats_packages, tidyverse_packages,
                      viz_packages, spatial_packages, performance_packages,
                      dev_packages, repro_packages, utility_packages)

missing_packages <- setdiff(required_packages, installed_packages)

if (length(missing_packages) == 0) {
  cat("🎉 All required packages installed successfully!\n")
} else {
  cat("⚠ The following packages could not be installed:\n")
  for (pkg in missing_packages) {
    cat("  -", pkg, "\n")
  }
}

cat("\nR environment setup complete!\n")
cat("Next steps:\n")
cat("1. Restart your R session\n")
cat("2. Test the integration with: source('test_r_integration.R')\n")
cat("3. Start developing your ABM models!\n")