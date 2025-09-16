# RSiena Tolerance Intervention Research: Demonstration Usage Guide

## Overview

This package contains three comprehensive R demonstrations showcasing the power of RSiena for tolerance intervention research. Each demonstration builds on the previous one, progressing from basic concepts to advanced custom implementations.

## Demonstration Files

### 1. `tolerance_basic_demo.R` - Foundation
**Purpose**: Complete workflow demonstration of basic RSiena usage for tolerance research

**Key Features**:
- Data generation with realistic tolerance dynamics
- Basic RSiena model specification
- Parameter estimation and convergence checking
- Goodness-of-fit testing
- Results interpretation and visualization
- Educational notes and best practices

**Learning Objectives**:
- Understand fundamental RSiena concepts (selection vs. influence)
- Learn proper workflow from data to publication-ready results
- Master convergence diagnostics and model validation
- Practice interpreting tolerance-specific effects

**Runtime**: ~10-15 minutes

### 2. `intervention_simulation_demo.R` - Applied Research
**Purpose**: Complete intervention experiment from hypothesis to policy recommendations

**Key Features**:
- Research hypothesis formulation
- Multiple intervention targeting strategies (central, peripheral, clustered, random)
- Statistical comparison across strategies
- Policy recommendation generation
- Publication-ready visualization suite

**Learning Objectives**:
- Design and execute complete intervention studies
- Compare targeting strategies systematically
- Translate statistical findings into policy guidance
- Create comprehensive research reports

**Runtime**: ~15-20 minutes

### 3. `custom_effects_demo.R` - Advanced Implementation
**Purpose**: Custom C++ effects with complex theoretical mechanisms

**Key Features**:
- Attraction-repulsion influence functions
- Complex contagion threshold models
- Multi-network co-evolution (friendship + cooperation)
- Advanced diagnostic procedures
- Sophisticated visualization techniques

**Learning Objectives**:
- Implement custom theoretical mechanisms
- Handle multi-network co-evolution models
- Perform advanced model validation
- Create publication-quality technical figures

**Runtime**: ~20-30 minutes

## Quick Start Guide

### Prerequisites

```r
# Install required packages
install.packages(c(
    "RSiena", "network", "sna", "igraph",
    "ggplot2", "dplyr", "tidyr", "purrr",
    "RColorBrewer", "gridExtra", "knitr"
))
```

### Running the Demonstrations

#### Option 1: Run Individual Demonstrations
```r
# Basic demonstration
source("tolerance_basic_demo.R")

# Intervention simulation
source("intervention_simulation_demo.R")

# Custom effects
source("custom_effects_demo.R")
```

#### Option 2: Run All Demonstrations
```r
# Run complete demonstration suite
demos <- c(
    "tolerance_basic_demo.R",
    "intervention_simulation_demo.R",
    "custom_effects_demo.R"
)

results <- list()
for (demo in demos) {
    cat("Running", demo, "...\n")
    source(demo)
    # Results automatically saved to .RData files
}
```

#### Option 3: Skip Execution for Code Review
```r
# Set flag to skip execution and review code only
skip_demo <- TRUE
source("tolerance_basic_demo.R")  # Will load functions without running
```

## Expected Outputs

### Files Generated
- `tolerance_basic_demo_results.RData` - Basic demo results
- `intervention_simulation_results.RData` - Intervention study results
- `custom_effects_demo_results.RData` - Custom effects results

### Visualizations Created
- Network evolution plots with tolerance coloring
- Time series of intervention effects
- Strategy comparison charts
- Goodness-of-fit diagnostic plots
- Custom effects validation graphs
- Multi-network co-evolution displays

### Research Products
- Statistical parameter tables
- Policy recommendation reports
- Research methodology documentation
- Technical implementation notes

## Customization Options

### Modify Data Generation Parameters
```r
# In any demo, adjust these parameters:
demo_results <- run_tolerance_basic_demo(
    n_actors = 100,        # Network size
    n_waves = 5,           # Number of time points
    tolerance_sd = 0.4     # Initial tolerance variance
)
```

### Adjust Model Specifications
```r
# Add custom effects to basic model
effects <- includeEffects(effects,
                         gwespFF,         # Geometrically weighted ESP
                         name = "friendship")

# Modify intervention parameters
intervention_data <- generate_intervention_data(
    intervention_proportion = 0.3,    # Target 30% of actors
    effect_size = 0.5,               # Stronger intervention
    decay_rate = 0.05                # Slower decay
)
```

### Custom Visualization Themes
```r
# Modify plot aesthetics
library(ggplot2)
theme_set(theme_minimal() +
    theme(
        text = element_text(size = 12),
        plot.title = element_text(size = 14, face = "bold")
    ))
```

## Troubleshooting

### Common Issues and Solutions

#### Convergence Problems
```r
# If models don't converge, try:
algorithm <- sienaAlgorithmCreate(
    nsub = 6,           # More sub-phases
    n3 = 5000,          # More iterations
    MaxDegree = c(friendship = 10)  # Higher degree ceiling
)
```

#### Memory Issues
```r
# For large networks, reduce memory usage:
gc()  # Garbage collection
options(java.parameters = "-Xmx8g")  # Increase Java heap size
```

#### Visualization Errors
```r
# If plots fail, check:
dev.off()  # Close any open graphics devices
par(mfrow = c(1, 1))  # Reset plot parameters
```

### Getting Help

1. **RSiena Documentation**: `help(package = "RSiena")`
2. **RSiena Manual**: Comprehensive PDF guide available on CRAN
3. **Tolerance Research**: See educational notes in each demo file
4. **Technical Issues**: Check convergence diagnostics and error messages

## Educational Resources

### Key Concepts Covered

#### Statistical Concepts
- **Selection vs. Influence**: Fundamental distinction in social network analysis
- **Co-evolution**: Simultaneous change in networks and behaviors
- **Convergence**: Ensuring reliable parameter estimates
- **Goodness-of-fit**: Validating model adequacy

#### Tolerance Research Concepts
- **Attraction-Repulsion**: Non-linear influence based on attitude similarity
- **Complex Contagion**: Multiple exposure requirements for attitude change
- **Intervention Targeting**: Strategic selection of intervention participants
- **Spillover Effects**: Indirect effects on non-intervention actors

#### Technical Implementation
- **Custom Effects**: Implementing theoretical mechanisms in RSiena
- **Multi-network Models**: Handling multiple relationship types
- **Simulation Studies**: Systematic comparison of intervention strategies
- **Policy Translation**: Converting statistical findings to actionable guidance

### Learning Progression

1. **Beginner**: Start with `tolerance_basic_demo.R`
   - Focus on understanding basic RSiena workflow
   - Practice interpreting tolerance influence effects
   - Master convergence checking and goodness-of-fit

2. **Intermediate**: Progress to `intervention_simulation_demo.R`
   - Learn experimental design for intervention studies
   - Practice systematic strategy comparison
   - Develop policy recommendation skills

3. **Advanced**: Master `custom_effects_demo.R`
   - Implement custom theoretical mechanisms
   - Handle complex multi-network models
   - Create sophisticated research products

### Assessment Questions

#### Basic Level
1. What is the difference between selection and influence in tolerance networks?
2. How do you interpret a positive vs. negative avAlt effect?
3. What does a convergence t-ratio > 0.25 indicate?

#### Intermediate Level
1. Why might central targeting be more effective than random targeting?
2. How do spillover effects differ from direct intervention effects?
3. What factors should inform targeting strategy selection?

#### Advanced Level
1. How would you implement a custom attraction-repulsion effect in C++?
2. What are the computational challenges of multi-network co-evolution models?
3. How do you validate complex contagion mechanisms empirically?

## Research Applications

### Suitable Research Questions
- How does tolerance spread through social networks?
- Which intervention strategies maximize tolerance improvement?
- What role does network structure play in tolerance dynamics?
- How do tolerance and cooperation co-evolve?
- What mechanisms drive attitude polarization vs. consensus?

### Data Requirements
- **Minimum**: Network ties and tolerance measures across 2+ time points
- **Recommended**: 3+ waves, individual attributes, multiple network types
- **Advanced**: High-frequency measurements, intervention data, behavioral outcomes

### Publication Guidelines
- Report all model specifications and convergence diagnostics
- Include goodness-of-fit testing results
- Provide substantive interpretation beyond statistical significance
- Document data collection and measurement procedures
- Consider replication materials and code availability

## Citation and Attribution

When using these demonstrations in research or teaching:

```
These RSiena demonstrations were developed for tolerance intervention research
in statistical sociology. They provide educational workflows for agent-based
network analysis using the RSiena package (Ripley et al., 2024).

For technical RSiena details, cite:
Ripley, R. M., Snijders, T. A. B., Boda, Z., Vörös, A., & Preciado, P. (2024).
Manual for RSiena. University of Oxford, Department of Statistics;
Nuffield College.
```

## Further Development

### Extensions and Modifications
- Implement additional custom effects (reputation, status, etc.)
- Add time-varying covariates for intervention timing analysis
- Extend to multi-group models for population heterogeneity
- Incorporate economic evaluation frameworks
- Develop longitudinal follow-up analysis capabilities

### Contributing
- Report issues or suggest improvements via project documentation
- Share additional tolerance research applications
- Contribute custom effects implementations
- Develop teaching materials based on demonstrations

---

*This usage guide provides comprehensive documentation for using the RSiena tolerance intervention demonstrations. Each demonstration is self-contained and includes extensive educational materials for learning both RSiena methodology and tolerance research applications.*