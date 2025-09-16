# CLAUDE.md

This file provides guidance to Claude Code when working with Agent-Based Models (ABM) for statistical sociology research.

## Repository Overview

This repository contains a PhD dissertation project focused on Stochastic Actor-Oriented Models (SAOM) using RSiena for investigating social norm interventions to promote interethnic cooperation through tolerance. The research utilizes parallel AI agents through Claude Code for state-of-the-art computational social science research.

## Research Context

**Discipline**: Statistical Sociology / Computational Social Science
**Focus**: Social norm interventions for interethnic cooperation using SAOM/ABM
**Objective**: PhD dissertation investigating how tolerance interventions spread through social networks and affect interethnic cooperation
**Primary Tools**: R with RSiena, Python for supplementary analysis, Claude Code coordination
**Data**: 5825 observations, 2585 respondents, 105 classes, 3 schools, 3 time waves

## Research Ambitions (from Full Presentation)

**Core Research Question**: How can individual-level changes in tolerance from interventions spread and persist in social networks to increase sustained interethnic cooperation?

**Key Components**:
1. **Tolerance as Target**: Moving beyond prejudice reduction to promote tolerance (value-based acceptance despite principled disapproval)
2. **Social Network Dynamics**: Using SAOM to model how tolerance spreads through friendship networks via attraction-repulsion mechanisms
3. **Intervention Design**: Testing different targeting strategies (popular actors vs. peripheral actors, centrality measures, complex vs. simple contagion)
4. **Empirical Calibration**: Using real-world intervention data from German high schools (Shani et al., 2023 follow-up study)

**Theoretical Innovation**:
- Attraction-repulsion social influence mechanism among friends (not classroom-wide)
- Complex contagion modeling requiring multiple simultaneous exposures
- Tolerance → cooperation pathway via expanded "radius of trust"

## Development Guidelines

### R Environment and RSiena Commands

**R Installation Path**: `C:\Program Files\R\R-4.5.1\`

```bash
# R environment setup and package installation
"C:\Program Files\R\R-4.5.1\bin\x64\Rscript.exe" -e "install.packages(c('RSiena', 'RSienaTest', 'network', 'sna', 'igraph', 'tidyverse', 'data.table', 'ggplot2', 'ggraph', 'parallel', 'foreach', 'doParallel'))"

# Run R interactively
"C:\Program Files\R\R-4.5.1\bin\x64\R.exe"

# Execute RSiena analysis scripts
"C:\Program Files\R\R-4.5.1\bin\x64\Rscript.exe" R/siena_models/main_analysis.R
"C:\Program Files\R\R-4.5.1\bin\x64\Rscript.exe" R/data_analysis/descriptive_stats.R
"C:\Program Files\R\R-4.5.1\bin\x64\Rscript.exe" R/intervention/scenario_testing.R

# Model diagnostics and convergence checking
"C:\Program Files\R\R-4.5.1\bin\x64\Rscript.exe" R/siena_models/model_diagnostics.R

# Custom effects testing
"C:\Program Files\R\R-4.5.1\bin\x64\Rscript.exe" R/custom_effects/test_attraction_repulsion.R

# Parallel intervention scenario analysis
"C:\Program Files\R\R-4.5.1\bin\x64\Rscript.exe" R/intervention/parallel_scenarios.R --cores 4
```

### Python Support Commands

```bash
# Python environment setup (for supplementary analysis)
python -m venv .venv
.venv\Scripts\activate

# Install Python dependencies
pip install pandas numpy scipy matplotlib seaborn networkx rpy2 jupyter

# R-Python integration testing
python scripts/test_rpy2_integration.py

# Data preprocessing for RSiena
python src/data_prep/prepare_siena_data.py

# Visualization support
python src/visualization/network_plots.py
python src/visualization/intervention_results.py
```

### Code Quality Standards

**Academic Rigor Requirements**:
- All RSiena models must achieve proper convergence (t-ratios < 0.1)
- Code must be reproducible with documented random seeds
- Comprehensive roxygen2 documentation for all R functions
- Statistical validation of model assumptions and goodness-of-fit
- Peer-reviewed methodology following SAOM best practices

**RSiena-Specific Requirements**:
- Use RSienaTest package for model diagnostics and validation
- Implement proper handling of missing data in longitudinal networks
- Custom C++ effects must be tested against known benchmarks
- Parameter estimation should use appropriate algorithms (Method of Moments, Maximum Likelihood)
- Model selection should follow information criteria (AIC, BIC) when appropriate

### SAOM Architecture and Implementation

**Core RSiena Components**:
- `sienaDataCreate()`: Network-behavior data object construction
- `getEffects()`: Effect specification and modification
- `sienaAlgorithmCreate()`: Estimation algorithm configuration
- `siena07()`: Model estimation with convergence monitoring
- `simulate.sienaFit()`: Forward simulation from fitted models

**Custom Effects Requirements**:
- **Attraction-Repulsion Effect**: Friend-based influence with latitude of acceptance
- **Complex Contagion**: Multiple simultaneous exposure threshold effects
- **Tolerance-Cooperation Link**: Behavioral dependence between tolerance and cooperation
- **Intervention Scenarios**: Exogenous shocks to tolerance levels at specific time points

**Implementation Standards**:
- Use sienaDataCreate() for proper data object construction
- Implement modular effect specifications for easy modification
- Support multi-level analysis across classes and schools
- Use parallel processing for parameter sweeps and scenario testing
- Implement comprehensive model diagnostics and goodness-of-fit testing

### Statistical Analysis Requirements

**SAOM-Specific Methodology**:
- Conduct proper RSiena model selection using score tests and Wald tests
- Perform goodness-of-fit assessment using mahalanobisDistance() and plot()
- Implement intervention scenario design with realistic parameter ranges
- Use Monte Carlo simulation for intervention effectiveness testing
- Calculate and report intervention effect sizes with confidence intervals
- Conduct sensitivity analysis across different targeting strategies
- Document model assumptions (Markov property, conditional independence)
- Use network autocorrelation tests for residual analysis
- Implement multilevel analysis accounting for class and school nesting

**Output Standards**:
- Generate publication-ready network plots using ggraph and igraph
- Export RSiena results in multiple formats (CSV, LaTeX tables, RDS objects)
- Create reproducible R Markdown analysis reports
- Include comprehensive model diagnostics and convergence assessments
- Document intervention scenario results with statistical significance testing

### Parallel AI Agent Coordination

**Claude Code Integration for SAOM Research**:
- Use multiple Claude Code instances for different research components
- Coordinate R and Python development across analysis pipeline
- Support distributed intervention scenario testing
- Enable collaborative model development and peer review

**Coordination Patterns**:
- **Data Preparation**: One agent for network construction, another for behavior coding
- **Model Development**: Parallel development of custom effects and main model
- **Intervention Scenarios**: Distributed testing of targeting strategies and timing
- **Analysis and Validation**: Independent model verification and results interpretation
- **Documentation**: Collaborative writing of methodology and results sections

## Repository Structure

```
ABM/
├── R/                         # Primary R analysis directory
│   ├── siena_models/          # Core RSiena SAOM implementations
│   ├── custom_effects/        # Attraction-repulsion and complex contagion effects
│   ├── data_analysis/         # Descriptive statistics and exploratory analysis
│   ├── intervention/          # Intervention scenario testing and simulation
│   ├── diagnostics/           # Model convergence and goodness-of-fit testing
│   └── visualization/         # ggplot2/ggraph network and results plots
├── src/                       # Supporting Python analysis
│   ├── data_prep/            # Data cleaning and RSiena object preparation
│   ├── integration/          # R-Python interface via rpy2
│   ├── visualization/        # Supplementary Python plotting
│   └── utils/               # Helper functions and configuration
├── data/
│   ├── raw/                 # Original survey data (5825 obs, 2585 respondents)
│   ├── processed/           # RSiena-ready network and behavior data
│   ├── networks/            # Friendship and cooperation network matrices
│   └── simulated/           # Monte Carlo simulation outputs
├── internal/                # Project management and planning
│   └── planning/            # Research plans, presentation materials
├── scripts/                 # Analysis pipeline orchestration
├── tests/                   # R and Python test suites
├── docs/                    # Methodology documentation
├── configs/                 # Model specifications and parameter sets
└── outputs/                 # Publication-ready results and figures
```

## Research Workflow

### SAOM Development Cycle
1. **Literature Review**: Review SAOM methodology and tolerance intervention theory
2. **Data Preparation**: Clean longitudinal network-behavior data for RSiena
3. **Model Specification**: Define network and behavior effects using getEffects()
4. **Custom Effects Development**: Implement attraction-repulsion and complex contagion in C++
5. **Model Estimation**: Run siena07() with proper convergence monitoring
6. **Model Validation**: Assess goodness-of-fit and perform diagnostic checks
7. **Intervention Design**: Specify realistic scenario parameters and targeting strategies
8. **Simulation**: Run Monte Carlo simulations using fitted model parameters
9. **Analysis**: Statistical comparison of intervention effectiveness
10. **Documentation**: Write methodology following SAOM best practices

### Reproducibility Standards
- Version control all R scripts, custom effects, and configuration files
- Document exact R version (4.5.1), RSiena version, and all package versions
- Include random seeds for all RSiena estimations and simulations
- Provide detailed documentation of all model effects and parameters
- Create automated R workflows for complete analysis pipeline
- Use session_info() to document computational environment

### Publication Requirements
- Generate network visualizations and results plots at 300+ DPI using ggplot2/ggraph
- Include comprehensive statistical testing of intervention effectiveness
- Document RSiena model specifications, convergence, and goodness-of-fit
- Provide complete R code availability following open science practices
- Follow SAOM reporting standards and computational social science guidelines
- Include detailed description of custom effects and their theoretical justification

## Performance Optimization

### Computational Efficiency
- Profile RSiena model estimation time using system.time() and Rprof()
- Optimize custom C++ effects for computational speed
- Use RSiena's built-in parallel processing capabilities
- Implement batch processing for intervention scenario testing
- Consider using high-performance computing resources for large-scale simulations

### Memory Management
- Monitor memory usage during RSiena estimation (can be memory-intensive)
- Use efficient data.table operations for large longitudinal datasets
- Implement checkpointing for long-running intervention scenario testing
- Clear large RSiena objects using rm() and gc() after model fitting
- Manage R workspace memory efficiently during batch processing

## Quality Assurance

### Testing Strategy
- **Model Convergence Tests**: Verify RSiena models achieve proper convergence
- **Custom Effects Validation**: Test attraction-repulsion and complex contagion against known benchmarks
- **Integration Tests**: Complete SAOM estimation and simulation pipeline
- **Statistical Tests**: Verify intervention scenario outcomes are statistically meaningful
- **Performance Tests**: Ensure estimation scales appropriately with network size
- **Regression Tests**: Detect changes in model behavior across software versions

### Code Review Process
- Peer review for all model implementations
- Statistical validation by domain experts
- Performance benchmarking against baseline implementations
- Documentation review for clarity and completeness

## Research Ethics and Data

### Data Management
- Follow institutional data management policies
- Implement proper anonymization for sensitive data
- Document data provenance and transformation steps
- Backup research data and code regularly

### Open Science Practices
- Prepare code for public release upon publication
- Include comprehensive README and setup instructions
- Provide example datasets and usage demonstrations
- Consider pre-registration of analysis plans

## Notes

**Primary Focus**: Developing rigorous, reproducible Stochastic Actor-Oriented Models using RSiena to investigate social norm interventions for promoting interethnic cooperation through tolerance. This research meets PhD dissertation standards and contributes novel methodology to computational social science.

**Key Success Metrics**:
- **Model Validity**: Proper RSiena convergence, goodness-of-fit, and theoretical consistency
- **Empirical Calibration**: Models fitted to real intervention data from German high schools
- **Intervention Effectiveness**: Statistically significant improvements in interethnic cooperation
- **Methodological Innovation**: Successfully implemented attraction-repulsion and complex contagion effects
- **Reproducibility**: Complete R code pipeline enabling independent replication
- **Academic Impact**: Contribution to tolerance intervention theory and SAOM methodology

**Research Impact Goal**: Provide evidence-based recommendations for designing effective tolerance interventions that can create sustained improvements in interethnic cooperation within educational settings.