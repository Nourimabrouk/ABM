# Agent Instructions for SAOM/RSiena Research Coordination

This file provides specialized instructions for Claude Code agents working collaboratively on the Agent-Based Models (SAOM) research project investigating social norm interventions for interethnic cooperation through tolerance.

## Project Overview

**Research Goal**: Investigate how tolerance interventions spread through social networks and promote sustained interethnic cooperation using Stochastic Actor-Oriented Models (SAOM) in RSiena.

**Data Context**: 5825 observations, 2585 respondents, 105 classes, 3 schools, 3 time waves from German high school intervention study (Shani et al., 2023 follow-up).

**Key Innovation**: Implementing attraction-repulsion social influence mechanisms and complex contagion effects for friend-based tolerance diffusion.

## Agent Specialization Roles

### Agent 1: Data Preparation & Network Construction
**Primary Responsibilities**:
- Clean and validate longitudinal survey data
- Construct friendship and cooperation network matrices
- Prepare RSiena data objects using `sienaDataCreate()`
- Handle missing data appropriately for network analysis
- Generate descriptive statistics and network diagnostics

**Key Files to Work With**:
- `R/data_analysis/data_cleaning.R`
- `R/data_analysis/network_construction.R`
- `data/raw/` - Original survey data
- `data/processed/` - Cleaned datasets
- `data/networks/` - Network matrices

**Critical Tasks**:
- Ensure proper handling of three-wave longitudinal structure
- Validate network data for structural consistency
- Create behavior variables (tolerance, cooperation) in RSiena format
- Generate network descriptive statistics (density, transitivity, etc.)

### Agent 2: RSiena Model Development & Custom Effects
**Primary Responsibilities**:
- Develop core SAOM model specifications
- Implement custom attraction-repulsion effects in C++
- Create complex contagion mechanisms
- Ensure proper model convergence and estimation
- Conduct model selection and goodness-of-fit testing

**Key Files to Work With**:
- `R/siena_models/main_model.R`
- `R/custom_effects/attraction_repulsion.cpp`
- `R/custom_effects/complex_contagion.cpp`
- `R/diagnostics/convergence_testing.R`

**Critical Tasks**:
- Achieve RSiena model convergence (t-ratios < 0.1, overall < 0.25)
- Validate custom effects against theoretical expectations
- Implement proper friend-based influence (not classroom-wide)
- Test multiple model specifications for robustness

### Agent 3: Intervention Scenario Design & Simulation
**Primary Responsibilities**:
- Design realistic intervention scenarios
- Implement different targeting strategies (centrality, popularity, random)
- Run Monte Carlo simulations from fitted models
- Test complex vs. simple contagion scenarios
- Evaluate intervention effectiveness across different designs

**Key Files to Work With**:
- `R/intervention/scenario_design.R`
- `R/intervention/targeting_strategies.R`
- `R/intervention/monte_carlo_simulation.R`
- `configs/intervention_parameters.yaml`

**Critical Tasks**:
- Define realistic intervention intensity and duration parameters
- Implement various targeting algorithms (betweenness, closeness, eigenvector centrality)
- Compare clustered vs. random intervention delivery
- Calculate intervention effect sizes and confidence intervals

### Agent 4: Statistical Analysis & Visualization
**Primary Responsibilities**:
- Analyze intervention effectiveness using appropriate statistical tests
- Generate publication-quality network visualizations
- Create results plots and tables
- Conduct sensitivity analyses
- Document statistical methodology

**Key Files to Work With**:
- `R/visualization/network_plots.R`
- `R/visualization/results_visualization.R`
- `src/visualization/supplementary_plots.py`
- `outputs/figures/`

**Critical Tasks**:
- Create network plots showing tolerance and cooperation levels
- Generate intervention effectiveness comparison plots
- Implement proper statistical testing with multiple comparison corrections
- Produce publication-ready figures at 300+ DPI

## Coordination Protocols

### Communication Standards
- Use clear, descriptive commit messages referencing agent role and task
- Document all model assumptions and parameter choices
- Share intermediate results and diagnostic information
- Report convergence issues and model fitting challenges immediately

### File Naming Conventions
```
R/siena_models/
├── 01_data_preparation.R      # Agent 1
├── 02_model_specification.R   # Agent 2
├── 03_custom_effects.R        # Agent 2
├── 04_model_estimation.R      # Agent 2
├── 05_intervention_scenarios.R # Agent 3
├── 06_simulation_analysis.R   # Agent 3
├── 07_statistical_tests.R     # Agent 4
└── 08_visualization.R         # Agent 4
```

### Quality Assurance Checkpoints
1. **Data Preparation Complete**: Agent 1 confirms clean data and network objects
2. **Model Convergence Achieved**: Agent 2 confirms proper SAOM estimation
3. **Custom Effects Validated**: Agent 2 confirms attraction-repulsion implementation
4. **Intervention Scenarios Designed**: Agent 3 confirms realistic parameter ranges
5. **Statistical Analysis Complete**: Agent 4 confirms effect size calculations

## R Environment Setup (All Agents)

### Essential Commands
```bash
# Install core packages (run once)
"C:\Program Files\R\R-4.5.1\bin\x64\Rscript.exe" -e "install.packages(c('RSiena', 'RSienaTest', 'network', 'sna', 'igraph', 'tidyverse', 'data.table', 'ggplot2', 'ggraph', 'parallel'))"

# Run interactive R
"C:\Program Files\R\R-4.5.1\bin\x64\R.exe"

# Execute specific analysis scripts
"C:\Program Files\R\R-4.5.1\bin\x64\Rscript.exe" R/siena_models/01_data_preparation.R
```

### Required R Packages by Agent
- **Agent 1**: `tidyverse`, `data.table`, `network`, `sna`, `RSiena`
- **Agent 2**: `RSiena`, `RSienaTest`, `network`, `igraph`, `Rcpp`
- **Agent 3**: `RSiena`, `parallel`, `foreach`, `doParallel`
- **Agent 4**: `ggplot2`, `ggraph`, `igraph`, `gridExtra`, `RColorBrewer`

## Common Challenges & Solutions

### RSiena Model Convergence
- **Issue**: Poor convergence (t-ratios > 0.1)
- **Solutions**: Adjust estimation parameters, increase iterations, modify effects
- **Command**: Use `sienaAlgorithmCreate(nsub=8, n3=5000)` for difficult models

### Memory Management
- **Issue**: Large networks causing memory problems
- **Solutions**: Use `gc()` after large objects, process data in chunks
- **Monitor**: Use `pryr::mem_used()` to track memory usage

### Custom Effects Implementation
- **Issue**: C++ compilation errors
- **Solutions**: Test effects incrementally, validate against simple cases
- **Reference**: RSiena manual Section 12 on user-defined effects

## Success Criteria

### Model Quality
- All RSiena models achieve convergence (t-ratios < 0.1)
- Goodness-of-fit tests show acceptable model performance
- Custom effects behave according to theoretical expectations

### Intervention Analysis
- Statistically significant differences between intervention scenarios
- Effect sizes large enough to be practically meaningful
- Robust results across different model specifications

### Reproducibility
- Complete R scripts with documented random seeds
- Clear documentation of all parameter choices
- Automated pipeline from raw data to final results

## Emergency Protocols

### When Models Fail to Converge
1. Check data quality and network structure
2. Simplify model specification temporarily
3. Adjust algorithm parameters (increase nsub, n3)
4. Consider alternative effects or specifications
5. Consult RSiena documentation and support forums

### When Custom Effects Don't Work
1. Test with simple network examples first
2. Validate C++ code compilation
3. Check effect definitions against RSiena standards
4. Consider using existing similar effects as templates

### When Results Don't Make Sense
1. Verify data preprocessing steps
2. Check model specification for errors
3. Validate intervention scenario parameters
4. Review statistical analysis methods
5. Cross-check with theoretical expectations

Remember: This is PhD-level research with high standards for methodological rigor. Every decision should be theoretically justified and empirically validated. Communication between agents is essential for maintaining consistency and quality across the entire analysis pipeline.