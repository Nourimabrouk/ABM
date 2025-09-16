# Agent-Based Models for Statistical Sociology

**PhD Dissertation Research Repository**

This repository contains Agent-Based Model implementations for studying social phenomena through computational sociology methods. The project utilizes parallel AI agents via Claude Code to develop, validate, and analyze sophisticated ABMs for academic research.

## Research Overview

**Objective**: Develop and validate Agent-Based Models for statistical sociology research  
**Level**: PhD Dissertation  
**Field**: Computational Social Science / Statistical Sociology  
**Methods**: Agent-Based Modeling, Statistical Analysis, Parallel Computing  

## Repository Structure

```
ABM/
├── src/
│   ├── models/          # ABM implementations using Mesa framework
│   ├── agents/          # Agent behavior definitions and classes
│   ├── environments/    # Spatial and network environment models
│   ├── analysis/        # Statistical analysis and validation modules
│   ├── visualization/   # Publication-quality plotting and figures
│   └── utils/          # Shared utilities and helper functions
├── data/
│   ├── raw/            # Original datasets and empirical data
│   ├── processed/      # Cleaned and transformed datasets
│   └── simulated/      # Model outputs and simulation results
├── notebooks/          # Jupyter notebooks for exploration and analysis
├── tests/              # Unit tests and model validation tests
├── docs/               # Documentation and methodology
├── configs/            # Model configuration and parameter files
└── outputs/            # Final results, figures, and publications
```

## Quick Start

### Environment Setup

```bash
# Clone the repository
git clone <repository-url>
cd ABM

# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Install development tools
pip install -e .[dev]
```

### Running Models

```bash
# Run a basic ABM simulation
python src/models/run_simulation.py

# Execute statistical analysis
python src/analysis/statistical_analysis.py

# Generate visualizations
python src/visualization/generate_plots.py

# Run test suite
python -m pytest tests/ -v
```

## Key Features

### ABM Framework
- **Mesa Integration**: Built on Mesa ABM framework for robust modeling
- **Modular Design**: Composable agent, environment, and model components
- **Scalable Architecture**: Support for large-scale simulations with thousands of agents
- **Statistical Validation**: Comprehensive model verification and validation protocols

### Research Standards
- **Reproducibility**: All simulations use documented random seeds
- **Academic Rigor**: Publication-ready code with comprehensive documentation
- **Statistical Methods**: Proper hypothesis testing, confidence intervals, and effect sizes
- **Performance Optimization**: Parallel execution and efficient algorithms

### AI Agent Coordination
- **Claude Code Integration**: Multiple AI agents for collaborative development
- **Parallel Development**: Distributed model design and implementation
- **Automated Analysis**: AI-assisted statistical interpretation and validation
- **Quality Assurance**: Automated testing and code review processes

## Dependencies

### Core Libraries
- `mesa`: Agent-Based Modeling framework
- `numpy`, `pandas`, `scipy`: Scientific computing and data analysis
- `networkx`: Network analysis for agent interactions
- `matplotlib`, `seaborn`, `plotly`: Statistical visualization

### Research Tools
- `statsmodels`, `scikit-learn`: Advanced statistical analysis
- `pytest`: Testing framework with coverage reports
- `jupyter`: Interactive development and analysis
- `numba`: Performance optimization for compute-intensive operations

## Development Guidelines

### Code Quality
- Type hints required for all functions
- NumPy-style docstrings for all modules
- 90%+ test coverage for core components
- PEP 8 compliance with academic documentation standards

### Research Standards
- Document all methodological assumptions
- Include statistical significance testing
- Generate publication-ready figures (300+ DPI)
- Maintain reproducible analysis workflows

### Version Control
- Descriptive commit messages with research context
- Feature branches for experimental models
- Tagged releases for paper submissions
- Comprehensive changelog for major versions

## Testing

```bash
# Run all tests
python -m pytest

# Run with coverage report
python -m pytest --cov=src --cov-report=html

# Run specific test categories
python -m pytest -m "statistical"  # Statistical validation tests
python -m pytest -m "integration"  # Full model integration tests
```

## Documentation

Detailed documentation is available in the `docs/` directory:

- **Methodology**: Theoretical foundations and modeling approach
- **API Reference**: Complete function and class documentation  
- **Tutorials**: Step-by-step model development guides
- **Validation**: Model verification and statistical validation protocols

## Contributing

This repository follows academic research standards:

1. All code must include comprehensive tests
2. Statistical methods require validation against known benchmarks
3. Documentation must enable replication by independent researchers
4. Performance optimizations should not compromise model accuracy

## Research Ethics

- All data handling follows institutional research protocols
- Model assumptions and limitations are clearly documented
- Results include appropriate statistical uncertainty quantification
- Code will be made publicly available upon publication

## Contact

For questions about the research methodology or technical implementation, please refer to the documentation or create an issue in the repository.

## Citation

If you use this code in your research, please cite:

```
[Citation will be added upon publication]
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---

**Status**: Active Development  
**Last Updated**: September 2025  
**Python Version**: 3.9+