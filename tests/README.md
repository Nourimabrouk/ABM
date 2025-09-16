# ABM-RSiena Tolerance Intervention Research - Test Suite

## Overview

This comprehensive test suite validates the ABM-RSiena tolerance intervention system for statistical sociology research. The suite ensures research reliability, methodological rigor, and reproducibility for PhD-level dissertation work.

## Test Categories

### 1. RSiena Integration Tests (`test_rsiena_integration.py`)
- **Attraction-repulsion mechanisms**: Validates tolerance similarity effects and influence patterns
- **Complex contagion**: Tests threshold-based behavior change mechanisms
- **Model convergence**: Ensures SAOM estimation converges with t-ratios < 0.1
- **Custom effects**: Validates tolerance-specific RSiena effects implementation
- **Data conversion**: Tests ABM ↔ RSiena data format conversion accuracy

### 2. Intervention Simulation Tests (`test_intervention_simulations.py`)
- **Targeting strategies**: Central, peripheral, random, and clustered intervention targeting
- **Intervention persistence**: Tests tolerance change persistence and decay over time
- **Dose-response**: Validates intervention strength effects and dose-response relationships
- **Spillover effects**: Tests network diffusion and spillover of intervention effects
- **Multi-level effects**: Individual, network, and group-level intervention impacts

### 3. Data Processing Tests (`test_data_processing.py`)
- **Classroom data handling**: Multi-level data structure validation (105 classrooms)
- **RSiena format conversion**: Ensures data compatibility with RSiena requirements
- **Missing data handling**: Tests imputation and listwise deletion strategies
- **Temporal alignment**: Validates synchronization between networks and behavior data
- **Data integrity**: Comprehensive consistency and validity checks

### 4. Statistical Analysis Tests (`test_statistical_analysis.py`)
- **Parameter estimation**: Validates accuracy, convergence, and stability
- **Meta-analysis**: Cross-classroom analysis with fixed and random effects
- **Effect size calculations**: Cohen's d, eta-squared, confidence intervals
- **Significance testing**: Multiple comparison corrections, power analysis
- **Robustness**: Bootstrap analysis and sensitivity testing

### 5. Visualization Tests (`test_visualizations.py`)
- **Network animations**: Rendering accuracy and temporal evolution visualization
- **Interactive dashboards**: Parameter controls, real-time updates, export functionality
- **Publication figures**: High-resolution, vector formats, accessibility features
- **Data correspondence**: Validates visual accuracy against underlying data
- **Export capabilities**: Multiple formats (PNG, PDF, SVG, EPS)

## Quality Assurance Framework

### Validation Metrics
- **Model Convergence Rate**: ≥90% (t-ratios < 0.25 for testing, < 0.1 for production)
- **Parameter Accuracy**: ≥85% within confidence intervals
- **Data Integrity**: ≥95% consistency across checks
- **Visualization Accuracy**: ≥90% data-visual correspondence
- **Statistical Validity**: ≥90% test passage rate
- **Reproducibility**: ≥95% identical results with same seeds

### Quality Standards
- **Research rigor**: PhD dissertation-level methodological standards
- **Statistical validity**: Appropriate tests, effect sizes, confidence intervals
- **Reproducibility**: Fixed random seeds, documented procedures
- **Performance**: Test execution < 5 minutes, memory efficiency
- **Documentation**: Comprehensive API documentation and usage examples

## Usage

### Quick Start
```bash
# Run all tests
python test_runner.py

# Run specific categories
python test_runner.py --categories rsiena_integration statistical_analysis

# Verbose output with detailed logging
python test_runner.py --verbose

# Custom output directory
python test_runner.py --output-dir ./custom_reports
```

### Individual Test Modules
```bash
# Run specific test module
python -m unittest test_rsiena_integration.py -v
python -m unittest test_intervention_simulations.py -v
python -m unittest test_data_processing.py -v
python -m unittest test_statistical_analysis.py -v
python -m unittest test_visualizations.py -v
```

### Continuous Integration
```bash
# CI-friendly run with XML output
python test_runner.py --output-dir ./ci_reports
```

## Dependencies

### Core Requirements
```
numpy>=1.21.0
pandas>=1.3.0
networkx>=2.6.0
scipy>=1.7.0
matplotlib>=3.4.0
```

### RSiena Integration (Optional)
```
rpy2>=3.4.0  # For R interface
```

### Interactive Visualizations (Optional)
```
streamlit>=1.10.0
plotly>=5.0.0
```

### Testing Dependencies
```
unittest2
coverage
pytest  # Alternative test runner
```

## Output Reports

### Comprehensive Test Report (`test_report.html`)
- Executive summary with quality scores
- Detailed test category results
- Quality assurance checklist
- Performance metrics
- Validation recommendations

### Quality Assurance Checklist (`quality_assurance_checklist.md`)
- Research validation checklist
- Compliance verification
- Improvement recommendations
- Publication readiness assessment

### JSON Report (`comprehensive_test_report.json`)
- Machine-readable test results
- Quality metrics data
- Performance benchmarks
- Integration with CI/CD systems

## Test Data

### Synthetic Data Generation
All tests use controlled synthetic data to ensure:
- **Reproducibility**: Fixed random seeds for consistent results
- **Coverage**: Edge cases and boundary conditions
- **Validation**: Known ground truth for accuracy verification
- **Performance**: Optimized data sizes for test efficiency

### Realistic Scenarios
- **Network structures**: Small-world, scale-free, random graphs
- **Tolerance distributions**: Multi-modal, realistic variance
- **Intervention patterns**: Various targeting strategies and intensities
- **Temporal dynamics**: Network evolution and behavior change

## Performance Benchmarks

### Execution Time Targets
- **Individual tests**: < 1 second average
- **Test categories**: < 30 seconds each
- **Full suite**: < 5 minutes total
- **Memory usage**: < 2GB peak

### Scalability Testing
- **Network sizes**: 20-100 actors for testing, 500+ for validation
- **Time periods**: 3-10 periods for testing, 20+ for research
- **Classrooms**: 5 for testing, 105 for full research dataset

## Troubleshooting

### Common Issues

#### RSiena Dependencies
```bash
# Install R and RSiena package
R -e "install.packages('RSiena')"

# Install rpy2 Python interface
pip install rpy2
```

#### Memory Issues
```bash
# Reduce test data size
export TEST_SCALE_FACTOR=0.5

# Monitor memory usage
python test_runner.py --verbose
```

#### Visualization Dependencies
```bash
# Install optional visualization packages
pip install streamlit plotly

# Use non-interactive matplotlib backend
export MPLBACKEND=Agg
```

### Test Debugging
```python
# Enable detailed logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Run individual test methods
python -m unittest test_rsiena_integration.TestRSienaIntegration.test_attraction_repulsion_mechanism -v
```

## Contributing

### Adding New Tests
1. Follow naming convention: `test_[category]_[specific_feature].py`
2. Include comprehensive docstrings and validation
3. Use synthetic data with known ground truth
4. Add performance benchmarks
5. Update quality metrics calculations

### Test Categories
- **Unit tests**: Individual component validation
- **Integration tests**: Cross-component interaction
- **System tests**: End-to-end workflow validation
- **Performance tests**: Execution time and memory usage
- **Regression tests**: Prevent functionality degradation

### Quality Guidelines
- **Test coverage**: Aim for >90% code coverage
- **Assertion clarity**: Descriptive failure messages
- **Data validation**: Comprehensive input/output checks
- **Error handling**: Graceful failure modes
- **Documentation**: Clear test purpose and methodology

## Research Integration

### Academic Standards
- **Methodological rigor**: Validates statistical sociology methods
- **Reproducibility**: Ensures research replicability
- **Documentation**: Publication-ready methodology descriptions
- **Compliance**: Ethics and data management standards

### Publication Support
- **Figure generation**: Publication-quality visualizations
- **Statistical validation**: Comprehensive analysis verification
- **Methodology documentation**: Detailed procedural descriptions
- **Supplementary materials**: Complete test suite as research artifact

---

**Validation Specialist**
*PhD Dissertation Research Support*
*Agent-Based Models for Statistical Sociology*

For questions or support, see the comprehensive test documentation and quality assurance guidelines.