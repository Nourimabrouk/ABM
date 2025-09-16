"""
Test Suite for ABM-RSiena Tolerance Intervention Research

This package contains comprehensive tests for validating the ABM-RSiena
tolerance intervention system for statistical sociology research.

Test Categories:
- RSiena model validation
- Intervention simulation testing
- Data processing validation
- Statistical analysis verification
- Visualization testing
- Performance and integration tests

Author: Validation Specialist
Created: 2025-09-16
"""

import logging
import warnings
from pathlib import Path

# Configure test logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Suppress common warnings during testing
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="rpy2")

# Test data directory
TEST_DATA_DIR = Path(__file__).parent / "data"
TEST_OUTPUT_DIR = Path(__file__).parent / "outputs"

# Create test directories if they don't exist
TEST_DATA_DIR.mkdir(exist_ok=True)
TEST_OUTPUT_DIR.mkdir(exist_ok=True)

__version__ = "1.0.0"
__author__ = "Validation Specialist"