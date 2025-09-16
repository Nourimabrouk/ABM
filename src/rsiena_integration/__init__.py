"""
RSiena Integration Package for ABM-RSiena Framework

This package provides comprehensive integration between Mesa Agent-Based Models
and RSiena (Simulation Investigation for Empirical Network Analysis) for
longitudinal social network analysis.

Key Components:
- r_interface: Python-R bridge for RSiena communication
- data_converters: Data transformation between ABM and RSiena formats
- statistical_estimation: RSiena parameter estimation and model fitting
- validation: Model validation and goodness-of-fit testing

Author: Beta Agent - Implementation Specialist
Created: 2025-09-15
"""

from .r_interface import RInterface
from .data_converters import ABMRSienaConverter
from .statistical_estimation import StatisticalEstimator

__all__ = [
    'RInterface',
    'ABMRSienaConverter',
    'StatisticalEstimator'
]