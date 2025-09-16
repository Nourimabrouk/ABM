"""
Advanced Visualization Suite for ABM-RSiena Integration Research

This module provides state-of-the-art visualization capabilities for Agent-Based Models
integrated with RSiena, supporting:

- Publication-quality static figures and plots
- Dynamic network animations and temporal visualizations
- Interactive dashboards and exploration tools
- Statistical validation and comparison visualizations
- Multi-scale and advanced analytical visualizations

Author: Delta Agent - Visualization & Interactive Demos Specialist
Project: ABM-RSiena Integration for Statistical Sociology PhD Research
"""

from .static_figures import (
    PublicationPlots,
    NetworkPlots,
    StatisticalPlots
)

from .animations import (
    NetworkEvolutionAnimator,
    AgentBehaviorAnimator,
    ParameterSensitivityAnimator
)

from .interactive import (
    ModelDashboard,
    NetworkExplorer,
    ComparativeAnalysisTool
)

from .advanced import (
    MultiScaleVisualizer,
    TemporalAnalysisViz,
    UncertaintyVisualizer
)

from .utils import (
    ColorSchemes,
    LayoutAlgorithms,
    ExportUtilities
)

__version__ = "1.0.0"
__author__ = "Delta Agent"
__all__ = [
    'PublicationPlots', 'NetworkPlots', 'StatisticalPlots',
    'NetworkEvolutionAnimator', 'AgentBehaviorAnimator', 'ParameterSensitivityAnimator',
    'ModelDashboard', 'NetworkExplorer', 'ComparativeAnalysisTool',
    'MultiScaleVisualizer', 'TemporalAnalysisViz', 'UncertaintyVisualizer',
    'ColorSchemes', 'LayoutAlgorithms', 'ExportUtilities'
]