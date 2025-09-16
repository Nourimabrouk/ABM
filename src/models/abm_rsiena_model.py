"""
ABM-RSiena Integrated Model

This module implements the core integrated model class that seamlessly combines
Mesa Agent-Based Models with RSiena longitudinal network analysis. This represents
the technical heart of the PhD dissertation methodology.

Key Features:
- Bidirectional data flow: ABM → RSiena → ABM feedback loops
- Temporal alignment between ABM discrete steps and RSiena continuous-time models
- Automated parameter estimation and statistical validation
- Multi-level agent types: individuals, groups, institutions
- Network-dependent behavior change mechanisms

Author: Implementation Agent (Beta)
Created: 2025-09-15
"""

import logging
import warnings
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
import pandas as pd
import networkx as nx
from abc import ABC, abstractmethod

import mesa
from mesa import Agent, Model
from mesa.time import RandomActivation, StagedActivation
from mesa.datacollection import DataCollector

# Import integration modules
from ..rsiena_integration.r_interface import RInterface
from ..rsiena_integration.data_converters import ABMRSienaConverter
from ..rsiena_integration.statistical_estimation import StatisticalEstimator
from ..agents.social_agent import SocialAgent, AgentAttributes
from ..utils.config_manager import ModelConfiguration
from ..utils.performance_profiler import PerformanceProfiler

logger = logging.getLogger(__name__)


@dataclass
class NetworkEvolutionParameters:
    """Parameters controlling network evolution dynamics."""
    # Structural effects
    density_effect: float = -2.0
    reciprocity_effect: float = 2.0
    transitivity_effect: float = 0.5
    three_cycle_effect: float = -0.1

    # Individual effects
    outdegree_activity_effect: float = 0.0
    indegree_popularity_effect: float = 0.0
    outdegree_activity_sqrt_effect: float = 0.0

    # Homophily effects
    age_similarity_effect: float = 0.0
    gender_similarity_effect: float = 0.0
    ses_similarity_effect: float = 0.0

    # Behavior evolution effects
    opinion_linear_shape_effect: float = 0.0
    opinion_quadratic_shape_effect: float = 0.0
    opinion_average_similarity_effect: float = 0.0

    # Co-evolution effects
    network_behavior_effect: float = 0.0
    behavior_network_effect: float = 0.0


@dataclass
class SimulationMetrics:
    """Container for simulation performance metrics."""
    step_time: float = 0.0
    memory_usage: float = 0.0
    network_density: float = 0.0
    convergence_ratio: float = 0.0
    rsiena_goodness_of_fit: float = 0.0


class TemporalAlignment:
    """
    Manages temporal alignment between ABM discrete steps and RSiena continuous-time models.

    This class handles the complex task of bridging between Mesa's discrete-time stepping
    and RSiena's continuous-time Markov chain modeling approach.
    """

    def __init__(self, abm_steps_per_period: int = 50):
        """
        Initialize temporal alignment manager.

        Args:
            abm_steps_per_period: Number of ABM steps that correspond to one RSiena period
        """
        self.abm_steps_per_period = abm_steps_per_period
        self.current_period = 0
        self.current_step_in_period = 0
        self.period_snapshots = []

    def advance_step(self) -> bool:
        """
        Advance one ABM step and check if a new RSiena period has started.

        Returns:
            True if we've moved to a new RSiena period
        """
        self.current_step_in_period += 1

        if self.current_step_in_period >= self.abm_steps_per_period:
            self.current_period += 1
            self.current_step_in_period = 0
            return True
        return False

    def get_period_progress(self) -> float:
        """Get progress through current period (0.0 to 1.0)."""
        return self.current_step_in_period / self.abm_steps_per_period

    def should_collect_snapshot(self) -> bool:
        """Check if we should collect a network snapshot for RSiena."""
        return self.current_step_in_period == 0


class ABMRSienaModel(Model):
    """
    Integrated ABM-RSiena Model for longitudinal social network analysis.

    This model combines Mesa's agent-based modeling framework with RSiena's
    longitudinal network analysis capabilities, enabling:
    - Network-behavior co-evolution modeling
    - Empirical validation against real network data
    - Parameter estimation using Method of Moments
    - Multi-scale temporal dynamics
    """

    def __init__(
        self,
        config: ModelConfiguration,
        enable_rsiena: bool = True,
        performance_monitoring: bool = True
    ):
        """
        Initialize the integrated ABM-RSiena model.

        Args:
            config: Model configuration object
            enable_rsiena: Whether to enable RSiena integration
            performance_monitoring: Whether to enable performance profiling
        """
        super().__init__()

        self.config = config
        self.enable_rsiena = enable_rsiena
        self.performance_monitoring = performance_monitoring

        # Initialize components
        self._initialize_rsiena_integration()
        self._initialize_temporal_alignment()
        self._initialize_performance_monitoring()

        # Model state
        self.network = nx.Graph()
        self.network_snapshots = []
        self.behavior_snapshots = []
        self.current_parameters = NetworkEvolutionParameters()
        self.metrics = SimulationMetrics()

        # Agent management
        self.schedule = StagedActivation(
            self,
            stage_list=["network_formation", "behavior_update", "network_dissolution"],
            shuffle=True
        )

        # Create agents
        self._create_agents()

        # Data collection
        self._setup_data_collection()

        # Initial state
        self.running = True
        self.datacollector.collect(self)

        logger.info(f"ABM-RSiena model initialized with {self.config.n_agents} agents")

    def _initialize_rsiena_integration(self):
        """Initialize RSiena integration components."""
        if not self.enable_rsiena:
            self.r_interface = None
            self.converter = None
            self.statistical_estimator = None
            return

        try:
            self.r_interface = RInterface()
            self.converter = ABMRSienaConverter()
            self.statistical_estimator = StatisticalEstimator(self.r_interface)
            logger.info("RSiena integration initialized successfully")
        except Exception as e:
            logger.warning(f"RSiena integration failed: {e}")
            self.enable_rsiena = False
            self.r_interface = None
            self.converter = None
            self.statistical_estimator = None

    def _initialize_temporal_alignment(self):
        """Initialize temporal alignment system."""
        self.temporal_alignment = TemporalAlignment(
            abm_steps_per_period=self.config.abm_steps_per_period
        )

    def _initialize_performance_monitoring(self):
        """Initialize performance monitoring."""
        if self.performance_monitoring:
            self.profiler = PerformanceProfiler()
        else:
            self.profiler = None

    def _create_agents(self):
        """Create agents based on configuration."""
        for i in range(self.config.n_agents):
            # Create agent attributes
            attributes = self._generate_agent_attributes(i)

            # Create agent
            agent = SocialAgent(
                unique_id=i,
                model=self,
                attributes=attributes,
                behavior_variables=self.config.behavior_variables
            )

            # Add to schedule and network
            self.schedule.add(agent)
            self.network.add_node(i, **attributes.__dict__)

    def _generate_agent_attributes(self, agent_id: int) -> AgentAttributes:
        """Generate attributes for a single agent."""
        np.random.seed(self.config.random_seed + agent_id)

        return AgentAttributes(
            age=np.random.normal(self.config.mean_age, self.config.sd_age),
            gender=np.random.choice(['M', 'F'], p=[0.5, 0.5]),
            socioeconomic_status=np.random.normal(0, 1),
            extroversion=np.random.beta(2, 2),
            academic_performance=np.random.normal(0, 1),
            opinion=np.random.uniform(-1, 1) if 'opinion' in self.config.behavior_variables else None
        )

    def _setup_data_collection(self):
        """Setup comprehensive data collection."""
        self.datacollector = DataCollector(
            model_reporters={
                # Network structure metrics
                "network_density": self._get_network_density,
                "average_degree": self._get_average_degree,
                "clustering_coefficient": self._get_clustering_coefficient,
                "degree_centralization": self._get_degree_centralization,
                "average_path_length": self._get_average_path_length,
                "number_of_components": self._get_number_of_components,

                # Network evolution metrics
                "edge_stability": self._get_edge_stability,
                "jaccard_similarity": self._get_jaccard_similarity,
                "degree_correlation": self._get_degree_correlation,

                # Behavior metrics
                "opinion_variance": lambda m: self._get_behavior_variance("opinion"),
                "opinion_polarization": lambda m: self._get_polarization("opinion"),

                # Performance metrics
                "step_time": lambda m: self.metrics.step_time,
                "memory_usage": lambda m: self.metrics.memory_usage,

                # RSiena integration metrics
                "rsiena_convergence": lambda m: self.metrics.convergence_ratio,
                "rsiena_gof": lambda m: self.metrics.rsiena_goodness_of_fit
            },
            agent_reporters={
                # Agent attributes
                "age": lambda a: a.attributes.age,
                "gender": lambda a: a.attributes.gender,
                "ses": lambda a: a.attributes.socioeconomic_status,
                "extroversion": lambda a: a.attributes.extroversion,

                # Agent behaviors
                "opinion": lambda a: getattr(a, 'opinion', None),

                # Agent network position
                "degree": lambda a: len(a.friends),
                "clustering": lambda a: a.local_clustering_coefficient(),
                "betweenness": lambda a: a.betweenness_centrality(),

                # Agent dynamics
                "friendship_attempts": lambda a: a.friendship_attempts,
                "friendship_successes": lambda a: a.friendship_successes,
                "opinion_changes": lambda a: getattr(a, 'opinion_changes', 0)
            }
        )

    def step(self):
        """Execute one step of the integrated model."""
        if self.profiler:
            self.profiler.start_timer('model_step')

        # Execute agent steps in stages
        self.schedule.step()

        # Check for period transition
        new_period = self.temporal_alignment.advance_step()

        if new_period and self.enable_rsiena:
            self._handle_period_transition()

        # Collect data
        self.datacollector.collect(self)

        # Update metrics
        self._update_metrics()

        if self.profiler:
            self.metrics.step_time = self.profiler.end_timer('model_step')

    def _handle_period_transition(self):
        """Handle transition to new RSiena period."""
        # Store network and behavior snapshots
        self.network_snapshots.append(self.network.copy())
        self._collect_behavior_snapshot()

        # Update RSiena parameters if enough periods have passed
        if len(self.network_snapshots) >= 3:
            self._update_rsiena_parameters()

    def _collect_behavior_snapshot(self):
        """Collect current behavior states for all agents."""
        behavior_data = {}
        for agent in self.schedule.agents:
            agent_behaviors = {}
            for var in self.config.behavior_variables:
                agent_behaviors[var] = getattr(agent, var, None)
            behavior_data[agent.unique_id] = agent_behaviors

        self.behavior_snapshots.append(behavior_data)

    def _update_rsiena_parameters(self):
        """Update model parameters based on RSiena estimation."""
        try:
            # Convert data to RSiena format
            rsiena_data = self.converter.convert_to_rsiena(
                networks=self.network_snapshots[-3:],
                behaviors=self.behavior_snapshots[-3:] if self.behavior_snapshots else None,
                agent_attributes=self._get_agent_attributes_dict()
            )

            # Estimate RSiena model
            results = self.statistical_estimator.estimate_model(
                rsiena_data=rsiena_data,
                effects_specification=self.config.rsiena_effects
            )

            # Update model parameters
            if results['converged']:
                self._update_parameters_from_rsiena(results)
                self.metrics.convergence_ratio = results['max_convergence_ratio']
                logger.info(f"RSiena parameters updated at step {self.schedule.steps}")
            else:
                logger.warning(f"RSiena estimation did not converge at step {self.schedule.steps}")

        except Exception as e:
            logger.error(f"Failed to update RSiena parameters: {e}")

    def _update_parameters_from_rsiena(self, rsiena_results: Dict[str, Any]):
        """Update ABM parameters based on RSiena estimation results."""
        parameter_mapping = {
            'density': 'density_effect',
            'recip': 'reciprocity_effect',
            'transTrip': 'transitivity_effect',
            'cycle3': 'three_cycle_effect'
        }

        for rsiena_param, abm_param in parameter_mapping.items():
            if rsiena_param in rsiena_results['effect_names']:
                idx = rsiena_results['effect_names'].index(rsiena_param)
                new_value = rsiena_results['parameters'][idx]
                setattr(self.current_parameters, abm_param, new_value)

        # Update agents with new parameters
        for agent in self.schedule.agents:
            agent.update_parameters(self.current_parameters)

    def _get_agent_attributes_dict(self) -> Dict[str, List]:
        """Get agent attributes formatted for RSiena."""
        attributes = {}

        for agent in self.schedule.agents:
            for attr_name in ['age', 'gender', 'socioeconomic_status', 'extroversion']:
                if attr_name not in attributes:
                    attributes[attr_name] = []
                attributes[attr_name].append(getattr(agent.attributes, attr_name))

        return attributes

    def _update_metrics(self):
        """Update performance and model metrics."""
        if self.profiler:
            self.metrics.memory_usage = self.profiler.get_memory_usage()

        self.metrics.network_density = self._get_network_density()

    def validate_against_empirical_data(
        self,
        empirical_networks: List[nx.Graph],
        empirical_behaviors: Optional[List[Dict]] = None,
        validation_metrics: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Validate the model against empirical network data.

        Args:
            empirical_networks: List of empirical network snapshots
            empirical_behaviors: List of empirical behavior data
            validation_metrics: Specific metrics to validate

        Returns:
            Comprehensive validation results
        """
        if not self.enable_rsiena:
            raise ValueError("RSiena integration required for empirical validation")

        if validation_metrics is None:
            validation_metrics = [
                'density', 'transitivity', 'degree_distribution',
                'assortativity', 'average_path_length'
            ]

        # Estimate RSiena model on empirical data
        empirical_rsiena = self.converter.convert_to_rsiena(
            networks=empirical_networks,
            behaviors=empirical_behaviors,
            agent_attributes=self._get_empirical_attributes(empirical_networks)
        )

        empirical_results = self.statistical_estimator.estimate_model(
            rsiena_data=empirical_rsiena,
            effects_specification=self.config.rsiena_effects
        )

        # Compare with ABM results
        validation_results = self.statistical_estimator.validate_model(
            abm_networks=self.network_snapshots,
            empirical_results=empirical_results,
            metrics=validation_metrics
        )

        # Calculate goodness of fit
        gof_score = self._calculate_goodness_of_fit(validation_results)
        self.metrics.rsiena_goodness_of_fit = gof_score

        return {
            'validation_metrics': validation_results,
            'empirical_parameters': empirical_results,
            'goodness_of_fit': gof_score,
            'model_comparison': self._compare_models(empirical_results)
        }

    def _get_empirical_attributes(self, networks: List[nx.Graph]) -> Dict[str, List]:
        """Extract attributes from empirical network data."""
        # This would be implemented based on how empirical data is structured
        # For now, return empty dict
        return {}

    def _calculate_goodness_of_fit(self, validation_results: Dict) -> float:
        """Calculate overall goodness of fit score."""
        scores = []
        for metric, results in validation_results.items():
            if isinstance(results, dict) and 'correlation' in results:
                scores.append(abs(results['correlation']))

        return np.mean(scores) if scores else 0.0

    def _compare_models(self, empirical_results: Dict) -> Dict[str, Any]:
        """Compare ABM parameters with empirical estimates."""
        comparison = {}

        # Compare structural effects
        our_params = self.current_parameters
        their_params = empirical_results.get('parameters', [])
        effect_names = empirical_results.get('effect_names', [])

        for i, effect_name in enumerate(effect_names):
            if effect_name in ['density', 'recip', 'transTrip']:
                abm_value = getattr(our_params, f'{effect_name}_effect', None)
                empirical_value = their_params[i] if i < len(their_params) else None

                if abm_value is not None and empirical_value is not None:
                    comparison[effect_name] = {
                        'abm_value': abm_value,
                        'empirical_value': empirical_value,
                        'difference': abs(abm_value - empirical_value),
                        'relative_error': abs(abm_value - empirical_value) / abs(empirical_value)
                    }

        return comparison

    # Network analysis methods
    def _get_network_density(self) -> float:
        """Calculate network density."""
        return nx.density(self.network)

    def _get_average_degree(self) -> float:
        """Calculate average degree."""
        if self.network.number_of_nodes() == 0:
            return 0.0
        return sum(dict(self.network.degree()).values()) / self.network.number_of_nodes()

    def _get_clustering_coefficient(self) -> float:
        """Calculate global clustering coefficient."""
        try:
            return nx.transitivity(self.network)
        except:
            return 0.0

    def _get_degree_centralization(self) -> float:
        """Calculate degree centralization."""
        degrees = [d for n, d in self.network.degree()]
        if len(degrees) < 2:
            return 0.0

        max_degree = max(degrees)
        n = self.network.number_of_nodes()

        numerator = sum(max_degree - d for d in degrees)
        denominator = (n - 1) * (n - 2)

        return numerator / denominator if denominator > 0 else 0.0

    def _get_average_path_length(self) -> float:
        """Calculate average path length."""
        try:
            if nx.is_connected(self.network):
                return nx.average_shortest_path_length(self.network)
            else:
                # Use largest connected component
                components = list(nx.connected_components(self.network))
                if not components:
                    return float('inf')
                largest_component = max(components, key=len)
                subgraph = self.network.subgraph(largest_component)
                return nx.average_shortest_path_length(subgraph)
        except:
            return float('inf')

    def _get_number_of_components(self) -> int:
        """Get number of connected components."""
        return nx.number_connected_components(self.network)

    def _get_edge_stability(self) -> float:
        """Calculate edge stability between consecutive snapshots."""
        if len(self.network_snapshots) < 2:
            return 1.0

        current = set(self.network_snapshots[-1].edges())
        previous = set(self.network_snapshots[-2].edges())

        if not current and not previous:
            return 1.0

        intersection = len(current.intersection(previous))
        union = len(current.union(previous))

        return intersection / union if union > 0 else 0.0

    def _get_jaccard_similarity(self) -> float:
        """Calculate Jaccard similarity between consecutive network snapshots."""
        return self._get_edge_stability()  # Same calculation

    def _get_degree_correlation(self) -> float:
        """Calculate degree correlation between consecutive snapshots."""
        if len(self.network_snapshots) < 2:
            return 1.0

        current_degrees = dict(self.network_snapshots[-1].degree())
        previous_degrees = dict(self.network_snapshots[-2].degree())

        # Get degrees for common nodes
        common_nodes = set(current_degrees.keys()).intersection(previous_degrees.keys())
        if len(common_nodes) < 2:
            return 1.0

        current_vals = [current_degrees[node] for node in common_nodes]
        previous_vals = [previous_degrees[node] for node in common_nodes]

        try:
            correlation = np.corrcoef(current_vals, previous_vals)[0, 1]
            return correlation if not np.isnan(correlation) else 0.0
        except:
            return 0.0

    def _get_behavior_variance(self, behavior: str) -> float:
        """Calculate variance in behavior variable."""
        values = []
        for agent in self.schedule.agents:
            value = getattr(agent, behavior, None)
            if value is not None:
                values.append(value)

        return np.var(values) if values else 0.0

    def _get_polarization(self, behavior: str) -> float:
        """Calculate polarization in behavior variable."""
        values = []
        for agent in self.schedule.agents:
            value = getattr(agent, behavior, None)
            if value is not None:
                values.append(value)

        if len(values) < 2:
            return 0.0

        # Calculate as standard deviation (higher = more polarized)
        return np.std(values)


def run_integrated_simulation(
    config: ModelConfiguration,
    n_steps: int = 1000,
    enable_rsiena: bool = True,
    progress_callback: Optional[Callable[[int, Dict], None]] = None
) -> Tuple[ABMRSienaModel, pd.DataFrame]:
    """
    Run an integrated ABM-RSiena simulation.

    Args:
        config: Model configuration
        n_steps: Number of simulation steps
        enable_rsiena: Whether to enable RSiena integration
        progress_callback: Optional callback for progress updates

    Returns:
        Tuple of (model, collected_data)
    """
    logger.info(f"Starting integrated ABM-RSiena simulation with {n_steps} steps")

    # Create model
    model = ABMRSienaModel(
        config=config,
        enable_rsiena=enable_rsiena,
        performance_monitoring=True
    )

    # Run simulation
    for step in range(n_steps):
        model.step()

        # Progress reporting
        if step % 100 == 0:
            metrics = {
                'step': step,
                'density': model._get_network_density(),
                'avg_degree': model._get_average_degree(),
                'clustering': model._get_clustering_coefficient(),
                'memory_mb': model.metrics.memory_usage,
                'step_time_ms': model.metrics.step_time * 1000
            }

            logger.info(f"Step {step}: Density={metrics['density']:.3f}, "
                       f"AvgDegree={metrics['avg_degree']:.2f}, "
                       f"Clustering={metrics['clustering']:.3f}")

            if progress_callback:
                progress_callback(step, metrics)

    # Get final data
    data = model.datacollector.get_model_vars_dataframe()

    logger.info("Integrated simulation completed successfully")
    logger.info(f"Final metrics: Density={model._get_network_density():.3f}, "
               f"Components={model._get_number_of_components()}")

    return model, data


if __name__ == "__main__":
    # Example usage with configuration
    from ..utils.config_manager import create_default_config

    logging.basicConfig(level=logging.INFO)

    # Create default configuration
    config = create_default_config(n_agents=100)

    # Run simulation
    model, data = run_integrated_simulation(
        config=config,
        n_steps=500,
        enable_rsiena=False  # Disable for quick testing
    )

    # Display results
    print("\nFinal Simulation Results:")
    print(f"Network Density: {model._get_network_density():.3f}")
    print(f"Average Degree: {model._get_average_degree():.2f}")
    print(f"Clustering Coefficient: {model._get_clustering_coefficient():.3f}")
    print(f"Connected Components: {model._get_number_of_components()}")

    # Show data evolution
    key_metrics = ['network_density', 'average_degree', 'clustering_coefficient']
    if not data.empty and any(col in data.columns for col in key_metrics):
        print("\nNetwork Evolution (Last 10 Steps):")
        available_metrics = [col for col in key_metrics if col in data.columns]
        print(data[available_metrics].tail(10))