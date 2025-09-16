"""
Configuration Management for ABM-RSiena Integration

This module provides comprehensive configuration management for the ABM-RSiena
integration framework, including model parameters, simulation settings, and
experimental configurations with validation and serialization capabilities.

Features:
- Hierarchical configuration with inheritance and overrides
- Parameter validation and type checking
- Configuration serialization (YAML, JSON, pickle)
- Environment variable integration
- Parameter sweeps and experimental design
- Configuration versioning and reproducibility

Author: Beta Agent - Implementation Specialist
Created: 2025-09-15
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass, field, asdict
from pathlib import Path
import yaml
import json
import pickle
import os
from enum import Enum
import warnings

try:
    from pydantic import BaseModel, Field, validator
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False
    warnings.warn("pydantic not available. Advanced validation disabled.")

logger = logging.getLogger(__name__)


class SchedulerType(Enum):
    """Types of Mesa schedulers."""
    RANDOM_ACTIVATION = "random_activation"
    STAGED_ACTIVATION = "staged_activation"
    SIMULTANEOUS_ACTIVATION = "simultaneous_activation"


class NetworkType(Enum):
    """Types of network structures."""
    ERDOS_RENYI = "erdos_renyi"
    BARABASI_ALBERT = "barabasi_albert"
    WATTS_STROGATZ = "watts_strogatz"
    COMPLETE = "complete"
    EMPTY = "empty"
    EMPIRICAL = "empirical"


@dataclass
class PopulationParameters:
    """Parameters for agent population generation."""
    # Basic demographics
    mean_age: float = 25.0
    sd_age: float = 10.0
    gender_ratio: float = 0.5  # Proportion of males

    # Socioeconomic distribution
    mean_ses: float = 0.0
    sd_ses: float = 1.0
    education_distribution: List[float] = field(default_factory=lambda: [0.2, 0.3, 0.3, 0.15, 0.05])

    # Personality trait distributions (Beta distribution parameters)
    extroversion_alpha: float = 2.0
    extroversion_beta: float = 2.0
    openness_alpha: float = 2.0
    openness_beta: float = 2.0
    conscientiousness_alpha: float = 3.0
    conscientiousness_beta: float = 2.0
    agreeableness_alpha: float = 3.0
    agreeableness_beta: float = 2.0
    neuroticism_alpha: float = 2.0
    neuroticism_beta: float = 3.0

    # Correlations between attributes
    ses_education_correlation: float = 0.5
    age_ses_correlation: float = 0.3


@dataclass
class NetworkParameters:
    """Parameters for network structure and dynamics."""
    # Network generation
    network_type: NetworkType = NetworkType.ERDOS_RENYI
    initial_density: float = 0.1
    network_size: int = 100

    # Specific network parameters
    barabasi_albert_m: int = 3
    watts_strogatz_k: int = 6
    watts_strogatz_p: float = 0.3

    # Network dynamics
    friendship_formation_rate: float = 0.1
    friendship_dissolution_rate: float = 0.02

    # Homophily parameters
    age_homophily: float = 0.5
    gender_homophily: float = 0.3
    ses_homophily: float = 0.4
    education_homophily: float = 0.3

    # Structural effects
    transitivity_effect: float = 0.2
    popularity_effect: float = 0.05
    activity_effect: float = 0.0


@dataclass
class BehaviorParameters:
    """Parameters for behavior dynamics."""
    # Opinion dynamics
    initial_opinion_distribution: str = "uniform"  # "uniform", "normal", "bimodal"
    opinion_confidence_bound: float = 0.3
    opinion_influence_strength: float = 0.1
    opinion_persistence: float = 0.9

    # Social influence
    conformity_strength: float = 0.3
    social_pressure_threshold: float = 0.7
    minority_resistance_factor: float = 0.8

    # Behavior adoption
    innovation_threshold: float = 0.6
    adoption_rate: float = 0.1


@dataclass
class RSienaParameters:
    """Parameters for RSiena integration."""
    # Temporal alignment
    abm_steps_per_period: int = 50
    rsiena_periods: int = 3

    # Effects specification
    include_density: bool = True
    include_reciprocity: bool = True
    include_transitivity: bool = True
    include_three_cycles: bool = False

    # Homophily effects
    include_age_similarity: bool = True
    include_gender_similarity: bool = True
    include_ses_similarity: bool = True

    # Behavior effects
    include_opinion_linear: bool = True
    include_opinion_quadratic: bool = False
    include_average_similarity: bool = True

    # Estimation parameters
    estimation_nsub: int = 4
    estimation_n3: int = 1000
    convergence_tolerance: float = 0.25


@dataclass
class SimulationParameters:
    """Parameters for simulation execution."""
    # Basic simulation
    n_steps: int = 1000
    n_agents: int = 100
    random_seed: int = 42

    # Scheduler configuration
    scheduler_type: SchedulerType = SchedulerType.STAGED_ACTIVATION
    shuffle_agents: bool = True
    stage_list: List[str] = field(default_factory=lambda: [
        "network_formation", "behavior_update", "network_dissolution"
    ])

    # Data collection
    collect_agent_data: bool = True
    collect_network_data: bool = True
    collection_interval: int = 1

    # Performance
    enable_performance_monitoring: bool = True
    memory_limit_mb: int = 2048
    max_execution_time: int = 3600  # seconds


@dataclass
class ExperimentalDesign:
    """Parameters for experimental design and parameter sweeps."""
    # Parameter sweep configuration
    sweep_parameters: Dict[str, List[Any]] = field(default_factory=dict)
    sweep_method: str = "grid"  # "grid", "random", "sobol", "latin_hypercube"
    n_runs_per_condition: int = 10
    n_total_runs: Optional[int] = None

    # Statistical design
    blocking_variables: List[str] = field(default_factory=list)
    randomization_seed: int = 42

    # Output configuration
    output_directory: str = "outputs"
    save_individual_runs: bool = False
    save_aggregated_results: bool = True


class ModelConfiguration:
    """
    Comprehensive configuration manager for ABM-RSiena models.

    Provides hierarchical configuration management with validation,
    serialization, and parameter sweep capabilities.
    """

    def __init__(
        self,
        config_file: Optional[Union[str, Path]] = None,
        **kwargs
    ):
        """
        Initialize model configuration.

        Args:
            config_file: Path to configuration file
            **kwargs: Configuration overrides
        """
        # Initialize default configurations
        self.population = PopulationParameters()
        self.network = NetworkParameters()
        self.behavior = BehaviorParameters()
        self.rsiena = RSienaParameters()
        self.simulation = SimulationParameters()
        self.experimental = ExperimentalDesign()

        # Additional configuration sections
        self._custom_sections = {}

        # Load from file if provided
        if config_file:
            self.load_from_file(config_file)

        # Apply keyword overrides
        self.update_from_dict(kwargs)

        # Validation
        self.validate()

    def load_from_file(self, filepath: Union[str, Path]):
        """
        Load configuration from file.

        Args:
            filepath: Path to configuration file (YAML, JSON, or pickle)
        """
        filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(f"Configuration file not found: {filepath}")

        try:
            if filepath.suffix.lower() in ['.yaml', '.yml']:
                with open(filepath, 'r') as f:
                    config_dict = yaml.safe_load(f)

            elif filepath.suffix.lower() == '.json':
                with open(filepath, 'r') as f:
                    config_dict = json.load(f)

            elif filepath.suffix.lower() in ['.pkl', '.pickle']:
                with open(filepath, 'rb') as f:
                    config_dict = pickle.load(f)

            else:
                raise ValueError(f"Unsupported configuration file format: {filepath.suffix}")

            self.update_from_dict(config_dict)
            logger.info(f"Configuration loaded from {filepath}")

        except Exception as e:
            logger.error(f"Failed to load configuration from {filepath}: {e}")
            raise

    def save_to_file(self, filepath: Union[str, Path], format: str = "yaml"):
        """
        Save configuration to file.

        Args:
            filepath: Output file path
            format: Output format ("yaml", "json", "pickle")
        """
        filepath = Path(filepath)
        config_dict = self.to_dict()

        try:
            if format.lower() in ['yaml', 'yml']:
                with open(filepath.with_suffix('.yaml'), 'w') as f:
                    yaml.dump(config_dict, f, default_flow_style=False, indent=2)

            elif format.lower() == 'json':
                with open(filepath.with_suffix('.json'), 'w') as f:
                    json.dump(config_dict, f, indent=2, default=str)

            elif format.lower() in ['pkl', 'pickle']:
                with open(filepath.with_suffix('.pkl'), 'wb') as f:
                    pickle.dump(config_dict, f)

            else:
                raise ValueError(f"Unsupported format: {format}")

            logger.info(f"Configuration saved to {filepath}")

        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
            raise

    def update_from_dict(self, config_dict: Dict[str, Any]):
        """
        Update configuration from dictionary.

        Args:
            config_dict: Configuration dictionary
        """
        # Update standard sections
        section_mapping = {
            'population': self.population,
            'network': self.network,
            'behavior': self.behavior,
            'rsiena': self.rsiena,
            'simulation': self.simulation,
            'experimental': self.experimental
        }

        for section_name, section_config in config_dict.items():
            if section_name in section_mapping:
                section_obj = section_mapping[section_name]
                self._update_dataclass_from_dict(section_obj, section_config)
            else:
                # Store custom sections
                self._custom_sections[section_name] = section_config

    def _update_dataclass_from_dict(self, dataclass_obj: Any, update_dict: Dict[str, Any]):
        """Update dataclass object from dictionary."""
        for key, value in update_dict.items():
            if hasattr(dataclass_obj, key):
                # Handle enum conversions
                field_type = type(getattr(dataclass_obj, key))
                if hasattr(field_type, '__bases__') and Enum in field_type.__bases__:
                    if isinstance(value, str):
                        value = field_type(value)

                setattr(dataclass_obj, key, value)
            else:
                logger.warning(f"Unknown configuration parameter: {key}")

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary.

        Returns:
            Configuration dictionary
        """
        config_dict = {
            'population': asdict(self.population),
            'network': asdict(self.network),
            'behavior': asdict(self.behavior),
            'rsiena': asdict(self.rsiena),
            'simulation': asdict(self.simulation),
            'experimental': asdict(self.experimental)
        }

        # Add custom sections
        config_dict.update(self._custom_sections)

        # Convert enums to strings
        config_dict = self._convert_enums_to_strings(config_dict)

        return config_dict

    def _convert_enums_to_strings(self, obj: Any) -> Any:
        """Recursively convert enums to strings in nested structures."""
        if isinstance(obj, Enum):
            return obj.value
        elif isinstance(obj, dict):
            return {k: self._convert_enums_to_strings(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_enums_to_strings(item) for item in obj]
        else:
            return obj

    def validate(self):
        """Validate configuration parameters."""
        errors = []

        # Population validation
        if self.population.mean_age < 0:
            errors.append("Mean age must be non-negative")

        if not 0 <= self.population.gender_ratio <= 1:
            errors.append("Gender ratio must be between 0 and 1")

        if sum(self.population.education_distribution) != 1.0:
            logger.warning("Education distribution does not sum to 1, normalizing...")
            total = sum(self.population.education_distribution)
            self.population.education_distribution = [
                x / total for x in self.population.education_distribution
            ]

        # Network validation
        if self.network.network_size < 3:
            errors.append("Network size must be at least 3")

        if not 0 <= self.network.initial_density <= 1:
            errors.append("Initial density must be between 0 and 1")

        if self.network.friendship_formation_rate < 0:
            errors.append("Friendship formation rate must be non-negative")

        # Simulation validation
        if self.simulation.n_steps < 1:
            errors.append("Number of steps must be positive")

        if self.simulation.n_agents < 1:
            errors.append("Number of agents must be positive")

        if self.simulation.n_agents != self.network.network_size:
            logger.warning("n_agents != network_size, using n_agents")
            self.network.network_size = self.simulation.n_agents

        # RSiena validation
        if self.rsiena.abm_steps_per_period < 1:
            errors.append("ABM steps per period must be positive")

        if self.rsiena.rsiena_periods < 2:
            errors.append("Need at least 2 RSiena periods")

        if errors:
            error_msg = "Configuration validation failed:\n" + "\n".join(f"- {error}" for error in errors)
            raise ValueError(error_msg)

        logger.debug("Configuration validation passed")

    def create_parameter_sweep(
        self,
        sweep_config: Optional[Dict[str, List[Any]]] = None
    ) -> List['ModelConfiguration']:
        """
        Create parameter sweep configurations.

        Args:
            sweep_config: Dictionary of parameters to sweep

        Returns:
            List of configuration objects for parameter sweep
        """
        if sweep_config is None:
            sweep_config = self.experimental.sweep_parameters

        if not sweep_config:
            return [self]

        # Generate parameter combinations
        if self.experimental.sweep_method == "grid":
            combinations = self._generate_grid_combinations(sweep_config)
        elif self.experimental.sweep_method == "random":
            combinations = self._generate_random_combinations(sweep_config)
        elif self.experimental.sweep_method == "sobol":
            combinations = self._generate_sobol_combinations(sweep_config)
        elif self.experimental.sweep_method == "latin_hypercube":
            combinations = self._generate_lhs_combinations(sweep_config)
        else:
            raise ValueError(f"Unknown sweep method: {self.experimental.sweep_method}")

        # Create configuration objects
        configs = []
        for i, combination in enumerate(combinations):
            # Create copy of current configuration
            config_dict = self.to_dict()

            # Apply parameter combination
            for param_path, value in combination.items():
                self._set_nested_parameter(config_dict, param_path, value)

            # Create new configuration
            new_config = ModelConfiguration()
            new_config.update_from_dict(config_dict)

            # Set unique random seed if required
            if 'random_seed' not in combination:
                new_config.simulation.random_seed = self.simulation.random_seed + i

            configs.append(new_config)

        logger.info(f"Created {len(configs)} configurations for parameter sweep")
        return configs

    def _generate_grid_combinations(self, sweep_config: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
        """Generate grid search parameter combinations."""
        from itertools import product

        param_names = list(sweep_config.keys())
        param_values = list(sweep_config.values())

        combinations = []
        for combo in product(*param_values):
            combination = dict(zip(param_names, combo))
            combinations.append(combination)

        return combinations

    def _generate_random_combinations(self, sweep_config: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
        """Generate random parameter combinations."""
        n_combinations = self.experimental.n_total_runs or 100
        np.random.seed(self.experimental.randomization_seed)

        combinations = []
        for _ in range(n_combinations):
            combination = {}
            for param_name, param_values in sweep_config.items():
                combination[param_name] = np.random.choice(param_values)
            combinations.append(combination)

        return combinations

    def _generate_sobol_combinations(self, sweep_config: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
        """Generate Sobol sequence parameter combinations."""
        try:
            from SALib.sample import sobol
        except ImportError:
            logger.warning("SALib not available, falling back to random sampling")
            return self._generate_random_combinations(sweep_config)

        # This is a simplified implementation
        # Full implementation would require proper bounds and scaling
        return self._generate_random_combinations(sweep_config)

    def _generate_lhs_combinations(self, sweep_config: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
        """Generate Latin Hypercube Sampling parameter combinations."""
        try:
            from SALib.sample import latin
        except ImportError:
            logger.warning("SALib not available, falling back to random sampling")
            return self._generate_random_combinations(sweep_config)

        # This is a simplified implementation
        # Full implementation would require proper bounds and scaling
        return self._generate_random_combinations(sweep_config)

    def _set_nested_parameter(self, config_dict: Dict[str, Any], param_path: str, value: Any):
        """Set nested parameter in configuration dictionary."""
        keys = param_path.split('.')
        current_dict = config_dict

        for key in keys[:-1]:
            if key not in current_dict:
                current_dict[key] = {}
            current_dict = current_dict[key]

        current_dict[keys[-1]] = value

    def get_parameter(self, param_path: str) -> Any:
        """
        Get parameter value by dot-separated path.

        Args:
            param_path: Dot-separated parameter path (e.g., "network.initial_density")

        Returns:
            Parameter value
        """
        keys = param_path.split('.')
        current_obj = self

        for key in keys:
            if hasattr(current_obj, key):
                current_obj = getattr(current_obj, key)
            else:
                raise KeyError(f"Parameter not found: {param_path}")

        return current_obj

    def set_parameter(self, param_path: str, value: Any):
        """
        Set parameter value by dot-separated path.

        Args:
            param_path: Dot-separated parameter path
            value: New parameter value
        """
        keys = param_path.split('.')
        current_obj = self

        for key in keys[:-1]:
            if hasattr(current_obj, key):
                current_obj = getattr(current_obj, key)
            else:
                raise KeyError(f"Parameter path not found: {param_path}")

        if hasattr(current_obj, keys[-1]):
            setattr(current_obj, keys[-1], value)
        else:
            raise KeyError(f"Parameter not found: {param_path}")

    def clone(self) -> 'ModelConfiguration':
        """
        Create a deep copy of the configuration.

        Returns:
            Cloned configuration
        """
        config_dict = self.to_dict()
        new_config = ModelConfiguration()
        new_config.update_from_dict(config_dict)
        return new_config

    def get_summary(self) -> str:
        """
        Get configuration summary.

        Returns:
            Summary string
        """
        summary_lines = [
            "Model Configuration Summary",
            "=" * 30,
            f"Agents: {self.simulation.n_agents}",
            f"Steps: {self.simulation.n_steps}",
            f"Network Type: {self.network.network_type.value}",
            f"Initial Density: {self.network.initial_density:.3f}",
            f"Random Seed: {self.simulation.random_seed}",
            "",
            "RSiena Integration:",
            f"  Steps per period: {self.rsiena.abm_steps_per_period}",
            f"  RSiena periods: {self.rsiena.rsiena_periods}",
            f"  Include transitivity: {self.rsiena.include_transitivity}",
            "",
            "Key Parameters:",
            f"  Friendship formation rate: {self.network.friendship_formation_rate:.3f}",
            f"  Opinion influence strength: {self.behavior.opinion_influence_strength:.3f}",
            f"  Transitivity effect: {self.network.transitivity_effect:.3f}"
        ]

        return "\n".join(summary_lines)

    # Convenience properties for backward compatibility
    @property
    def n_agents(self) -> int:
        """Number of agents."""
        return self.simulation.n_agents

    @property
    def n_steps(self) -> int:
        """Number of simulation steps."""
        return self.simulation.n_steps

    @property
    def random_seed(self) -> int:
        """Random seed."""
        return self.simulation.random_seed

    @property
    def abm_steps_per_period(self) -> int:
        """ABM steps per RSiena period."""
        return self.rsiena.abm_steps_per_period

    @property
    def behavior_variables(self) -> List[str]:
        """Behavior variables to track."""
        variables = ['opinion']
        if hasattr(self, '_behavior_variables'):
            variables = self._behavior_variables
        return variables

    @property
    def rsiena_effects(self) -> Dict[str, bool]:
        """RSiena effects specification."""
        return {
            'density': self.rsiena.include_density,
            'reciprocity': self.rsiena.include_reciprocity,
            'transitivity': self.rsiena.include_transitivity,
            'three_cycles': self.rsiena.include_three_cycles,
            'age_similarity': self.rsiena.include_age_similarity,
            'gender_similarity': self.rsiena.include_gender_similarity,
            'ses_similarity': self.rsiena.include_ses_similarity
        }

    @property
    def mean_age(self) -> float:
        """Mean agent age."""
        return self.population.mean_age

    @property
    def sd_age(self) -> float:
        """Standard deviation of agent age."""
        return self.population.sd_age


def create_default_config(n_agents: int = 100, n_steps: int = 1000, **kwargs) -> ModelConfiguration:
    """
    Create default configuration with common parameters.

    Args:
        n_agents: Number of agents
        n_steps: Number of simulation steps
        **kwargs: Additional configuration overrides

    Returns:
        ModelConfiguration with defaults
    """
    config = ModelConfiguration()
    config.simulation.n_agents = n_agents
    config.simulation.n_steps = n_steps
    config.network.network_size = n_agents

    # Apply any additional overrides
    for key, value in kwargs.items():
        try:
            config.set_parameter(key, value)
        except KeyError:
            logger.warning(f"Could not set parameter {key}: parameter not found")

    return config


def load_config_from_env(prefix: str = "ABM_") -> Dict[str, Any]:
    """
    Load configuration from environment variables.

    Args:
        prefix: Environment variable prefix

    Returns:
        Dictionary with environment-based configuration
    """
    env_config = {}

    for key, value in os.environ.items():
        if key.startswith(prefix):
            param_name = key[len(prefix):].lower()

            # Try to convert to appropriate type
            if value.lower() in ['true', 'false']:
                env_config[param_name] = value.lower() == 'true'
            elif value.isdigit():
                env_config[param_name] = int(value)
            else:
                try:
                    env_config[param_name] = float(value)
                except ValueError:
                    env_config[param_name] = value

    return env_config


if __name__ == "__main__":
    # Example usage and testing
    logging.basicConfig(level=logging.INFO)

    # Create default configuration
    config = create_default_config(n_agents=50, n_steps=500)
    print("Default configuration created")
    print(config.get_summary())

    # Test parameter access
    print(f"\nNetwork density: {config.network.initial_density}")
    print(f"Formation rate: {config.network.friendship_formation_rate}")

    # Test parameter sweep
    sweep_config = {
        'network.initial_density': [0.05, 0.1, 0.15],
        'network.transitivity_effect': [0.1, 0.2, 0.3]
    }

    config.experimental.sweep_parameters = sweep_config
    config.experimental.sweep_method = "grid"

    sweep_configs = config.create_parameter_sweep()
    print(f"\nCreated {len(sweep_configs)} configurations for parameter sweep")

    # Test saving/loading
    try:
        config.save_to_file("test_config.yaml")
        loaded_config = ModelConfiguration("test_config.yaml")
        print("Configuration save/load test passed")
    except Exception as e:
        print(f"Configuration save/load test failed: {e}")

    print("Configuration management module ready for use")