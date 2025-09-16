"""
Data Converters for ABM-RSiena Integration

This module provides comprehensive data conversion utilities for transforming
Mesa ABM data structures into RSiena-compatible formats and vice versa.
Handles network data, actor attributes, and behavior variables with proper
temporal alignment.

Features:
- Network data conversion with consistent node ordering
- Actor attribute handling for both constant and time-varying covariates
- Behavior variable conversion for co-evolution models
- Temporal alignment and period management
- Data validation and quality checks

Author: Beta Agent - Implementation Specialist
Created: 2025-09-15
"""

import logging
import numpy as np
import pandas as pd
import networkx as nx
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
from pathlib import Path
import warnings

from .r_interface import RInterface

logger = logging.getLogger(__name__)


@dataclass
class RSienaDataSet:
    """Container for RSiena-formatted data."""
    siena_data_object: Any  # R object
    network_data: np.ndarray  # (n_periods, n_actors, n_actors)
    actor_attributes: Dict[str, np.ndarray] = field(default_factory=dict)
    behavior_data: Dict[str, np.ndarray] = field(default_factory=dict)
    dyadic_covariates: Dict[str, np.ndarray] = field(default_factory=dict)
    n_actors: int = 0
    n_periods: int = 0
    actor_ids: List[int] = field(default_factory=list)
    period_labels: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConversionConfig:
    """Configuration for data conversion process."""
    validate_networks: bool = True
    handle_missing_actors: str = "error"  # "error", "fill", "remove"
    missing_value_fill: float = 0.0
    standardize_attributes: bool = False
    temporal_alignment: str = "strict"  # "strict", "flexible"
    preserve_edge_weights: bool = False
    minimum_network_size: int = 3


class ABMRSienaConverter:
    """
    Comprehensive converter between ABM and RSiena data formats.

    Handles the complex process of converting Mesa ABM data structures
    into RSiena-compatible formats while preserving temporal dynamics
    and statistical properties.
    """

    def __init__(
        self,
        r_interface: Optional[RInterface] = None,
        config: Optional[ConversionConfig] = None
    ):
        """
        Initialize converter with R interface and configuration.

        Args:
            r_interface: R interface for RSiena operations
            config: Conversion configuration
        """
        self.r_interface = r_interface
        self.config = config or ConversionConfig()
        self.conversion_cache = {}
        self.last_conversion_info = {}

    def convert_to_rsiena(
        self,
        networks: List[nx.Graph],
        behaviors: Optional[List[Dict[int, Dict[str, Any]]]] = None,
        actor_attributes: Optional[Dict[str, List[Any]]] = None,
        dyadic_covariates: Optional[List[np.ndarray]] = None,
        actor_id_mapping: Optional[Dict[int, int]] = None
    ) -> RSienaDataSet:
        """
        Convert ABM data to RSiena format.

        Args:
            networks: List of NetworkX graphs (temporal sequence)
            behaviors: List of behavior dictionaries per period
            actor_attributes: Dictionary of actor-level attributes
            dyadic_covariates: List of dyadic covariate matrices
            actor_id_mapping: Optional mapping from original to sequential IDs

        Returns:
            RSienaDataSet with converted data
        """
        logger.info(f"Converting {len(networks)} networks to RSiena format")

        # Validate input data
        self._validate_input_data(networks, behaviors, actor_attributes)

        # Prepare actor ID mapping
        actor_mapping = self._prepare_actor_mapping(networks, actor_id_mapping)

        # Convert network data
        network_array, consistent_actor_ids = self._convert_networks(networks, actor_mapping)

        # Convert actor attributes
        attribute_arrays = self._convert_actor_attributes(
            actor_attributes, consistent_actor_ids, len(networks)
        )

        # Convert behavior data
        behavior_arrays = self._convert_behavior_data(
            behaviors, consistent_actor_ids, len(networks)
        )

        # Convert dyadic covariates
        dyadic_arrays = self._convert_dyadic_covariates(
            dyadic_covariates, consistent_actor_ids
        )

        # Create RSiena data object
        siena_data_object = self._create_rsiena_data_object(
            network_array, attribute_arrays, behavior_arrays, dyadic_arrays
        )

        # Create dataset container
        dataset = RSienaDataSet(
            siena_data_object=siena_data_object,
            network_data=network_array,
            actor_attributes=attribute_arrays,
            behavior_data=behavior_arrays,
            dyadic_covariates=dyadic_arrays,
            n_actors=len(consistent_actor_ids),
            n_periods=len(networks),
            actor_ids=consistent_actor_ids,
            period_labels=[f"Period_{i+1}" for i in range(len(networks))],
            metadata={
                'conversion_config': self.config,
                'original_network_sizes': [len(net.nodes()) for net in networks],
                'conversion_timestamp': pd.Timestamp.now(),
                'actor_mapping': actor_mapping
            }
        )

        logger.info(f"Conversion completed: {dataset.n_actors} actors, {dataset.n_periods} periods")
        return dataset

    def _validate_input_data(
        self,
        networks: List[nx.Graph],
        behaviors: Optional[List[Dict]],
        actor_attributes: Optional[Dict]
    ):
        """Validate input data consistency and quality."""
        if not networks:
            raise ValueError("Networks list cannot be empty")

        if len(networks) < 2:
            raise ValueError("Need at least 2 network periods for RSiena analysis")

        # Check network sizes
        network_sizes = [len(net.nodes()) for net in networks]
        if self.config.temporal_alignment == "strict":
            if len(set(network_sizes)) > 1:
                raise ValueError("All networks must have same number of nodes in strict alignment mode")

        if min(network_sizes) < self.config.minimum_network_size:
            raise ValueError(f"Networks too small (min size: {self.config.minimum_network_size})")

        # Validate behavior data if provided
        if behaviors:
            if len(behaviors) != len(networks):
                raise ValueError("Behavior data must match number of network periods")

        # Validate attribute data if provided
        if actor_attributes:
            for attr_name, attr_values in actor_attributes.items():
                if not isinstance(attr_values, list):
                    raise ValueError(f"Attribute '{attr_name}' must be a list")

        logger.debug("Input data validation passed")

    def _prepare_actor_mapping(
        self,
        networks: List[nx.Graph],
        provided_mapping: Optional[Dict[int, int]]
    ) -> Dict[int, int]:
        """Prepare consistent actor ID mapping across time periods."""
        if provided_mapping:
            return provided_mapping

        # Collect all unique actor IDs across periods
        all_actor_ids = set()
        for network in networks:
            all_actor_ids.update(network.nodes())

        # Create sequential mapping
        sorted_ids = sorted(all_actor_ids)
        mapping = {original_id: new_id for new_id, original_id in enumerate(sorted_ids)}

        logger.debug(f"Created actor mapping for {len(mapping)} actors")
        return mapping

    def _convert_networks(
        self,
        networks: List[nx.Graph],
        actor_mapping: Dict[int, int]
    ) -> Tuple[np.ndarray, List[int]]:
        """Convert NetworkX graphs to adjacency matrix array."""
        n_periods = len(networks)
        n_actors = len(actor_mapping)

        # Initialize network array
        network_array = np.zeros((n_periods, n_actors, n_actors), dtype=int)

        # Get consistent actor list
        consistent_actor_ids = [None] * n_actors
        for original_id, new_id in actor_mapping.items():
            consistent_actor_ids[new_id] = original_id

        # Convert each network
        for period, network in enumerate(networks):
            # Create adjacency matrix with consistent ordering
            adj_matrix = np.zeros((n_actors, n_actors), dtype=int)

            for edge in network.edges():
                source_orig, target_orig = edge
                if source_orig in actor_mapping and target_orig in actor_mapping:
                    source_new = actor_mapping[source_orig]
                    target_new = actor_mapping[target_orig]

                    if self.config.preserve_edge_weights and 'weight' in network[source_orig][target_orig]:
                        weight = network[source_orig][target_orig]['weight']
                        adj_matrix[source_new, target_new] = weight
                    else:
                        adj_matrix[source_new, target_new] = 1

            # Handle missing actors based on configuration
            if self.config.handle_missing_actors == "fill":
                # Fill missing actors with zeros (already done)
                pass
            elif self.config.handle_missing_actors == "error":
                missing_actors = set(actor_mapping.keys()) - set(network.nodes())
                if missing_actors:
                    raise ValueError(f"Missing actors in period {period}: {missing_actors}")

            network_array[period] = adj_matrix

        if self.config.validate_networks:
            self._validate_network_array(network_array)

        logger.debug(f"Converted {n_periods} networks to {n_actors}x{n_actors} matrices")
        return network_array, consistent_actor_ids

    def _validate_network_array(self, network_array: np.ndarray):
        """Validate converted network array."""
        n_periods, n_actors, _ = network_array.shape

        # Check for self-loops
        for period in range(n_periods):
            diagonal = np.diag(network_array[period])
            if np.any(diagonal != 0):
                warnings.warn(f"Self-loops detected in period {period}")

        # Check for reasonable sparsity
        for period in range(n_periods):
            density = np.sum(network_array[period]) / (n_actors * (n_actors - 1))
            if density > 0.8:
                warnings.warn(f"Very dense network in period {period}: density={density:.3f}")

        logger.debug("Network array validation completed")

    def _convert_actor_attributes(
        self,
        actor_attributes: Optional[Dict[str, List[Any]]],
        consistent_actor_ids: List[int],
        n_periods: int
    ) -> Dict[str, np.ndarray]:
        """Convert actor attributes to RSiena format."""
        if not actor_attributes:
            return {}

        attribute_arrays = {}
        n_actors = len(consistent_actor_ids)

        for attr_name, attr_values in actor_attributes.items():
            if len(attr_values) == n_actors:
                # Constant attribute (same across all periods)
                attr_array = np.array(attr_values, dtype=float)
                if self.config.standardize_attributes:
                    attr_array = (attr_array - np.mean(attr_array)) / np.std(attr_array)

            elif len(attr_values) == n_actors * n_periods:
                # Time-varying attribute
                attr_array = np.array(attr_values).reshape(n_periods, n_actors)
                if self.config.standardize_attributes:
                    # Standardize across all time points
                    flat_values = attr_array.flatten()
                    standardized = (flat_values - np.mean(flat_values)) / np.std(flat_values)
                    attr_array = standardized.reshape(n_periods, n_actors)

            else:
                raise ValueError(
                    f"Attribute '{attr_name}' has incompatible length: "
                    f"{len(attr_values)} (expected {n_actors} or {n_actors * n_periods})"
                )

            attribute_arrays[attr_name] = attr_array

        logger.debug(f"Converted {len(attribute_arrays)} actor attributes")
        return attribute_arrays

    def _convert_behavior_data(
        self,
        behaviors: Optional[List[Dict[int, Dict[str, Any]]]],
        consistent_actor_ids: List[int],
        n_periods: int
    ) -> Dict[str, np.ndarray]:
        """Convert behavior data to RSiena format."""
        if not behaviors:
            return {}

        behavior_arrays = {}
        n_actors = len(consistent_actor_ids)

        # Identify all behavior variables
        all_behavior_vars = set()
        for period_behaviors in behaviors:
            for actor_behaviors in period_behaviors.values():
                all_behavior_vars.update(actor_behaviors.keys())

        # Convert each behavior variable
        for behavior_var in all_behavior_vars:
            behavior_matrix = np.full(
                (n_periods, n_actors),
                self.config.missing_value_fill,
                dtype=float
            )

            for period, period_behaviors in enumerate(behaviors):
                for actor_idx, actor_id in enumerate(consistent_actor_ids):
                    if actor_id in period_behaviors:
                        actor_behaviors = period_behaviors[actor_id]
                        if behavior_var in actor_behaviors:
                            value = actor_behaviors[behavior_var]
                            if value is not None:
                                behavior_matrix[period, actor_idx] = float(value)

            behavior_arrays[behavior_var] = behavior_matrix

        logger.debug(f"Converted {len(behavior_arrays)} behavior variables")
        return behavior_arrays

    def _convert_dyadic_covariates(
        self,
        dyadic_covariates: Optional[List[np.ndarray]],
        consistent_actor_ids: List[int]
    ) -> Dict[str, np.ndarray]:
        """Convert dyadic covariates to RSiena format."""
        if not dyadic_covariates:
            return {}

        dyadic_arrays = {}
        n_actors = len(consistent_actor_ids)

        for i, covariate_matrix in enumerate(dyadic_covariates):
            if covariate_matrix.shape != (n_actors, n_actors):
                raise ValueError(
                    f"Dyadic covariate {i} has wrong shape: "
                    f"{covariate_matrix.shape} (expected {(n_actors, n_actors)})"
                )

            dyadic_arrays[f"dyadic_cov_{i}"] = covariate_matrix.astype(float)

        logger.debug(f"Converted {len(dyadic_arrays)} dyadic covariates")
        return dyadic_arrays

    def _create_rsiena_data_object(
        self,
        network_array: np.ndarray,
        attribute_arrays: Dict[str, np.ndarray],
        behavior_arrays: Dict[str, np.ndarray],
        dyadic_arrays: Dict[str, np.ndarray]
    ) -> Any:
        """Create RSiena data object using R interface."""
        if not self.r_interface:
            logger.warning("No R interface available, returning None for RSiena data object")
            return None

        try:
            # Transfer network data to R
            self.r_interface.create_r_object("network_array", network_array)

            # Create sienaDependent object for networks
            r_code = """
            library(RSiena)
            siena_network <- sienaDependent(network_array, type="oneMode")
            """
            self.r_interface.execute_r_code(r_code)

            # Create data objects dictionary
            data_objects = {"friendship": "siena_network"}

            # Add actor attributes
            for attr_name, attr_array in attribute_arrays.items():
                self.r_interface.create_r_object(f"attr_{attr_name}", attr_array)

                if len(attr_array.shape) == 1:
                    # Constant covariate
                    r_code = f"""
                    {attr_name}_cov <- coCovar(attr_{attr_name})
                    """
                else:
                    # Time-varying covariate
                    r_code = f"""
                    {attr_name}_cov <- varCovar(attr_{attr_name})
                    """

                self.r_interface.execute_r_code(r_code)
                data_objects[attr_name] = f"{attr_name}_cov"

            # Add behavior variables
            for behavior_name, behavior_array in behavior_arrays.items():
                self.r_interface.create_r_object(f"behavior_{behavior_name}", behavior_array)

                r_code = f"""
                {behavior_name}_dep <- sienaDependent(behavior_{behavior_name}, type="behavior")
                """
                self.r_interface.execute_r_code(r_code)
                data_objects[behavior_name] = f"{behavior_name}_dep"

            # Add dyadic covariates
            for dyadic_name, dyadic_array in dyadic_arrays.items():
                self.r_interface.create_r_object(f"dyadic_{dyadic_name}", dyadic_array)

                r_code = f"""
                {dyadic_name}_cov <- coDyadCovar(dyadic_{dyadic_name})
                """
                self.r_interface.execute_r_code(r_code)
                data_objects[dyadic_name] = f"{dyadic_name}_cov"

            # Create RSiena data object
            data_objects_r = ", ".join([f"{k}={v}" for k, v in data_objects.items()])
            r_code = f"""
            siena_data <- sienaDataCreate({data_objects_r})
            """
            self.r_interface.execute_r_code(r_code)

            # Return R object reference
            return self.r_interface.get_r_object("siena_data")

        except Exception as e:
            logger.error(f"Failed to create RSiena data object: {e}")
            return None

    def convert_from_rsiena(
        self,
        rsiena_results: Any,
        original_actor_ids: Optional[List[int]] = None
    ) -> Dict[str, Any]:
        """
        Convert RSiena results back to Python format.

        Args:
            rsiena_results: RSiena model results
            original_actor_ids: Original actor IDs for mapping

        Returns:
            Dictionary with converted results
        """
        if not self.r_interface:
            raise RuntimeError("R interface required for conversion from RSiena")

        try:
            # Create R object for results
            self.r_interface.create_r_object("rsiena_results", rsiena_results)

            # Extract key components
            r_code = """
            result_summary <- list(
                parameters = rsiena_results$theta,
                standard_errors = rsiena_results$se,
                tstatistics = rsiena_results$tstat,
                effect_names = rsiena_results$requestedEffects$effectName,
                convergence = rsiena_results$OK,
                max_convergence_ratio = max(abs(rsiena_results$tconv)),
                iterations = rsiena_results$n,
                likelihood = rsiena_results$ll
            )
            """
            self.r_interface.execute_r_code(r_code)

            # Get results
            result_summary = self.r_interface.get_r_object("result_summary")

            # Convert to Python format
            converted_results = {
                'parameters': np.array(result_summary['parameters']),
                'standard_errors': np.array(result_summary['standard_errors']),
                'tstatistics': np.array(result_summary['tstatistics']),
                'effect_names': list(result_summary['effect_names']),
                'converged': bool(result_summary['convergence']),
                'max_convergence_ratio': float(result_summary['max_convergence_ratio']),
                'iterations': int(result_summary['iterations']),
                'log_likelihood': float(result_summary['likelihood']) if result_summary['likelihood'] else None,
                'original_actor_ids': original_actor_ids
            }

            logger.debug("Successfully converted RSiena results to Python format")
            return converted_results

        except Exception as e:
            logger.error(f"Failed to convert RSiena results: {e}")
            return {}

    def export_to_file(
        self,
        dataset: RSienaDataSet,
        filepath: Union[str, Path],
        format: str = "rdata"
    ):
        """
        Export RSiena dataset to file.

        Args:
            dataset: RSiena dataset
            filepath: Output file path
            format: Export format ("rdata", "csv", "pickle")
        """
        filepath = Path(filepath)

        if format == "rdata" and self.r_interface:
            # Save as R data file
            self.r_interface.create_r_object("export_data", dataset.siena_data_object)
            self.r_interface.execute_r_code(f'save(export_data, file="{filepath}")')

        elif format == "csv":
            # Export as CSV files
            base_path = filepath.with_suffix("")

            # Export networks
            for period in range(dataset.n_periods):
                network_df = pd.DataFrame(
                    dataset.network_data[period],
                    index=dataset.actor_ids,
                    columns=dataset.actor_ids
                )
                network_df.to_csv(f"{base_path}_network_period_{period+1}.csv")

            # Export attributes
            for attr_name, attr_data in dataset.actor_attributes.items():
                if len(attr_data.shape) == 1:
                    attr_df = pd.DataFrame({
                        'actor_id': dataset.actor_ids,
                        attr_name: attr_data
                    })
                    attr_df.to_csv(f"{base_path}_attribute_{attr_name}.csv", index=False)

        elif format == "pickle":
            # Export as pickle
            import pickle
            with open(filepath, 'wb') as f:
                pickle.dump(dataset, f)

        else:
            raise ValueError(f"Unsupported export format: {format}")

        logger.info(f"Exported dataset to {filepath} in {format} format")

    def get_conversion_summary(self, dataset: RSienaDataSet) -> str:
        """
        Generate summary of conversion process.

        Args:
            dataset: Converted dataset

        Returns:
            Summary string
        """
        summary_lines = [
            "RSiena Data Conversion Summary",
            "=" * 35,
            f"Number of actors: {dataset.n_actors}",
            f"Number of periods: {dataset.n_periods}",
            f"Network dimensions: {dataset.network_data.shape}",
            "",
            "Actor Attributes:",
        ]

        for attr_name, attr_data in dataset.actor_attributes.items():
            if len(attr_data.shape) == 1:
                summary_lines.append(f"  - {attr_name}: constant (n={len(attr_data)})")
            else:
                summary_lines.append(f"  - {attr_name}: time-varying {attr_data.shape}")

        if dataset.behavior_data:
            summary_lines.append("")
            summary_lines.append("Behavior Variables:")
            for behavior_name, behavior_data in dataset.behavior_data.items():
                summary_lines.append(f"  - {behavior_name}: {behavior_data.shape}")

        if dataset.dyadic_covariates:
            summary_lines.append("")
            summary_lines.append("Dyadic Covariates:")
            for dyadic_name, dyadic_data in dataset.dyadic_covariates.items():
                summary_lines.append(f"  - {dyadic_name}: {dyadic_data.shape}")

        summary_lines.extend([
            "",
            f"Conversion timestamp: {dataset.metadata.get('conversion_timestamp', 'Unknown')}",
        ])

        return "\n".join(summary_lines)


if __name__ == "__main__":
    # Example usage and testing
    logging.basicConfig(level=logging.INFO)

    # Create test networks
    test_networks = []
    for period in range(3):
        G = nx.erdos_renyi_graph(20, 0.1, directed=True)
        test_networks.append(G)

    # Create test attributes
    actor_ids = list(range(20))
    test_attributes = {
        'age': [np.random.normal(25, 5) for _ in actor_ids],
        'gender': [np.random.choice([0, 1]) for _ in actor_ids]
    }

    # Test conversion
    try:
        converter = ABMRSienaConverter()
        dataset = converter.convert_to_rsiena(
            networks=test_networks,
            actor_attributes=test_attributes
        )

        print("Conversion successful!")
        print(converter.get_conversion_summary(dataset))

    except Exception as e:
        print(f"Conversion test failed: {e}")
        import traceback
        traceback.print_exc()