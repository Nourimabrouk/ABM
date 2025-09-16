"""
RSiena Integration Module for Agent-Based Models

This module provides utilities for integrating RSiena (Simulation Investigation for
Empirical Network Analysis) with Mesa-based Agent-Based Models. RSiena is particularly
useful for modeling longitudinal network dynamics in social science research.

Key Features:
- Convert Mesa network data to RSiena format
- Run RSiena models from Python using rpy2
- Extract and process RSiena results for ABM validation
- Support for both co-evolution and network-behavior dynamics

Dependencies:
- rpy2: Python-R interface
- R with RSiena package installed
- Mesa: For ABM framework
- NetworkX: For network manipulation
"""

import logging
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
import pandas as pd
import networkx as nx

try:
    import rpy2.robjects as robjects
    from rpy2.robjects import pandas2ri, numpy2ri
    from rpy2.robjects.packages import importr
    import rpy2.robjects.numpy2ri
    rpy2.robjects.numpy2ri.activate()
    pandas2ri.activate()

    # Import R base functions
    r_base = importr('base')
    r_utils = importr('utils')

    RPY2_AVAILABLE = True
except ImportError as e:
    RPY2_AVAILABLE = False
    warnings.warn(f"rpy2 not available: {e}. RSiena integration will not work.")

logger = logging.getLogger(__name__)


class RSienaIntegrator:
    """
    Main class for integrating RSiena with Agent-Based Models.

    This class handles the conversion of ABM network data to RSiena format,
    runs RSiena models, and processes results for use in ABM validation
    and parameterization.
    """

    def __init__(self, ensure_rsiena: bool = True):
        """
        Initialize the RSiena integrator.

        Args:
            ensure_rsiena: Whether to ensure RSiena package is installed
        """
        if not RPY2_AVAILABLE:
            raise ImportError("rpy2 is required for RSiena integration. Install with: pip install rpy2")

        self.r_session = robjects.r
        self.rsiena = None

        if ensure_rsiena:
            self._ensure_rsiena_installed()

    def _ensure_rsiena_installed(self) -> None:
        """Ensure RSiena package is installed in R."""
        try:
            self.rsiena = importr('RSiena')
            logger.info("RSiena package loaded successfully")
        except Exception as e:
            logger.warning(f"RSiena not found: {e}")
            logger.info("Installing RSiena package...")
            try:
                r_utils.install_packages('RSiena')
                self.rsiena = importr('RSiena')
                logger.info("RSiena package installed and loaded successfully")
            except Exception as install_error:
                raise RuntimeError(f"Failed to install RSiena: {install_error}")

    def mesa_networks_to_rsiena(
        self,
        networks: List[nx.Graph],
        actor_attributes: Optional[Dict[str, List]] = None,
        dyadic_covariates: Optional[List[np.ndarray]] = None
    ) -> Dict[str, Any]:
        """
        Convert Mesa network snapshots to RSiena data format.

        Args:
            networks: List of NetworkX graphs representing network at different time points
            actor_attributes: Dictionary of actor-level attributes over time
            dyadic_covariates: List of dyadic covariate matrices for each time point

        Returns:
            Dictionary containing RSiena data objects
        """
        if not self.rsiena:
            raise RuntimeError("RSiena package not available")

        n_actors = len(networks[0].nodes())
        n_periods = len(networks)

        # Convert networks to adjacency matrices
        adjacency_matrices = []
        for graph in networks:
            # Ensure consistent node ordering
            nodes = sorted(graph.nodes())
            adj_matrix = nx.adjacency_matrix(graph, nodelist=nodes).toarray()
            adjacency_matrices.append(adj_matrix)

        # Create RSiena network array
        network_array = np.array(adjacency_matrices)
        r_network_array = numpy2ri.py2rpy(network_array)

        # Create sienaDependent object for networks
        siena_net = self.rsiena.sienaDependent(r_network_array, type="oneMode")

        # Process actor attributes if provided
        siena_covariates = {}
        if actor_attributes:
            for attr_name, attr_values in actor_attributes.items():
                attr_array = np.array(attr_values).reshape(n_periods, n_actors)
                r_attr_array = numpy2ri.py2rpy(attr_array)

                # Determine if changing or constant covariate
                if np.all(attr_array[0] == attr_array):
                    # Constant covariate
                    siena_covariates[attr_name] = self.rsiena.coCovar(r_attr_array[0])
                else:
                    # Changing covariate
                    siena_covariates[attr_name] = self.rsiena.varCovar(r_attr_array)

        # Process dyadic covariates if provided
        siena_dyadic_covs = {}
        if dyadic_covariates:
            for i, dyad_cov in enumerate(dyadic_covariates):
                r_dyad_array = numpy2ri.py2rpy(dyad_cov)
                siena_dyadic_covs[f'dyadCov_{i}'] = self.rsiena.coDyadCovar(r_dyad_array)

        # Combine all data
        data_objects = {'network': siena_net}
        data_objects.update(siena_covariates)
        data_objects.update(siena_dyadic_covs)

        # Create RSiena data object
        siena_data = self.rsiena.sienaDataCreate(**data_objects)

        return {
            'siena_data': siena_data,
            'data_objects': data_objects,
            'n_actors': n_actors,
            'n_periods': n_periods
        }

    def create_rsiena_effects(
        self,
        siena_data: Any,
        include_structural: bool = True,
        include_individual: bool = True,
        custom_effects: Optional[List[str]] = None
    ) -> Any:
        """
        Create effects object for RSiena model specification.

        Args:
            siena_data: RSiena data object
            include_structural: Include basic structural effects
            include_individual: Include individual-level effects
            custom_effects: List of custom effect names to include

        Returns:
            RSiena effects object
        """
        # Get basic effects
        effects = self.rsiena.getEffects(siena_data)

        if include_structural:
            # Include common structural effects
            effects = self.rsiena.includeEffects(effects, transTrip=True)  # Transitivity
            effects = self.rsiena.includeEffects(effects, cycle3=True)     # 3-cycles

        if custom_effects:
            for effect in custom_effects:
                try:
                    effects = self.rsiena.includeEffects(effects, **{effect: True})
                except Exception as e:
                    logger.warning(f"Could not include effect {effect}: {e}")

        return effects

    def estimate_rsiena_model(
        self,
        siena_data: Any,
        effects: Any,
        algorithm_settings: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Estimate RSiena model using Method of Moments.

        Args:
            siena_data: RSiena data object
            effects: RSiena effects object
            algorithm_settings: Custom algorithm settings

        Returns:
            Dictionary containing estimation results
        """
        # Set algorithm parameters
        if algorithm_settings is None:
            algorithm_settings = {
                'nsub': 4,
                'n3': 1000,
                'simOnly': False
            }

        # Create algorithm object
        algorithm = self.rsiena.sienaAlgorithmCreate(**algorithm_settings)

        # Estimate model
        logger.info("Starting RSiena model estimation...")
        results = self.rsiena.siena07(algorithm, data=siena_data, effects=effects)

        # Extract key results
        theta = np.array(results.rx2('theta'))
        se = np.array(results.rx2('se'))
        tstat = theta / se

        # Get effect names
        effect_names = list(results.rx2('requestedEffects').rx2('effectName'))

        results_dict = {
            'raw_results': results,
            'parameters': theta,
            'standard_errors': se,
            'z_scores': tstat,
            'effect_names': effect_names,
            'convergence': results.rx2('OK')[0],
            'max_convergence_ratio': max(abs(results.rx2('tconv')))
        }

        logger.info(f"Model estimation completed. Convergence: {results_dict['convergence']}")

        return results_dict

    def simulate_rsiena_networks(
        self,
        results: Dict[str, Any],
        n_simulations: int = 100
    ) -> List[np.ndarray]:
        """
        Simulate networks using estimated RSiena model.

        Args:
            results: Results from estimate_rsiena_model
            n_simulations: Number of network simulations to generate

        Returns:
            List of simulated adjacency matrices
        """
        # Extract estimation results
        siena_results = results['raw_results']

        # Simulate networks
        logger.info(f"Simulating {n_simulations} networks...")
        simulations = self.rsiena.sienaGOF(
            sienaFitObject=siena_results,
            auxiliaryStatistics=self.rsiena.OutdegreeDistribution,
            verbose=False,
            join=True,
            varName="network"
        )

        # Extract simulated networks (this is simplified - actual extraction depends on RSiena version)
        simulated_networks = []
        # Note: Actual implementation would need to properly extract simulated networks
        # from RSiena output, which can be complex depending on the specific use case

        return simulated_networks

    def validate_abm_with_rsiena(
        self,
        abm_networks: List[nx.Graph],
        rsiena_results: Dict[str, Any],
        statistics: List[str] = None
    ) -> Dict[str, float]:
        """
        Validate ABM network evolution against RSiena model predictions.

        Args:
            abm_networks: Networks generated by ABM
            rsiena_results: Results from RSiena estimation
            statistics: Network statistics to compare

        Returns:
            Dictionary of validation metrics
        """
        if statistics is None:
            statistics = ['density', 'transitivity', 'degree_centralization']

        # Calculate statistics for ABM networks
        abm_stats = self._calculate_network_statistics(abm_networks, statistics)

        # Simulate RSiena networks for comparison
        rsiena_networks = self.simulate_rsiena_networks(rsiena_results)
        rsiena_stats = self._calculate_network_statistics(rsiena_networks, statistics)

        # Compare statistics
        validation_metrics = {}
        for stat in statistics:
            abm_values = abm_stats[stat]
            rsiena_values = rsiena_stats[stat]

            # Calculate similarity metrics
            validation_metrics[f'{stat}_correlation'] = np.corrcoef(abm_values, rsiena_values)[0, 1]
            validation_metrics[f'{stat}_rmse'] = np.sqrt(np.mean((abm_values - rsiena_values) ** 2))

        return validation_metrics

    def _calculate_network_statistics(
        self,
        networks: List[Union[nx.Graph, np.ndarray]],
        statistics: List[str]
    ) -> Dict[str, List[float]]:
        """Calculate network statistics for a list of networks."""
        stats_dict = {stat: [] for stat in statistics}

        for network in networks:
            if isinstance(network, np.ndarray):
                # Convert adjacency matrix to NetworkX graph
                network = nx.from_numpy_array(network)

            for stat in statistics:
                if stat == 'density':
                    stats_dict[stat].append(nx.density(network))
                elif stat == 'transitivity':
                    stats_dict[stat].append(nx.transitivity(network))
                elif stat == 'degree_centralization':
                    # Calculate degree centralization
                    degrees = [d for n, d in network.degree()]
                    max_deg = max(degrees)
                    n = len(network.nodes())
                    numerator = sum(max_deg - deg for deg in degrees)
                    denominator = (n - 1) * (n - 2)
                    stats_dict[stat].append(numerator / denominator if denominator > 0 else 0)

        return stats_dict


def create_example_longitudinal_networks(n_actors: int = 20, n_periods: int = 3) -> List[nx.Graph]:
    """
    Create example longitudinal networks for testing RSiena integration.

    Args:
        n_actors: Number of actors in the network
        n_periods: Number of time periods

    Returns:
        List of NetworkX graphs
    """
    networks = []
    np.random.seed(42)  # For reproducibility

    for period in range(n_periods):
        # Create network with evolving structure
        p = 0.1 + period * 0.05  # Increasing density over time
        G = nx.erdos_renyi_graph(n_actors, p, directed=True)
        networks.append(G)

    return networks


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    if RPY2_AVAILABLE:
        # Create example data
        networks = create_example_longitudinal_networks()

        # Initialize integrator
        integrator = RSienaIntegrator()

        # Convert to RSiena format
        rsiena_data = integrator.mesa_networks_to_rsiena(networks)

        # Create effects
        effects = integrator.create_rsiena_effects(rsiena_data['siena_data'])

        # Estimate model (commented out for quick testing)
        # results = integrator.estimate_rsiena_model(rsiena_data['siena_data'], effects)

        print("RSiena integration setup complete!")
    else:
        print("rpy2 not available. Please install with: pip install rpy2")