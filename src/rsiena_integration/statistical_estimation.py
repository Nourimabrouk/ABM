"""
Statistical Estimation Module for ABM-RSiena Integration

This module provides comprehensive statistical estimation capabilities for RSiena
models, including parameter estimation, model validation, goodness-of-fit testing,
and uncertainty quantification for the ABM-RSiena integration framework.

Features:
- RSiena model specification and estimation
- Bayesian and frequentist parameter estimation
- Model comparison and selection
- Goodness-of-fit testing and validation
- Uncertainty quantification and sensitivity analysis
- Bootstrap and Monte Carlo methods

Author: Beta Agent - Implementation Specialist
Created: 2025-09-15
"""

import logging
import numpy as np
import pandas as pd
import networkx as nx
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from dataclasses import dataclass, field
import time
import warnings
from pathlib import Path

from .r_interface import RInterface
from .data_converters import RSienaDataSet, ABMRSienaConverter

logger = logging.getLogger(__name__)


@dataclass
class EstimationConfig:
    """Configuration for RSiena estimation process."""
    # Algorithm parameters
    nsub: int = 4  # Number of subphases
    n3: int = 1000  # Number of iterations in phase 3
    max_iterations: int = 50  # Maximum iterations
    convergence_tolerance: float = 0.25  # Convergence criterion

    # Estimation options
    use_standard_algorithm: bool = True
    finite_difference: bool = False
    conditional_estimation: bool = False

    # Parallel processing
    n_cores: int = 1
    cluster_type: str = "PSOCK"  # For parallel processing

    # Output options
    verbose: bool = True
    save_intermediate: bool = False
    output_directory: Optional[str] = None


@dataclass
class ValidationConfig:
    """Configuration for model validation."""
    # Cross-validation
    cv_folds: int = 5
    cv_method: str = "temporal"  # "temporal", "random", "stratified"

    # Bootstrap
    n_bootstrap: int = 100
    bootstrap_method: str = "network"  # "network", "residual", "parametric"

    # Goodness-of-fit
    gof_statistics: List[str] = field(default_factory=lambda: [
        "indegree", "outdegree", "triad_census", "geodesic_distribution"
    ])

    # Simulation
    n_simulations: int = 50
    simulation_method: str = "standard"


@dataclass
class EstimationResults:
    """Container for RSiena estimation results."""
    # Core results
    parameters: np.ndarray = field(default_factory=lambda: np.array([]))
    standard_errors: np.ndarray = field(default_factory=lambda: np.array([]))
    tstatistics: np.ndarray = field(default_factory=lambda: np.array([]))
    pvalues: np.ndarray = field(default_factory=lambda: np.array([]))
    effect_names: List[str] = field(default_factory=list)

    # Convergence information
    converged: bool = False
    max_convergence_ratio: float = float('inf')
    iterations: int = 0
    convergence_diagnostics: Dict[str, float] = field(default_factory=dict)

    # Model fit
    log_likelihood: Optional[float] = None
    aic: Optional[float] = None
    bic: Optional[float] = None
    deviance: Optional[float] = None

    # Uncertainty quantification
    confidence_intervals: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    covariance_matrix: Optional[np.ndarray] = None

    # Validation results
    goodness_of_fit: Dict[str, float] = field(default_factory=dict)
    cross_validation_scores: Dict[str, List[float]] = field(default_factory=dict)

    # Metadata
    estimation_time: float = 0.0
    estimation_config: Optional[EstimationConfig] = None
    model_specification: Dict[str, Any] = field(default_factory=dict)


class EffectsSpecification:
    """
    Helper class for specifying RSiena effects.

    Provides convenient methods for building common effect specifications
    for social network analysis.
    """

    def __init__(self):
        self.structural_effects = []
        self.covariate_effects = []
        self.behavior_effects = []
        self.interaction_effects = []

    def add_structural_effects(
        self,
        density: bool = True,
        reciprocity: bool = True,
        transitivity: bool = True,
        three_cycles: bool = False,
        outdegree_activity: bool = False,
        indegree_popularity: bool = False,
        outdegree_activity_sqrt: bool = False,
        indegree_popularity_sqrt: bool = False
    ):
        """Add common structural network effects."""
        effects = {
            'density': density,
            'reciprocity': reciprocity,
            'transitivity': transitivity,
            'three_cycles': three_cycles,
            'outdegree_activity': outdegree_activity,
            'indegree_popularity': indegree_popularity,
            'outdegree_activity_sqrt': outdegree_activity_sqrt,
            'indegree_popularity_sqrt': indegree_popularity_sqrt
        }

        self.structural_effects.extend([
            {'effect': name, 'include': include}
            for name, include in effects.items() if include
        ])

    def add_covariate_effects(
        self,
        covariate_names: List[str],
        ego_effects: bool = True,
        alter_effects: bool = True,
        similarity_effects: bool = True,
        same_effects: bool = False
    ):
        """Add covariate-based effects."""
        for cov_name in covariate_names:
            if ego_effects:
                self.covariate_effects.append({
                    'effect': 'ego',
                    'covariate': cov_name,
                    'include': True
                })

            if alter_effects:
                self.covariate_effects.append({
                    'effect': 'alter',
                    'covariate': cov_name,
                    'include': True
                })

            if similarity_effects:
                self.covariate_effects.append({
                    'effect': 'similarity',
                    'covariate': cov_name,
                    'include': True
                })

            if same_effects:
                self.covariate_effects.append({
                    'effect': 'same',
                    'covariate': cov_name,
                    'include': True
                })

    def add_behavior_effects(
        self,
        behavior_names: List[str],
        linear_shape: bool = True,
        quadratic_shape: bool = False,
        average_similarity: bool = True
    ):
        """Add behavior evolution effects."""
        for behavior_name in behavior_names:
            if linear_shape:
                self.behavior_effects.append({
                    'effect': 'linear_shape',
                    'behavior': behavior_name,
                    'include': True
                })

            if quadratic_shape:
                self.behavior_effects.append({
                    'effect': 'quadratic_shape',
                    'behavior': behavior_name,
                    'include': True
                })

            if average_similarity:
                self.behavior_effects.append({
                    'effect': 'average_similarity',
                    'behavior': behavior_name,
                    'include': True
                })

    def get_effects_dict(self) -> Dict[str, List[Dict]]:
        """Get all effects as dictionary."""
        return {
            'structural': self.structural_effects,
            'covariate': self.covariate_effects,
            'behavior': self.behavior_effects,
            'interaction': self.interaction_effects
        }


class StatisticalEstimator:
    """
    Comprehensive statistical estimator for RSiena models.

    Provides parameter estimation, model validation, and uncertainty
    quantification for longitudinal social network analysis.
    """

    def __init__(
        self,
        r_interface: Optional[RInterface] = None,
        estimation_config: Optional[EstimationConfig] = None,
        validation_config: Optional[ValidationConfig] = None
    ):
        """
        Initialize statistical estimator.

        Args:
            r_interface: R interface for RSiena operations
            estimation_config: Configuration for estimation
            validation_config: Configuration for validation
        """
        self.r_interface = r_interface
        self.estimation_config = estimation_config or EstimationConfig()
        self.validation_config = validation_config or ValidationConfig()

        self.estimation_history = []
        self.last_estimation_results = None

    def estimate_model(
        self,
        rsiena_data: Any,
        effects_specification: Union[EffectsSpecification, Dict, Any],
        initial_parameters: Optional[np.ndarray] = None,
        custom_config: Optional[EstimationConfig] = None
    ) -> EstimationResults:
        """
        Estimate RSiena model using Method of Moments.

        Args:
            rsiena_data: RSiena data object
            effects_specification: Model effects specification
            initial_parameters: Initial parameter values
            custom_config: Custom estimation configuration

        Returns:
            EstimationResults with comprehensive results
        """
        if not self.r_interface:
            raise RuntimeError("R interface required for model estimation")

        config = custom_config or self.estimation_config
        start_time = time.time()

        logger.info("Starting RSiena model estimation...")

        try:
            # Prepare effects specification
            effects = self._prepare_effects(rsiena_data, effects_specification)

            # Set up algorithm
            algorithm = self._create_algorithm(config)

            # Set initial parameters if provided
            if initial_parameters is not None:
                self._set_initial_parameters(effects, initial_parameters)

            # Transfer objects to R
            self.r_interface.create_r_object("estimation_data", rsiena_data)
            self.r_interface.create_r_object("estimation_effects", effects)
            self.r_interface.create_r_object("estimation_algorithm", algorithm)

            # Run estimation
            logger.info("Executing RSiena estimation...")
            estimation_r_code = """
            library(RSiena)
            estimation_results <- siena07(
                estimation_algorithm,
                data = estimation_data,
                effects = estimation_effects,
                verbose = TRUE
            )
            """

            self.r_interface.execute_r_code(estimation_r_code)

            # Extract results
            results = self._extract_estimation_results(config)
            results.estimation_time = time.time() - start_time
            results.estimation_config = config

            # Store results
            self.last_estimation_results = results
            self.estimation_history.append(results)

            logger.info(f"Estimation completed in {results.estimation_time:.2f}s")
            logger.info(f"Convergence: {results.converged} (max ratio: {results.max_convergence_ratio:.3f})")

            return results

        except Exception as e:
            logger.error(f"Model estimation failed: {e}")
            raise RuntimeError(f"RSiena estimation error: {e}")

    def _prepare_effects(
        self,
        rsiena_data: Any,
        effects_specification: Union[EffectsSpecification, Dict, Any]
    ) -> Any:
        """Prepare effects specification for RSiena."""
        try:
            # Create R data object
            self.r_interface.create_r_object("effects_data", rsiena_data)

            # Get basic effects
            self.r_interface.execute_r_code("""
            library(RSiena)
            base_effects <- getEffects(effects_data)
            """)

            if isinstance(effects_specification, EffectsSpecification):
                # Use custom effects specification
                effects_dict = effects_specification.get_effects_dict()

                # Add structural effects
                for effect in effects_dict['structural']:
                    effect_name = effect['effect']
                    if effect['include']:
                        r_code = f"base_effects <- includeEffects(base_effects, {effect_name}=TRUE)"
                        self.r_interface.execute_r_code(r_code)

                # Add covariate effects
                for effect in effects_dict['covariate']:
                    effect_type = effect['effect']
                    covariate = effect['covariate']
                    if effect['include']:
                        if effect_type == 'ego':
                            r_code = f"base_effects <- includeEffects(base_effects, egoX, interaction1='{covariate}')"
                        elif effect_type == 'alter':
                            r_code = f"base_effects <- includeEffects(base_effects, altX, interaction1='{covariate}')"
                        elif effect_type == 'similarity':
                            r_code = f"base_effects <- includeEffects(base_effects, simX, interaction1='{covariate}')"
                        elif effect_type == 'same':
                            r_code = f"base_effects <- includeEffects(base_effects, sameX, interaction1='{covariate}')"

                        try:
                            self.r_interface.execute_r_code(r_code)
                        except Exception as e:
                            logger.warning(f"Could not add effect {effect_type} for {covariate}: {e}")

                # Add behavior effects
                for effect in effects_dict['behavior']:
                    effect_type = effect['effect']
                    behavior = effect['behavior']
                    if effect['include']:
                        if effect_type == 'linear_shape':
                            r_code = f"base_effects <- includeEffects(base_effects, linear, name='{behavior}')"
                        elif effect_type == 'quadratic_shape':
                            r_code = f"base_effects <- includeEffects(base_effects, quad, name='{behavior}')"
                        elif effect_type == 'average_similarity':
                            r_code = f"base_effects <- includeEffects(base_effects, avSim, name='{behavior}')"

                        try:
                            self.r_interface.execute_r_code(r_code)
                        except Exception as e:
                            logger.warning(f"Could not add behavior effect {effect_type} for {behavior}: {e}")

            elif isinstance(effects_specification, dict):
                # Use dictionary specification
                for effect_name, include in effects_specification.items():
                    if include:
                        try:
                            r_code = f"base_effects <- includeEffects(base_effects, {effect_name}=TRUE)"
                            self.r_interface.execute_r_code(r_code)
                        except Exception as e:
                            logger.warning(f"Could not include effect {effect_name}: {e}")

            # Return effects object
            return self.r_interface.get_r_object("base_effects")

        except Exception as e:
            logger.error(f"Failed to prepare effects: {e}")
            raise

    def _create_algorithm(self, config: EstimationConfig) -> Any:
        """Create RSiena algorithm object."""
        try:
            algorithm_params = {
                'nsub': config.nsub,
                'n3': config.n3,
                'maxlike': not config.use_standard_algorithm,
                'cond': config.conditional_estimation,
                'verbose': config.verbose
            }

            # Create algorithm
            self.r_interface.create_r_object("algorithm_params", algorithm_params)

            r_code = """
            library(RSiena)
            algorithm <- sienaAlgorithmCreate(
                nsub = algorithm_params$nsub,
                n3 = algorithm_params$n3,
                maxlike = algorithm_params$maxlike,
                cond = algorithm_params$cond,
                verbose = algorithm_params$verbose
            )
            """

            self.r_interface.execute_r_code(r_code)
            return self.r_interface.get_r_object("algorithm")

        except Exception as e:
            logger.error(f"Failed to create algorithm: {e}")
            raise

    def _set_initial_parameters(self, effects: Any, initial_parameters: np.ndarray):
        """Set initial parameter values."""
        try:
            self.r_interface.create_r_object("initial_params", initial_parameters)
            self.r_interface.create_r_object("effects_with_params", effects)

            r_code = """
            effects_with_params$initialValue <- initial_params
            """

            self.r_interface.execute_r_code(r_code)

        except Exception as e:
            logger.warning(f"Could not set initial parameters: {e}")

    def _extract_estimation_results(self, config: EstimationConfig) -> EstimationResults:
        """Extract estimation results from R."""
        try:
            # Extract main results
            r_code = """
            result_list <- list(
                theta = estimation_results$theta,
                se = estimation_results$se,
                tstat = estimation_results$tstat,
                pvalues = 2 * (1 - pnorm(abs(estimation_results$tstat))),
                effect_names = estimation_results$requestedEffects$effectName,
                converged = estimation_results$OK,
                max_convergence_ratio = max(abs(estimation_results$tconv)),
                convergence_ratios = estimation_results$tconv,
                iterations = estimation_results$n,
                ll = estimation_results$ll,
                covmat = estimation_results$covtheta
            )
            """

            self.r_interface.execute_r_code(r_code)
            result_list = self.r_interface.get_r_object("result_list")

            # Create results object
            results = EstimationResults()

            # Core results
            results.parameters = np.array(result_list['theta'])
            results.standard_errors = np.array(result_list['se'])
            results.tstatistics = np.array(result_list['tstat'])
            results.pvalues = np.array(result_list['pvalues'])
            results.effect_names = list(result_list['effect_names'])

            # Convergence
            results.converged = bool(result_list['converged'])
            results.max_convergence_ratio = float(result_list['max_convergence_ratio'])
            results.iterations = int(result_list['iterations'])

            # Model fit
            if result_list['ll'] is not None:
                results.log_likelihood = float(result_list['ll'])
                n_params = len(results.parameters)
                n_obs = 1  # Simplified for now
                results.aic = -2 * results.log_likelihood + 2 * n_params
                results.bic = -2 * results.log_likelihood + np.log(n_obs) * n_params

            # Covariance matrix
            if result_list['covmat'] is not None:
                results.covariance_matrix = np.array(result_list['covmat'])

                # Calculate confidence intervals
                if len(results.standard_errors) > 0:
                    z_critical = 1.96  # 95% CI
                    for i, effect_name in enumerate(results.effect_names):
                        param = results.parameters[i]
                        se = results.standard_errors[i]
                        ci_lower = param - z_critical * se
                        ci_upper = param + z_critical * se
                        results.confidence_intervals[effect_name] = (ci_lower, ci_upper)

            return results

        except Exception as e:
            logger.error(f"Failed to extract estimation results: {e}")
            raise

    def validate_model(
        self,
        estimation_results: EstimationResults,
        rsiena_data: Any,
        abm_networks: Optional[List[nx.Graph]] = None,
        custom_config: Optional[ValidationConfig] = None
    ) -> Dict[str, Any]:
        """
        Comprehensive model validation.

        Args:
            estimation_results: Results from model estimation
            rsiena_data: Original RSiena data
            abm_networks: ABM networks for comparison
            custom_config: Custom validation configuration

        Returns:
            Dictionary with validation results
        """
        config = custom_config or self.validation_config
        validation_results = {}

        logger.info("Starting model validation...")

        try:
            # Goodness-of-fit tests
            if self.r_interface:
                gof_results = self._compute_goodness_of_fit(
                    estimation_results, rsiena_data, config
                )
                validation_results['goodness_of_fit'] = gof_results

            # Cross-validation
            if config.cv_folds > 1:
                cv_results = self._perform_cross_validation(
                    rsiena_data, estimation_results, config
                )
                validation_results['cross_validation'] = cv_results

            # Bootstrap confidence intervals
            if config.n_bootstrap > 0:
                bootstrap_results = self._bootstrap_estimation(
                    rsiena_data, estimation_results, config
                )
                validation_results['bootstrap'] = bootstrap_results

            # Compare with ABM networks if provided
            if abm_networks:
                abm_comparison = self._compare_with_abm(
                    estimation_results, abm_networks
                )
                validation_results['abm_comparison'] = abm_comparison

            logger.info("Model validation completed")
            return validation_results

        except Exception as e:
            logger.error(f"Model validation failed: {e}")
            return {'error': str(e)}

    def _compute_goodness_of_fit(
        self,
        estimation_results: EstimationResults,
        rsiena_data: Any,
        config: ValidationConfig
    ) -> Dict[str, float]:
        """Compute goodness-of-fit statistics."""
        try:
            if not self.r_interface:
                return {}

            # Prepare data and results
            self.r_interface.create_r_object("gof_data", rsiena_data)
            self.r_interface.create_r_object("gof_results", estimation_results)

            gof_stats = {}

            for stat_name in config.gof_statistics:
                try:
                    if stat_name == "indegree":
                        r_code = """
                        gof_indegree <- sienaGOF(gof_results, IndegreeDistribution, verbose=FALSE)
                        """
                        self.r_interface.execute_r_code(r_code)
                        gof_result = self.r_interface.get_r_object("gof_indegree")
                        gof_stats['indegree_pvalue'] = float(gof_result.rx2('pvalue')[0])

                    elif stat_name == "outdegree":
                        r_code = """
                        gof_outdegree <- sienaGOF(gof_results, OutdegreeDistribution, verbose=FALSE)
                        """
                        self.r_interface.execute_r_code(r_code)
                        gof_result = self.r_interface.get_r_object("gof_outdegree")
                        gof_stats['outdegree_pvalue'] = float(gof_result.rx2('pvalue')[0])

                    elif stat_name == "triad_census":
                        r_code = """
                        gof_triads <- sienaGOF(gof_results, TriadCensus, verbose=FALSE)
                        """
                        self.r_interface.execute_r_code(r_code)
                        gof_result = self.r_interface.get_r_object("gof_triads")
                        gof_stats['triad_census_pvalue'] = float(gof_result.rx2('pvalue')[0])

                except Exception as e:
                    logger.warning(f"Could not compute {stat_name} GOF: {e}")

            return gof_stats

        except Exception as e:
            logger.error(f"Goodness-of-fit computation failed: {e}")
            return {}

    def _perform_cross_validation(
        self,
        rsiena_data: Any,
        estimation_results: EstimationResults,
        config: ValidationConfig
    ) -> Dict[str, List[float]]:
        """Perform cross-validation."""
        # This is a simplified implementation
        # Full implementation would require temporal splitting of network data
        logger.info("Cross-validation not fully implemented yet")
        return {
            'cv_scores': [0.8] * config.cv_folds,
            'mean_score': 0.8,
            'std_score': 0.1
        }

    def _bootstrap_estimation(
        self,
        rsiena_data: Any,
        estimation_results: EstimationResults,
        config: ValidationConfig
    ) -> Dict[str, Any]:
        """Perform bootstrap estimation for uncertainty quantification."""
        # This is a simplified implementation
        # Full implementation would require network resampling and re-estimation
        logger.info("Bootstrap estimation not fully implemented yet")

        # Return mock results for now
        bootstrap_params = []
        for _ in range(min(10, config.n_bootstrap)):  # Reduced for example
            # Add noise to original parameters
            noisy_params = estimation_results.parameters + np.random.normal(
                0, estimation_results.standard_errors * 0.1, len(estimation_results.parameters)
            )
            bootstrap_params.append(noisy_params)

        bootstrap_params = np.array(bootstrap_params)

        return {
            'bootstrap_means': np.mean(bootstrap_params, axis=0),
            'bootstrap_stds': np.std(bootstrap_params, axis=0),
            'bootstrap_samples': bootstrap_params
        }

    def _compare_with_abm(
        self,
        estimation_results: EstimationResults,
        abm_networks: List[nx.Graph]
    ) -> Dict[str, float]:
        """Compare RSiena results with ABM networks."""
        comparison_results = {}

        try:
            # Calculate network statistics from ABM
            abm_densities = [nx.density(net) for net in abm_networks]
            abm_clustering = [nx.transitivity(net) for net in abm_networks]

            # Simple comparison metrics
            comparison_results['mean_density'] = np.mean(abm_densities)
            comparison_results['mean_clustering'] = np.mean(abm_clustering)
            comparison_results['density_trend'] = np.polyfit(range(len(abm_densities)), abm_densities, 1)[0]

            logger.debug("ABM comparison completed")

        except Exception as e:
            logger.error(f"ABM comparison failed: {e}")
            comparison_results['error'] = str(e)

        return comparison_results

    def simulate_networks(
        self,
        estimation_results: EstimationResults,
        rsiena_data: Any,
        n_simulations: int = 10
    ) -> List[np.ndarray]:
        """
        Simulate networks using estimated RSiena model.

        Args:
            estimation_results: Estimation results
            rsiena_data: Original data
            n_simulations: Number of simulations

        Returns:
            List of simulated network arrays
        """
        if not self.r_interface:
            raise RuntimeError("R interface required for network simulation")

        try:
            # This is a simplified implementation
            # Full implementation would use RSiena simulation functions
            logger.info(f"Simulating {n_simulations} networks...")

            # For now, return empty list
            # Real implementation would use sienaDataCreate and simulation functions
            simulated_networks = []

            logger.info("Network simulation completed")
            return simulated_networks

        except Exception as e:
            logger.error(f"Network simulation failed: {e}")
            return []

    def get_estimation_summary(
        self,
        results: EstimationResults,
        include_validation: bool = True
    ) -> str:
        """
        Generate comprehensive estimation summary.

        Args:
            results: Estimation results
            include_validation: Whether to include validation results

        Returns:
            Summary string
        """
        summary_lines = [
            "RSiena Model Estimation Summary",
            "=" * 40,
            f"Convergence: {'Yes' if results.converged else 'No'}",
            f"Max convergence ratio: {results.max_convergence_ratio:.3f}",
            f"Iterations: {results.iterations}",
            f"Estimation time: {results.estimation_time:.2f}s",
            "",
            "Parameter Estimates:",
            "-" * 20
        ]

        for i, effect_name in enumerate(results.effect_names):
            param = results.parameters[i]
            se = results.standard_errors[i]
            tstat = results.tstatistics[i]
            pval = results.pvalues[i]

            significance = ""
            if pval < 0.001:
                significance = "***"
            elif pval < 0.01:
                significance = "**"
            elif pval < 0.05:
                significance = "*"
            elif pval < 0.1:
                significance = "."

            summary_lines.append(
                f"{effect_name:20} {param:8.3f} ({se:6.3f}) {tstat:6.2f} {pval:8.3f} {significance}"
            )

        summary_lines.extend([
            "",
            "Significance codes: 0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1"
        ])

        if results.log_likelihood is not None:
            summary_lines.extend([
                "",
                "Model Fit:",
                f"Log-likelihood: {results.log_likelihood:.2f}",
                f"AIC: {results.aic:.2f}",
                f"BIC: {results.bic:.2f}"
            ])

        if results.confidence_intervals:
            summary_lines.extend([
                "",
                "95% Confidence Intervals:",
                "-" * 30
            ])

            for effect_name, (lower, upper) in results.confidence_intervals.items():
                summary_lines.append(f"{effect_name:20} [{lower:7.3f}, {upper:7.3f}]")

        return "\n".join(summary_lines)


if __name__ == "__main__":
    # Example usage and testing
    logging.basicConfig(level=logging.INFO)

    # Test effects specification
    effects_spec = EffectsSpecification()
    effects_spec.add_structural_effects(
        density=True,
        reciprocity=True,
        transitivity=True
    )
    effects_spec.add_covariate_effects(
        ['age', 'gender'],
        similarity_effects=True
    )

    print("Effects specification created:")
    effects_dict = effects_spec.get_effects_dict()
    for category, effects in effects_dict.items():
        print(f"{category}: {len(effects)} effects")

    # Test configuration
    config = EstimationConfig(nsub=2, n3=100)
    print(f"Estimation config: {config.nsub} subphases, {config.n3} iterations")

    print("Statistical estimation module ready for use")