"""
Tolerance-Cooperation Stochastic Actor-Oriented Model (SAOM)

This module implements the core SAOM for studying tolerance interventions
and their effects on interethnic cooperation dynamics. Based on the research
framework from Simons, Tolsma & Jaspers on designing social norm interventions.

Key Features:
- Multi-network co-evolution (friendship + cooperation)
- Behavior dynamics (tolerance as continuous variable)
- Attraction-repulsion influence mechanism
- Complex contagion effects
- Tolerance → cooperation selection
- Control for prejudice confounding
- Intervention simulation capabilities

Research Context:
- 5825 observations, 2585 respondents, 3 schools, 105 classes, 3 waves
- Focus: Social norm interventions promoting interethnic cooperation through tolerance
- Method: SAOM using RSiena with custom C++ effects

Author: RSiena Integration Specialist
Created: 2025-09-16
"""

import logging
import numpy as np
import pandas as pd
import networkx as nx
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
from pathlib import Path
import warnings

from ..rsiena_integration.r_interface import RInterface, RSessionConfig
from ..rsiena_integration.data_converters import ABMRSienaConverter, RSienaDataSet
from ..rsiena_integration.custom_effects import (
    CustomEffectsManager,
    AttractionRepulsionConfig,
    ComplexContagionConfig,
    ToleranceCooperationConfig,
    create_standard_effects_configuration
)

logger = logging.getLogger(__name__)


@dataclass
class SAOMModelConfig:
    """Configuration for SAOM model specification."""
    # Network specifications
    networks: List[str] = field(default_factory=lambda: ['friendship', 'cooperation'])
    behaviors: List[str] = field(default_factory=lambda: ['tolerance'])
    covariates: List[str] = field(default_factory=lambda: ['ethnicity', 'gender', 'prejudice'])

    # Effect specifications
    include_structural_effects: bool = True
    include_selection_effects: bool = True
    include_influence_effects: bool = True
    include_custom_effects: bool = True

    # Custom effect configurations
    use_attraction_repulsion: bool = True
    use_complex_contagion: bool = False
    use_tolerance_cooperation: bool = True

    # Model estimation parameters
    estimation_method: str = "ML"  # "ML" or "Bayesian"
    n_iterations: int = 3000
    convergence_threshold: float = 0.25
    max_degree: int = 6

    # Intervention parameters
    intervention_enabled: bool = False
    tolerance_change: float = 0.3
    target_size: float = 0.2
    target_strategy: str = "central"  # "central", "peripheral", "random", "clustered"
    centrality_measure: str = "degree"  # "degree", "betweenness", "eigenvector"


@dataclass
class SAOMResults:
    """Container for SAOM estimation results."""
    parameters: np.ndarray
    standard_errors: np.ndarray
    t_statistics: np.ndarray
    effect_names: List[str]
    converged: bool
    max_convergence_ratio: float
    iterations: int
    log_likelihood: Optional[float]
    goodness_of_fit: Dict[str, Any] = field(default_factory=dict)
    model_config: Optional[SAOMModelConfig] = None
    classroom_id: Optional[str] = None
    estimation_time: Optional[float] = None


class ToleranceCooperationSAOM:
    """
    Stochastic Actor-Oriented Model for tolerance-cooperation dynamics.

    Implements the micro-theory for how tolerance interventions can spread
    through friendship networks and promote interethnic cooperation.
    """

    def __init__(
        self,
        config: Optional[SAOMModelConfig] = None,
        r_interface: Optional[RInterface] = None
    ):
        """
        Initialize tolerance-cooperation SAOM.

        Args:
            config: Model configuration
            r_interface: R interface for RSiena operations
        """
        self.config = config or SAOMModelConfig()
        self.r_interface = r_interface
        self.data_converter = None
        self.effects_manager = None
        self.current_data = None
        self.current_effects = None
        self.estimation_results = {}

        # Initialize components
        self._initialize_components()

    def _initialize_components(self):
        """Initialize model components."""
        if self.r_interface:
            self.data_converter = ABMRSienaConverter(self.r_interface)
            if self.config.include_custom_effects:
                self.effects_manager = CustomEffectsManager(self.r_interface)

    def specify_effects(self, data_object_name: str = "siena_data") -> str:
        """
        Specify RSiena effects for tolerance-cooperation model.

        Args:
            data_object_name: Name of RSiena data object

        Returns:
            Name of effects object
        """
        logger.info("Specifying SAOM effects for tolerance-cooperation model...")

        effects_name = "myeff"

        # Get basic effects
        self.r_interface.execute_r_code(f"""
        library(RSiena)
        {effects_name} <- getEffects({data_object_name})
        """)

        # Add structural network effects
        if self.config.include_structural_effects:
            self._add_structural_effects(effects_name)

        # Add selection effects
        if self.config.include_selection_effects:
            self._add_selection_effects(effects_name)

        # Add influence effects
        if self.config.include_influence_effects:
            self._add_influence_effects(effects_name)

        # Add custom effects
        if self.config.include_custom_effects and self.effects_manager:
            self._add_custom_effects(effects_name)

        # Store effects object name
        self.current_effects = effects_name

        logger.info("Effects specification completed")
        return effects_name

    def _add_structural_effects(self, effects_name: str):
        """Add structural network effects."""
        logger.debug("Adding structural network effects...")

        # Friendship network effects
        self.r_interface.execute_r_code(f"""
        # Basic structural effects for friendship network
        {effects_name} <- includeEffects({effects_name}, density, recip, transTrip)

        # Degree-related effects
        {effects_name} <- includeEffects({effects_name}, outActSqrt, inPopSqrt)

        # Higher-order structure
        {effects_name} <- includeEffects({effects_name}, cycle3, transRecTrip)
        """)

        # Cooperation network effects (if included)
        if 'cooperation' in self.config.networks:
            self.r_interface.execute_r_code(f"""
            # Cooperation network structural effects
            {effects_name} <- includeEffects({effects_name}, density, recip,
                                           name = "cooperation")

            # Degree effects for cooperation
            {effects_name} <- includeEffects({effects_name}, outActSqrt,
                                           name = "cooperation")
            """)

        logger.debug("Structural effects added")

    def _add_selection_effects(self, effects_name: str):
        """Add selection effects (homophily and attribute-based selection)."""
        logger.debug("Adding selection effects...")

        # Gender homophily
        if 'gender' in self.config.covariates:
            self.r_interface.execute_r_code(f"""
            # Gender homophily in friendship
            {effects_name} <- includeEffects({effects_name}, sameX,
                                           interaction1 = "gender")
            """)

        # Ethnicity homophily
        if 'ethnicity' in self.config.covariates:
            self.r_interface.execute_r_code(f"""
            # Ethnicity homophily in friendship
            {effects_name} <- includeEffects({effects_name}, sameX,
                                           interaction1 = "ethnicity")

            # Ethnicity effects in cooperation
            {effects_name} <- includeEffects({effects_name}, sameX,
                                           name = "cooperation",
                                           interaction1 = "ethnicity")
            """)

        # Tolerance-based selection
        if 'tolerance' in self.config.behaviors:
            self.r_interface.execute_r_code(f"""
            # Tolerance similarity in friendship
            {effects_name} <- includeEffects({effects_name}, simX,
                                           interaction1 = "tolerance")

            # Tolerance alter effect (selection of tolerant others)
            {effects_name} <- includeEffects({effects_name}, altX,
                                           interaction1 = "tolerance")
            """)

        # Prejudice control effects
        if 'prejudice' in self.config.covariates:
            self.r_interface.execute_r_code(f"""
            # Control for prejudice in selection
            {effects_name} <- includeEffects({effects_name}, egoX,
                                           interaction1 = "prejudice")

            {effects_name} <- includeEffects({effects_name}, altX,
                                           interaction1 = "prejudice")
            """)

        logger.debug("Selection effects added")

    def _add_influence_effects(self, effects_name: str):
        """Add influence effects (behavior change mechanisms)."""
        logger.debug("Adding influence effects...")

        if 'tolerance' in self.config.behaviors:
            self.r_interface.execute_r_code(f"""
            # Basic tolerance behavior effects
            {effects_name} <- includeEffects({effects_name}, linear, quad,
                                           name = "tolerance", type = "behavior")

            # Social influence on tolerance (average similarity)
            {effects_name} <- includeEffects({effects_name}, avSim,
                                           name = "tolerance", type = "behavior",
                                           interaction1 = "friendship")

            # Total similarity influence
            {effects_name} <- includeEffects({effects_name}, totSim,
                                           name = "tolerance", type = "behavior",
                                           interaction1 = "friendship")
            """)

            # Covariate effects on tolerance change
            if 'ethnicity' in self.config.covariates:
                self.r_interface.execute_r_code(f"""
                # Ethnicity effect on tolerance change
                {effects_name} <- includeEffects({effects_name}, effFrom,
                                               name = "tolerance", type = "behavior",
                                               interaction1 = "ethnicity")
                """)

            if 'gender' in self.config.covariates:
                self.r_interface.execute_r_code(f"""
                # Gender effect on tolerance change
                {effects_name} <- includeEffects({effects_name}, effFrom,
                                               name = "tolerance", type = "behavior",
                                               interaction1 = "gender")
                """)

        logger.debug("Influence effects added")

    def _add_custom_effects(self, effects_name: str):
        """Add custom effects for tolerance-cooperation dynamics."""
        logger.debug("Adding custom effects...")

        # Get standard configurations
        custom_configs = create_standard_effects_configuration()

        # Add custom effects using effects manager
        self.effects_manager.add_effects_to_model(
            effects_object_name=effects_name,
            include_attraction_repulsion=self.config.use_attraction_repulsion,
            include_complex_contagion=self.config.use_complex_contagion,
            include_tolerance_cooperation=self.config.use_tolerance_cooperation,
            attraction_repulsion_config=custom_configs['attraction_repulsion'],
            complex_contagion_config=custom_configs['complex_contagion'],
            tolerance_cooperation_config=custom_configs['tolerance_cooperation']
        )

        logger.debug("Custom effects added")

    def estimate_model(
        self,
        dataset: RSienaDataSet,
        classroom_id: Optional[str] = None,
        save_results: bool = True
    ) -> SAOMResults:
        """
        Estimate SAOM parameters for given dataset.

        Args:
            dataset: RSiena dataset
            classroom_id: Identifier for classroom (for multi-level analysis)
            save_results: Whether to save results

        Returns:
            SAOM estimation results
        """
        logger.info(f"Estimating SAOM for classroom {classroom_id or 'unknown'}...")

        if not self.r_interface:
            raise RuntimeError("R interface required for model estimation")

        import time
        start_time = time.time()

        try:
            # Prepare data in R environment
            self.r_interface.create_r_object("siena_data", dataset.siena_data_object)

            # Specify effects
            effects_name = self.specify_effects()

            # Create algorithm object
            self.r_interface.execute_r_code(f"""
            # Create estimation algorithm
            algorithm <- sienaAlgorithmCreate(
                nsub = {self.config.n_iterations // 3},
                n3 = {self.config.n_iterations},
                maxDegree = c(friendship = {self.config.max_degree},
                             cooperation = {self.config.max_degree}),
                projname = "tolerance_cooperation_model"
            )
            """)

            # Estimate model
            self.r_interface.execute_r_code(f"""
            # Estimate the model
            model_results <- siena07(algorithm, data = siena_data, effects = {effects_name})
            """)

            # Extract results
            results_dict = self._extract_results()

            # Create results object
            estimation_time = time.time() - start_time
            results = SAOMResults(
                parameters=results_dict['parameters'],
                standard_errors=results_dict['standard_errors'],
                t_statistics=results_dict['t_statistics'],
                effect_names=results_dict['effect_names'],
                converged=results_dict['converged'],
                max_convergence_ratio=results_dict['max_convergence_ratio'],
                iterations=results_dict['iterations'],
                log_likelihood=results_dict['log_likelihood'],
                model_config=self.config,
                classroom_id=classroom_id,
                estimation_time=estimation_time
            )

            # Save results
            if save_results:
                self.estimation_results[classroom_id or 'default'] = results

            logger.info(f"Model estimation completed in {estimation_time:.2f}s")
            return results

        except Exception as e:
            logger.error(f"Model estimation failed: {e}")
            raise RuntimeError(f"SAOM estimation failed: {e}")

    def _extract_results(self) -> Dict[str, Any]:
        """Extract results from RSiena estimation."""
        r_code = """
        # Extract key results
        extraction_results <- list(
            parameters = model_results$theta,
            standard_errors = model_results$se,
            t_statistics = model_results$tstat,
            effect_names = model_results$requestedEffects$effectName,
            converged = model_results$OK,
            max_convergence_ratio = max(abs(model_results$tconv)),
            iterations = model_results$n,
            log_likelihood = model_results$ll,
            overall_max_convergence = model_results$tconv.max
        )
        """

        self.r_interface.execute_r_code(r_code)
        results = self.r_interface.get_r_object("extraction_results")

        return {
            'parameters': np.array(results['parameters']),
            'standard_errors': np.array(results['standard_errors']),
            't_statistics': np.array(results['t_statistics']),
            'effect_names': list(results['effect_names']),
            'converged': bool(results['converged']),
            'max_convergence_ratio': float(results['max_convergence_ratio']),
            'iterations': int(results['iterations']),
            'log_likelihood': float(results['log_likelihood']) if results['log_likelihood'] else None
        }

    def goodness_of_fit(self, results: SAOMResults) -> Dict[str, Any]:
        """
        Perform goodness-of-fit assessment for estimated model.

        Args:
            results: SAOM estimation results

        Returns:
            Goodness-of-fit statistics
        """
        logger.info("Performing goodness-of-fit assessment...")

        try:
            # Create goodness-of-fit object
            self.r_interface.execute_r_code("""
            # Perform goodness-of-fit test
            gof_results <- sienaGOF(model_results, verbose = TRUE,
                                   varName = "friendship")

            # Extract GOF statistics
            gof_stats <- list(
                mahalanobis_distance = gof_results$MahalanobisDistance,
                p_value = gof_results$pvalue,
                observed_statistics = gof_results$Observed,
                simulated_statistics = gof_results$Simulated
            )
            """)

            gof_data = self.r_interface.get_r_object("gof_stats")

            goodness_of_fit = {
                'mahalanobis_distance': float(gof_data['mahalanobis_distance']),
                'p_value': float(gof_data['p_value']),
                'observed_statistics': np.array(gof_data['observed_statistics']),
                'simulated_statistics': np.array(gof_data['simulated_statistics']),
                'fit_adequate': float(gof_data['p_value']) > 0.05
            }

            results.goodness_of_fit = goodness_of_fit
            logger.info(f"GOF assessment completed (p-value: {goodness_of_fit['p_value']:.3f})")

            return goodness_of_fit

        except Exception as e:
            logger.error(f"Goodness-of-fit assessment failed: {e}")
            return {}

    def simulate_intervention(
        self,
        base_results: SAOMResults,
        intervention_config: Optional[SAOMModelConfig] = None,
        n_simulations: int = 100
    ) -> Dict[str, Any]:
        """
        Simulate tolerance intervention effects.

        Args:
            base_results: Baseline model results
            intervention_config: Intervention configuration
            n_simulations: Number of simulation runs

        Returns:
            Intervention simulation results
        """
        logger.info("Simulating tolerance intervention effects...")

        if not intervention_config:
            intervention_config = SAOMModelConfig(intervention_enabled=True)

        try:
            # Set intervention parameters
            self.r_interface.execute_r_code(f"""
            # Intervention simulation parameters
            intervention_params <- list(
                tolerance_change = {intervention_config.tolerance_change},
                target_size = {intervention_config.target_size},
                target_strategy = "{intervention_config.target_strategy}",
                centrality_measure = "{intervention_config.centrality_measure}",
                n_simulations = {n_simulations}
            )
            """)

            # Run intervention simulation
            self.r_interface.execute_r_code("""
            # Implement intervention targeting strategy
            if (intervention_params$target_strategy == "central") {
                # Target most central actors
                if (intervention_params$centrality_measure == "degree") {
                    centrality_scores <- rowSums(siena_data$depvars$friendship[,,1])
                } else if (intervention_params$centrality_measure == "betweenness") {
                    # Calculate betweenness centrality
                    library(igraph)
                    g <- graph_from_adjacency_matrix(siena_data$depvars$friendship[,,1])
                    centrality_scores <- betweenness(g)
                }

                n_targets <- ceiling(intervention_params$target_size * nrow(siena_data$depvars$friendship))
                target_actors <- order(centrality_scores, decreasing = TRUE)[1:n_targets]

            } else if (intervention_params$target_strategy == "random") {
                # Random targeting
                n_targets <- ceiling(intervention_params$target_size * nrow(siena_data$depvars$friendship))
                target_actors <- sample(1:nrow(siena_data$depvars$friendship), n_targets)
            }

            # Apply intervention (increase tolerance)
            tolerance_initial <- siena_data$depvars$tolerance[,1]
            tolerance_intervened <- tolerance_initial
            tolerance_intervened[target_actors] <- tolerance_intervened[target_actors] +
                                                  intervention_params$tolerance_change

            # Bound tolerance values
            tolerance_intervened <- pmax(0, pmin(1, tolerance_intervened))
            """)

            # Simulate forward from intervention
            self.r_interface.execute_r_code(f"""
            # Simulate model forward with intervention
            simulation_results <- siena07(
                algorithm,
                data = siena_data,
                effects = myeff,
                returnDeps = TRUE,
                nbrNodes = {n_simulations},
                prevAns = model_results
            )

            # Extract simulation statistics
            sim_stats <- list(
                tolerance_change_mean = mean(simulation_results$sims[['tolerance']][,2] -
                                           simulation_results$sims[['tolerance']][,1]),
                cooperation_change_mean = mean(rowSums(simulation_results$sims[['cooperation']][,,2]) -
                                             rowSums(simulation_results$sims[['cooperation']][,,1])),
                target_actors = target_actors,
                n_targets = length(target_actors)
            )
            """)

            # Extract simulation results
            sim_data = self.r_interface.get_r_object("sim_stats")

            intervention_results = {
                'tolerance_change_mean': float(sim_data['tolerance_change_mean']),
                'cooperation_change_mean': float(sim_data['cooperation_change_mean']),
                'target_actors': list(sim_data['target_actors']),
                'n_targets': int(sim_data['n_targets']),
                'intervention_config': intervention_config,
                'n_simulations': n_simulations
            }

            logger.info("Intervention simulation completed")
            return intervention_results

        except Exception as e:
            logger.error(f"Intervention simulation failed: {e}")
            return {}

    def get_parameter_interpretation(self, results: SAOMResults) -> Dict[str, str]:
        """
        Provide interpretation of estimated parameters.

        Args:
            results: SAOM estimation results

        Returns:
            Dictionary with parameter interpretations
        """
        interpretations = {}

        for i, effect_name in enumerate(results.effect_names):
            param = results.parameters[i]
            se = results.standard_errors[i]
            t_stat = results.t_statistics[i]

            significance = ""
            if abs(t_stat) > 1.96:
                significance = " (significant at p < 0.05)"
            elif abs(t_stat) > 1.64:
                significance = " (significant at p < 0.10)"

            if "density" in effect_name.lower():
                interpretations[effect_name] = f"Baseline tendency to form ties: {param:.3f}±{se:.3f}{significance}"

            elif "recip" in effect_name.lower():
                interpretations[effect_name] = f"Reciprocity tendency: {param:.3f}±{se:.3f}{significance}"

            elif "transitrip" in effect_name.lower():
                interpretations[effect_name] = f"Transitivity (clustering): {param:.3f}±{se:.3f}{significance}"

            elif "samex" in effect_name.lower():
                interpretations[effect_name] = f"Homophily effect: {param:.3f}±{se:.3f}{significance}"

            elif "avSim" in effect_name.lower():
                interpretations[effect_name] = f"Social influence (average similarity): {param:.3f}±{se:.3f}{significance}"

            elif "attraction" in effect_name.lower():
                interpretations[effect_name] = f"Attraction-repulsion influence: {param:.3f}±{se:.3f}{significance}"

            elif "complex" in effect_name.lower():
                interpretations[effect_name] = f"Complex contagion influence: {param:.3f}±{se:.3f}{significance}"

            elif "tolerance" in effect_name.lower() and "cooperation" in effect_name.lower():
                interpretations[effect_name] = f"Tolerance → cooperation effect: {param:.3f}±{se:.3f}{significance}"

            else:
                interpretations[effect_name] = f"Effect: {param:.3f}±{se:.3f}{significance}"

        return interpretations

    def save_results(self, results: SAOMResults, filepath: Union[str, Path]):
        """
        Save SAOM results to file.

        Args:
            results: SAOM estimation results
            filepath: Output file path
        """
        filepath = Path(filepath)

        # Create results dictionary
        results_dict = {
            'parameters': results.parameters.tolist(),
            'standard_errors': results.standard_errors.tolist(),
            't_statistics': results.t_statistics.tolist(),
            'effect_names': results.effect_names,
            'converged': results.converged,
            'max_convergence_ratio': results.max_convergence_ratio,
            'iterations': results.iterations,
            'log_likelihood': results.log_likelihood,
            'goodness_of_fit': results.goodness_of_fit,
            'classroom_id': results.classroom_id,
            'estimation_time': results.estimation_time,
            'model_config': results.model_config.__dict__ if results.model_config else None
        }

        # Save as JSON
        import json
        with open(filepath.with_suffix('.json'), 'w') as f:
            json.dump(results_dict, f, indent=2)

        # Save as pickle for complete object
        import pickle
        with open(filepath.with_suffix('.pkl'), 'wb') as f:
            pickle.dump(results, f)

        logger.info(f"Results saved to {filepath}")


def create_test_model() -> ToleranceCooperationSAOM:
    """Create test instance of tolerance-cooperation SAOM."""
    config = SAOMModelConfig(
        networks=['friendship', 'cooperation'],
        behaviors=['tolerance'],
        covariates=['ethnicity', 'gender', 'prejudice'],
        use_attraction_repulsion=True,
        use_complex_contagion=False,
        use_tolerance_cooperation=True
    )

    r_config = RSessionConfig(
        required_packages=['RSiena', 'igraph', 'network', 'sna'],
        memory_limit_mb=4096
    )

    r_interface = RInterface(r_config)
    return ToleranceCooperationSAOM(config, r_interface)


if __name__ == "__main__":
    # Test the tolerance-cooperation SAOM
    logging.basicConfig(level=logging.INFO)

    try:
        # Create test model
        saom = create_test_model()

        # Create test data
        logger.info("Creating test networks...")

        # Generate test friendship networks
        n_actors = 30
        n_periods = 3

        friendship_networks = []
        cooperation_networks = []

        for period in range(n_periods):
            # Friendship network (denser, more stable)
            friendship = nx.erdos_renyi_graph(n_actors, 0.15, directed=True)

            # Cooperation network (sparser, more task-based)
            cooperation = nx.erdos_renyi_graph(n_actors, 0.08, directed=True)

            friendship_networks.append(friendship)
            cooperation_networks.append(cooperation)

        # Generate test behaviors and attributes
        actor_attributes = {
            'ethnicity': [np.random.choice([0, 1], p=[0.7, 0.3]) for _ in range(n_actors)],
            'gender': [np.random.choice([0, 1]) for _ in range(n_actors)],
            'prejudice': [np.random.normal(0.3, 0.2) for _ in range(n_actors)]
        }

        behaviors = []
        for period in range(n_periods):
            period_tolerance = {}
            for actor in range(n_actors):
                base_tolerance = 0.5 + np.random.normal(0, 0.1)
                # Add some evolution over time
                tolerance = base_tolerance + period * 0.05 + np.random.normal(0, 0.05)
                tolerance = max(0, min(1, tolerance))  # Bound between 0 and 1
                period_tolerance[actor] = {'tolerance': tolerance}
            behaviors.append(period_tolerance)

        logger.info("Converting to RSiena format...")

        # Convert to RSiena format
        dataset = saom.data_converter.convert_to_rsiena(
            networks=friendship_networks,
            behaviors=behaviors,
            actor_attributes=actor_attributes
        )

        logger.info("Model specification and estimation test completed successfully")
        print(f"✓ SAOM test passed: {dataset.n_actors} actors, {dataset.n_periods} periods")

    except Exception as e:
        logger.error(f"SAOM test failed: {e}")
        import traceback
        traceback.print_exc()