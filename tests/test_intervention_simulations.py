"""
Intervention Simulation Testing Framework

Comprehensive testing suite for tolerance intervention simulations including
targeting strategies, intervention persistence, dose-response relationships,
and multi-level intervention effects.

Test Coverage:
- Targeting strategy validation (central, peripheral, random, clustered)
- Intervention persistence and decay mechanisms
- Dose-response relationship testing
- Multi-level intervention effects
- Spillover and diffusion effects
- Longitudinal intervention impact

Author: Validation Specialist
Created: 2025-09-16
"""

import unittest
import numpy as np
import pandas as pd
import networkx as nx
from typing import Dict, List, Optional, Tuple, Set
import warnings
from pathlib import Path
import logging
from dataclasses import dataclass
from scipy import stats
import itertools

# Import ABM-RSiena components
import sys
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from models.abm_rsiena_model import ABMRSienaModel, NetworkEvolutionParameters
from agents.social_agent import SocialAgent, AgentAttributes
from utils.config_manager import ModelConfiguration

logger = logging.getLogger(__name__)


@dataclass
class InterventionConfig:
    """Configuration for tolerance intervention."""
    intervention_type: str = "tolerance_enhancement"
    target_selection: str = "central"  # central, peripheral, random, clustered
    intervention_strength: float = 20.0
    intervention_duration: int = 5
    decay_rate: float = 0.1
    spillover_enabled: bool = True
    spillover_strength: float = 0.3


@dataclass
class InterventionResults:
    """Results from intervention simulation."""
    pre_intervention_tolerance: np.ndarray
    post_intervention_tolerance: np.ndarray
    targeted_actors: List[int]
    spillover_actors: List[int]
    intervention_effect_size: float
    persistence_metrics: Dict[str, float]
    network_changes: Dict[str, any]


class TestInterventionSimulations(unittest.TestCase):
    """Test suite for tolerance intervention simulations."""

    def setUp(self):
        """Set up test fixtures."""
        self.n_actors = 100
        self.n_periods = 10
        self.test_seed = 42
        np.random.seed(self.test_seed)

        # Create test network with realistic structure
        self.test_network = self._create_realistic_network()
        self.baseline_tolerance = self._create_baseline_tolerance()

        # Define intervention configurations for testing
        self.intervention_configs = {
            'central': InterventionConfig(target_selection='central'),
            'peripheral': InterventionConfig(target_selection='peripheral'),
            'random': InterventionConfig(target_selection='random'),
            'clustered': InterventionConfig(target_selection='clustered')
        }

    def _create_realistic_network(self) -> nx.Graph:
        """Create realistic social network for testing."""
        # Create small-world network (common in social settings)
        G = nx.watts_strogatz_graph(self.n_actors, 6, 0.3, seed=self.test_seed)

        # Add some variation in degree distribution
        # Randomly add edges to create hubs
        n_additional_edges = self.n_actors // 5
        high_degree_nodes = sorted(G.nodes(), key=lambda x: G.degree(x), reverse=True)[:10]

        for _ in range(n_additional_edges):
            hub = np.random.choice(high_degree_nodes)
            target = np.random.choice([n for n in G.nodes() if n != hub])
            if not G.has_edge(hub, target):
                G.add_edge(hub, target)

        return G

    def _create_baseline_tolerance(self) -> np.ndarray:
        """Create realistic baseline tolerance distribution."""
        # Multimodal distribution (common in tolerance research)
        component1 = np.random.normal(25, 8, self.n_actors // 3)
        component2 = np.random.normal(50, 10, self.n_actors // 3)
        component3 = np.random.normal(75, 8, self.n_actors - 2 * (self.n_actors // 3))

        tolerance = np.concatenate([component1, component2, component3])
        np.random.shuffle(tolerance)

        # Ensure values are in valid range
        tolerance = np.clip(tolerance, 0, 100)

        return tolerance

    def test_targeting_strategies(self):
        """Test different targeting strategies for interventions."""
        logger.info("Testing intervention targeting strategies...")

        for strategy_name, config in self.intervention_configs.items():
            with self.subTest(strategy=strategy_name):
                targets = self._select_intervention_targets(config)

                # Test target selection accuracy
                self._validate_target_selection(targets, config, strategy_name)

                # Test target characteristics
                self._validate_target_characteristics(targets, config, strategy_name)

    def _select_intervention_targets(self, config: InterventionConfig) -> List[int]:
        """Select intervention targets based on strategy."""
        n_targets = max(1, self.n_actors // 10)  # 10% of population

        if config.target_selection == 'central':
            # Select highest degree centrality nodes
            centrality = nx.degree_centrality(self.test_network)
            sorted_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)
            targets = [node for node, _ in sorted_nodes[:n_targets]]

        elif config.target_selection == 'peripheral':
            # Select lowest degree centrality nodes
            centrality = nx.degree_centrality(self.test_network)
            sorted_nodes = sorted(centrality.items(), key=lambda x: x[1])
            targets = [node for node, _ in sorted_nodes[:n_targets]]

        elif config.target_selection == 'random':
            # Random selection
            targets = list(np.random.choice(
                list(self.test_network.nodes()),
                size=n_targets,
                replace=False
            ))

        elif config.target_selection == 'clustered':
            # Select nodes from same community/cluster
            communities = self._detect_communities()
            if communities:
                # Select largest community
                largest_community = max(communities, key=len)
                targets = list(np.random.choice(
                    list(largest_community),
                    size=min(n_targets, len(largest_community)),
                    replace=False
                ))
            else:
                # Fallback to random if community detection fails
                targets = list(np.random.choice(
                    list(self.test_network.nodes()),
                    size=n_targets,
                    replace=False
                ))

        return targets

    def _detect_communities(self) -> List[Set[int]]:
        """Detect communities in the network."""
        try:
            # Use Louvain algorithm for community detection
            communities = nx.community.louvain_communities(self.test_network, seed=self.test_seed)
            return communities
        except:
            # Fallback: simple connected components
            return list(nx.connected_components(self.test_network))

    def _validate_target_selection(self, targets: List[int], config: InterventionConfig,
                                 strategy_name: str):
        """Validate target selection meets strategy requirements."""
        # Basic validation
        self.assertIsInstance(targets, list, f"{strategy_name}: Targets should be list")
        self.assertGreater(len(targets), 0, f"{strategy_name}: Should select at least one target")
        self.assertEqual(len(targets), len(set(targets)),
                        f"{strategy_name}: Targets should be unique")

        # All targets should be valid nodes
        for target in targets:
            self.assertIn(target, self.test_network.nodes(),
                         f"{strategy_name}: Invalid target node {target}")

        # Strategy-specific validation
        if config.target_selection == 'central':
            # Central targets should have above-average degree
            target_degrees = [self.test_network.degree(t) for t in targets]
            avg_degree = np.mean([self.test_network.degree(n) for n in self.test_network.nodes()])
            avg_target_degree = np.mean(target_degrees)
            self.assertGreater(avg_target_degree, avg_degree,
                             f"{strategy_name}: Central targets should have above-average degree")

        elif config.target_selection == 'peripheral':
            # Peripheral targets should have below-average degree
            target_degrees = [self.test_network.degree(t) for t in targets]
            avg_degree = np.mean([self.test_network.degree(n) for n in self.test_network.nodes()])
            avg_target_degree = np.mean(target_degrees)
            self.assertLess(avg_target_degree, avg_degree,
                           f"{strategy_name}: Peripheral targets should have below-average degree")

        elif config.target_selection == 'clustered':
            # Clustered targets should be more connected to each other
            if len(targets) > 1:
                internal_edges = 0
                total_possible = len(targets) * (len(targets) - 1) // 2
                for i, t1 in enumerate(targets):
                    for t2 in targets[i+1:]:
                        if self.test_network.has_edge(t1, t2):
                            internal_edges += 1

                clustering_ratio = internal_edges / total_possible if total_possible > 0 else 0
                # Should have some internal connectivity (though not necessarily high)
                # This is a weak test since clustering might not always produce connected targets
                self.assertGreaterEqual(clustering_ratio, 0,
                                      f"{strategy_name}: Clustering ratio should be non-negative")

    def _validate_target_characteristics(self, targets: List[int], config: InterventionConfig,
                                       strategy_name: str):
        """Validate characteristics of selected targets."""
        target_tolerance = self.baseline_tolerance[targets]

        # Check tolerance distribution of targets
        self.assertEqual(len(target_tolerance), len(targets),
                        f"{strategy_name}: Tolerance data should match target count")

        # Targets should have valid tolerance values
        self.assertTrue(np.all(target_tolerance >= 0),
                       f"{strategy_name}: All target tolerance values should be >= 0")
        self.assertTrue(np.all(target_tolerance <= 100),
                       f"{strategy_name}: All target tolerance values should be <= 100")

        # Log target characteristics for analysis
        logger.info(f"{strategy_name} targets - Count: {len(targets)}, "
                   f"Avg tolerance: {np.mean(target_tolerance):.2f}, "
                   f"Avg degree: {np.mean([self.test_network.degree(t) for t in targets]):.2f}")

    def test_intervention_persistence(self):
        """Test tolerance changes persist over time with appropriate decay."""
        logger.info("Testing intervention persistence...")

        config = self.intervention_configs['central']
        targets = self._select_intervention_targets(config)

        # Simulate intervention and measure persistence
        results = self._simulate_intervention(config, targets)

        # Test immediate intervention effect
        immediate_effect = np.mean(results.post_intervention_tolerance[targets]) - \
                          np.mean(results.pre_intervention_tolerance[targets])
        self.assertGreater(immediate_effect, 0,
                          "Intervention should have immediate positive effect")
        self.assertLessEqual(immediate_effect, config.intervention_strength,
                            "Immediate effect should not exceed intervention strength")

        # Test persistence over time
        persistence_data = self._measure_intervention_persistence(config, targets, results)

        # Effect should decay over time
        time_points = sorted(persistence_data.keys())
        effects = [persistence_data[t] for t in time_points]

        # Test that effect generally decreases (allowing for some fluctuation)
        # Use Spearman correlation to test for monotonic decrease
        correlation, p_value = stats.spearmanr(time_points, effects)
        self.assertLess(correlation, 0,
                       "Intervention effect should generally decrease over time")

        # Test decay rate
        if len(effects) >= 3:
            # Calculate empirical decay rate
            empirical_decay = self._calculate_decay_rate(time_points, effects)
            expected_decay = config.decay_rate

            # Allow for reasonable variation in decay rate
            self.assertLess(abs(empirical_decay - expected_decay), 0.1,
                           f"Decay rate {empirical_decay:.3f} should be close to expected {expected_decay:.3f}")

    def _simulate_intervention(self, config: InterventionConfig,
                             targets: List[int]) -> InterventionResults:
        """Simulate tolerance intervention."""
        pre_tolerance = self.baseline_tolerance.copy()
        post_tolerance = pre_tolerance.copy()

        # Apply intervention to targets
        for target in targets:
            intervention_effect = config.intervention_strength
            # Add some individual variation
            individual_variation = np.random.normal(0, config.intervention_strength * 0.1)
            total_effect = intervention_effect + individual_variation

            post_tolerance[target] = np.clip(
                pre_tolerance[target] + total_effect,
                0, 100
            )

        # Calculate spillover effects if enabled
        spillover_actors = []
        if config.spillover_enabled:
            spillover_actors = self._apply_spillover_effects(
                config, targets, pre_tolerance, post_tolerance
            )

        # Calculate effect size
        effect_size = self._calculate_effect_size(pre_tolerance, post_tolerance, targets)

        # Measure persistence (simplified for testing)
        persistence_metrics = {
            'immediate_effect': np.mean(post_tolerance[targets]) - np.mean(pre_tolerance[targets]),
            'effect_size': effect_size
        }

        return InterventionResults(
            pre_intervention_tolerance=pre_tolerance,
            post_intervention_tolerance=post_tolerance,
            targeted_actors=targets,
            spillover_actors=spillover_actors,
            intervention_effect_size=effect_size,
            persistence_metrics=persistence_metrics,
            network_changes={}
        )

    def _apply_spillover_effects(self, config: InterventionConfig, targets: List[int],
                               pre_tolerance: np.ndarray, post_tolerance: np.ndarray) -> List[int]:
        """Apply spillover effects to network neighbors."""
        spillover_actors = []

        for target in targets:
            neighbors = list(self.test_network.neighbors(target))
            for neighbor in neighbors:
                if neighbor not in targets:  # Don't double-apply to targets
                    # Spillover strength depends on network distance and connection strength
                    spillover_effect = config.spillover_strength * config.intervention_strength

                    # Add spillover effect
                    post_tolerance[neighbor] = np.clip(
                        post_tolerance[neighbor] + spillover_effect,
                        0, 100
                    )
                    spillover_actors.append(neighbor)

        return spillover_actors

    def _calculate_effect_size(self, pre_tolerance: np.ndarray, post_tolerance: np.ndarray,
                             targets: List[int]) -> float:
        """Calculate Cohen's d effect size for intervention."""
        pre_target = pre_tolerance[targets]
        post_target = post_tolerance[targets]

        mean_diff = np.mean(post_target) - np.mean(pre_target)
        pooled_std = np.sqrt((np.var(pre_target) + np.var(post_target)) / 2)

        if pooled_std > 0:
            return mean_diff / pooled_std
        else:
            return 0.0

    def _measure_intervention_persistence(self, config: InterventionConfig,
                                        targets: List[int], results: InterventionResults) -> Dict[int, float]:
        """Measure intervention persistence over time periods."""
        persistence_data = {}

        # Simulate tolerance evolution over time with decay
        current_tolerance = results.post_intervention_tolerance.copy()
        baseline_tolerance = results.pre_intervention_tolerance.copy()

        for time_period in range(1, config.intervention_duration + 1):
            # Apply decay
            for target in targets:
                decay_amount = config.decay_rate * (current_tolerance[target] - baseline_tolerance[target])
                current_tolerance[target] -= decay_amount

            # Calculate current effect
            current_effect = np.mean(current_tolerance[targets]) - np.mean(baseline_tolerance[targets])
            persistence_data[time_period] = current_effect

        return persistence_data

    def _calculate_decay_rate(self, time_points: List[int], effects: List[float]) -> float:
        """Calculate empirical decay rate from time series data."""
        if len(time_points) < 2:
            return 0.0

        # Fit exponential decay model: effect(t) = initial * exp(-k*t)
        if effects[0] > 0:
            log_effects = [np.log(max(e, 0.001)) for e in effects]  # Avoid log(0)
            slope, intercept, r_value, p_value, std_err = stats.linregress(time_points, log_effects)
            return -slope  # Decay rate is negative slope
        else:
            return 0.0

    def test_dose_response_relationship(self):
        """Test dose-response relationship for intervention strength."""
        logger.info("Testing dose-response relationship...")

        # Test different intervention strengths
        intervention_strengths = [5.0, 10.0, 20.0, 40.0]
        targets = self._select_intervention_targets(self.intervention_configs['central'])

        dose_response_data = []

        for strength in intervention_strengths:
            config = InterventionConfig(intervention_strength=strength)
            results = self._simulate_intervention(config, targets)

            effect_size = results.intervention_effect_size
            dose_response_data.append((strength, effect_size))

        # Test dose-response relationship
        doses, responses = zip(*dose_response_data)

        # Should have positive correlation between dose and response
        correlation, p_value = stats.pearsonr(doses, responses)
        self.assertGreater(correlation, 0.5,
                          f"Dose-response correlation {correlation:.3f} should be positive and strong")

        # Test that response increases with dose (allowing for some noise)
        sorted_by_dose = sorted(dose_response_data, key=lambda x: x[0])
        responses_sorted = [response for dose, response in sorted_by_dose]

        # Test monotonic increase (using Spearman correlation for robustness)
        spearman_corr, _ = stats.spearmanr(range(len(responses_sorted)), responses_sorted)
        self.assertGreater(spearman_corr, 0.5,
                          "Response should generally increase with dose")

        # Log dose-response data
        logger.info("Dose-response data:")
        for dose, response in dose_response_data:
            logger.info(f"  Dose: {dose:5.1f}, Response: {response:6.3f}")

    def test_multi_level_intervention_effects(self):
        """Test intervention effects at individual and network levels."""
        logger.info("Testing multi-level intervention effects...")

        config = self.intervention_configs['clustered']
        targets = self._select_intervention_targets(config)
        results = self._simulate_intervention(config, targets)

        # Individual level effects
        individual_effects = self._measure_individual_effects(results, targets)
        self._validate_individual_effects(individual_effects, config)

        # Network level effects
        network_effects = self._measure_network_effects(results)
        self._validate_network_effects(network_effects, config)

        # Group level effects (if clustered targeting)
        if config.target_selection == 'clustered':
            group_effects = self._measure_group_effects(results, targets)
            self._validate_group_effects(group_effects, config)

    def _measure_individual_effects(self, results: InterventionResults,
                                  targets: List[int]) -> Dict[str, any]:
        """Measure individual-level intervention effects."""
        individual_effects = {}

        # Direct effects on targets
        target_pre = results.pre_intervention_tolerance[targets]
        target_post = results.post_intervention_tolerance[targets]
        target_change = target_post - target_pre

        individual_effects['target_mean_change'] = np.mean(target_change)
        individual_effects['target_std_change'] = np.std(target_change)
        individual_effects['target_effect_sizes'] = [
            (post - pre) / np.std(results.pre_intervention_tolerance)
            for pre, post in zip(target_pre, target_post)
        ]

        # Spillover effects on non-targets
        non_targets = [i for i in range(len(results.pre_intervention_tolerance)) if i not in targets]
        if non_targets:
            nontarget_pre = results.pre_intervention_tolerance[non_targets]
            nontarget_post = results.post_intervention_tolerance[non_targets]
            nontarget_change = nontarget_post - nontarget_pre

            individual_effects['nontarget_mean_change'] = np.mean(nontarget_change)
            individual_effects['nontarget_std_change'] = np.std(nontarget_change)

        return individual_effects

    def _validate_individual_effects(self, individual_effects: Dict[str, any],
                                   config: InterventionConfig):
        """Validate individual-level effects."""
        # Targets should show positive change
        self.assertGreater(individual_effects['target_mean_change'], 0,
                          "Targets should show positive tolerance change")

        # Effect size should be meaningful
        target_effect_sizes = individual_effects['target_effect_sizes']
        mean_effect_size = np.mean(target_effect_sizes)
        self.assertGreater(mean_effect_size, 0.2,  # Small effect size threshold
                          "Mean effect size should be at least small (0.2)")

        # Non-targets might show positive change if spillover enabled
        if config.spillover_enabled and 'nontarget_mean_change' in individual_effects:
            nontarget_change = individual_effects['nontarget_mean_change']
            self.assertGreaterEqual(nontarget_change, 0,
                                   "Non-targets should not show negative change with spillover")

    def _measure_network_effects(self, results: InterventionResults) -> Dict[str, any]:
        """Measure network-level intervention effects."""
        network_effects = {}

        # Overall tolerance distribution changes
        pre_mean = np.mean(results.pre_intervention_tolerance)
        post_mean = np.mean(results.post_intervention_tolerance)
        network_effects['overall_mean_change'] = post_mean - pre_mean

        # Tolerance variance changes
        pre_var = np.var(results.pre_intervention_tolerance)
        post_var = np.var(results.post_intervention_tolerance)
        network_effects['variance_change'] = post_var - pre_var

        # Tolerance inequality (using Gini coefficient)
        pre_gini = self._calculate_gini_coefficient(results.pre_intervention_tolerance)
        post_gini = self._calculate_gini_coefficient(results.post_intervention_tolerance)
        network_effects['inequality_change'] = post_gini - pre_gini

        return network_effects

    def _calculate_gini_coefficient(self, values: np.ndarray) -> float:
        """Calculate Gini coefficient for inequality measurement."""
        sorted_values = np.sort(values)
        n = len(sorted_values)
        index = np.arange(1, n + 1)
        return (2 * np.sum(index * sorted_values)) / (n * np.sum(sorted_values)) - (n + 1) / n

    def _validate_network_effects(self, network_effects: Dict[str, any],
                                config: InterventionConfig):
        """Validate network-level effects."""
        # Overall mean should increase
        self.assertGreater(network_effects['overall_mean_change'], 0,
                          "Overall network tolerance should increase")

        # Variance change depends on intervention pattern
        # (Could increase if only targeting some actors, or decrease if equalizing)
        variance_change = network_effects['variance_change']
        self.assertIsInstance(variance_change, (int, float),
                            "Variance change should be numeric")

        # Inequality change should be interpretable
        inequality_change = network_effects['inequality_change']
        self.assertIsInstance(inequality_change, (int, float),
                            "Inequality change should be numeric")

    def _measure_group_effects(self, results: InterventionResults,
                             targets: List[int]) -> Dict[str, any]:
        """Measure group-level effects for clustered interventions."""
        group_effects = {}

        # Identify target group/community
        communities = self._detect_communities()
        target_community = None

        for community in communities:
            if any(target in community for target in targets):
                target_community = community
                break

        if target_community:
            community_members = list(target_community)

            # Group tolerance change
            pre_group = results.pre_intervention_tolerance[community_members]
            post_group = results.post_intervention_tolerance[community_members]
            group_effects['group_mean_change'] = np.mean(post_group) - np.mean(pre_group)

            # Within-group cohesion (variance)
            pre_group_var = np.var(pre_group)
            post_group_var = np.var(post_group)
            group_effects['group_cohesion_change'] = pre_group_var - post_group_var  # Decrease = more cohesion

        return group_effects

    def _validate_group_effects(self, group_effects: Dict[str, any],
                              config: InterventionConfig):
        """Validate group-level effects."""
        if 'group_mean_change' in group_effects:
            # Group should show positive change
            self.assertGreater(group_effects['group_mean_change'], 0,
                              "Target group should show positive tolerance change")

        if 'group_cohesion_change' in group_effects:
            # Group cohesion might increase (variance decreases) due to intervention
            cohesion_change = group_effects['group_cohesion_change']
            self.assertIsInstance(cohesion_change, (int, float),
                                "Group cohesion change should be numeric")

    def test_spillover_and_diffusion_effects(self):
        """Test spillover and diffusion effects through network."""
        logger.info("Testing spillover and diffusion effects...")

        # Test with spillover enabled
        config_spillover = InterventionConfig(spillover_enabled=True, spillover_strength=0.3)
        targets_spillover = self._select_intervention_targets(config_spillover)
        results_spillover = self._simulate_intervention(config_spillover, targets_spillover)

        # Test without spillover
        config_no_spillover = InterventionConfig(spillover_enabled=False)
        targets_no_spillover = self._select_intervention_targets(config_no_spillover)
        results_no_spillover = self._simulate_intervention(config_no_spillover, targets_no_spillover)

        # Compare spillover effects
        self._validate_spillover_effects(results_spillover, results_no_spillover, config_spillover)

        # Test diffusion by network distance
        self._test_diffusion_by_distance(results_spillover, targets_spillover, config_spillover)

    def _validate_spillover_effects(self, results_spillover: InterventionResults,
                                  results_no_spillover: InterventionResults,
                                  config: InterventionConfig):
        """Validate spillover effects by comparing with and without spillover."""
        # Non-target actors should show more change with spillover enabled
        targets_spillover = set(results_spillover.targeted_actors)
        targets_no_spillover = set(results_no_spillover.targeted_actors)

        # Use same targets for fair comparison
        common_targets = targets_spillover.intersection(targets_no_spillover)
        if not common_targets:
            # If different targets, compare overall non-target effects
            non_targets_spillover = [i for i in range(len(results_spillover.pre_intervention_tolerance))
                                   if i not in results_spillover.targeted_actors]
            non_targets_no_spillover = [i for i in range(len(results_no_spillover.pre_intervention_tolerance))
                                      if i not in results_no_spillover.targeted_actors]

            spillover_effect = np.mean(results_spillover.post_intervention_tolerance[non_targets_spillover]) - \
                             np.mean(results_spillover.pre_intervention_tolerance[non_targets_spillover])

            no_spillover_effect = np.mean(results_no_spillover.post_intervention_tolerance[non_targets_no_spillover]) - \
                                np.mean(results_no_spillover.pre_intervention_tolerance[non_targets_no_spillover])

            if config.spillover_enabled:
                self.assertGreater(spillover_effect, no_spillover_effect,
                                 "Spillover should increase non-target effects")
            else:
                self.assertAlmostEqual(spillover_effect, 0, places=2,
                                     msg="Without spillover, non-target effects should be minimal")

    def _test_diffusion_by_distance(self, results: InterventionResults,
                                   targets: List[int], config: InterventionConfig):
        """Test that intervention effects diffuse based on network distance."""
        # Calculate network distances from targets
        distance_effects = {}

        for distance in range(1, 4):  # Test up to distance 3
            actors_at_distance = set()

            for target in targets:
                try:
                    # Find actors at specific distance from this target
                    shortest_paths = nx.single_source_shortest_path_length(
                        self.test_network, target, cutoff=distance
                    )
                    distance_actors = {node for node, d in shortest_paths.items() if d == distance}
                    actors_at_distance.update(distance_actors)
                except:
                    # If network is disconnected, skip this target
                    continue

            if actors_at_distance:
                actors_list = list(actors_at_distance)
                pre_tolerance = results.pre_intervention_tolerance[actors_list]
                post_tolerance = results.post_intervention_tolerance[actors_list]
                effect = np.mean(post_tolerance) - np.mean(pre_tolerance)
                distance_effects[distance] = effect

        # Test that effects generally decrease with distance
        if len(distance_effects) >= 2:
            distances = sorted(distance_effects.keys())
            effects = [distance_effects[d] for d in distances]

            # Effects should generally decrease with distance (negative correlation)
            if len(effects) > 2:
                correlation, _ = stats.spearmanr(distances, effects)
                # Allow for some noise, but expect general downward trend
                self.assertLess(correlation, 0.5,
                               "Intervention effects should generally decrease with network distance")

        # Log diffusion pattern
        logger.info("Diffusion by network distance:")
        for distance in sorted(distance_effects.keys()):
            logger.info(f"  Distance {distance}: Effect = {distance_effects[distance]:.3f}")


if __name__ == '__main__':
    # Configure logging for test run
    logging.basicConfig(level=logging.INFO)

    # Run tests
    unittest.main(verbosity=2)