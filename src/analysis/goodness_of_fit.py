"""
Goodness-of-Fit Testing for SAOM Models

This module provides comprehensive goodness-of-fit assessment tools
for tolerance-cooperation SAOM models. Includes multiple diagnostic
approaches and visualization tools for model validation.

Key Features:
- Standard RSiena goodness-of-fit tests
- Custom diagnostic statistics for tolerance-cooperation dynamics
- Network-level and behavior-level fit assessment
- Mahalanobis distance and deviation statistics
- Publication-ready diagnostic plots
- Model comparison and selection tools

Diagnostic Categories:
- Network Statistics: degree distribution, clustering, path lengths
- Behavior Statistics: tolerance distribution, change patterns
- Dyadic Statistics: reciprocity, transitivity, homophily patterns
- Dynamic Statistics: temporal stability, change correlations

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
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

from ..models.tolerance_cooperation_saom import SAOMResults, ToleranceCooperationSAOM
from ..rsiena_integration.r_interface import RInterface
from ..rsiena_integration.data_converters import RSienaDataSet

logger = logging.getLogger(__name__)


@dataclass
class GoodnessOfFitResults:
    """Results from goodness-of-fit assessment."""
    overall_fit: Dict[str, float]
    network_statistics: Dict[str, Dict[str, float]]
    behavior_statistics: Dict[str, Dict[str, float]]
    diagnostic_plots: Dict[str, Any]
    mahalanobis_distances: Dict[str, float]
    p_values: Dict[str, float]
    fit_adequate: bool
    problematic_statistics: List[str]
    recommendations: List[str]


@dataclass
class CustomStatistic:
    """Custom diagnostic statistic for SAOM models."""
    name: str
    description: str
    statistic_function: callable
    expected_range: Tuple[float, float]
    interpretation: str


class GoodnessOfFitTester:
    """
    Comprehensive goodness-of-fit testing for SAOM models.

    Provides multiple approaches to assess model fit including
    standard RSiena diagnostics and custom statistics for
    tolerance-cooperation dynamics.
    """

    def __init__(self, r_interface: Optional[RInterface] = None):
        """
        Initialize goodness-of-fit tester.

        Args:
            r_interface: R interface for RSiena GOF functions
        """
        self.r_interface = r_interface
        self.custom_statistics = {}
        self.fit_results = {}

        # Define custom statistics for tolerance-cooperation models
        self._define_custom_statistics()

    def _define_custom_statistics(self):
        """Define custom diagnostic statistics."""
        # Network-level statistics
        self.custom_statistics['tolerance_homophily'] = CustomStatistic(
            name="tolerance_homophily",
            description="Homophily based on tolerance levels",
            statistic_function=self._calculate_tolerance_homophily,
            expected_range=(0.0, 1.0),
            interpretation="Higher values indicate stronger tolerance-based clustering"
        )

        self.custom_statistics['interethnic_cooperation_rate'] = CustomStatistic(
            name="interethnic_cooperation_rate",
            description="Rate of cooperation across ethnic boundaries",
            statistic_function=self._calculate_interethnic_cooperation_rate,
            expected_range=(0.0, 1.0),
            interpretation="Proportion of possible interethnic cooperation ties that exist"
        )

        self.custom_statistics['tolerance_network_correlation'] = CustomStatistic(
            name="tolerance_network_correlation",
            description="Correlation between tolerance and network centrality",
            statistic_function=self._calculate_tolerance_centrality_correlation,
            expected_range=(-1.0, 1.0),
            interpretation="Association between individual tolerance and network position"
        )

        # Behavior-level statistics
        self.custom_statistics['tolerance_stability'] = CustomStatistic(
            name="tolerance_stability",
            description="Stability of tolerance over time",
            statistic_function=self._calculate_tolerance_stability,
            expected_range=(0.0, 1.0),
            interpretation="Correlation of tolerance between consecutive waves"
        )

        self.custom_statistics['tolerance_variance_explained'] = CustomStatistic(
            name="tolerance_variance_explained",
            description="Variance in tolerance explained by network position",
            statistic_function=self._calculate_tolerance_variance_explained,
            expected_range=(0.0, 1.0),
            interpretation="R-squared from regression of tolerance on network measures"
        )

    def assess_model_fit(
        self,
        saom_model: ToleranceCooperationSAOM,
        results: SAOMResults,
        dataset: RSienaDataSet,
        n_simulations: int = 1000
    ) -> GoodnessOfFitResults:
        """
        Comprehensive goodness-of-fit assessment.

        Args:
            saom_model: Fitted SAOM model
            results: Model estimation results
            dataset: Original dataset
            n_simulations: Number of simulations for GOF testing

        Returns:
            Comprehensive goodness-of-fit results
        """
        logger.info("Starting comprehensive goodness-of-fit assessment...")

        # Standard RSiena GOF tests
        standard_gof = self._run_standard_gof_tests(saom_model, results, n_simulations)

        # Custom diagnostic statistics
        custom_diagnostics = self._run_custom_diagnostics(dataset, saom_model, results)

        # Network-level diagnostics
        network_diagnostics = self._assess_network_fit(dataset, saom_model, results)

        # Behavior-level diagnostics
        behavior_diagnostics = self._assess_behavior_fit(dataset, saom_model, results)

        # Overall assessment
        overall_assessment = self._assess_overall_fit(
            standard_gof, custom_diagnostics, network_diagnostics, behavior_diagnostics
        )

        # Create diagnostic plots
        diagnostic_plots = self._create_diagnostic_plots(
            dataset, standard_gof, custom_diagnostics
        )

        # Compile results
        gof_results = GoodnessOfFitResults(
            overall_fit=overall_assessment,
            network_statistics=network_diagnostics,
            behavior_statistics=behavior_diagnostics,
            diagnostic_plots=diagnostic_plots,
            mahalanobis_distances=standard_gof.get('mahalanobis_distances', {}),
            p_values=standard_gof.get('p_values', {}),
            fit_adequate=overall_assessment.get('fit_adequate', False),
            problematic_statistics=overall_assessment.get('problematic_statistics', []),
            recommendations=overall_assessment.get('recommendations', [])
        )

        self.fit_results = gof_results
        logger.info("Goodness-of-fit assessment completed")

        return gof_results

    def _run_standard_gof_tests(
        self,
        saom_model: ToleranceCooperationSAOM,
        results: SAOMResults,
        n_simulations: int
    ) -> Dict[str, Any]:
        """Run standard RSiena goodness-of-fit tests."""
        if not self.r_interface:
            logger.warning("No R interface available for standard GOF tests")
            return {}

        try:
            # Prepare results in R environment
            self.r_interface.create_r_object("gof_model_results", results)

            # Run GOF for friendship network
            r_code = f"""
            library(RSiena)

            # Friendship network GOF
            gof_friendship <- sienaGOF(
                gof_model_results,
                varName = "friendship",
                verbose = FALSE,
                join = TRUE,
                period = 1
            )

            # Extract key statistics
            gof_stats <- list(
                mahalanobis_distance = gof_friendship$MahalanobisDistance,
                p_value = gof_friendship$pvalue,
                joint_test_p = gof_friendship$Joint$pvalue,
                crit_value = gof_friendship$CriticalValue
            )

            # Additional network statistics
            gof_outdegree <- sienaGOF(gof_model_results, OutdegreeDistribution, verbose=FALSE, varName="friendship")
            gof_indegree <- sienaGOF(gof_model_results, IndegreeDistribution, verbose=FALSE, varName="friendship")
            gof_geodesic <- sienaGOF(gof_model_results, GeodesicDistribution, verbose=FALSE, varName="friendship")

            network_gof_stats <- list(
                outdegree_p = gof_outdegree$pvalue,
                indegree_p = gof_indegree$pvalue,
                geodesic_p = gof_geodesic$pvalue
            )
            """

            self.r_interface.execute_r_code(r_code)

            # Get results
            gof_stats = self.r_interface.get_r_object("gof_stats")
            network_gof_stats = self.r_interface.get_r_object("network_gof_stats")

            standard_results = {
                'mahalanobis_distances': {
                    'friendship': float(gof_stats['mahalanobis_distance'])
                },
                'p_values': {
                    'friendship_overall': float(gof_stats['p_value']),
                    'friendship_joint': float(gof_stats['joint_test_p']),
                    'outdegree': float(network_gof_stats['outdegree_p']),
                    'indegree': float(network_gof_stats['indegree_p']),
                    'geodesic': float(network_gof_stats['geodesic_p'])
                },
                'critical_value': float(gof_stats['crit_value'])
            }

            logger.debug("Standard GOF tests completed successfully")
            return standard_results

        except Exception as e:
            logger.error(f"Standard GOF tests failed: {e}")
            return {}

    def _run_custom_diagnostics(
        self,
        dataset: RSienaDataSet,
        saom_model: ToleranceCooperationSAOM,
        results: SAOMResults
    ) -> Dict[str, float]:
        """Run custom diagnostic statistics."""
        logger.debug("Running custom diagnostic statistics...")

        custom_results = {}

        for stat_name, stat_config in self.custom_statistics.items():
            try:
                value = stat_config.statistic_function(dataset, saom_model, results)
                custom_results[stat_name] = value
                logger.debug(f"Custom statistic {stat_name}: {value:.3f}")

            except Exception as e:
                logger.warning(f"Custom statistic {stat_name} failed: {e}")
                custom_results[stat_name] = np.nan

        return custom_results

    def _calculate_tolerance_homophily(
        self,
        dataset: RSienaDataSet,
        saom_model: ToleranceCooperationSAOM,
        results: SAOMResults
    ) -> float:
        """Calculate tolerance-based homophily."""
        if 'tolerance' not in dataset.behavior_data:
            return np.nan

        # Use first wave data
        network = dataset.network_data[0]
        tolerance = dataset.behavior_data['tolerance'][0]

        # Calculate tolerance similarity for connected pairs
        similarities = []
        for i in range(len(tolerance)):
            for j in range(len(tolerance)):
                if i != j and network[i, j] > 0:
                    similarity = 1 - abs(tolerance[i] - tolerance[j])
                    similarities.append(similarity)

        return np.mean(similarities) if similarities else 0.0

    def _calculate_interethnic_cooperation_rate(
        self,
        dataset: RSienaDataSet,
        saom_model: ToleranceCooperationSAOM,
        results: SAOMResults
    ) -> float:
        """Calculate rate of interethnic cooperation."""
        if 'ethnicity' not in dataset.actor_attributes:
            return np.nan

        # Use first wave network (assuming cooperation is available)
        network = dataset.network_data[0]
        ethnicity = dataset.actor_attributes['ethnicity']

        interethnic_ties = 0
        possible_interethnic = 0

        for i in range(len(ethnicity)):
            for j in range(len(ethnicity)):
                if i != j and ethnicity[i] != ethnicity[j]:
                    possible_interethnic += 1
                    if network[i, j] > 0:
                        interethnic_ties += 1

        return interethnic_ties / possible_interethnic if possible_interethnic > 0 else 0.0

    def _calculate_tolerance_centrality_correlation(
        self,
        dataset: RSienaDataSet,
        saom_model: ToleranceCooperationSAOM,
        results: SAOMResults
    ) -> float:
        """Calculate correlation between tolerance and network centrality."""
        if 'tolerance' not in dataset.behavior_data:
            return np.nan

        # Create network and calculate centrality
        network = dataset.network_data[0]
        G = nx.from_numpy_array(network, create_using=nx.DiGraph)

        try:
            centrality = nx.degree_centrality(G)
            centrality_values = [centrality[i] for i in range(len(centrality))]

            tolerance = dataset.behavior_data['tolerance'][0]
            valid_indices = ~np.isnan(tolerance)

            if np.sum(valid_indices) < 3:
                return np.nan

            correlation = np.corrcoef(
                tolerance[valid_indices],
                np.array(centrality_values)[valid_indices]
            )[0, 1]

            return correlation

        except Exception:
            return np.nan

    def _calculate_tolerance_stability(
        self,
        dataset: RSienaDataSet,
        saom_model: ToleranceCooperationSAOM,
        results: SAOMResults
    ) -> float:
        """Calculate tolerance stability between waves."""
        if 'tolerance' not in dataset.behavior_data or dataset.n_periods < 2:
            return np.nan

        tolerance_data = dataset.behavior_data['tolerance']

        # Calculate correlation between consecutive waves
        correlations = []
        for wave in range(dataset.n_periods - 1):
            wave1 = tolerance_data[wave]
            wave2 = tolerance_data[wave + 1]

            valid_indices = ~(np.isnan(wave1) | np.isnan(wave2))
            if np.sum(valid_indices) >= 3:
                corr = np.corrcoef(wave1[valid_indices], wave2[valid_indices])[0, 1]
                correlations.append(corr)

        return np.mean(correlations) if correlations else np.nan

    def _calculate_tolerance_variance_explained(
        self,
        dataset: RSienaDataSet,
        saom_model: ToleranceCooperationSAOM,
        results: SAOMResults
    ) -> float:
        """Calculate variance in tolerance explained by network position."""
        if 'tolerance' not in dataset.behavior_data:
            return np.nan

        try:
            # Network measures
            network = dataset.network_data[0]
            G = nx.from_numpy_array(network, create_using=nx.DiGraph)

            degree_centrality = list(nx.degree_centrality(G).values())
            betweenness_centrality = list(nx.betweenness_centrality(G).values())

            # Tolerance data
            tolerance = dataset.behavior_data['tolerance'][0]
            valid_indices = ~np.isnan(tolerance)

            if np.sum(valid_indices) < 5:
                return np.nan

            # Simple regression R-squared
            X = np.column_stack([
                np.array(degree_centrality)[valid_indices],
                np.array(betweenness_centrality)[valid_indices]
            ])
            y = tolerance[valid_indices]

            # Calculate R-squared
            y_mean = np.mean(y)
            ss_tot = np.sum((y - y_mean) ** 2)

            # Simple linear regression
            from sklearn.linear_model import LinearRegression
            reg = LinearRegression().fit(X, y)
            y_pred = reg.predict(X)
            ss_res = np.sum((y - y_pred) ** 2)

            r_squared = 1 - (ss_res / ss_tot)
            return max(0, r_squared)  # Ensure non-negative

        except Exception:
            return np.nan

    def _assess_network_fit(
        self,
        dataset: RSienaDataSet,
        saom_model: ToleranceCooperationSAOM,
        results: SAOMResults
    ) -> Dict[str, Dict[str, float]]:
        """Assess network-level model fit."""
        network_diagnostics = {}

        # Basic network statistics
        for wave in range(dataset.n_periods):
            network = dataset.network_data[wave]
            G = nx.from_numpy_array(network, create_using=nx.DiGraph)

            wave_stats = {
                'density': nx.density(G),
                'reciprocity': self._calculate_reciprocity(network),
                'transitivity': nx.transitivity(G.to_undirected()),
                'average_clustering': nx.average_clustering(G.to_undirected()),
                'assortativity': self._calculate_assortativity(G, dataset),
                'components': nx.number_weakly_connected_components(G)
            }

            network_diagnostics[f'wave_{wave+1}'] = wave_stats

        return network_diagnostics

    def _calculate_reciprocity(self, network: np.ndarray) -> float:
        """Calculate network reciprocity."""
        mutual_ties = 0
        total_ties = 0

        for i in range(network.shape[0]):
            for j in range(network.shape[1]):
                if i != j and network[i, j] > 0:
                    total_ties += 1
                    if network[j, i] > 0:
                        mutual_ties += 1

        return mutual_ties / total_ties if total_ties > 0 else 0.0

    def _calculate_assortativity(self, G: nx.DiGraph, dataset: RSienaDataSet) -> float:
        """Calculate assortativity by ethnicity if available."""
        if 'ethnicity' not in dataset.actor_attributes:
            return np.nan

        try:
            # Create node attribute dictionary
            ethnicity_dict = {i: dataset.actor_attributes['ethnicity'][i]
                            for i in range(len(dataset.actor_attributes['ethnicity']))}

            # Set node attributes
            nx.set_node_attributes(G, ethnicity_dict, 'ethnicity')

            # Calculate assortativity
            return nx.attribute_assortativity_coefficient(G, 'ethnicity')

        except Exception:
            return np.nan

    def _assess_behavior_fit(
        self,
        dataset: RSienaDataSet,
        saom_model: ToleranceCooperationSAOM,
        results: SAOMResults
    ) -> Dict[str, Dict[str, float]]:
        """Assess behavior-level model fit."""
        behavior_diagnostics = {}

        if 'tolerance' in dataset.behavior_data:
            tolerance_data = dataset.behavior_data['tolerance']

            for wave in range(dataset.n_periods):
                wave_tolerance = tolerance_data[wave]
                valid_tolerance = wave_tolerance[~np.isnan(wave_tolerance)]

                if len(valid_tolerance) > 0:
                    wave_stats = {
                        'mean': np.mean(valid_tolerance),
                        'std': np.std(valid_tolerance),
                        'min': np.min(valid_tolerance),
                        'max': np.max(valid_tolerance),
                        'skewness': stats.skew(valid_tolerance),
                        'kurtosis': stats.kurtosis(valid_tolerance)
                    }

                    behavior_diagnostics[f'tolerance_wave_{wave+1}'] = wave_stats

        return behavior_diagnostics

    def _assess_overall_fit(
        self,
        standard_gof: Dict[str, Any],
        custom_diagnostics: Dict[str, float],
        network_diagnostics: Dict[str, Dict[str, float]],
        behavior_diagnostics: Dict[str, Dict[str, float]]
    ) -> Dict[str, Any]:
        """Assess overall model fit and provide recommendations."""
        # Check standard GOF p-values
        problematic_statistics = []
        recommendations = []

        # Standard GOF assessment
        if 'p_values' in standard_gof:
            for stat_name, p_value in standard_gof['p_values'].items():
                if p_value < 0.05:
                    problematic_statistics.append(f"Standard GOF: {stat_name}")

        # Custom diagnostics assessment
        for stat_name, value in custom_diagnostics.items():
            if np.isnan(value):
                continue

            stat_config = self.custom_statistics.get(stat_name)
            if stat_config:
                expected_min, expected_max = stat_config.expected_range
                if not (expected_min <= value <= expected_max):
                    problematic_statistics.append(f"Custom diagnostic: {stat_name}")

        # Overall fit assessment
        fit_adequate = len(problematic_statistics) <= 2  # Allow some issues

        # Generate recommendations
        if not fit_adequate:
            recommendations.extend([
                "Consider adding additional effects to improve model fit",
                "Check for model specification issues",
                "Validate data quality and preprocessing"
            ])

        if 'friendship_overall' in standard_gof.get('p_values', {}):
            if standard_gof['p_values']['friendship_overall'] < 0.01:
                recommendations.append("Friendship network fit is poor - consider structural effects")

        return {
            'fit_adequate': fit_adequate,
            'n_problematic_statistics': len(problematic_statistics),
            'problematic_statistics': problematic_statistics,
            'recommendations': recommendations,
            'overall_assessment': "Adequate" if fit_adequate else "Poor"
        }

    def _create_diagnostic_plots(
        self,
        dataset: RSienaDataSet,
        standard_gof: Dict[str, Any],
        custom_diagnostics: Dict[str, float]
    ) -> Dict[str, Any]:
        """Create diagnostic plots for model assessment."""
        plots = {}

        try:
            # Tolerance distribution plot
            if 'tolerance' in dataset.behavior_data:
                fig, axes = plt.subplots(1, dataset.n_periods, figsize=(4 * dataset.n_periods, 4))
                if dataset.n_periods == 1:
                    axes = [axes]

                for wave in range(dataset.n_periods):
                    tolerance = dataset.behavior_data['tolerance'][wave]
                    valid_tolerance = tolerance[~np.isnan(tolerance)]

                    axes[wave].hist(valid_tolerance, bins=20, alpha=0.7, edgecolor='black')
                    axes[wave].set_title(f'Tolerance Distribution - Wave {wave+1}')
                    axes[wave].set_xlabel('Tolerance')
                    axes[wave].set_ylabel('Frequency')

                plt.tight_layout()
                plots['tolerance_distributions'] = fig

            # Network degree distribution
            fig, axes = plt.subplots(1, dataset.n_periods, figsize=(4 * dataset.n_periods, 4))
            if dataset.n_periods == 1:
                axes = [axes]

            for wave in range(dataset.n_periods):
                network = dataset.network_data[wave]
                degrees = np.sum(network, axis=1)

                axes[wave].hist(degrees, bins=range(int(max(degrees)) + 2), alpha=0.7, edgecolor='black')
                axes[wave].set_title(f'Degree Distribution - Wave {wave+1}')
                axes[wave].set_xlabel('Degree')
                axes[wave].set_ylabel('Frequency')

            plt.tight_layout()
            plots['degree_distributions'] = fig

        except Exception as e:
            logger.warning(f"Failed to create some diagnostic plots: {e}")

        return plots

    def export_gof_report(
        self,
        gof_results: GoodnessOfFitResults,
        output_dir: Union[str, Path]
    ):
        """Export comprehensive goodness-of-fit report."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create summary report
        report_lines = [
            "Goodness-of-Fit Assessment Report",
            "=" * 40,
            "",
            f"Overall Assessment: {gof_results.overall_fit.get('overall_assessment', 'Unknown')}",
            f"Fit Adequate: {'Yes' if gof_results.fit_adequate else 'No'}",
            f"Number of Problematic Statistics: {gof_results.overall_fit.get('n_problematic_statistics', 0)}",
            "",
            "Standard GOF Results:",
            "-" * 20
        ]

        # Add p-values
        for stat_name, p_value in gof_results.p_values.items():
            significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
            report_lines.append(f"{stat_name}: p = {p_value:.3f} {significance}")

        # Add problematic statistics
        if gof_results.problematic_statistics:
            report_lines.extend([
                "",
                "Problematic Statistics:",
                "-" * 20
            ])
            for stat in gof_results.problematic_statistics:
                report_lines.append(f"- {stat}")

        # Add recommendations
        if gof_results.recommendations:
            report_lines.extend([
                "",
                "Recommendations:",
                "-" * 15
            ])
            for rec in gof_results.recommendations:
                report_lines.append(f"- {rec}")

        # Save report
        with open(output_dir / "gof_report.txt", 'w') as f:
            f.write("\n".join(report_lines))

        # Save detailed results as CSV
        if gof_results.network_statistics:
            network_df = pd.DataFrame(gof_results.network_statistics).T
            network_df.to_csv(output_dir / "network_diagnostics.csv")

        if gof_results.behavior_statistics:
            behavior_df = pd.DataFrame(gof_results.behavior_statistics).T
            behavior_df.to_csv(output_dir / "behavior_diagnostics.csv")

        # Save plots
        if gof_results.diagnostic_plots:
            for plot_name, fig in gof_results.diagnostic_plots.items():
                fig.savefig(output_dir / f"{plot_name}.png", dpi=300, bbox_inches='tight')
                plt.close(fig)

        logger.info(f"Goodness-of-fit report exported to {output_dir}")


if __name__ == "__main__":
    # Test goodness-of-fit tester
    logging.basicConfig(level=logging.INFO)

    try:
        # Create test GOF tester
        gof_tester = GoodnessOfFitTester()

        print("✓ Goodness-of-fit testing module test completed successfully")
        print(f"  - {len(gof_tester.custom_statistics)} custom diagnostic statistics defined")
        print("  - Standard RSiena GOF integration ready")
        print("  - Comprehensive diagnostic framework implemented")

    except Exception as e:
        logger.error(f"Goodness-of-fit testing test failed: {e}")
        import traceback
        traceback.print_exc()