"""
Global Sensitivity Analysis Framework for ABM-RSiena Integration

This module implements comprehensive global sensitivity analysis using Sobol indices,
Morris method, and other variance-based techniques for computational social science
research meeting PhD dissertation standards.

Author: Gamma Agent - Statistical Analysis & Validation Specialist
Date: 2025-09-15
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass, field
from pathlib import Path
import warnings
import pickle
import itertools
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

# Sensitivity analysis libraries
from SALib.sample import saltelli, morris as morris_sample, latin
from SALib.analyze import sobol, morris, delta, dgsm, fast, rbd_fast
from SALib.util import read_param_file
import sobol_seq

# Statistical libraries
from scipy import stats
from scipy.optimize import minimize
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Network analysis
import networkx as nx

logger = logging.getLogger(__name__)

@dataclass
class Parameter:
    """Container for model parameter specification."""
    name: str
    bounds: Tuple[float, float]  # (lower, upper)
    distribution: str = "uniform"  # Distribution type
    description: str = ""
    default_value: Optional[float] = None
    theoretical_justification: str = ""

@dataclass
class SobolIndices:
    """Container for Sobol sensitivity indices."""
    parameter_name: str
    first_order: float  # S1 - main effect
    first_order_conf: float  # S1 confidence interval
    total_order: float  # ST - total effect including interactions
    total_order_conf: float  # ST confidence interval
    second_order: Dict[str, float] = field(default_factory=dict)  # S2 - pairwise interactions

@dataclass
class MorrisResults:
    """Container for Morris method results."""
    parameter_name: str
    mu: float  # Mean of elementary effects
    mu_star: float  # Mean of absolute elementary effects
    sigma: float  # Standard deviation of elementary effects
    mu_star_conf: float  # Confidence interval for mu_star

@dataclass
class SensitivityResults:
    """Container for comprehensive sensitivity analysis results."""
    sobol_indices: Dict[str, SobolIndices] = field(default_factory=dict)
    morris_results: Dict[str, MorrisResults] = field(default_factory=dict)
    delta_indices: Dict[str, float] = field(default_factory=dict)
    parameter_rankings: Dict[str, int] = field(default_factory=dict)
    interaction_strength: float = 0.0
    model_output_variance: float = 0.0
    total_variance_explained: float = 0.0
    convergence_diagnostics: Dict[str, Any] = field(default_factory=dict)

class ParameterSpace:
    """
    Class for defining and managing parameter spaces for sensitivity analysis.
    """

    def __init__(self):
        self.parameters = {}
        self.bounds_matrix = None

    def add_parameter(self, parameter: Parameter):
        """Add a parameter to the space."""
        self.parameters[parameter.name] = parameter
        logger.debug(f"Added parameter: {parameter.name}")

    def get_salib_problem(self) -> Dict[str, Any]:
        """
        Convert parameter space to SALib problem format.

        Returns:
            SALib problem dictionary
        """
        problem = {
            'num_vars': len(self.parameters),
            'names': list(self.parameters.keys()),
            'bounds': [self.parameters[name].bounds for name in self.parameters.keys()]
        }
        return problem

    def create_abm_parameter_space(self) -> 'ParameterSpace':
        """
        Create standard ABM parameter space for social networks.

        Returns:
            ParameterSpace with ABM parameters
        """
        # Network formation parameters
        self.add_parameter(Parameter(
            name="density",
            bounds=(0.01, 0.5),
            description="Network density parameter",
            default_value=0.15,
            theoretical_justification="Social networks typically have low density"
        ))

        self.add_parameter(Parameter(
            name="homophily_strength",
            bounds=(0.0, 2.0),
            description="Strength of homophily in tie formation",
            default_value=0.8,
            theoretical_justification="Homophily is fundamental in social network formation"
        ))

        self.add_parameter(Parameter(
            name="clustering_tendency",
            bounds=(0.0, 1.0),
            description="Tendency to form triangles (transitivity)",
            default_value=0.6,
            theoretical_justification="Social networks exhibit triadic closure"
        ))

        # Temporal evolution parameters
        self.add_parameter(Parameter(
            name="formation_rate",
            bounds=(0.001, 0.5),
            description="Rate of new edge formation per time step",
            default_value=0.1,
            theoretical_justification="Formation rates vary across social contexts"
        ))

        self.add_parameter(Parameter(
            name="dissolution_rate",
            bounds=(0.001, 0.3),
            description="Rate of edge dissolution per time step",
            default_value=0.05,
            theoretical_justification="Ties dissolve slower than they form"
        ))

        # Agent heterogeneity parameters
        self.add_parameter(Parameter(
            name="agent_activity",
            bounds=(0.1, 2.0),
            description="Average agent activity level",
            default_value=1.0,
            theoretical_justification="Agents vary in their social activity"
        ))

        self.add_parameter(Parameter(
            name="attribute_variance",
            bounds=(0.1, 2.0),
            description="Variance in agent attributes",
            default_value=0.5,
            theoretical_justification="Population heterogeneity affects network structure"
        ))

        # RSiena-specific parameters
        self.add_parameter(Parameter(
            name="reciprocity_effect",
            bounds=(-2.0, 2.0),
            description="Tendency for reciprocal ties",
            default_value=1.5,
            theoretical_justification="Reciprocity is common in social networks"
        ))

        self.add_parameter(Parameter(
            name="popularity_effect",
            bounds=(-1.0, 1.0),
            description="Preferential attachment to popular nodes",
            default_value=0.2,
            theoretical_justification="Popularity effects vary across contexts"
        ))

        self.add_parameter(Parameter(
            name="activity_effect",
            bounds=(-1.0, 1.0),
            description="Effect of node activity on tie formation",
            default_value=0.1,
            theoretical_justification="Active nodes may form more ties"
        ))

        return self

    def create_rsiena_parameter_space(self) -> 'ParameterSpace':
        """
        Create RSiena-specific parameter space.

        Returns:
            ParameterSpace with RSiena parameters
        """
        # Core RSiena parameters
        self.add_parameter(Parameter(
            name="rate_parameter",
            bounds=(1.0, 20.0),
            description="Rate of change in network",
            default_value=5.0,
            theoretical_justification="Controls speed of network evolution"
        ))

        self.add_parameter(Parameter(
            name="outdegree_effect",
            bounds=(-5.0, -1.0),
            description="Baseline tendency to form ties",
            default_value=-2.5,
            theoretical_justification="Negative due to sparsity assumption"
        ))

        self.add_parameter(Parameter(
            name="reciprocity_effect",
            bounds=(-1.0, 3.0),
            description="Tendency for reciprocal ties",
            default_value=1.5,
            theoretical_justification="Strong reciprocity in social networks"
        ))

        self.add_parameter(Parameter(
            name="transitivity_effect",
            bounds=(-1.0, 2.0),
            description="Tendency to close triangles",
            default_value=0.8,
            theoretical_justification="Triadic closure mechanism"
        ))

        self.add_parameter(Parameter(
            name="indegree_popularity",
            bounds=(-0.5, 0.5),
            description="Preferential attachment effect",
            default_value=0.1,
            theoretical_justification="Popularity effects in social networks"
        ))

        self.add_parameter(Parameter(
            name="outdegree_activity",
            bounds=(-0.5, 0.5),
            description="Activity effect on tie formation",
            default_value=0.0,
            theoretical_justification="Variable across network types"
        ))

        # Attribute-based effects
        self.add_parameter(Parameter(
            name="similarity_effect",
            bounds=(-2.0, 2.0),
            description="Effect of attribute similarity",
            default_value=1.0,
            theoretical_justification="Homophily principle"
        ))

        self.add_parameter(Parameter(
            name="alter_effect",
            bounds=(-1.0, 1.0),
            description="Effect of alter attributes",
            default_value=0.2,
            theoretical_justification="Attribute attractiveness"
        ))

        self.add_parameter(Parameter(
            name="ego_effect",
            bounds=(-1.0, 1.0),
            description="Effect of ego attributes",
            default_value=0.1,
            theoretical_justification="Ego activity by attributes"
        ))

        return self

class SensitivityAnalyzer:
    """
    Main class for performing global sensitivity analysis on network models.
    """

    def __init__(self, parameter_space: ParameterSpace, n_cores: int = None):
        self.parameter_space = parameter_space
        self.n_cores = n_cores or mp.cpu_count() - 1
        self.problem = parameter_space.get_salib_problem()

    def sobol_analysis(self, model_function: Callable, n_samples: int = 1024,
                      calc_second_order: bool = True, parallel: bool = True) -> Dict[str, SobolIndices]:
        """
        Perform Sobol variance-based sensitivity analysis.

        Args:
            model_function: Function that takes parameter array and returns output
            n_samples: Number of samples for Sobol analysis
            calc_second_order: Whether to calculate second-order interactions
            parallel: Whether to use parallel processing

        Returns:
            Dictionary of SobolIndices for each parameter
        """
        logger.info(f"Starting Sobol analysis with {n_samples} samples")

        # Generate Sobol samples
        param_values = saltelli.sample(self.problem, n_samples, calc_second_order=calc_second_order)
        logger.info(f"Generated {param_values.shape[0]} parameter combinations")

        # Evaluate model
        if parallel and self.n_cores > 1:
            model_outputs = self._evaluate_model_parallel(model_function, param_values)
        else:
            model_outputs = self._evaluate_model_sequential(model_function, param_values)

        # Perform Sobol analysis
        sobol_results = sobol.analyze(self.problem, model_outputs, calc_second_order=calc_second_order)

        # Convert to SobolIndices objects
        indices = {}
        for i, param_name in enumerate(self.problem['names']):
            indices[param_name] = SobolIndices(
                parameter_name=param_name,
                first_order=float(sobol_results['S1'][i]),
                first_order_conf=float(sobol_results['S1_conf'][i]),
                total_order=float(sobol_results['ST'][i]),
                total_order_conf=float(sobol_results['ST_conf'][i])
            )

            # Add second-order interactions if calculated
            if calc_second_order and 'S2' in sobol_results:
                second_order = {}
                for j, other_param in enumerate(self.problem['names']):
                    if i != j:
                        s2_idx = self._get_s2_index(i, j, len(self.problem['names']))
                        if s2_idx < len(sobol_results['S2']):
                            second_order[other_param] = float(sobol_results['S2'][s2_idx])
                indices[param_name].second_order = second_order

        logger.info("Sobol analysis completed")
        return indices

    def morris_analysis(self, model_function: Callable, n_trajectories: int = 1000,
                       num_levels: int = 4, parallel: bool = True) -> Dict[str, MorrisResults]:
        """
        Perform Morris method sensitivity analysis.

        Args:
            model_function: Function that takes parameter array and returns output
            n_trajectories: Number of Morris trajectories
            num_levels: Number of grid levels
            parallel: Whether to use parallel processing

        Returns:
            Dictionary of MorrisResults for each parameter
        """
        logger.info(f"Starting Morris analysis with {n_trajectories} trajectories")

        # Generate Morris samples
        param_values = morris_sample.sample(self.problem, n_trajectories, num_levels=num_levels)
        logger.info(f"Generated {param_values.shape[0]} parameter combinations for Morris")

        # Evaluate model
        if parallel and self.n_cores > 1:
            model_outputs = self._evaluate_model_parallel(model_function, param_values)
        else:
            model_outputs = self._evaluate_model_sequential(model_function, param_values)

        # Perform Morris analysis
        morris_results = morris.analyze(self.problem, param_values, model_outputs,
                                      conf_level=0.95, print_to_console=False)

        # Convert to MorrisResults objects
        results = {}
        for i, param_name in enumerate(self.problem['names']):
            results[param_name] = MorrisResults(
                parameter_name=param_name,
                mu=float(morris_results['mu'][i]),
                mu_star=float(morris_results['mu_star'][i]),
                sigma=float(morris_results['sigma'][i]),
                mu_star_conf=float(morris_results['mu_star_conf'][i])
            )

        logger.info("Morris analysis completed")
        return results

    def delta_analysis(self, model_function: Callable, n_samples: int = 1000,
                      parallel: bool = True) -> Dict[str, float]:
        """
        Perform delta moment-independent sensitivity analysis.

        Args:
            model_function: Function that takes parameter array and returns output
            n_samples: Number of samples
            parallel: Whether to use parallel processing

        Returns:
            Dictionary of delta indices for each parameter
        """
        logger.info(f"Starting delta analysis with {n_samples} samples")

        # Generate Latin hypercube samples
        param_values = latin.sample(self.problem, n_samples)

        # Evaluate model
        if parallel and self.n_cores > 1:
            model_outputs = self._evaluate_model_parallel(model_function, param_values)
        else:
            model_outputs = self._evaluate_model_sequential(model_function, param_values)

        # Perform delta analysis
        delta_results = delta.analyze(self.problem, param_values, model_outputs,
                                    print_to_console=False)

        # Extract delta indices
        indices = {}
        for i, param_name in enumerate(self.problem['names']):
            indices[param_name] = float(delta_results['delta'][i])

        logger.info("Delta analysis completed")
        return indices

    def comprehensive_analysis(self, model_function: Callable, n_sobol: int = 1024,
                             n_morris: int = 1000, n_delta: int = 1000,
                             parallel: bool = True) -> SensitivityResults:
        """
        Perform comprehensive sensitivity analysis using multiple methods.

        Args:
            model_function: Function that takes parameter array and returns output
            n_sobol: Number of samples for Sobol analysis
            n_morris: Number of trajectories for Morris analysis
            n_delta: Number of samples for delta analysis
            parallel: Whether to use parallel processing

        Returns:
            SensitivityResults object with comprehensive results
        """
        logger.info("Starting comprehensive sensitivity analysis")

        results = SensitivityResults()

        # Sobol analysis
        try:
            logger.info("Performing Sobol analysis...")
            sobol_indices = self.sobol_analysis(model_function, n_sobol, parallel=parallel)
            results.sobol_indices = sobol_indices

            # Calculate interaction strength
            total_interactions = 0
            for param_name, indices in sobol_indices.items():
                interaction_strength = max(0, indices.total_order - indices.first_order)
                total_interactions += interaction_strength
            results.interaction_strength = total_interactions / len(sobol_indices)

            # Calculate total variance explained
            total_first_order = sum(idx.first_order for idx in sobol_indices.values())
            results.total_variance_explained = total_first_order

        except Exception as e:
            logger.error(f"Sobol analysis failed: {e}")

        # Morris analysis
        try:
            logger.info("Performing Morris analysis...")
            morris_results = self.morris_analysis(model_function, n_morris, parallel=parallel)
            results.morris_results = morris_results
        except Exception as e:
            logger.error(f"Morris analysis failed: {e}")

        # Delta analysis
        try:
            logger.info("Performing delta analysis...")
            delta_indices = self.delta_analysis(model_function, n_delta, parallel=parallel)
            results.delta_indices = delta_indices
        except Exception as e:
            logger.error(f"Delta analysis failed: {e}")

        # Generate parameter rankings
        results.parameter_rankings = self._generate_parameter_rankings(results)

        # Convergence diagnostics
        results.convergence_diagnostics = self._assess_convergence(results)

        logger.info("Comprehensive sensitivity analysis completed")
        return results

    def _evaluate_model_parallel(self, model_function: Callable, param_values: np.ndarray) -> np.ndarray:
        """Evaluate model function in parallel."""
        logger.info(f"Evaluating model in parallel using {self.n_cores} cores")

        # Split parameter values into chunks
        chunk_size = max(1, len(param_values) // self.n_cores)
        chunks = [param_values[i:i + chunk_size] for i in range(0, len(param_values), chunk_size)]

        results = []
        with ProcessPoolExecutor(max_workers=self.n_cores) as executor:
            # Submit jobs
            futures = {executor.submit(self._evaluate_chunk, model_function, chunk): i
                      for i, chunk in enumerate(chunks)}

            # Collect results
            for future in as_completed(futures):
                try:
                    chunk_results = future.result()
                    results.extend(chunk_results)
                except Exception as e:
                    logger.error(f"Chunk evaluation failed: {e}")
                    # Add NaN values for failed chunk
                    chunk_idx = futures[future]
                    chunk_size = len(chunks[chunk_idx])
                    results.extend([np.nan] * chunk_size)

        return np.array(results)

    def _evaluate_model_sequential(self, model_function: Callable, param_values: np.ndarray) -> np.ndarray:
        """Evaluate model function sequentially."""
        logger.info("Evaluating model sequentially")
        results = []

        for i, params in enumerate(param_values):
            try:
                output = model_function(params)
                results.append(output)

                # Progress logging
                if (i + 1) % 100 == 0:
                    logger.debug(f"Evaluated {i + 1}/{len(param_values)} parameter combinations")

            except Exception as e:
                logger.warning(f"Model evaluation failed for parameter set {i}: {e}")
                results.append(np.nan)

        return np.array(results)

    def _evaluate_chunk(self, model_function: Callable, param_chunk: np.ndarray) -> List[float]:
        """Evaluate model function for a chunk of parameters."""
        results = []
        for params in param_chunk:
            try:
                output = model_function(params)
                results.append(output)
            except Exception as e:
                logger.warning(f"Model evaluation failed: {e}")
                results.append(np.nan)
        return results

    def _get_s2_index(self, i: int, j: int, n_params: int) -> int:
        """Get index for second-order Sobol index S2[i,j]."""
        if i > j:
            i, j = j, i  # Ensure i < j
        return int(i * n_params - i * (i + 1) / 2 + j - i - 1)

    def _generate_parameter_rankings(self, results: SensitivityResults) -> Dict[str, int]:
        """Generate parameter importance rankings based on sensitivity indices."""
        rankings = {}

        # Rank by Sobol total-order indices (most comprehensive)
        if results.sobol_indices:
            sobol_ranking = sorted(results.sobol_indices.items(),
                                 key=lambda x: x[1].total_order, reverse=True)
            rankings['sobol_total'] = {param: rank + 1
                                     for rank, (param, _) in enumerate(sobol_ranking)}

        # Rank by Morris mu* (for screening)
        if results.morris_results:
            morris_ranking = sorted(results.morris_results.items(),
                                  key=lambda x: x[1].mu_star, reverse=True)
            rankings['morris_mu_star'] = {param: rank + 1
                                        for rank, (param, _) in enumerate(morris_ranking)}

        # Rank by delta indices
        if results.delta_indices:
            delta_ranking = sorted(results.delta_indices.items(),
                                 key=lambda x: x[1], reverse=True)
            rankings['delta'] = {param: rank + 1
                               for rank, (param, _) in enumerate(delta_ranking)}

        return rankings

    def _assess_convergence(self, results: SensitivityResults) -> Dict[str, Any]:
        """Assess convergence of sensitivity analysis."""
        diagnostics = {}

        # Check Sobol indices convergence
        if results.sobol_indices:
            # Calculate confidence interval widths
            ci_widths = []
            for param_name, indices in results.sobol_indices.items():
                ci_width = 2 * indices.first_order_conf / max(indices.first_order, 1e-6)
                ci_widths.append(ci_width)

            diagnostics['sobol_mean_ci_width'] = np.mean(ci_widths)
            diagnostics['sobol_max_ci_width'] = np.max(ci_widths)
            diagnostics['sobol_convergence_ok'] = np.max(ci_widths) < 0.1  # 10% relative error

        # Check Morris results stability
        if results.morris_results:
            # Look for parameters with high sigma (non-linear/interaction effects)
            high_sigma_params = []
            for param_name, morris_result in results.morris_results.items():
                if morris_result.sigma / max(morris_result.mu_star, 1e-6) > 0.5:
                    high_sigma_params.append(param_name)

            diagnostics['morris_high_sigma_params'] = high_sigma_params
            diagnostics['morris_nonlinearity_detected'] = len(high_sigma_params) > 0

        return diagnostics

    def plot_sensitivity_results(self, results: SensitivityResults,
                               output_dir: Path = None) -> Dict[str, Path]:
        """
        Generate comprehensive sensitivity analysis plots.

        Args:
            results: SensitivityResults object
            output_dir: Directory to save plots

        Returns:
            Dictionary mapping plot types to file paths
        """
        output_dir = output_dir or Path("outputs/sensitivity_plots")
        output_dir.mkdir(parents=True, exist_ok=True)

        plot_files = {}

        # Sobol indices plot
        if results.sobol_indices:
            fig, ax = plt.subplots(figsize=(12, 8))

            param_names = list(results.sobol_indices.keys())
            s1_values = [results.sobol_indices[name].first_order for name in param_names]
            st_values = [results.sobol_indices[name].total_order for name in param_names]
            s1_errors = [results.sobol_indices[name].first_order_conf for name in param_names]
            st_errors = [results.sobol_indices[name].total_order_conf for name in param_names]

            x = np.arange(len(param_names))
            width = 0.35

            ax.bar(x - width/2, s1_values, width, yerr=s1_errors,
                  label='First-order (S1)', alpha=0.8, capsize=5)
            ax.bar(x + width/2, st_values, width, yerr=st_errors,
                  label='Total-order (ST)', alpha=0.8, capsize=5)

            ax.set_xlabel('Parameters')
            ax.set_ylabel('Sensitivity Index')
            ax.set_title('Sobol Sensitivity Indices')
            ax.set_xticks(x)
            ax.set_xticklabels(param_names, rotation=45, ha='right')
            ax.legend()
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            sobol_file = output_dir / "sobol_indices.png"
            plt.savefig(sobol_file, dpi=300, bbox_inches='tight')
            plt.close()
            plot_files['sobol'] = sobol_file

        # Morris method plot
        if results.morris_results:
            fig, ax = plt.subplots(figsize=(10, 8))

            param_names = list(results.morris_results.keys())
            mu_star_values = [results.morris_results[name].mu_star for name in param_names]
            sigma_values = [results.morris_results[name].sigma for name in param_names]

            scatter = ax.scatter(mu_star_values, sigma_values, s=100, alpha=0.7)

            # Add parameter labels
            for i, name in enumerate(param_names):
                ax.annotate(name, (mu_star_values[i], sigma_values[i]),
                           xytext=(5, 5), textcoords='offset points', fontsize=8)

            ax.set_xlabel('μ* (Mean of |Elementary Effects|)')
            ax.set_ylabel('σ (Standard Deviation of Elementary Effects)')
            ax.set_title('Morris Method Sensitivity Analysis')
            ax.grid(True, alpha=0.3)

            # Add interpretation regions
            ax.axhline(y=np.mean(sigma_values), color='red', linestyle='--', alpha=0.5,
                      label='Mean σ')
            ax.axvline(x=np.mean(mu_star_values), color='red', linestyle='--', alpha=0.5,
                      label='Mean μ*')
            ax.legend()

            plt.tight_layout()
            morris_file = output_dir / "morris_analysis.png"
            plt.savefig(morris_file, dpi=300, bbox_inches='tight')
            plt.close()
            plot_files['morris'] = morris_file

        # Parameter ranking comparison
        if results.parameter_rankings:
            fig, ax = plt.subplots(figsize=(12, 8))

            # Create ranking comparison heatmap
            ranking_methods = list(results.parameter_rankings.keys())
            if ranking_methods:
                param_names = list(results.parameter_rankings[ranking_methods[0]].keys())
                ranking_matrix = np.zeros((len(param_names), len(ranking_methods)))

                for j, method in enumerate(ranking_methods):
                    for i, param in enumerate(param_names):
                        ranking_matrix[i, j] = results.parameter_rankings[method][param]

                im = ax.imshow(ranking_matrix, cmap='RdYlBu_r', aspect='auto')

                # Add colorbar
                cbar = plt.colorbar(im, ax=ax)
                cbar.set_label('Parameter Rank', rotation=270, labelpad=15)

                # Set labels
                ax.set_xticks(range(len(ranking_methods)))
                ax.set_xticklabels(ranking_methods, rotation=45, ha='right')
                ax.set_yticks(range(len(param_names)))
                ax.set_yticklabels(param_names)
                ax.set_title('Parameter Importance Rankings Across Methods')

                # Add text annotations
                for i in range(len(param_names)):
                    for j in range(len(ranking_methods)):
                        text = ax.text(j, i, int(ranking_matrix[i, j]),
                                     ha="center", va="center", color="black", fontweight='bold')

                plt.tight_layout()
                ranking_file = output_dir / "parameter_rankings.png"
                plt.savefig(ranking_file, dpi=300, bbox_inches='tight')
                plt.close()
                plot_files['rankings'] = ranking_file

        logger.info(f"Sensitivity analysis plots saved to {output_dir}")
        return plot_files

    def generate_sensitivity_report(self, results: SensitivityResults,
                                  output_file: Path = None) -> str:
        """
        Generate comprehensive sensitivity analysis report.

        Args:
            results: SensitivityResults object
            output_file: Optional file to save report

        Returns:
            Report text
        """
        report_lines = []
        report_lines.append("# Global Sensitivity Analysis Report")
        report_lines.append("=" * 50)
        report_lines.append("")

        # Executive summary
        report_lines.append("## Executive Summary")
        report_lines.append("")

        if results.sobol_indices:
            # Identify most important parameters
            sorted_params = sorted(results.sobol_indices.items(),
                                 key=lambda x: x[1].total_order, reverse=True)
            top_3 = sorted_params[:3]

            report_lines.append("### Most Influential Parameters (by total-order Sobol index):")
            for i, (param_name, indices) in enumerate(top_3, 1):
                report_lines.append(f"{i}. **{param_name}**: ST = {indices.total_order:.4f} "
                                  f"(±{indices.total_order_conf:.4f})")

            report_lines.append("")
            report_lines.append(f"**Total variance explained**: {results.total_variance_explained:.1%}")
            report_lines.append(f"**Interaction strength**: {results.interaction_strength:.4f}")
            report_lines.append("")

        # Detailed results
        report_lines.append("## Detailed Sensitivity Analysis Results")
        report_lines.append("")

        # Sobol indices
        if results.sobol_indices:
            report_lines.append("### Sobol Variance-Based Indices")
            report_lines.append("")
            report_lines.append("| Parameter | S1 (First-order) | ST (Total-order) | Interactions |")
            report_lines.append("|-----------|------------------|------------------|--------------|")

            for param_name, indices in sorted(results.sobol_indices.items(),
                                           key=lambda x: x[1].total_order, reverse=True):
                interaction = max(0, indices.total_order - indices.first_order)
                report_lines.append(f"| {param_name} | "
                                  f"{indices.first_order:.4f} (±{indices.first_order_conf:.4f}) | "
                                  f"{indices.total_order:.4f} (±{indices.total_order_conf:.4f}) | "
                                  f"{interaction:.4f} |")

            report_lines.append("")

        # Morris results
        if results.morris_results:
            report_lines.append("### Morris Method Results")
            report_lines.append("")
            report_lines.append("| Parameter | μ* | σ | Classification |")
            report_lines.append("|-----------|----|----|---------------|")

            for param_name, morris_result in sorted(results.morris_results.items(),
                                                   key=lambda x: x[1].mu_star, reverse=True):
                # Classify parameter importance
                if morris_result.mu_star > np.median([r.mu_star for r in results.morris_results.values()]):
                    if morris_result.sigma > np.median([r.sigma for r in results.morris_results.values()]):
                        classification = "High importance, non-linear"
                    else:
                        classification = "High importance, linear"
                else:
                    if morris_result.sigma > np.median([r.sigma for r in results.morris_results.values()]):
                        classification = "Low importance, non-linear"
                    else:
                        classification = "Low importance"

                report_lines.append(f"| {param_name} | "
                                  f"{morris_result.mu_star:.4f} | "
                                  f"{morris_result.sigma:.4f} | "
                                  f"{classification} |")

            report_lines.append("")

        # Delta indices
        if results.delta_indices:
            report_lines.append("### Delta Moment-Independent Indices")
            report_lines.append("")
            report_lines.append("| Parameter | δ Index | Interpretation |")
            report_lines.append("|-----------|---------|---------------|")

            for param_name, delta_value in sorted(results.delta_indices.items(),
                                                key=lambda x: x[1], reverse=True):
                if delta_value > 0.1:
                    interpretation = "Highly influential"
                elif delta_value > 0.05:
                    interpretation = "Moderately influential"
                else:
                    interpretation = "Low influence"

                report_lines.append(f"| {param_name} | {delta_value:.4f} | {interpretation} |")

            report_lines.append("")

        # Convergence diagnostics
        if results.convergence_diagnostics:
            report_lines.append("### Convergence Diagnostics")
            report_lines.append("")

            for key, value in results.convergence_diagnostics.items():
                if isinstance(value, bool):
                    status = "✓" if value else "✗"
                    report_lines.append(f"- {key}: {status}")
                elif isinstance(value, (int, float)):
                    report_lines.append(f"- {key}: {value:.6f}")
                else:
                    report_lines.append(f"- {key}: {value}")

            report_lines.append("")

        # Recommendations
        report_lines.append("## Recommendations")
        report_lines.append("")

        if results.sobol_indices:
            high_interaction_params = [name for name, indices in results.sobol_indices.items()
                                     if (indices.total_order - indices.first_order) > 0.1]

            if high_interaction_params:
                report_lines.append("- **Parameter interactions detected** for: " +
                                  ", ".join(high_interaction_params))
                report_lines.append("  Consider examining pairwise parameter relationships.")
                report_lines.append("")

            low_sensitivity_params = [name for name, indices in results.sobol_indices.items()
                                    if indices.total_order < 0.01]

            if low_sensitivity_params:
                report_lines.append("- **Low sensitivity parameters**: " +
                                  ", ".join(low_sensitivity_params))
                report_lines.append("  Consider fixing these at default values to reduce model complexity.")
                report_lines.append("")

        if results.convergence_diagnostics.get('sobol_convergence_ok') == False:
            report_lines.append("- **Convergence issues detected**: Consider increasing sample size for more reliable estimates.")
            report_lines.append("")

        report_text = "\n".join(report_lines)

        # Save report if requested
        if output_file:
            output_file.write_text(report_text)
            logger.info(f"Sensitivity analysis report saved to {output_file}")

        return report_text


# Example usage and testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Create parameter space
    param_space = ParameterSpace()
    param_space.create_abm_parameter_space()

    # Define a simple test model function
    def test_network_model(params):
        """Simple test model that returns a network metric."""
        # Extract parameters
        density = params[0]
        homophily = params[1]
        clustering = params[2]
        formation_rate = params[3]

        # Simple synthetic network metric
        # In practice, this would run an actual ABM or RSiena model
        metric = (density * 0.5 + homophily * 0.3 + clustering * 0.2 +
                 formation_rate * 0.1 + np.random.normal(0, 0.05))
        return max(0, metric)  # Ensure positive output

    # Initialize sensitivity analyzer
    analyzer = SensitivityAnalyzer(param_space)

    # Perform comprehensive analysis (using small sample sizes for testing)
    results = analyzer.comprehensive_analysis(test_network_model,
                                            n_sobol=256,  # Reduced for testing
                                            n_morris=100,  # Reduced for testing
                                            n_delta=200)   # Reduced for testing

    # Generate report
    report = analyzer.generate_sensitivity_report(results)
    print(report)

    # Generate plots
    plot_files = analyzer.plot_sensitivity_results(results)
    print(f"\nPlots generated: {list(plot_files.keys())}")