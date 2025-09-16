"""
Bootstrap Analysis Module for Uncertainty Quantification

This module implements comprehensive bootstrap methods for uncertainty quantification,
confidence interval estimation, and robust statistical inference for ABM-RSiena
integration studies meeting PhD dissertation standards.

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
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import multiprocessing as mp
from functools import partial

# Statistical libraries
from scipy import stats
from scipy.optimize import minimize
from sklearn.utils import resample
import matplotlib.pyplot as plt
import seaborn as sns

# Network analysis
import networkx as nx

logger = logging.getLogger(__name__)

@dataclass
class BootstrapResult:
    """Container for bootstrap analysis results."""
    statistic_name: str
    original_statistic: float
    bootstrap_samples: np.ndarray
    confidence_intervals: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    bias: float = 0.0
    bias_corrected_statistic: float = 0.0
    standard_error: float = 0.0
    distribution_type: str = "empirical"
    n_bootstrap: int = 0

@dataclass
class BootstrapComparison:
    """Container for bootstrap comparison between two groups/methods."""
    group1_name: str
    group2_name: str
    difference_samples: np.ndarray
    original_difference: float
    p_value_bootstrap: float
    effect_size_samples: np.ndarray
    confidence_intervals: Dict[str, Tuple[float, float]] = field(default_factory=dict)

class NetworkBootstrapper:
    """
    Specialized bootstrap methods for network data.
    Handles the unique challenges of resampling network structures.
    """

    def __init__(self, network: nx.Graph):
        self.network = network
        self.nodes = list(network.nodes())
        self.edges = list(network.edges())

    def node_bootstrap(self, n_bootstrap: int = 1000) -> List[nx.Graph]:
        """
        Bootstrap by resampling nodes (with replacement).
        Maintains original network structure where possible.

        Args:
            n_bootstrap: Number of bootstrap samples

        Returns:
            List of bootstrap network samples
        """
        bootstrap_networks = []

        for _ in range(n_bootstrap):
            # Sample nodes with replacement
            bootstrap_nodes = np.random.choice(self.nodes, size=len(self.nodes), replace=True)

            # Create mapping for resampled nodes
            node_mapping = {old: new for new, old in enumerate(bootstrap_nodes)}

            # Create new network
            bootstrap_net = nx.Graph()
            bootstrap_net.add_nodes_from(range(len(bootstrap_nodes)))

            # Add edges that exist in original network between resampled nodes
            for i, node1 in enumerate(bootstrap_nodes):
                for j, node2 in enumerate(bootstrap_nodes[i+1:], i+1):
                    if self.network.has_edge(node1, node2):
                        bootstrap_net.add_edge(i, j)

            bootstrap_networks.append(bootstrap_net)

        return bootstrap_networks

    def edge_bootstrap(self, n_bootstrap: int = 1000) -> List[nx.Graph]:
        """
        Bootstrap by resampling edges (with replacement).

        Args:
            n_bootstrap: Number of bootstrap samples

        Returns:
            List of bootstrap network samples
        """
        bootstrap_networks = []

        for _ in range(n_bootstrap):
            # Sample edges with replacement
            bootstrap_edges = resample(self.edges, n_samples=len(self.edges), replace=True)

            # Create new network
            bootstrap_net = nx.Graph()
            bootstrap_net.add_nodes_from(self.nodes)
            bootstrap_net.add_edges_from(bootstrap_edges)

            bootstrap_networks.append(bootstrap_net)

        return bootstrap_networks

    def subgraph_bootstrap(self, subsample_ratio: float = 0.8,
                          n_bootstrap: int = 1000) -> List[nx.Graph]:
        """
        Bootstrap by sampling subgraphs.

        Args:
            subsample_ratio: Fraction of nodes to include in each bootstrap sample
            n_bootstrap: Number of bootstrap samples

        Returns:
            List of bootstrap network samples
        """
        bootstrap_networks = []
        n_subsample = max(1, int(subsample_ratio * len(self.nodes)))

        for _ in range(n_bootstrap):
            # Sample subset of nodes
            subsample_nodes = np.random.choice(self.nodes, size=n_subsample, replace=False)

            # Extract subgraph
            bootstrap_net = self.network.subgraph(subsample_nodes).copy()

            bootstrap_networks.append(bootstrap_net)

        return bootstrap_networks

    def block_bootstrap(self, block_attribute: str, n_bootstrap: int = 1000) -> List[nx.Graph]:
        """
        Block bootstrap based on node attributes (e.g., communities).

        Args:
            block_attribute: Node attribute to use for blocking
            n_bootstrap: Number of bootstrap samples

        Returns:
            List of bootstrap network samples
        """
        # Group nodes by attribute
        blocks = {}
        for node in self.nodes:
            attr_value = self.network.nodes[node].get(block_attribute, 'default')
            if attr_value not in blocks:
                blocks[attr_value] = []
            blocks[attr_value].append(node)

        bootstrap_networks = []

        for _ in range(n_bootstrap):
            # Resample within each block
            bootstrap_nodes = []
            node_mapping = {}
            new_id = 0

            for block_value, block_nodes in blocks.items():
                # Sample with replacement within block
                resampled_block = np.random.choice(block_nodes,
                                                 size=len(block_nodes),
                                                 replace=True)

                for old_node in resampled_block:
                    node_mapping[old_node] = node_mapping.get(old_node, [])
                    node_mapping[old_node].append(new_id)
                    bootstrap_nodes.append(new_id)
                    new_id += 1

            # Create bootstrap network
            bootstrap_net = nx.Graph()
            bootstrap_net.add_nodes_from(bootstrap_nodes)

            # Add edges between resampled nodes
            for (u, v) in self.edges:
                if u in node_mapping and v in node_mapping:
                    for new_u in node_mapping[u]:
                        for new_v in node_mapping[v]:
                            if new_u != new_v:  # Avoid self-loops
                                bootstrap_net.add_edge(new_u, new_v)

            bootstrap_networks.append(bootstrap_net)

        return bootstrap_networks

class BootstrapAnalyzer:
    """
    Comprehensive bootstrap analysis for statistical inference and uncertainty quantification.
    """

    def __init__(self, n_bootstrap: int = 1000, confidence_levels: List[float] = None,
                 n_jobs: int = None, random_state: int = None):
        self.n_bootstrap = n_bootstrap
        self.confidence_levels = confidence_levels or [0.90, 0.95, 0.99]
        self.n_jobs = n_jobs or mp.cpu_count() - 1
        self.random_state = random_state

        if random_state is not None:
            np.random.seed(random_state)

    def basic_bootstrap(self, data: np.ndarray, statistic_func: Callable,
                       method: str = "percentile") -> BootstrapResult:
        """
        Perform basic bootstrap analysis for a given statistic.

        Args:
            data: Original data array
            statistic_func: Function that computes the statistic from data
            method: CI method ('percentile', 'bias_corrected', 'bca')

        Returns:
            BootstrapResult object
        """
        logger.info(f"Performing basic bootstrap with {self.n_bootstrap} samples")

        # Calculate original statistic
        original_statistic = statistic_func(data)

        # Generate bootstrap samples
        bootstrap_samples = []
        for _ in range(self.n_bootstrap):
            bootstrap_data = resample(data, n_samples=len(data), replace=True,
                                    random_state=None)  # Use global random state
            try:
                bootstrap_stat = statistic_func(bootstrap_data)
                bootstrap_samples.append(bootstrap_stat)
            except Exception as e:
                logger.warning(f"Bootstrap sample failed: {e}")
                continue

        bootstrap_samples = np.array(bootstrap_samples)

        # Calculate bias
        bias = np.mean(bootstrap_samples) - original_statistic
        bias_corrected_statistic = original_statistic - bias

        # Calculate standard error
        standard_error = np.std(bootstrap_samples, ddof=1)

        # Calculate confidence intervals
        confidence_intervals = {}

        if method == "percentile":
            confidence_intervals = self._percentile_ci(bootstrap_samples)
        elif method == "bias_corrected":
            confidence_intervals = self._bias_corrected_ci(data, statistic_func,
                                                         bootstrap_samples, original_statistic)
        elif method == "bca":
            confidence_intervals = self._bca_ci(data, statistic_func,
                                              bootstrap_samples, original_statistic)

        return BootstrapResult(
            statistic_name="custom_statistic",
            original_statistic=original_statistic,
            bootstrap_samples=bootstrap_samples,
            confidence_intervals=confidence_intervals,
            bias=bias,
            bias_corrected_statistic=bias_corrected_statistic,
            standard_error=standard_error,
            n_bootstrap=len(bootstrap_samples)
        )

    def paired_bootstrap_test(self, group1: np.ndarray, group2: np.ndarray,
                             statistic_func: Callable = np.mean) -> BootstrapComparison:
        """
        Perform bootstrap test for paired/dependent samples.

        Args:
            group1: First group data
            group2: Second group data (same length as group1)
            statistic_func: Function to compute statistic

        Returns:
            BootstrapComparison object
        """
        if len(group1) != len(group2):
            raise ValueError("Paired samples must have equal length")

        logger.info(f"Performing paired bootstrap test with {self.n_bootstrap} samples")

        # Calculate original statistics
        original_stat1 = statistic_func(group1)
        original_stat2 = statistic_func(group2)
        original_difference = original_stat1 - original_stat2

        # Generate bootstrap samples of differences
        differences = group1 - group2
        bootstrap_differences = []

        for _ in range(self.n_bootstrap):
            bootstrap_diff = resample(differences, n_samples=len(differences), replace=True)
            bootstrap_differences.append(statistic_func(bootstrap_diff))

        bootstrap_differences = np.array(bootstrap_differences)

        # Calculate p-value (two-tailed test for no difference)
        p_value = np.mean(np.abs(bootstrap_differences) >= np.abs(original_difference))

        # Calculate effect size (Cohen's d for means)
        if statistic_func == np.mean:
            pooled_std = np.sqrt((np.var(group1, ddof=1) + np.var(group2, ddof=1)) / 2)
            effect_size_samples = bootstrap_differences / pooled_std
        else:
            effect_size_samples = bootstrap_differences  # Raw differences

        # Calculate confidence intervals for difference
        confidence_intervals = self._percentile_ci(bootstrap_differences)

        return BootstrapComparison(
            group1_name="group1",
            group2_name="group2",
            difference_samples=bootstrap_differences,
            original_difference=original_difference,
            p_value_bootstrap=p_value,
            effect_size_samples=effect_size_samples,
            confidence_intervals=confidence_intervals
        )

    def independent_bootstrap_test(self, group1: np.ndarray, group2: np.ndarray,
                                  statistic_func: Callable = np.mean) -> BootstrapComparison:
        """
        Perform bootstrap test for independent samples.

        Args:
            group1: First group data
            group2: Second group data
            statistic_func: Function to compute statistic

        Returns:
            BootstrapComparison object
        """
        logger.info(f"Performing independent bootstrap test with {self.n_bootstrap} samples")

        # Calculate original statistics
        original_stat1 = statistic_func(group1)
        original_stat2 = statistic_func(group2)
        original_difference = original_stat1 - original_stat2

        # Generate bootstrap samples
        bootstrap_stat1 = []
        bootstrap_stat2 = []

        for _ in range(self.n_bootstrap):
            boot_group1 = resample(group1, n_samples=len(group1), replace=True)
            boot_group2 = resample(group2, n_samples=len(group2), replace=True)

            bootstrap_stat1.append(statistic_func(boot_group1))
            bootstrap_stat2.append(statistic_func(boot_group2))

        bootstrap_differences = np.array(bootstrap_stat1) - np.array(bootstrap_stat2)

        # Calculate p-value using permutation test logic
        # Combine groups and permute for null distribution
        combined_data = np.concatenate([group1, group2])
        n1 = len(group1)

        null_differences = []
        for _ in range(self.n_bootstrap):
            permuted_data = np.random.permutation(combined_data)
            perm_group1 = permuted_data[:n1]
            perm_group2 = permuted_data[n1:]

            null_diff = statistic_func(perm_group1) - statistic_func(perm_group2)
            null_differences.append(null_diff)

        null_differences = np.array(null_differences)
        p_value = np.mean(np.abs(null_differences) >= np.abs(original_difference))

        # Calculate effect size
        if statistic_func == np.mean:
            pooled_std = np.sqrt((np.var(group1, ddof=1) * (len(group1) - 1) +
                                np.var(group2, ddof=1) * (len(group2) - 1)) /
                               (len(group1) + len(group2) - 2))
            effect_size_samples = bootstrap_differences / pooled_std
        else:
            effect_size_samples = bootstrap_differences

        # Calculate confidence intervals
        confidence_intervals = self._percentile_ci(bootstrap_differences)

        return BootstrapComparison(
            group1_name="group1",
            group2_name="group2",
            difference_samples=bootstrap_differences,
            original_difference=original_difference,
            p_value_bootstrap=p_value,
            effect_size_samples=effect_size_samples,
            confidence_intervals=confidence_intervals
        )

    def network_bootstrap_analysis(self, networks: List[nx.Graph],
                                  metric_func: Callable,
                                  bootstrap_method: str = "node") -> BootstrapResult:
        """
        Perform bootstrap analysis on network metrics.

        Args:
            networks: List of network objects
            metric_func: Function that computes network metric
            bootstrap_method: Method for network bootstrap ('node', 'edge', 'subgraph')

        Returns:
            BootstrapResult object
        """
        logger.info(f"Performing network bootstrap analysis using {bootstrap_method} method")

        if not networks:
            raise ValueError("No networks provided")

        # Calculate original metric
        original_metrics = [metric_func(net) for net in networks]
        original_statistic = np.mean(original_metrics)

        # Generate bootstrap samples
        bootstrap_statistics = []

        for _ in range(self.n_bootstrap):
            # Sample networks with replacement
            bootstrap_networks = resample(networks, n_samples=len(networks), replace=True)

            # For each sampled network, apply network-specific bootstrap
            bootstrap_metrics = []
            for net in bootstrap_networks:
                bootstrapper = NetworkBootstrapper(net)

                if bootstrap_method == "node":
                    boot_nets = bootstrapper.node_bootstrap(n_bootstrap=1)
                elif bootstrap_method == "edge":
                    boot_nets = bootstrapper.edge_bootstrap(n_bootstrap=1)
                elif bootstrap_method == "subgraph":
                    boot_nets = bootstrapper.subgraph_bootstrap(n_bootstrap=1)
                else:
                    raise ValueError(f"Unknown bootstrap method: {bootstrap_method}")

                # Calculate metric for bootstrap network
                try:
                    boot_metric = metric_func(boot_nets[0])
                    bootstrap_metrics.append(boot_metric)
                except Exception as e:
                    logger.warning(f"Network metric calculation failed: {e}")
                    continue

            if bootstrap_metrics:
                bootstrap_statistics.append(np.mean(bootstrap_metrics))

        bootstrap_statistics = np.array(bootstrap_statistics)

        # Calculate results
        bias = np.mean(bootstrap_statistics) - original_statistic
        standard_error = np.std(bootstrap_statistics, ddof=1)
        confidence_intervals = self._percentile_ci(bootstrap_statistics)

        return BootstrapResult(
            statistic_name=f"network_{metric_func.__name__}",
            original_statistic=original_statistic,
            bootstrap_samples=bootstrap_statistics,
            confidence_intervals=confidence_intervals,
            bias=bias,
            bias_corrected_statistic=original_statistic - bias,
            standard_error=standard_error,
            n_bootstrap=len(bootstrap_statistics)
        )

    def temporal_bootstrap(self, time_series: np.ndarray, block_size: int = None,
                          statistic_func: Callable = np.mean) -> BootstrapResult:
        """
        Perform block bootstrap for time series data.

        Args:
            time_series: Time series data
            block_size: Size of blocks for block bootstrap
            statistic_func: Function to compute statistic

        Returns:
            BootstrapResult object
        """
        if block_size is None:
            # Rule of thumb: block size ~ n^(1/3)
            block_size = max(1, int(len(time_series) ** (1/3)))

        logger.info(f"Performing temporal bootstrap with block size {block_size}")

        original_statistic = statistic_func(time_series)

        # Generate bootstrap samples using moving block bootstrap
        bootstrap_samples = []
        n_blocks = len(time_series) - block_size + 1

        for _ in range(self.n_bootstrap):
            # Sample starting positions for blocks
            n_blocks_needed = int(np.ceil(len(time_series) / block_size))
            block_starts = np.random.choice(n_blocks, size=n_blocks_needed, replace=True)

            # Construct bootstrap series
            bootstrap_series = []
            for start in block_starts:
                end = min(start + block_size, len(time_series))
                bootstrap_series.extend(time_series[start:end])

            # Trim to original length
            bootstrap_series = bootstrap_series[:len(time_series)]

            try:
                bootstrap_stat = statistic_func(np.array(bootstrap_series))
                bootstrap_samples.append(bootstrap_stat)
            except Exception as e:
                logger.warning(f"Temporal bootstrap sample failed: {e}")
                continue

        bootstrap_samples = np.array(bootstrap_samples)

        # Calculate results
        bias = np.mean(bootstrap_samples) - original_statistic
        standard_error = np.std(bootstrap_samples, ddof=1)
        confidence_intervals = self._percentile_ci(bootstrap_samples)

        return BootstrapResult(
            statistic_name="temporal_statistic",
            original_statistic=original_statistic,
            bootstrap_samples=bootstrap_samples,
            confidence_intervals=confidence_intervals,
            bias=bias,
            bias_corrected_statistic=original_statistic - bias,
            standard_error=standard_error,
            n_bootstrap=len(bootstrap_samples)
        )

    def _percentile_ci(self, bootstrap_samples: np.ndarray) -> Dict[str, Tuple[float, float]]:
        """Calculate percentile confidence intervals."""
        confidence_intervals = {}

        for conf_level in self.confidence_levels:
            alpha = 1 - conf_level
            lower_percentile = (alpha / 2) * 100
            upper_percentile = (1 - alpha / 2) * 100

            lower_bound = np.percentile(bootstrap_samples, lower_percentile)
            upper_bound = np.percentile(bootstrap_samples, upper_percentile)

            confidence_intervals[f"{conf_level:.0%}"] = (lower_bound, upper_bound)

        return confidence_intervals

    def _bias_corrected_ci(self, original_data: np.ndarray, statistic_func: Callable,
                          bootstrap_samples: np.ndarray, original_statistic: float) -> Dict[str, Tuple[float, float]]:
        """Calculate bias-corrected confidence intervals."""
        confidence_intervals = {}

        # Calculate bias-correction
        n_less = np.sum(bootstrap_samples < original_statistic)
        bias_correction = stats.norm.ppf(n_less / len(bootstrap_samples))

        for conf_level in self.confidence_levels:
            alpha = 1 - conf_level
            z_alpha_2 = stats.norm.ppf(alpha / 2)
            z_1_alpha_2 = stats.norm.ppf(1 - alpha / 2)

            # Adjusted percentiles
            lower_p = stats.norm.cdf(2 * bias_correction + z_alpha_2)
            upper_p = stats.norm.cdf(2 * bias_correction + z_1_alpha_2)

            # Ensure percentiles are in valid range
            lower_p = max(0, min(1, lower_p))
            upper_p = max(0, min(1, upper_p))

            lower_bound = np.percentile(bootstrap_samples, lower_p * 100)
            upper_bound = np.percentile(bootstrap_samples, upper_p * 100)

            confidence_intervals[f"{conf_level:.0%}_BC"] = (lower_bound, upper_bound)

        return confidence_intervals

    def _bca_ci(self, original_data: np.ndarray, statistic_func: Callable,
               bootstrap_samples: np.ndarray, original_statistic: float) -> Dict[str, Tuple[float, float]]:
        """Calculate bias-corrected and accelerated (BCa) confidence intervals."""
        # This is a simplified implementation
        # Full BCa requires jackknife estimation of acceleration parameter

        # For now, return bias-corrected intervals
        # In a full implementation, would calculate acceleration parameter
        return self._bias_corrected_ci(original_data, statistic_func,
                                     bootstrap_samples, original_statistic)

    def bootstrap_hypothesis_test(self, data1: np.ndarray, data2: np.ndarray,
                                 null_difference: float = 0.0,
                                 alternative: str = "two-sided",
                                 statistic_func: Callable = np.mean) -> Dict[str, Any]:
        """
        Perform bootstrap-based hypothesis test.

        Args:
            data1: First sample
            data2: Second sample
            null_difference: Hypothesized difference under null
            alternative: Alternative hypothesis ('two-sided', 'greater', 'less')
            statistic_func: Function to compute test statistic

        Returns:
            Dictionary with test results
        """
        logger.info("Performing bootstrap hypothesis test")

        # Observed test statistic
        observed_stat = statistic_func(data1) - statistic_func(data2)

        # Generate null distribution by centering data
        mean1, mean2 = np.mean(data1), np.mean(data2)
        grand_mean = (len(data1) * mean1 + len(data2) * mean2) / (len(data1) + len(data2))

        # Center data to create null hypothesis scenario
        centered_data1 = data1 - mean1 + grand_mean - null_difference/2
        centered_data2 = data2 - mean2 + grand_mean + null_difference/2

        # Bootstrap null distribution
        null_statistics = []
        for _ in range(self.n_bootstrap):
            boot_data1 = resample(centered_data1, n_samples=len(data1), replace=True)
            boot_data2 = resample(centered_data2, n_samples=len(data2), replace=True)

            null_stat = statistic_func(boot_data1) - statistic_func(boot_data2)
            null_statistics.append(null_stat)

        null_statistics = np.array(null_statistics)

        # Calculate p-value based on alternative hypothesis
        if alternative == "two-sided":
            p_value = np.mean(np.abs(null_statistics) >= np.abs(observed_stat))
        elif alternative == "greater":
            p_value = np.mean(null_statistics >= observed_stat)
        elif alternative == "less":
            p_value = np.mean(null_statistics <= observed_stat)
        else:
            raise ValueError(f"Unknown alternative hypothesis: {alternative}")

        return {
            "observed_statistic": observed_stat,
            "null_statistics": null_statistics,
            "p_value": p_value,
            "alternative": alternative,
            "n_bootstrap": self.n_bootstrap
        }

    def generate_bootstrap_plots(self, result: BootstrapResult,
                               output_dir: Path = None) -> Dict[str, Path]:
        """
        Generate comprehensive bootstrap analysis plots.

        Args:
            result: BootstrapResult object
            output_dir: Directory to save plots

        Returns:
            Dictionary mapping plot types to file paths
        """
        output_dir = output_dir or Path("outputs/bootstrap_plots")
        output_dir.mkdir(parents=True, exist_ok=True)

        plot_files = {}

        # Bootstrap distribution histogram
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Histogram of bootstrap samples
        ax1.hist(result.bootstrap_samples, bins=50, alpha=0.7, density=True,
                color='skyblue', edgecolor='black')
        ax1.axvline(result.original_statistic, color='red', linestyle='--',
                   linewidth=2, label=f'Original: {result.original_statistic:.4f}')
        ax1.axvline(result.bias_corrected_statistic, color='orange', linestyle='--',
                   linewidth=2, label=f'Bias-corrected: {result.bias_corrected_statistic:.4f}')

        # Add confidence intervals
        for conf_level, (lower, upper) in result.confidence_intervals.items():
            if not conf_level.endswith('_BC'):  # Only plot percentile CIs
                ax1.axvspan(lower, upper, alpha=0.2,
                           label=f'{conf_level} CI')

        ax1.set_xlabel('Bootstrap Statistic Values')
        ax1.set_ylabel('Density')
        ax1.set_title(f'Bootstrap Distribution ({result.statistic_name})')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Q-Q plot for normality check
        stats.probplot(result.bootstrap_samples, dist="norm", plot=ax2)
        ax2.set_title('Q-Q Plot: Bootstrap Samples vs Normal Distribution')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        bootstrap_file = output_dir / f"bootstrap_{result.statistic_name}.png"
        plt.savefig(bootstrap_file, dpi=300, bbox_inches='tight')
        plt.close()
        plot_files['distribution'] = bootstrap_file

        # Convergence plot
        if len(result.bootstrap_samples) > 100:
            fig, ax = plt.subplots(figsize=(10, 6))

            # Calculate running mean and CI
            running_means = np.cumsum(result.bootstrap_samples) / np.arange(1, len(result.bootstrap_samples) + 1)

            ax.plot(running_means, color='blue', linewidth=2, label='Running Mean')
            ax.axhline(result.original_statistic, color='red', linestyle='--',
                      label=f'True Value: {result.original_statistic:.4f}')

            # Add confidence bands
            n_samples = np.arange(1, len(result.bootstrap_samples) + 1)
            running_stds = np.array([np.std(result.bootstrap_samples[:i+1]) for i in range(len(result.bootstrap_samples))])
            standard_errors = running_stds / np.sqrt(n_samples)

            ax.fill_between(n_samples, running_means - 1.96 * standard_errors,
                           running_means + 1.96 * standard_errors,
                           alpha=0.2, label='95% Confidence Band')

            ax.set_xlabel('Bootstrap Sample Number')
            ax.set_ylabel('Running Mean')
            ax.set_title('Bootstrap Convergence')
            ax.legend()
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            convergence_file = output_dir / f"convergence_{result.statistic_name}.png"
            plt.savefig(convergence_file, dpi=300, bbox_inches='tight')
            plt.close()
            plot_files['convergence'] = convergence_file

        logger.info(f"Bootstrap plots saved to {output_dir}")
        return plot_files

    def generate_bootstrap_report(self, results: List[BootstrapResult],
                                output_file: Path = None) -> str:
        """
        Generate comprehensive bootstrap analysis report.

        Args:
            results: List of BootstrapResult objects
            output_file: Optional file to save report

        Returns:
            Report text
        """
        report_lines = []
        report_lines.append("# Bootstrap Analysis Report")
        report_lines.append("=" * 50)
        report_lines.append("")

        # Summary table
        report_lines.append("## Summary of Bootstrap Results")
        report_lines.append("")
        report_lines.append("| Statistic | Original | Bias-Corrected | Std Error | 95% CI |")
        report_lines.append("|-----------|----------|----------------|-----------|---------|")

        for result in results:
            ci_95 = result.confidence_intervals.get('95%', (np.nan, np.nan))
            report_lines.append(f"| {result.statistic_name} | "
                              f"{result.original_statistic:.4f} | "
                              f"{result.bias_corrected_statistic:.4f} | "
                              f"{result.standard_error:.4f} | "
                              f"[{ci_95[0]:.4f}, {ci_95[1]:.4f}] |")

        report_lines.append("")

        # Detailed results for each statistic
        for result in results:
            report_lines.append(f"## {result.statistic_name}")
            report_lines.append("")
            report_lines.append(f"- **Original Statistic**: {result.original_statistic:.6f}")
            report_lines.append(f"- **Bootstrap Samples**: {result.n_bootstrap:,}")
            report_lines.append(f"- **Bias**: {result.bias:.6f}")
            report_lines.append(f"- **Bias-Corrected**: {result.bias_corrected_statistic:.6f}")
            report_lines.append(f"- **Standard Error**: {result.standard_error:.6f}")
            report_lines.append("")

            # Confidence intervals
            report_lines.append("### Confidence Intervals:")
            for conf_level, (lower, upper) in result.confidence_intervals.items():
                width = upper - lower
                report_lines.append(f"- **{conf_level}**: [{lower:.6f}, {upper:.6f}] (width: {width:.6f})")

            report_lines.append("")

            # Distribution properties
            skewness = stats.skew(result.bootstrap_samples)
            kurtosis = stats.kurtosis(result.bootstrap_samples)

            report_lines.append("### Bootstrap Distribution Properties:")
            report_lines.append(f"- **Skewness**: {skewness:.4f}")
            report_lines.append(f"- **Kurtosis**: {kurtosis:.4f}")
            report_lines.append(f"- **Min**: {np.min(result.bootstrap_samples):.6f}")
            report_lines.append(f"- **Max**: {np.max(result.bootstrap_samples):.6f}")
            report_lines.append("")

        # Methodological notes
        report_lines.append("## Methodological Notes")
        report_lines.append("")
        report_lines.append("- Bootstrap resampling performed with replacement")
        report_lines.append("- Bias-corrected estimates account for bootstrap bias")
        report_lines.append("- Confidence intervals computed using percentile method")
        report_lines.append("- Standard errors estimated from bootstrap distribution")
        report_lines.append("")

        report_text = "\n".join(report_lines)

        # Save report if requested
        if output_file:
            output_file.write_text(report_text)
            logger.info(f"Bootstrap analysis report saved to {output_file}")

        return report_text


# Example usage and testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Initialize bootstrap analyzer
    analyzer = BootstrapAnalyzer(n_bootstrap=1000, random_state=42)

    # Generate sample data
    np.random.seed(42)
    data1 = np.random.normal(10, 2, 100)  # ABM results
    data2 = np.random.normal(10.5, 2.2, 100)  # RSiena results

    # Test basic bootstrap
    result = analyzer.basic_bootstrap(data1, np.mean)
    print("Basic Bootstrap Results:")
    print(f"Original mean: {result.original_statistic:.4f}")
    print(f"Bias: {result.bias:.4f}")
    print(f"Standard error: {result.standard_error:.4f}")
    print(f"95% CI: {result.confidence_intervals.get('95%', 'N/A')}")
    print()

    # Test comparison between methods
    comparison = analyzer.independent_bootstrap_test(data1, data2, np.mean)
    print("Bootstrap Comparison Results:")
    print(f"Difference: {comparison.original_difference:.4f}")
    print(f"Bootstrap p-value: {comparison.p_value_bootstrap:.4f}")
    print(f"95% CI for difference: {comparison.confidence_intervals.get('95%', 'N/A')}")
    print()

    # Test hypothesis test
    test_result = analyzer.bootstrap_hypothesis_test(data1, data2,
                                                   null_difference=0.0,
                                                   alternative="two-sided")
    print("Bootstrap Hypothesis Test:")
    print(f"Test statistic: {test_result['observed_statistic']:.4f}")
    print(f"p-value: {test_result['p_value']:.4f}")

    # Generate plots
    plot_files = analyzer.generate_bootstrap_plots(result)
    print(f"\nPlots generated: {list(plot_files.keys())}")

    # Generate report
    report = analyzer.generate_bootstrap_report([result])
    print("\nBootstrap Report Preview:")
    print("=" * 50)
    print(report[:500] + "..." if len(report) > 500 else report)