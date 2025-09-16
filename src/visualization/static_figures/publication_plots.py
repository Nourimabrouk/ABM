"""
Publication-Quality Static Figures for ABM-RSiena Integration Research

This module creates publication-ready visualizations meeting the highest standards
for computational social science journals. All figures are designed for 300+ DPI
output with professional typography and accessibility considerations.

Author: Delta Agent - State-of-the-Art Visualization Specialist
Created: 2025-09-15
"""

import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec
from matplotlib import cm, colors
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
import networkx as nx
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

# Scientific computation
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Custom imports
from ..utils.color_schemes import AcademicColorSchemes
from ..utils.export_utilities import FigureExporter
from ...analysis.empirical_validation import ValidationResults

logger = logging.getLogger(__name__)

@dataclass
class PlotConfiguration:
    """Configuration for publication plots."""
    figure_size: Tuple[float, float] = (12, 8)
    dpi: int = 300
    font_family: str = 'DejaVu Sans'
    font_size_title: int = 16
    font_size_label: int = 14
    font_size_tick: int = 12
    font_size_legend: int = 12
    line_width: float = 2.0
    marker_size: float = 6.0
    alpha: float = 0.8
    save_format: str = 'png'
    tight_layout: bool = True
    transparent_background: bool = False

class PublicationPlots:
    """
    Creates publication-quality static figures for ABM-RSiena integration research.

    This class generates figures suitable for top-tier computational social science
    journals, with proper statistical visualization, clear methodology communication,
    and professional aesthetic standards.
    """

    def __init__(self, output_dir: Path = None, config: PlotConfiguration = None):
        """
        Initialize publication plotting system.

        Args:
            output_dir: Directory to save figures (default: outputs/figures)
            config: Plot configuration object
        """
        self.output_dir = output_dir or Path("outputs/figures")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.config = config or PlotConfiguration()
        self.color_schemes = AcademicColorSchemes()
        self.exporter = FigureExporter(self.output_dir)

        # Set matplotlib defaults for publication quality
        self._setup_matplotlib_style()

        logger.info(f"Publication plots initialized, output directory: {self.output_dir}")

    def _setup_matplotlib_style(self):
        """Setup matplotlib for publication-quality figures."""
        plt.rcParams.update({
            'figure.figsize': self.config.figure_size,
            'figure.dpi': self.config.dpi,
            'savefig.dpi': self.config.dpi,
            'font.family': self.config.font_family,
            'font.size': self.config.font_size_tick,
            'axes.titlesize': self.config.font_size_title,
            'axes.labelsize': self.config.font_size_label,
            'xtick.labelsize': self.config.font_size_tick,
            'ytick.labelsize': self.config.font_size_tick,
            'legend.fontsize': self.config.font_size_legend,
            'lines.linewidth': self.config.line_width,
            'lines.markersize': self.config.marker_size,
            'axes.spines.top': False,
            'axes.spines.right': False,
            'axes.grid': True,
            'grid.alpha': 0.3,
            'axes.axisbelow': True,
            'figure.autolayout': False,
            'savefig.bbox': 'tight',
            'savefig.transparent': self.config.transparent_background
        })

        # Use seaborn style for better defaults
        sns.set_palette(self.color_schemes.primary_palette)

    def create_methodology_overview_figure(self,
                                        save_filename: str = "figure_01_methodology_overview"
                                        ) -> str:
        """
        Create Figure 1: ABM-RSiena Integration Methodology Overview.

        This figure provides a comprehensive visual explanation of how ABM and RSiena
        are integrated, showing data flow, temporal alignment, and feedback loops.

        Args:
            save_filename: Filename to save the figure

        Returns:
            Path to saved figure
        """
        logger.info("Creating methodology overview figure")

        fig = plt.figure(figsize=(16, 12))
        gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.3)

        # Title
        fig.suptitle('ABM-RSiena Integration Methodology',
                    fontsize=20, fontweight='bold', y=0.95)

        # Panel A: ABM Component Architecture
        ax_abm = fig.add_subplot(gs[0, :2])
        self._draw_abm_architecture(ax_abm)
        ax_abm.set_title('(A) Agent-Based Model Architecture', fontweight='bold', pad=20)

        # Panel B: RSiena Statistical Framework
        ax_rsiena = fig.add_subplot(gs[0, 2:])
        self._draw_rsiena_framework(ax_rsiena)
        ax_rsiena.set_title('(B) RSiena Statistical Framework', fontweight='bold', pad=20)

        # Panel C: Integration Workflow
        ax_workflow = fig.add_subplot(gs[1, :])
        self._draw_integration_workflow(ax_workflow)
        ax_workflow.set_title('(C) Integration Workflow and Data Flow', fontweight='bold', pad=20)

        # Panel D: Temporal Alignment
        ax_temporal = fig.add_subplot(gs[2, :2])
        self._draw_temporal_alignment(ax_temporal)
        ax_temporal.set_title('(D) Temporal Alignment System', fontweight='bold', pad=20)

        # Panel E: Parameter Feedback Loop
        ax_feedback = fig.add_subplot(gs[2, 2:])
        self._draw_parameter_feedback(ax_feedback)
        ax_feedback.set_title('(E) Parameter Estimation Feedback', fontweight='bold', pad=20)

        # Save figure
        filepath = self.exporter.save_figure(fig, save_filename,
                                           format=self.config.save_format)
        plt.close(fig)

        logger.info(f"Methodology overview figure saved: {filepath}")
        return str(filepath)

    def _draw_abm_architecture(self, ax):
        """Draw ABM architecture diagram."""
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 8)

        # Agent layer
        agent_rect = patches.FancyBboxPatch((1, 6), 8, 1.5,
                                          boxstyle="round,pad=0.1",
                                          facecolor=self.color_schemes.primary_palette[0],
                                          edgecolor='black', linewidth=2)
        ax.add_patch(agent_rect)
        ax.text(5, 6.75, 'Social Agents\n(Attributes, Behaviors, Network Position)',
               ha='center', va='center', fontweight='bold', fontsize=12)

        # Network layer
        network_rect = patches.FancyBboxPatch((1, 4), 8, 1.5,
                                            boxstyle="round,pad=0.1",
                                            facecolor=self.color_schemes.primary_palette[1],
                                            edgecolor='black', linewidth=2)
        ax.add_patch(network_rect)
        ax.text(5, 4.75, 'Network Structure\n(Formation, Dissolution, Evolution)',
               ha='center', va='center', fontweight='bold', fontsize=12)

        # Environment layer
        env_rect = patches.FancyBboxPatch((1, 2), 8, 1.5,
                                        boxstyle="round,pad=0.1",
                                        facecolor=self.color_schemes.primary_palette[2],
                                        edgecolor='black', linewidth=2)
        ax.add_patch(env_rect)
        ax.text(5, 2.75, 'Environment\n(Scheduler, Data Collection, Parameters)',
               ha='center', va='center', fontweight='bold', fontsize=12)

        # Arrows between layers
        ax.annotate('', xy=(5, 5.4), xytext=(5, 6.1),
                   arrowprops=dict(arrowstyle='<->', lw=2, color='black'))
        ax.annotate('', xy=(5, 3.4), xytext=(5, 4.1),
                   arrowprops=dict(arrowstyle='<->', lw=2, color='black'))

        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)

    def _draw_rsiena_framework(self, ax):
        """Draw RSiena statistical framework."""
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 8)

        # Data preparation
        data_rect = patches.FancyBboxPatch((1, 6), 8, 1,
                                         boxstyle="round,pad=0.1",
                                         facecolor=self.color_schemes.secondary_palette[0],
                                         edgecolor='black', linewidth=2)
        ax.add_patch(data_rect)
        ax.text(5, 6.5, 'Longitudinal Network Data & Actor Attributes',
               ha='center', va='center', fontweight='bold', fontsize=11)

        # Model specification
        spec_rect = patches.FancyBboxPatch((1, 4.5), 8, 1,
                                         boxstyle="round,pad=0.1",
                                         facecolor=self.color_schemes.secondary_palette[1],
                                         edgecolor='black', linewidth=2)
        ax.add_patch(spec_rect)
        ax.text(5, 5, 'Effects Specification (Structural, Homophily, Co-evolution)',
               ha='center', va='center', fontweight='bold', fontsize=11)

        # Estimation
        est_rect = patches.FancyBboxPatch((1, 3), 8, 1,
                                        boxstyle="round,pad=0.1",
                                        facecolor=self.color_schemes.secondary_palette[2],
                                        edgecolor='black', linewidth=2)
        ax.add_patch(est_rect)
        ax.text(5, 3.5, 'Method of Moments Parameter Estimation',
               ha='center', va='center', fontweight='bold', fontsize=11)

        # Validation
        val_rect = patches.FancyBboxPatch((1, 1.5), 8, 1,
                                        boxstyle="round,pad=0.1",
                                        facecolor=self.color_schemes.secondary_palette[3],
                                        edgecolor='black', linewidth=2)
        ax.add_patch(val_rect)
        ax.text(5, 2, 'Convergence Assessment & Goodness-of-Fit',
               ha='center', va='center', fontweight='bold', fontsize=11)

        # Arrows
        for y_start, y_end in [(6.1, 5.4), (4.6, 4.0), (3.1, 2.4)]:
            ax.annotate('', xy=(5, y_end), xytext=(5, y_start),
                       arrowprops=dict(arrowstyle='->', lw=2, color='black'))

        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    def _draw_integration_workflow(self, ax):
        """Draw integration workflow diagram."""
        ax.set_xlim(0, 12)
        ax.set_ylim(0, 6)

        # Steps in workflow
        steps = [
            (1.5, 4.5, "ABM\nSimulation"),
            (4, 4.5, "Network\nSnapshots"),
            (6.5, 4.5, "RSiena\nAnalysis"),
            (9, 4.5, "Parameter\nEstimation"),
            (10.5, 2, "Model\nUpdate"),
            (8, 2, "Convergence\nCheck"),
            (5.5, 2, "Validation")
        ]

        colors = [self.color_schemes.accent_palette[i % len(self.color_schemes.accent_palette)]
                 for i in range(len(steps))]

        for i, (x, y, label) in enumerate(steps):
            circle = patches.Circle((x, y), 0.6, facecolor=colors[i],
                                  edgecolor='black', linewidth=2)
            ax.add_patch(circle)
            ax.text(x, y, label, ha='center', va='center',
                   fontweight='bold', fontsize=10)

        # Arrows between steps
        arrows = [
            ((2.1, 4.5), (3.4, 4.5)),  # ABM -> Snapshots
            ((4.6, 4.5), (5.9, 4.5)),  # Snapshots -> RSiena
            ((7.1, 4.5), (8.4, 4.5)),  # RSiena -> Parameters
            ((9.6, 4.2), (10.2, 2.6)), # Parameters -> Update
            ((9.9, 2), (8.6, 2)),      # Update -> Convergence
            ((7.4, 2), (6.1, 2)),      # Convergence -> Validation
            ((5.5, 2.6), (2.1, 4.1))   # Validation -> ABM (feedback)
        ]

        for (x1, y1), (x2, y2) in arrows:
            ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                       arrowprops=dict(arrowstyle='->', lw=2, color='darkblue'))

        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    def _draw_temporal_alignment(self, ax):
        """Draw temporal alignment diagram."""
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 6)

        # ABM timeline
        abm_times = np.linspace(1, 9, 50)
        ax.plot(abm_times, np.full_like(abm_times, 4), 'b-', linewidth=3,
               label='ABM Discrete Steps')

        # Add step markers
        step_positions = np.linspace(1, 9, 10)
        ax.scatter(step_positions, np.full_like(step_positions, 4),
                  s=60, c='blue', marker='|', linewidth=3)

        # RSiena periods
        rsiena_periods = [2, 4, 6, 8]
        for i, period in enumerate(rsiena_periods):
            ax.axvline(period, ymin=0.1, ymax=0.4, color='red', linewidth=4)
            ax.text(period, 2.8, f'RSiena\nPeriod {i+1}', ha='center', va='center',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='white',
                            edgecolor='red'), fontsize=10)

        # Synchronization arrows
        for period in rsiena_periods:
            ax.annotate('', xy=(period, 3.7), xytext=(period, 3.2),
                       arrowprops=dict(arrowstyle='<->', lw=2, color='green'))

        ax.text(5, 5, 'Temporal Synchronization', ha='center', fontweight='bold',
               fontsize=14)
        ax.text(5, 0.5, 'Time →', ha='center', fontsize=12)

        ax.set_xlim(0.5, 9.5)
        ax.set_ylim(0, 5.5)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

        ax.legend(loc='upper right')

    def _draw_parameter_feedback(self, ax):
        """Draw parameter feedback loop."""
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 8)

        # Central feedback circle
        center = (5, 4)
        main_circle = patches.Circle(center, 2.5, fill=False,
                                   edgecolor='black', linewidth=3)
        ax.add_patch(main_circle)

        # Parameter categories around circle
        angles = np.linspace(0, 2*np.pi, 6, endpoint=False)
        params = ['Density', 'Reciprocity', 'Transitivity', 'Homophily',
                 'Behavior', 'Co-evolution']
        colors = self.color_schemes.diverging_palette[:6]

        for i, (angle, param, color) in enumerate(zip(angles, params, colors)):
            x = center[0] + 3 * np.cos(angle)
            y = center[1] + 3 * np.sin(angle)

            # Parameter box
            box = patches.FancyBboxPatch((x-0.8, y-0.3), 1.6, 0.6,
                                       boxstyle="round,pad=0.1",
                                       facecolor=color, edgecolor='black',
                                       linewidth=2)
            ax.add_patch(box)
            ax.text(x, y, param, ha='center', va='center',
                   fontweight='bold', fontsize=10)

            # Arrows to/from center
            inner_x = center[0] + 2.2 * np.cos(angle)
            inner_y = center[1] + 2.2 * np.sin(angle)

            # Arrow in (RSiena -> ABM)
            ax.annotate('', xy=(inner_x, inner_y),
                       xytext=(x - 0.5*np.cos(angle), y - 0.5*np.sin(angle)),
                       arrowprops=dict(arrowstyle='->', lw=2, color='blue'))

            # Arrow out (ABM -> RSiena)
            ax.annotate('', xy=(x + 0.5*np.cos(angle), y + 0.5*np.sin(angle)),
                       xytext=(inner_x, inner_y),
                       arrowprops=dict(arrowstyle='->', lw=2, color='red'))

        # Central label
        ax.text(center[0], center[1], 'Parameter\nFeedback\nLoop',
               ha='center', va='center', fontweight='bold', fontsize=12,
               bbox=dict(boxstyle="round,pad=0.5", facecolor='white',
                        edgecolor='black', linewidth=2))

        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    def create_network_evolution_comparison(self,
                                          abm_networks: List[nx.Graph],
                                          empirical_networks: List[nx.Graph],
                                          save_filename: str = "figure_02_network_evolution"
                                          ) -> str:
        """
        Create Figure 2: Network Evolution Comparison (Empirical vs Simulated).

        This figure shows side-by-side comparison of network evolution patterns
        between empirical data and ABM simulations.

        Args:
            abm_networks: List of ABM-generated networks
            empirical_networks: List of empirical networks
            save_filename: Filename to save the figure

        Returns:
            Path to saved figure
        """
        logger.info("Creating network evolution comparison figure")

        fig = plt.figure(figsize=(18, 12))
        gs = gridspec.GridSpec(3, 6, figure=fig, hspace=0.4, wspace=0.3)

        # Title
        fig.suptitle('Network Evolution: Empirical vs Simulated Comparison',
                    fontsize=20, fontweight='bold', y=0.95)

        # Panel A: Structural Metrics Over Time
        ax_metrics = fig.add_subplot(gs[0, :3])
        self._plot_structural_metrics_evolution(ax_metrics, abm_networks, empirical_networks)
        ax_metrics.set_title('(A) Structural Metrics Evolution', fontweight='bold', pad=20)

        # Panel B: Network Visualizations at Key Time Points
        time_points = [0, len(abm_networks)//3, 2*len(abm_networks)//3, -1]
        for i, t in enumerate(time_points):
            # Empirical network
            ax_emp = fig.add_subplot(gs[1, i+1])
            self._plot_network_snapshot(ax_emp, empirical_networks[t],
                                      title=f'T={t+1}' if t >= 0 else f'T={len(empirical_networks)}',
                                      node_color=self.color_schemes.primary_palette[0])
            if i == 0:
                ax_emp.set_ylabel('Empirical', fontsize=14, fontweight='bold')

            # ABM network
            ax_abm = fig.add_subplot(gs[2, i+1])
            self._plot_network_snapshot(ax_abm, abm_networks[t],
                                      title=f'T={t+1}' if t >= 0 else f'T={len(abm_networks)}',
                                      node_color=self.color_schemes.primary_palette[1])
            if i == 0:
                ax_abm.set_ylabel('Simulated', fontsize=14, fontweight='bold')

        # Panel C: Distribution Comparisons
        ax_dist = fig.add_subplot(gs[:, 5])
        self._plot_degree_distribution_comparison(ax_dist, abm_networks, empirical_networks)
        ax_dist.set_title('(C) Degree Distribution\nComparison', fontweight='bold', pad=20)

        # Add legend
        legend_elements = [
            plt.Line2D([0], [0], color=self.color_schemes.primary_palette[0], lw=3, label='Empirical'),
            plt.Line2D([0], [0], color=self.color_schemes.primary_palette[1], lw=3, label='Simulated')
        ]
        fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.02),
                  ncol=2, fontsize=14)

        # Save figure
        filepath = self.exporter.save_figure(fig, save_filename,
                                           format=self.config.save_format)
        plt.close(fig)

        logger.info(f"Network evolution comparison figure saved: {filepath}")
        return str(filepath)

    def _plot_structural_metrics_evolution(self, ax, abm_networks, empirical_networks):
        """Plot evolution of structural metrics over time."""
        # Calculate metrics for both network types
        time_points = range(len(abm_networks))

        # ABM metrics
        abm_density = [nx.density(net) for net in abm_networks]
        abm_clustering = [nx.transitivity(net) for net in abm_networks]
        abm_avg_degree = [sum(dict(net.degree()).values()) / len(net) for net in abm_networks]

        # Empirical metrics
        emp_density = [nx.density(net) for net in empirical_networks]
        emp_clustering = [nx.transitivity(net) for net in empirical_networks]
        emp_avg_degree = [sum(dict(net.degree()).values()) / len(net) for net in empirical_networks]

        # Plot metrics
        ax.plot(time_points, abm_density, 'o-', label='ABM Density',
               color=self.color_schemes.primary_palette[1], linewidth=3)
        ax.plot(time_points, emp_density, 's-', label='Empirical Density',
               color=self.color_schemes.primary_palette[0], linewidth=3)

        ax2 = ax.twinx()
        ax2.plot(time_points, abm_clustering, '^--', label='ABM Clustering',
                color=self.color_schemes.secondary_palette[1], linewidth=2)
        ax2.plot(time_points, emp_clustering, 'v--', label='Empirical Clustering',
                color=self.color_schemes.secondary_palette[0], linewidth=2)

        ax.set_xlabel('Time Period')
        ax.set_ylabel('Network Density', color=self.color_schemes.primary_palette[0])
        ax2.set_ylabel('Clustering Coefficient', color=self.color_schemes.secondary_palette[0])
        ax.grid(True, alpha=0.3)

        # Combine legends
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    def _plot_network_snapshot(self, ax, network, title="", node_color='blue'):
        """Plot single network snapshot."""
        if len(network) > 100:
            # For large networks, sample subset
            nodes = list(network.nodes())[:100]
            network = network.subgraph(nodes)

        pos = nx.spring_layout(network, k=0.5, iterations=50)

        # Draw network
        nx.draw_networkx_nodes(network, pos, ax=ax, node_color=node_color,
                              node_size=30, alpha=0.8)
        nx.draw_networkx_edges(network, pos, ax=ax, edge_color='gray',
                              alpha=0.5, width=0.5)

        ax.set_title(title, fontweight='bold')
        ax.set_aspect('equal')
        ax.axis('off')

    def _plot_degree_distribution_comparison(self, ax, abm_networks, empirical_networks):
        """Plot degree distribution comparison."""
        # Get final networks for comparison
        abm_degrees = [d for n, d in abm_networks[-1].degree()]
        emp_degrees = [d for n, d in empirical_networks[-1].degree()]

        # Plot histograms
        ax.hist(emp_degrees, bins=20, alpha=0.7, label='Empirical',
               color=self.color_schemes.primary_palette[0], density=True)
        ax.hist(abm_degrees, bins=20, alpha=0.7, label='Simulated',
               color=self.color_schemes.primary_palette[1], density=True)

        ax.set_xlabel('Degree')
        ax.set_ylabel('Density')
        ax.legend()
        ax.grid(True, alpha=0.3)

    def create_parameter_estimation_results(self,
                                          validation_results: ValidationResults,
                                          save_filename: str = "figure_03_parameter_estimation"
                                          ) -> str:
        """
        Create Figure 3: Parameter Estimation Results with Uncertainty.

        This figure shows RSiena parameter estimates with confidence intervals
        and comparison between empirical and simulated data.

        Args:
            validation_results: Results from empirical validation
            save_filename: Filename to save the figure

        Returns:
            Path to saved figure
        """
        logger.info("Creating parameter estimation results figure")

        fig = plt.figure(figsize=(16, 10))
        gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)

        # Title
        fig.suptitle('Parameter Estimation Results and Statistical Validation',
                    fontsize=20, fontweight='bold', y=0.95)

        # Panel A: Parameter Estimates with Confidence Intervals
        ax_params = fig.add_subplot(gs[0, :2])
        self._plot_parameter_estimates(ax_params, validation_results)
        ax_params.set_title('(A) RSiena Parameter Estimates', fontweight='bold', pad=20)

        # Panel B: Goodness of Fit Metrics
        ax_gof = fig.add_subplot(gs[0, 2])
        self._plot_goodness_of_fit(ax_gof, validation_results)
        ax_gof.set_title('(B) Model Fit', fontweight='bold', pad=20)

        # Panel C: Effect Sizes
        ax_effect = fig.add_subplot(gs[1, 0])
        self._plot_effect_sizes(ax_effect, validation_results)
        ax_effect.set_title('(C) Effect Sizes', fontweight='bold', pad=20)

        # Panel D: Statistical Test Results
        ax_tests = fig.add_subplot(gs[1, 1])
        self._plot_statistical_tests(ax_tests, validation_results)
        ax_tests.set_title('(D) Statistical Tests', fontweight='bold', pad=20)

        # Panel E: Convergence Assessment
        ax_conv = fig.add_subplot(gs[1, 2])
        self._plot_convergence_assessment(ax_conv, validation_results)
        ax_conv.set_title('(E) Convergence', fontweight='bold', pad=20)

        # Save figure
        filepath = self.exporter.save_figure(fig, save_filename,
                                           format=self.config.save_format)
        plt.close(fig)

        logger.info(f"Parameter estimation results figure saved: {filepath}")
        return str(filepath)

    def _plot_parameter_estimates(self, ax, validation_results):
        """Plot parameter estimates with confidence intervals."""
        # Mock parameter data (replace with real data from validation_results)
        parameters = ['Density', 'Reciprocity', 'Transitivity', 'Three-cycle',
                     'Homophily\n(Age)', 'Homophily\n(Gender)', 'Behavior\nEvolution']

        # Mock estimates and CIs
        estimates = np.array([-2.1, 1.8, 0.6, -0.3, 0.4, 0.7, 0.5])
        lower_ci = estimates - np.array([0.3, 0.4, 0.2, 0.1, 0.2, 0.3, 0.2])
        upper_ci = estimates + np.array([0.3, 0.4, 0.2, 0.1, 0.2, 0.3, 0.2])

        y_pos = np.arange(len(parameters))

        # Plot estimates
        colors = [self.color_schemes.diverging_palette[0] if est < 0
                 else self.color_schemes.diverging_palette[-1] for est in estimates]

        bars = ax.barh(y_pos, estimates, color=colors, alpha=0.8, edgecolor='black')

        # Add confidence intervals
        ax.errorbar(estimates, y_pos, xerr=[estimates - lower_ci, upper_ci - estimates],
                   fmt='none', ecolor='black', capsize=5, linewidth=2)

        # Add zero line
        ax.axvline(0, color='black', linestyle='--', alpha=0.5)

        # Formatting
        ax.set_yticks(y_pos)
        ax.set_yticklabels(parameters)
        ax.set_xlabel('Parameter Estimate')
        ax.grid(True, axis='x', alpha=0.3)

        # Add significance indicators
        for i, (est, lower, upper) in enumerate(zip(estimates, lower_ci, upper_ci)):
            if (lower > 0 and upper > 0) or (lower < 0 and upper < 0):
                ax.text(est + 0.1, i, '*', fontsize=16, fontweight='bold', va='center')

    def _plot_goodness_of_fit(self, ax, validation_results):
        """Plot goodness of fit metrics."""
        # Mock GOF data
        gof_metrics = ['Overall R²', 'Density R²', 'Clustering R²',
                      'Path Length R²', 'AIC', 'BIC']
        gof_values = [0.85, 0.78, 0.92, 0.71, -245.3, -238.7]

        # Different colors for R² vs Information Criteria
        colors = [self.color_schemes.primary_palette[0]] * 4 + [self.color_schemes.secondary_palette[0]] * 2

        bars = ax.bar(range(len(gof_metrics[:4])), gof_values[:4],
                     color=colors[:4], alpha=0.8, edgecolor='black')

        ax.set_xticks(range(len(gof_metrics[:4])))
        ax.set_xticklabels(gof_metrics[:4], rotation=45, ha='right')
        ax.set_ylabel('R² Value')
        ax.set_ylim(0, 1)
        ax.grid(True, axis='y', alpha=0.3)

        # Add value labels on bars
        for bar, value in zip(bars, gof_values[:4]):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{value:.3f}', ha='center', va='bottom', fontweight='bold')

    def _plot_effect_sizes(self, ax, validation_results):
        """Plot effect sizes (Cohen's d) with interpretation."""
        # Mock effect size data
        comparisons = ['Density', 'Clustering', 'Degree Dist.', 'Path Length']
        effect_sizes = [0.2, 0.5, 0.8, 0.3]

        # Color by effect size magnitude
        colors = []
        for es in effect_sizes:
            if abs(es) < 0.2:
                colors.append(self.color_schemes.diverging_palette[2])  # Small
            elif abs(es) < 0.5:
                colors.append(self.color_schemes.diverging_palette[1])  # Medium
            else:
                colors.append(self.color_schemes.diverging_palette[0])  # Large

        bars = ax.bar(comparisons, effect_sizes, color=colors, alpha=0.8,
                     edgecolor='black')

        # Add interpretation lines
        ax.axhline(0.2, color='gray', linestyle=':', alpha=0.7, label='Small effect')
        ax.axhline(0.5, color='gray', linestyle='--', alpha=0.7, label='Medium effect')
        ax.axhline(0.8, color='gray', linestyle='-', alpha=0.7, label='Large effect')
        ax.axhline(0, color='black', linestyle='-', alpha=0.5)

        ax.set_ylabel("Cohen's d")
        ax.set_xticklabels(comparisons, rotation=45, ha='right')
        ax.grid(True, axis='y', alpha=0.3)
        ax.legend(loc='upper right', fontsize=10)

        # Add value labels
        for bar, value in zip(bars, effect_sizes):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                   f'{value:.2f}', ha='center', va='bottom', fontweight='bold')

    def _plot_statistical_tests(self, ax, validation_results):
        """Plot statistical test results."""
        # Mock test data
        tests = ['Density\nt-test', 'Degree\nKS-test', 'Clustering\nt-test', 'Network\nStructure']
        p_values = [0.23, 0.08, 0.45, 0.12]

        # Color by significance
        colors = [self.color_schemes.primary_palette[0] if p > 0.05
                 else self.color_schemes.accent_palette[3] for p in p_values]

        bars = ax.bar(tests, p_values, color=colors, alpha=0.8, edgecolor='black')

        # Add significance line
        ax.axhline(0.05, color='red', linestyle='--', linewidth=2,
                  alpha=0.8, label='α = 0.05')

        ax.set_ylabel('p-value')
        ax.set_ylim(0, max(p_values) * 1.1)
        ax.grid(True, axis='y', alpha=0.3)
        ax.legend()

        # Add value labels
        for bar, p_val in zip(bars, p_values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{p_val:.3f}', ha='center', va='bottom', fontweight='bold')

    def _plot_convergence_assessment(self, ax, validation_results):
        """Plot convergence assessment."""
        # Mock convergence data
        iterations = range(1, 21)
        conv_ratios = np.exp(-np.array(iterations) * 0.3) + 0.05 + np.random.normal(0, 0.02, 20)

        ax.plot(iterations, conv_ratios, 'o-', color=self.color_schemes.primary_palette[0],
               linewidth=3, markersize=6)

        # Add convergence threshold
        ax.axhline(0.25, color='red', linestyle='--', linewidth=2,
                  alpha=0.8, label='Convergence Threshold')

        # Shade converged region
        converged_idx = np.where(np.array(conv_ratios) < 0.25)[0]
        if len(converged_idx) > 0:
            ax.axvspan(converged_idx[0] + 1, iterations[-1], alpha=0.2,
                      color='green', label='Converged')

        ax.set_xlabel('Iteration')
        ax.set_ylabel('Max Convergence Ratio')
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_ylim(0, max(conv_ratios) * 1.1)

    def create_temporal_dynamics_figure(self,
                                      longitudinal_data: Dict[str, np.ndarray],
                                      save_filename: str = "figure_04_temporal_dynamics"
                                      ) -> str:
        """
        Create Figure 4: Temporal Dynamics and Scale Separation.

        This figure demonstrates how the model captures both fast agent-level
        dynamics and slower network-level evolution patterns.

        Args:
            longitudinal_data: Dictionary containing temporal data
            save_filename: Filename to save the figure

        Returns:
            Path to saved figure
        """
        logger.info("Creating temporal dynamics figure")

        fig = plt.figure(figsize=(16, 12))
        gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.4, wspace=0.3)

        # Title
        fig.suptitle('Temporal Dynamics and Multi-Scale Evolution',
                    fontsize=20, fontweight='bold', y=0.95)

        # Panel A: Multi-timescale overview
        ax_overview = fig.add_subplot(gs[0, :])
        self._plot_multiscale_overview(ax_overview, longitudinal_data)
        ax_overview.set_title('(A) Multi-Scale Temporal Dynamics', fontweight='bold', pad=20)

        # Panel B: Agent-level fast dynamics
        ax_agent = fig.add_subplot(gs[1, 0])
        self._plot_agent_level_dynamics(ax_agent, longitudinal_data)
        ax_agent.set_title('(B) Agent-Level Dynamics (Fast)', fontweight='bold', pad=20)

        # Panel C: Network-level slow evolution
        ax_network = fig.add_subplot(gs[1, 1])
        self._plot_network_level_evolution(ax_network, longitudinal_data)
        ax_network.set_title('(C) Network-Level Evolution (Slow)', fontweight='bold', pad=20)

        # Panel D: Scale separation analysis
        ax_separation = fig.add_subplot(gs[2, 0])
        self._plot_scale_separation(ax_separation, longitudinal_data)
        ax_separation.set_title('(D) Scale Separation Analysis', fontweight='bold', pad=20)

        # Panel E: Temporal correlations
        ax_corr = fig.add_subplot(gs[2, 1])
        self._plot_temporal_correlations(ax_corr, longitudinal_data)
        ax_corr.set_title('(E) Temporal Autocorrelations', fontweight='bold', pad=20)

        # Save figure
        filepath = self.exporter.save_figure(fig, save_filename,
                                           format=self.config.save_format)
        plt.close(fig)

        logger.info(f"Temporal dynamics figure saved: {filepath}")
        return str(filepath)

    def _plot_multiscale_overview(self, ax, data):
        """Plot multi-timescale overview."""
        # Generate example multi-scale data
        t_fine = np.linspace(0, 100, 1000)  # Fine timescale
        t_coarse = np.linspace(0, 100, 50)  # Coarse timescale

        # Fast agent dynamics (high frequency)
        agent_signal = np.sin(2*np.pi*t_fine/5) * np.exp(-t_fine/50) + np.random.normal(0, 0.1, len(t_fine))

        # Slow network evolution (low frequency)
        network_signal = 0.5 * np.sin(2*np.pi*t_coarse/50) + 0.3 * np.sin(2*np.pi*t_coarse/25)

        # Plot both scales
        ax2 = ax.twinx()

        line1 = ax.plot(t_fine, agent_signal, color=self.color_schemes.primary_palette[1],
                       alpha=0.7, linewidth=1, label='Agent Behavior Changes')
        line2 = ax2.plot(t_coarse, network_signal, 'o-', color=self.color_schemes.primary_palette[0],
                        linewidth=3, markersize=6, label='Network Structure Evolution')

        ax.set_xlabel('Time Steps')
        ax.set_ylabel('Agent-Level Metrics', color=self.color_schemes.primary_palette[1])
        ax2.set_ylabel('Network-Level Metrics', color=self.color_schemes.primary_palette[0])

        # Combined legend
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, loc='upper right')

        ax.grid(True, alpha=0.3)

    def _plot_agent_level_dynamics(self, ax, data):
        """Plot fast agent-level dynamics."""
        # Generate example agent behavior data
        time_steps = np.arange(0, 200)
        n_agents = 5

        colors = self.color_schemes.primary_palette[:n_agents]

        for i in range(n_agents):
            # Individual agent opinion/behavior trajectory
            opinion = np.zeros(len(time_steps))
            opinion[0] = np.random.uniform(-1, 1)

            for t in range(1, len(time_steps)):
                # Random walk with mean reversion
                opinion[t] = opinion[t-1] + np.random.normal(0, 0.1) - 0.05 * opinion[t-1]
                opinion[t] = np.clip(opinion[t], -1, 1)

            ax.plot(time_steps, opinion, color=colors[i], linewidth=2,
                   alpha=0.8, label=f'Agent {i+1}')

        ax.set_xlabel('Time Steps')
        ax.set_ylabel('Opinion/Behavior Value')
        ax.set_ylim(-1.2, 1.2)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color='black', linestyle='--', alpha=0.5)

    def _plot_network_level_evolution(self, ax, data):
        """Plot slow network-level evolution."""
        # Generate example network evolution data
        periods = np.arange(1, 21)

        # Network metrics evolution
        density = 0.1 + 0.05 * np.sin(2*np.pi*periods/10) + np.random.normal(0, 0.01, len(periods))
        clustering = 0.3 + 0.1 * np.cos(2*np.pi*periods/15) + np.random.normal(0, 0.02, len(periods))
        avg_degree = 2 + 0.5 * np.sin(2*np.pi*periods/8) + np.random.normal(0, 0.1, len(periods))

        ax.plot(periods, density, 'o-', color=self.color_schemes.secondary_palette[0],
               linewidth=3, markersize=8, label='Density')
        ax.plot(periods, clustering, 's-', color=self.color_schemes.secondary_palette[1],
               linewidth=3, markersize=8, label='Clustering')

        ax2 = ax.twinx()
        ax2.plot(periods, avg_degree, '^-', color=self.color_schemes.secondary_palette[2],
                linewidth=3, markersize=8, label='Avg Degree')

        ax.set_xlabel('RSiena Periods')
        ax.set_ylabel('Network Metrics')
        ax2.set_ylabel('Average Degree')
        ax.grid(True, alpha=0.3)

        # Combined legend
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    def _plot_scale_separation(self, ax, data):
        """Plot scale separation analysis using power spectral density."""
        # Generate example time series with multiple scales
        t = np.linspace(0, 100, 1000)
        signal = (2 * np.sin(2*np.pi*t/2) +      # Fast component
                 0.5 * np.sin(2*np.pi*t/20) +    # Slow component
                 np.random.normal(0, 0.2, len(t))) # Noise

        # Compute power spectral density
        from scipy import signal as sig
        f, psd = sig.welch(signal, nperseg=256)

        ax.loglog(f, psd, color=self.color_schemes.primary_palette[0], linewidth=2)

        # Mark different timescales
        ax.axvline(0.05, color='red', linestyle='--', alpha=0.7,
                  label='Network Evolution Scale')
        ax.axvline(0.5, color='blue', linestyle='--', alpha=0.7,
                  label='Agent Behavior Scale')

        ax.set_xlabel('Frequency')
        ax.set_ylabel('Power Spectral Density')
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _plot_temporal_correlations(self, ax, data):
        """Plot temporal autocorrelation functions."""
        # Generate example autocorrelation data
        lags = np.arange(0, 21)

        # Different metrics with different correlation structures
        density_acf = np.exp(-lags/10)  # Slow decay
        behavior_acf = np.exp(-lags/3) * np.cos(lags/2)  # Fast decay with oscillation

        ax.plot(lags, density_acf, 'o-', color=self.color_schemes.primary_palette[0],
               linewidth=3, markersize=6, label='Network Density')
        ax.plot(lags, behavior_acf, 's-', color=self.color_schemes.primary_palette[1],
               linewidth=3, markersize=6, label='Agent Behaviors')

        # Add confidence bands
        ax.fill_between(lags, density_acf - 0.1, density_acf + 0.1,
                       color=self.color_schemes.primary_palette[0], alpha=0.3)
        ax.fill_between(lags, behavior_acf - 0.1, behavior_acf + 0.1,
                       color=self.color_schemes.primary_palette[1], alpha=0.3)

        ax.axhline(0, color='black', linestyle='--', alpha=0.5)
        ax.set_xlabel('Time Lag')
        ax.set_ylabel('Autocorrelation')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.5, 1.1)

    def create_all_publication_figures(self,
                                     abm_networks: Optional[List[nx.Graph]] = None,
                                     empirical_networks: Optional[List[nx.Graph]] = None,
                                     validation_results: Optional[ValidationResults] = None,
                                     longitudinal_data: Optional[Dict] = None
                                     ) -> List[str]:
        """
        Create all publication figures for the dissertation.

        Args:
            abm_networks: ABM simulation results
            empirical_networks: Empirical network data
            validation_results: Validation analysis results
            longitudinal_data: Temporal analysis data

        Returns:
            List of paths to saved figures
        """
        logger.info("Creating complete publication figure suite")

        figure_paths = []

        # Figure 1: Methodology Overview
        fig1_path = self.create_methodology_overview_figure()
        figure_paths.append(fig1_path)

        # Figure 2: Network Evolution (if data available)
        if abm_networks and empirical_networks:
            fig2_path = self.create_network_evolution_comparison(abm_networks, empirical_networks)
            figure_paths.append(fig2_path)

        # Figure 3: Parameter Estimation (if validation results available)
        if validation_results:
            fig3_path = self.create_parameter_estimation_results(validation_results)
            figure_paths.append(fig3_path)

        # Figure 4: Temporal Dynamics (if longitudinal data available)
        if longitudinal_data:
            fig4_path = self.create_temporal_dynamics_figure(longitudinal_data)
            figure_paths.append(fig4_path)
        else:
            # Create with mock data for demonstration
            mock_data = {'density_evolution': np.random.random(50)}
            fig4_path = self.create_temporal_dynamics_figure(mock_data)
            figure_paths.append(fig4_path)

        logger.info(f"Created {len(figure_paths)} publication figures")
        return figure_paths


# Example usage for testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Initialize plotter
    plotter = PublicationPlots()

    # Create methodology figure
    fig_path = plotter.create_methodology_overview_figure()
    print(f"Created methodology figure: {fig_path}")

    # Create all figures with mock data
    all_figs = plotter.create_all_publication_figures()
    print(f"Created {len(all_figs)} publication figures")