"""
Advanced Multi-Scale Visualization System

This module creates sophisticated visualizations that capture multiple scales
of social phenomena, from individual agent behaviors to network-level patterns
and emergent macro-structures.

Author: Delta Agent - State-of-the-Art Visualization Specialist
Created: 2025-09-15
"""

import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec
from matplotlib.patches import Circle, FancyBboxPatch, Rectangle
from matplotlib.collections import LineCollection, PatchCollection
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass
import networkx as nx
import warnings
warnings.filterwarnings('ignore')

# Scientific computation
from scipy import stats, spatial, cluster
from scipy.signal import find_peaks, savgol_filter
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.animation as animation

# Custom imports
from ..utils.color_schemes import AcademicColorSchemes
from ..utils.export_utilities import FigureExporter

logger = logging.getLogger(__name__)

@dataclass
class MultiScaleConfiguration:
    """Configuration for multi-scale visualizations."""
    figure_size: Tuple[float, float] = (16, 12)
    dpi: int = 300
    agent_marker_size: float = 50
    network_alpha: float = 0.7
    macro_alpha: float = 0.8
    time_window: int = 50
    spatial_bins: int = 20
    clustering_method: str = 'kmeans'
    n_clusters: int = 5
    color_intensity_range: Tuple[float, float] = (0.3, 1.0)

class MultiScaleVisualizer:
    """
    Advanced multi-scale visualization system for ABM-RSiena research.

    Creates sophisticated visualizations showing the interplay between:
    - Individual agent behaviors (micro-scale)
    - Local network structures (meso-scale)
    - Global network patterns (macro-scale)
    - Temporal evolution across all scales
    """

    def __init__(self,
                 output_dir: Path = None,
                 config: MultiScaleConfiguration = None):
        """
        Initialize multi-scale visualizer.

        Args:
            output_dir: Directory for saving visualizations
            config: Configuration for visualizations
        """
        self.output_dir = output_dir or Path("outputs/advanced_visualizations")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.config = config or MultiScaleConfiguration()
        self.color_schemes = AcademicColorSchemes()
        self.exporter = FigureExporter(self.output_dir)

        logger.info(f"Multi-scale visualizer initialized, output: {self.output_dir}")

    def create_multiscale_overview(self,
                                 networks: List[nx.Graph],
                                 agent_data: List[Dict[int, Dict]],
                                 time_series_data: pd.DataFrame,
                                 title: str = "Multi-Scale Social Network Analysis",
                                 save_filename: str = "multiscale_overview") -> str:
        """
        Create comprehensive multi-scale visualization overview.

        Args:
            networks: List of network snapshots over time
            agent_data: Agent attribute data for each time point
            time_series_data: Aggregated time series metrics
            title: Figure title
            save_filename: Output filename

        Returns:
            Path to saved figure
        """
        logger.info("Creating multi-scale overview visualization")

        # Create figure with complex grid layout
        fig = plt.figure(figsize=self.config.figure_size, dpi=self.config.dpi)
        gs = gridspec.GridSpec(
            4, 6, figure=fig,
            height_ratios=[1.5, 1.5, 1, 1],
            width_ratios=[1, 1, 1, 1, 1, 0.8],
            hspace=0.4, wspace=0.4
        )

        # Main title
        fig.suptitle(title, fontsize=20, fontweight='bold', y=0.95)

        # Panel A: Micro-scale - Individual agent behaviors
        ax_micro = fig.add_subplot(gs[0, :2])
        self._plot_agent_trajectories(ax_micro, agent_data, networks)
        ax_micro.set_title('(A) Micro-Scale: Individual Agent Behaviors',
                          fontweight='bold', fontsize=14, pad=15)

        # Panel B: Meso-scale - Local network structures
        ax_meso = fig.add_subplot(gs[0, 2:4])
        self._plot_local_structures(ax_meso, networks[-1] if networks else nx.Graph())
        ax_meso.set_title('(B) Meso-Scale: Local Network Structures',
                         fontweight='bold', fontsize=14, pad=15)

        # Panel C: Macro-scale - Global network patterns
        ax_macro = fig.add_subplot(gs[0, 4:])
        self._plot_global_patterns(ax_macro, networks, time_series_data)
        ax_macro.set_title('(C) Macro-Scale: Global Patterns',
                          fontweight='bold', fontsize=14, pad=15)

        # Panel D: Temporal evolution across scales
        ax_temporal = fig.add_subplot(gs[1, :3])
        self._plot_temporal_multiscale(ax_temporal, time_series_data, agent_data)
        ax_temporal.set_title('(D) Temporal Evolution Across Scales',
                            fontweight='bold', fontsize=14, pad=15)

        # Panel E: Scale interactions and feedback
        ax_interactions = fig.add_subplot(gs[1, 3:])
        self._plot_scale_interactions(ax_interactions, networks, agent_data, time_series_data)
        ax_interactions.set_title('(E) Cross-Scale Interactions',
                                fontweight='bold', fontsize=14, pad=15)

        # Panel F: Emergence patterns
        ax_emergence = fig.add_subplot(gs[2, :3])
        self._plot_emergence_patterns(ax_emergence, networks, time_series_data)
        ax_emergence.set_title('(F) Emergent Pattern Detection',
                             fontweight='bold', fontsize=14, pad=15)

        # Panel G: Multi-scale clustering
        ax_clustering = fig.add_subplot(gs[2, 3:])
        self._plot_multiscale_clustering(ax_clustering, networks[-1] if networks else nx.Graph(),
                                       agent_data[-1] if agent_data else {})
        ax_clustering.set_title('(G) Multi-Scale Clustering',
                              fontweight='bold', fontsize=14, pad=15)

        # Panel H: Scale separation analysis
        ax_separation = fig.add_subplot(gs[3, :2])
        self._plot_scale_separation(ax_separation, time_series_data)
        ax_separation.set_title('(H) Scale Separation Analysis',
                              fontweight='bold', fontsize=14, pad=15)

        # Panel I: Hierarchical structure
        ax_hierarchy = fig.add_subplot(gs[3, 2:4])
        self._plot_hierarchical_structure(ax_hierarchy, networks[-1] if networks else nx.Graph())
        ax_hierarchy.set_title('(I) Hierarchical Network Structure',
                             fontweight='bold', fontsize=14, pad=15)

        # Panel J: Summary metrics
        ax_summary = fig.add_subplot(gs[3, 4:])
        self._plot_summary_metrics(ax_summary, networks, time_series_data)
        ax_summary.set_title('(J) Multi-Scale Summary',
                           fontweight='bold', fontsize=14, pad=15)

        # Save figure
        filepath = self.exporter.save_figure(fig, save_filename, 'png', 'publication')
        plt.close(fig)

        logger.info(f"Multi-scale overview saved: {filepath}")
        return str(filepath)

    def _plot_agent_trajectories(self, ax, agent_data, networks):
        """Plot individual agent behavior trajectories."""
        if not agent_data or not networks:
            ax.text(0.5, 0.5, 'No agent data available', ha='center', va='center',
                   transform=ax.transAxes, fontsize=12, color='gray')
            return

        # Select representative agents
        n_agents_to_show = min(8, len(agent_data[0]) if agent_data[0] else 0)
        if n_agents_to_show == 0:
            return

        selected_agents = list(agent_data[0].keys())[:n_agents_to_show]
        colors = self.color_schemes.get_palette('qualitative', n_agents_to_show)

        # Extract behavior trajectories
        time_points = range(len(agent_data))

        for i, agent_id in enumerate(selected_agents):
            trajectory = []
            for t in time_points:
                if agent_id in agent_data[t]:
                    # Use opinion as primary behavior if available
                    opinion = agent_data[t][agent_id].get('opinion', np.random.random() - 0.5)
                    trajectory.append(opinion)
                else:
                    trajectory.append(np.nan)

            if trajectory and not all(np.isnan(trajectory)):
                ax.plot(time_points, trajectory, 'o-', color=colors[i],
                       label=f'Agent {agent_id}', alpha=0.8, linewidth=2, markersize=4)

        ax.set_xlabel('Time Steps')
        ax.set_ylabel('Opinion/Behavior Value')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color='black', linestyle='--', alpha=0.5)

    def _plot_local_structures(self, ax, network):
        """Plot local network structures and motifs."""
        if len(network) == 0:
            ax.text(0.5, 0.5, 'No network data', ha='center', va='center',
                   transform=ax.transAxes, fontsize=12, color='gray')
            return

        # Calculate local clustering for each node
        clustering = nx.clustering(network)

        # Get node positions
        pos = nx.spring_layout(network, k=1/np.sqrt(len(network)), iterations=50)

        # Create color map based on local clustering
        if clustering:
            node_colors = [clustering[node] for node in network.nodes()]
            vmin, vmax = min(node_colors), max(node_colors)
        else:
            node_colors = ['lightblue'] * len(network)
            vmin, vmax = 0, 1

        # Draw network with clustering-based coloring
        nodes = nx.draw_networkx_nodes(
            network, pos, ax=ax,
            node_color=node_colors,
            node_size=100,
            cmap=plt.cm.RdYlBu_r,
            vmin=vmin, vmax=vmax,
            alpha=self.config.network_alpha
        )

        nx.draw_networkx_edges(
            network, pos, ax=ax,
            edge_color='gray',
            alpha=0.5,
            width=0.5
        )

        # Highlight triangles (3-cycles)
        triangles = [cycle for cycle in nx.simple_cycles(network, length_limit=3) if len(cycle) == 3]
        for triangle in triangles[:10]:  # Show first 10 triangles
            triangle_pos = [pos[node] for node in triangle]
            triangle_patch = patches.Polygon(triangle_pos, closed=True,
                                           fill=False, edgecolor='red',
                                           linewidth=2, alpha=0.7)
            ax.add_patch(triangle_patch)

        # Add colorbar for clustering
        if hasattr(nodes, 'get_array') and nodes.get_array() is not None:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cbar = plt.colorbar(nodes, cax=cax)
            cbar.set_label('Local Clustering', fontsize=10)

        ax.set_aspect('equal')
        ax.axis('off')

    def _plot_global_patterns(self, ax, networks, time_series_data):
        """Plot global network patterns and evolution."""
        if not networks:
            ax.text(0.5, 0.5, 'No network data', ha='center', va='center',
                   transform=ax.transAxes, fontsize=12, color='gray')
            return

        # Calculate global metrics over time
        time_points = range(len(networks))
        densities = []
        avg_clusterings = []
        assortativity = []

        for net in networks:
            if len(net) > 0:
                densities.append(nx.density(net))
                avg_clusterings.append(nx.transitivity(net))
                try:
                    assort = nx.degree_assortativity_coefficient(net)
                    assortativity.append(assort if not np.isnan(assort) else 0)
                except:
                    assortativity.append(0)
            else:
                densities.append(0)
                avg_clusterings.append(0)
                assortativity.append(0)

        # Plot global metrics
        ax2 = ax.twinx()
        ax3 = ax.twinx()

        # Offset the third axis
        ax3.spines['right'].set_position(('outward', 60))

        line1 = ax.plot(time_points, densities, 'o-',
                       color=self.color_schemes.primary_palette[0],
                       linewidth=2, markersize=4, label='Density')

        line2 = ax2.plot(time_points, avg_clusterings, 's-',
                        color=self.color_schemes.primary_palette[1],
                        linewidth=2, markersize=4, label='Clustering')

        line3 = ax3.plot(time_points, assortativity, '^-',
                        color=self.color_schemes.primary_palette[2],
                        linewidth=2, markersize=4, label='Assortativity')

        # Labels and colors
        ax.set_xlabel('Time Steps')
        ax.set_ylabel('Density', color=self.color_schemes.primary_palette[0])
        ax2.set_ylabel('Clustering', color=self.color_schemes.primary_palette[1])
        ax3.set_ylabel('Assortativity', color=self.color_schemes.primary_palette[2])

        ax.tick_params(axis='y', labelcolor=self.color_schemes.primary_palette[0])
        ax2.tick_params(axis='y', labelcolor=self.color_schemes.primary_palette[1])
        ax3.tick_params(axis='y', labelcolor=self.color_schemes.primary_palette[2])

        # Combined legend
        lines = line1 + line2 + line3
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, loc='upper left', fontsize=10)

        ax.grid(True, alpha=0.3)

    def _plot_temporal_multiscale(self, ax, time_series_data, agent_data):
        """Plot temporal evolution across multiple scales."""
        if time_series_data is None or len(time_series_data) == 0:
            ax.text(0.5, 0.5, 'No time series data', ha='center', va='center',
                   transform=ax.transAxes, fontsize=12, color='gray')
            return

        # Create multi-scale time series
        time_points = time_series_data.index if hasattr(time_series_data, 'index') else range(len(time_series_data))

        # Smooth the signals at different scales
        scales = [1, 5, 15]  # Different temporal smoothing scales
        colors = self.color_schemes.get_palette('primary', len(scales))

        if 'network_density' in time_series_data.columns:
            density_values = time_series_data['network_density'].values

            for i, scale in enumerate(scales):
                if len(density_values) > scale * 2:
                    # Apply smoothing
                    if scale == 1:
                        smoothed = density_values
                        label = 'Raw (micro-scale)'
                    else:
                        smoothed = savgol_filter(density_values,
                                               min(scale * 2 + 1, len(density_values) // 2),
                                               1)
                        label = f'Smoothed (scale {scale})'

                    ax.plot(time_points[:len(smoothed)], smoothed,
                           color=colors[i], linewidth=2 + i,
                           alpha=0.7 + i * 0.1, label=label)

        # Add phase space representation
        if len(agent_data) > 10:
            # Calculate agent behavior variance over time
            behavior_variance = []
            for t in range(len(agent_data)):
                if agent_data[t]:
                    opinions = [data.get('opinion', 0) for data in agent_data[t].values()]
                    behavior_variance.append(np.var(opinions) if opinions else 0)
                else:
                    behavior_variance.append(0)

            # Normalize and plot as second series
            if behavior_variance and max(behavior_variance) > 0:
                behavior_variance = np.array(behavior_variance) / max(behavior_variance)
                ax2 = ax.twinx()
                ax2.plot(range(len(behavior_variance)), behavior_variance,
                        '--', color=self.color_schemes.accent_palette[0],
                        linewidth=2, alpha=0.8, label='Behavior Diversity')
                ax2.set_ylabel('Normalized Behavior Diversity')
                ax2.tick_params(axis='y', labelcolor=self.color_schemes.accent_palette[0])

        ax.set_xlabel('Time Steps')
        ax.set_ylabel('Network Density')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

    def _plot_scale_interactions(self, ax, networks, agent_data, time_series_data):
        """Plot interactions and feedback between scales."""
        if not networks or not agent_data:
            ax.text(0.5, 0.5, 'Insufficient data\nfor scale interactions',
                   ha='center', va='center', transform=ax.transAxes,
                   fontsize=12, color='gray')
            return

        # Calculate cross-scale correlations
        n_timepoints = min(len(networks), len(agent_data))
        if n_timepoints < 5:
            ax.text(0.5, 0.5, 'Need more timepoints\nfor correlation analysis',
                   ha='center', va='center', transform=ax.transAxes,
                   fontsize=12, color='gray')
            return

        # Micro-scale: individual behavior variance
        micro_values = []
        for t in range(n_timepoints):
            if agent_data[t]:
                opinions = [data.get('opinion', 0) for data in agent_data[t].values()]
                micro_values.append(np.var(opinions) if opinions else 0)
            else:
                micro_values.append(0)

        # Macro-scale: network density
        macro_values = []
        for t in range(n_timepoints):
            if len(networks[t]) > 0:
                macro_values.append(nx.density(networks[t]))
            else:
                macro_values.append(0)

        # Create scatter plot showing relationship
        if len(micro_values) == len(macro_values) and len(micro_values) > 0:
            scatter = ax.scatter(micro_values, macro_values,
                               c=range(len(micro_values)),
                               cmap='viridis', s=60, alpha=0.8)

            # Add correlation line if significant correlation
            if len(micro_values) > 3:
                try:
                    correlation, p_value = stats.pearsonr(micro_values, macro_values)
                    if p_value < 0.05:
                        z = np.polyfit(micro_values, macro_values, 1)
                        p = np.poly1d(z)
                        ax.plot(micro_values, p(micro_values), "r--", alpha=0.8,
                               linewidth=2, label=f'r={correlation:.3f}, p={p_value:.3f}')
                except:
                    correlation = 0

            # Add colorbar for time
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cbar = plt.colorbar(scatter, cax=cax)
            cbar.set_label('Time Step', fontsize=10)

            ax.set_xlabel('Micro-Scale (Behavior Variance)')
            ax.set_ylabel('Macro-Scale (Network Density)')
            if 'correlation' in locals():
                ax.legend(loc='best')

        ax.grid(True, alpha=0.3)

    def _plot_emergence_patterns(self, ax, networks, time_series_data):
        """Plot emergent pattern detection across scales."""
        if not networks:
            ax.text(0.5, 0.5, 'No network data', ha='center', va='center',
                   transform=ax.transAxes, fontsize=12, color='gray')
            return

        # Calculate emergence indicators
        time_points = range(len(networks))
        emergence_metrics = {
            'complexity': [],
            'order': [],
            'adaptability': []
        }

        for net in networks:
            if len(net) == 0:
                emergence_metrics['complexity'].append(0)
                emergence_metrics['order'].append(0)
                emergence_metrics['adaptability'].append(0)
                continue

            # Complexity: network entropy based on degree distribution
            degrees = [d for n, d in net.degree()]
            if degrees:
                degree_counts = np.bincount(degrees)
                degree_probs = degree_counts / sum(degree_counts)
                entropy = -sum(p * np.log(p) for p in degree_probs if p > 0)
                emergence_metrics['complexity'].append(entropy)
            else:
                emergence_metrics['complexity'].append(0)

            # Order: clustering coefficient
            emergence_metrics['order'].append(nx.transitivity(net))

            # Adaptability: edge changes (approximated as density variation)
            emergence_metrics['adaptability'].append(nx.density(net))

        # Normalize metrics
        for key in emergence_metrics:
            values = emergence_metrics[key]
            if max(values) > 0:
                emergence_metrics[key] = [v / max(values) for v in values]

        # Plot emergence patterns
        colors = self.color_schemes.get_palette('accent', 3)
        for i, (metric, values) in enumerate(emergence_metrics.items()):
            ax.plot(time_points, values, 'o-', color=colors[i],
                   linewidth=2, markersize=4, label=metric.capitalize(),
                   alpha=0.8)

        ax.set_xlabel('Time Steps')
        ax.set_ylabel('Normalized Emergence Metric')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.1)

    def _plot_multiscale_clustering(self, ax, network, agent_attributes):
        """Plot multi-scale clustering analysis."""
        if len(network) == 0:
            ax.text(0.5, 0.5, 'No network data', ha='center', va='center',
                   transform=ax.transAxes, fontsize=12, color='gray')
            return

        # Perform community detection at multiple scales
        try:
            communities = nx.community.greedy_modularity_communities(network)
        except:
            # Fallback to simple clustering
            communities = [set(network.nodes())]

        # Get positions
        pos = nx.spring_layout(network, k=1/np.sqrt(len(network)), iterations=50)

        # Color nodes by community
        node_colors = []
        community_colors = self.color_schemes.get_palette('qualitative', len(communities))

        for node in network.nodes():
            for i, community in enumerate(communities):
                if node in community:
                    node_colors.append(community_colors[i % len(community_colors)])
                    break
            else:
                node_colors.append('gray')

        # Draw network with community coloring
        nx.draw_networkx_nodes(network, pos, ax=ax,
                              node_color=node_colors,
                              node_size=80, alpha=0.8)

        nx.draw_networkx_edges(network, pos, ax=ax,
                              edge_color='gray', alpha=0.5, width=0.5)

        # Draw community boundaries
        for i, community in enumerate(communities):
            if len(community) > 2:
                community_pos = [pos[node] for node in community]
                hull_points = spatial.ConvexHull(community_pos)
                hull_vertices = [community_pos[i] for i in hull_points.vertices]
                hull_patch = patches.Polygon(hull_vertices, closed=True,
                                           fill=False, edgecolor=community_colors[i],
                                           linewidth=2, alpha=0.7, linestyle='--')
                ax.add_patch(hull_patch)

        ax.set_aspect('equal')
        ax.axis('off')

        # Add modularity score
        try:
            modularity = nx.community.modularity(network, communities)
            ax.text(0.02, 0.98, f'Modularity: {modularity:.3f}\nCommunities: {len(communities)}',
                   transform=ax.transAxes, fontsize=10,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8),
                   verticalalignment='top')
        except:
            pass

    def _plot_scale_separation(self, ax, time_series_data):
        """Plot scale separation analysis using spectral methods."""
        if time_series_data is None or len(time_series_data) < 10:
            ax.text(0.5, 0.5, 'Insufficient data\nfor spectral analysis',
                   ha='center', va='center', transform=ax.transAxes,
                   fontsize=12, color='gray')
            return

        # Use network density as the main signal
        if 'network_density' in time_series_data.columns:
            signal = time_series_data['network_density'].values
        else:
            # Create synthetic signal for demonstration
            t = np.arange(len(time_series_data))
            signal = 0.5 + 0.3 * np.sin(2 * np.pi * t / 20) + 0.1 * np.sin(2 * np.pi * t / 5) + \
                    0.05 * np.random.random(len(t))

        # Compute power spectral density
        from scipy import signal as sig
        frequencies, psd = sig.welch(signal, nperseg=min(len(signal)//4, 32))

        # Plot power spectrum
        ax.loglog(frequencies[1:], psd[1:], 'b-', linewidth=2,
                 color=self.color_schemes.primary_palette[0])

        # Mark different scales
        ax.axvline(0.1, color='red', linestyle='--', alpha=0.7, linewidth=2,
                  label='Network Evolution Scale')
        ax.axvline(0.3, color='green', linestyle='--', alpha=0.7, linewidth=2,
                  label='Agent Behavior Scale')

        ax.set_xlabel('Frequency')
        ax.set_ylabel('Power Spectral Density')
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _plot_hierarchical_structure(self, ax, network):
        """Plot hierarchical network structure analysis."""
        if len(network) == 0:
            ax.text(0.5, 0.5, 'No network data', ha='center', va='center',
                   transform=ax.transAxes, fontsize=12, color='gray')
            return

        # Calculate centrality measures
        centralities = {
            'degree': nx.degree_centrality(network),
            'betweenness': nx.betweenness_centrality(network),
            'closeness': nx.closeness_centrality(network)
        }

        # Create hierarchical layout based on betweenness centrality
        betweenness = centralities['betweenness']

        # Sort nodes by centrality
        sorted_nodes = sorted(network.nodes(), key=lambda x: betweenness[x], reverse=True)

        # Create hierarchical positions
        levels = 5
        nodes_per_level = len(sorted_nodes) // levels + 1
        pos = {}

        for i, node in enumerate(sorted_nodes):
            level = i // nodes_per_level
            position_in_level = i % nodes_per_level
            x = (position_in_level - nodes_per_level/2) * 0.8 / max(nodes_per_level/2, 1)
            y = (levels - level - 1) * 0.8
            pos[node] = (x, y)

        # Node colors based on centrality
        node_colors = [betweenness[node] for node in network.nodes()]

        # Draw network
        nodes = nx.draw_networkx_nodes(network, pos, ax=ax,
                                      node_color=node_colors,
                                      node_size=[centralities['degree'][node] * 500 + 50
                                               for node in network.nodes()],
                                      cmap=plt.cm.Reds, alpha=0.8)

        nx.draw_networkx_edges(network, pos, ax=ax,
                              edge_color='gray', alpha=0.5, width=0.5)

        # Add colorbar
        if hasattr(nodes, 'get_array') and nodes.get_array() is not None:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cbar = plt.colorbar(nodes, cax=cax)
            cbar.set_label('Betweenness Centrality', fontsize=10)

        ax.set_aspect('equal')
        ax.axis('off')

    def _plot_summary_metrics(self, ax, networks, time_series_data):
        """Plot summary metrics across all scales."""
        if not networks:
            ax.text(0.5, 0.5, 'No data for\nsummary metrics', ha='center', va='center',
                   transform=ax.transAxes, fontsize=12, color='gray')
            return

        # Calculate summary statistics
        metrics = {
            'Micro\n(Agents)': 0,
            'Meso\n(Local)': 0,
            'Macro\n(Global)': 0,
            'Temporal\n(Evolution)': 0
        }

        # Micro-scale: average degree
        final_network = networks[-1]
        if len(final_network) > 0:
            metrics['Micro\n(Agents)'] = sum(dict(final_network.degree()).values()) / len(final_network)

        # Meso-scale: clustering coefficient
        if len(final_network) > 0:
            metrics['Meso\n(Local)'] = nx.transitivity(final_network)

        # Macro-scale: network efficiency
        if len(final_network) > 0 and nx.is_connected(final_network):
            try:
                metrics['Macro\n(Global)'] = nx.global_efficiency(final_network)
            except:
                metrics['Macro\n(Global)'] = nx.density(final_network)

        # Temporal: stability (inverse of density variance)
        densities = [nx.density(net) for net in networks if len(net) > 0]
        if densities:
            metrics['Temporal\n(Evolution)'] = 1 - (np.var(densities) / max(np.mean(densities), 0.001))

        # Create radar/bar chart
        scales = list(metrics.keys())
        values = list(metrics.values())

        # Normalize values to 0-1 range
        max_val = max(values) if max(values) > 0 else 1
        normalized_values = [v / max_val for v in values]

        bars = ax.bar(scales, normalized_values,
                     color=self.color_schemes.get_palette('primary', len(scales)),
                     alpha=0.8, edgecolor='black', linewidth=1)

        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                   f'{value:.3f}', ha='center', va='bottom', fontweight='bold')

        ax.set_ylabel('Normalized Metric Value')
        ax.set_ylim(0, 1.2)
        ax.grid(True, axis='y', alpha=0.3)

        # Rotate x labels for better readability
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')


# Example usage and testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Create sample data for testing
    def create_sample_multiscale_data():
        """Create sample data for multi-scale visualization testing."""
        networks = []
        agent_data = []

        # Create evolving network
        n_agents = 30
        base_network = nx.erdos_renyi_graph(n_agents, 0.1, seed=42)

        for t in range(20):
            # Evolve network
            evolved = base_network.copy()

            # Add some edges
            non_edges = list(nx.non_edges(evolved))
            if non_edges and len(non_edges) > 2:
                new_edges = np.random.choice(len(non_edges), size=2, replace=False)
                for idx in new_edges:
                    evolved.add_edge(*non_edges[idx])

            networks.append(evolved)

            # Create agent data
            agent_dict = {}
            for agent in range(n_agents):
                agent_dict[agent] = {
                    'opinion': np.sin(t * 0.3 + agent * 0.1) + np.random.normal(0, 0.1),
                    'age': 25 + agent,
                    'active': np.random.random() > 0.1
                }
            agent_data.append(agent_dict)

            base_network = evolved.copy()

        # Create time series data
        time_series = pd.DataFrame({
            'network_density': [nx.density(net) for net in networks],
            'clustering_coefficient': [nx.transitivity(net) for net in networks],
            'average_degree': [sum(dict(net.degree()).values()) / max(len(net), 1) for net in networks]
        })

        return networks, agent_data, time_series

    # Initialize visualizer
    visualizer = MultiScaleVisualizer()

    # Create sample data
    sample_networks, sample_agent_data, sample_time_series = create_sample_multiscale_data()

    # Create multi-scale overview
    overview_path = visualizer.create_multiscale_overview(
        networks=sample_networks,
        agent_data=sample_agent_data,
        time_series_data=sample_time_series,
        title="Sample Multi-Scale Analysis",
        save_filename="sample_multiscale_overview"
    )

    print(f"Multi-scale overview created: {overview_path}")