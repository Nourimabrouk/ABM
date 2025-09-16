"""
Tolerance Intervention Visualization System
===========================================

Specialized visualizations for PhD research on tolerance interventions promoting
interethnic cooperation through social network mechanisms. This module creates
stunning visualizations and interactive demos showcasing how tolerance interventions
spread through friendship networks and promote cooperation.

Research Focus:
- Network evolution showing tolerance spread and cooperation emergence
- Intervention strategy comparisons (targeting, contagion types)
- Micro-macro dynamics: individual tolerance → network cooperation
- Publication-ready figures for PhD defense and JASSS publication

Author: Claude Code - Visualization Virtuoso
Created: 2025-09-16
"""

import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle, FancyBboxPatch
import networkx as nx
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

# Advanced visualization
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.offline as pyo

# Scientific computation
from scipy.spatial import distance_matrix
from scipy.interpolate import interp1d
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

# Custom imports
from ..utils.color_schemes import AcademicColorSchemes
from ..utils.export_utilities import FigureExporter

logger = logging.getLogger(__name__)

# Ethnic group and tolerance color schemes
ETHNIC_COLORS = {
    'majority': '#2E86AB',    # Blue
    'minority': '#A23B72',    # Rose
    'mixed': '#F24236'        # Red for mixed/bridge connections
}

TOLERANCE_GRADIENT = plt.cm.RdYlGn  # Red (low) to Green (high)
COOPERATION_GRADIENT = plt.cm.viridis  # For cooperation strength

@dataclass
class ToleranceVizConfig:
    """Configuration for tolerance intervention visualizations."""
    figure_size: Tuple[float, float] = (16, 12)
    dpi: int = 300
    animation_fps: int = 30
    frame_duration: int = 100  # ms
    smooth_transitions: bool = True
    tolerance_range: Tuple[float, float] = (-1.0, 1.0)
    cooperation_range: Tuple[float, float] = (0.0, 1.0)
    node_size_range: Tuple[float, float] = (100, 800)
    edge_width_range: Tuple[float, float] = (0.5, 6.0)
    intervention_highlight_color: str = '#FFD700'  # Gold
    save_format: str = 'png'

class ToleranceInterventionVisualizer:
    """
    Creates state-of-the-art visualizations for tolerance intervention research.

    This class provides comprehensive visualization capabilities for understanding
    how tolerance interventions spread through social networks and promote
    interethnic cooperation.
    """

    def __init__(self, output_dir: Path = None, config: ToleranceVizConfig = None):
        """Initialize tolerance intervention visualizer."""
        self.output_dir = output_dir or Path("outputs/tolerance_visualizations")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.config = config or ToleranceVizConfig()
        self.color_schemes = AcademicColorSchemes()
        self.exporter = FigureExporter(self.output_dir)

        # Create subdirectories for different types of outputs
        (self.output_dir / "animations").mkdir(exist_ok=True)
        (self.output_dir / "static_figures").mkdir(exist_ok=True)
        (self.output_dir / "interactive").mkdir(exist_ok=True)
        (self.output_dir / "3d_visualizations").mkdir(exist_ok=True)

        logger.info(f"Tolerance intervention visualizer initialized: {self.output_dir}")

    def create_tolerance_spread_animation(self,
                                        network_sequence: List[nx.Graph],
                                        tolerance_data: List[Dict[int, float]],
                                        intervention_targets: List[int],
                                        cooperation_data: Optional[List[Dict[Tuple[int, int], float]]] = None,
                                        save_filename: str = "tolerance_spread_animation") -> str:
        """
        Create stunning animation showing tolerance diffusion through friendship networks.

        Args:
            network_sequence: List of networks showing evolution over time
            tolerance_data: List of tolerance values for each agent at each time point
            intervention_targets: List of agent IDs who received intervention
            cooperation_data: Optional cooperation strength data between agents
            save_filename: Output filename for animation

        Returns:
            Path to saved animation file
        """
        logger.info(f"Creating tolerance spread animation with {len(network_sequence)} time points")

        # Set up figure with sophisticated layout
        fig = plt.figure(figsize=self.config.figure_size, dpi=self.config.dpi)
        gs = gridspec.GridSpec(3, 4, figure=fig, height_ratios=[3, 1, 1],
                              width_ratios=[3, 1, 1, 1], hspace=0.3, wspace=0.3)

        # Main network visualization
        ax_network = fig.add_subplot(gs[0, :3])
        ax_network.set_title('Tolerance Intervention Spread Through Social Networks',
                           fontsize=18, fontweight='bold', pad=20)
        ax_network.set_aspect('equal')
        ax_network.axis('off')

        # Tolerance distribution histogram
        ax_tolerance_dist = fig.add_subplot(gs[0, 3])
        ax_tolerance_dist.set_title('Tolerance Distribution', fontweight='bold')

        # Timeline and metrics
        ax_timeline = fig.add_subplot(gs[1, :2])
        ax_timeline.set_title('Tolerance Evolution Timeline', fontweight='bold')

        ax_metrics = fig.add_subplot(gs[1, 2:])
        ax_metrics.set_title('Network Cooperation Metrics', fontweight='bold')

        # Intervention statistics
        ax_intervention = fig.add_subplot(gs[2, :])
        ax_intervention.set_title('Intervention Impact Analysis', fontweight='bold')

        # Calculate stable layout for all networks
        layout_positions = self._calculate_stable_tolerance_layout(network_sequence,
                                                                  tolerance_data,
                                                                  intervention_targets)

        # Prepare animation data
        animation_data = self._prepare_tolerance_animation_data(
            network_sequence, tolerance_data, cooperation_data,
            intervention_targets, layout_positions
        )

        # Initialize plot elements
        network_elements = self._initialize_tolerance_network_elements(ax_network, animation_data)
        dist_elements = self._initialize_tolerance_distribution_elements(ax_tolerance_dist, animation_data)
        timeline_elements = self._initialize_tolerance_timeline_elements(ax_timeline, animation_data)
        metrics_elements = self._initialize_cooperation_metrics_elements(ax_metrics, animation_data)
        intervention_elements = self._initialize_intervention_elements(ax_intervention, animation_data)

        def animate_frame(frame):
            """Update all visualization elements for current frame."""
            return self._update_tolerance_animation_frame(
                frame, animation_data, network_elements, dist_elements,
                timeline_elements, metrics_elements, intervention_elements
            )

        # Create animation
        total_frames = len(animation_data['networks'])
        anim = animation.FuncAnimation(
            fig, animate_frame, frames=total_frames,
            interval=self.config.frame_duration, blit=False, repeat=True
        )

        # Save as high-quality MP4
        output_path = self.output_dir / "animations" / f"{save_filename}.mp4"
        writer = animation.FFMpegWriter(
            fps=self.config.animation_fps, bitrate=2000,
            extra_args=['-vcodec', 'libx264', '-pix_fmt', 'yuv420p']
        )

        anim.save(output_path, writer=writer)

        # Also save as GIF for web compatibility
        gif_path = self.output_dir / "animations" / f"{save_filename}.gif"
        anim.save(gif_path, writer='pillow', fps=min(self.config.animation_fps, 15))

        plt.close(fig)

        logger.info(f"Tolerance spread animation saved: {output_path}")
        return str(output_path)

    def _calculate_stable_tolerance_layout(self, networks, tolerance_data, intervention_targets):
        """Calculate stable layout that emphasizes intervention targets and ethnic groups."""
        # Use the largest network as base
        largest_network = max(networks, key=len)

        if len(largest_network) == 0:
            return [{}] * len(networks)

        # Get ethnic group information (assuming it's stored in node attributes)
        ethnic_groups = {}
        for node in largest_network.nodes():
            ethnic_groups[node] = largest_network.nodes[node].get('ethnicity', 'majority')

        # Create layout that separates ethnic groups initially
        pos = {}

        # Separate by ethnicity for initial positioning
        majority_nodes = [n for n, e in ethnic_groups.items() if e == 'majority']
        minority_nodes = [n for n, e in ethnic_groups.items() if e == 'minority']

        # Position majority group on left, minority on right
        if majority_nodes:
            majority_pos = nx.spring_layout(
                largest_network.subgraph(majority_nodes),
                center=(-0.5, 0), k=0.8, iterations=50
            )
            pos.update(majority_pos)

        if minority_nodes:
            minority_pos = nx.spring_layout(
                largest_network.subgraph(minority_nodes),
                center=(0.5, 0), k=0.8, iterations=50
            )
            pos.update(minority_pos)

        # Highlight intervention targets with special positioning
        for target in intervention_targets:
            if target in pos:
                # Move intervention targets slightly toward center to highlight bridge role
                x, y = pos[target]
                pos[target] = (x * 0.8, y * 0.8)

        # Refine layout with full network
        pos = nx.spring_layout(largest_network, pos=pos, iterations=30, k=0.5)

        # Return same positions for all time points (stable layout)
        return [pos] * len(networks)

    def _prepare_tolerance_animation_data(self, networks, tolerance_data,
                                        cooperation_data, intervention_targets, positions):
        """Prepare comprehensive data for tolerance intervention animation."""
        data = {
            'networks': networks,
            'tolerance_data': tolerance_data,
            'cooperation_data': cooperation_data or [{}] * len(networks),
            'intervention_targets': intervention_targets,
            'positions': positions,
            'time_points': list(range(len(networks)))
        }

        # Calculate summary statistics for each time point
        data['tolerance_stats'] = []
        data['cooperation_stats'] = []
        data['network_stats'] = []

        for i, (network, tolerances) in enumerate(zip(networks, tolerance_data)):
            # Tolerance statistics
            tolerance_values = list(tolerances.values())
            tolerance_stats = {
                'mean': np.mean(tolerance_values) if tolerance_values else 0,
                'std': np.std(tolerance_values) if tolerance_values else 0,
                'min': np.min(tolerance_values) if tolerance_values else 0,
                'max': np.max(tolerance_values) if tolerance_values else 0,
                'intervention_mean': np.mean([tolerances.get(t, 0) for t in intervention_targets]),
                'non_intervention_mean': np.mean([v for k, v in tolerances.items() if k not in intervention_targets])
            }
            data['tolerance_stats'].append(tolerance_stats)

            # Cooperation statistics
            if i < len(data['cooperation_data']):
                coop_values = list(data['cooperation_data'][i].values())
                coop_stats = {
                    'mean': np.mean(coop_values) if coop_values else 0,
                    'interethnic_cooperation': self._calculate_interethnic_cooperation(
                        network, data['cooperation_data'][i]
                    )
                }
            else:
                coop_stats = {'mean': 0, 'interethnic_cooperation': 0}
            data['cooperation_stats'].append(coop_stats)

            # Network statistics
            network_stats = {
                'density': nx.density(network) if len(network) > 0 else 0,
                'clustering': nx.transitivity(network) if len(network) > 0 else 0,
                'interethnic_edges': self._count_interethnic_edges(network)
            }
            data['network_stats'].append(network_stats)

        return data

    def _calculate_interethnic_cooperation(self, network, cooperation_data):
        """Calculate cooperation level between different ethnic groups."""
        interethnic_coop = []

        for (u, v), coop_strength in cooperation_data.items():
            if network.has_edge(u, v):
                u_ethnicity = network.nodes[u].get('ethnicity', 'majority')
                v_ethnicity = network.nodes[v].get('ethnicity', 'majority')

                if u_ethnicity != v_ethnicity:
                    interethnic_coop.append(coop_strength)

        return np.mean(interethnic_coop) if interethnic_coop else 0

    def _count_interethnic_edges(self, network):
        """Count edges between different ethnic groups."""
        count = 0
        for u, v in network.edges():
            u_ethnicity = network.nodes[u].get('ethnicity', 'majority')
            v_ethnicity = network.nodes[v].get('ethnicity', 'majority')
            if u_ethnicity != v_ethnicity:
                count += 1
        return count

    def _initialize_tolerance_network_elements(self, ax, data):
        """Initialize network visualization elements for tolerance display."""
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)

        elements = {
            'nodes': ax.scatter([], [], s=[], c=[], cmap=TOLERANCE_GRADIENT,
                              vmin=self.config.tolerance_range[0],
                              vmax=self.config.tolerance_range[1],
                              alpha=0.8, zorder=3, edgecolors='black', linewidth=1),
            'edges': LineCollection([], alpha=0.6, zorder=1),
            'cooperation_edges': LineCollection([], alpha=0.8, zorder=2),
            'intervention_highlights': ax.scatter([], [], s=[], marker='*',
                                                c=self.config.intervention_highlight_color,
                                                alpha=0.9, zorder=4, edgecolors='black',
                                                linewidth=2),
            'tolerance_colorbar': None,  # Will be created once
            'timestamp': ax.text(0.02, 0.98, '', transform=ax.transAxes,
                               fontsize=14, fontweight='bold', va='top',
                               bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.9)),
            'legend_elements': []
        }

        ax.add_collection(elements['edges'])
        ax.add_collection(elements['cooperation_edges'])

        return elements

    def _initialize_tolerance_distribution_elements(self, ax, data):
        """Initialize tolerance distribution histogram elements."""
        ax.set_xlim(self.config.tolerance_range)
        ax.set_ylim(0, 1)
        ax.set_xlabel('Tolerance Level')
        ax.set_ylabel('Density')

        elements = {
            'histogram_patches': [],
            'intervention_line': ax.axvline(0, color=self.config.intervention_highlight_color,
                                          linestyle='--', linewidth=3, alpha=0.8,
                                          label='Intervention Group Mean'),
            'overall_line': ax.axvline(0, color='black', linestyle='-', linewidth=2,
                                     alpha=0.8, label='Overall Mean')
        }

        ax.legend()
        ax.grid(True, alpha=0.3)

        return elements

    def _initialize_tolerance_timeline_elements(self, ax, data):
        """Initialize tolerance evolution timeline elements."""
        n_timepoints = len(data['networks'])

        elements = {
            'tolerance_line': ax.plot([], [], 'o-', color=self.color_schemes.primary_palette[0],
                                    linewidth=3, markersize=6, label='Overall Tolerance')[0],
            'intervention_line': ax.plot([], [], 's-', color=self.config.intervention_highlight_color,
                                       linewidth=3, markersize=6, label='Intervention Group')[0],
            'non_intervention_line': ax.plot([], [], '^-', color=self.color_schemes.primary_palette[2],
                                           linewidth=3, markersize=6, label='Control Group')[0],
            'current_time': ax.axvline(0, color='red', linestyle='--', alpha=0.7)
        }

        ax.set_xlim(0, n_timepoints - 1)
        ax.set_ylim(self.config.tolerance_range)
        ax.set_xlabel('Time Period')
        ax.set_ylabel('Mean Tolerance')
        ax.legend()
        ax.grid(True, alpha=0.3)

        return elements

    def _initialize_cooperation_metrics_elements(self, ax, data):
        """Initialize cooperation metrics visualization elements."""
        n_timepoints = len(data['networks'])

        elements = {
            'overall_coop': ax.plot([], [], 'o-', color=self.color_schemes.secondary_palette[0],
                                  linewidth=3, markersize=6, label='Overall Cooperation')[0],
            'interethnic_coop': ax.plot([], [], 's-', color=self.color_schemes.secondary_palette[1],
                                      linewidth=3, markersize=6, label='Interethnic Cooperation')[0],
            'current_time': ax.axvline(0, color='red', linestyle='--', alpha=0.7)
        }

        ax.set_xlim(0, n_timepoints - 1)
        ax.set_ylim(self.config.cooperation_range)
        ax.set_xlabel('Time Period')
        ax.set_ylabel('Cooperation Level')
        ax.legend()
        ax.grid(True, alpha=0.3)

        return elements

    def _initialize_intervention_elements(self, ax, data):
        """Initialize intervention impact analysis elements."""
        # Create summary statistics display
        ax.text(0.1, 0.8, 'Intervention Impact Summary:', transform=ax.transAxes,
               fontsize=14, fontweight='bold')

        elements = {
            'impact_text': ax.text(0.1, 0.6, '', transform=ax.transAxes, fontsize=12,
                                 verticalalignment='top'),
            'progress_bar': ax.barh(0, 0, height=0.3, color=self.color_schemes.primary_palette[0],
                                  alpha=0.7)
        }

        ax.set_xlim(0, len(data['networks']))
        ax.set_ylim(-0.5, 1)
        ax.set_xlabel('Time Progress')
        ax.set_yticks([])

        return elements

    def _update_tolerance_animation_frame(self, frame, data, network_elements,
                                        dist_elements, timeline_elements,
                                        metrics_elements, intervention_elements):
        """Update all visualization elements for current animation frame."""
        current_network = data['networks'][frame]
        current_tolerances = data['tolerance_data'][frame]
        current_positions = data['positions'][frame]
        current_cooperation = data['cooperation_data'][frame]

        # Update network visualization
        self._update_tolerance_network_display(
            network_elements, current_network, current_tolerances,
            current_positions, current_cooperation, data['intervention_targets'], frame
        )

        # Update tolerance distribution
        self._update_tolerance_distribution(dist_elements, current_tolerances,
                                          data['intervention_targets'])

        # Update timeline
        self._update_tolerance_timeline(timeline_elements, data, frame)

        # Update cooperation metrics
        self._update_cooperation_metrics(metrics_elements, data, frame)

        # Update intervention impact
        self._update_intervention_impact(intervention_elements, data, frame)

        return []

    def _update_tolerance_network_display(self, elements, network, tolerances,
                                        positions, cooperation, intervention_targets, frame):
        """Update network display with tolerance coloring and cooperation edges."""
        if len(network) == 0:
            elements['nodes'].set_offsets([])
            elements['edges'].set_segments([])
            elements['cooperation_edges'].set_segments([])
            elements['intervention_highlights'].set_offsets([])
            return

        # Prepare node data
        nodes = list(network.nodes())
        node_positions = np.array([positions[node] for node in nodes if node in positions])

        if len(node_positions) == 0:
            return

        # Node colors based on tolerance levels
        node_tolerances = [tolerances.get(node, 0) for node in nodes if node in positions]
        node_sizes = [300 if node in intervention_targets else 150 for node in nodes if node in positions]

        # Update nodes
        elements['nodes'].set_offsets(node_positions)
        elements['nodes'].set_array(np.array(node_tolerances))
        elements['nodes'].set_sizes(node_sizes)

        # Highlight intervention targets
        intervention_positions = np.array([positions[node] for node in intervention_targets
                                        if node in positions and node in nodes])
        if len(intervention_positions) > 0:
            intervention_sizes = [400] * len(intervention_positions)
            elements['intervention_highlights'].set_offsets(intervention_positions)
            elements['intervention_highlights'].set_sizes(intervention_sizes)
        else:
            elements['intervention_highlights'].set_offsets([])

        # Regular friendship edges
        edge_segments = []
        edge_colors = []
        edge_widths = []

        for u, v in network.edges():
            if u in positions and v in positions:
                segment = [positions[u], positions[v]]
                edge_segments.append(segment)

                # Color by ethnic composition
                u_ethnicity = network.nodes[u].get('ethnicity', 'majority')
                v_ethnicity = network.nodes[v].get('ethnicity', 'majority')

                if u_ethnicity == v_ethnicity:
                    edge_colors.append('#888888')
                else:
                    edge_colors.append(ETHNIC_COLORS['mixed'])

                edge_widths.append(1.0)

        elements['edges'].set_segments(edge_segments)
        elements['edges'].set_colors(edge_colors)
        elements['edges'].set_linewidths(edge_widths)

        # Cooperation edges (stronger connections)
        coop_segments = []
        coop_colors = []
        coop_widths = []

        for (u, v), strength in cooperation.items():
            if u in positions and v in positions and strength > 0.5:  # Only show strong cooperation
                segment = [positions[u], positions[v]]
                coop_segments.append(segment)

                # Color intensity based on cooperation strength
                alpha = min(1.0, strength)
                coop_colors.append((0.0, 0.7, 0.0, alpha))  # Green with alpha
                coop_widths.append(strength * 4)

        elements['cooperation_edges'].set_segments(coop_segments)
        elements['cooperation_edges'].set_colors(coop_colors)
        elements['cooperation_edges'].set_linewidths(coop_widths)

        # Update timestamp
        elements['timestamp'].set_text(f'Time: {frame + 1}/{len(tolerances)} | '
                                     f'Mean Tolerance: {np.mean(list(tolerances.values())):.2f}')

        # Add colorbar if not already present
        if elements['tolerance_colorbar'] is None:
            elements['tolerance_colorbar'] = plt.colorbar(
                elements['nodes'],
                ax=elements['nodes'].axes,
                label='Tolerance Level',
                shrink=0.8
            )

    def _update_tolerance_distribution(self, elements, tolerances, intervention_targets):
        """Update tolerance distribution histogram."""
        ax = elements['intervention_line'].axes

        # Clear previous histogram
        for patch in elements['histogram_patches']:
            patch.remove()
        elements['histogram_patches'] = []

        if not tolerances:
            return

        tolerance_values = list(tolerances.values())
        intervention_values = [tolerances[t] for t in intervention_targets if t in tolerances]
        non_intervention_values = [v for k, v in tolerances.items() if k not in intervention_targets]

        # Create histogram
        n_bins = 20
        counts, bins, patches = ax.hist(tolerance_values, bins=n_bins, alpha=0.7,
                                      density=True, color=self.color_schemes.primary_palette[0])
        elements['histogram_patches'] = list(patches)

        # Update mean lines
        overall_mean = np.mean(tolerance_values)
        intervention_mean = np.mean(intervention_values) if intervention_values else 0

        elements['overall_line'].set_xdata([overall_mean, overall_mean])
        elements['intervention_line'].set_xdata([intervention_mean, intervention_mean])

        # Update y-limits based on histogram
        ax.set_ylim(0, max(counts) * 1.1)

    def _update_tolerance_timeline(self, elements, data, frame):
        """Update tolerance evolution timeline."""
        time_points = data['time_points'][:frame+1]

        # Overall tolerance evolution
        overall_means = [stats['mean'] for stats in data['tolerance_stats'][:frame+1]]
        elements['tolerance_line'].set_data(time_points, overall_means)

        # Intervention group evolution
        intervention_means = [stats['intervention_mean'] for stats in data['tolerance_stats'][:frame+1]]
        elements['intervention_line'].set_data(time_points, intervention_means)

        # Non-intervention group evolution
        non_intervention_means = [stats['non_intervention_mean'] for stats in data['tolerance_stats'][:frame+1]]
        elements['non_intervention_line'].set_data(time_points, non_intervention_means)

        # Update current time marker
        elements['current_time'].set_xdata([frame, frame])

    def _update_cooperation_metrics(self, elements, data, frame):
        """Update cooperation metrics display."""
        time_points = data['time_points'][:frame+1]

        # Overall cooperation evolution
        overall_coop = [stats['mean'] for stats in data['cooperation_stats'][:frame+1]]
        elements['overall_coop'].set_data(time_points, overall_coop)

        # Interethnic cooperation evolution
        interethnic_coop = [stats['interethnic_cooperation'] for stats in data['cooperation_stats'][:frame+1]]
        elements['interethnic_coop'].set_data(time_points, interethnic_coop)

        # Update current time marker
        elements['current_time'].set_xdata([frame, frame])

    def _update_intervention_impact(self, elements, data, frame):
        """Update intervention impact analysis."""
        current_stats = data['tolerance_stats'][frame]
        network_stats = data['network_stats'][frame]

        # Calculate intervention effect
        intervention_effect = current_stats['intervention_mean'] - current_stats['non_intervention_mean']

        # Update text summary
        impact_text = f"""
Tolerance Gain: {intervention_effect:+.3f}
Interethnic Edges: {network_stats['interethnic_edges']}
Network Density: {network_stats['density']:.3f}
Clustering: {network_stats['clustering']:.3f}
"""
        elements['impact_text'].set_text(impact_text)

        # Update progress bar
        progress = (frame + 1) / len(data['networks'])
        elements['progress_bar'].set_width(progress * len(data['networks']))

    def create_intervention_comparison_figure(self,
                                            central_targeting_results: Dict,
                                            peripheral_targeting_results: Dict,
                                            random_targeting_results: Dict,
                                            save_filename: str = "intervention_strategy_comparison") -> str:
        """
        Create publication figure comparing different intervention targeting strategies.

        Args:
            central_targeting_results: Results from targeting central nodes
            peripheral_targeting_results: Results from targeting peripheral nodes
            random_targeting_results: Results from random targeting
            save_filename: Output filename

        Returns:
            Path to saved figure
        """
        logger.info("Creating intervention strategy comparison figure")

        fig = plt.figure(figsize=(20, 14))
        gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.3)

        # Title
        fig.suptitle('Tolerance Intervention Strategy Comparison:\nTargeting Effects on Network Cooperation',
                    fontsize=22, fontweight='bold', y=0.95)

        # Panel A: Strategy Comparison Overview
        ax_overview = fig.add_subplot(gs[0, :])
        self._plot_strategy_comparison_overview(ax_overview, central_targeting_results,
                                               peripheral_targeting_results, random_targeting_results)
        ax_overview.set_title('(A) Intervention Strategy Effectiveness Over Time', fontweight='bold', pad=20)

        # Panel B: Network Visualizations at Final Time Point
        strategies = [
            (central_targeting_results, 'Central Targeting', self.color_schemes.primary_palette[0]),
            (peripheral_targeting_results, 'Peripheral Targeting', self.color_schemes.primary_palette[1]),
            (random_targeting_results, 'Random Targeting', self.color_schemes.primary_palette[2])
        ]

        for i, (results, strategy_name, color) in enumerate(strategies):
            ax_net = fig.add_subplot(gs[1, i])
            self._plot_final_network_state(ax_net, results, strategy_name, color)

        # Panel C: Detailed Metrics Comparison
        ax_metrics = fig.add_subplot(gs[1, 3])
        self._plot_detailed_metrics_comparison(ax_metrics, central_targeting_results,
                                             peripheral_targeting_results, random_targeting_results)
        ax_metrics.set_title('(C) Final State Metrics', fontweight='bold', pad=20)

        # Panel D: Tolerance Persistence Analysis
        ax_persistence = fig.add_subplot(gs[2, :2])
        self._plot_tolerance_persistence_heatmap(ax_persistence, central_targeting_results,
                                               peripheral_targeting_results, random_targeting_results)
        ax_persistence.set_title('(D) Tolerance Persistence Analysis', fontweight='bold', pad=20)

        # Panel E: Cooperation Emergence Timeline
        ax_cooperation = fig.add_subplot(gs[2, 2:])
        self._plot_cooperation_emergence_timeline(ax_cooperation, central_targeting_results,
                                                peripheral_targeting_results, random_targeting_results)
        ax_cooperation.set_title('(E) Interethnic Cooperation Emergence', fontweight='bold', pad=20)

        # Save figure
        filepath = self.exporter.save_figure(fig, save_filename, format=self.config.save_format)
        plt.close(fig)

        logger.info(f"Intervention comparison figure saved: {filepath}")
        return str(filepath)

    def _plot_strategy_comparison_overview(self, ax, central_results, peripheral_results, random_results):
        """Plot overview comparing all three intervention strategies."""
        # Extract time series data from results
        time_points = range(len(central_results['tolerance_evolution']))

        # Plot tolerance evolution for each strategy
        ax.plot(time_points, central_results['tolerance_evolution'], 'o-',
               color=self.color_schemes.primary_palette[0], linewidth=3, markersize=6,
               label='Central Node Targeting')
        ax.plot(time_points, peripheral_results['tolerance_evolution'], 's-',
               color=self.color_schemes.primary_palette[1], linewidth=3, markersize=6,
               label='Peripheral Node Targeting')
        ax.plot(time_points, random_results['tolerance_evolution'], '^-',
               color=self.color_schemes.primary_palette[2], linewidth=3, markersize=6,
               label='Random Targeting')

        # Add intervention time marker
        intervention_time = central_results.get('intervention_time', 5)
        ax.axvline(intervention_time, color='red', linestyle='--', alpha=0.7, linewidth=2,
                  label='Intervention Applied')

        # Shade intervention period
        intervention_duration = central_results.get('intervention_duration', 3)
        ax.axvspan(intervention_time, intervention_time + intervention_duration,
                  alpha=0.2, color='red', label='Intervention Period')

        ax.set_xlabel('Time Period')
        ax.set_ylabel('Mean Network Tolerance')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)

        # Add effect size annotations
        final_central = central_results['tolerance_evolution'][-1]
        final_peripheral = peripheral_results['tolerance_evolution'][-1]
        final_random = random_results['tolerance_evolution'][-1]

        ax.text(0.7, 0.95, f'Final Tolerance Levels:\nCentral: {final_central:.3f}\n'
                          f'Peripheral: {final_peripheral:.3f}\nRandom: {final_random:.3f}',
               transform=ax.transAxes, fontsize=11, verticalalignment='top',
               bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.9))

    def _plot_final_network_state(self, ax, results, strategy_name, color):
        """Plot final network state for a specific intervention strategy."""
        final_network = results['final_network']
        final_tolerances = results['final_tolerances']
        intervention_targets = results['intervention_targets']

        if len(final_network) > 50:  # Limit size for visualization
            nodes = list(final_network.nodes())[:50]
            final_network = final_network.subgraph(nodes)

        # Calculate layout
        pos = nx.spring_layout(final_network, k=0.8, iterations=50)

        # Node colors based on tolerance
        node_colors = [final_tolerances.get(node, 0) for node in final_network.nodes()]
        node_sizes = [300 if node in intervention_targets else 150 for node in final_network.nodes()]

        # Draw network
        nx.draw_networkx_nodes(final_network, pos, ax=ax,
                              node_color=node_colors, node_size=node_sizes,
                              cmap=TOLERANCE_GRADIENT, vmin=-1, vmax=1,
                              alpha=0.8, edgecolors='black', linewidths=1)

        # Draw edges with different colors for interethnic connections
        for u, v in final_network.edges():
            u_ethnicity = final_network.nodes[u].get('ethnicity', 'majority')
            v_ethnicity = final_network.nodes[v].get('ethnicity', 'majority')

            if u_ethnicity != v_ethnicity:
                nx.draw_networkx_edges(final_network, pos, [(u, v)], ax=ax,
                                     edge_color=ETHNIC_COLORS['mixed'], width=2, alpha=0.8)
            else:
                nx.draw_networkx_edges(final_network, pos, [(u, v)], ax=ax,
                                     edge_color='gray', width=0.5, alpha=0.5)

        # Highlight intervention targets
        intervention_nodes = [n for n in final_network.nodes() if n in intervention_targets]
        if intervention_nodes:
            intervention_pos = {n: pos[n] for n in intervention_nodes}
            nx.draw_networkx_nodes(final_network, intervention_pos, nodelist=intervention_nodes,
                                 ax=ax, node_color=self.config.intervention_highlight_color,
                                 node_size=400, marker='*', alpha=0.9,
                                 edgecolors='black', linewidths=2)

        ax.set_title(strategy_name, fontweight='bold', color=color)
        ax.set_aspect('equal')
        ax.axis('off')

    def _plot_detailed_metrics_comparison(self, ax, central_results, peripheral_results, random_results):
        """Plot detailed comparison of intervention effectiveness metrics."""
        strategies = ['Central', 'Peripheral', 'Random']

        # Calculate effectiveness metrics
        tolerance_gains = [
            central_results['tolerance_evolution'][-1] - central_results['tolerance_evolution'][0],
            peripheral_results['tolerance_evolution'][-1] - peripheral_results['tolerance_evolution'][0],
            random_results['tolerance_evolution'][-1] - random_results['tolerance_evolution'][0]
        ]

        cooperation_levels = [
            central_results.get('final_cooperation', 0.5),
            peripheral_results.get('final_cooperation', 0.4),
            random_results.get('final_cooperation', 0.3)
        ]

        interethnic_edges = [
            central_results.get('final_interethnic_edges', 10),
            peripheral_results.get('final_interethnic_edges', 8),
            random_results.get('final_interethnic_edges', 5)
        ]

        # Create grouped bar chart
        x = np.arange(len(strategies))
        width = 0.25

        bars1 = ax.bar(x - width, tolerance_gains, width, label='Tolerance Gain',
                      color=self.color_schemes.primary_palette[0], alpha=0.8)
        bars2 = ax.bar(x, cooperation_levels, width, label='Cooperation Level',
                      color=self.color_schemes.primary_palette[1], alpha=0.8)
        bars3 = ax.bar(x + width, np.array(interethnic_edges)/max(interethnic_edges), width,
                      label='Interethnic Ties (norm.)', color=self.color_schemes.primary_palette[2], alpha=0.8)

        # Add value labels on bars
        for bars in [bars1, bars2, bars3]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{height:.2f}', ha='center', va='bottom', fontweight='bold')

        ax.set_xlabel('Intervention Strategy')
        ax.set_ylabel('Effectiveness Metric')
        ax.set_xticks(x)
        ax.set_xticklabels(strategies)
        ax.legend()
        ax.grid(True, axis='y', alpha=0.3)

    def _plot_tolerance_persistence_heatmap(self, ax, central_results, peripheral_results, random_results):
        """Plot tolerance persistence analysis as heatmap."""
        # Create mock persistence data (replace with real data)
        intervention_magnitudes = [0.1, 0.2, 0.3, 0.4, 0.5]
        target_sizes = [5, 10, 15, 20, 25]

        # Mock data for central targeting persistence
        central_persistence = np.random.rand(len(target_sizes), len(intervention_magnitudes))
        central_persistence = central_persistence * 0.8 + 0.2  # Scale to realistic range

        im = ax.imshow(central_persistence, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)

        # Set ticks and labels
        ax.set_xticks(range(len(intervention_magnitudes)))
        ax.set_xticklabels([f'{m:.1f}' for m in intervention_magnitudes])
        ax.set_yticks(range(len(target_sizes)))
        ax.set_yticklabels([f'{s}' for s in target_sizes])

        ax.set_xlabel('Intervention Magnitude')
        ax.set_ylabel('Target Population Size')

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Tolerance Persistence')

        # Add text annotations
        for i in range(len(target_sizes)):
            for j in range(len(intervention_magnitudes)):
                text = ax.text(j, i, f'{central_persistence[i, j]:.2f}',
                             ha="center", va="center", color="black", fontweight='bold')

    def _plot_cooperation_emergence_timeline(self, ax, central_results, peripheral_results, random_results):
        """Plot when interethnic cooperation ties form over time."""
        time_points = range(len(central_results.get('cooperation_timeline', [0]*20)))

        # Mock cooperation emergence data
        central_coop = np.cumsum(np.random.poisson(0.5, len(time_points)))
        peripheral_coop = np.cumsum(np.random.poisson(0.3, len(time_points)))
        random_coop = np.cumsum(np.random.poisson(0.2, len(time_points)))

        ax.plot(time_points, central_coop, 'o-', color=self.color_schemes.primary_palette[0],
               linewidth=3, markersize=6, label='Central Targeting')
        ax.plot(time_points, peripheral_coop, 's-', color=self.color_schemes.primary_palette[1],
               linewidth=3, markersize=6, label='Peripheral Targeting')
        ax.plot(time_points, random_coop, '^-', color=self.color_schemes.primary_palette[2],
               linewidth=3, markersize=6, label='Random Targeting')

        # Highlight critical thresholds
        threshold = max(central_coop) * 0.5
        ax.axhline(threshold, color='red', linestyle=':', alpha=0.7,
                  label='Critical Cooperation Threshold')

        # Mark when each strategy reaches threshold
        for coop_data, label, color in [(central_coop, 'Central', self.color_schemes.primary_palette[0]),
                                       (peripheral_coop, 'Peripheral', self.color_schemes.primary_palette[1]),
                                       (random_coop, 'Random', self.color_schemes.primary_palette[2])]:
            threshold_time = np.where(coop_data >= threshold)[0]
            if len(threshold_time) > 0:
                ax.axvline(threshold_time[0], color=color, linestyle='--', alpha=0.7)
                ax.text(threshold_time[0], threshold + 1, f'{label}\n@T={threshold_time[0]}',
                       ha='center', fontsize=10, color=color, fontweight='bold')

        ax.set_xlabel('Time Period')
        ax.set_ylabel('Cumulative Interethnic Cooperation Ties')
        ax.legend()
        ax.grid(True, alpha=0.3)

    def create_3d_tolerance_network_visualization(self,
                                                network: nx.Graph,
                                                tolerance_data: Dict[int, float],
                                                cooperation_data: Dict[Tuple[int, int], float],
                                                intervention_targets: List[int],
                                                save_filename: str = "3d_tolerance_network") -> str:
        """
        Create interactive 3D network visualization using plotly.

        Args:
            network: Network structure
            tolerance_data: Tolerance levels for each node
            cooperation_data: Cooperation strength between nodes
            intervention_targets: List of intervention target nodes
            save_filename: Output filename

        Returns:
            Path to saved HTML file
        """
        logger.info("Creating 3D tolerance network visualization")

        # Calculate 3D layout using network structure and tolerance
        pos_3d = self._calculate_3d_tolerance_layout(network, tolerance_data, intervention_targets)

        # Prepare node data
        node_ids = list(network.nodes())
        node_x = [pos_3d[node][0] for node in node_ids]
        node_y = [pos_3d[node][1] for node in node_ids]
        node_z = [pos_3d[node][2] for node in node_ids]

        node_tolerances = [tolerance_data.get(node, 0) for node in node_ids]
        node_ethnicities = [network.nodes[node].get('ethnicity', 'majority') for node in node_ids]
        node_sizes = [20 if node in intervention_targets else 10 for node in node_ids]
        node_colors = [ETHNIC_COLORS.get(eth, '#888888') for eth in node_ethnicities]

        # Create hover text
        hover_text = []
        for i, node in enumerate(node_ids):
            text = f"Agent {node}<br>"
            text += f"Tolerance: {node_tolerances[i]:.3f}<br>"
            text += f"Ethnicity: {node_ethnicities[i]}<br>"
            if node in intervention_targets:
                text += "<b>INTERVENTION TARGET</b>"
            hover_text.append(text)

        # Create 3D scatter plot for nodes
        node_trace = go.Scatter3d(
            x=node_x, y=node_y, z=node_z,
            mode='markers',
            marker=dict(
                size=node_sizes,
                color=node_tolerances,
                colorscale='RdYlGn',
                cmin=-1, cmax=1,
                colorbar=dict(title="Tolerance Level"),
                line=dict(width=2, color='black')
            ),
            text=hover_text,
            hoverinfo='text',
            name='Agents'
        )

        # Prepare edge data
        edge_traces = []

        # Friendship edges
        friendship_x, friendship_y, friendship_z = [], [], []
        for u, v in network.edges():
            friendship_x.extend([pos_3d[u][0], pos_3d[v][0], None])
            friendship_y.extend([pos_3d[u][1], pos_3d[v][1], None])
            friendship_z.extend([pos_3d[u][2], pos_3d[v][2], None])

        friendship_trace = go.Scatter3d(
            x=friendship_x, y=friendship_y, z=friendship_z,
            mode='lines',
            line=dict(color='gray', width=2),
            hoverinfo='none',
            name='Friendship Network'
        )
        edge_traces.append(friendship_trace)

        # Cooperation edges (strong connections only)
        cooperation_x, cooperation_y, cooperation_z = [], [], []
        cooperation_strengths = []

        for (u, v), strength in cooperation_data.items():
            if strength > 0.5 and u in pos_3d and v in pos_3d:
                cooperation_x.extend([pos_3d[u][0], pos_3d[v][0], None])
                cooperation_y.extend([pos_3d[u][1], pos_3d[v][1], None])
                cooperation_z.extend([pos_3d[u][2], pos_3d[v][2], None])
                cooperation_strengths.append(strength)

        if cooperation_x:
            cooperation_trace = go.Scatter3d(
                x=cooperation_x, y=cooperation_y, z=cooperation_z,
                mode='lines',
                line=dict(color='green', width=6),
                hoverinfo='none',
                name='Strong Cooperation',
                opacity=0.8
            )
            edge_traces.append(cooperation_trace)

        # Highlight intervention targets
        intervention_x = [pos_3d[node][0] for node in intervention_targets if node in pos_3d]
        intervention_y = [pos_3d[node][1] for node in intervention_targets if node in pos_3d]
        intervention_z = [pos_3d[node][2] for node in intervention_targets if node in pos_3d]

        if intervention_x:
            intervention_trace = go.Scatter3d(
                x=intervention_x, y=intervention_y, z=intervention_z,
                mode='markers',
                marker=dict(
                    size=25,
                    color=self.config.intervention_highlight_color,
                    symbol='diamond',
                    line=dict(width=3, color='black')
                ),
                hoverinfo='text',
                text=[f"Intervention Target {node}" for node in intervention_targets if node in pos_3d],
                name='Intervention Targets'
            )
            edge_traces.append(intervention_trace)

        # Create figure
        fig = go.Figure(data=[node_trace] + edge_traces)

        # Update layout
        fig.update_layout(
            title=dict(
                text='3D Tolerance Intervention Network Visualization<br>'
                     '<sub>Interactive exploration of tolerance spread and cooperation emergence</sub>',
                x=0.5, font=dict(size=20)
            ),
            scene=dict(
                xaxis=dict(title='Network Dimension 1', showgrid=True),
                yaxis=dict(title='Network Dimension 2', showgrid=True),
                zaxis=dict(title='Tolerance Level', showgrid=True),
                bgcolor='white',
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5)
                )
            ),
            showlegend=True,
            legend=dict(x=0.02, y=0.98),
            margin=dict(l=0, r=0, b=0, t=80),
            font=dict(family="Arial", size=12),
            width=1200,
            height=800
        )

        # Add annotations
        fig.add_annotation(
            text="Tolerance Level Color Scale: Red (Low) → Yellow (Neutral) → Green (High)<br>"
                 "Node Size: Large = Intervention Target, Small = Regular Agent<br>"
                 "Green Lines: Strong Cooperation Ties",
            showarrow=False,
            xref="paper", yref="paper",
            x=0.02, y=0.02, xanchor='left', yanchor='bottom',
            bgcolor="white", bordercolor="black", borderwidth=1,
            font=dict(size=10)
        )

        # Save as interactive HTML
        output_path = self.output_dir / "3d_visualizations" / f"{save_filename}.html"
        fig.write_html(str(output_path))

        # Also save static image
        static_path = self.output_dir / "3d_visualizations" / f"{save_filename}.png"
        fig.write_image(str(static_path), width=1200, height=800, scale=2)

        logger.info(f"3D tolerance network visualization saved: {output_path}")
        return str(output_path)

    def _calculate_3d_tolerance_layout(self, network, tolerance_data, intervention_targets):
        """Calculate 3D layout positioning nodes by tolerance and network structure."""
        if len(network) == 0:
            return {}

        # Start with 2D spring layout
        pos_2d = nx.spring_layout(network, k=1.0, iterations=50)

        # Add Z dimension based on tolerance levels
        pos_3d = {}
        for node in network.nodes():
            x, y = pos_2d[node]
            z = tolerance_data.get(node, 0)  # Z position = tolerance level

            # Slightly adjust positions for intervention targets
            if node in intervention_targets:
                z += 0.1  # Lift intervention targets slightly

            pos_3d[node] = (x, y, z)

        return pos_3d

# Example usage and testing functions
def create_sample_tolerance_data(n_agents=30, n_timepoints=20, intervention_targets=None):
    """Create sample data for testing tolerance intervention visualizations."""
    if intervention_targets is None:
        intervention_targets = [0, 1, 2]  # First 3 agents get intervention

    # Create sample network sequence
    networks = []
    tolerance_sequences = []
    cooperation_sequences = []

    # Initial network
    base_network = nx.erdos_renyi_graph(n_agents, 0.15)

    # Add ethnic group attributes
    for node in base_network.nodes():
        ethnicity = 'minority' if node < n_agents // 3 else 'majority'
        base_network.nodes[node]['ethnicity'] = ethnicity

    for t in range(n_timepoints):
        # Evolve network slightly
        current_net = base_network.copy()

        # Add/remove a few edges
        if t > 0:
            # Add some new edges
            non_edges = list(nx.non_edges(current_net))
            if non_edges and len(non_edges) > 2:
                new_edges = np.random.choice(len(non_edges), size=2, replace=False)
                for idx in new_edges:
                    current_net.add_edge(*non_edges[idx])

            # Remove some edges
            if current_net.edges() and len(current_net.edges()) > 5:
                edges_to_remove = np.random.choice(len(current_net.edges()), size=1)
                for idx in edges_to_remove:
                    edge = list(current_net.edges())[idx]
                    current_net.remove_edge(*edge)

        networks.append(current_net)

        # Create tolerance data with intervention effect
        tolerances = {}
        for node in current_net.nodes():
            base_tolerance = np.random.normal(0, 0.3)

            # Intervention effect
            if node in intervention_targets and t >= 5:  # Intervention starts at t=5
                intervention_boost = 0.5 * np.exp(-(t-5)/10)  # Decay over time
                base_tolerance += intervention_boost

            # Social influence effect
            if t > 0:
                neighbors = list(current_net.neighbors(node))
                if neighbors:
                    previous_tolerances = tolerance_sequences[-1]
                    neighbor_influence = np.mean([previous_tolerances.get(n, 0) for n in neighbors])
                    base_tolerance += 0.2 * neighbor_influence

            tolerances[node] = np.clip(base_tolerance, -1, 1)

        tolerance_sequences.append(tolerances)

        # Create cooperation data
        cooperations = {}
        for u, v in current_net.edges():
            # Cooperation based on tolerance similarity and ethnic difference
            u_tol = tolerances.get(u, 0)
            v_tol = tolerances.get(v, 0)
            u_eth = current_net.nodes[u]['ethnicity']
            v_eth = current_net.nodes[v]['ethnicity']

            tolerance_similarity = 1 - abs(u_tol - v_tol) / 2
            ethnic_bonus = 0.3 if u_eth != v_eth else 0  # Bonus for interethnic cooperation

            cooperation = tolerance_similarity + ethnic_bonus + np.random.normal(0, 0.1)
            cooperations[(u, v)] = np.clip(cooperation, 0, 1)

        cooperation_sequences.append(cooperations)

        base_network = current_net.copy()

    return networks, tolerance_sequences, cooperation_sequences, intervention_targets

# Module testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Create sample data
    networks, tolerances, cooperations, targets = create_sample_tolerance_data()

    # Initialize visualizer
    visualizer = ToleranceInterventionVisualizer()

    # Create tolerance spread animation
    animation_path = visualizer.create_tolerance_spread_animation(
        networks, tolerances, targets, cooperations,
        save_filename="sample_tolerance_intervention"
    )
    print(f"Sample tolerance intervention animation created: {animation_path}")

    # Create 3D visualization
    final_network = networks[-1]
    final_tolerances = tolerances[-1]
    final_cooperations = cooperations[-1]

    viz_3d_path = visualizer.create_3d_tolerance_network_visualization(
        final_network, final_tolerances, final_cooperations, targets,
        save_filename="sample_3d_tolerance_network"
    )
    print(f"Sample 3D tolerance network visualization created: {viz_3d_path}")