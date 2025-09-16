"""
Network Evolution Animation System

This module creates sophisticated animations showing network evolution over time,
with smooth temporal transitions, agent behavior dynamics, and multi-scale visualizations.

Author: Delta Agent - State-of-the-Art Visualization Specialist
Created: 2025-09-15
"""

import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle, FancyBboxPatch
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection
import networkx as nx
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

# Scientific computation
from scipy.spatial import distance_matrix
from scipy.interpolate import interp1d
from sklearn.decomposition import PCA

# Custom imports
from ..utils.color_schemes import AcademicColorSchemes
from ..utils.export_utilities import FigureExporter

logger = logging.getLogger(__name__)

@dataclass
class AnimationConfiguration:
    """Configuration for network evolution animations."""
    fps: int = 30
    duration_seconds: float = 10.0
    figure_size: Tuple[float, float] = (12, 10)
    dpi: int = 100
    bitrate: int = 1800
    interpolation_frames: int = 10
    smooth_transitions: bool = True
    show_metrics: bool = True
    show_timestamp: bool = True
    node_size_range: Tuple[float, float] = (50, 500)
    edge_width_range: Tuple[float, float] = (0.5, 4.0)
    layout_stability: float = 0.8

class NetworkEvolutionAnimator:
    """
    Creates sophisticated animations of network evolution over time.

    This class handles temporal network data and creates smooth animations
    showing network structure changes, agent behavior evolution, and
    statistical metrics over time.
    """

    def __init__(self,
                 output_dir: Path = None,
                 config: AnimationConfiguration = None):
        """
        Initialize network evolution animator.

        Args:
            output_dir: Directory to save animations
            config: Animation configuration
        """
        self.output_dir = output_dir or Path("outputs/animations")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.config = config or AnimationConfiguration()
        self.color_schemes = AcademicColorSchemes()
        self.exporter = FigureExporter(self.output_dir)

        logger.info(f"Network evolution animator initialized, output: {self.output_dir}")

    def animate_network_evolution(self,
                                networks: List[nx.Graph],
                                agent_attributes: Optional[List[Dict]] = None,
                                metrics_data: Optional[pd.DataFrame] = None,
                                title: str = "Network Evolution Over Time",
                                save_filename: str = "network_evolution") -> str:
        """
        Create animation of network evolution over time.

        Args:
            networks: List of NetworkX graphs in temporal order
            agent_attributes: List of agent attribute dictionaries for each time point
            metrics_data: DataFrame with network metrics over time
            title: Animation title
            save_filename: Output filename

        Returns:
            Path to saved animation file
        """
        logger.info(f"Creating network evolution animation with {len(networks)} time points")

        # Prepare data for animation
        animation_data = self._prepare_animation_data(networks, agent_attributes, metrics_data)

        # Create figure and subplots
        fig = plt.figure(figsize=self.config.figure_size, dpi=self.config.dpi)
        gs = fig.add_gridspec(2, 3, height_ratios=[3, 1], width_ratios=[2, 2, 1],
                             hspace=0.3, wspace=0.3)

        # Main network plot
        ax_network = fig.add_subplot(gs[0, :2])
        ax_network.set_aspect('equal')
        ax_network.set_title(title, fontsize=16, fontweight='bold')

        # Metrics plot
        ax_metrics = fig.add_subplot(gs[0, 2])
        ax_metrics.set_title('Network Metrics', fontweight='bold')

        # Timeline and controls
        ax_timeline = fig.add_subplot(gs[1, :])
        ax_timeline.set_title('Timeline', fontweight='bold')

        # Initialize animation elements
        network_elements = self._initialize_network_elements(ax_network, animation_data)
        metrics_elements = self._initialize_metrics_elements(ax_metrics, animation_data)
        timeline_elements = self._initialize_timeline_elements(ax_timeline, animation_data)

        # Create animation function
        def animate(frame):
            return self._update_animation_frame(
                frame, animation_data, network_elements, metrics_elements, timeline_elements
            )

        # Calculate total frames
        total_frames = len(animation_data['interpolated_positions'])

        # Create animation
        anim = animation.FuncAnimation(
            fig, animate, frames=total_frames,
            interval=1000/self.config.fps, blit=False, repeat=True
        )

        # Save animation
        output_path = self.output_dir / f"{save_filename}.mp4"
        writer = animation.FFMpegWriter(
            fps=self.config.fps,
            bitrate=self.config.bitrate,
            extra_args=['-vcodec', 'libx264']
        )

        anim.save(output_path, writer=writer, progress_callback=self._progress_callback)

        # Also save as GIF for web compatibility
        gif_path = self.output_dir / f"{save_filename}.gif"
        anim.save(gif_path, writer='pillow', fps=min(self.config.fps, 20))

        plt.close(fig)

        logger.info(f"Network evolution animation saved: {output_path}")
        return str(output_path)

    def _prepare_animation_data(self, networks, agent_attributes, metrics_data):
        """Prepare and interpolate data for smooth animation."""
        data = {
            'networks': networks,
            'agent_attributes': agent_attributes or [{}] * len(networks),
            'metrics_data': metrics_data,
            'time_points': list(range(len(networks)))
        }

        # Calculate stable layout positions
        data['positions'] = self._calculate_stable_layouts(networks)

        # Interpolate positions for smooth transitions
        if self.config.smooth_transitions:
            data['interpolated_positions'] = self._interpolate_positions(data['positions'])
            data['interpolated_networks'] = self._interpolate_networks(networks)
        else:
            data['interpolated_positions'] = data['positions']
            data['interpolated_networks'] = networks

        # Prepare node and edge attributes
        data['node_attributes'] = self._extract_node_attributes(networks, agent_attributes)
        data['edge_attributes'] = self._extract_edge_attributes(networks)

        # Calculate network metrics if not provided
        if metrics_data is None:
            data['metrics_data'] = self._calculate_network_metrics(networks)
        else:
            data['metrics_data'] = metrics_data

        return data

    def _calculate_stable_layouts(self, networks):
        """Calculate stable layout positions across time points."""
        positions = []

        # Get all unique nodes across time
        all_nodes = set()
        for network in networks:
            all_nodes.update(network.nodes())
        all_nodes = sorted(list(all_nodes))

        # Use the largest network to establish base layout
        largest_network = max(networks, key=len)
        if len(largest_network) > 0:
            # Use spring layout with fixed random state for consistency
            base_pos = nx.spring_layout(
                largest_network,
                k=1/np.sqrt(len(largest_network)),
                iterations=100,
                seed=42
            )
        else:
            base_pos = {}

        # Calculate positions for each time point
        for network in networks:
            pos = {}

            # Use base positions for existing nodes
            for node in network.nodes():
                if node in base_pos:
                    pos[node] = base_pos[node]
                else:
                    # Place new nodes randomly but consistently
                    np.random.seed(hash(str(node)) % 2**32)
                    pos[node] = np.random.random(2) * 2 - 1

            # Refine layout for current network
            if len(network) > 1:
                refined_pos = nx.spring_layout(
                    network,
                    pos=pos,
                    iterations=10,
                    k=1/np.sqrt(len(network))
                )
                pos.update(refined_pos)

            positions.append(pos)

        return positions

    def _interpolate_positions(self, positions):
        """Create smooth position interpolation between time points."""
        if len(positions) < 2:
            return positions

        interpolated = []

        for i in range(len(positions) - 1):
            current_pos = positions[i]
            next_pos = positions[i + 1]

            # Get common nodes
            common_nodes = set(current_pos.keys()) & set(next_pos.keys())

            # Create interpolation for each frame
            for frame in range(self.config.interpolation_frames):
                t = frame / self.config.interpolation_frames
                interp_pos = {}

                for node in common_nodes:
                    curr_xy = current_pos[node]
                    next_xy = next_pos[node]
                    interp_xy = (1-t) * np.array(curr_xy) + t * np.array(next_xy)
                    interp_pos[node] = tuple(interp_xy)

                # Add nodes that appear/disappear
                for node in current_pos:
                    if node not in common_nodes:
                        if t < 0.5:  # Fade out
                            interp_pos[node] = current_pos[node]

                for node in next_pos:
                    if node not in common_nodes:
                        if t >= 0.5:  # Fade in
                            interp_pos[node] = next_pos[node]

                interpolated.append(interp_pos)

        # Add final position
        interpolated.append(positions[-1])

        return interpolated

    def _interpolate_networks(self, networks):
        """Create interpolated networks for smooth edge transitions."""
        if not self.config.smooth_transitions:
            return networks

        interpolated = []

        for i in range(len(networks) - 1):
            current_net = networks[i]
            next_net = networks[i + 1]

            for frame in range(self.config.interpolation_frames):
                t = frame / self.config.interpolation_frames

                # Create interpolated network
                interp_net = nx.Graph()

                # Add all nodes from both networks
                all_nodes = set(current_net.nodes()) | set(next_net.nodes())
                interp_net.add_nodes_from(all_nodes)

                # Interpolate edges
                current_edges = set(current_net.edges())
                next_edges = set(next_net.edges())

                # Persistent edges (in both networks)
                persistent_edges = current_edges & next_edges
                interp_net.add_edges_from(persistent_edges)

                # Disappearing edges
                disappearing_edges = current_edges - next_edges
                if t < 0.7:  # Keep edges for most of transition
                    interp_net.add_edges_from(disappearing_edges)

                # Appearing edges
                appearing_edges = next_edges - current_edges
                if t > 0.3:  # Add edges in latter part of transition
                    interp_net.add_edges_from(appearing_edges)

                interpolated.append(interp_net)

        interpolated.append(networks[-1])
        return interpolated

    def _extract_node_attributes(self, networks, agent_attributes):
        """Extract node attributes for visualization."""
        node_attrs = []

        for i, network in enumerate(networks):
            attrs = {}
            agent_data = agent_attributes[i] if i < len(agent_attributes) else {}

            for node in network.nodes():
                node_data = network.nodes[node].copy()
                if node in agent_data:
                    node_data.update(agent_data[node])

                # Default values
                attrs[node] = {
                    'size': node_data.get('degree', 1) * 20 + 50,
                    'color': self._get_node_color(node_data),
                    'alpha': 0.8,
                    'type': node_data.get('type', 'individual')
                }

            node_attrs.append(attrs)

        return node_attrs

    def _extract_edge_attributes(self, networks):
        """Extract edge attributes for visualization."""
        edge_attrs = []

        for network in networks:
            attrs = {}

            for edge in network.edges():
                edge_data = network.edges[edge]
                attrs[edge] = {
                    'width': edge_data.get('weight', 1.0) * 2,
                    'color': edge_data.get('color', '#333333'),
                    'alpha': edge_data.get('strength', 0.5),
                    'style': edge_data.get('style', '-')
                }

            edge_attrs.append(attrs)

        return edge_attrs

    def _get_node_color(self, node_data):
        """Determine node color based on attributes."""
        # Color by type if available
        node_type = node_data.get('type', 'default')
        if node_type == 'institution':
            return self.color_schemes.primary_palette[3]
        elif node_type == 'group':
            return self.color_schemes.primary_palette[2]
        else:
            # Color by some continuous attribute if available
            if 'opinion' in node_data:
                opinion = node_data['opinion']
                if opinion > 0.5:
                    return self.color_schemes.diverging_palette[-1]
                elif opinion < -0.5:
                    return self.color_schemes.diverging_palette[0]
                else:
                    return self.color_schemes.diverging_palette[4]
            else:
                return self.color_schemes.primary_palette[0]

    def _calculate_network_metrics(self, networks):
        """Calculate network metrics over time."""
        metrics = []

        for network in networks:
            if len(network) == 0:
                metric_dict = {
                    'density': 0,
                    'average_degree': 0,
                    'clustering': 0,
                    'components': 1
                }
            else:
                metric_dict = {
                    'density': nx.density(network),
                    'average_degree': sum(dict(network.degree()).values()) / len(network),
                    'clustering': nx.transitivity(network),
                    'components': nx.number_connected_components(network)
                }

            metrics.append(metric_dict)

        return pd.DataFrame(metrics)

    def _initialize_network_elements(self, ax, data):
        """Initialize network visualization elements."""
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.axis('off')

        elements = {
            'nodes': ax.scatter([], [], s=[], c=[], alpha=0.8, zorder=2),
            'edges': LineCollection([], alpha=0.6, zorder=1),
            'timestamp': ax.text(0.02, 0.98, '', transform=ax.transAxes,
                               fontsize=12, fontweight='bold', va='top',
                               bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
        }

        ax.add_collection(elements['edges'])
        return elements

    def _initialize_metrics_elements(self, ax, data):
        """Initialize metrics visualization elements."""
        metrics_df = data['metrics_data']
        time_points = range(len(metrics_df))

        # Plot full time series in light colors
        ax.plot(time_points, metrics_df['density'], 'o-', alpha=0.3,
               color=self.color_schemes.primary_palette[0], label='Density')
        ax.plot(time_points, metrics_df['clustering'], 's-', alpha=0.3,
               color=self.color_schemes.primary_palette[1], label='Clustering')

        # Current position indicators
        elements = {
            'density_point': ax.scatter([], [], s=100, c=self.color_schemes.primary_palette[0], zorder=3),
            'clustering_point': ax.scatter([], [], s=100, c=self.color_schemes.primary_palette[1], zorder=3),
            'vertical_line': ax.axvline(0, color='red', linestyle='--', alpha=0.7)
        }

        ax.set_xlabel('Time Period')
        ax.set_ylabel('Metric Value')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

        return elements

    def _initialize_timeline_elements(self, ax, data):
        """Initialize timeline visualization elements."""
        n_timepoints = len(data['networks'])
        timeline_x = np.arange(n_timepoints)

        # Timeline bar
        ax.barh(0, n_timepoints, height=0.3, alpha=0.3,
               color=self.color_schemes.primary_palette[2])

        elements = {
            'progress_bar': ax.barh(0, 0, height=0.3,
                                  color=self.color_schemes.primary_palette[2]),
            'current_marker': ax.scatter([], [], s=200, c='red', marker='|',
                                       linewidth=4, zorder=3)
        }

        ax.set_xlim(-0.5, n_timepoints - 0.5)
        ax.set_ylim(-0.5, 0.5)
        ax.set_xlabel('Time Period')
        ax.set_yticks([])

        return elements

    def _update_animation_frame(self, frame, data, network_elements, metrics_elements, timeline_elements):
        """Update all elements for current animation frame."""
        # Determine current time point
        frames_per_period = self.config.interpolation_frames if self.config.smooth_transitions else 1
        time_point = min(frame // frames_per_period, len(data['networks']) - 1)
        interp_progress = (frame % frames_per_period) / frames_per_period if frames_per_period > 1 else 0

        # Get current network and positions
        if frame < len(data['interpolated_positions']):
            current_positions = data['interpolated_positions'][frame]
            current_network = data['interpolated_networks'][frame]
        else:
            current_positions = data['positions'][-1]
            current_network = data['networks'][-1]

        # Update network visualization
        self._update_network_elements(network_elements, current_network, current_positions,
                                    data, time_point, frame)

        # Update metrics visualization
        self._update_metrics_elements(metrics_elements, data, time_point, interp_progress)

        # Update timeline
        self._update_timeline_elements(timeline_elements, data, time_point, frame)

        # Return all artists for blitting
        artists = list(network_elements.values()) + list(metrics_elements.values()) + list(timeline_elements.values())
        return [artist for artist in artists if hasattr(artist, 'set_visible')]

    def _update_network_elements(self, elements, network, positions, data, time_point, frame):
        """Update network visualization elements."""
        if len(network) == 0 or len(positions) == 0:
            elements['nodes'].set_offsets([])
            elements['edges'].set_segments([])
            return

        # Get node positions and attributes
        nodes = list(network.nodes())
        node_positions = np.array([positions[node] for node in nodes])
        node_attrs = data['node_attributes'][time_point] if time_point < len(data['node_attributes']) else {}

        # Update nodes
        node_sizes = [node_attrs.get(node, {}).get('size', 50) for node in nodes]
        node_colors = [node_attrs.get(node, {}).get('color', self.color_schemes.primary_palette[0])
                      for node in nodes]

        elements['nodes'].set_offsets(node_positions)
        elements['nodes'].set_sizes(node_sizes)
        elements['nodes'].set_color(node_colors)

        # Update edges
        edge_segments = []
        edge_colors = []
        edge_widths = []

        edge_attrs = data['edge_attributes'][time_point] if time_point < len(data['edge_attributes']) else {}

        for edge in network.edges():
            if edge[0] in positions and edge[1] in positions:
                segment = [positions[edge[0]], positions[edge[1]]]
                edge_segments.append(segment)

                edge_attr = edge_attrs.get(edge, {})
                edge_colors.append(edge_attr.get('color', '#333333'))
                edge_widths.append(edge_attr.get('width', 1.0))

        elements['edges'].set_segments(edge_segments)
        elements['edges'].set_colors(edge_colors)
        elements['edges'].set_linewidths(edge_widths)

        # Update timestamp
        elements['timestamp'].set_text(f'Time: {time_point + 1}/{len(data["networks"])}')

    def _update_metrics_elements(self, elements, data, time_point, interp_progress):
        """Update metrics visualization elements."""
        metrics_df = data['metrics_data']

        # Update current position markers
        current_density = metrics_df.iloc[time_point]['density']
        current_clustering = metrics_df.iloc[time_point]['clustering']

        elements['density_point'].set_offsets([(time_point, current_density)])
        elements['clustering_point'].set_offsets([(time_point, current_clustering)])

        # Update vertical line
        elements['vertical_line'].set_xdata([time_point, time_point])

    def _update_timeline_elements(self, elements, data, time_point, frame):
        """Update timeline visualization elements."""
        total_frames = len(data['interpolated_positions'])
        progress = frame / max(total_frames - 1, 1)

        # Update progress bar
        elements['progress_bar'].set_width(time_point + progress / self.config.interpolation_frames)

        # Update current marker
        elements['current_marker'].set_offsets([(time_point, 0)])

    def _progress_callback(self, current_frame, total_frames):
        """Callback for animation saving progress."""
        if current_frame % 50 == 0:  # Log every 50 frames
            progress = (current_frame / total_frames) * 100
            logger.info(f"Animation progress: {progress:.1f}% ({current_frame}/{total_frames} frames)")

    def animate_network_comparison(self,
                                 empirical_networks: List[nx.Graph],
                                 simulated_networks: List[nx.Graph],
                                 title: str = "Network Evolution: Empirical vs Simulated",
                                 save_filename: str = "network_comparison") -> str:
        """
        Create side-by-side comparison animation of empirical vs simulated networks.

        Args:
            empirical_networks: List of empirical networks
            simulated_networks: List of simulated networks
            title: Animation title
            save_filename: Output filename

        Returns:
            Path to saved animation file
        """
        logger.info("Creating network comparison animation")

        # Ensure same number of time points
        min_length = min(len(empirical_networks), len(simulated_networks))
        empirical_networks = empirical_networks[:min_length]
        simulated_networks = simulated_networks[:min_length]

        # Create figure with side-by-side layout
        fig = plt.figure(figsize=(16, 8), dpi=self.config.dpi)
        gs = fig.add_gridspec(2, 2, height_ratios=[3, 1], hspace=0.3, wspace=0.3)

        fig.suptitle(title, fontsize=18, fontweight='bold')

        # Network plots
        ax_empirical = fig.add_subplot(gs[0, 0])
        ax_empirical.set_title('Empirical Networks', fontweight='bold')
        ax_empirical.set_aspect('equal')
        ax_empirical.axis('off')

        ax_simulated = fig.add_subplot(gs[0, 1])
        ax_simulated.set_title('Simulated Networks', fontweight='bold')
        ax_simulated.set_aspect('equal')
        ax_simulated.axis('off')

        # Metrics comparison plot
        ax_metrics = fig.add_subplot(gs[1, :])
        ax_metrics.set_title('Network Metrics Comparison', fontweight='bold')

        # Prepare data
        emp_data = self._prepare_animation_data(empirical_networks, None, None)
        sim_data = self._prepare_animation_data(simulated_networks, None, None)

        # Initialize elements
        emp_elements = self._initialize_network_elements(ax_empirical, emp_data)
        sim_elements = self._initialize_network_elements(ax_simulated, sim_data)
        comp_elements = self._initialize_comparison_elements(ax_metrics, emp_data, sim_data)

        def animate(frame):
            self._update_comparison_frame(frame, emp_data, sim_data,
                                        emp_elements, sim_elements, comp_elements)
            return []

        # Create and save animation
        total_frames = min(len(emp_data['interpolated_positions']),
                          len(sim_data['interpolated_positions']))

        anim = animation.FuncAnimation(fig, animate, frames=total_frames,
                                     interval=1000/self.config.fps, blit=False)

        output_path = self.output_dir / f"{save_filename}.mp4"
        writer = animation.FFMpegWriter(fps=self.config.fps, bitrate=self.config.bitrate)
        anim.save(output_path, writer=writer)

        plt.close(fig)
        logger.info(f"Network comparison animation saved: {output_path}")
        return str(output_path)

    def _initialize_comparison_elements(self, ax, emp_data, sim_data):
        """Initialize comparison metrics elements."""
        emp_metrics = emp_data['metrics_data']
        sim_metrics = sim_data['metrics_data']
        time_points = range(len(emp_metrics))

        # Plot full time series
        ax.plot(time_points, emp_metrics['density'], 'o-', alpha=0.4,
               color=self.color_schemes.primary_palette[0], label='Empirical Density')
        ax.plot(time_points, sim_metrics['density'], 's-', alpha=0.4,
               color=self.color_schemes.primary_palette[1], label='Simulated Density')

        elements = {
            'emp_density': ax.scatter([], [], s=100, c=self.color_schemes.primary_palette[0], zorder=3),
            'sim_density': ax.scatter([], [], s=100, c=self.color_schemes.primary_palette[1], zorder=3),
            'vertical_line': ax.axvline(0, color='red', linestyle='--', alpha=0.7)
        }

        ax.set_xlabel('Time Period')
        ax.set_ylabel('Network Density')
        ax.legend()
        ax.grid(True, alpha=0.3)

        return elements

    def _update_comparison_frame(self, frame, emp_data, sim_data, emp_elements, sim_elements, comp_elements):
        """Update comparison animation frame."""
        frames_per_period = self.config.interpolation_frames
        time_point = min(frame // frames_per_period, len(emp_data['networks']) - 1)

        # Update empirical network
        if frame < len(emp_data['interpolated_positions']):
            emp_pos = emp_data['interpolated_positions'][frame]
            emp_net = emp_data['interpolated_networks'][frame]
            self._update_network_elements(emp_elements, emp_net, emp_pos, emp_data, time_point, frame)

        # Update simulated network
        if frame < len(sim_data['interpolated_positions']):
            sim_pos = sim_data['interpolated_positions'][frame]
            sim_net = sim_data['interpolated_networks'][frame]
            self._update_network_elements(sim_elements, sim_net, sim_pos, sim_data, time_point, frame)

        # Update comparison metrics
        if time_point < len(emp_data['metrics_data']) and time_point < len(sim_data['metrics_data']):
            emp_density = emp_data['metrics_data'].iloc[time_point]['density']
            sim_density = sim_data['metrics_data'].iloc[time_point]['density']

            comp_elements['emp_density'].set_offsets([(time_point, emp_density)])
            comp_elements['sim_density'].set_offsets([(time_point, sim_density)])
            comp_elements['vertical_line'].set_xdata([time_point, time_point])

# Example usage and testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Create sample network sequence for testing
    def create_sample_networks(n_networks=10, n_nodes=20):
        """Create sample evolving networks."""
        networks = []
        base_network = nx.erdos_renyi_graph(n_nodes, 0.1, seed=42)

        for i in range(n_networks):
            # Evolve network slightly
            evolved = base_network.copy()

            # Add some edges
            non_edges = list(nx.non_edges(evolved))
            if non_edges:
                new_edges = np.random.choice(len(non_edges), size=min(2, len(non_edges)), replace=False)
                for idx in new_edges:
                    evolved.add_edge(*non_edges[idx])

            # Remove some edges
            if evolved.edges():
                edges_to_remove = list(np.random.choice(len(evolved.edges()), size=1))
                for idx in edges_to_remove:
                    edge = list(evolved.edges())[idx]
                    evolved.remove_edge(*edge)

            networks.append(evolved)
            base_network = evolved.copy()

        return networks

    # Initialize animator
    animator = NetworkEvolutionAnimator()

    # Create sample data
    sample_networks = create_sample_networks()

    # Create animation
    animation_path = animator.animate_network_evolution(
        networks=sample_networks,
        title="Sample Network Evolution Animation",
        save_filename="sample_evolution"
    )

    print(f"Sample animation created: {animation_path}")

    # Create comparison animation
    empirical_nets = create_sample_networks(n_networks=8, n_nodes=15)
    simulated_nets = create_sample_networks(n_networks=8, n_nodes=15)

    comparison_path = animator.animate_network_comparison(
        empirical_networks=empirical_nets,
        simulated_networks=simulated_nets,
        save_filename="sample_comparison"
    )

    print(f"Sample comparison animation created: {comparison_path}")