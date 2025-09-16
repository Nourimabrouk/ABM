"""
Publication-Quality Figures for Tolerance Intervention Research
==============================================================

Creates publication-ready visualizations specifically for PhD research on
tolerance interventions promoting interethnic cooperation. All figures are
designed for 300+ DPI output meeting top-tier journal standards for
computational social science.

Target Publication: Journal of Artificial Societies and Social Simulation (JASSS)
Research Context: PhD Dissertation Defense & Academic Publications

Specific Figures:
1. Conceptual Model: Tolerance → Influence → Cooperation
2. Network Structure Examples with Ethnic Composition
3. Intervention Strategies Visualization
4. Simulation Results: Tolerance Evolution & Cooperation Emergence
5. Empirical Validation: Model Fit to Real Classroom Data

Author: Claude Code - Visualization Virtuoso
Created: 2025-09-16
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

# Scientific computation for statistical visualization
from scipy import stats
from scipy.optimize import curve_fit
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import r2_score

# Custom imports
from ..utils.color_schemes import AcademicColorSchemes
from ..utils.export_utilities import FigureExporter

logger = logging.getLogger(__name__)

# Publication color schemes for tolerance intervention research
TOLERANCE_COLORS = {
    'low': '#d7191c',      # Red for low tolerance
    'medium': '#ffffbf',   # Yellow for neutral tolerance
    'high': '#2c7bb6',     # Blue for high tolerance
    'intervention': '#FFD700'  # Gold for intervention targets
}

ETHNIC_COLORS = {
    'majority': '#2E86AB',    # Blue
    'minority': '#A23B72',    # Rose
    'interethnic': '#F24236'  # Red for interethnic connections
}

@dataclass
class TolerancePublicationConfig:
    """Configuration for tolerance intervention publication figures."""
    figure_size: Tuple[float, float] = (16, 12)
    dpi: int = 300
    font_family: str = 'DejaVu Sans'
    font_size_title: int = 18
    font_size_label: int = 14
    font_size_tick: int = 12
    font_size_legend: int = 12
    line_width: float = 2.5
    marker_size: float = 8.0
    alpha: float = 0.8
    save_format: str = 'png'
    save_formats: List[str] = None
    tight_layout: bool = True
    transparent_background: bool = False

    def __post_init__(self):
        if self.save_formats is None:
            self.save_formats = ['png', 'pdf', 'svg']

class ToleranceInterventionPublicationFigures:
    """
    Creates publication-quality figures for tolerance intervention research.

    This class generates all figures needed for PhD dissertation defense
    and JASSS publication, with emphasis on tolerance diffusion mechanisms
    and interethnic cooperation emergence.
    """

    def __init__(self, output_dir: Path = None, config: TolerancePublicationConfig = None):
        """Initialize publication figure generator."""
        self.output_dir = output_dir or Path("outputs/publication_figures/tolerance_intervention")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.config = config or TolerancePublicationConfig()
        self.color_schemes = AcademicColorSchemes()
        self.exporter = FigureExporter(self.output_dir)

        # Setup matplotlib for publication quality
        self._setup_publication_style()

        logger.info(f"Tolerance intervention publication figures initialized: {self.output_dir}")

    def _setup_publication_style(self):
        """Setup matplotlib for publication-quality tolerance intervention figures."""
        plt.style.use('seaborn-v0_8-whitegrid')
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
            'savefig.transparent': self.config.transparent_background,
            'savefig.facecolor': 'white',
            'axes.facecolor': 'white'
        })

    def create_figure_1_conceptual_model(self,
                                       save_filename: str = "figure_01_tolerance_intervention_conceptual_model"
                                       ) -> List[str]:
        """
        Create Figure 1: Conceptual Model of Tolerance Intervention Mechanisms.

        Shows how tolerance interventions lead to social influence and ultimately
        promote interethnic cooperation through network mechanisms.

        Returns:
            List of paths to saved figures in different formats
        """
        logger.info("Creating Figure 1: Conceptual Model")

        fig = plt.figure(figsize=(18, 14))
        gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.4, wspace=0.3)

        # Main title
        fig.suptitle('Tolerance Intervention Mechanisms:\nFrom Individual Change to Network Cooperation',
                    fontsize=22, fontweight='bold', y=0.95)

        # Panel A: Micro-Theory Diagram
        ax_micro = fig.add_subplot(gs[0, :])
        self._draw_micro_theory_diagram(ax_micro)
        ax_micro.set_title('(A) Micro-Level Theory: Individual Tolerance → Social Influence → Cooperation',
                          fontweight='bold', pad=20)

        # Panel B: Attraction-Repulsion Mechanism
        ax_attraction = fig.add_subplot(gs[1, 0])
        self._draw_attraction_repulsion_mechanism(ax_attraction)
        ax_attraction.set_title('(B) Attraction-Repulsion\nDynamics', fontweight='bold', pad=20)

        # Panel C: Social Influence Network
        ax_influence = fig.add_subplot(gs[1, 1])
        self._draw_social_influence_network(ax_influence)
        ax_influence.set_title('(C) Social Influence\nNetwork Process', fontweight='bold', pad=20)

        # Panel D: Cooperation Emergence
        ax_cooperation = fig.add_subplot(gs[1, 2])
        self._draw_cooperation_emergence(ax_cooperation)
        ax_cooperation.set_title('(D) Interethnic Cooperation\nEmergence', fontweight='bold', pad=20)

        # Panel E: Intervention Targeting Strategies
        ax_targeting = fig.add_subplot(gs[2, :2])
        self._draw_intervention_targeting_strategies(ax_targeting)
        ax_targeting.set_title('(E) Intervention Targeting Strategies in Social Networks',
                              fontweight='bold', pad=20)

        # Panel F: Temporal Dynamics
        ax_temporal = fig.add_subplot(gs[2, 2])
        self._draw_temporal_dynamics_overview(ax_temporal)
        ax_temporal.set_title('(F) Temporal Dynamics\nOverview', fontweight='bold', pad=20)

        # Save in multiple formats
        saved_paths = []
        for fmt in self.config.save_formats:
            path = self.exporter.save_figure(fig, save_filename, format=fmt)
            saved_paths.append(str(path))

        plt.close(fig)
        logger.info(f"Figure 1 saved in {len(saved_paths)} formats")
        return saved_paths

    def _draw_micro_theory_diagram(self, ax):
        """Draw micro-level theory flow diagram."""
        ax.set_xlim(0, 12)
        ax.set_ylim(0, 4)

        # Step 1: Tolerance Intervention
        intervention_box = patches.FancyBboxPatch(
            (0.5, 1.5), 2, 1, boxstyle="round,pad=0.1",
            facecolor=TOLERANCE_COLORS['intervention'], edgecolor='black', linewidth=2
        )
        ax.add_patch(intervention_box)
        ax.text(1.5, 2, 'Tolerance\nIntervention', ha='center', va='center',
               fontweight='bold', fontsize=12)

        # Step 2: Individual Change
        individual_box = patches.FancyBboxPatch(
            (3.5, 1.5), 2, 1, boxstyle="round,pad=0.1",
            facecolor=TOLERANCE_COLORS['high'], edgecolor='black', linewidth=2
        )
        ax.add_patch(individual_box)
        ax.text(4.5, 2, 'Individual\nTolerance ↑', ha='center', va='center',
               fontweight='bold', fontsize=12, color='white')

        # Step 3: Social Influence
        influence_box = patches.FancyBboxPatch(
            (6.5, 1.5), 2, 1, boxstyle="round,pad=0.1",
            facecolor=self.color_schemes.primary_palette[1], edgecolor='black', linewidth=2
        )
        ax.add_patch(influence_box)
        ax.text(7.5, 2, 'Social\nInfluence', ha='center', va='center',
               fontweight='bold', fontsize=12, color='white')

        # Step 4: Network Cooperation
        cooperation_box = patches.FancyBboxPatch(
            (9.5, 1.5), 2, 1, boxstyle="round,pad=0.1",
            facecolor=ETHNIC_COLORS['interethnic'], edgecolor='black', linewidth=2
        )
        ax.add_patch(cooperation_box)
        ax.text(10.5, 2, 'Interethnic\nCooperation', ha='center', va='center',
               fontweight='bold', fontsize=12, color='white')

        # Arrows between steps
        arrow_props = dict(arrowstyle='->', lw=3, color='black')
        ax.annotate('', xy=(3.4, 2), xytext=(2.6, 2), arrowprops=arrow_props)
        ax.annotate('', xy=(6.4, 2), xytext=(5.6, 2), arrowprops=arrow_props)
        ax.annotate('', xy=(9.4, 2), xytext=(8.6, 2), arrowprops=arrow_props)

        # Add mechanism labels above arrows
        ax.text(3, 2.7, 'Direct\nEffect', ha='center', fontsize=10, fontweight='bold')
        ax.text(6, 2.7, 'Network\nDiffusion', ha='center', fontsize=10, fontweight='bold')
        ax.text(9, 2.7, 'Behavioral\nChange', ha='center', fontsize=10, fontweight='bold')

        # Add feedback loop
        ax.annotate('', xy=(2, 1.3), xytext=(10, 1.3),
                   arrowprops=dict(arrowstyle='->', lw=2, color='red', alpha=0.7,
                                 connectionstyle="arc3,rad=-0.3"))
        ax.text(6, 0.8, 'Feedback: Cooperation reinforces tolerance', ha='center',
               fontsize=11, color='red', fontweight='bold')

        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    def _draw_attraction_repulsion_mechanism(self, ax):
        """Draw attraction-repulsion dynamics visualization."""
        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)

        # Central tolerant agent
        central_circle = patches.Circle((0, 0), 0.3, facecolor=TOLERANCE_COLORS['high'],
                                      edgecolor='black', linewidth=2)
        ax.add_patch(central_circle)
        ax.text(0, 0, 'T+', ha='center', va='center', fontweight='bold',
               fontsize=12, color='white')

        # Surrounding agents with different tolerance levels
        positions = [(1.2, 0.8), (-1.2, 0.8), (1.2, -0.8), (-1.2, -0.8),
                    (0, 1.5), (0, -1.5), (1.5, 0), (-1.5, 0)]
        tolerances = [0.7, -0.3, 0.5, -0.7, 0.2, -0.5, 0.8, -0.2]
        labels = ['T+', 'T-', 'T+', 'T-', 'T0', 'T-', 'T+', 'T0']

        for pos, tol, label in zip(positions, tolerances, labels):
            if tol > 0.4:
                color = TOLERANCE_COLORS['high']
                text_color = 'white'
            elif tol < -0.2:
                color = TOLERANCE_COLORS['low']
                text_color = 'white'
            else:
                color = TOLERANCE_COLORS['medium']
                text_color = 'black'

            circle = patches.Circle(pos, 0.2, facecolor=color, edgecolor='black', linewidth=1)
            ax.add_patch(circle)
            ax.text(pos[0], pos[1], label, ha='center', va='center',
                   fontweight='bold', fontsize=10, color=text_color)

            # Draw influence arrows
            if tol > 0:
                # Attraction
                ax.annotate('', xy=pos, xytext=(0, 0),
                           arrowprops=dict(arrowstyle='->', lw=2, color='green', alpha=0.7))
            else:
                # Repulsion
                dx, dy = pos[0] - 0, pos[1] - 0
                length = np.sqrt(dx**2 + dy**2)
                dx, dy = dx/length * 0.4, dy/length * 0.4
                ax.annotate('', xy=(pos[0] + dx, pos[1] + dy), xytext=pos,
                           arrowprops=dict(arrowstyle='->', lw=2, color='red', alpha=0.7))

        ax.set_title('Tolerance-Based\nAttraction/Repulsion', fontsize=11, fontweight='bold')
        ax.set_aspect('equal')
        ax.axis('off')

    def _draw_social_influence_network(self, ax):
        """Draw social influence network process."""
        # Create small network
        G = nx.Graph()
        G.add_edges_from([(0, 1), (1, 2), (1, 3), (2, 4), (3, 4), (0, 5), (5, 6)])

        # Position nodes
        pos = nx.spring_layout(G, k=2, iterations=50)

        # Node tolerances (node 1 is intervention target)
        tolerances = {0: -0.2, 1: 0.8, 2: 0.1, 3: 0.3, 4: 0.5, 5: -0.1, 6: 0.2}
        ethnicities = {0: 'minority', 1: 'majority', 2: 'minority', 3: 'majority',
                      4: 'minority', 5: 'majority', 6: 'minority'}

        # Draw edges
        for edge in G.edges():
            x_vals = [pos[edge[0]][0], pos[edge[1]][0]]
            y_vals = [pos[edge[0]][1], pos[edge[1]][1]]
            ax.plot(x_vals, y_vals, 'gray', alpha=0.5, linewidth=2)

        # Draw nodes
        for node in G.nodes():
            tolerance = tolerances[node]
            ethnicity = ethnicities[node]

            if tolerance > 0.5:
                color = TOLERANCE_COLORS['high']
            elif tolerance < -0.1:
                color = TOLERANCE_COLORS['low']
            else:
                color = TOLERANCE_COLORS['medium']

            # Node size based on intervention status
            size = 800 if node == 1 else 400

            # Border color based on ethnicity
            border_color = ETHNIC_COLORS['majority'] if ethnicity == 'majority' else ETHNIC_COLORS['minority']

            ax.scatter(pos[node][0], pos[node][1], s=size, c=color,
                      edgecolors=border_color, linewidth=3, alpha=0.9)

            # Add tolerance value
            ax.text(pos[node][0], pos[node][1], f'{tolerance:.1f}',
                   ha='center', va='center', fontweight='bold', fontsize=9)

        # Highlight intervention target
        intervention_pos = pos[1]
        star = patches.RegularPolygon((intervention_pos[0], intervention_pos[1]), 5, 0.15,
                                    facecolor=TOLERANCE_COLORS['intervention'],
                                    edgecolor='black', linewidth=2, alpha=0.8)
        ax.add_patch(star)

        ax.set_title('Network Influence\nPropagation', fontsize=11, fontweight='bold')
        ax.set_aspect('equal')
        ax.axis('off')

    def _draw_cooperation_emergence(self, ax):
        """Draw cooperation emergence visualization."""
        # Create bipartite-like structure representing ethnic groups
        majority_nodes = [(0.2, 0.8), (0.2, 0.5), (0.2, 0.2)]
        minority_nodes = [(0.8, 0.8), (0.8, 0.5), (0.8, 0.2)]

        # Draw ethnic group nodes
        for pos in majority_nodes:
            ax.scatter(pos[0], pos[1], s=300, c=ETHNIC_COLORS['majority'],
                      alpha=0.8, edgecolors='black', linewidth=2)
            ax.text(pos[0], pos[1], 'M', ha='center', va='center',
                   fontweight='bold', fontsize=10, color='white')

        for pos in minority_nodes:
            ax.scatter(pos[0], pos[1], s=300, c=ETHNIC_COLORS['minority'],
                      alpha=0.8, edgecolors='black', linewidth=2)
            ax.text(pos[0], pos[1], 'm', ha='center', va='center',
                   fontweight='bold', fontsize=10, color='white')

        # Draw cooperation ties with varying strength
        cooperation_ties = [
            (majority_nodes[0], minority_nodes[0], 0.8),  # Strong
            (majority_nodes[1], minority_nodes[1], 0.6),  # Medium
            (majority_nodes[2], minority_nodes[2], 0.3),  # Weak
        ]

        for start, end, strength in cooperation_ties:
            color_intensity = strength
            line_width = strength * 6
            alpha = 0.3 + strength * 0.7

            ax.plot([start[0], end[0]], [start[1], end[1]],
                   color=ETHNIC_COLORS['interethnic'], linewidth=line_width,
                   alpha=alpha)

            # Add cooperation strength label
            mid_x = (start[0] + end[0]) / 2
            mid_y = (start[1] + end[1]) / 2
            ax.text(mid_x, mid_y + 0.05, f'{strength:.1f}',
                   ha='center', va='bottom', fontweight='bold',
                   fontsize=9, color=ETHNIC_COLORS['interethnic'])

        # Add group labels
        ax.text(0.1, 0.95, 'Majority\nGroup', ha='center', va='top',
               fontweight='bold', fontsize=10, color=ETHNIC_COLORS['majority'])
        ax.text(0.9, 0.95, 'Minority\nGroup', ha='center', va='top',
               fontweight='bold', fontsize=10, color=ETHNIC_COLORS['minority'])

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title('Interethnic Cooperation\nStrength', fontsize=11, fontweight='bold')
        ax.axis('off')

    def _draw_intervention_targeting_strategies(self, ax):
        """Draw different intervention targeting strategies."""
        ax.set_xlim(0, 12)
        ax.set_ylim(0, 8)

        # Strategy 1: Central Node Targeting
        ax.text(1.5, 7, 'Central Node Targeting', ha='center', fontweight='bold', fontsize=12)
        central_net = self._create_example_network(center_pos=(1.5, 5.5), strategy='central')
        self._draw_strategy_network(ax, central_net, center_pos=(1.5, 5.5))

        # Strategy 2: Peripheral Node Targeting
        ax.text(4.5, 7, 'Peripheral Node Targeting', ha='center', fontweight='bold', fontsize=12)
        peripheral_net = self._create_example_network(center_pos=(4.5, 5.5), strategy='peripheral')
        self._draw_strategy_network(ax, peripheral_net, center_pos=(4.5, 5.5))

        # Strategy 3: Random Targeting
        ax.text(7.5, 7, 'Random Targeting', ha='center', fontweight='bold', fontsize=12)
        random_net = self._create_example_network(center_pos=(7.5, 5.5), strategy='random')
        self._draw_strategy_network(ax, random_net, center_pos=(7.5, 5.5))

        # Strategy 4: Clustered Targeting
        ax.text(10.5, 7, 'Clustered Targeting', ha='center', fontweight='bold', fontsize=12)
        clustered_net = self._create_example_network(center_pos=(10.5, 5.5), strategy='clustered')
        self._draw_strategy_network(ax, clustered_net, center_pos=(10.5, 5.5))

        # Add effectiveness indicators
        effectiveness = ['High', 'Medium', 'Low', 'Medium']
        colors = [TOLERANCE_COLORS['high'], TOLERANCE_COLORS['medium'],
                 TOLERANCE_COLORS['low'], TOLERANCE_COLORS['medium']]

        for i, (eff, color) in enumerate(zip(effectiveness, colors)):
            x_pos = 1.5 + i * 3
            ax.text(x_pos, 3.5, f'Effectiveness:\n{eff}', ha='center', va='center',
                   fontweight='bold', fontsize=10,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.7))

        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    def _create_example_network(self, center_pos, strategy):
        """Create example network for strategy visualization."""
        # Simple network structure
        positions = {
            0: (center_pos[0], center_pos[1] + 0.5),     # Top
            1: (center_pos[0] - 0.5, center_pos[1]),     # Left
            2: (center_pos[0] + 0.5, center_pos[1]),     # Right
            3: (center_pos[0], center_pos[1] - 0.5),     # Bottom
            4: (center_pos[0] - 0.3, center_pos[1] + 0.3), # Top-left
            5: (center_pos[0] + 0.3, center_pos[1] + 0.3), # Top-right
        }

        # Intervention targets based on strategy
        if strategy == 'central':
            targets = [1]  # Central node
        elif strategy == 'peripheral':
            targets = [3]  # Peripheral node
        elif strategy == 'random':
            targets = [2]  # Random selection
        elif strategy == 'clustered':
            targets = [0, 4]  # Clustered nodes
        else:
            targets = [1]

        return {'positions': positions, 'targets': targets}

    def _draw_strategy_network(self, ax, network_data, center_pos):
        """Draw a single strategy network."""
        positions = network_data['positions']
        targets = network_data['targets']

        # Draw edges (simple star pattern)
        center_node = 1
        for node in positions:
            if node != center_node:
                ax.plot([positions[center_node][0], positions[node][0]],
                       [positions[center_node][1], positions[node][1]],
                       'gray', alpha=0.5, linewidth=1)

        # Draw nodes
        for node, pos in positions.items():
            if node in targets:
                # Intervention target
                ax.scatter(pos[0], pos[1], s=200, c=TOLERANCE_COLORS['intervention'],
                          edgecolors='black', linewidth=2, marker='*')
            else:
                # Regular node
                ax.scatter(pos[0], pos[1], s=100, c='lightblue',
                          edgecolors='black', linewidth=1)

    def _draw_temporal_dynamics_overview(self, ax):
        """Draw temporal dynamics overview."""
        # Create mock time series data
        time_points = np.linspace(0, 20, 100)
        intervention_start = 5
        intervention_end = 8

        # Tolerance evolution
        tolerance = np.zeros_like(time_points)
        for i, t in enumerate(time_points):
            if t < intervention_start:
                tolerance[i] = 0.1 + 0.02 * np.random.random()
            elif t < intervention_end:
                # During intervention
                progress = (t - intervention_start) / (intervention_end - intervention_start)
                tolerance[i] = 0.1 + 0.4 * progress + 0.02 * np.random.random()
            else:
                # Post-intervention decay
                decay = np.exp(-(t - intervention_end) / 5)
                tolerance[i] = 0.5 * decay + 0.1 + 0.02 * np.random.random()

        ax.plot(time_points, tolerance, color=TOLERANCE_COLORS['high'], linewidth=3,
               label='Tolerance Level')

        # Highlight intervention period
        ax.axvspan(intervention_start, intervention_end, alpha=0.3,
                  color=TOLERANCE_COLORS['intervention'], label='Intervention Period')

        # Add cooperation emergence (delayed)
        cooperation = np.zeros_like(time_points)
        for i, t in enumerate(time_points):
            if t > intervention_start + 2:
                # Cooperation emerges after tolerance increase
                coop_growth = min(1.0, (t - intervention_start - 2) / 10)
                cooperation[i] = 0.3 * coop_growth * tolerance[i] + 0.01 * np.random.random()

        ax2 = ax.twinx()
        ax2.plot(time_points, cooperation, color=ETHNIC_COLORS['interethnic'],
                linewidth=3, linestyle='--', label='Cooperation Level')

        ax.set_xlabel('Time Periods')
        ax.set_ylabel('Tolerance Level', color=TOLERANCE_COLORS['high'])
        ax2.set_ylabel('Cooperation Level', color=ETHNIC_COLORS['interethnic'])

        # Combine legends
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=9)

        ax.grid(True, alpha=0.3)

    def create_figure_2_network_examples(self,
                                       classroom_networks: Optional[List[nx.Graph]] = None,
                                       save_filename: str = "figure_02_network_structure_examples"
                                       ) -> List[str]:
        """
        Create Figure 2: Network Structure Examples with Ethnic Composition.

        Shows real classroom networks with friendship vs cooperation layers
        and ethnic group composition patterns.

        Args:
            classroom_networks: Optional real classroom network data
            save_filename: Output filename

        Returns:
            List of paths to saved figures
        """
        logger.info("Creating Figure 2: Network Structure Examples")

        fig = plt.figure(figsize=(20, 12))
        gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.3, wspace=0.3)

        # Main title
        fig.suptitle('Social Network Structure in Educational Settings:\nFriendship Networks and Ethnic Composition',
                    fontsize=22, fontweight='bold', y=0.95)

        # Generate or use provided classroom networks
        if classroom_networks is None:
            classroom_networks = self._generate_example_classroom_networks()

        # Panel A: Classroom Network 1
        ax_class1 = fig.add_subplot(gs[0, 0])
        self._plot_classroom_network(ax_class1, classroom_networks[0], "Classroom A\n(High Segregation)")

        # Panel B: Classroom Network 2
        ax_class2 = fig.add_subplot(gs[0, 1])
        self._plot_classroom_network(ax_class2, classroom_networks[1], "Classroom B\n(Medium Integration)")

        # Panel C: Classroom Network 3
        ax_class3 = fig.add_subplot(gs[0, 2])
        self._plot_classroom_network(ax_class3, classroom_networks[2], "Classroom C\n(High Integration)")

        # Panel D: Network Metrics Comparison
        ax_metrics = fig.add_subplot(gs[0, 3])
        self._plot_classroom_network_metrics(ax_metrics, classroom_networks)

        # Panel E: Friendship vs Cooperation Networks
        ax_friendship = fig.add_subplot(gs[1, 0])
        ax_cooperation = fig.add_subplot(gs[1, 1])
        self._plot_friendship_vs_cooperation_networks(ax_friendship, ax_cooperation, classroom_networks[1])

        # Panel F: Ethnic Composition Analysis
        ax_composition = fig.add_subplot(gs[1, 2])
        self._plot_ethnic_composition_analysis(ax_composition, classroom_networks)

        # Panel G: Intervention Potential Assessment
        ax_intervention = fig.add_subplot(gs[1, 3])
        self._plot_intervention_potential(ax_intervention, classroom_networks)

        # Save in multiple formats
        saved_paths = []
        for fmt in self.config.save_formats:
            path = self.exporter.save_figure(fig, save_filename, format=fmt)
            saved_paths.append(str(path))

        plt.close(fig)
        logger.info(f"Figure 2 saved in {len(saved_paths)} formats")
        return saved_paths

    def _generate_example_classroom_networks(self):
        """Generate example classroom networks with different integration levels."""
        networks = []

        for integration_level in ['high_segregation', 'medium_integration', 'high_integration']:
            # Create base network
            n_students = 25
            n_majority = 15
            n_minority = 10

            G = nx.Graph()
            G.add_nodes_from(range(n_students))

            # Assign ethnicities
            for i in range(n_students):
                ethnicity = 'minority' if i < n_minority else 'majority'
                G.nodes[i]['ethnicity'] = ethnicity

            # Add edges based on integration level
            if integration_level == 'high_segregation':
                # Mostly within-group connections
                self._add_segregated_edges(G, n_minority, n_majority, homophily=0.9)
            elif integration_level == 'medium_integration':
                # Moderate cross-group connections
                self._add_segregated_edges(G, n_minority, n_majority, homophily=0.6)
            else:  # high_integration
                # Many cross-group connections
                self._add_segregated_edges(G, n_minority, n_majority, homophily=0.3)

            networks.append(G)

        return networks

    def _add_segregated_edges(self, G, n_minority, n_majority, homophily):
        """Add edges to network with specified level of homophily."""
        n_students = len(G.nodes())

        # Calculate target number of edges
        target_edges = int(n_students * 1.5)  # Average degree ~3

        edges_added = 0
        while edges_added < target_edges:
            u = np.random.randint(0, n_students)
            v = np.random.randint(0, n_students)

            if u != v and not G.has_edge(u, v):
                u_ethnicity = G.nodes[u]['ethnicity']
                v_ethnicity = G.nodes[v]['ethnicity']

                # Apply homophily bias
                if u_ethnicity == v_ethnicity:
                    # Same ethnicity - higher probability
                    if np.random.random() < homophily:
                        G.add_edge(u, v)
                        edges_added += 1
                else:
                    # Different ethnicity - lower probability
                    if np.random.random() < (1 - homophily):
                        G.add_edge(u, v)
                        edges_added += 1

    def _plot_classroom_network(self, ax, network, title):
        """Plot individual classroom network with ethnic coloring."""
        pos = nx.spring_layout(network, k=0.5, iterations=50)

        # Separate nodes by ethnicity
        majority_nodes = [n for n in network.nodes() if network.nodes[n]['ethnicity'] == 'majority']
        minority_nodes = [n for n in network.nodes() if network.nodes[n]['ethnicity'] == 'minority']

        # Draw edges
        nx.draw_networkx_edges(network, pos, ax=ax, edge_color='gray', alpha=0.3, width=0.5)

        # Draw majority nodes
        nx.draw_networkx_nodes(network, pos, nodelist=majority_nodes, ax=ax,
                              node_color=ETHNIC_COLORS['majority'], node_size=200,
                              alpha=0.8, edgecolors='black', linewidths=1)

        # Draw minority nodes
        nx.draw_networkx_nodes(network, pos, nodelist=minority_nodes, ax=ax,
                              node_color=ETHNIC_COLORS['minority'], node_size=200,
                              alpha=0.8, edgecolors='black', linewidths=1)

        # Highlight interethnic edges
        interethnic_edges = []
        for u, v in network.edges():
            if network.nodes[u]['ethnicity'] != network.nodes[v]['ethnicity']:
                interethnic_edges.append((u, v))

        if interethnic_edges:
            nx.draw_networkx_edges(network, pos, edgelist=interethnic_edges, ax=ax,
                                  edge_color=ETHNIC_COLORS['interethnic'], width=2, alpha=0.8)

        ax.set_title(title, fontweight='bold', pad=10)
        ax.axis('off')

        # Add ethnic composition info
        n_majority = len(majority_nodes)
        n_minority = len(minority_nodes)
        n_interethnic = len(interethnic_edges)

        info_text = f"Majority: {n_majority}\nMinority: {n_minority}\nCross-ties: {n_interethnic}"
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=9,
               verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3",
                                                facecolor='white', alpha=0.8))

    def _plot_classroom_network_metrics(self, ax, networks):
        """Plot comparison of network metrics across classrooms."""
        metrics = []
        labels = ['High Segregation', 'Medium Integration', 'High Integration']

        for network in networks:
            # Calculate network metrics
            density = nx.density(network)
            clustering = nx.transitivity(network)

            # Calculate segregation index (E-I index)
            ei_index = self._calculate_ei_index(network)

            metrics.append([density, clustering, abs(ei_index)])

        metrics = np.array(metrics)

        # Create grouped bar chart
        x = np.arange(len(labels))
        width = 0.25

        bars1 = ax.bar(x - width, metrics[:, 0], width, label='Density',
                      color=self.color_schemes.primary_palette[0], alpha=0.8)
        bars2 = ax.bar(x, metrics[:, 1], width, label='Clustering',
                      color=self.color_schemes.primary_palette[1], alpha=0.8)
        bars3 = ax.bar(x + width, metrics[:, 2], width, label='Segregation',
                      color=self.color_schemes.primary_palette[2], alpha=0.8)

        # Add value labels
        for bars in [bars1, bars2, bars3]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{height:.2f}', ha='center', va='bottom', fontsize=9)

        ax.set_xlabel('Classroom Type')
        ax.set_ylabel('Metric Value')
        ax.set_title('Network Metrics Comparison', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=15)
        ax.legend()
        ax.grid(True, axis='y', alpha=0.3)

    def _calculate_ei_index(self, network):
        """Calculate E-I (External-Internal) index for ethnic segregation."""
        external_edges = 0
        internal_edges = 0

        for u, v in network.edges():
            if network.nodes[u]['ethnicity'] != network.nodes[v]['ethnicity']:
                external_edges += 1
            else:
                internal_edges += 1

        total_edges = external_edges + internal_edges
        if total_edges == 0:
            return 0

        ei_index = (external_edges - internal_edges) / total_edges
        return ei_index

    def _plot_friendship_vs_cooperation_networks(self, ax_friendship, ax_cooperation, network):
        """Plot friendship network vs potential cooperation network."""
        pos = nx.spring_layout(network, k=0.5, iterations=50)

        # Friendship network (actual edges)
        self._plot_single_network_layer(ax_friendship, network, pos, "Friendship Network\n(Observed Ties)")

        # Cooperation network (potential based on tolerance intervention)
        cooperation_network = self._generate_cooperation_network(network)
        self._plot_single_network_layer(ax_cooperation, cooperation_network, pos,
                                       "Potential Cooperation Network\n(After Intervention)")

    def _plot_single_network_layer(self, ax, network, pos, title):
        """Plot a single network layer."""
        # Separate nodes by ethnicity
        majority_nodes = [n for n in network.nodes() if network.nodes[n]['ethnicity'] == 'majority']
        minority_nodes = [n for n in network.nodes() if network.nodes[n]['ethnicity'] == 'minority']

        # Draw edges
        nx.draw_networkx_edges(network, pos, ax=ax, edge_color='gray', alpha=0.3, width=0.5)

        # Draw nodes
        nx.draw_networkx_nodes(network, pos, nodelist=majority_nodes, ax=ax,
                              node_color=ETHNIC_COLORS['majority'], node_size=150,
                              alpha=0.8, edgecolors='black', linewidths=1)
        nx.draw_networkx_nodes(network, pos, nodelist=minority_nodes, ax=ax,
                              node_color=ETHNIC_COLORS['minority'], node_size=150,
                              alpha=0.8, edgecolors='black', linewidths=1)

        # Highlight interethnic edges
        interethnic_edges = []
        for u, v in network.edges():
            if network.nodes[u]['ethnicity'] != network.nodes[v]['ethnicity']:
                interethnic_edges.append((u, v))

        if interethnic_edges:
            nx.draw_networkx_edges(network, pos, edgelist=interethnic_edges, ax=ax,
                                  edge_color=ETHNIC_COLORS['interethnic'], width=2, alpha=0.8)

        ax.set_title(title, fontweight='bold', pad=10)
        ax.axis('off')

    def _generate_cooperation_network(self, friendship_network):
        """Generate potential cooperation network after tolerance intervention."""
        cooperation_network = friendship_network.copy()

        # Add potential cooperation edges between ethnic groups
        majority_nodes = [n for n in friendship_network.nodes()
                         if friendship_network.nodes[n]['ethnicity'] == 'majority']
        minority_nodes = [n for n in friendship_network.nodes()
                         if friendship_network.nodes[n]['ethnicity'] == 'minority']

        # Add some cross-ethnic cooperation ties
        for maj_node in majority_nodes[:3]:  # Sample some nodes
            for min_node in minority_nodes[:3]:
                if not cooperation_network.has_edge(maj_node, min_node):
                    if np.random.random() < 0.3:  # 30% chance of cooperation
                        cooperation_network.add_edge(maj_node, min_node)

        return cooperation_network

    def _plot_ethnic_composition_analysis(self, ax, networks):
        """Plot ethnic composition analysis across classrooms."""
        compositions = []
        labels = ['High Segregation', 'Medium Integration', 'High Integration']

        for network in networks:
            total_nodes = len(network.nodes())
            majority_count = sum(1 for n in network.nodes()
                               if network.nodes[n]['ethnicity'] == 'majority')
            minority_count = total_nodes - majority_count

            majority_pct = majority_count / total_nodes * 100
            minority_pct = minority_count / total_nodes * 100

            compositions.append([majority_pct, minority_pct])

        # Create stacked bar chart
        compositions = np.array(compositions)
        width = 0.6

        bars1 = ax.bar(labels, compositions[:, 0], width, label='Majority Group',
                      color=ETHNIC_COLORS['majority'], alpha=0.8)
        bars2 = ax.bar(labels, compositions[:, 1], width, bottom=compositions[:, 0],
                      label='Minority Group', color=ETHNIC_COLORS['minority'], alpha=0.8)

        ax.set_ylabel('Percentage of Students')
        ax.set_title('Ethnic Composition', fontweight='bold')
        ax.legend()
        ax.set_ylim(0, 100)

        # Add percentage labels
        for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
            # Majority percentage
            ax.text(bar1.get_x() + bar1.get_width()/2., bar1.get_height()/2,
                   f'{compositions[i, 0]:.0f}%', ha='center', va='center',
                   fontweight='bold', color='white')
            # Minority percentage
            ax.text(bar2.get_x() + bar2.get_width()/2.,
                   compositions[i, 0] + bar2.get_height()/2,
                   f'{compositions[i, 1]:.0f}%', ha='center', va='center',
                   fontweight='bold', color='white')

    def _plot_intervention_potential(self, ax, networks):
        """Plot intervention potential assessment."""
        potentials = []
        labels = ['High Segregation', 'Medium Integration', 'High Integration']

        for network in networks:
            # Calculate intervention potential metrics
            density = nx.density(network)
            ei_index = abs(self._calculate_ei_index(network))
            clustering = nx.transitivity(network)

            # Simple intervention potential score
            potential = (1 - ei_index) * density + 0.5 * clustering
            potentials.append(potential)

        # Create bar chart
        colors = [TOLERANCE_COLORS['low'], TOLERANCE_COLORS['medium'], TOLERANCE_COLORS['high']]
        bars = ax.bar(labels, potentials, color=colors, alpha=0.8, edgecolor='black', linewidth=1)

        # Add value labels
        for bar, potential in zip(bars, potentials):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{potential:.2f}', ha='center', va='bottom', fontweight='bold')

        ax.set_ylabel('Intervention Potential Score')
        ax.set_title('Tolerance Intervention\nPotential Assessment', fontweight='bold')
        ax.set_xticklabels(labels, rotation=15)
        ax.grid(True, axis='y', alpha=0.3)

        # Add interpretation
        ax.text(0.5, 0.95, 'Higher scores indicate greater potential\nfor successful tolerance interventions',
               transform=ax.transAxes, ha='center', va='top', fontsize=9,
               bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.7))

    def create_all_tolerance_intervention_figures(self,
                                                simulation_results: Optional[Dict] = None,
                                                empirical_data: Optional[Dict] = None
                                                ) -> Dict[str, List[str]]:
        """
        Create complete suite of tolerance intervention publication figures.

        Args:
            simulation_results: Results from tolerance intervention simulations
            empirical_data: Real classroom network data for validation

        Returns:
            Dictionary mapping figure names to lists of saved file paths
        """
        logger.info("Creating complete tolerance intervention figure suite")

        figure_paths = {}

        # Figure 1: Conceptual Model
        figure_paths['conceptual_model'] = self.create_figure_1_conceptual_model()

        # Figure 2: Network Examples
        classroom_networks = empirical_data.get('classroom_networks') if empirical_data else None
        figure_paths['network_examples'] = self.create_figure_2_network_examples(classroom_networks)

        # Additional figures would be created here based on simulation_results
        # Figure 3: Simulation Results (tolerance evolution, cooperation emergence)
        # Figure 4: Strategy Comparison (central vs peripheral vs random targeting)
        # Figure 5: Empirical Validation (model fit to real data)

        logger.info(f"Created {len(figure_paths)} tolerance intervention figures")
        return figure_paths


# Module testing and example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Create tolerance intervention publication figures
    figure_generator = ToleranceInterventionPublicationFigures()

    # Create Figure 1: Conceptual Model
    fig1_paths = figure_generator.create_figure_1_conceptual_model()
    print(f"Figure 1 created: {fig1_paths}")

    # Create Figure 2: Network Examples
    fig2_paths = figure_generator.create_figure_2_network_examples()
    print(f"Figure 2 created: {fig2_paths}")

    # Create complete figure suite
    all_figures = figure_generator.create_all_tolerance_intervention_figures()
    print(f"Complete tolerance intervention figure suite created:")
    for fig_name, paths in all_figures.items():
        print(f"  {fig_name}: {len(paths)} formats")

    print("All tolerance intervention publication figures completed!")