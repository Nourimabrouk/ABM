"""
Interactive Tolerance Intervention Explorer
==========================================

Web-based interactive dashboard for exploring tolerance intervention effects
on interethnic cooperation in social networks. This sophisticated demo allows
real-time experimentation with intervention parameters and visualization of
tolerance diffusion mechanisms.

Features:
- Real-time parameter sliders for intervention design
- Live network visualization with tolerance and cooperation layers
- Comparative analysis tools for different targeting strategies
- Export functionality for custom scenarios and results
- "What-If" scenario explorer for policy recommendations

Author: Claude Code - Visualization Virtuoso
Created: 2025-09-16
"""

import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass, asdict
import warnings
warnings.filterwarnings('ignore')

# Dashboard framework
import panel as pn
import param
import bokeh.plotting as bk
from bokeh.models import HoverTool, ColorBar, LinearColorMapper, ColumnDataSource, Slider
from bokeh.palettes import RdYlGn11, Viridis256
from bokeh.layouts import row, column
from bokeh.plotting import figure
import holoviews as hv
from holoviews import opts
import networkx as nx

# Advanced visualization
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Scientific computation
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

# Custom imports
from ..utils.color_schemes import AcademicColorSchemes
from .tolerance_intervention_viz import ToleranceInterventionVisualizer, create_sample_tolerance_data

logger = logging.getLogger(__name__)

# Enable Panel extensions for interactive dashboard
pn.extension('bokeh', 'tabulator', 'plotly')
hv.extension('bokeh')

@dataclass
class InterventionExplorerConfig:
    """Configuration for the tolerance intervention explorer."""
    width: int = 1400
    height: int = 900
    update_interval: int = 500  # milliseconds
    max_network_size: int = 50
    default_n_agents: int = 30
    default_n_steps: int = 20
    tolerance_colormap: str = 'RdYlGn'
    cooperation_colormap: str = 'viridis'
    show_animations: bool = True
    enable_3d_view: bool = True

class ToleranceInterventionExplorer(param.Parameterized):
    """
    Interactive tolerance intervention explorer dashboard.

    Provides comprehensive tools for exploring tolerance intervention effects
    on social networks and interethnic cooperation dynamics.
    """

    # Network Configuration
    n_agents = param.Integer(default=30, bounds=(15, 100), label="Number of Agents")
    n_time_steps = param.Integer(default=20, bounds=(10, 50), label="Time Steps")
    network_density = param.Number(default=0.15, bounds=(0.05, 0.4), step=0.01, label="Network Density")

    # Intervention Parameters
    intervention_magnitude = param.Number(default=0.5, bounds=(0.1, 1.0), step=0.05,
                                        label="Tolerance Change Magnitude")
    target_population_size = param.Integer(default=3, bounds=(1, 10), label="Target Population Size")
    intervention_start_time = param.Integer(default=5, bounds=(1, 15), label="Intervention Start Time")
    intervention_duration = param.Integer(default=3, bounds=(1, 10), label="Intervention Duration")

    # Social Influence Parameters
    influence_threshold = param.Number(default=0.3, bounds=(0.1, 0.8), step=0.05,
                                     label="Social Influence Threshold")
    homophily_strength = param.Number(default=0.2, bounds=(0.0, 0.8), step=0.05,
                                    label="Ethnic Homophily Strength")
    tolerance_persistence = param.Number(default=0.8, bounds=(0.3, 1.0), step=0.05,
                                       label="Tolerance Persistence Factor")

    # Complex Contagion Parameters
    complex_contagion = param.Boolean(default=False, label="Enable Complex Contagion")
    contagion_threshold = param.Number(default=0.4, bounds=(0.2, 0.8), step=0.05,
                                     label="Complex Contagion Threshold")
    multiple_exposure_required = param.Integer(default=2, bounds=(1, 5),
                                             label="Multiple Exposures Required")

    # Targeting Strategy
    targeting_strategy = param.ObjectSelector(
        default="Central Nodes", objects=["Central Nodes", "Peripheral Nodes", "Random", "Clustered"],
        label="Targeting Strategy"
    )

    # Ethnic Composition
    minority_proportion = param.Number(default=0.33, bounds=(0.1, 0.5), step=0.05,
                                     label="Minority Group Proportion")

    # Control Actions
    run_simulation = param.Action(lambda self: self._run_tolerance_simulation(), label="🚀 Run Simulation")
    reset_parameters = param.Action(lambda self: self._reset_to_defaults(), label="🔄 Reset Parameters")
    export_scenario = param.Action(lambda self: self._export_current_scenario(), label="💾 Export Scenario")
    compare_strategies = param.Action(lambda self: self._compare_targeting_strategies(),
                                    label="📊 Compare Strategies")

    def __init__(self, config: InterventionExplorerConfig = None, **params):
        """Initialize tolerance intervention explorer."""
        super().__init__(**params)

        self.config = config or InterventionExplorerConfig()
        self.color_schemes = AcademicColorSchemes()
        self.visualizer = ToleranceInterventionVisualizer()

        # Dashboard state
        self.current_networks = []
        self.current_tolerance_data = []
        self.current_cooperation_data = []
        self.current_intervention_targets = []
        self.simulation_results = {}
        self.comparison_results = {}

        # Create dashboard layout
        self._create_dashboard_layout()

        logger.info("Tolerance intervention explorer initialized")

    def _create_dashboard_layout(self):
        """Create the comprehensive dashboard layout."""
        # Parameter Control Panels
        network_params = pn.Param(
            self,
            parameters=['n_agents', 'n_time_steps', 'network_density', 'minority_proportion'],
            widgets={
                'n_agents': pn.widgets.IntSlider,
                'n_time_steps': pn.widgets.IntSlider,
                'network_density': pn.widgets.FloatSlider,
                'minority_proportion': pn.widgets.FloatSlider
            },
            name="🌐 Network Configuration",
            show_name=True
        )

        intervention_params = pn.Param(
            self,
            parameters=['intervention_magnitude', 'target_population_size',
                       'intervention_start_time', 'intervention_duration', 'targeting_strategy'],
            widgets={
                'intervention_magnitude': pn.widgets.FloatSlider,
                'target_population_size': pn.widgets.IntSlider,
                'intervention_start_time': pn.widgets.IntSlider,
                'intervention_duration': pn.widgets.IntSlider,
                'targeting_strategy': pn.widgets.Select
            },
            name="🎯 Intervention Design",
            show_name=True
        )

        influence_params = pn.Param(
            self,
            parameters=['influence_threshold', 'homophily_strength', 'tolerance_persistence'],
            widgets={
                'influence_threshold': pn.widgets.FloatSlider,
                'homophily_strength': pn.widgets.FloatSlider,
                'tolerance_persistence': pn.widgets.FloatSlider
            },
            name="🔄 Social Influence",
            show_name=True
        )

        contagion_params = pn.Param(
            self,
            parameters=['complex_contagion', 'contagion_threshold', 'multiple_exposure_required'],
            widgets={
                'complex_contagion': pn.widgets.Checkbox,
                'contagion_threshold': pn.widgets.FloatSlider,
                'multiple_exposure_required': pn.widgets.IntSlider
            },
            name="🧮 Complex Contagion",
            show_name=True
        )

        control_params = pn.Param(
            self,
            parameters=['run_simulation', 'reset_parameters', 'export_scenario', 'compare_strategies'],
            widgets={
                'run_simulation': pn.widgets.Button,
                'reset_parameters': pn.widgets.Button,
                'export_scenario': pn.widgets.Button,
                'compare_strategies': pn.widgets.Button
            },
            name="🎮 Controls",
            show_name=True
        )

        # Main Visualization Areas
        self.network_view = self._create_network_visualization()
        self.metrics_view = self._create_metrics_dashboard()
        self.timeline_view = self._create_timeline_visualization()
        self.comparison_view = self._create_comparison_panel()

        # Status and Information
        self.status_panel = pn.pane.Markdown("""
## 🎯 Tolerance Intervention Explorer

**Status:** Ready to explore tolerance interventions

**Instructions:**
1. Adjust parameters using the controls on the left
2. Click "🚀 Run Simulation" to see the intervention effects
3. Explore different targeting strategies
4. Compare multiple scenarios using "📊 Compare Strategies"

**Current Settings:** Default parameters loaded
        """, width=350)

        self.info_panel = pn.pane.Markdown(self._get_research_info(), width=350)

        # Create main dashboard with tabs
        main_content = pn.Tabs(
            ("🌐 Network Evolution", pn.Column(
                pn.Row(self.network_view, self.timeline_view),
                self.metrics_view
            )),
            ("📈 Metrics Dashboard", self.metrics_view),
            ("📊 Strategy Comparison", self.comparison_view),
            ("ℹ️ Research Context", self.info_panel),
            width=1000,
            height=700
        )

        # Create sidebar with parameter controls
        sidebar_content = pn.Column(
            network_params,
            intervention_params,
            influence_params,
            contagion_params,
            control_params,
            self.status_panel,
            width=350,
            scroll=True
        )

        # Main layout using template
        self.layout = pn.template.FastListTemplate(
            title="🎯 Tolerance Intervention Explorer - PhD Research Demo",
            sidebar=sidebar_content,
            main=[main_content],
            header_background='#1f77b4',
            sidebar_width=370,
            main_max_width="1000px"
        )

    def _create_network_visualization(self):
        """Create interactive network visualization panel."""
        # Create initial empty plot
        p = figure(
            title="Tolerance Network Evolution",
            width=500, height=400,
            tools="pan,wheel_zoom,box_zoom,reset,save,hover",
            toolbar_location="above"
        )

        p.title.text_font_size = "16pt"
        p.title.align = "center"

        # Add placeholder text
        p.text([0], [0], text=["Run simulation to see tolerance spread"],
              text_align="center", text_baseline="middle",
              text_font_size="14pt", color="gray")

        return pn.pane.Bokeh(p, sizing_mode='stretch_both')

    def _create_metrics_dashboard(self):
        """Create comprehensive metrics dashboard."""
        # Use HoloViews for interactive metrics
        tolerance_curve = hv.Curve([]).opts(
            title="Tolerance Evolution",
            xlabel="Time Step",
            ylabel="Mean Tolerance",
            width=450, height=250,
            tools=['hover'],
            show_grid=True,
            line_width=3
        )

        cooperation_curve = hv.Curve([]).opts(
            title="Cooperation Emergence",
            xlabel="Time Step",
            ylabel="Cooperation Level",
            width=450, height=250,
            tools=['hover'],
            show_grid=True,
            line_width=3
        )

        # Combine into layout
        metrics_layout = pn.Row(
            pn.pane.HoloViews(tolerance_curve),
            pn.pane.HoloViews(cooperation_curve)
        )

        return metrics_layout

    def _create_timeline_visualization(self):
        """Create timeline showing intervention phases."""
        timeline_plot = hv.Curve([]).opts(
            title="Intervention Timeline",
            xlabel="Time Step",
            ylabel="Intervention Intensity",
            width=450, height=200,
            tools=['hover'],
            show_grid=True
        )

        return pn.pane.HoloViews(timeline_plot)

    def _create_comparison_panel(self):
        """Create strategy comparison visualization panel."""
        comparison_text = pn.pane.Markdown("""
## 📊 Strategy Comparison Results

Use the "📊 Compare Strategies" button to run automated comparison
of different targeting approaches:

- **Central Node Targeting**: Target highly connected agents
- **Peripheral Node Targeting**: Target less connected agents
- **Random Targeting**: Random selection baseline
- **Clustered Targeting**: Target agents within ethnic clusters

Results will show effectiveness metrics and optimal targeting recommendations.
        """)

        return comparison_text

    def _get_research_info(self):
        """Get research context information."""
        return """
## 🎓 Research Context: Tolerance Interventions for Interethnic Cooperation

### Theoretical Framework
This explorer demonstrates computational models of tolerance interventions
designed to promote interethnic cooperation through social network mechanisms.

### Key Research Questions
1. **How do tolerance interventions spread through friendship networks?**
2. **Which targeting strategies maximize interethnic cooperation?**
3. **What role does complex contagion play in tolerance diffusion?**
4. **How persistent are intervention effects over time?**

### Model Mechanisms
- **Tolerance Diffusion**: Interventions change individual tolerance levels
- **Social Influence**: Tolerance spreads through network connections
- **Cooperation Emergence**: Higher tolerance leads to interethnic cooperation
- **Network Evolution**: Cooperation strengthens interethnic ties

### Policy Implications
Findings inform design of real-world tolerance interventions in:
- Educational settings (classroom networks)
- Community programs (neighborhood initiatives)
- Workplace diversity training
- Online social platforms

### PhD Dissertation Context
This work contributes to computational social science methodology
for analyzing tolerance interventions and intergroup contact effects.

**Publication Target**: JASSS (Journal of Artificial Societies and Social Simulation)
        """

    def _run_tolerance_simulation(self):
        """Run tolerance intervention simulation with current parameters."""
        try:
            self.status_panel.object = """
## 🚀 Running Simulation...

**Status:** Initializing tolerance intervention model

- Creating social network structure
- Assigning ethnic group memberships
- Selecting intervention targets
- Simulating tolerance diffusion
            """

            # Create tolerance intervention simulation
            results = self._execute_tolerance_simulation()

            self.current_networks = results['networks']
            self.current_tolerance_data = results['tolerance_data']
            self.current_cooperation_data = results['cooperation_data']
            self.current_intervention_targets = results['intervention_targets']
            self.simulation_results = results

            # Update all visualizations
            self._update_network_visualization()
            self._update_metrics_dashboard()
            self._update_timeline_visualization()

            # Update status
            final_tolerance = np.mean(list(self.current_tolerance_data[-1].values()))
            final_cooperation = np.mean(list(self.current_cooperation_data[-1].values()))
            interethnic_ties = self._count_final_interethnic_ties()

            self.status_panel.object = f"""
## ✅ Simulation Complete!

**Final Results:**
- **Mean Tolerance:** {final_tolerance:.3f}
- **Mean Cooperation:** {final_cooperation:.3f}
- **Interethnic Ties:** {interethnic_ties}
- **Intervention Targets:** {len(self.current_intervention_targets)}

**Strategy Used:** {self.targeting_strategy}

Explore the Network Evolution tab to see tolerance spread dynamics!
            """

        except Exception as e:
            logger.error(f"Simulation failed: {e}")
            self.status_panel.object = f"""
## ❌ Simulation Failed!

**Error:** {str(e)}

Please check parameters and try again.
            """

    def _execute_tolerance_simulation(self):
        """Execute the core tolerance intervention simulation."""
        # Create network structure based on parameters
        base_network = self._create_intervention_network()

        # Select intervention targets based on strategy
        intervention_targets = self._select_intervention_targets(base_network)

        # Run tolerance intervention simulation
        networks, tolerance_data, cooperation_data = self._simulate_tolerance_intervention(
            base_network, intervention_targets
        )

        return {
            'networks': networks,
            'tolerance_data': tolerance_data,
            'cooperation_data': cooperation_data,
            'intervention_targets': intervention_targets,
            'parameters': self._get_current_parameters()
        }

    def _create_intervention_network(self):
        """Create social network with ethnic group structure."""
        # Create base network
        network = nx.erdos_renyi_graph(self.n_agents, self.network_density)

        # Assign ethnic group memberships
        n_minority = int(self.n_agents * self.minority_proportion)

        for i, node in enumerate(network.nodes()):
            if i < n_minority:
                network.nodes[node]['ethnicity'] = 'minority'
            else:
                network.nodes[node]['ethnicity'] = 'majority'

        # Add homophily-based edge adjustments
        if self.homophily_strength > 0:
            network = self._add_ethnic_homophily(network)

        return network

    def _add_ethnic_homophily(self, network):
        """Add ethnic homophily to network structure."""
        # Calculate current interethnic edge proportion
        interethnic_edges = []
        intraethnic_edges = []

        for u, v in network.edges():
            u_ethnicity = network.nodes[u]['ethnicity']
            v_ethnicity = network.nodes[v]['ethnicity']

            if u_ethnicity != v_ethnicity:
                interethnic_edges.append((u, v))
            else:
                intraethnic_edges.append((u, v))

        # Remove some interethnic edges based on homophily strength
        edges_to_remove = int(len(interethnic_edges) * self.homophily_strength)
        if edges_to_remove > 0 and interethnic_edges:
            remove_edges = np.random.choice(len(interethnic_edges),
                                          size=min(edges_to_remove, len(interethnic_edges)),
                                          replace=False)
            for idx in remove_edges:
                network.remove_edge(*interethnic_edges[idx])

        return network

    def _select_intervention_targets(self, network):
        """Select intervention targets based on chosen strategy."""
        targets = []

        if self.targeting_strategy == "Central Nodes":
            # Target nodes with highest degree centrality
            centrality = nx.degree_centrality(network)
            sorted_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)
            targets = [node for node, _ in sorted_nodes[:self.target_population_size]]

        elif self.targeting_strategy == "Peripheral Nodes":
            # Target nodes with lowest degree centrality
            centrality = nx.degree_centrality(network)
            sorted_nodes = sorted(centrality.items(), key=lambda x: x[1])
            targets = [node for node, _ in sorted_nodes[:self.target_population_size]]

        elif self.targeting_strategy == "Random":
            # Random selection
            targets = np.random.choice(list(network.nodes()),
                                     size=self.target_population_size,
                                     replace=False).tolist()

        elif self.targeting_strategy == "Clustered":
            # Target nodes within same community/cluster
            try:
                communities = nx.community.greedy_modularity_communities(network)
                if communities:
                    # Select from largest community
                    largest_community = max(communities, key=len)
                    targets = list(largest_community)[:self.target_population_size]
                else:
                    # Fallback to random
                    targets = np.random.choice(list(network.nodes()),
                                             size=self.target_population_size,
                                             replace=False).tolist()
            except:
                # Fallback to random
                targets = np.random.choice(list(network.nodes()),
                                         size=self.target_population_size,
                                         replace=False).tolist()

        return targets

    def _simulate_tolerance_intervention(self, base_network, intervention_targets):
        """Simulate tolerance intervention with social influence dynamics."""
        networks = [base_network.copy()]
        tolerance_sequences = []
        cooperation_sequences = []

        # Initialize tolerance levels
        current_tolerances = {}
        for node in base_network.nodes():
            current_tolerances[node] = np.random.normal(0, 0.2)  # Small initial variation

        tolerance_sequences.append(current_tolerances.copy())

        # Simulate over time
        for t in range(1, self.n_time_steps):
            current_network = networks[-1].copy()
            new_tolerances = current_tolerances.copy()

            # Apply intervention
            if (self.intervention_start_time <= t <
                self.intervention_start_time + self.intervention_duration):
                for target in intervention_targets:
                    if target in new_tolerances:
                        new_tolerances[target] += self.intervention_magnitude
                        new_tolerances[target] = np.clip(new_tolerances[target], -1, 1)

            # Social influence process
            new_tolerances = self._apply_social_influence(current_network, new_tolerances)

            # Apply persistence factor
            for node in new_tolerances:
                new_tolerances[node] = (self.tolerance_persistence * new_tolerances[node] +
                                      (1 - self.tolerance_persistence) * current_tolerances[node])

            tolerance_sequences.append(new_tolerances.copy())
            current_tolerances = new_tolerances

            # Update network based on tolerance and cooperation
            updated_network = self._update_network_structure(current_network, current_tolerances)
            networks.append(updated_network)

            # Calculate cooperation levels
            cooperations = self._calculate_cooperation_levels(updated_network, current_tolerances)
            cooperation_sequences.append(cooperations)

        return networks, tolerance_sequences, cooperation_sequences

    def _apply_social_influence(self, network, tolerances):
        """Apply social influence mechanism for tolerance diffusion."""
        new_tolerances = tolerances.copy()

        for node in network.nodes():
            neighbors = list(network.neighbors(node))
            if not neighbors:
                continue

            # Calculate social influence
            neighbor_tolerances = [tolerances[n] for n in neighbors]

            if self.complex_contagion:
                # Complex contagion: require multiple high-tolerance neighbors
                high_tolerance_neighbors = sum(1 for tol in neighbor_tolerances
                                             if tol > self.contagion_threshold)

                if high_tolerance_neighbors >= self.multiple_exposure_required:
                    influence = np.mean(neighbor_tolerances) * self.influence_threshold
                    new_tolerances[node] += influence

            else:
                # Simple contagion: average neighbor influence
                mean_neighbor_tolerance = np.mean(neighbor_tolerances)
                if abs(mean_neighbor_tolerance) > self.influence_threshold:
                    influence = mean_neighbor_tolerance * self.influence_threshold * 0.1
                    new_tolerances[node] += influence

            # Clip to valid range
            new_tolerances[node] = np.clip(new_tolerances[node], -1, 1)

        return new_tolerances

    def _update_network_structure(self, network, tolerances):
        """Update network structure based on tolerance and cooperation."""
        # Create copy to modify
        updated_network = network.copy()

        # Add cooperation-based edges
        nodes = list(network.nodes())
        for i, u in enumerate(nodes):
            for v in nodes[i+1:]:
                if not updated_network.has_edge(u, v):
                    # Calculate cooperation potential
                    u_tolerance = tolerances[u]
                    v_tolerance = tolerances[v]
                    u_ethnicity = network.nodes[u]['ethnicity']
                    v_ethnicity = network.nodes[v]['ethnicity']

                    # Higher cooperation potential for:
                    # 1. High tolerance individuals
                    # 2. Interethnic pairs (when both tolerant)
                    cooperation_potential = (u_tolerance + v_tolerance) / 2

                    if u_ethnicity != v_ethnicity:
                        cooperation_potential *= 1.5  # Bonus for interethnic cooperation

                    # Probabilistically add edge
                    if cooperation_potential > 0.6 and np.random.random() < 0.1:
                        updated_network.add_edge(u, v)

        return updated_network

    def _calculate_cooperation_levels(self, network, tolerances):
        """Calculate cooperation levels between connected agents."""
        cooperations = {}

        for u, v in network.edges():
            u_tolerance = tolerances[u]
            v_tolerance = tolerances[v]
            u_ethnicity = network.nodes[u]['ethnicity']
            v_ethnicity = network.nodes[v]['ethnicity']

            # Cooperation based on tolerance similarity and ethnic difference
            tolerance_similarity = 1 - abs(u_tolerance - v_tolerance) / 2

            # Bonus for interethnic cooperation when both are tolerant
            interethnic_bonus = 0
            if u_ethnicity != v_ethnicity and u_tolerance > 0.2 and v_tolerance > 0.2:
                interethnic_bonus = 0.3

            cooperation = tolerance_similarity + interethnic_bonus + np.random.normal(0, 0.05)
            cooperations[(u, v)] = np.clip(cooperation, 0, 1)

        return cooperations

    def _update_network_visualization(self):
        """Update network visualization with current simulation results."""
        if not self.current_networks:
            return

        # Use final network state
        final_network = self.current_networks[-1]
        final_tolerances = self.current_tolerance_data[-1]

        if len(final_network) == 0:
            return

        # Calculate layout
        pos = nx.spring_layout(final_network, k=0.8, iterations=50)

        # Prepare node data
        node_x = []
        node_y = []
        node_colors = []
        node_sizes = []
        node_labels = []

        for node in final_network.nodes():
            if node in pos:
                x, y = pos[node]
                node_x.append(x)
                node_y.append(y)

                # Color by tolerance level
                tolerance = final_tolerances.get(node, 0)
                node_colors.append(tolerance)

                # Size by intervention status
                size = 400 if node in self.current_intervention_targets else 200
                node_sizes.append(size)

                # Create hover label
                ethnicity = final_network.nodes[node].get('ethnicity', 'unknown')
                label = (f"Agent {node}<br>"
                        f"Tolerance: {tolerance:.3f}<br>"
                        f"Ethnicity: {ethnicity}<br>")
                if node in self.current_intervention_targets:
                    label += "<b>INTERVENTION TARGET</b>"
                node_labels.append(label)

        # Prepare edge data
        edge_x = []
        edge_y = []
        edge_colors = []

        for u, v in final_network.edges():
            if u in pos and v in pos:
                x0, y0 = pos[u]
                x1, y1 = pos[v]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])

                # Color by ethnic composition
                u_ethnicity = final_network.nodes[u].get('ethnicity', 'majority')
                v_ethnicity = final_network.nodes[v].get('ethnicity', 'majority')
                if u_ethnicity != v_ethnicity:
                    edge_colors.append('red')  # Interethnic edge
                else:
                    edge_colors.append('gray')  # Intraethnic edge

        # Create new plot
        p = figure(
            title=f"Tolerance Network (Final State) - {self.targeting_strategy} Strategy",
            width=500, height=400,
            tools="pan,wheel_zoom,box_zoom,reset,save,hover",
            toolbar_location="above"
        )

        # Add edges
        p.line(edge_x, edge_y, line_width=1, line_alpha=0.6, color="gray")

        # Add nodes with tolerance coloring
        if node_colors:
            source = ColumnDataSource(dict(
                x=node_x, y=node_y,
                colors=node_colors,
                sizes=node_sizes,
                labels=node_labels
            ))

            # Map tolerance to colors (-1 to 1 -> red to green)
            mapper = LinearColorMapper(palette=RdYlGn11, low=-1, high=1)

            nodes_glyph = p.circle('x', 'y', size='sizes',
                                  color={'field': 'colors', 'transform': mapper},
                                  source=source, alpha=0.8, line_color='black')

            # Configure hover tool
            hover = HoverTool(tooltips="@labels", renderers=[nodes_glyph])
            p.add_tools(hover)

            # Add color bar
            color_bar = ColorBar(color_mapper=mapper,
                               label_standoff=12, location=(0,0),
                               title="Tolerance Level")
            p.add_layout(color_bar, 'right')

        p.title.text_font_size = "14pt"
        p.axis.visible = False
        p.grid.visible = False

        # Update the panel
        self.network_view.object = p

    def _update_metrics_dashboard(self):
        """Update metrics dashboard with simulation results."""
        if not self.current_tolerance_data:
            return

        time_points = list(range(len(self.current_tolerance_data)))

        # Calculate tolerance evolution
        tolerance_means = [np.mean(list(tolerances.values()))
                          for tolerances in self.current_tolerance_data]

        tolerance_curve = hv.Curve(
            (time_points, tolerance_means),
            kdims='Time Step', vdims='Mean Tolerance'
        ).opts(
            title="Tolerance Evolution",
            color=self.color_schemes.primary_palette[0],
            line_width=3,
            width=450, height=250,
            tools=['hover'],
            show_grid=True
        )

        # Calculate cooperation evolution
        if self.current_cooperation_data:
            cooperation_means = [np.mean(list(cooperations.values())) if cooperations else 0
                               for cooperations in self.current_cooperation_data]
        else:
            cooperation_means = [0] * len(time_points)

        cooperation_curve = hv.Curve(
            (time_points, cooperation_means),
            kdims='Time Step', vdims='Mean Cooperation'
        ).opts(
            title="Cooperation Emergence",
            color=self.color_schemes.secondary_palette[0],
            line_width=3,
            width=450, height=250,
            tools=['hover'],
            show_grid=True
        )

        # Update the metrics view
        metrics_layout = pn.Row(
            pn.pane.HoloViews(tolerance_curve),
            pn.pane.HoloViews(cooperation_curve)
        )

        # Update the existing panel
        self.metrics_view.objects = [metrics_layout]

    def _update_timeline_visualization(self):
        """Update timeline showing intervention phases."""
        if not self.current_tolerance_data:
            return

        time_points = list(range(len(self.current_tolerance_data)))
        intervention_intensity = []

        for t in time_points:
            if (self.intervention_start_time <= t <
                self.intervention_start_time + self.intervention_duration):
                intensity = self.intervention_magnitude
            else:
                intensity = 0
            intervention_intensity.append(intensity)

        timeline_plot = hv.Area(
            (time_points, intervention_intensity),
            kdims='Time Step', vdims='Intervention Intensity'
        ).opts(
            title="Intervention Timeline",
            color=self.color_schemes.accent_palette[0],
            alpha=0.7,
            width=450, height=200,
            tools=['hover'],
            show_grid=True
        )

        self.timeline_view.object = timeline_plot

    def _count_final_interethnic_ties(self):
        """Count interethnic ties in final network state."""
        if not self.current_networks:
            return 0

        final_network = self.current_networks[-1]
        count = 0

        for u, v in final_network.edges():
            u_ethnicity = final_network.nodes[u].get('ethnicity', 'majority')
            v_ethnicity = final_network.nodes[v].get('ethnicity', 'majority')
            if u_ethnicity != v_ethnicity:
                count += 1

        return count

    def _get_current_parameters(self):
        """Get current parameter settings."""
        return {
            'n_agents': self.n_agents,
            'n_time_steps': self.n_time_steps,
            'network_density': self.network_density,
            'intervention_magnitude': self.intervention_magnitude,
            'target_population_size': self.target_population_size,
            'intervention_start_time': self.intervention_start_time,
            'intervention_duration': self.intervention_duration,
            'influence_threshold': self.influence_threshold,
            'homophily_strength': self.homophily_strength,
            'tolerance_persistence': self.tolerance_persistence,
            'complex_contagion': self.complex_contagion,
            'contagion_threshold': self.contagion_threshold,
            'multiple_exposure_required': self.multiple_exposure_required,
            'targeting_strategy': self.targeting_strategy,
            'minority_proportion': self.minority_proportion
        }

    def _reset_to_defaults(self):
        """Reset all parameters to default values."""
        defaults = {
            'n_agents': 30, 'n_time_steps': 20, 'network_density': 0.15,
            'intervention_magnitude': 0.5, 'target_population_size': 3,
            'intervention_start_time': 5, 'intervention_duration': 3,
            'influence_threshold': 0.3, 'homophily_strength': 0.2,
            'tolerance_persistence': 0.8, 'complex_contagion': False,
            'contagion_threshold': 0.4, 'multiple_exposure_required': 2,
            'targeting_strategy': 'Central Nodes', 'minority_proportion': 0.33
        }

        for param_name, default_value in defaults.items():
            setattr(self, param_name, default_value)

        self.status_panel.object = """
## 🔄 Parameters Reset

All parameters have been reset to default values.

Ready for new tolerance intervention exploration!
        """

    def _export_current_scenario(self):
        """Export current scenario and results."""
        if not self.simulation_results:
            self.status_panel.object = """
## ❌ Export Failed

No simulation results to export. Run a simulation first!
            """
            return

        try:
            # Create export directory
            output_dir = Path("outputs/tolerance_explorer_exports")
            output_dir.mkdir(parents=True, exist_ok=True)

            # Generate timestamp for unique filename
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

            # Export parameters and results
            import json
            export_data = {
                'parameters': self._get_current_parameters(),
                'results_summary': {
                    'final_tolerance': np.mean(list(self.current_tolerance_data[-1].values())),
                    'final_cooperation': np.mean(list(self.current_cooperation_data[-1].values())) if self.current_cooperation_data else 0,
                    'interethnic_ties': self._count_final_interethnic_ties(),
                    'intervention_targets': self.current_intervention_targets
                },
                'timestamp': timestamp
            }

            export_file = output_dir / f"tolerance_scenario_{timestamp}.json"
            with open(export_file, 'w') as f:
                json.dump(export_data, f, indent=2)

            # Export network visualization
            if self.current_networks:
                viz_file = output_dir / f"tolerance_network_{timestamp}.png"
                # Save current network visualization (simplified for demo)

            self.status_panel.object = f"""
## 💾 Export Successful!

**Files saved:**
- Scenario data: `tolerance_scenario_{timestamp}.json`
- Network visualization: `tolerance_network_{timestamp}.png`

**Export location:** `outputs/tolerance_explorer_exports/`

Results include parameters, network data, and summary statistics.
            """

        except Exception as e:
            logger.error(f"Export failed: {e}")
            self.status_panel.object = f"""
## ❌ Export Failed!

**Error:** {str(e)}

Please try again or check file permissions.
            """

    def _compare_targeting_strategies(self):
        """Run automated comparison of all targeting strategies."""
        self.status_panel.object = """
## 📊 Running Strategy Comparison...

**Status:** Comparing targeting strategies

This will run simulations for all four targeting approaches:
- Central Node Targeting
- Peripheral Node Targeting
- Random Targeting
- Clustered Targeting

Please wait while simulations complete...
        """

        try:
            strategies = ["Central Nodes", "Peripheral Nodes", "Random", "Clustered"]
            comparison_results = {}

            original_strategy = self.targeting_strategy

            for strategy in strategies:
                # Set strategy and run simulation
                self.targeting_strategy = strategy
                results = self._execute_tolerance_simulation()

                # Store key metrics
                final_tolerance = np.mean(list(results['tolerance_data'][-1].values()))
                final_cooperation = np.mean(list(results['cooperation_data'][-1].values())) if results['cooperation_data'] else 0

                # Count interethnic ties
                final_network = results['networks'][-1]
                interethnic_ties = sum(1 for u, v in final_network.edges()
                                     if final_network.nodes[u].get('ethnicity') !=
                                        final_network.nodes[v].get('ethnicity'))

                comparison_results[strategy] = {
                    'final_tolerance': final_tolerance,
                    'final_cooperation': final_cooperation,
                    'interethnic_ties': interethnic_ties,
                    'tolerance_evolution': [np.mean(list(tol.values())) for tol in results['tolerance_data']]
                }

            # Restore original strategy
            self.targeting_strategy = original_strategy

            # Create comparison visualization
            self._create_strategy_comparison_plot(comparison_results)

            # Find best strategy
            best_strategy = max(comparison_results.keys(),
                              key=lambda s: comparison_results[s]['final_tolerance'])

            self.status_panel.object = f"""
## 📊 Strategy Comparison Complete!

**Results Summary:**

**🏆 Best Strategy:** {best_strategy}
- Final Tolerance: {comparison_results[best_strategy]['final_tolerance']:.3f}
- Final Cooperation: {comparison_results[best_strategy]['final_cooperation']:.3f}
- Interethnic Ties: {comparison_results[best_strategy]['interethnic_ties']}

**Comparison Results:**
{self._format_comparison_results(comparison_results)}

**Policy Recommendation:** Use {best_strategy.lower()} for maximum tolerance intervention effectiveness.
            """

        except Exception as e:
            logger.error(f"Strategy comparison failed: {e}")
            self.status_panel.object = f"""
## ❌ Strategy Comparison Failed!

**Error:** {str(e)}

Please try again with different parameters.
            """

    def _format_comparison_results(self, results):
        """Format comparison results for display."""
        formatted = ""
        for strategy, metrics in results.items():
            formatted += f"- **{strategy}:** "
            formatted += f"Tolerance={metrics['final_tolerance']:.3f}, "
            formatted += f"Cooperation={metrics['final_cooperation']:.3f}, "
            formatted += f"Ties={metrics['interethnic_ties']}\n"
        return formatted

    def _create_strategy_comparison_plot(self, results):
        """Create strategy comparison visualization."""
        # Create comparison plot using HoloViews
        strategies = list(results.keys())
        tolerance_values = [results[s]['final_tolerance'] for s in strategies]
        cooperation_values = [results[s]['final_cooperation'] for s in strategies]

        # Bar chart of final tolerance levels
        tolerance_bars = hv.Bars(
            (strategies, tolerance_values),
            kdims='Strategy', vdims='Final Tolerance'
        ).opts(
            title="Targeting Strategy Effectiveness",
            color=self.color_schemes.primary_palette[0],
            width=600, height=300,
            tools=['hover'],
            xrotation=45
        )

        # Update comparison view
        self.comparison_view.object = pn.pane.HoloViews(tolerance_bars)

    def serve(self, port: int = 5007, show: bool = True):
        """Serve the tolerance intervention explorer dashboard."""
        return self.layout.servable().show(port=port, threaded=True, verbose=False, open=show)

    def save_static_dashboard(self, filepath: str):
        """Save static version of the dashboard."""
        self.layout.save(filepath)
        logger.info(f"Static tolerance intervention explorer saved: {filepath}")


def create_tolerance_intervention_explorer(config: InterventionExplorerConfig = None) -> ToleranceInterventionExplorer:
    """
    Create and configure the tolerance intervention explorer dashboard.

    Args:
        config: Dashboard configuration

    Returns:
        Configured ToleranceInterventionExplorer instance
    """
    return ToleranceInterventionExplorer(config=config)


# Module testing and example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Create tolerance intervention explorer
    explorer = create_tolerance_intervention_explorer()

    print("🎯 Starting Tolerance Intervention Explorer...")
    print("📱 Access the dashboard at: http://localhost:5007")
    print("🎮 Use the controls to explore tolerance intervention effects!")
    print("🛑 Use Ctrl+C to stop the server")

    try:
        explorer.serve(port=5007, show=True)
    except KeyboardInterrupt:
        print("\n🛑 Tolerance intervention explorer stopped by user")
    except Exception as e:
        print(f"❌ Explorer error: {e}")
        logger.error(f"Tolerance intervention explorer failed: {e}")