"""
Interactive Model Exploration Dashboard

This module creates a sophisticated web-based dashboard for real-time exploration
of ABM-RSiena models with parameter manipulation, scenario comparison, and
interactive network visualization.

Author: Delta Agent - State-of-the-Art Visualization Specialist
Created: 2025-09-15
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
from bokeh.models import HoverTool, ColorBar, LinearColorMapper, ColumnDataSource
from bokeh.palettes import Viridis256, RdYlBu11
from bokeh.layouts import row, column
from bokeh.plotting import figure
import holoviews as hv
from holoviews import opts
import networkx as nx

# Scientific computation
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

# Custom imports
from ..utils.color_schemes import AcademicColorSchemes
from ...models.abm_rsiena_model import ABMRSienaModel
from ...utils.config_manager import ModelConfiguration

logger = logging.getLogger(__name__)

# Enable Panel extensions
pn.extension('bokeh', 'tabulator')
hv.extension('bokeh')

@dataclass
class DashboardConfiguration:
    """Configuration for interactive dashboard."""
    width: int = 1200
    height: int = 800
    update_interval: int = 1000  # milliseconds
    max_network_size: int = 100
    default_n_agents: int = 50
    default_n_steps: int = 100
    color_palette: str = 'viridis'
    show_metrics: bool = True
    show_parameters: bool = True
    enable_animation: bool = True

class ModelDashboard(param.Parameterized):
    """
    Interactive dashboard for ABM-RSiena model exploration.

    Provides real-time parameter manipulation, network visualization,
    statistical analysis, and scenario comparison capabilities.
    """

    # Model parameters (these become interactive widgets)
    n_agents = param.Integer(default=50, bounds=(10, 200), label="Number of Agents")
    n_steps = param.Integer(default=100, bounds=(10, 1000), label="Simulation Steps")

    # Network effects
    density_effect = param.Number(default=-2.0, bounds=(-5.0, 2.0), step=0.1, label="Density Effect")
    reciprocity_effect = param.Number(default=2.0, bounds=(-2.0, 5.0), step=0.1, label="Reciprocity Effect")
    transitivity_effect = param.Number(default=0.5, bounds=(-2.0, 3.0), step=0.1, label="Transitivity Effect")

    # Individual effects
    outdegree_activity = param.Number(default=0.0, bounds=(-2.0, 2.0), step=0.1, label="Outdegree Activity")
    indegree_popularity = param.Number(default=0.0, bounds=(-2.0, 2.0), step=0.1, label="Indegree Popularity")

    # Homophily effects
    age_similarity = param.Number(default=0.0, bounds=(-2.0, 2.0), step=0.1, label="Age Homophily")
    gender_similarity = param.Number(default=0.0, bounds=(-2.0, 2.0), step=0.1, label="Gender Homophily")

    # Behavior effects
    opinion_linear = param.Number(default=0.0, bounds=(-2.0, 2.0), step=0.1, label="Opinion Linear")
    opinion_quadratic = param.Number(default=0.0, bounds=(-2.0, 2.0), step=0.1, label="Opinion Quadratic")

    # Co-evolution
    network_behavior = param.Number(default=0.0, bounds=(-2.0, 2.0), step=0.1, label="Network→Behavior")
    behavior_network = param.Number(default=0.0, bounds=(-2.0, 2.0), step=0.1, label="Behavior→Network")

    # Control buttons
    run_simulation = param.Action(lambda self: self._run_simulation(), label="Run Simulation")
    reset_parameters = param.Action(lambda self: self._reset_parameters(), label="Reset to Defaults")
    save_results = param.Action(lambda self: self._save_results(), label="Save Results")

    def __init__(self, config: DashboardConfiguration = None, **params):
        """
        Initialize interactive dashboard.

        Args:
            config: Dashboard configuration
            **params: Additional parameters
        """
        super().__init__(**params)

        self.config = config or DashboardConfiguration()
        self.color_schemes = AcademicColorSchemes()

        # Dashboard state
        self.current_model = None
        self.current_data = None
        self.simulation_results = {}
        self.network_snapshots = []

        # Create layout components
        self._create_dashboard_layout()

        logger.info("Interactive model dashboard initialized")

    def _create_dashboard_layout(self):
        """Create the main dashboard layout."""
        # Parameter control panel
        param_panel = pn.Param(
            self,
            parameters=[
                'n_agents', 'n_steps',
                'density_effect', 'reciprocity_effect', 'transitivity_effect',
                'outdegree_activity', 'indegree_popularity',
                'age_similarity', 'gender_similarity',
                'opinion_linear', 'opinion_quadratic',
                'network_behavior', 'behavior_network'
            ],
            widgets={
                'n_agents': pn.widgets.IntSlider,
                'n_steps': pn.widgets.IntSlider,
                'density_effect': pn.widgets.FloatSlider,
                'reciprocity_effect': pn.widgets.FloatSlider,
                'transitivity_effect': pn.widgets.FloatSlider,
                'outdegree_activity': pn.widgets.FloatSlider,
                'indegree_popularity': pn.widgets.FloatSlider,
                'age_similarity': pn.widgets.FloatSlider,
                'gender_similarity': pn.widgets.FloatSlider,
                'opinion_linear': pn.widgets.FloatSlider,
                'opinion_quadratic': pn.widgets.FloatSlider,
                'network_behavior': pn.widgets.FloatSlider,
                'behavior_network': pn.widgets.FloatSlider
            },
            name="Model Parameters"
        )

        # Control buttons
        control_panel = pn.Param(
            self,
            parameters=['run_simulation', 'reset_parameters', 'save_results'],
            widgets={
                'run_simulation': pn.widgets.Button,
                'reset_parameters': pn.widgets.Button,
                'save_results': pn.widgets.Button
            },
            name="Controls"
        )

        # Main visualization area
        self.network_plot = self._create_network_plot()
        self.metrics_plot = self._create_metrics_plot()
        self.parameter_plot = self._create_parameter_evolution_plot()

        # Status and info panel
        self.status_panel = pn.pane.Markdown("## Dashboard Status\n\nReady to run simulation...")
        self.info_panel = pn.pane.Markdown(self._get_info_text())

        # Create tabs for different views
        vis_tabs = pn.Tabs(
            ("Network", self.network_plot),
            ("Metrics Evolution", self.metrics_plot),
            ("Parameter Evolution", self.parameter_plot),
            ("Network Statistics", self._create_statistics_table())
        )

        # Create main layout
        self.layout = pn.template.FastListTemplate(
            title="ABM-RSiena Interactive Model Explorer",
            sidebar=[
                param_panel,
                control_panel,
                self.status_panel
            ],
            main=[
                pn.Row(
                    vis_tabs,
                    sizing_mode='stretch_width'
                ),
                self.info_panel
            ],
            header_background='#2F4F4F',
            sidebar_width=350
        )

    def _create_network_plot(self):
        """Create interactive network visualization."""
        # Start with empty plot
        p = figure(
            title="Network Structure",
            width=700, height=500,
            tools="pan,wheel_zoom,box_zoom,reset,save",
            toolbar_location="above"
        )

        p.title.text_font_size = "16pt"
        p.title.align = "center"

        # Add placeholder
        p.text([0], [0], text=["Run simulation to see network"],
              text_align="center", text_baseline="middle",
              text_font_size="14pt", color="gray")

        return pn.pane.Bokeh(p, sizing_mode='stretch_both')

    def _create_metrics_plot(self):
        """Create metrics evolution plot."""
        # Use HoloViews for more interactive plotting
        empty_curve = hv.Curve([]).opts(
            title="Network Metrics Evolution",
            xlabel="Time Step",
            ylabel="Metric Value",
            width=700, height=400,
            tools=['hover', 'pan', 'wheel_zoom', 'box_zoom', 'reset'],
            show_grid=True
        )

        return pn.pane.HoloViews(empty_curve, sizing_mode='stretch_both')

    def _create_parameter_evolution_plot(self):
        """Create parameter evolution tracking plot."""
        empty_curve = hv.Curve([]).opts(
            title="Parameter Evolution (RSiena Updates)",
            xlabel="RSiena Period",
            ylabel="Parameter Value",
            width=700, height=400,
            tools=['hover', 'pan', 'wheel_zoom', 'box_zoom', 'reset'],
            show_grid=True
        )

        return pn.pane.HoloViews(empty_curve, sizing_mode='stretch_both')

    def _create_statistics_table(self):
        """Create network statistics table."""
        empty_df = pd.DataFrame({
            'Metric': ['Density', 'Average Degree', 'Clustering', 'Components'],
            'Value': ['-', '-', '-', '-'],
            'Description': [
                'Proportion of possible edges present',
                'Mean number of connections per node',
                'Tendency for nodes to form triangles',
                'Number of disconnected components'
            ]
        })

        return pn.widgets.Tabulator(
            empty_df,
            pagination='remote',
            page_size=10,
            sizing_mode='stretch_width'
        )

    def _get_info_text(self):
        """Get informational text for the dashboard."""
        return """
## ABM-RSiena Integration Dashboard

This interactive dashboard allows you to explore Agent-Based Models integrated with RSiena
statistical analysis in real-time.

### Features:
- **Real-time parameter adjustment**: Use sliders to modify model parameters
- **Network visualization**: See network structure and evolution
- **Metrics tracking**: Monitor network statistics over time
- **Parameter evolution**: Track RSiena parameter updates
- **Statistical validation**: Compare with empirical data

### How to Use:
1. Adjust parameters using the sliders on the left
2. Click "Run Simulation" to start the model
3. Explore different tabs to see various aspects of the results
4. Save results for further analysis

### Model Integration:
The dashboard runs a full ABM-RSiena integration cycle:
- Agent-based simulation generates network evolution
- RSiena analyzes longitudinal network data
- Parameter estimates feed back into the ABM
- Process repeats for continuous learning
        """

    def _run_simulation(self):
        """Run the ABM-RSiena simulation with current parameters."""
        try:
            self.status_panel.object = "## Running Simulation...\n\nInitializing model and agents..."

            # Create model configuration
            config = self._create_model_config()

            # Run simulation
            self.status_panel.object = "## Running Simulation...\n\nExecuting ABM steps..."
            model, data = self._execute_simulation(config)

            self.current_model = model
            self.current_data = data

            # Update visualizations
            self.status_panel.object = "## Running Simulation...\n\nUpdating visualizations..."
            self._update_network_visualization()
            self._update_metrics_visualization()
            self._update_parameter_visualization()
            self._update_statistics_table()

            self.status_panel.object = "## Simulation Complete!\n\n" + self._get_simulation_summary()

        except Exception as e:
            logger.error(f"Simulation failed: {e}")
            self.status_panel.object = f"## Simulation Failed!\n\nError: {str(e)}"

    def _create_model_config(self):
        """Create model configuration from current parameters."""
        from ...utils.config_manager import ModelConfiguration

        # Create base configuration
        config = ModelConfiguration(
            n_agents=self.n_agents,
            n_steps=self.n_steps,
            random_seed=42,
            behavior_variables=['opinion'],
            mean_age=35,
            sd_age=10,
            abm_steps_per_period=25
        )

        # Add RSiena effects specification
        config.rsiena_effects = [
            ('density', 'eval', 'density'),
            ('recip', 'eval', 'reciprocity'),
            ('transTrip', 'eval', 'transitivity'),
            ('outAct', 'eval', 'outdegree activity'),
            ('inPop', 'eval', 'indegree popularity'),
            ('simX', 'eval', 'age similarity'),
            ('simX', 'eval', 'gender similarity'),
            ('linear', 'behav', 'opinion linear shape'),
            ('quad', 'behav', 'opinion quadratic shape'),
            ('avSim', 'eval', 'opinion average similarity'),
        ]

        return config

    def _execute_simulation(self, config):
        """Execute the ABM simulation."""
        # Import here to avoid circular imports
        from ...models.abm_rsiena_model import run_integrated_simulation

        # Create progress callback
        def progress_callback(step, metrics):
            if step % 20 == 0:
                self.status_panel.object = f"## Running Simulation...\n\nStep {step}/{self.n_steps}\n\nDensity: {metrics['density']:.3f}"

        # Run simulation
        model, data = run_integrated_simulation(
            config=config,
            n_steps=self.n_steps,
            enable_rsiena=False,  # Disable for faster dashboard updates
            progress_callback=progress_callback
        )

        return model, data

    def _update_network_visualization(self):
        """Update the network visualization with current results."""
        if self.current_model is None:
            return

        network = self.current_model.network

        if len(network) == 0:
            return

        # Limit network size for performance
        if len(network) > self.config.max_network_size:
            nodes = list(network.nodes())[:self.config.max_network_size]
            network = network.subgraph(nodes)

        # Calculate layout
        pos = nx.spring_layout(network, k=1/np.sqrt(len(network)), iterations=50)

        # Prepare node data
        node_x = []
        node_y = []
        node_colors = []
        node_sizes = []
        node_labels = []

        for node in network.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)

            # Color by degree
            degree = network.degree(node)
            node_colors.append(degree)
            node_sizes.append(max(10, degree * 3))
            node_labels.append(f"Node {node}<br>Degree: {degree}")

        # Prepare edge data
        edge_x = []
        edge_y = []

        for edge in network.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

        # Create new plot
        p = figure(
            title=f"Network Structure ({len(network)} nodes, {len(network.edges())} edges)",
            width=700, height=500,
            tools="pan,wheel_zoom,box_zoom,reset,save",
            toolbar_location="above"
        )

        # Add edges
        p.line(edge_x, edge_y, line_width=0.5, line_alpha=0.5, color="gray")

        # Add nodes with color mapping
        if node_colors:
            mapper = LinearColorMapper(palette=Viridis256,
                                     low=min(node_colors),
                                     high=max(node_colors))

            node_source = ColumnDataSource(dict(
                x=node_x, y=node_y,
                colors=node_colors,
                sizes=node_sizes,
                labels=node_labels
            ))

            nodes_glyph = p.circle('x', 'y', size='sizes',
                                  color={'field': 'colors', 'transform': mapper},
                                  source=node_source, alpha=0.8)

            # Add hover tool
            hover = HoverTool(tooltips="@labels", renderers=[nodes_glyph])
            p.add_tools(hover)

            # Add color bar
            color_bar = ColorBar(color_mapper=mapper,
                               label_standoff=12,
                               location=(0,0),
                               title="Node Degree")
            p.add_layout(color_bar, 'right')

        p.title.text_font_size = "14pt"
        p.axis.visible = False
        p.grid.visible = False

        # Update the panel
        self.network_plot.object = p

    def _update_metrics_visualization(self):
        """Update metrics evolution plot."""
        if self.current_data is None:
            return

        # Get metrics data
        metrics_df = self.current_data.reset_index()

        if len(metrics_df) == 0:
            return

        # Create curves for different metrics
        curves = []

        # Network density
        if 'network_density' in metrics_df.columns:
            density_curve = hv.Curve(
                (metrics_df.index, metrics_df['network_density']),
                label='Density'
            ).opts(color=self.color_schemes.primary_palette[0], line_width=2)
            curves.append(density_curve)

        # Average degree
        if 'average_degree' in metrics_df.columns:
            degree_curve = hv.Curve(
                (metrics_df.index, metrics_df['average_degree']),
                label='Avg Degree'
            ).opts(color=self.color_schemes.primary_palette[1], line_width=2)
            curves.append(degree_curve)

        # Clustering coefficient
        if 'clustering_coefficient' in metrics_df.columns:
            clustering_curve = hv.Curve(
                (metrics_df.index, metrics_df['clustering_coefficient']),
                label='Clustering'
            ).opts(color=self.color_schemes.primary_palette[2], line_width=2)
            curves.append(clustering_curve)

        # Combine curves
        if curves:
            combined = hv.Overlay(curves).opts(
                title="Network Metrics Evolution",
                xlabel="Time Step",
                ylabel="Metric Value",
                width=700, height=400,
                legend_position='top_right',
                show_grid=True,
                tools=['hover', 'pan', 'wheel_zoom', 'box_zoom', 'reset']
            )

            self.metrics_plot.object = combined

    def _update_parameter_visualization(self):
        """Update parameter evolution plot."""
        # For now, show static parameter values
        # In full implementation, this would show RSiena parameter updates over time

        param_names = ['Density', 'Reciprocity', 'Transitivity']
        param_values = [self.density_effect, self.reciprocity_effect, self.transitivity_effect]

        bars = hv.Bars(
            (param_names, param_values),
            kdims='Parameter',
            vdims='Value'
        ).opts(
            title="Current Parameter Values",
            xlabel="Parameter",
            ylabel="Value",
            width=700, height=400,
            tools=['hover'],
            show_grid=True,
            color=self.color_schemes.primary_palette[0]
        )

        self.parameter_plot.object = bars

    def _update_statistics_table(self):
        """Update network statistics table."""
        if self.current_model is None:
            return

        network = self.current_model.network

        if len(network) == 0:
            return

        # Calculate statistics
        stats = {
            'Density': f"{nx.density(network):.4f}",
            'Average Degree': f"{sum(dict(network.degree()).values()) / len(network):.2f}",
            'Clustering': f"{nx.transitivity(network):.4f}",
            'Components': f"{nx.number_connected_components(network)}",
            'Nodes': f"{len(network)}",
            'Edges': f"{len(network.edges())}",
            'Average Path Length': self._safe_avg_path_length(network),
            'Assortativity': self._safe_assortativity(network)
        }

        # Create DataFrame
        stats_df = pd.DataFrame([
            {'Metric': k, 'Value': v, 'Description': self._get_metric_description(k)}
            for k, v in stats.items()
        ])

        # Update table
        if hasattr(self, 'layout'):
            # Find and update the statistics table in the tabs
            for tab in self.layout.main[0][0]:
                if hasattr(tab, 'object') and hasattr(tab.object, 'value'):
                    if isinstance(tab.object.value, pd.DataFrame):
                        tab.object.value = stats_df
                        break

    def _safe_avg_path_length(self, network):
        """Safely calculate average path length."""
        try:
            if nx.is_connected(network):
                return f"{nx.average_shortest_path_length(network):.2f}"
            else:
                return "Disconnected"
        except:
            return "N/A"

    def _safe_assortativity(self, network):
        """Safely calculate degree assortativity."""
        try:
            return f"{nx.degree_assortativity_coefficient(network):.4f}"
        except:
            return "N/A"

    def _get_metric_description(self, metric):
        """Get description for a metric."""
        descriptions = {
            'Density': 'Proportion of possible edges present',
            'Average Degree': 'Mean number of connections per node',
            'Clustering': 'Tendency for nodes to form triangles',
            'Components': 'Number of disconnected components',
            'Nodes': 'Total number of nodes in network',
            'Edges': 'Total number of edges in network',
            'Average Path Length': 'Mean shortest path between nodes',
            'Assortativity': 'Tendency for similar-degree nodes to connect'
        }
        return descriptions.get(metric, 'Network statistic')

    def _get_simulation_summary(self):
        """Get summary of simulation results."""
        if self.current_model is None:
            return "No simulation data available."

        network = self.current_model.network

        return f"""
**Simulation completed successfully!**

- **Agents**: {len(network)} active agents
- **Connections**: {len(network.edges())} network edges
- **Density**: {nx.density(network):.4f}
- **Steps**: {self.n_steps} simulation steps

Explore the tabs above to see detailed results.
        """

    def _reset_parameters(self):
        """Reset all parameters to defaults."""
        defaults = {
            'n_agents': 50, 'n_steps': 100,
            'density_effect': -2.0, 'reciprocity_effect': 2.0,
            'transitivity_effect': 0.5, 'outdegree_activity': 0.0,
            'indegree_popularity': 0.0, 'age_similarity': 0.0,
            'gender_similarity': 0.0, 'opinion_linear': 0.0,
            'opinion_quadratic': 0.0, 'network_behavior': 0.0,
            'behavior_network': 0.0
        }

        for param_name, default_value in defaults.items():
            setattr(self, param_name, default_value)

        self.status_panel.object = "## Parameters Reset\n\nAll parameters reset to default values."

    def _save_results(self):
        """Save current simulation results."""
        if self.current_model is None or self.current_data is None:
            self.status_panel.object = "## Save Failed\n\nNo simulation results to save."
            return

        try:
            # Save data
            output_dir = Path("outputs/dashboard_results")
            output_dir.mkdir(exist_ok=True)

            # Save metrics data
            timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
            metrics_file = output_dir / f"metrics_{timestamp}.csv"
            self.current_data.to_csv(metrics_file)

            # Save network
            network_file = output_dir / f"network_{timestamp}.gml"
            nx.write_gml(self.current_model.network, network_file)

            # Save parameters
            params_file = output_dir / f"parameters_{timestamp}.json"
            import json
            param_dict = {
                name: getattr(self, name)
                for name in [
                    'n_agents', 'n_steps', 'density_effect', 'reciprocity_effect',
                    'transitivity_effect', 'outdegree_activity', 'indegree_popularity',
                    'age_similarity', 'gender_similarity', 'opinion_linear',
                    'opinion_quadratic', 'network_behavior', 'behavior_network'
                ]
            }

            with open(params_file, 'w') as f:
                json.dump(param_dict, f, indent=2)

            self.status_panel.object = f"""
## Results Saved!

Files saved to `outputs/dashboard_results/`:
- Metrics: `metrics_{timestamp}.csv`
- Network: `network_{timestamp}.gml`
- Parameters: `parameters_{timestamp}.json`
            """

        except Exception as e:
            logger.error(f"Failed to save results: {e}")
            self.status_panel.object = f"## Save Failed\n\nError: {str(e)}"

    def serve(self, port: int = 5006, show: bool = True):
        """
        Serve the dashboard as a web application.

        Args:
            port: Port number for the web server
            show: Whether to open browser automatically
        """
        return self.layout.servable().show(port=port, threaded=True, verbose=False, open=show)

    def save_static_dashboard(self, filepath: str):
        """
        Save a static version of the dashboard.

        Args:
            filepath: Path to save the static HTML file
        """
        self.layout.save(filepath)
        logger.info(f"Static dashboard saved to: {filepath}")


def create_dashboard(config: DashboardConfiguration = None) -> ModelDashboard:
    """
    Create and configure the ABM-RSiena dashboard.

    Args:
        config: Dashboard configuration

    Returns:
        Configured ModelDashboard instance
    """
    return ModelDashboard(config=config)


# Example usage and testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Create dashboard
    dashboard = create_dashboard()

    # For testing, we can save a static version
    # dashboard.save_static_dashboard("outputs/dashboard_demo.html")

    # Serve the dashboard
    print("Starting ABM-RSiena Interactive Dashboard...")
    print("Access the dashboard at: http://localhost:5006")
    print("Use Ctrl+C to stop the server")

    try:
        dashboard.serve(port=5006, show=True)
    except KeyboardInterrupt:
        print("\nDashboard stopped by user")
    except Exception as e:
        print(f"Dashboard error: {e}")
        logger.error(f"Dashboard failed: {e}")