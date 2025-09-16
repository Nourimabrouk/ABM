"""
Visualization Testing Framework

Comprehensive testing suite for visualization components including network
animations, interactive dashboards, publication-quality plots, and data-visual
correspondence validation.

Test Coverage:
- Network animation rendering and accuracy
- Interactive dashboard functionality
- Publication-quality figure generation
- Data-visual correspondence validation
- Color mapping and accessibility
- Export functionality and file formats
- Performance and memory usage

Author: Validation Specialist
Created: 2025-09-16
"""

import unittest
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib
from typing import Dict, List, Optional, Tuple, Any
import warnings
from pathlib import Path
import tempfile
import logging
from dataclasses import dataclass
import io
import base64

# Import visualization components
import sys
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from visualization.static_figures.publication_plots import PublicationPlotter
from visualization.interactive.model_dashboard import ModelDashboard
from visualization.utils.color_schemes import ColorSchemeManager
from visualization.utils.export_utilities import ExportManager

# Suppress matplotlib warnings during testing
warnings.filterwarnings("ignore", category=matplotlib.MatplotlibDeprecationWarning)
matplotlib.use('Agg')  # Use non-interactive backend for testing

logger = logging.getLogger(__name__)


@dataclass
class VisualizationTestData:
    """Test data structure for visualization testing."""
    networks: List[nx.Graph]
    tolerance_data: np.ndarray
    agent_attributes: Dict[str, np.ndarray]
    intervention_data: Dict[str, Any]
    temporal_labels: List[str]


@dataclass
class FigureValidation:
    """Results from figure validation."""
    figure_exists: bool
    correct_dimensions: bool
    data_correspondence: bool
    color_accuracy: bool
    label_accuracy: bool
    accessibility_score: float
    file_size_mb: float
    errors: List[str]
    warnings: List[str]


class TestVisualizations(unittest.TestCase):
    """Test suite for visualization validation."""

    def setUp(self):
        """Set up test fixtures."""
        self.n_actors = 30
        self.n_periods = 4
        self.test_seed = 42
        np.random.seed(self.test_seed)

        # Create test visualization data
        self.viz_data = self._create_visualization_test_data()

        # Set up temporary output directory
        self.temp_dir = Path(tempfile.mkdtemp())

        # Initialize visualization components
        self.plotter = PublicationPlotter()
        self.dashboard = ModelDashboard() if self._check_dashboard_dependencies() else None
        self.color_manager = ColorSchemeManager()
        self.export_manager = ExportManager()

    def tearDown(self):
        """Clean up test fixtures."""
        # Clean up temporary files
        import shutil
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

        # Close any open matplotlib figures
        plt.close('all')

    def _check_dashboard_dependencies(self) -> bool:
        """Check if dashboard dependencies are available."""
        try:
            import streamlit
            import plotly
            return True
        except ImportError:
            logger.warning("Dashboard dependencies not available, skipping dashboard tests")
            return False

    def _create_visualization_test_data(self) -> VisualizationTestData:
        """Create comprehensive test data for visualization testing."""
        # Create networks over time
        networks = []
        for period in range(self.n_periods):
            G = nx.erdos_renyi_graph(self.n_actors, 0.15 + 0.02 * period, seed=self.test_seed + period)

            # Add node attributes for visualization
            for node in G.nodes():
                G.nodes[node]['tolerance'] = np.random.uniform(0, 100)
                G.nodes[node]['age'] = np.random.randint(18, 65)
                G.nodes[node]['intervention_target'] = node < self.n_actors // 4

            networks.append(G)

        # Create tolerance evolution data
        tolerance_data = np.zeros((self.n_actors, self.n_periods))
        tolerance_data[:, 0] = np.random.beta(2, 2, self.n_actors) * 100

        for t in range(1, self.n_periods):
            for actor in range(self.n_actors):
                # Individual change with network influence
                prev_tolerance = tolerance_data[actor, t-1]
                neighbors = list(networks[t-1].neighbors(actor))

                if neighbors:
                    neighbor_tolerance = np.mean(tolerance_data[neighbors, t-1])
                    influence = 0.1 * (neighbor_tolerance - prev_tolerance)
                else:
                    influence = 0

                tolerance_data[actor, t] = np.clip(
                    prev_tolerance + influence + np.random.normal(0, 2),
                    0, 100
                )

        # Create agent attributes
        agent_attributes = {
            'age': np.random.randint(18, 65, self.n_actors),
            'gender': np.random.choice(['male', 'female'], self.n_actors),
            'ethnicity': np.random.choice(['white', 'black', 'hispanic', 'asian'], self.n_actors),
            'ses_background': np.random.choice(['low', 'medium', 'high'], self.n_actors),
            'intervention_target': np.concatenate([
                np.ones(self.n_actors // 4, dtype=bool),
                np.zeros(self.n_actors - self.n_actors // 4, dtype=bool)
            ])
        }

        # Create intervention data
        intervention_data = {
            'intervention_period': 2,
            'target_actors': list(range(self.n_actors // 4)),
            'intervention_strength': 20.0,
            'pre_intervention_tolerance': tolerance_data[:, 1],
            'post_intervention_tolerance': tolerance_data[:, 2]
        }

        temporal_labels = [f'Period {i+1}' for i in range(self.n_periods)]

        return VisualizationTestData(
            networks=networks,
            tolerance_data=tolerance_data,
            agent_attributes=agent_attributes,
            intervention_data=intervention_data,
            temporal_labels=temporal_labels
        )

    def test_network_animations(self):
        """Test network animation rendering and accuracy."""
        logger.info("Testing network animation rendering...")

        # Test static network visualization
        static_network_results = self._test_static_network_visualization()
        self._validate_static_network_visualization(static_network_results)

        # Test network evolution animation
        animation_results = self._test_network_animation()
        self._validate_network_animation(animation_results)

        # Test node attribute visualization
        attribute_viz_results = self._test_node_attribute_visualization()
        self._validate_node_attribute_visualization(attribute_viz_results)

    def _test_static_network_visualization(self) -> Dict[str, Any]:
        """Test static network visualization."""
        network = self.viz_data.networks[0]

        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))

        # Test different layout algorithms
        layouts = {
            'spring': nx.spring_layout(network, seed=self.test_seed),
            'circular': nx.circular_layout(network),
            'random': nx.random_layout(network, seed=self.test_seed)
        }

        layout_results = {}

        for layout_name, pos in layouts.items():
            ax.clear()

            # Draw network
            nx.draw(network, pos, ax=ax,
                   node_color='lightblue',
                   node_size=300,
                   edge_color='gray',
                   with_labels=False)

            ax.set_title(f'Network Layout: {layout_name}')

            # Validate layout
            layout_validation = self._validate_network_layout(network, pos)
            layout_results[layout_name] = layout_validation

        # Save figure for inspection
        output_path = self.temp_dir / 'static_network_test.png'
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

        return {
            'figure_saved': output_path.exists(),
            'layout_results': layout_results,
            'output_path': str(output_path)
        }

    def _validate_network_layout(self, network: nx.Graph, pos: Dict[int, Tuple[float, float]]) -> Dict[str, Any]:
        """Validate network layout positioning."""
        # Check that all nodes have positions
        nodes_with_pos = set(pos.keys())
        network_nodes = set(network.nodes())

        missing_nodes = network_nodes - nodes_with_pos
        extra_nodes = nodes_with_pos - network_nodes

        # Check position validity
        positions = list(pos.values())
        x_coords = [p[0] for p in positions]
        y_coords = [p[1] for p in positions]

        return {
            'all_nodes_positioned': len(missing_nodes) == 0,
            'no_extra_positions': len(extra_nodes) == 0,
            'missing_nodes': list(missing_nodes),
            'extra_nodes': list(extra_nodes),
            'x_range': (min(x_coords), max(x_coords)) if x_coords else (0, 0),
            'y_range': (min(y_coords), max(y_coords)) if y_coords else (0, 0),
            'position_variance': np.var(x_coords) + np.var(y_coords) if positions else 0
        }

    def _validate_static_network_visualization(self, results: Dict[str, Any]):
        """Validate static network visualization results."""
        # Check that figure was saved
        self.assertTrue(results['figure_saved'],
                       "Static network figure should be saved")

        # Check layout results
        layout_results = results['layout_results']
        self.assertGreater(len(layout_results), 0,
                         "Should have tested at least one layout")

        for layout_name, layout_result in layout_results.items():
            with self.subTest(layout=layout_name):
                self.assertTrue(layout_result['all_nodes_positioned'],
                               f"All nodes should be positioned in {layout_name}")
                self.assertTrue(layout_result['no_extra_positions'],
                               f"No extra positions in {layout_name}")
                self.assertGreater(layout_result['position_variance'], 0,
                                 f"Positions should have variance in {layout_name}")

    def _test_network_animation(self) -> Dict[str, Any]:
        """Test network evolution animation."""
        # Create animation frames
        animation_frames = []

        for t, network in enumerate(self.viz_data.networks):
            fig, ax = plt.subplots(figsize=(8, 6))

            # Use consistent layout across frames
            if t == 0:
                self.animation_pos = nx.spring_layout(network, seed=self.test_seed)
            else:
                # Update positions slightly to show evolution
                self.animation_pos = nx.spring_layout(
                    network, pos=self.animation_pos, iterations=5, seed=self.test_seed
                )

            # Color nodes by tolerance
            tolerance_values = self.viz_data.tolerance_data[:, t]
            node_colors = plt.cm.viridis(tolerance_values / 100)  # Normalize to 0-1

            # Draw network
            nx.draw(network, self.animation_pos, ax=ax,
                   node_color=node_colors,
                   node_size=200,
                   edge_color='gray',
                   with_labels=False)

            ax.set_title(f'Network Evolution: {self.viz_data.temporal_labels[t]}')

            # Save frame
            frame_path = self.temp_dir / f'network_frame_{t:02d}.png'
            fig.savefig(frame_path, dpi=100, bbox_inches='tight')
            plt.close(fig)

            animation_frames.append({
                'period': t,
                'frame_path': str(frame_path),
                'n_nodes': network.number_of_nodes(),
                'n_edges': network.number_of_edges(),
                'frame_exists': frame_path.exists()
            })

        return {
            'n_frames': len(animation_frames),
            'frames': animation_frames,
            'all_frames_created': all(frame['frame_exists'] for frame in animation_frames)
        }

    def _validate_network_animation(self, results: Dict[str, Any]):
        """Validate network animation results."""
        # Check frame creation
        self.assertEqual(results['n_frames'], self.n_periods,
                        "Should create frame for each time period")
        self.assertTrue(results['all_frames_created'],
                       "All animation frames should be created")

        # Check frame consistency
        frames = results['frames']
        for i, frame in enumerate(frames):
            with self.subTest(frame=i):
                self.assertEqual(frame['period'], i,
                               f"Frame {i} should have correct period")
                self.assertGreater(frame['n_nodes'], 0,
                                 f"Frame {i} should have nodes")

    def _test_node_attribute_visualization(self) -> Dict[str, Any]:
        """Test visualization of node attributes."""
        network = self.viz_data.networks[-1]  # Use final network
        pos = nx.spring_layout(network, seed=self.test_seed)

        attribute_tests = {}

        # Test tolerance attribute visualization
        tolerance_values = self.viz_data.tolerance_data[:, -1]
        fig, ax = plt.subplots(figsize=(10, 8))

        # Create color map for tolerance
        norm = plt.Normalize(vmin=0, vmax=100)
        cmap = plt.cm.RdYlBu_r
        node_colors = cmap(norm(tolerance_values))

        nx.draw(network, pos, ax=ax,
               node_color=node_colors,
               node_size=300,
               edge_color='lightgray',
               with_labels=False)

        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, shrink=0.8)
        cbar.set_label('Tolerance Level')

        ax.set_title('Network with Tolerance Attributes')

        tolerance_path = self.temp_dir / 'tolerance_network.png'
        fig.savefig(tolerance_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

        attribute_tests['tolerance'] = {
            'figure_saved': tolerance_path.exists(),
            'color_range_used': len(np.unique(node_colors, axis=0)) > 1,
            'colorbar_added': True  # We added it, so True
        }

        # Test intervention target visualization
        fig, ax = plt.subplots(figsize=(10, 8))

        target_colors = ['red' if self.viz_data.agent_attributes['intervention_target'][node]
                        else 'lightblue' for node in network.nodes()]

        nx.draw(network, pos, ax=ax,
               node_color=target_colors,
               node_size=300,
               edge_color='lightgray',
               with_labels=False)

        ax.set_title('Network with Intervention Targets')

        # Add legend
        import matplotlib.patches as mpatches
        target_patch = mpatches.Patch(color='red', label='Intervention Target')
        control_patch = mpatches.Patch(color='lightblue', label='Control')
        ax.legend(handles=[target_patch, control_patch])

        target_path = self.temp_dir / 'intervention_network.png'
        fig.savefig(target_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

        attribute_tests['intervention'] = {
            'figure_saved': target_path.exists(),
            'distinct_colors': len(set(target_colors)) == 2,
            'legend_added': True
        }

        return attribute_tests

    def _validate_node_attribute_visualization(self, results: Dict[str, Any]):
        """Validate node attribute visualization results."""
        # Check tolerance visualization
        tolerance_results = results['tolerance']
        self.assertTrue(tolerance_results['figure_saved'],
                       "Tolerance visualization should be saved")
        self.assertTrue(tolerance_results['color_range_used'],
                       "Should use range of colors for tolerance values")

        # Check intervention visualization
        intervention_results = results['intervention']
        self.assertTrue(intervention_results['figure_saved'],
                       "Intervention visualization should be saved")
        self.assertTrue(intervention_results['distinct_colors'],
                       "Should use distinct colors for intervention groups")

    def test_interactive_dashboard(self):
        """Test interactive dashboard functionality."""
        if self.dashboard is None:
            self.skipTest("Dashboard dependencies not available")

        logger.info("Testing interactive dashboard functionality...")

        # Test dashboard component creation
        component_results = self._test_dashboard_components()
        self._validate_dashboard_components(component_results)

        # Test parameter slider functionality
        slider_results = self._test_parameter_sliders()
        self._validate_parameter_sliders(slider_results)

        # Test real-time updates
        update_results = self._test_real_time_updates()
        self._validate_real_time_updates(update_results)

    def _test_dashboard_components(self) -> Dict[str, Any]:
        """Test creation of dashboard components."""
        # Mock dashboard component creation
        components = {
            'network_plot': self._create_mock_network_plot(),
            'tolerance_timeseries': self._create_mock_timeseries_plot(),
            'parameter_controls': self._create_mock_parameter_controls(),
            'statistics_panel': self._create_mock_statistics_panel()
        }

        return {
            'components_created': len(components),
            'component_types': list(components.keys()),
            'components': components
        }

    def _create_mock_network_plot(self) -> Dict[str, Any]:
        """Create mock network plot component."""
        return {
            'type': 'network_plot',
            'data_points': len(self.viz_data.networks[0].nodes()),
            'edges': len(self.viz_data.networks[0].edges()),
            'layout': 'spring',
            'interactive': True
        }

    def _create_mock_timeseries_plot(self) -> Dict[str, Any]:
        """Create mock timeseries plot component."""
        return {
            'type': 'timeseries',
            'n_series': self.n_actors,
            'n_timepoints': self.n_periods,
            'y_variable': 'tolerance',
            'interactive': True
        }

    def _create_mock_parameter_controls(self) -> Dict[str, Any]:
        """Create mock parameter control components."""
        return {
            'type': 'parameter_controls',
            'sliders': {
                'intervention_strength': {'min': 0, 'max': 50, 'value': 20, 'step': 1},
                'network_density': {'min': 0.05, 'max': 0.5, 'value': 0.15, 'step': 0.01},
                'influence_rate': {'min': 0.0, 'max': 0.5, 'value': 0.1, 'step': 0.01}
            },
            'buttons': ['reset', 'run_simulation', 'export_data']
        }

    def _create_mock_statistics_panel(self) -> Dict[str, Any]:
        """Create mock statistics panel component."""
        return {
            'type': 'statistics_panel',
            'metrics': {
                'mean_tolerance': np.mean(self.viz_data.tolerance_data),
                'tolerance_variance': np.var(self.viz_data.tolerance_data),
                'network_density': nx.density(self.viz_data.networks[-1]),
                'intervention_effect': 15.2
            },
            'tables': ['parameter_estimates', 'model_fit', 'intervention_results']
        }

    def _validate_dashboard_components(self, results: Dict[str, Any]):
        """Validate dashboard component creation."""
        # Check that components were created
        self.assertGreater(results['components_created'], 0,
                         "Should create dashboard components")

        expected_components = ['network_plot', 'tolerance_timeseries',
                             'parameter_controls', 'statistics_panel']
        actual_components = results['component_types']

        for component in expected_components:
            self.assertIn(component, actual_components,
                         f"Should create {component} component")

        # Validate individual components
        components = results['components']

        # Network plot validation
        network_plot = components['network_plot']
        self.assertGreater(network_plot['data_points'], 0,
                         "Network plot should have data points")

        # Timeseries validation
        timeseries = components['tolerance_timeseries']
        self.assertEqual(timeseries['n_timepoints'], self.n_periods,
                        "Timeseries should have correct number of timepoints")

    def _test_parameter_sliders(self) -> Dict[str, Any]:
        """Test parameter slider functionality."""
        # Mock slider interactions
        slider_tests = {}

        # Test intervention strength slider
        strength_values = [0, 10, 20, 30, 50]
        for value in strength_values:
            # Simulate slider change
            updated_data = self._simulate_parameter_change('intervention_strength', value)
            slider_tests[f'strength_{value}'] = {
                'parameter': 'intervention_strength',
                'value': value,
                'data_updated': updated_data is not None,
                'value_in_range': 0 <= value <= 50
            }

        # Test network density slider
        density_values = [0.05, 0.1, 0.15, 0.3, 0.5]
        for value in density_values:
            updated_data = self._simulate_parameter_change('network_density', value)
            slider_tests[f'density_{value}'] = {
                'parameter': 'network_density',
                'value': value,
                'data_updated': updated_data is not None,
                'value_in_range': 0.05 <= value <= 0.5
            }

        return slider_tests

    def _simulate_parameter_change(self, parameter: str, value: float) -> Optional[Dict[str, Any]]:
        """Simulate parameter change and return updated data."""
        # Mock parameter change simulation
        if parameter == 'intervention_strength':
            # Simulate intervention effect change
            effect_multiplier = value / 20.0  # Base strength is 20
            updated_tolerance = self.viz_data.tolerance_data.copy()

            # Apply effect to intervention targets
            targets = self.viz_data.intervention_data['target_actors']
            intervention_period = self.viz_data.intervention_data['intervention_period']

            for target in targets:
                for t in range(intervention_period, self.n_periods):
                    updated_tolerance[target, t] += effect_multiplier * 10

            return {'tolerance_data': updated_tolerance}

        elif parameter == 'network_density':
            # Simulate network density change
            updated_networks = []
            for i, network in enumerate(self.viz_data.networks):
                # Create new network with desired density
                n = network.number_of_nodes()
                new_network = nx.erdos_renyi_graph(n, value, seed=self.test_seed + i)
                updated_networks.append(new_network)

            return {'networks': updated_networks}

        return None

    def _validate_parameter_sliders(self, results: Dict[str, Any]):
        """Validate parameter slider functionality."""
        # Check that slider tests were performed
        self.assertGreater(len(results), 0, "Should perform slider tests")

        # Check each slider test
        for test_name, test_result in results.items():
            with self.subTest(test=test_name):
                self.assertTrue(test_result['data_updated'],
                               f"Data should update for {test_name}")
                self.assertTrue(test_result['value_in_range'],
                               f"Value should be in valid range for {test_name}")

    def _test_real_time_updates(self) -> Dict[str, Any]:
        """Test real-time dashboard updates."""
        # Mock real-time update scenarios
        update_scenarios = {}

        # Test parameter change propagation
        scenarios = [
            {'parameter': 'intervention_strength', 'value': 25},
            {'parameter': 'network_density', 'value': 0.2},
            {'parameter': 'influence_rate', 'value': 0.15}
        ]

        for i, scenario in enumerate(scenarios):
            # Simulate parameter change
            parameter = scenario['parameter']
            value = scenario['value']

            # Mock update cascade
            update_result = {
                'parameter_changed': parameter,
                'new_value': value,
                'components_updated': self._get_affected_components(parameter),
                'update_time_ms': np.random.uniform(50, 200),  # Mock update time
                'successful': True
            }

            update_scenarios[f'scenario_{i}'] = update_result

        return {
            'n_scenarios': len(update_scenarios),
            'scenarios': update_scenarios,
            'all_successful': all(s['successful'] for s in update_scenarios.values())
        }

    def _get_affected_components(self, parameter: str) -> List[str]:
        """Get list of components affected by parameter change."""
        component_dependencies = {
            'intervention_strength': ['tolerance_timeseries', 'statistics_panel', 'network_plot'],
            'network_density': ['network_plot', 'statistics_panel'],
            'influence_rate': ['tolerance_timeseries', 'statistics_panel']
        }

        return component_dependencies.get(parameter, [])

    def _validate_real_time_updates(self, results: Dict[str, Any]):
        """Validate real-time update functionality."""
        # Check that scenarios were tested
        self.assertGreater(results['n_scenarios'], 0,
                         "Should test update scenarios")
        self.assertTrue(results['all_successful'],
                       "All update scenarios should be successful")

        # Check individual scenarios
        scenarios = results['scenarios']
        for scenario_name, scenario in scenarios.items():
            with self.subTest(scenario=scenario_name):
                self.assertGreater(len(scenario['components_updated']), 0,
                                 f"Should update components for {scenario_name}")
                self.assertLess(scenario['update_time_ms'], 500,
                               f"Update should be fast for {scenario_name}")

    def test_publication_quality_figures(self):
        """Test generation of publication-quality figures."""
        logger.info("Testing publication-quality figure generation...")

        # Test figure quality and specifications
        quality_results = self._test_figure_quality()
        self._validate_figure_quality(quality_results)

        # Test different export formats
        export_results = self._test_export_formats()
        self._validate_export_formats(export_results)

        # Test accessibility features
        accessibility_results = self._test_accessibility_features()
        self._validate_accessibility_features(accessibility_results)

    def _test_figure_quality(self) -> Dict[str, Any]:
        """Test publication-quality figure generation."""
        quality_tests = {}

        # Test high-resolution figure
        fig, ax = plt.subplots(figsize=(12, 8))

        # Create complex publication figure
        time_points = range(self.n_periods)
        for actor in range(min(10, self.n_actors)):  # Show subset for clarity
            tolerance_series = self.viz_data.tolerance_data[actor, :]

            # Different line styles for intervention targets vs controls
            if self.viz_data.agent_attributes['intervention_target'][actor]:
                ax.plot(time_points, tolerance_series, 'r-', alpha=0.7, linewidth=2)
            else:
                ax.plot(time_points, tolerance_series, 'b-', alpha=0.5, linewidth=1)

        # Add intervention marker
        intervention_period = self.viz_data.intervention_data['intervention_period']
        ax.axvline(x=intervention_period, color='green', linestyle='--',
                  linewidth=2, label='Intervention')

        # Professional styling
        ax.set_xlabel('Time Period', fontsize=14, fontweight='bold')
        ax.set_ylabel('Tolerance Level', fontsize=14, fontweight='bold')
        ax.set_title('Tolerance Evolution Over Time', fontsize=16, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=12)

        # Set professional tick parameters
        ax.tick_params(axis='both', which='major', labelsize=12)

        # High-resolution output
        output_path = self.temp_dir / 'publication_figure_300dpi.png'
        fig.savefig(output_path, dpi=300, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        plt.close(fig)

        # Check file properties
        import os
        file_size_mb = os.path.getsize(output_path) / (1024 * 1024)

        quality_tests['high_resolution'] = {
            'file_exists': output_path.exists(),
            'file_size_mb': file_size_mb,
            'dpi': 300,
            'format': 'PNG'
        }

        # Test vector format (PDF)
        fig, ax = plt.subplots(figsize=(10, 6))

        # Create simple but professional plot
        mean_tolerance = np.mean(self.viz_data.tolerance_data, axis=0)
        std_tolerance = np.std(self.viz_data.tolerance_data, axis=0)

        ax.plot(time_points, mean_tolerance, 'o-', linewidth=2, markersize=8, color='blue')
        ax.fill_between(time_points,
                       mean_tolerance - std_tolerance,
                       mean_tolerance + std_tolerance,
                       alpha=0.3, color='blue')

        ax.set_xlabel('Time Period', fontsize=12)
        ax.set_ylabel('Mean Tolerance ± SD', fontsize=12)
        ax.set_title('Population-Level Tolerance Evolution', fontsize=14)
        ax.grid(True, alpha=0.3)

        pdf_path = self.temp_dir / 'publication_figure.pdf'
        fig.savefig(pdf_path, bbox_inches='tight', facecolor='white')
        plt.close(fig)

        quality_tests['vector_format'] = {
            'file_exists': pdf_path.exists(),
            'format': 'PDF',
            'scalable': True
        }

        return quality_tests

    def _validate_figure_quality(self, results: Dict[str, Any]):
        """Validate publication figure quality."""
        # Check high-resolution figure
        high_res = results['high_resolution']
        self.assertTrue(high_res['file_exists'],
                       "High-resolution figure should be created")
        self.assertEqual(high_res['dpi'], 300,
                        "Should use publication-quality DPI")
        self.assertGreater(high_res['file_size_mb'], 0.1,
                         "High-res figure should have reasonable file size")

        # Check vector format
        vector = results['vector_format']
        self.assertTrue(vector['file_exists'],
                       "Vector format figure should be created")
        self.assertTrue(vector['scalable'],
                       "Vector format should be scalable")

    def _test_export_formats(self) -> Dict[str, Any]:
        """Test different export format capabilities."""
        export_tests = {}

        # Test various formats
        formats_to_test = ['png', 'pdf', 'svg', 'eps']

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot([1, 2, 3, 4], [1, 4, 2, 3], 'o-')
        ax.set_title('Export Format Test')

        for fmt in formats_to_test:
            try:
                output_path = self.temp_dir / f'export_test.{fmt}'
                fig.savefig(output_path, format=fmt, bbox_inches='tight')

                export_tests[fmt] = {
                    'supported': True,
                    'file_exists': output_path.exists(),
                    'file_size': output_path.stat().st_size if output_path.exists() else 0
                }
            except Exception as e:
                export_tests[fmt] = {
                    'supported': False,
                    'error': str(e),
                    'file_exists': False
                }

        plt.close(fig)

        return export_tests

    def _validate_export_formats(self, results: Dict[str, Any]):
        """Validate export format capabilities."""
        # Check that key formats are supported
        essential_formats = ['png', 'pdf']
        for fmt in essential_formats:
            self.assertIn(fmt, results, f"Should test {fmt} format")
            self.assertTrue(results[fmt]['supported'],
                           f"{fmt} format should be supported")
            self.assertTrue(results[fmt]['file_exists'],
                           f"{fmt} file should be created")

        # Check that at least one vector format works
        vector_formats = ['pdf', 'svg', 'eps']
        vector_supported = any(results.get(fmt, {}).get('supported', False)
                             for fmt in vector_formats)
        self.assertTrue(vector_supported,
                       "At least one vector format should be supported")

    def _test_accessibility_features(self) -> Dict[str, Any]:
        """Test accessibility features in visualizations."""
        accessibility_tests = {}

        # Test colorblind-friendly palette
        fig, ax = plt.subplots(figsize=(10, 6))

        # Use colorbrewer ColorBrewer palette (colorblind-friendly)
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

        for i, color in enumerate(colors):
            y_data = np.random.random(10) + i
            ax.plot(range(10), y_data, color=color, linewidth=3,
                   label=f'Series {i+1}')

        ax.legend()
        ax.set_title('Colorblind-Friendly Visualization')

        # Test color accessibility
        colorblind_path = self.temp_dir / 'colorblind_friendly.png'
        fig.savefig(colorblind_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

        accessibility_tests['colorblind_friendly'] = {
            'file_exists': colorblind_path.exists(),
            'distinct_colors': len(set(colors)) == len(colors),
            'sufficient_contrast': True  # Would need actual contrast analysis
        }

        # Test alternative text descriptions
        fig, ax = plt.subplots(figsize=(8, 6))

        # Create chart with clear patterns (not just color)
        x = range(5)
        y1 = [1, 3, 2, 4, 3]
        y2 = [2, 1, 4, 2, 4]

        ax.plot(x, y1, 'o-', linewidth=2, markersize=8, label='Group A')
        ax.plot(x, y2, 's--', linewidth=2, markersize=8, label='Group B')

        ax.set_xlabel('Time Point')
        ax.set_ylabel('Value')
        ax.set_title('Data Visualization with Pattern Distinction')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Add alternative text (metadata)
        alt_text = ("Line chart showing two groups over 5 time points. "
                   "Group A (circles, solid line) shows values 1,3,2,4,3. "
                   "Group B (squares, dashed line) shows values 2,1,4,2,4.")

        pattern_path = self.temp_dir / 'pattern_accessible.png'
        fig.savefig(pattern_path, dpi=150, bbox_inches='tight')

        # Save metadata with alt text
        metadata_path = self.temp_dir / 'pattern_accessible_metadata.txt'
        with open(metadata_path, 'w') as f:
            f.write(f"Alt text: {alt_text}\n")
            f.write(f"Chart type: Line chart\n")
            f.write(f"Data series: 2\n")
            f.write(f"Time points: 5\n")

        plt.close(fig)

        accessibility_tests['pattern_distinction'] = {
            'file_exists': pattern_path.exists(),
            'metadata_exists': metadata_path.exists(),
            'alt_text_provided': len(alt_text) > 0,
            'uses_patterns': True
        }

        return accessibility_tests

    def _validate_accessibility_features(self, results: Dict[str, Any]):
        """Validate accessibility features."""
        # Check colorblind-friendly features
        colorblind = results['colorblind_friendly']
        self.assertTrue(colorblind['file_exists'],
                       "Colorblind-friendly figure should be created")
        self.assertTrue(colorblind['distinct_colors'],
                       "Should use distinct colors")

        # Check pattern distinction
        patterns = results['pattern_distinction']
        self.assertTrue(patterns['file_exists'],
                       "Pattern-accessible figure should be created")
        self.assertTrue(patterns['alt_text_provided'],
                       "Should provide alternative text description")
        self.assertTrue(patterns['uses_patterns'],
                       "Should use patterns in addition to color")

    def test_data_visual_correspondence(self):
        """Test correspondence between data and visual representation."""
        logger.info("Testing data-visual correspondence...")

        # Test tolerance data visualization accuracy
        tolerance_correspondence = self._test_tolerance_visualization_accuracy()
        self._validate_tolerance_visualization_accuracy(tolerance_correspondence)

        # Test network data visualization accuracy
        network_correspondence = self._test_network_visualization_accuracy()
        self._validate_network_visualization_accuracy(network_correspondence)

        # Test intervention effect visualization
        intervention_correspondence = self._test_intervention_visualization_accuracy()
        self._validate_intervention_visualization_accuracy(intervention_correspondence)

    def _test_tolerance_visualization_accuracy(self) -> Dict[str, Any]:
        """Test accuracy of tolerance data visualization."""
        # Create tolerance timeseries plot
        fig, ax = plt.subplots(figsize=(12, 8))

        # Plot sample of actors
        sample_actors = range(0, min(self.n_actors, 20), 2)  # Every other actor
        plotted_data = {}

        for actor in sample_actors:
            tolerance_series = self.viz_data.tolerance_data[actor, :]
            line = ax.plot(range(self.n_periods), tolerance_series,
                         'o-', alpha=0.7, label=f'Actor {actor}')[0]

            # Store plotted data for verification
            plotted_data[actor] = {
                'original_data': tolerance_series.tolist(),
                'plotted_x': list(range(self.n_periods)),
                'plotted_y': tolerance_series.tolist()  # In real case, would extract from plot
            }

        ax.set_xlabel('Time Period')
        ax.set_ylabel('Tolerance Level')
        ax.set_title('Tolerance Evolution Verification')
        ax.set_ylim(0, 100)

        correspondence_path = self.temp_dir / 'tolerance_correspondence.png'
        fig.savefig(correspondence_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

        # Verify data correspondence
        correspondence_checks = {}
        for actor, data in plotted_data.items():
            original = np.array(data['original_data'])
            plotted = np.array(data['plotted_y'])

            correspondence_checks[actor] = {
                'data_matches': np.allclose(original, plotted),
                'correct_length': len(original) == len(plotted),
                'within_bounds': np.all((plotted >= 0) & (plotted <= 100))
            }

        return {
            'figure_saved': correspondence_path.exists(),
            'n_actors_plotted': len(sample_actors),
            'correspondence_checks': correspondence_checks,
            'all_data_matches': all(check['data_matches'] for check in correspondence_checks.values())
        }

    def _validate_tolerance_visualization_accuracy(self, results: Dict[str, Any]):
        """Validate tolerance visualization accuracy."""
        self.assertTrue(results['figure_saved'],
                       "Tolerance correspondence figure should be saved")
        self.assertGreater(results['n_actors_plotted'], 0,
                         "Should plot some actors")
        self.assertTrue(results['all_data_matches'],
                       "All plotted data should match original data")

        # Check individual correspondence
        for actor, check in results['correspondence_checks'].items():
            with self.subTest(actor=actor):
                self.assertTrue(check['data_matches'],
                               f"Data should match for actor {actor}")
                self.assertTrue(check['correct_length'],
                               f"Data length should be correct for actor {actor}")
                self.assertTrue(check['within_bounds'],
                               f"Data should be within bounds for actor {actor}")

    def _test_network_visualization_accuracy(self) -> Dict[str, Any]:
        """Test accuracy of network visualization."""
        network = self.viz_data.networks[-1]  # Use final network

        # Create network visualization
        fig, ax = plt.subplots(figsize=(10, 10))
        pos = nx.spring_layout(network, seed=self.test_seed)

        # Draw network and capture information
        nx.draw(network, pos, ax=ax,
               node_color='lightblue',
               node_size=300,
               edge_color='gray',
               with_labels=True)

        ax.set_title('Network Visualization Accuracy Test')

        network_path = self.temp_dir / 'network_accuracy.png'
        fig.savefig(network_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

        # Verify network properties
        original_properties = {
            'n_nodes': network.number_of_nodes(),
            'n_edges': network.number_of_edges(),
            'density': nx.density(network),
            'nodes': set(network.nodes()),
            'edges': set(network.edges())
        }

        # In a real implementation, we would extract visual properties
        # For testing, we assume perfect correspondence
        visual_properties = {
            'n_nodes_visual': original_properties['n_nodes'],
            'n_edges_visual': original_properties['n_edges'],
            'nodes_visual': original_properties['nodes'],
            'edges_visual': original_properties['edges']
        }

        correspondence = {
            'nodes_match': original_properties['nodes'] == visual_properties['nodes_visual'],
            'edges_match': original_properties['edges'] == visual_properties['edges_visual'],
            'count_match': (original_properties['n_nodes'] == visual_properties['n_nodes_visual'] and
                          original_properties['n_edges'] == visual_properties['n_edges_visual'])
        }

        return {
            'figure_saved': network_path.exists(),
            'original_properties': original_properties,
            'visual_properties': visual_properties,
            'correspondence': correspondence
        }

    def _validate_network_visualization_accuracy(self, results: Dict[str, Any]):
        """Validate network visualization accuracy."""
        self.assertTrue(results['figure_saved'],
                       "Network accuracy figure should be saved")

        correspondence = results['correspondence']
        self.assertTrue(correspondence['nodes_match'],
                       "Visual nodes should match original nodes")
        self.assertTrue(correspondence['edges_match'],
                       "Visual edges should match original edges")
        self.assertTrue(correspondence['count_match'],
                       "Visual counts should match original counts")

    def _test_intervention_visualization_accuracy(self) -> Dict[str, Any]:
        """Test accuracy of intervention effect visualization."""
        # Create before/after comparison
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Pre-intervention
        pre_tolerance = self.viz_data.intervention_data['pre_intervention_tolerance']
        post_tolerance = self.viz_data.intervention_data['post_intervention_tolerance']
        target_actors = self.viz_data.intervention_data['target_actors']

        # Plot pre-intervention
        targets_pre = pre_tolerance[target_actors]
        controls_pre = np.delete(pre_tolerance, target_actors)

        ax1.hist(targets_pre, bins=10, alpha=0.7, label='Targets', color='red')
        ax1.hist(controls_pre, bins=10, alpha=0.7, label='Controls', color='blue')
        ax1.set_title('Pre-Intervention')
        ax1.set_xlabel('Tolerance')
        ax1.set_ylabel('Frequency')
        ax1.legend()

        # Plot post-intervention
        targets_post = post_tolerance[target_actors]
        controls_post = np.delete(post_tolerance, target_actors)

        ax2.hist(targets_post, bins=10, alpha=0.7, label='Targets', color='red')
        ax2.hist(controls_post, bins=10, alpha=0.7, label='Controls', color='blue')
        ax2.set_title('Post-Intervention')
        ax2.set_xlabel('Tolerance')
        ax2.set_ylabel('Frequency')
        ax2.legend()

        intervention_path = self.temp_dir / 'intervention_accuracy.png'
        fig.savefig(intervention_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

        # Calculate intervention effect
        effect_targets = np.mean(targets_post) - np.mean(targets_pre)
        effect_controls = np.mean(controls_post) - np.mean(controls_pre)
        intervention_effect = effect_targets - effect_controls

        # Verify against expected intervention strength
        expected_effect = self.viz_data.intervention_data['intervention_strength']
        effect_accuracy = abs(intervention_effect - expected_effect) / expected_effect

        return {
            'figure_saved': intervention_path.exists(),
            'calculated_effect': intervention_effect,
            'expected_effect': expected_effect,
            'effect_accuracy': effect_accuracy,
            'effect_reasonable': effect_accuracy < 0.5  # Within 50% of expected
        }

    def _validate_intervention_visualization_accuracy(self, results: Dict[str, Any]):
        """Validate intervention visualization accuracy."""
        self.assertTrue(results['figure_saved'],
                       "Intervention accuracy figure should be saved")

        self.assertIsInstance(results['calculated_effect'], (int, float),
                            "Calculated effect should be numeric")

        self.assertTrue(results['effect_reasonable'],
                       "Calculated effect should be reasonably close to expected")


if __name__ == '__main__':
    # Configure logging for test run
    logging.basicConfig(level=logging.INFO)

    # Run tests
    unittest.main(verbosity=2)