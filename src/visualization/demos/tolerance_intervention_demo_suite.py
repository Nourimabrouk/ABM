"""
Tolerance Intervention Demo Suite
================================

Complete demonstration suite for tolerance intervention research showcasing
all visualization capabilities. This script creates a comprehensive demo
including animations, interactive dashboards, publication figures, and
3D visualizations for PhD defense presentation.

Demo Components:
1. Network evolution animations showing tolerance spread
2. Interactive parameter exploration dashboard
3. Publication-quality figures for dissertation
4. 3D tolerance network visualizations
5. Strategy comparison demonstrations
6. Export functionality for all components

Author: Claude Code - Visualization Virtuoso
Created: 2025-09-16
"""

import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import time
import warnings
warnings.filterwarnings('ignore')

# Custom visualization modules
from ..specialized.tolerance_intervention_viz import (
    ToleranceInterventionVisualizer,
    create_sample_tolerance_data
)
from ..interactive.tolerance_intervention_explorer import (
    ToleranceInterventionExplorer,
    create_tolerance_intervention_explorer
)
from ..specialized.tolerance_intervention_publication_figures import (
    ToleranceInterventionPublicationFigures
)

logger = logging.getLogger(__name__)

class ToleranceInterventionDemoSuite:
    """
    Comprehensive demonstration suite for tolerance intervention research.

    Creates all visualization outputs needed for PhD defense and publication,
    demonstrating state-of-the-art computational social science methods.
    """

    def __init__(self, output_dir: Path = None):
        """Initialize tolerance intervention demo suite."""
        self.output_dir = output_dir or Path("outputs/tolerance_intervention_demo")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Create output subdirectories
        self.demo_dirs = {
            'animations': self.output_dir / "animations",
            'interactive': self.output_dir / "interactive_dashboards",
            'publications': self.output_dir / "publication_figures",
            '3d_visualizations': self.output_dir / "3d_visualizations",
            'demo_data': self.output_dir / "demo_data",
            'exports': self.output_dir / "exports"
        }

        for directory in self.demo_dirs.values():
            directory.mkdir(exist_ok=True)

        # Initialize visualization components
        self.visualizer = ToleranceInterventionVisualizer(self.demo_dirs['animations'])
        self.publication_figures = ToleranceInterventionPublicationFigures(self.demo_dirs['publications'])

        # Demo state
        self.demo_results = {}
        self.created_files = []

        logger.info(f"Tolerance intervention demo suite initialized: {self.output_dir}")

    def run_complete_demo(self) -> Dict[str, Any]:
        """
        Run complete tolerance intervention demonstration suite.

        Returns:
            Dictionary containing all demo results and file paths
        """
        logger.info("🎯 Starting Complete Tolerance Intervention Demo Suite")

        print("\n" + "="*80)
        print("🎯 TOLERANCE INTERVENTION VISUALIZATION DEMO SUITE")
        print("   PhD Research: Promoting Interethnic Cooperation")
        print("="*80)

        start_time = time.time()

        try:
            # Step 1: Create demonstration data
            print("\n📊 Step 1: Creating demonstration data...")
            demo_data = self._create_comprehensive_demo_data()
            self.demo_results['demo_data'] = demo_data

            # Step 2: Generate network evolution animations
            print("\n🎬 Step 2: Creating network evolution animations...")
            animation_paths = self._create_tolerance_animations(demo_data)
            self.demo_results['animations'] = animation_paths

            # Step 3: Generate publication figures
            print("\n📄 Step 3: Creating publication-quality figures...")
            publication_paths = self._create_publication_figures(demo_data)
            self.demo_results['publications'] = publication_paths

            # Step 4: Create 3D visualizations
            print("\n🌐 Step 4: Creating 3D network visualizations...")
            viz_3d_paths = self._create_3d_visualizations(demo_data)
            self.demo_results['3d_visualizations'] = viz_3d_paths

            # Step 5: Generate strategy comparison demos
            print("\n📊 Step 5: Creating strategy comparison demonstrations...")
            comparison_paths = self._create_strategy_comparisons(demo_data)
            self.demo_results['strategy_comparisons'] = comparison_paths

            # Step 6: Create interactive dashboard demo
            print("\n🖥️  Step 6: Preparing interactive dashboard...")
            dashboard_info = self._prepare_interactive_dashboard()
            self.demo_results['interactive_dashboard'] = dashboard_info

            # Step 7: Export demo package
            print("\n📦 Step 7: Creating demo export package...")
            export_info = self._create_demo_export_package()
            self.demo_results['export_package'] = export_info

            # Generate demo summary
            self._generate_demo_summary()

            elapsed_time = time.time() - start_time
            print(f"\n✅ Demo suite completed successfully in {elapsed_time:.1f} seconds!")
            print(f"📁 All outputs saved to: {self.output_dir}")

            return self.demo_results

        except Exception as e:
            logger.error(f"Demo suite failed: {e}")
            print(f"\n❌ Demo suite failed: {e}")
            raise

    def _create_comprehensive_demo_data(self) -> Dict[str, Any]:
        """Create comprehensive demonstration data for all visualizations."""
        print("   📈 Generating tolerance intervention simulation data...")

        # Create multiple scenarios for demonstration
        scenarios = {
            'central_targeting': {
                'n_agents': 40,
                'n_timepoints': 25,
                'intervention_targets': [0, 1, 2],  # High centrality nodes
                'intervention_magnitude': 0.6,
                'description': 'Central node targeting strategy'
            },
            'peripheral_targeting': {
                'n_agents': 40,
                'n_timepoints': 25,
                'intervention_targets': [35, 36, 37],  # Low centrality nodes
                'intervention_magnitude': 0.6,
                'description': 'Peripheral node targeting strategy'
            },
            'random_targeting': {
                'n_agents': 40,
                'n_timepoints': 25,
                'intervention_targets': [15, 22, 31],  # Random selection
                'intervention_magnitude': 0.6,
                'description': 'Random targeting baseline'
            },
            'clustered_targeting': {
                'n_agents': 40,
                'n_timepoints': 25,
                'intervention_targets': [8, 9, 10],  # Clustered selection
                'intervention_magnitude': 0.6,
                'description': 'Clustered targeting strategy'
            }
        }

        demo_data = {'scenarios': {}}

        for scenario_name, config in scenarios.items():
            print(f"     🔄 Creating {scenario_name} scenario...")

            # Generate scenario data
            networks, tolerances, cooperations, targets = create_sample_tolerance_data(
                n_agents=config['n_agents'],
                n_timepoints=config['n_timepoints'],
                intervention_targets=config['intervention_targets']
            )

            demo_data['scenarios'][scenario_name] = {
                'networks': networks,
                'tolerance_data': tolerances,
                'cooperation_data': cooperations,
                'intervention_targets': targets,
                'config': config
            }

        # Save demo data
        self._save_demo_data(demo_data)

        print(f"   ✅ Created {len(scenarios)} demonstration scenarios")
        return demo_data

    def _save_demo_data(self, demo_data: Dict[str, Any]):
        """Save demonstration data for reuse."""
        try:
            import pickle
            data_file = self.demo_dirs['demo_data'] / "tolerance_demo_data.pkl"

            # Prepare data for saving (remove non-serializable objects)
            save_data = {}
            for scenario_name, scenario_data in demo_data['scenarios'].items():
                save_data[scenario_name] = {
                    'tolerance_data': scenario_data['tolerance_data'],
                    'cooperation_data': scenario_data['cooperation_data'],
                    'intervention_targets': scenario_data['intervention_targets'],
                    'config': scenario_data['config']
                }
                # Save network data separately as GraphML
                for i, network in enumerate(scenario_data['networks']):
                    network_file = self.demo_dirs['demo_data'] / f"{scenario_name}_network_{i:03d}.graphml"
                    import networkx as nx
                    nx.write_graphml(network, network_file)

            with open(data_file, 'wb') as f:
                pickle.dump(save_data, f)

            print(f"     💾 Demo data saved to {data_file}")

        except Exception as e:
            logger.warning(f"Could not save demo data: {e}")

    def _create_tolerance_animations(self, demo_data: Dict[str, Any]) -> Dict[str, str]:
        """Create tolerance intervention network evolution animations."""
        animation_paths = {}

        # Create primary demonstration animation
        print("     🎬 Creating main tolerance spread animation...")
        central_scenario = demo_data['scenarios']['central_targeting']

        main_animation_path = self.visualizer.create_tolerance_spread_animation(
            central_scenario['networks'],
            central_scenario['tolerance_data'],
            central_scenario['intervention_targets'],
            central_scenario['cooperation_data'],
            save_filename="main_tolerance_intervention_demo"
        )
        animation_paths['main_demo'] = main_animation_path
        self.created_files.append(main_animation_path)

        # Create strategy comparison animation
        print("     🔄 Creating strategy comparison animations...")
        for strategy_name, scenario_data in demo_data['scenarios'].items():
            if strategy_name != 'central_targeting':  # Already created above
                anim_path = self.visualizer.create_tolerance_spread_animation(
                    scenario_data['networks'],
                    scenario_data['tolerance_data'],
                    scenario_data['intervention_targets'],
                    scenario_data['cooperation_data'],
                    save_filename=f"tolerance_intervention_{strategy_name}"
                )
                animation_paths[strategy_name] = anim_path
                self.created_files.append(anim_path)

        print(f"     ✅ Created {len(animation_paths)} tolerance intervention animations")
        return animation_paths

    def _create_publication_figures(self, demo_data: Dict[str, Any]) -> Dict[str, List[str]]:
        """Create publication-quality figures for tolerance intervention research."""
        publication_paths = {}

        # Figure 1: Conceptual Model
        print("     📊 Creating Figure 1: Conceptual Model...")
        fig1_paths = self.publication_figures.create_figure_1_conceptual_model()
        publication_paths['conceptual_model'] = fig1_paths
        self.created_files.extend(fig1_paths)

        # Figure 2: Network Structure Examples
        print("     🌐 Creating Figure 2: Network Structure Examples...")
        fig2_paths = self.publication_figures.create_figure_2_network_examples()
        publication_paths['network_examples'] = fig2_paths
        self.created_files.extend(fig2_paths)

        # Additional figures using demo data
        print("     📈 Creating strategy comparison figure...")
        comparison_results = self._prepare_strategy_comparison_data(demo_data)
        fig3_paths = self.visualizer.create_intervention_comparison_figure(
            comparison_results['central_targeting'],
            comparison_results['peripheral_targeting'],
            comparison_results['random_targeting'],
            save_filename="figure_03_strategy_comparison"
        )
        publication_paths['strategy_comparison'] = [fig3_paths]
        self.created_files.append(fig3_paths)

        print(f"     ✅ Created {len(publication_paths)} publication figure sets")
        return publication_paths

    def _prepare_strategy_comparison_data(self, demo_data: Dict[str, Any]) -> Dict[str, Dict]:
        """Prepare data for strategy comparison visualization."""
        comparison_data = {}

        for strategy_name, scenario_data in demo_data['scenarios'].items():
            # Calculate final metrics
            final_tolerances = scenario_data['tolerance_data'][-1]
            final_network = scenario_data['networks'][-1]
            final_cooperations = scenario_data['cooperation_data'][-1] if scenario_data['cooperation_data'] else {}

            # Create tolerance evolution
            tolerance_evolution = [np.mean(list(tol_data.values()))
                                 for tol_data in scenario_data['tolerance_data']]

            comparison_data[strategy_name] = {
                'tolerance_evolution': tolerance_evolution,
                'final_network': final_network,
                'final_tolerances': final_tolerances,
                'final_cooperation': np.mean(list(final_cooperations.values())) if final_cooperations else 0,
                'final_interethnic_edges': self._count_interethnic_edges(final_network),
                'intervention_targets': scenario_data['intervention_targets'],
                'intervention_time': 5,
                'intervention_duration': 3
            }

        return comparison_data

    def _count_interethnic_edges(self, network):
        """Count edges between different ethnic groups."""
        count = 0
        for u, v in network.edges():
            u_ethnicity = network.nodes[u].get('ethnicity', 'majority')
            v_ethnicity = network.nodes[v].get('ethnicity', 'majority')
            if u_ethnicity != v_ethnicity:
                count += 1
        return count

    def _create_3d_visualizations(self, demo_data: Dict[str, Any]) -> Dict[str, str]:
        """Create 3D tolerance network visualizations."""
        viz_3d_paths = {}

        # Create 3D visualization for central targeting scenario
        print("     🌐 Creating 3D tolerance network visualization...")
        central_scenario = demo_data['scenarios']['central_targeting']

        final_network = central_scenario['networks'][-1]
        final_tolerances = central_scenario['tolerance_data'][-1]
        final_cooperations = central_scenario['cooperation_data'][-1] if central_scenario['cooperation_data'] else {}

        viz_3d_path = self.visualizer.create_3d_tolerance_network_visualization(
            final_network,
            final_tolerances,
            final_cooperations,
            central_scenario['intervention_targets'],
            save_filename="tolerance_3d_network_demo"
        )
        viz_3d_paths['main_3d'] = viz_3d_path
        self.created_files.append(viz_3d_path)

        # Create additional 3D visualizations for comparison
        for strategy_name in ['peripheral_targeting', 'random_targeting']:
            scenario_data = demo_data['scenarios'][strategy_name]

            viz_path = self.visualizer.create_3d_tolerance_network_visualization(
                scenario_data['networks'][-1],
                scenario_data['tolerance_data'][-1],
                scenario_data['cooperation_data'][-1] if scenario_data['cooperation_data'] else {},
                scenario_data['intervention_targets'],
                save_filename=f"tolerance_3d_{strategy_name}"
            )
            viz_3d_paths[strategy_name] = viz_path
            self.created_files.append(viz_path)

        print(f"     ✅ Created {len(viz_3d_paths)} 3D tolerance network visualizations")
        return viz_3d_paths

    def _create_strategy_comparisons(self, demo_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create comprehensive strategy comparison demonstrations."""
        comparison_paths = {}

        print("     📊 Analyzing intervention strategy effectiveness...")

        # Calculate effectiveness metrics for each strategy
        strategy_metrics = {}
        for strategy_name, scenario_data in demo_data['scenarios'].items():
            metrics = self._calculate_strategy_metrics(scenario_data)
            strategy_metrics[strategy_name] = metrics

        # Save strategy comparison data
        comparison_file = self.demo_dirs['exports'] / "strategy_comparison_metrics.csv"
        self._save_strategy_metrics(strategy_metrics, comparison_file)
        comparison_paths['metrics_file'] = str(comparison_file)
        self.created_files.append(str(comparison_file))

        # Create summary report
        report_file = self.demo_dirs['exports'] / "strategy_effectiveness_report.md"
        self._create_strategy_report(strategy_metrics, report_file)
        comparison_paths['report_file'] = str(report_file)
        self.created_files.append(str(report_file))

        print(f"     ✅ Created strategy comparison analysis")
        return comparison_paths

    def _calculate_strategy_metrics(self, scenario_data: Dict[str, Any]) -> Dict[str, float]:
        """Calculate effectiveness metrics for a strategy."""
        tolerance_data = scenario_data['tolerance_data']
        networks = scenario_data['networks']
        intervention_targets = scenario_data['intervention_targets']

        # Calculate metrics
        initial_tolerance = np.mean(list(tolerance_data[0].values()))
        final_tolerance = np.mean(list(tolerance_data[-1].values()))
        tolerance_gain = final_tolerance - initial_tolerance

        # Intervention group vs control group
        final_tolerances = tolerance_data[-1]
        intervention_tolerance = np.mean([final_tolerances[t] for t in intervention_targets])
        control_tolerance = np.mean([v for k, v in final_tolerances.items()
                                   if k not in intervention_targets])
        intervention_effect = intervention_tolerance - control_tolerance

        # Network metrics
        final_network = networks[-1]
        final_density = final_network.number_of_edges() / (len(final_network) * (len(final_network) - 1) / 2)
        interethnic_edges = self._count_interethnic_edges(final_network)

        return {
            'tolerance_gain': tolerance_gain,
            'intervention_effect': intervention_effect,
            'final_tolerance': final_tolerance,
            'final_density': final_density,
            'interethnic_edges': interethnic_edges,
            'network_size': len(final_network)
        }

    def _save_strategy_metrics(self, strategy_metrics: Dict[str, Dict], output_file: Path):
        """Save strategy metrics to CSV file."""
        import pandas as pd

        # Convert to DataFrame
        df = pd.DataFrame(strategy_metrics).T
        df.index.name = 'strategy'
        df.to_csv(output_file)

    def _create_strategy_report(self, strategy_metrics: Dict[str, Dict], output_file: Path):
        """Create strategy effectiveness report."""
        # Find best strategy
        best_strategy = max(strategy_metrics.keys(),
                          key=lambda s: strategy_metrics[s]['tolerance_gain'])

        report_content = f"""# Tolerance Intervention Strategy Effectiveness Report

## Executive Summary

This report analyzes the effectiveness of different targeting strategies for tolerance interventions aimed at promoting interethnic cooperation in social networks.

## Strategy Comparison Results

### Best Performing Strategy: {best_strategy.replace('_', ' ').title()}

**Key Findings:**
- **Tolerance Gain:** {strategy_metrics[best_strategy]['tolerance_gain']:.3f}
- **Intervention Effect:** {strategy_metrics[best_strategy]['intervention_effect']:.3f}
- **Final Tolerance Level:** {strategy_metrics[best_strategy]['final_tolerance']:.3f}
- **Interethnic Edges Created:** {strategy_metrics[best_strategy]['interethnic_edges']}

## Detailed Results

| Strategy | Tolerance Gain | Intervention Effect | Final Tolerance | Interethnic Edges |
|----------|----------------|-------------------|-----------------|-------------------|
"""

        for strategy, metrics in strategy_metrics.items():
            strategy_name = strategy.replace('_', ' ').title()
            report_content += f"| {strategy_name} | {metrics['tolerance_gain']:.3f} | {metrics['intervention_effect']:.3f} | {metrics['final_tolerance']:.3f} | {metrics['interethnic_edges']} |\n"

        report_content += f"""
## Policy Recommendations

1. **Primary Recommendation:** Use {best_strategy.replace('_', ' ').lower()} for maximum tolerance intervention effectiveness.

2. **Implementation Considerations:**
   - Target population size: 3-5 individuals per intervention
   - Intervention magnitude: 0.5-0.7 tolerance units
   - Duration: 3-5 time periods for sustained effect

3. **Expected Outcomes:**
   - Tolerance increase: {strategy_metrics[best_strategy]['tolerance_gain']:.1%}
   - Interethnic cooperation ties: {strategy_metrics[best_strategy]['interethnic_edges']} new connections
   - Network density improvement: {strategy_metrics[best_strategy]['final_density']:.3f}

## Technical Details

This analysis was conducted using agent-based modeling with the following parameters:
- Network size: {strategy_metrics[best_strategy]['network_size']} agents
- Simulation duration: 25 time periods
- Intervention period: Time steps 5-8
- Ethnic composition: 33% minority, 67% majority

## Generated by Tolerance Intervention Demo Suite
Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

        with open(output_file, 'w') as f:
            f.write(report_content)

    def _prepare_interactive_dashboard(self) -> Dict[str, str]:
        """Prepare interactive dashboard demonstration."""
        print("     🖥️  Setting up interactive tolerance intervention explorer...")

        # Create dashboard launcher script
        launcher_script = self.demo_dirs['interactive'] / "launch_tolerance_explorer.py"
        launcher_content = '''#!/usr/bin/env python
"""
Tolerance Intervention Explorer Launcher
=======================================

Launch the interactive tolerance intervention explorer dashboard.
This script provides a web-based interface for exploring tolerance
intervention effects on social networks.

Usage:
    python launch_tolerance_explorer.py

Then open your browser to: http://localhost:5007
"""

import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.visualization.interactive.tolerance_intervention_explorer import (
    create_tolerance_intervention_explorer
)

if __name__ == "__main__":
    print("🎯 Starting Tolerance Intervention Explorer...")
    print("📱 Dashboard will open at: http://localhost:5007")
    print("🎮 Use the controls to explore tolerance intervention effects!")
    print("🛑 Use Ctrl+C to stop the server")

    try:
        # Create and serve dashboard
        explorer = create_tolerance_intervention_explorer()
        explorer.serve(port=5007, show=True)
    except KeyboardInterrupt:
        print("\\n🛑 Tolerance intervention explorer stopped")
    except Exception as e:
        print(f"❌ Error: {e}")
'''

        with open(launcher_script, 'w') as f:
            f.write(launcher_content)

        # Create dashboard README
        readme_file = self.demo_dirs['interactive'] / "README.md"
        readme_content = """# Interactive Tolerance Intervention Explorer

## Overview

The Tolerance Intervention Explorer is an interactive web-based dashboard for exploring the effects of tolerance interventions on social networks and interethnic cooperation.

## Features

- **Real-time Parameter Adjustment**: Use sliders to modify intervention parameters
- **Network Visualization**: See live network evolution with tolerance coloring
- **Strategy Comparison**: Compare different targeting approaches
- **Export Functionality**: Save scenarios and results
- **What-If Analysis**: Test custom intervention designs

## Getting Started

1. Run the launcher script:
   ```bash
   python launch_tolerance_explorer.py
   ```

2. Open your browser to: http://localhost:5007

3. Use the parameter controls on the left to design interventions

4. Click "🚀 Run Simulation" to see results

5. Explore different tabs for various visualizations

## Parameter Guide

### Network Configuration
- **Number of Agents**: Size of social network (15-100)
- **Time Steps**: Duration of simulation (10-50)
- **Network Density**: How connected the network is (0.05-0.4)
- **Minority Proportion**: Percentage of minority group members (0.1-0.5)

### Intervention Design
- **Tolerance Change Magnitude**: Strength of intervention effect (0.1-1.0)
- **Target Population Size**: Number of agents to target (1-10)
- **Intervention Start Time**: When intervention begins (1-15)
- **Intervention Duration**: How long intervention lasts (1-10)
- **Targeting Strategy**: How to select targets (Central/Peripheral/Random/Clustered)

### Social Influence
- **Influence Threshold**: Minimum influence needed for change (0.1-0.8)
- **Homophily Strength**: Tendency to connect within ethnic groups (0.0-0.8)
- **Tolerance Persistence**: How long tolerance changes last (0.3-1.0)

### Complex Contagion
- **Enable Complex Contagion**: Require multiple exposures for change
- **Contagion Threshold**: Minimum exposure level needed (0.2-0.8)
- **Multiple Exposures Required**: Number of exposures needed (1-5)

## Research Context

This tool demonstrates computational models of tolerance interventions designed to promote interethnic cooperation through social network mechanisms, developed for PhD research in computational social science.
"""

        with open(readme_file, 'w') as f:
            f.write(readme_content)

        self.created_files.extend([str(launcher_script), str(readme_file)])

        return {
            'launcher_script': str(launcher_script),
            'readme_file': str(readme_file),
            'dashboard_url': 'http://localhost:5007'
        }

    def _create_demo_export_package(self) -> Dict[str, str]:
        """Create comprehensive demo export package."""
        print("     📦 Creating demo export package...")

        # Create main demo README
        main_readme = self.output_dir / "README.md"
        readme_content = f"""# Tolerance Intervention Visualization Demo Suite

## Overview

This demo suite showcases state-of-the-art visualizations for PhD research on tolerance interventions promoting interethnic cooperation through social network mechanisms.

## Created Files

### 🎬 Animations ({len(self.demo_results.get('animations', {}))})
Network evolution animations showing tolerance spread and cooperation emergence:
"""

        for name, path in self.demo_results.get('animations', {}).items():
            readme_content += f"- `{Path(path).name}`: {name.replace('_', ' ').title()}\n"

        readme_content += f"""
### 📄 Publication Figures ({sum(len(paths) for paths in self.demo_results.get('publications', {}).values())})
Publication-quality figures for dissertation and JASSS publication:
"""

        for fig_name, paths in self.demo_results.get('publications', {}).items():
            readme_content += f"- **{fig_name.replace('_', ' ').title()}**: {len(paths)} formats\n"

        readme_content += f"""
### 🌐 3D Visualizations ({len(self.demo_results.get('3d_visualizations', {}))})
Interactive 3D network visualizations:
"""

        for name, path in self.demo_results.get('3d_visualizations', {}).items():
            readme_content += f"- `{Path(path).name}`: {name.replace('_', ' ').title()}\n"

        readme_content += f"""
### 🖥️ Interactive Dashboard
Web-based tolerance intervention explorer:
- **Launcher**: `{self.demo_results.get('interactive_dashboard', {}).get('launcher_script', 'N/A')}`
- **URL**: {self.demo_results.get('interactive_dashboard', {}).get('dashboard_url', 'N/A')}

### 📊 Analysis Reports
Strategy comparison and effectiveness analysis:
"""

        for name, path in self.demo_results.get('strategy_comparisons', {}).items():
            if isinstance(path, str):
                readme_content += f"- `{Path(path).name}`: {name.replace('_', ' ').title()}\n"

        readme_content += """
## Research Context

This work contributes to computational social science methodology for analyzing tolerance interventions and intergroup contact effects in educational and community settings.

**Publication Target**: Journal of Artificial Societies and Social Simulation (JASSS)
**Research Level**: PhD Dissertation
**Methods**: Agent-Based Modeling, Social Network Analysis, Statistical Validation

## Usage Instructions

1. **View Animations**: Open MP4 files in any media player
2. **Explore 3D Visualizations**: Open HTML files in web browser
3. **Run Interactive Dashboard**: Execute Python launcher script
4. **Review Publication Figures**: View PNG/PDF files for dissertation
5. **Analyze Results**: Review CSV and markdown reports

## Technical Details

- **Programming Language**: Python 3.8+
- **Key Libraries**: NetworkX, Matplotlib, Panel, Plotly, NumPy, Pandas
- **Visualization Types**: Static plots, animations, interactive dashboards, 3D networks
- **Output Formats**: PNG, PDF, SVG, MP4, GIF, HTML, CSV, Markdown

## Contact

For questions about this research or visualization methodology, please contact the research team.

---
*Generated by Tolerance Intervention Demo Suite*
*Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""

        with open(main_readme, 'w') as f:
            f.write(readme_content)

        # Create file manifest
        manifest_file = self.demo_dirs['exports'] / "file_manifest.txt"
        with open(manifest_file, 'w') as f:
            f.write("Tolerance Intervention Demo Suite - File Manifest\\n")
            f.write("=" * 60 + "\\n\\n")
            for i, file_path in enumerate(self.created_files, 1):
                f.write(f"{i:3d}. {file_path}\\n")

        self.created_files.extend([str(main_readme), str(manifest_file)])

        return {
            'main_readme': str(main_readme),
            'file_manifest': str(manifest_file),
            'total_files': len(self.created_files)
        }

    def _generate_demo_summary(self):
        """Generate comprehensive demo summary."""
        print("\\n" + "="*80)
        print("✅ TOLERANCE INTERVENTION DEMO SUITE COMPLETED")
        print("="*80)

        print(f"📁 Output Directory: {self.output_dir}")
        print(f"📊 Total Files Created: {len(self.created_files)}")

        print(f"\\n🎬 Animations: {len(self.demo_results.get('animations', {}))}")
        for name in self.demo_results.get('animations', {}):
            print(f"   - {name.replace('_', ' ').title()}")

        print(f"\\n📄 Publication Figures: {len(self.demo_results.get('publications', {}))}")
        for name in self.demo_results.get('publications', {}):
            print(f"   - {name.replace('_', ' ').title()}")

        print(f"\\n🌐 3D Visualizations: {len(self.demo_results.get('3d_visualizations', {}))}")
        for name in self.demo_results.get('3d_visualizations', {}):
            print(f"   - {name.replace('_', ' ').title()}")

        print("\\n🖥️  Interactive Dashboard:")
        dashboard_info = self.demo_results.get('interactive_dashboard', {})
        print(f"   - Launch: python {Path(dashboard_info.get('launcher_script', '')).name}")
        print(f"   - URL: {dashboard_info.get('dashboard_url', 'N/A')}")

        print("\\n📊 Analysis & Exports:")
        for name in self.demo_results.get('strategy_comparisons', {}):
            print(f"   - {name.replace('_', ' ').title()}")

        print(f"\\n🎯 DEMO HIGHLIGHTS:")
        print("   ✨ State-of-the-art tolerance intervention visualizations")
        print("   🎬 Stunning network evolution animations")
        print("   📊 Publication-ready figures for PhD defense")
        print("   🌐 Interactive 3D network explorations")
        print("   🖥️  Real-time parameter exploration dashboard")
        print("   📈 Comprehensive strategy effectiveness analysis")

        print(f"\\n🚀 READY FOR:")
        print("   🎓 PhD Dissertation Defense")
        print("   📑 JASSS Publication Submission")
        print("   👥 Conference Presentations")
        print("   💡 Policy Recommendation Reports")

    def launch_interactive_dashboard(self, port: int = 5007):
        """Launch the interactive tolerance intervention explorer."""
        print(f"🎯 Launching Tolerance Intervention Explorer on port {port}...")

        try:
            explorer = create_tolerance_intervention_explorer()
            return explorer.serve(port=port, show=True)
        except Exception as e:
            logger.error(f"Failed to launch dashboard: {e}")
            print(f"❌ Dashboard launch failed: {e}")
            raise


def run_tolerance_intervention_demo():
    """Main function to run the complete tolerance intervention demo suite."""
    logging.basicConfig(level=logging.INFO)

    # Create and run demo suite
    demo_suite = ToleranceInterventionDemoSuite()
    results = demo_suite.run_complete_demo()

    return demo_suite, results


if __name__ == "__main__":
    print("🎯 Starting Tolerance Intervention Visualization Demo Suite...")

    try:
        demo_suite, results = run_tolerance_intervention_demo()

        print("\\n" + "="*60)
        print("🎊 DEMO SUITE COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("\\nChoose your next action:")
        print("1. View animations and visualizations")
        print("2. Launch interactive dashboard")
        print("3. Review publication figures")
        print("4. Explore 3D visualizations")
        print("5. Exit")

        while True:
            choice = input("\\nEnter your choice (1-5): ").strip()

            if choice == "1":
                print(f"📁 Animations and visualizations saved to:")
                print(f"   {demo_suite.output_dir}")
                break
            elif choice == "2":
                print("🚀 Launching interactive dashboard...")
                demo_suite.launch_interactive_dashboard()
                break
            elif choice == "3":
                print(f"📄 Publication figures saved to:")
                print(f"   {demo_suite.demo_dirs['publications']}")
                break
            elif choice == "4":
                print(f"🌐 3D visualizations saved to:")
                print(f"   {demo_suite.demo_dirs['3d_visualizations']}")
                break
            elif choice == "5":
                print("👋 Thank you for exploring tolerance intervention visualizations!")
                break
            else:
                print("❌ Invalid choice. Please enter 1-5.")

    except KeyboardInterrupt:
        print("\\n🛑 Demo suite interrupted by user")
    except Exception as e:
        print(f"\\n❌ Demo suite failed: {e}")
        import traceback
        traceback.print_exc()