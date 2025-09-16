#!/usr/bin/env python
"""
Tolerance Intervention Visualization Showcase
============================================

🎯 VISUALIZATION VIRTUOSO: STATE-OF-THE-ART TOLERANCE INTERVENTION VISUALS

Complete showcase of stunning visualizations and interactive demos for PhD research
on tolerance interventions promoting interethnic cooperation through social networks.

This showcase creates:
1. 🎬 Network Evolution Animations - Tolerance spread and cooperation emergence
2. 🖥️  Interactive Dashboard - Real-time parameter exploration
3. 📄 Publication Figures - PhD defense and JASSS publication ready
4. 🌐 3D Visualizations - Interactive network explorations
5. 📊 Strategy Comparisons - Targeting effectiveness analysis

Research Context:
- Visualize tolerance diffusion through friendship networks
- Compare intervention designs (targeting strategies, contagion types)
- Show micro-macro dynamics: individual tolerance → network cooperation
- Present results for PhD defense and JASSS publication

Author: Claude Code - Visualization Virtuoso
Created: 2025-09-16
"""

import sys
import logging
from pathlib import Path

# Add src to Python path
project_root = Path(__file__).parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

def main():
    """Main function to run tolerance intervention visualization showcase."""

    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║  🎯 TOLERANCE INTERVENTION VISUALIZATION SHOWCASE                           ║
║                                                                              ║
║     Stunning Visualizations for PhD Research on Interethnic Cooperation     ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

🔬 RESEARCH CONTEXT:
   📖 PhD Dissertation: Tolerance Interventions in Social Networks
   🎯 Target Journal: Journal of Artificial Societies and Social Simulation (JASSS)
   💡 Innovation: Agent-Based Models + Network Visualization + Interactive Demos

🚀 VISUALIZATION DELIVERABLES:
   🎬 Network Evolution Animations (tolerance spread dynamics)
   🖥️  Interactive Parameter Explorer (real-time simulations)
   📄 Publication Figures (300+ DPI, multiple formats)
   🌐 3D Network Visualizations (interactive explorations)
   📊 Strategy Effectiveness Analysis (policy recommendations)

🎊 DEMO FEATURES:
   ✨ What-If Scenario Explorer
   📈 Real-time network updates
   🎨 Color-coded tolerance levels
   🔄 Comparative strategy analysis
   💾 Export functionality
   📱 Web-based dashboard
    """)

    print("\nChoose your exploration path:")
    print("1. 🚀 Run Complete Demo Suite (create all visualizations)")
    print("2. 🖥️  Launch Interactive Dashboard Only")
    print("3. 🎬 Create Tolerance Spread Animations")
    print("4. 📄 Generate Publication Figures")
    print("5. 🌐 Create 3D Network Visualizations")
    print("6. ℹ️  View Documentation")
    print("7. ❌ Exit")

    while True:
        try:
            choice = input("\n🎯 Enter your choice (1-7): ").strip()

            if choice == "1":
                run_complete_demo_suite()
                break
            elif choice == "2":
                launch_interactive_dashboard()
                break
            elif choice == "3":
                create_tolerance_animations()
                break
            elif choice == "4":
                generate_publication_figures()
                break
            elif choice == "5":
                create_3d_visualizations()
                break
            elif choice == "6":
                show_documentation()
                break
            elif choice == "7":
                print("👋 Thank you for exploring tolerance intervention visualizations!")
                break
            else:
                print("❌ Invalid choice. Please enter 1-7.")

        except KeyboardInterrupt:
            print("\n🛑 Showcase interrupted by user")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()

def run_complete_demo_suite():
    """Run the complete tolerance intervention demo suite."""
    print("\n🚀 Launching Complete Tolerance Intervention Demo Suite...")
    print("📊 This will create all visualizations and demonstrations.")

    try:
        from visualization.demos.tolerance_intervention_demo_suite import run_tolerance_intervention_demo

        demo_suite, results = run_tolerance_intervention_demo()

        print("\n🎊 Complete demo suite finished successfully!")
        print(f"📁 All outputs saved to: {demo_suite.output_dir}")

        # Ask if user wants to launch dashboard
        launch_choice = input("\n🖥️  Would you like to launch the interactive dashboard? (y/n): ").strip().lower()
        if launch_choice in ['y', 'yes']:
            demo_suite.launch_interactive_dashboard()

    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Make sure you're running from the project root directory")
    except Exception as e:
        print(f"❌ Demo suite failed: {e}")
        import traceback
        traceback.print_exc()

def launch_interactive_dashboard():
    """Launch only the interactive tolerance intervention explorer."""
    print("\n🖥️  Launching Interactive Tolerance Intervention Explorer...")
    print("📱 Dashboard will open at: http://localhost:5007")
    print("🎮 Use the controls to explore tolerance intervention effects!")

    try:
        from visualization.interactive.tolerance_intervention_explorer import create_tolerance_intervention_explorer

        explorer = create_tolerance_intervention_explorer()
        print("🚀 Starting dashboard server...")
        explorer.serve(port=5007, show=True)

    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Make sure you're running from the project root directory")
    except Exception as e:
        print(f"❌ Dashboard launch failed: {e}")

def create_tolerance_animations():
    """Create tolerance spread animations only."""
    print("\n🎬 Creating Tolerance Spread Animations...")

    try:
        from visualization.specialized.tolerance_intervention_viz import (
            ToleranceInterventionVisualizer,
            create_sample_tolerance_data
        )

        # Create visualizer
        visualizer = ToleranceInterventionVisualizer()

        # Create sample data
        print("📊 Generating sample tolerance intervention data...")
        networks, tolerances, cooperations, targets = create_sample_tolerance_data(
            n_agents=30, n_timepoints=20, intervention_targets=[0, 1, 2]
        )

        # Create animation
        print("🎬 Creating tolerance spread animation...")
        animation_path = visualizer.create_tolerance_spread_animation(
            networks, tolerances, targets, cooperations,
            save_filename="tolerance_intervention_showcase"
        )

        print(f"✅ Animation created successfully!")
        print(f"📁 Saved to: {animation_path}")

    except ImportError as e:
        print(f"❌ Import error: {e}")
    except Exception as e:
        print(f"❌ Animation creation failed: {e}")

def generate_publication_figures():
    """Generate publication-quality figures only."""
    print("\n📄 Creating Publication-Quality Figures...")

    try:
        from visualization.specialized.tolerance_intervention_publication_figures import (
            ToleranceInterventionPublicationFigures
        )

        # Create figure generator
        figure_generator = ToleranceInterventionPublicationFigures()

        # Create figures
        print("📊 Creating Figure 1: Conceptual Model...")
        fig1_paths = figure_generator.create_figure_1_conceptual_model()

        print("🌐 Creating Figure 2: Network Structure Examples...")
        fig2_paths = figure_generator.create_figure_2_network_examples()

        print("✅ Publication figures created successfully!")
        print(f"📁 Figure 1 saved in {len(fig1_paths)} formats")
        print(f"📁 Figure 2 saved in {len(fig2_paths)} formats")

    except ImportError as e:
        print(f"❌ Import error: {e}")
    except Exception as e:
        print(f"❌ Figure generation failed: {e}")

def create_3d_visualizations():
    """Create 3D network visualizations only."""
    print("\n🌐 Creating 3D Tolerance Network Visualizations...")

    try:
        from visualization.specialized.tolerance_intervention_viz import (
            ToleranceInterventionVisualizer,
            create_sample_tolerance_data
        )
        import networkx as nx

        # Create visualizer
        visualizer = ToleranceInterventionVisualizer()

        # Create sample data
        print("📊 Generating sample tolerance network data...")
        networks, tolerances, cooperations, targets = create_sample_tolerance_data(
            n_agents=25, n_timepoints=15, intervention_targets=[0, 1, 2]
        )

        # Use final network state
        final_network = networks[-1]
        final_tolerances = tolerances[-1]
        final_cooperations = cooperations[-1] if cooperations else {}

        # Create 3D visualization
        print("🌐 Creating 3D tolerance network visualization...")
        viz_3d_path = visualizer.create_3d_tolerance_network_visualization(
            final_network, final_tolerances, final_cooperations, targets,
            save_filename="tolerance_3d_showcase"
        )

        print(f"✅ 3D visualization created successfully!")
        print(f"📁 Saved to: {viz_3d_path}")
        print("💡 Open the HTML file in your browser to explore the 3D network")

    except ImportError as e:
        print(f"❌ Import error: {e}")
    except Exception as e:
        print(f"❌ 3D visualization creation failed: {e}")

def show_documentation():
    """Show comprehensive documentation and research context."""
    print("""
📚 TOLERANCE INTERVENTION VISUALIZATION DOCUMENTATION
══════════════════════════════════════════════════════

🎓 RESEARCH CONTEXT:
   This visualization suite supports PhD research on tolerance interventions
   designed to promote interethnic cooperation through social network mechanisms.

🔬 THEORETICAL FRAMEWORK:
   • Tolerance interventions change individual tolerance levels
   • Social influence spreads tolerance through network connections
   • Higher tolerance leads to increased interethnic cooperation
   • Cooperation strengthens interethnic network ties over time

🎯 KEY RESEARCH QUESTIONS:
   1. How do tolerance interventions spread through friendship networks?
   2. Which targeting strategies maximize interethnic cooperation?
   3. What role does complex contagion play in tolerance diffusion?
   4. How persistent are intervention effects over time?

📊 VISUALIZATION COMPONENTS:

1. 🎬 NETWORK EVOLUTION ANIMATIONS
   • Show tolerance spreading via social influence
   • Display attraction-repulsion dynamics
   • Visualize cooperation ties emerging over time
   • Compare different intervention strategies

2. 🖥️  INTERACTIVE DASHBOARD
   • Real-time parameter manipulation
   • Live network visualization updates
   • Comparative analysis tools
   • Export functionality for scenarios

3. 📄 PUBLICATION FIGURES (300+ DPI)
   • Figure 1: Conceptual model diagram
   • Figure 2: Network structure examples
   • Figure 3: Intervention strategies
   • Figure 4: Simulation results
   • Figure 5: Empirical validation

4. 🌐 3D NETWORK VISUALIZATIONS
   • Interactive exploration of network structure
   • Tolerance levels as 3D positioning
   • Cooperation strength as edge thickness
   • Ethnic groups with distinct colors

5. 📈 STRATEGY EFFECTIVENESS ANALYSIS
   • Central vs peripheral targeting
   • Random vs clustered delivery
   • Simple vs complex contagion
   • Cost-effectiveness comparisons

🎨 VISUAL DESIGN PRINCIPLES:
   • Color schemes: Red (low tolerance) → Green (high tolerance)
   • Node sizes: Larger = intervention targets
   • Edge colors: Red = interethnic, Gray = intraethnic
   • Animations: Smooth transitions, clear temporal progression

💡 POLICY IMPLICATIONS:
   • Educational settings (classroom networks)
   • Community programs (neighborhood initiatives)
   • Workplace diversity training
   • Online social platforms

📑 PUBLICATION TARGET:
   Journal of Artificial Societies and Social Simulation (JASSS)
   https://www.jasss.org/

🔧 TECHNICAL DETAILS:
   • Programming: Python 3.8+
   • Key Libraries: NetworkX, Matplotlib, Panel, Plotly
   • Model Framework: Mesa ABM + Custom Extensions
   • Statistics: R integration via RSiena
   • Output Formats: PNG, PDF, SVG, MP4, GIF, HTML

📞 RESEARCH SUPPORT:
   For questions about methodology or implementation,
   refer to the PhD dissertation documentation.
""")

if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Run main showcase
    main()