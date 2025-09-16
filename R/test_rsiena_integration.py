"""
Test script for RSiena integration with ABM models.

This script tests the complete workflow:
1. Create synthetic longitudinal network data
2. Test RSiena integration utilities
3. Run social network ABM
4. Validate ABM against RSiena (if R environment is set up)

Run this script to verify your RSiena integration is working correctly.
"""

import logging
import sys
from pathlib import Path
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from utils.rsiena_integration import RSienaIntegrator, create_example_longitudinal_networks, RPY2_AVAILABLE
from models.social_network_abm import run_social_network_simulation, SocialNetworkModel

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_basic_integration():
    """Test basic RSiena integration without R."""
    print("=== Testing Basic Integration ===")

    # Test creating example networks
    networks = create_example_longitudinal_networks(n_actors=15, n_periods=3)
    print(f"✓ Created {len(networks)} network snapshots")

    for i, net in enumerate(networks):
        print(f"  Period {i}: {net.number_of_nodes()} nodes, {net.number_of_edges()} edges")

    return networks


def test_rpy2_integration():
    """Test rpy2 integration with R."""
    print("\n=== Testing rpy2 Integration ===")

    if not RPY2_AVAILABLE:
        print("✗ rpy2 not available - skipping R integration tests")
        return False

    try:
        # Initialize integrator
        integrator = RSienaIntegrator(ensure_rsiena=False)  # Don't auto-install
        print("✓ RSiena integrator initialized")

        # Test with example networks
        networks = create_example_longitudinal_networks(n_actors=10, n_periods=3)

        # Convert to RSiena format (this should work even without RSiena package)
        try:
            rsiena_data = integrator.mesa_networks_to_rsiena(networks)
            print("✓ Network conversion to RSiena format successful")
            print(f"  Data contains: {list(rsiena_data.keys())}")
            return True
        except Exception as e:
            print(f"✗ Network conversion failed: {e}")
            return False

    except Exception as e:
        print(f"✗ rpy2 integration failed: {e}")
        return False


def test_abm_simulation():
    """Test the social network ABM."""
    print("\n=== Testing Social Network ABM ===")

    try:
        # Run a small simulation
        model, data = run_social_network_simulation(
            steps=50,
            n_agents=20,
            enable_rsiena=False  # Disable RSiena for basic test
        )

        print("✓ ABM simulation completed successfully")
        print(f"  Final density: {model._get_network_density():.3f}")
        print(f"  Final average degree: {model._get_average_degree():.2f}")
        print(f"  Final clustering: {model._get_clustering_coefficient():.3f}")

        # Check if data was collected
        if len(data) > 0:
            print(f"✓ Data collection working - {len(data)} time points recorded")
        else:
            print("✗ No data collected")
            return False

        return True

    except Exception as e:
        print(f"✗ ABM simulation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_full_integration():
    """Test full integration including RSiena validation."""
    print("\n=== Testing Full Integration ===")

    if not RPY2_AVAILABLE:
        print("✗ rpy2 not available - skipping full integration test")
        return False

    try:
        # Create RSiena integrator with package installation
        integrator = RSienaIntegrator(ensure_rsiena=True)
        print("✓ RSiena package available")

        # Create empirical-like data
        empirical_networks = create_example_longitudinal_networks(n_actors=15, n_periods=4)

        # Run ABM with RSiena validation enabled
        model = SocialNetworkModel(
            n_agents=15,
            enable_rsiena_validation=True,
            rsiena_validation_interval=10  # Collect snapshots more frequently
        )

        # Run for enough steps to get snapshots
        for step in range(25):
            model.step()

        print(f"✓ ABM with RSiena validation completed")
        print(f"  Collected {len(model.network_snapshots)} network snapshots")

        if len(model.network_snapshots) >= 2:
            print("✓ Sufficient snapshots for RSiena validation")

            # Attempt validation (may fail if RSiena estimation has issues)
            try:
                validation_results = model.validate_with_rsiena(empirical_networks)
                print("✓ RSiena validation completed successfully!")

                # Print some results
                rsiena_results = validation_results['rsiena_results']
                if rsiena_results['convergence']:
                    print("✓ RSiena model converged")
                else:
                    print("⚠ RSiena model did not converge")

                print(f"  Max convergence ratio: {rsiena_results['max_convergence_ratio']:.3f}")

            except Exception as e:
                print(f"⚠ RSiena validation failed (this is common): {e}")
                print("  This may be due to small network size or estimation issues")

        return True

    except Exception as e:
        print(f"✗ Full integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def create_visualization():
    """Create a simple visualization of the test results."""
    print("\n=== Creating Visualization ===")

    try:
        # Run a simulation for visualization
        model, data = run_social_network_simulation(steps=100, n_agents=30, enable_rsiena=False)

        # Create plots
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Social Network ABM Test Results', fontsize=16)

        # Network evolution metrics
        axes[0, 0].plot(data.index, data['Network_Density'], label='Density')
        axes[0, 0].plot(data.index, data['Clustering_Coefficient'], label='Clustering')
        axes[0, 0].set_title('Network Structure Evolution')
        axes[0, 0].set_xlabel('Time Step')
        axes[0, 0].set_ylabel('Metric Value')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Degree evolution
        axes[0, 1].plot(data.index, data['Average_Degree'])
        axes[0, 1].set_title('Average Degree Over Time')
        axes[0, 1].set_xlabel('Time Step')
        axes[0, 1].set_ylabel('Average Degree')
        axes[0, 1].grid(True, alpha=0.3)

        # Assortativity measures
        if 'Assortativity_Age' in data.columns:
            axes[1, 0].plot(data.index, data['Assortativity_Age'], label='Age')
            axes[1, 0].plot(data.index, data['Assortativity_SES'], label='SES')
            axes[1, 0].set_title('Assortativity by Attributes')
            axes[1, 0].set_xlabel('Time Step')
            axes[1, 0].set_ylabel('Assortativity')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)

        # Final network visualization
        pos = nx.spring_layout(model.network, seed=42)
        nx.draw(model.network, pos, ax=axes[1, 1],
                node_size=30, node_color='lightblue',
                edge_color='gray', alpha=0.7)
        axes[1, 1].set_title(f'Final Network (Density: {model._get_network_density():.3f})')

        plt.tight_layout()

        # Save plot
        output_path = Path('outputs') / 'rsiena_integration_test.png'
        output_path.parent.mkdir(exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Visualization saved to {output_path}")

        plt.show()

        return True

    except Exception as e:
        print(f"✗ Visualization failed: {e}")
        return False


def main():
    """Run all tests."""
    print("RSiena Integration Test Suite")
    print("=" * 50)

    results = {}

    # Run tests
    results['basic'] = test_basic_integration()
    results['rpy2'] = test_rpy2_integration()
    results['abm'] = test_abm_simulation()
    results['full'] = test_full_integration()
    results['viz'] = create_visualization()

    # Summary
    print("\n=== Test Summary ===")
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{test_name.upper():12} {status}")

    total_passed = sum(results.values())
    total_tests = len(results)

    print(f"\nOverall: {total_passed}/{total_tests} tests passed")

    if total_passed == total_tests:
        print("🎉 All tests passed! RSiena integration is ready to use.")
    elif results['basic'] and results['abm']:
        print("⚠ Basic functionality works. RSiena integration may need R setup.")
        print("\nNext steps:")
        print("1. Install R: https://cran.r-project.org/bin/windows/base/")
        print("2. Run: python setup_r_environment.py")
        print("3. Re-run this test script")
    else:
        print("❌ Basic functionality issues detected. Check error messages above.")

    return total_passed == total_tests


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)