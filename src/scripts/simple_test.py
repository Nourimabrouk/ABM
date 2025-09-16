"""
Simple test script for basic RSiena integration functionality.
Tests core functionality without requiring full dependency installation.
"""

import sys
from pathlib import Path
import numpy as np
import networkx as nx

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

try:
    from utils.rsiena_integration import create_example_longitudinal_networks, RPY2_AVAILABLE
    print("OK RSiena integration module imported successfully")
except ImportError as e:
    print(f"✗ Failed to import RSiena integration: {e}")
    sys.exit(1)

def test_network_creation():
    """Test creating example longitudinal networks."""
    print("\n=== Testing Network Creation ===")

    networks = create_example_longitudinal_networks(n_actors=10, n_periods=3)
    print(f"OK Created {len(networks)} network snapshots")

    for i, net in enumerate(networks):
        print(f"  Period {i}: {net.number_of_nodes()} nodes, {net.number_of_edges()} edges")

    return True

def test_rpy2_availability():
    """Test if rpy2 is available."""
    print("\n=== Testing rpy2 Availability ===")

    if RPY2_AVAILABLE:
        print("OK rpy2 is available for R integration")
        try:
            from utils.rsiena_integration import RSienaIntegrator
            integrator = RSienaIntegrator(ensure_rsiena=False)
            print("OK RSienaIntegrator can be instantiated")
            return True
        except Exception as e:
            print(f"⚠ RSienaIntegrator instantiation failed: {e}")
            return False
    else:
        print("X rpy2 is not available")
        print("  Install with: pip install rpy2")
        return False

def test_basic_abm():
    """Test basic ABM functionality."""
    print("\n=== Testing Basic ABM ===")

    try:
        # Import mesa components
        import mesa
        print("OK Mesa is available")

        # Try to import our ABM model
        from models.social_network_abm import SocialNetworkModel
        print("OK Social network ABM imported successfully")

        # Create a small model
        model = SocialNetworkModel(n_agents=5, enable_rsiena_validation=False)
        print("OK ABM model created successfully")

        # Run a few steps
        for _ in range(5):
            model.step()

        print(f"OK ABM simulation ran successfully")
        print(f"  Final density: {model._get_network_density():.3f}")

        return True

    except ImportError as e:
        print(f"X Mesa/ABM import failed: {e}")
        return False
    except Exception as e:
        print(f"X ABM test failed: {e}")
        return False

def main():
    """Run simple tests."""
    print("Simple RSiena Integration Test")
    print("=" * 40)

    results = []

    # Test network creation
    results.append(test_network_creation())

    # Test rpy2
    results.append(test_rpy2_availability())

    # Test basic ABM
    results.append(test_basic_abm())

    # Summary
    print("\n=== Test Summary ===")
    passed = sum(results)
    total = len(results)

    print(f"Tests passed: {passed}/{total}")

    if passed == total:
        print("SUCCESS: All basic tests passed!")
        print("\nNext steps:")
        print("1. Install R from: https://cran.r-project.org/")
        print("2. Run: python setup_r_environment.py")
        print("3. Try full integration tests")
    elif passed >= 1:
        print("WARNING: Some tests passed. Check errors above.")
    else:
        print("ERROR: All tests failed. Check dependencies.")

    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)