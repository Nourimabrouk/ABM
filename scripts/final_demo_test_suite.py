#!/usr/bin/env python3
"""
FINAL DEMO TEST SUITE
Comprehensive testing and validation of all tolerance intervention demos
Author: AI Agent Coordination Team
Date: 2025-09-16
"""

import time
import logging
from pathlib import Path
import pandas as pd
import numpy as np
import json
import subprocess
import sys
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('final_demo_validation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class FinalDemoTestSuite:
    """Comprehensive testing suite for tolerance intervention demos"""

    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        self.start_time = time.time()
        self.test_results = {}
        self.performance_metrics = {}

        logger.info("Final Demo Test Suite initialized")
        logger.info(f"Project root: {self.project_root.absolute()}")

    def test_data_generation(self) -> Dict[str, bool]:
        """Test data generation components"""
        logger.info("Testing data generation...")

        start_time = time.time()
        tests = {}

        # Test 1: Check if data files exist
        data_dir = self.project_root / "outputs" / "tolerance_data"
        required_files = [
            "students.csv",
            "tolerance_evolution_complete.csv",
            "network_statistics.csv",
            "config.json",
            "summary.json"
        ]

        tests["data_files_exist"] = all((data_dir / file).exists() for file in required_files)

        # Test 2: Validate data integrity
        if tests["data_files_exist"]:
            try:
                students = pd.read_csv(data_dir / "students.csv")
                tolerance_data = pd.read_csv(data_dir / "tolerance_evolution_complete.csv")

                tests["data_integrity"] = (
                    len(students) == 30 and
                    len(tolerance_data) > 0 and
                    tolerance_data['scenario'].nunique() == 5 and
                    tolerance_data['wave'].nunique() == 4
                )
            except Exception as e:
                logger.error(f"Data integrity test failed: {e}")
                tests["data_integrity"] = False
        else:
            tests["data_integrity"] = False

        # Test 3: Check network files
        network_files = [f"network_wave_{i}.csv" for i in range(1, 4)]
        tests["network_files_exist"] = all((data_dir / file).exists() for file in network_files)

        # Test 4: Validate network properties
        if tests["network_files_exist"]:
            try:
                networks_valid = True
                for i in range(1, 4):
                    network = pd.read_csv(data_dir / f"network_wave_{i}.csv", header=None).values
                    if network.shape != (30, 30) or not np.array_equal(network, network.T):
                        networks_valid = False
                        break
                tests["network_properties_valid"] = networks_valid
            except Exception as e:
                logger.error(f"Network validation failed: {e}")
                tests["network_properties_valid"] = False
        else:
            tests["network_properties_valid"] = False

        end_time = time.time()
        self.performance_metrics["data_generation"] = end_time - start_time

        logger.info(f"Data generation tests: {sum(tests.values())}/{len(tests)} passed")
        return tests

    def test_visualizations(self) -> Dict[str, bool]:
        """Test visualization generation"""
        logger.info("Testing visualizations...")

        start_time = time.time()
        tests = {}

        viz_dir = self.project_root / "outputs" / "visualizations"

        # Test 1: Check if visualization directory exists
        tests["viz_directory_exists"] = viz_dir.exists()

        # Test 2: Check for required visualization files
        if tests["viz_directory_exists"]:
            required_viz_files = [
                "figure_1_network_evolution.png",
                "figure_2_tolerance_evolution.png",
                "figure_3_intervention_effectiveness.png",
                "figure_4_network_metrics.png",
                "figure_5_comprehensive_summary.png"
            ]

            tests["required_figures_exist"] = all((viz_dir / file).exists() for file in required_viz_files)

            # Test 3: Check file sizes (should be reasonable for 300 DPI images)
            if tests["required_figures_exist"]:
                file_sizes_ok = True
                for file in required_viz_files:
                    file_path = viz_dir / file
                    file_size = file_path.stat().st_size
                    # Images should be at least 100KB and less than 20MB
                    if not (100_000 <= file_size <= 20_000_000):
                        logger.warning(f"Unusual file size for {file}: {file_size} bytes")
                        file_sizes_ok = False

                tests["file_sizes_reasonable"] = file_sizes_ok
            else:
                tests["file_sizes_reasonable"] = False

            # Test 4: Check visualization index
            tests["viz_index_exists"] = (viz_dir / "visualization_index.txt").exists()

        else:
            tests["required_figures_exist"] = False
            tests["file_sizes_reasonable"] = False
            tests["viz_index_exists"] = False

        end_time = time.time()
        self.performance_metrics["visualizations"] = end_time - start_time

        logger.info(f"Visualization tests: {sum(tests.values())}/{len(tests)} passed")
        return tests

    def test_r_scripts(self) -> Dict[str, bool]:
        """Test R script functionality (dry run)"""
        logger.info("Testing R scripts...")

        start_time = time.time()
        tests = {}

        r_dir = self.project_root / "R"

        # Test 1: Check if R scripts exist
        required_r_files = [
            "final_rsiena_comprehensive_demo.R",
            "final_tolerance_data_generator.R",
            "tolerance_basic_demo.R",
            "intervention_simulation_demo.R"
        ]

        tests["r_scripts_exist"] = all((r_dir / file).exists() for file in required_r_files)

        # Test 2: Basic syntax validation (check for common R patterns)
        if tests["r_scripts_exist"]:
            syntax_valid = True
            for file in required_r_files:
                try:
                    with open(r_dir / file, 'r') as f:
                        content = f.read()
                        # Basic checks for R syntax
                        if "library(" not in content or "function" not in content:
                            logger.warning(f"Potential syntax issues in {file}")
                            syntax_valid = False
                except Exception as e:
                    logger.error(f"Error reading {file}: {e}")
                    syntax_valid = False

            tests["r_syntax_valid"] = syntax_valid
        else:
            tests["r_syntax_valid"] = False

        # Test 3: Check for RSiena integration
        if tests["r_scripts_exist"]:
            siena_integration = False
            try:
                with open(r_dir / "final_rsiena_comprehensive_demo.R", 'r') as f:
                    content = f.read()
                    siena_integration = (
                        "library(RSiena)" in content and
                        "sienaDataCreate" in content and
                        "getEffects" in content
                    )
            except Exception as e:
                logger.error(f"Error checking RSiena integration: {e}")

            tests["siena_integration"] = siena_integration
        else:
            tests["siena_integration"] = False

        end_time = time.time()
        self.performance_metrics["r_scripts"] = end_time - start_time

        logger.info(f"R script tests: {sum(tests.values())}/{len(tests)} passed")
        return tests

    def test_python_components(self) -> Dict[str, bool]:
        """Test Python components"""
        logger.info("Testing Python components...")

        start_time = time.time()
        tests = {}

        src_dir = self.project_root / "src"

        # Test 1: Check if Python scripts exist
        required_python_files = [
            "tolerance_data_generator.py",
            "final_visualization_generator.py",
            "advanced_visualization_system.py"
        ]

        tests["python_scripts_exist"] = all((src_dir / file).exists() for file in required_python_files)

        # Test 2: Test import capabilities
        if tests["python_scripts_exist"]:
            import_success = True
            for file in required_python_files:
                try:
                    # Basic syntax check by attempting to compile
                    with open(src_dir / file, 'r') as f:
                        content = f.read()
                        compile(content, file, 'exec')
                except SyntaxError as e:
                    logger.error(f"Syntax error in {file}: {e}")
                    import_success = False
                except Exception as e:
                    logger.warning(f"Compilation warning for {file}: {e}")

            tests["python_syntax_valid"] = import_success
        else:
            tests["python_syntax_valid"] = False

        # Test 3: Check for required dependencies
        tests["dependencies_importable"] = self._check_python_dependencies()

        end_time = time.time()
        self.performance_metrics["python_components"] = end_time - start_time

        logger.info(f"Python component tests: {sum(tests.values())}/{len(tests)} passed")
        return tests

    def _check_python_dependencies(self) -> bool:
        """Check if required Python dependencies are importable"""
        required_packages = [
            'numpy', 'pandas', 'matplotlib', 'seaborn', 'networkx', 'plotly'
        ]

        for package in required_packages:
            try:
                __import__(package)
            except ImportError:
                logger.error(f"Required package not available: {package}")
                return False

        return True

    def test_output_completeness(self) -> Dict[str, bool]:
        """Test completeness of all outputs"""
        logger.info("Testing output completeness...")

        start_time = time.time()
        tests = {}

        outputs_dir = self.project_root / "outputs"

        # Test 1: Main directories exist
        required_dirs = ["tolerance_data", "visualizations"]
        tests["output_directories_exist"] = all((outputs_dir / dir_name).exists() for dir_name in required_dirs)

        # Test 2: Data completeness
        if tests["output_directories_exist"]:
            data_dir = outputs_dir / "tolerance_data"
            data_files = list(data_dir.glob("*.csv")) + list(data_dir.glob("*.json"))
            tests["sufficient_data_files"] = len(data_files) >= 8  # Minimum expected files

            viz_dir = outputs_dir / "visualizations"
            viz_files = list(viz_dir.glob("*.png")) + list(viz_dir.glob("*.html"))
            tests["sufficient_viz_files"] = len(viz_files) >= 5  # Minimum expected visualizations
        else:
            tests["sufficient_data_files"] = False
            tests["sufficient_viz_files"] = False

        # Test 3: Summary files exist
        summary_files = ["research_summary.txt"]
        tests["summary_files_exist"] = all((outputs_dir / file).exists() for file in summary_files)

        end_time = time.time()
        self.performance_metrics["output_completeness"] = end_time - start_time

        logger.info(f"Output completeness tests: {sum(tests.values())}/{len(tests)} passed")
        return tests

    def performance_benchmark(self) -> Dict[str, float]:
        """Run performance benchmarks"""
        logger.info("Running performance benchmarks...")

        start_time = time.time()
        benchmarks = {}

        # Benchmark 1: Data loading speed
        data_load_start = time.time()
        try:
            data_dir = self.project_root / "outputs" / "tolerance_data"
            students = pd.read_csv(data_dir / "students.csv")
            tolerance_data = pd.read_csv(data_dir / "tolerance_evolution_complete.csv")
            network_stats = pd.read_csv(data_dir / "network_statistics.csv")

            benchmarks["data_loading_time"] = time.time() - data_load_start
        except Exception as e:
            logger.error(f"Data loading benchmark failed: {e}")
            benchmarks["data_loading_time"] = float('inf')

        # Benchmark 2: Visualization file access
        viz_access_start = time.time()
        try:
            viz_dir = self.project_root / "outputs" / "visualizations"
            viz_files = list(viz_dir.glob("*.png"))
            file_sizes = [f.stat().st_size for f in viz_files]
            benchmarks["viz_access_time"] = time.time() - viz_access_start
        except Exception as e:
            logger.error(f"Visualization access benchmark failed: {e}")
            benchmarks["viz_access_time"] = float('inf')

        # Benchmark 3: Overall system responsiveness
        benchmarks["total_test_time"] = time.time() - start_time

        logger.info(f"Performance benchmarks completed in {benchmarks['total_test_time']:.2f}s")
        return benchmarks

    def generate_comprehensive_report(self) -> Dict:
        """Generate comprehensive test report"""
        logger.info("Generating comprehensive test report...")

        # Run all test suites
        test_results = {
            "data_generation": self.test_data_generation(),
            "visualizations": self.test_visualizations(),
            "r_scripts": self.test_r_scripts(),
            "python_components": self.test_python_components(),
            "output_completeness": self.test_output_completeness()
        }

        # Run performance benchmarks
        performance_results = self.performance_benchmark()

        # Calculate overall metrics
        total_tests = sum(len(suite_results) for suite_results in test_results.values())
        passed_tests = sum(sum(suite_results.values()) for suite_results in test_results.values())
        overall_success_rate = (passed_tests / total_tests) * 100 if total_tests > 0 else 0

        # Calculate total execution time
        total_time = time.time() - self.start_time

        # Generate report
        report = {
            "timestamp": pd.Timestamp.now().isoformat(),
            "total_execution_time": total_time,
            "overall_success_rate": overall_success_rate,
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "failed_tests": total_tests - passed_tests,
            "test_results": test_results,
            "performance_metrics": {**self.performance_metrics, **performance_results},
            "system_ready": overall_success_rate >= 90.0 and total_time <= 1800  # 30 minutes
        }

        # Save report
        report_file = self.project_root / "outputs" / "final_demo_validation_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)

        # Generate human-readable summary
        self.generate_summary_report(report)

        return report

    def generate_summary_report(self, report: Dict):
        """Generate human-readable summary report"""
        summary_file = self.project_root / "outputs" / "demo_validation_summary.txt"

        with open(summary_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("FINAL TOLERANCE INTERVENTION DEMO VALIDATION REPORT\n")
            f.write("=" * 80 + "\n\n")

            f.write(f"Generated: {report['timestamp']}\n")
            f.write(f"Total Execution Time: {report['total_execution_time']:.2f} seconds\n")
            f.write(f"Overall Success Rate: {report['overall_success_rate']:.1f}%\n")
            f.write(f"Tests Passed: {report['passed_tests']}/{report['total_tests']}\n\n")

            f.write("SYSTEM STATUS: ")
            if report['system_ready']:
                f.write("✅ READY FOR PRESENTATION\n\n")
            else:
                f.write("⚠️ REQUIRES ATTENTION\n\n")

            f.write("DETAILED TEST RESULTS:\n")
            f.write("-" * 40 + "\n")

            for suite_name, suite_results in report['test_results'].items():
                passed = sum(suite_results.values())
                total = len(suite_results)
                f.write(f"\n{suite_name.upper().replace('_', ' ')}: {passed}/{total} passed\n")

                for test_name, result in suite_results.items():
                    status = "✅ PASS" if result else "❌ FAIL"
                    f.write(f"  - {test_name.replace('_', ' ')}: {status}\n")

            f.write("\nPERFORMANCE METRICS:\n")
            f.write("-" * 40 + "\n")
            for metric_name, value in report['performance_metrics'].items():
                if isinstance(value, (int, float)) and not np.isinf(value):
                    f.write(f"- {metric_name.replace('_', ' ')}: {value:.3f}s\n")

            f.write("\nRECOMMendATIONS:\n")
            f.write("-" * 40 + "\n")
            if report['overall_success_rate'] >= 95:
                f.write("🏆 Excellent! System is fully ready for demonstration.\n")
            elif report['overall_success_rate'] >= 90:
                f.write("✅ Good! System is ready with minor recommendations.\n")
            elif report['overall_success_rate'] >= 80:
                f.write("⚠️ Acceptable but some issues need addressing.\n")
            else:
                f.write("❌ Critical issues detected. Review failed tests.\n")

            if report['total_execution_time'] > 1800:  # 30 minutes
                f.write("⏰ Execution time exceeds 30-minute target.\n")
            else:
                f.write("⚡ Execution time meets 30-minute target.\n")

        logger.info(f"Summary report saved to {summary_file}")

def main():
    """Main execution function"""
    print("=" * 80)
    print("FINAL TOLERANCE INTERVENTION DEMO TEST SUITE")
    print("Comprehensive validation of all system components")
    print("=" * 80)

    try:
        # Initialize test suite
        test_suite = FinalDemoTestSuite(".")

        # Generate comprehensive report
        report = test_suite.generate_comprehensive_report()

        # Print summary
        print(f"\n🏁 VALIDATION COMPLETE!")
        print(f"⏱️  Total time: {report['total_execution_time']:.2f} seconds")
        print(f"✅ Success rate: {report['overall_success_rate']:.1f}%")
        print(f"📊 Tests: {report['passed_tests']}/{report['total_tests']} passed")

        if report['system_ready']:
            print("🎯 SYSTEM READY FOR PRESENTATION!")
        else:
            print("⚠️  System requires attention - check validation report")

        print(f"\n📄 Detailed reports saved to outputs/")
        print("   - final_demo_validation_report.json")
        print("   - demo_validation_summary.txt")

        return report

    except Exception as e:
        logger.error(f"Test suite execution failed: {e}")
        print(f"❌ Test suite failed: {e}")
        return None

if __name__ == "__main__":
    main()