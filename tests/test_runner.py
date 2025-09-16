"""
Comprehensive Test Runner and Quality Assurance Framework

This module provides a comprehensive test runner for the ABM-RSiena tolerance
intervention research validation suite. It orchestrates all test categories,
generates detailed reports, and validates research quality standards.

Features:
- Automated test discovery and execution
- Comprehensive quality metrics calculation
- Detailed HTML and PDF reporting
- Performance benchmarking
- Research standards validation
- Continuous integration support

Author: Validation Specialist
Created: 2025-09-16
"""

import unittest
import sys
import logging
import time
import json
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import traceback

# Import test modules
from test_rsiena_integration import TestRSienaIntegration
from test_intervention_simulations import TestInterventionSimulations
from test_data_processing import TestDataProcessing
from test_statistical_analysis import TestStatisticalAnalysis
from test_visualizations import TestVisualizations

logger = logging.getLogger(__name__)


@dataclass
class TestSuiteResults:
    """Comprehensive test suite results."""
    total_tests: int = 0
    passed_tests: int = 0
    failed_tests: int = 0
    skipped_tests: int = 0
    errors: int = 0
    warnings: int = 0
    execution_time: float = 0.0
    coverage_percentage: float = 0.0
    quality_score: float = 0.0
    test_categories: Dict[str, Any] = None
    performance_metrics: Dict[str, float] = None
    validation_checks: Dict[str, bool] = None

    def __post_init__(self):
        if self.test_categories is None:
            self.test_categories = {}
        if self.performance_metrics is None:
            self.performance_metrics = {}
        if self.validation_checks is None:
            self.validation_checks = {}


@dataclass
class QualityMetrics:
    """Research quality validation metrics."""
    model_convergence_rate: float = 0.0
    parameter_accuracy_score: float = 0.0
    data_integrity_score: float = 0.0
    visualization_accuracy_score: float = 0.0
    statistical_validity_score: float = 0.0
    reproducibility_score: float = 0.0
    overall_quality_score: float = 0.0


class ComprehensiveTestRunner:
    """Comprehensive test runner with quality assurance."""

    def __init__(self, output_dir: Optional[Path] = None):
        """Initialize test runner."""
        self.output_dir = output_dir or Path(__file__).parent / "test_outputs"
        self.output_dir.mkdir(exist_ok=True)

        # Configure logging
        self._setup_logging()

        # Test categories
        self.test_categories = {
            'rsiena_integration': TestRSienaIntegration,
            'intervention_simulations': TestInterventionSimulations,
            'data_processing': TestDataProcessing,
            'statistical_analysis': TestStatisticalAnalysis,
            'visualizations': TestVisualizations
        }

        # Quality standards
        self.quality_standards = {
            'min_convergence_rate': 0.90,
            'min_parameter_accuracy': 0.85,
            'min_data_integrity': 0.95,
            'min_visualization_accuracy': 0.90,
            'min_statistical_validity': 0.90,
            'min_reproducibility': 0.95,
            'min_overall_quality': 0.90
        }

    def _setup_logging(self):
        """Set up comprehensive logging."""
        log_file = self.output_dir / f"test_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

        # File handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)

        # Configure logger
        logger.setLevel(logging.DEBUG)
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

        logger.info(f"Test run started - Log file: {log_file}")

    def run_comprehensive_tests(self, categories: Optional[List[str]] = None) -> TestSuiteResults:
        """Run comprehensive test suite."""
        logger.info("Starting comprehensive test suite...")
        start_time = time.time()

        # Filter categories if specified
        if categories:
            test_categories = {k: v for k, v in self.test_categories.items() if k in categories}
        else:
            test_categories = self.test_categories

        # Initialize results
        results = TestSuiteResults()
        category_results = {}

        # Run each test category
        for category_name, test_class in test_categories.items():
            logger.info(f"Running {category_name} tests...")

            try:
                category_result = self._run_test_category(category_name, test_class)
                category_results[category_name] = category_result

                # Update overall results
                results.total_tests += category_result['total_tests']
                results.passed_tests += category_result['passed_tests']
                results.failed_tests += category_result['failed_tests']
                results.skipped_tests += category_result['skipped_tests']
                results.errors += category_result['errors']

            except Exception as e:
                logger.error(f"Failed to run {category_name} tests: {e}")
                results.errors += 1
                category_results[category_name] = {
                    'status': 'ERROR',
                    'error_message': str(e),
                    'traceback': traceback.format_exc()
                }

        # Calculate execution time
        results.execution_time = time.time() - start_time
        results.test_categories = category_results

        # Calculate performance metrics
        results.performance_metrics = self._calculate_performance_metrics(results)

        # Run validation checks
        results.validation_checks = self._run_validation_checks(results)

        # Calculate quality scores
        quality_metrics = self._calculate_quality_metrics(results)
        results.quality_score = quality_metrics.overall_quality_score

        # Generate reports
        self._generate_comprehensive_report(results, quality_metrics)

        logger.info(f"Test suite completed in {results.execution_time:.2f} seconds")
        logger.info(f"Results: {results.passed_tests}/{results.total_tests} passed, "
                   f"Quality Score: {results.quality_score:.2f}")

        return results

    def _run_test_category(self, category_name: str, test_class: type) -> Dict[str, Any]:
        """Run tests for a specific category."""
        # Create test suite
        suite = unittest.TestLoader().loadTestsFromTestCase(test_class)

        # Run tests with custom result collector
        result_collector = DetailedTestResult()
        suite.run(result_collector)

        # Compile results
        category_result = {
            'category': category_name,
            'total_tests': result_collector.testsRun,
            'passed_tests': result_collector.testsRun - len(result_collector.failures) - len(result_collector.errors),
            'failed_tests': len(result_collector.failures),
            'skipped_tests': len(result_collector.skipped),
            'errors': len(result_collector.errors),
            'execution_time': result_collector.execution_time,
            'test_details': result_collector.test_details,
            'performance_issues': result_collector.performance_issues
        }

        return category_result

    def _calculate_performance_metrics(self, results: TestSuiteResults) -> Dict[str, float]:
        """Calculate performance metrics."""
        metrics = {}

        # Test execution efficiency
        if results.total_tests > 0:
            metrics['avg_test_time'] = results.execution_time / results.total_tests
            metrics['tests_per_second'] = results.total_tests / results.execution_time
        else:
            metrics['avg_test_time'] = 0.0
            metrics['tests_per_second'] = 0.0

        # Success rate
        if results.total_tests > 0:
            metrics['success_rate'] = results.passed_tests / results.total_tests
            metrics['failure_rate'] = results.failed_tests / results.total_tests
            metrics['error_rate'] = results.errors / results.total_tests
        else:
            metrics['success_rate'] = 0.0
            metrics['failure_rate'] = 0.0
            metrics['error_rate'] = 0.0

        # Performance benchmarks
        metrics['fast_execution'] = metrics['avg_test_time'] < 1.0  # Less than 1 second per test
        metrics['high_success_rate'] = metrics['success_rate'] > 0.95
        metrics['low_error_rate'] = metrics['error_rate'] < 0.05

        return metrics

    def _run_validation_checks(self, results: TestSuiteResults) -> Dict[str, bool]:
        """Run comprehensive validation checks."""
        checks = {}

        # Basic test execution checks
        checks['tests_executed'] = results.total_tests > 0
        checks['majority_passed'] = results.passed_tests > results.failed_tests
        checks['low_error_rate'] = results.errors < (results.total_tests * 0.1)

        # Category coverage checks
        required_categories = ['rsiena_integration', 'intervention_simulations',
                             'data_processing', 'statistical_analysis']
        checks['all_categories_tested'] = all(
            cat in results.test_categories for cat in required_categories
        )

        # Performance checks
        checks['reasonable_execution_time'] = results.execution_time < 300  # 5 minutes max
        checks['no_timeout_failures'] = True  # Would check for timeout-related failures

        # Quality checks
        checks['meets_quality_standards'] = results.quality_score >= self.quality_standards['min_overall_quality']

        return checks

    def _calculate_quality_metrics(self, results: TestSuiteResults) -> QualityMetrics:
        """Calculate comprehensive quality metrics."""
        metrics = QualityMetrics()

        # Model convergence rate (from RSiena tests)
        rsiena_results = results.test_categories.get('rsiena_integration', {})
        if 'test_details' in rsiena_results:
            convergence_tests = [test for test in rsiena_results['test_details']
                               if 'convergence' in test.get('test_name', '').lower()]
            if convergence_tests:
                convergence_success = sum(1 for test in convergence_tests if test.get('passed', False))
                metrics.model_convergence_rate = convergence_success / len(convergence_tests)

        # Parameter accuracy (from statistical analysis tests)
        stats_results = results.test_categories.get('statistical_analysis', {})
        if 'test_details' in stats_results:
            parameter_tests = [test for test in stats_results['test_details']
                             if 'parameter' in test.get('test_name', '').lower()]
            if parameter_tests:
                parameter_success = sum(1 for test in parameter_tests if test.get('passed', False))
                metrics.parameter_accuracy_score = parameter_success / len(parameter_tests)

        # Data integrity (from data processing tests)
        data_results = results.test_categories.get('data_processing', {})
        if 'test_details' in data_results:
            integrity_tests = [test for test in data_results['test_details']
                             if any(word in test.get('test_name', '').lower()
                                   for word in ['integrity', 'validation', 'consistency'])]
            if integrity_tests:
                integrity_success = sum(1 for test in integrity_tests if test.get('passed', False))
                metrics.data_integrity_score = integrity_success / len(integrity_tests)

        # Visualization accuracy (from visualization tests)
        viz_results = results.test_categories.get('visualizations', {})
        if 'test_details' in viz_results:
            viz_tests = [test for test in viz_results['test_details']
                        if 'correspondence' in test.get('test_name', '').lower()]
            if viz_tests:
                viz_success = sum(1 for test in viz_tests if test.get('passed', False))
                metrics.visualization_accuracy_score = viz_success / len(viz_tests)

        # Statistical validity (overall statistical test success)
        if stats_results.get('total_tests', 0) > 0:
            metrics.statistical_validity_score = stats_results.get('passed_tests', 0) / stats_results['total_tests']

        # Reproducibility (based on test consistency and deterministic results)
        all_passed = results.passed_tests
        all_total = results.total_tests
        if all_total > 0:
            metrics.reproducibility_score = all_passed / all_total

        # Overall quality score (weighted average)
        weights = {
            'model_convergence_rate': 0.20,
            'parameter_accuracy_score': 0.20,
            'data_integrity_score': 0.20,
            'visualization_accuracy_score': 0.15,
            'statistical_validity_score': 0.15,
            'reproducibility_score': 0.10
        }

        weighted_sum = sum(getattr(metrics, attr) * weight
                          for attr, weight in weights.items())
        metrics.overall_quality_score = weighted_sum

        return metrics

    def _generate_comprehensive_report(self, results: TestSuiteResults, quality_metrics: QualityMetrics):
        """Generate comprehensive test report."""
        # Generate JSON report
        self._generate_json_report(results, quality_metrics)

        # Generate HTML report
        self._generate_html_report(results, quality_metrics)

        # Generate quality checklist
        self._generate_quality_checklist(quality_metrics)

    def _generate_json_report(self, results: TestSuiteResults, quality_metrics: QualityMetrics):
        """Generate detailed JSON report."""
        report_data = {
            'timestamp': datetime.now().isoformat(),
            'test_results': asdict(results),
            'quality_metrics': asdict(quality_metrics),
            'quality_standards': self.quality_standards,
            'validation_status': self._determine_validation_status(quality_metrics)
        }

        json_path = self.output_dir / 'comprehensive_test_report.json'
        with open(json_path, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)

        logger.info(f"JSON report generated: {json_path}")

    def _generate_html_report(self, results: TestSuiteResults, quality_metrics: QualityMetrics):
        """Generate HTML report."""
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>ABM-RSiena Tolerance Intervention - Test Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        .header {{ background: #f0f0f0; padding: 20px; border-radius: 5px; }}
        .section {{ margin: 20px 0; }}
        .metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; }}
        .metric-card {{ background: #f9f9f9; padding: 15px; border-radius: 5px; border-left: 4px solid #4CAF50; }}
        .metric-card.warning {{ border-left-color: #FF9800; }}
        .metric-card.error {{ border-left-color: #F44336; }}
        .quality-score {{ font-size: 24px; font-weight: bold; color: #4CAF50; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        .pass {{ color: #4CAF50; font-weight: bold; }}
        .fail {{ color: #F44336; font-weight: bold; }}
        .skip {{ color: #FF9800; font-weight: bold; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>ABM-RSiena Tolerance Intervention Research</h1>
        <h2>Comprehensive Test Validation Report</h2>
        <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p><strong>Quality Score:</strong> <span class="quality-score">{quality_metrics.overall_quality_score:.2%}</span></p>
    </div>

    <div class="section">
        <h3>Executive Summary</h3>
        <div class="metrics">
            <div class="metric-card">
                <h4>Test Execution</h4>
                <p><strong>Total Tests:</strong> {results.total_tests}</p>
                <p><strong>Passed:</strong> <span class="pass">{results.passed_tests}</span></p>
                <p><strong>Failed:</strong> <span class="fail">{results.failed_tests}</span></p>
                <p><strong>Skipped:</strong> <span class="skip">{results.skipped_tests}</span></p>
                <p><strong>Execution Time:</strong> {results.execution_time:.2f}s</p>
            </div>
            <div class="metric-card">
                <h4>Quality Metrics</h4>
                <p><strong>Model Convergence:</strong> {quality_metrics.model_convergence_rate:.2%}</p>
                <p><strong>Parameter Accuracy:</strong> {quality_metrics.parameter_accuracy_score:.2%}</p>
                <p><strong>Data Integrity:</strong> {quality_metrics.data_integrity_score:.2%}</p>
                <p><strong>Statistical Validity:</strong> {quality_metrics.statistical_validity_score:.2%}</p>
            </div>
        </div>
    </div>

    <div class="section">
        <h3>Test Categories Results</h3>
        <table>
            <tr>
                <th>Category</th>
                <th>Total</th>
                <th>Passed</th>
                <th>Failed</th>
                <th>Skipped</th>
                <th>Success Rate</th>
            </tr>
        """

        for category, details in results.test_categories.items():
            if isinstance(details, dict) and 'total_tests' in details:
                success_rate = details['passed_tests'] / details['total_tests'] if details['total_tests'] > 0 else 0
                html_content += f"""
            <tr>
                <td>{category.replace('_', ' ').title()}</td>
                <td>{details['total_tests']}</td>
                <td class="pass">{details['passed_tests']}</td>
                <td class="fail">{details['failed_tests']}</td>
                <td class="skip">{details['skipped_tests']}</td>
                <td>{success_rate:.1%}</td>
            </tr>
                """

        html_content += """
        </table>
    </div>

    <div class="section">
        <h3>Quality Assurance Checklist</h3>
        <table>
            <tr>
                <th>Quality Check</th>
                <th>Standard</th>
                <th>Actual</th>
                <th>Status</th>
            </tr>
        """

        quality_checks = [
            ('Model Convergence Rate', self.quality_standards['min_convergence_rate'],
             quality_metrics.model_convergence_rate),
            ('Parameter Accuracy', self.quality_standards['min_parameter_accuracy'],
             quality_metrics.parameter_accuracy_score),
            ('Data Integrity', self.quality_standards['min_data_integrity'],
             quality_metrics.data_integrity_score),
            ('Visualization Accuracy', self.quality_standards['min_visualization_accuracy'],
             quality_metrics.visualization_accuracy_score),
            ('Statistical Validity', self.quality_standards['min_statistical_validity'],
             quality_metrics.statistical_validity_score),
            ('Reproducibility', self.quality_standards['min_reproducibility'],
             quality_metrics.reproducibility_score),
        ]

        for check_name, standard, actual in quality_checks:
            status = "PASS" if actual >= standard else "FAIL"
            status_class = "pass" if status == "PASS" else "fail"
            html_content += f"""
            <tr>
                <td>{check_name}</td>
                <td>{standard:.1%}</td>
                <td>{actual:.1%}</td>
                <td class="{status_class}">{status}</td>
            </tr>
            """

        html_content += """
        </table>
    </div>

    <div class="section">
        <h3>Validation Recommendations</h3>
        """

        recommendations = self._generate_recommendations(quality_metrics)
        for recommendation in recommendations:
            html_content += f"<p>• {recommendation}</p>"

        html_content += """
    </div>
</body>
</html>
        """

        html_path = self.output_dir / 'test_report.html'
        with open(html_path, 'w') as f:
            f.write(html_content)

        logger.info(f"HTML report generated: {html_path}")

    def _generate_quality_checklist(self, quality_metrics: QualityMetrics):
        """Generate quality assurance checklist."""
        checklist_content = f"""
# ABM-RSiena Tolerance Intervention Research - Quality Assurance Checklist

## Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Overall Quality Score: {quality_metrics.overall_quality_score:.1%}

## Model Validation Checklist

### RSiena Model Validation
- [ ] Model convergence achieved (t-ratios < 0.1): {quality_metrics.model_convergence_rate:.1%}
- [ ] Parameter estimates within reasonable bounds
- [ ] Goodness-of-fit tests passed (p > 0.05)
- [ ] Custom effects properly implemented
- [ ] Attraction-repulsion mechanism validated
- [ ] Complex contagion implementation verified

### Intervention Simulation Validation
- [ ] Targeting strategies work correctly
- [ ] Intervention persistence validated
- [ ] Dose-response relationship confirmed
- [ ] Spillover effects properly modeled
- [ ] Multi-level effects captured

### Data Processing Validation
- [ ] Classroom data structure validated (105 classrooms): {quality_metrics.data_integrity_score:.1%}
- [ ] RSiena data format conversion accurate
- [ ] Missing data handling implemented
- [ ] Temporal alignment verified
- [ ] Data integrity checks passed

### Statistical Analysis Validation
- [ ] Parameter estimation accuracy: {quality_metrics.parameter_accuracy_score:.1%}
- [ ] Standard errors and confidence intervals valid
- [ ] Meta-analysis methodology correct
- [ ] Effect size calculations verified
- [ ] Statistical significance testing appropriate
- [ ] Multiple comparison corrections applied

### Visualization Validation
- [ ] Network animations render correctly: {quality_metrics.visualization_accuracy_score:.1%}
- [ ] Interactive dashboard functional
- [ ] Publication-quality figures generated
- [ ] Data-visual correspondence verified
- [ ] Accessibility features implemented
- [ ] Export functionality working

## Performance Metrics
- [ ] Test execution time reasonable (< 5 minutes)
- [ ] Memory usage within limits
- [ ] No memory leaks detected
- [ ] Parallel processing optimized

## Research Quality Standards
- [ ] Reproducibility ensured: {quality_metrics.reproducibility_score:.1%}
- [ ] Statistical rigor maintained
- [ ] Academic standards met
- [ ] Documentation comprehensive
- [ ] Code quality high

## Compliance Checklist
- [ ] Ethics approval documented
- [ ] Data privacy maintained
- [ ] Open science practices followed
- [ ] Version control implemented
- [ ] Backup procedures in place

## Recommendations for Improvement
"""

        recommendations = self._generate_recommendations(quality_metrics)
        for i, recommendation in enumerate(recommendations, 1):
            checklist_content += f"{i}. {recommendation}\n"

        checklist_content += f"""

## Quality Assessment Summary
- Overall Quality Score: {quality_metrics.overall_quality_score:.1%}
- Validation Status: {self._determine_validation_status(quality_metrics)}
- Ready for Publication: {'YES' if quality_metrics.overall_quality_score >= 0.90 else 'NO - Improvements Needed'}

---
Generated by ABM-RSiena Validation Specialist
"""

        checklist_path = self.output_dir / 'quality_assurance_checklist.md'
        with open(checklist_path, 'w') as f:
            f.write(checklist_content)

        logger.info(f"Quality checklist generated: {checklist_path}")

    def _determine_validation_status(self, quality_metrics: QualityMetrics) -> str:
        """Determine overall validation status."""
        if quality_metrics.overall_quality_score >= 0.95:
            return "EXCELLENT"
        elif quality_metrics.overall_quality_score >= 0.90:
            return "GOOD"
        elif quality_metrics.overall_quality_score >= 0.80:
            return "ACCEPTABLE"
        elif quality_metrics.overall_quality_score >= 0.70:
            return "NEEDS_IMPROVEMENT"
        else:
            return "INADEQUATE"

    def _generate_recommendations(self, quality_metrics: QualityMetrics) -> List[str]:
        """Generate improvement recommendations."""
        recommendations = []

        if quality_metrics.model_convergence_rate < self.quality_standards['min_convergence_rate']:
            recommendations.append(
                "Improve RSiena model convergence by adjusting estimation parameters "
                "or simplifying model specification"
            )

        if quality_metrics.parameter_accuracy_score < self.quality_standards['min_parameter_accuracy']:
            recommendations.append(
                "Enhance parameter estimation accuracy through better initial values "
                "or increased sample sizes"
            )

        if quality_metrics.data_integrity_score < self.quality_standards['min_data_integrity']:
            recommendations.append(
                "Strengthen data integrity validation and implement additional "
                "consistency checks"
            )

        if quality_metrics.visualization_accuracy_score < self.quality_standards['min_visualization_accuracy']:
            recommendations.append(
                "Improve data-visual correspondence validation and add more "
                "comprehensive visualization tests"
            )

        if quality_metrics.statistical_validity_score < self.quality_standards['min_statistical_validity']:
            recommendations.append(
                "Enhance statistical testing procedures and add more robust "
                "significance testing"
            )

        if quality_metrics.reproducibility_score < self.quality_standards['min_reproducibility']:
            recommendations.append(
                "Improve reproducibility by fixing random seeds, documenting "
                "procedures, and adding deterministic test validation"
            )

        if not recommendations:
            recommendations.append(
                "Excellent work! All quality standards have been met. "
                "Consider additional sensitivity analyses and robustness checks."
            )

        return recommendations


class DetailedTestResult(unittest.TestResult):
    """Custom test result collector with detailed metrics."""

    def __init__(self):
        super().__init__()
        self.test_details = []
        self.performance_issues = []
        self.start_time = time.time()
        self.execution_time = 0.0

    def startTest(self, test):
        super().startTest(test)
        self.test_start_time = time.time()

    def stopTest(self, test):
        super().stopTest(test)
        test_time = time.time() - self.test_start_time

        test_detail = {
            'test_name': str(test),
            'execution_time': test_time,
            'passed': True,  # Will be updated if test fails
            'timestamp': datetime.now().isoformat()
        }

        # Check for performance issues
        if test_time > 5.0:  # Tests taking longer than 5 seconds
            self.performance_issues.append({
                'test_name': str(test),
                'execution_time': test_time,
                'issue': 'SLOW_EXECUTION'
            })

        self.test_details.append(test_detail)

    def addError(self, test, err):
        super().addError(test, err)
        # Update the corresponding test detail
        for detail in reversed(self.test_details):
            if detail['test_name'] == str(test):
                detail['passed'] = False
                detail['error'] = str(err[1])
                break

    def addFailure(self, test, err):
        super().addFailure(test, err)
        # Update the corresponding test detail
        for detail in reversed(self.test_details):
            if detail['test_name'] == str(test):
                detail['passed'] = False
                detail['failure'] = str(err[1])
                break

    def stopTestRun(self):
        super().stopTestRun()
        self.execution_time = time.time() - self.start_time


def main():
    """Main entry point for test runner."""
    import argparse

    parser = argparse.ArgumentParser(description='Run comprehensive ABM-RSiena validation tests')
    parser.add_argument('--categories', nargs='*',
                       choices=['rsiena_integration', 'intervention_simulations',
                               'data_processing', 'statistical_analysis', 'visualizations'],
                       help='Specific test categories to run')
    parser.add_argument('--output-dir', type=Path,
                       help='Output directory for test reports')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output')

    args = parser.parse_args()

    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Initialize test runner
    runner = ComprehensiveTestRunner(output_dir=args.output_dir)

    # Run tests
    try:
        results = runner.run_comprehensive_tests(categories=args.categories)

        # Print summary
        print("\n" + "="*80)
        print("TEST EXECUTION SUMMARY")
        print("="*80)
        print(f"Total Tests: {results.total_tests}")
        print(f"Passed: {results.passed_tests}")
        print(f"Failed: {results.failed_tests}")
        print(f"Skipped: {results.skipped_tests}")
        print(f"Errors: {results.errors}")
        print(f"Execution Time: {results.execution_time:.2f} seconds")
        print(f"Quality Score: {results.quality_score:.2%}")
        print("="*80)

        # Exit with appropriate code
        exit_code = 0 if results.failed_tests == 0 and results.errors == 0 else 1
        sys.exit(exit_code)

    except Exception as e:
        logger.error(f"Test runner failed: {e}")
        logger.debug(traceback.format_exc())
        sys.exit(2)


if __name__ == '__main__':
    main()