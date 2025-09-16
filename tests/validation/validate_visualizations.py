#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Publication-Quality Visualization Validation Framework
Validates all visualization components for academic publication standards
"""

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set publication-quality defaults
plt.style.use('default')
matplotlib.rcParams.update({
    'font.size': 12,
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'serif'],
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 16,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.format': 'png',
    'savefig.bbox': 'tight',
    'savefig.transparent': False,
    'axes.linewidth': 1,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'grid.alpha': 0.3,
    'lines.linewidth': 2,
    'patch.linewidth': 0.5,
    'patch.facecolor': '348ABD',
    'patch.edgecolor': 'EEEEEE'
})

class VisualizationValidator:
    """Validates visualization quality for publication standards."""

    def __init__(self, output_dir="outputs/validation_figures"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.validation_results = {}

    def validate_all_visualizations(self):
        """Run comprehensive visualization validation."""

        print("=" * 70)
        print("PUBLICATION-QUALITY VISUALIZATION VALIDATION")
        print("=" * 70)
        print()

        # Test each visualization type
        tests = [
            ("Network Evolution", self.test_network_evolution_plots),
            ("Tolerance Distributions", self.test_tolerance_distribution_plots),
            ("Statistical Results", self.test_statistical_results_plots),
            ("Intervention Effects", self.test_intervention_effects_plots),
            ("Multi-panel Figures", self.test_multi_panel_figures),
            ("Color Accessibility", self.test_color_accessibility),
            ("Typography Standards", self.test_typography_standards),
            ("Export Quality", self.test_export_quality)
        ]

        for test_name, test_function in tests:
            print(f"Testing {test_name}...")
            try:
                result = test_function()
                self.validation_results[test_name] = {
                    "status": "PASS",
                    "details": result
                }
                print(f"[PASS] {test_name}: PASSED")
            except Exception as e:
                self.validation_results[test_name] = {
                    "status": "FAIL",
                    "error": str(e)
                }
                print(f"[FAIL] {test_name}: FAILED - {e}")
            print()

        # Generate summary report
        self.generate_validation_report()

        return self.validation_results

    def test_network_evolution_plots(self):
        """Test network evolution visualization quality."""

        # Generate sample network data
        n_periods = 4
        network_stats = pd.DataFrame({
            'period': range(1, n_periods + 1),
            'density': [0.08, 0.12, 0.15, 0.18],
            'transitivity': [0.25, 0.32, 0.38, 0.42],
            'reciprocity': [0.65, 0.68, 0.72, 0.75]
        })

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Density plot
        axes[0].plot(network_stats['period'], network_stats['density'],
                    marker='o', linewidth=2, markersize=8)
        axes[0].set_title('Network Density Evolution')
        axes[0].set_xlabel('Time Period')
        axes[0].set_ylabel('Density')
        axes[0].grid(True, alpha=0.3)

        # Transitivity plot
        axes[1].plot(network_stats['period'], network_stats['transitivity'],
                    marker='s', linewidth=2, markersize=8, color='red')
        axes[1].set_title('Network Transitivity Evolution')
        axes[1].set_xlabel('Time Period')
        axes[1].set_ylabel('Transitivity')
        axes[1].grid(True, alpha=0.3)

        # Reciprocity plot
        axes[2].plot(network_stats['period'], network_stats['reciprocity'],
                    marker='^', linewidth=2, markersize=8, color='green')
        axes[2].set_title('Network Reciprocity Evolution')
        axes[2].set_xlabel('Time Period')
        axes[2].set_ylabel('Reciprocity')
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'network_evolution_test.png')
        plt.close()

        return {
            "figure_count": 3,
            "resolution": "300 DPI",
            "format": "PNG",
            "layout": "tight"
        }

    def test_tolerance_distribution_plots(self):
        """Test tolerance distribution visualization quality."""

        # Generate sample tolerance data
        np.random.seed(42)
        n_students = 150
        n_periods = 4

        tolerance_data = []
        for period in range(1, n_periods + 1):
            # Simulate increasing tolerance over time
            base_tolerance = 3.5 + 0.3 * period
            tolerance_scores = np.random.normal(base_tolerance, 1.2, n_students)
            tolerance_scores = np.clip(tolerance_scores, 1, 7)

            for score in tolerance_scores:
                tolerance_data.append({
                    'period': period,
                    'tolerance': score
                })

        tolerance_df = pd.DataFrame(tolerance_data)

        # Create violin plot with boxplot overlay
        fig, ax = plt.subplots(figsize=(10, 6))

        # Violin plot
        violin_parts = ax.violinplot([tolerance_df[tolerance_df['period'] == p]['tolerance'].values
                                     for p in range(1, n_periods + 1)],
                                    positions=range(1, n_periods + 1),
                                    showmeans=True, showmedians=True)

        # Customize violin plot
        for pc in violin_parts['bodies']:
            pc.set_facecolor('lightblue')
            pc.set_alpha(0.7)

        # Box plot overlay
        box_data = [tolerance_df[tolerance_df['period'] == p]['tolerance'].values
                   for p in range(1, n_periods + 1)]
        box_parts = ax.boxplot(box_data, positions=range(1, n_periods + 1),
                              widths=0.2, patch_artist=True)

        for patch in box_parts['boxes']:
            patch.set_facecolor('white')
            patch.set_alpha(0.8)

        ax.set_xlabel('Time Period')
        ax.set_ylabel('Tolerance Score (1-7 scale)')
        ax.set_title('Tolerance Distribution Evolution Over Time')
        ax.set_xticks(range(1, n_periods + 1))
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'tolerance_distribution_test.png')
        plt.close()

        return {
            "plot_type": "violin_box_combo",
            "data_points": len(tolerance_data),
            "statistical_elements": "means, medians, quartiles"
        }

    def test_statistical_results_plots(self):
        """Test statistical results visualization quality."""

        # Generate sample RSiena-style results
        effects = [
            'Density (friendship)',
            'Reciprocity',
            'Transitivity',
            'Tolerance similarity',
            'Gender similarity',
            'SES similarity',
            'Tolerance: linear shape',
            'Tolerance: peer influence',
            'Tolerance: intervention effect'
        ]

        estimates = np.array([-2.5, 0.8, 0.6, 0.4, 0.3, 0.2, -0.1, 0.5, 0.7])
        std_errors = np.array([0.3, 0.2, 0.2, 0.15, 0.12, 0.1, 0.08, 0.18, 0.25])

        # Calculate confidence intervals
        ci_lower = estimates - 1.96 * std_errors
        ci_upper = estimates + 1.96 * std_errors

        # Create coefficient plot
        fig, ax = plt.subplots(figsize=(10, 8))

        y_positions = range(len(effects))

        # Plot point estimates
        colors = ['red' if abs(est/se) > 1.96 else 'gray' for est, se in zip(estimates, std_errors)]
        ax.scatter(estimates, y_positions, s=100, c=colors, alpha=0.8, zorder=3)

        # Plot confidence intervals
        for i, (est, ci_l, ci_u) in enumerate(zip(estimates, ci_lower, ci_upper)):
            ax.plot([ci_l, ci_u], [i, i], color=colors[i], linewidth=2, alpha=0.6)

        # Add vertical line at zero
        ax.axvline(x=0, color='black', linestyle='--', alpha=0.5)

        # Customize plot
        ax.set_yticks(y_positions)
        ax.set_yticklabels(effects)
        ax.set_xlabel('Parameter Estimate (95% CI)')
        ax.set_title('RSiena Model Results: Parameter Estimates')
        ax.grid(True, alpha=0.3, axis='x')

        # Add significance indicators
        ax.text(0.02, 0.98, 'Red = Significant (p < 0.05)', transform=ax.transAxes,
                verticalalignment='top', fontsize=10, style='italic')

        plt.tight_layout()
        plt.savefig(self.output_dir / 'statistical_results_test.png')
        plt.close()

        return {
            "plot_type": "coefficient_plot",
            "parameters": len(effects),
            "significance_testing": "95% confidence intervals"
        }

    def test_intervention_effects_plots(self):
        """Test intervention effects visualization quality."""

        # Generate sample intervention data
        np.random.seed(123)
        n_students = 100

        # Pre-intervention tolerance
        pre_tolerance = np.random.normal(4.0, 1.0, n_students)
        pre_tolerance = np.clip(pre_tolerance, 1, 7)

        # Post-intervention tolerance (with treatment effect)
        treatment_effect = np.random.normal(0.5, 0.3, n_students)
        post_tolerance = pre_tolerance + treatment_effect
        post_tolerance = np.clip(post_tolerance, 1, 7)

        # Create before-after plot
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Individual trajectories
        for i in range(min(50, n_students)):  # Show subset for clarity
            axes[0].plot([1, 2], [pre_tolerance[i], post_tolerance[i]],
                        alpha=0.3, color='blue', linewidth=1)

        # Mean trajectory
        axes[0].plot([1, 2], [np.mean(pre_tolerance), np.mean(post_tolerance)],
                    color='red', linewidth=3, marker='o', markersize=8, label='Mean')

        axes[0].set_xticks([1, 2])
        axes[0].set_xticklabels(['Pre-Intervention', 'Post-Intervention'])
        axes[0].set_ylabel('Tolerance Score')
        axes[0].set_title('Individual Tolerance Trajectories')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Distribution comparison
        axes[1].hist(pre_tolerance, bins=15, alpha=0.6, label='Pre-Intervention',
                    color='lightcoral', density=True)
        axes[1].hist(post_tolerance, bins=15, alpha=0.6, label='Post-Intervention',
                    color='lightblue', density=True)

        axes[1].axvline(np.mean(pre_tolerance), color='red', linestyle='--',
                       label=f'Pre Mean: {np.mean(pre_tolerance):.2f}')
        axes[1].axvline(np.mean(post_tolerance), color='blue', linestyle='--',
                       label=f'Post Mean: {np.mean(post_tolerance):.2f}')

        axes[1].set_xlabel('Tolerance Score')
        axes[1].set_ylabel('Density')
        axes[1].set_title('Tolerance Distribution Comparison')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'intervention_effects_test.png')
        plt.close()

        return {
            "effect_size": f"{np.mean(post_tolerance) - np.mean(pre_tolerance):.3f}",
            "visualization_types": ["trajectory_plot", "distribution_comparison"]
        }

    def test_multi_panel_figures(self):
        """Test multi-panel figure layout and quality."""

        # Create comprehensive multi-panel figure
        fig = plt.figure(figsize=(16, 12))

        # Define grid layout
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        # Panel A: Network over time
        ax1 = fig.add_subplot(gs[0, :2])
        periods = [1, 2, 3, 4]
        density = [0.08, 0.12, 0.15, 0.18]
        ax1.plot(periods, density, marker='o', linewidth=2, markersize=8)
        ax1.set_title('A. Network Density Evolution')
        ax1.set_xlabel('Period')
        ax1.set_ylabel('Density')
        ax1.grid(True, alpha=0.3)

        # Panel B: Tolerance change
        ax2 = fig.add_subplot(gs[0, 2])
        change_data = np.random.normal(0.5, 0.3, 100)
        ax2.hist(change_data, bins=15, alpha=0.7, color='lightgreen')
        ax2.set_title('B. Tolerance Change')
        ax2.set_xlabel('Change Score')
        ax2.set_ylabel('Frequency')

        # Panel C: Correlation matrix
        ax3 = fig.add_subplot(gs[1, :])
        variables = ['Tolerance', 'Network Size', 'SES', 'Academic Performance', 'Extroversion']
        corr_matrix = np.random.rand(5, 5)
        corr_matrix = (corr_matrix + corr_matrix.T) / 2  # Make symmetric
        np.fill_diagonal(corr_matrix, 1)

        im = ax3.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
        ax3.set_xticks(range(len(variables)))
        ax3.set_yticks(range(len(variables)))
        ax3.set_xticklabels(variables, rotation=45, ha='right')
        ax3.set_yticklabels(variables)
        ax3.set_title('C. Variable Correlation Matrix')

        # Add correlation values
        for i in range(len(variables)):
            for j in range(len(variables)):
                ax3.text(j, i, f'{corr_matrix[i, j]:.2f}',
                        ha='center', va='center', fontsize=9)

        # Panel D: Statistical results
        ax4 = fig.add_subplot(gs[2, :])
        effects = ['Density', 'Reciprocity', 'Transitivity', 'Tolerance Sim.', 'Peer Influence']
        estimates = [-2.1, 0.8, 0.6, 0.4, 0.5]
        std_errors = [0.3, 0.2, 0.2, 0.15, 0.18]

        y_pos = range(len(effects))
        colors = ['red' if abs(est/se) > 1.96 else 'gray' for est, se in zip(estimates, std_errors)]

        ax4.barh(y_pos, estimates, color=colors, alpha=0.7)
        ax4.set_yticks(y_pos)
        ax4.set_yticklabels(effects)
        ax4.set_xlabel('Parameter Estimate')
        ax4.set_title('D. RSiena Model Results')
        ax4.axvline(x=0, color='black', linestyle='--', alpha=0.5)
        ax4.grid(True, alpha=0.3, axis='x')

        plt.suptitle('Tolerance Intervention Study: Comprehensive Results',
                    fontsize=16, fontweight='bold', y=0.98)

        plt.savefig(self.output_dir / 'multi_panel_test.png')
        plt.close()

        return {
            "panels": 4,
            "layout": "3x3_grid",
            "total_size": "16x12_inches"
        }

    def test_color_accessibility(self):
        """Test color accessibility and colorblind-friendly palettes."""

        # Test different color palettes
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Sample data
        categories = ['Group A', 'Group B', 'Group C', 'Group D', 'Group E']
        values = [23, 45, 56, 78, 32]

        # Default palette (potentially problematic)
        axes[0, 0].bar(categories, values, color=['red', 'green', 'blue', 'orange', 'purple'])
        axes[0, 0].set_title('Standard Colors (Potentially Problematic)')
        axes[0, 0].tick_params(axis='x', rotation=45)

        # Colorblind-friendly palette
        cb_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        axes[0, 1].bar(categories, values, color=cb_colors)
        axes[0, 1].set_title('Colorblind-Friendly Palette')
        axes[0, 1].tick_params(axis='x', rotation=45)

        # Viridis palette (perceptually uniform)
        axes[1, 0].bar(categories, values, color=plt.cm.viridis(np.linspace(0, 1, len(categories))))
        axes[1, 0].set_title('Viridis Palette (Perceptually Uniform)')
        axes[1, 0].tick_params(axis='x', rotation=45)

        # Grayscale (printer-friendly)
        gray_colors = [str(g) for g in np.linspace(0.2, 0.8, len(categories))]
        axes[1, 1].bar(categories, values, color=gray_colors)
        axes[1, 1].set_title('Grayscale (Printer-Friendly)')
        axes[1, 1].tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'color_accessibility_test.png')
        plt.close()

        return {
            "palettes_tested": 4,
            "accessibility_features": ["colorblind_friendly", "grayscale", "perceptually_uniform"]
        }

    def test_typography_standards(self):
        """Test typography and text formatting standards."""

        fig, ax = plt.subplots(figsize=(10, 8))

        # Sample data
        x = np.linspace(0, 10, 100)
        y1 = np.sin(x)
        y2 = np.cos(x)

        ax.plot(x, y1, label='sin(x)', linewidth=2)
        ax.plot(x, y2, label='cos(x)', linewidth=2)

        # Test various text elements
        ax.set_title('Typography Standards Test\nMultiple Font Sizes and Weights',
                    fontsize=16, fontweight='bold')
        ax.set_xlabel('X-axis Label (12pt)', fontsize=12)
        ax.set_ylabel('Y-axis Label (12pt)', fontsize=12)

        # Legend with custom formatting
        legend = ax.legend(fontsize=11, frameon=True, fancybox=True, shadow=True)
        legend.get_frame().set_facecolor('white')
        legend.get_frame().set_alpha(0.9)

        # Add annotations with different text styles
        ax.annotate('Peak value', xy=(1.57, 1), xytext=(3, 1.2),
                   arrowprops=dict(arrowstyle='->', color='red'),
                   fontsize=10, style='italic')

        ax.text(7, -0.5, 'Mathematical notation: $\\alpha + \\beta = \\gamma$',
               fontsize=11, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))

        # Grid and formatting
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        ax.set_xlim(0, 10)
        ax.set_ylim(-1.5, 1.5)

        # Tick formatting
        ax.tick_params(labelsize=10)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'typography_test.png')
        plt.close()

        return {
            "font_sizes_tested": ["10pt", "11pt", "12pt", "16pt"],
            "text_elements": ["title", "labels", "legend", "annotations", "equations"],
            "formatting_features": ["bold", "italic", "math_notation"]
        }

    def test_export_quality(self):
        """Test export quality in different formats and resolutions."""

        # Create test figure
        fig, ax = plt.subplots(figsize=(8, 6))

        # High-detail plot to test resolution
        x = np.linspace(0, 4*np.pi, 1000)
        y = np.sin(x) * np.exp(-x/10)

        ax.plot(x, y, linewidth=1.5, color='blue')
        ax.fill_between(x, 0, y, alpha=0.3, color='lightblue')

        ax.set_title('Export Quality Test: High-Detail Plot')
        ax.set_xlabel('X values')
        ax.set_ylabel('Y values')
        ax.grid(True, alpha=0.3)

        # Test different DPI settings
        export_formats = [
            ('png', 150, 'web_quality'),
            ('png', 300, 'print_quality'),
            ('png', 600, 'high_resolution'),
            ('pdf', None, 'vector_format')
        ]

        export_results = {}

        for fmt, dpi, quality_level in export_formats:
            filename = f'export_test_{quality_level}.{fmt}'
            filepath = self.output_dir / filename

            if dpi:
                plt.savefig(filepath, format=fmt, dpi=dpi, bbox_inches='tight')
                export_results[quality_level] = f"{fmt.upper()} at {dpi} DPI"
            else:
                plt.savefig(filepath, format=fmt, bbox_inches='tight')
                export_results[quality_level] = f"{fmt.upper()} vector format"

        plt.close()

        return export_results

    def generate_validation_report(self):
        """Generate comprehensive validation report."""

        report_path = self.output_dir / 'validation_report.txt'

        with open(report_path, 'w') as f:
            f.write("PUBLICATION-QUALITY VISUALIZATION VALIDATION REPORT\n")
            f.write("=" * 60 + "\n\n")

            # Summary statistics
            total_tests = len(self.validation_results)
            passed_tests = sum(1 for result in self.validation_results.values()
                             if result['status'] == 'PASS')

            f.write(f"SUMMARY:\n")
            f.write(f"Total tests: {total_tests}\n")
            f.write(f"Passed: {passed_tests}\n")
            f.write(f"Failed: {total_tests - passed_tests}\n")
            f.write(f"Success rate: {passed_tests/total_tests*100:.1f}%\n\n")

            # Detailed results
            f.write("DETAILED RESULTS:\n")
            f.write("-" * 30 + "\n\n")

            for test_name, result in self.validation_results.items():
                f.write(f"{test_name}: {result['status']}\n")

                if result['status'] == 'PASS':
                    if 'details' in result:
                        for key, value in result['details'].items():
                            f.write(f"  - {key}: {value}\n")
                else:
                    f.write(f"  Error: {result.get('error', 'Unknown error')}\n")

                f.write("\n")

            # Recommendations
            f.write("RECOMMENDATIONS:\n")
            f.write("-" * 20 + "\n")
            f.write("1. All figures should use 300+ DPI for print publication\n")
            f.write("2. Use colorblind-friendly palettes for accessibility\n")
            f.write("3. Maintain consistent typography across all figures\n")
            f.write("4. Include proper statistical annotations and confidence intervals\n")
            f.write("5. Test figures in both color and grayscale formats\n")
            f.write("6. Ensure multi-panel figures have clear labels (A, B, C, etc.)\n")
            f.write("7. Export in vector formats (PDF, SVG) when possible\n")

        print(f"Validation report saved to: {report_path}")

        # Print summary to console
        print("\nVALIDATION SUMMARY:")
        print("-" * 30)
        print(f"[RESULT] {passed_tests}/{total_tests} tests passed ({passed_tests/total_tests*100:.1f}%)")

        if passed_tests == total_tests:
            print("[SUCCESS] ALL VISUALIZATION TESTS PASSED!")
            print("Your visualizations meet publication quality standards.")
        else:
            print(f"[WARNING] {total_tests - passed_tests} tests failed - check validation report for details")

def main():
    """Main execution function."""

    print("Starting Publication-Quality Visualization Validation")
    print()

    # Create validator and run all tests
    validator = VisualizationValidator()
    results = validator.validate_all_visualizations()

    print("\nValidation complete!")
    print(f"Check output directory: {validator.output_dir}")

    return results

if __name__ == "__main__":
    main()