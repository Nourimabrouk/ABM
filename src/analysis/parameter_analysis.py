"""
Parameter Analysis and Interpretation Module

This module provides comprehensive parameter analysis and interpretation
for tolerance-cooperation SAOM models. Includes parameter significance testing,
effect size interpretation, and theoretical linking to research hypotheses.

Key Features:
- Parameter significance testing with multiple comparison corrections
- Effect size calculation and interpretation guidelines
- Theoretical hypothesis testing framework
- Parameter stability and sensitivity analysis
- Cross-classroom parameter comparison
- Publication-ready result formatting

Research Context:
Supports interpretation of micro-theory effects:
- Attraction-repulsion influence mechanism
- Complex contagion effects
- Tolerance → cooperation selection
- Control for prejudice and demographic factors

Author: RSiena Integration Specialist
Created: 2025-09-16
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
from pathlib import Path
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.multitest import multipletests

from ..models.tolerance_cooperation_saom import SAOMResults
from ..analysis.saom_estimation import MetaAnalysisResults

logger = logging.getLogger(__name__)


@dataclass
class ParameterInterpretation:
    """Interpretation guidelines for SAOM parameters."""
    effect_name: str
    parameter_estimate: float
    standard_error: float
    t_statistic: float
    p_value: float
    effect_size_category: str  # "small", "medium", "large", "very_large"
    significance_level: str  # "***", "**", "*", ".", ""
    theoretical_interpretation: str
    practical_significance: bool
    confidence_interval: Tuple[float, float]


@dataclass
class HypothesisTest:
    """Hypothesis test for tolerance-cooperation research."""
    hypothesis_name: str
    description: str
    target_effects: List[str]
    expected_direction: str  # "positive", "negative", "any"
    minimum_effect_size: float
    test_statistic: float
    p_value: float
    supported: bool
    evidence_strength: str  # "strong", "moderate", "weak", "none"


class ParameterAnalyzer:
    """
    Comprehensive parameter analysis and interpretation.

    Provides tools for analyzing SAOM parameter estimates,
    testing theoretical hypotheses, and interpreting results
    in the context of tolerance-cooperation dynamics.
    """

    def __init__(self):
        """Initialize parameter analyzer."""
        self.interpretation_cache = {}
        self.hypothesis_tests = {}

        # Effect size thresholds (Cohen's guidelines adapted for SAOM)
        self.effect_size_thresholds = {
            'small': 0.1,
            'medium': 0.3,
            'large': 0.5,
            'very_large': 0.8
        }

        # Define research hypotheses
        self._define_research_hypotheses()

    def _define_research_hypotheses(self):
        """Define theoretical hypotheses for tolerance-cooperation model."""
        self.research_hypotheses = [
            HypothesisTest(
                hypothesis_name="H1_Attraction_Repulsion",
                description="Tolerance spreads through attraction-repulsion influence mechanism",
                target_effects=["attractionRepulsionInfluence"],
                expected_direction="positive",
                minimum_effect_size=0.1,
                test_statistic=0.0,
                p_value=1.0,
                supported=False,
                evidence_strength="none"
            ),
            HypothesisTest(
                hypothesis_name="H2_Tolerance_Cooperation",
                description="Increased tolerance leads to more interethnic cooperation",
                target_effects=["toleranceCooperationSelection", "altX.tolerance"],
                expected_direction="positive",
                minimum_effect_size=0.15,
                test_statistic=0.0,
                p_value=1.0,
                supported=False,
                evidence_strength="none"
            ),
            HypothesisTest(
                hypothesis_name="H3_Ethnicity_Homophily",
                description="Baseline ethnic homophily in friendship and cooperation",
                target_effects=["sameX.ethnicity"],
                expected_direction="positive",
                minimum_effect_size=0.2,
                test_statistic=0.0,
                p_value=1.0,
                supported=False,
                evidence_strength="none"
            ),
            HypothesisTest(
                hypothesis_name="H4_Prejudice_Control",
                description="Prejudice negatively affects interethnic cooperation (control effect)",
                target_effects=["egoX.prejudice", "altX.prejudice"],
                expected_direction="negative",
                minimum_effect_size=0.1,
                test_statistic=0.0,
                p_value=1.0,
                supported=False,
                evidence_strength="none"
            ),
            HypothesisTest(
                hypothesis_name="H5_Social_Influence",
                description="Social influence on tolerance through friendship networks",
                target_effects=["avSim.tolerance", "totSim.tolerance"],
                expected_direction="positive",
                minimum_effect_size=0.1,
                test_statistic=0.0,
                p_value=1.0,
                supported=False,
                evidence_strength="none"
            )
        ]

    def analyze_parameters(
        self,
        results: Union[SAOMResults, MetaAnalysisResults],
        alpha: float = 0.05,
        correction_method: str = "holm"
    ) -> Dict[str, ParameterInterpretation]:
        """
        Comprehensive analysis of SAOM parameters.

        Args:
            results: SAOM or meta-analysis results
            alpha: Significance level
            correction_method: Multiple comparison correction method

        Returns:
            Dictionary of parameter interpretations
        """
        logger.info("Conducting comprehensive parameter analysis...")

        # Extract parameter data
        if isinstance(results, SAOMResults):
            param_data = self._extract_saom_parameters(results)
        else:  # MetaAnalysisResults
            param_data = self._extract_meta_parameters(results)

        # Apply multiple comparison correction
        if correction_method and len(param_data) > 1:
            param_data = self._apply_multiple_comparison_correction(param_data, alpha, correction_method)

        # Interpret each parameter
        interpretations = {}
        for effect_name, data in param_data.items():
            interpretation = self._interpret_parameter(effect_name, data, alpha)
            interpretations[effect_name] = interpretation

        # Store results
        self.interpretation_cache = interpretations

        logger.info(f"Parameter analysis completed for {len(interpretations)} effects")
        return interpretations

    def _extract_saom_parameters(self, results: SAOMResults) -> Dict[str, Dict[str, float]]:
        """Extract parameter data from SAOM results."""
        param_data = {}

        for i, effect_name in enumerate(results.effect_names):
            param_data[effect_name] = {
                'estimate': results.parameters[i],
                'se': results.standard_errors[i],
                't_statistic': results.t_statistics[i],
                'p_value': 2 * (1 - stats.t.cdf(abs(results.t_statistics[i]), df=100))  # Approximate
            }

        return param_data

    def _extract_meta_parameters(self, results: MetaAnalysisResults) -> Dict[str, Dict[str, float]]:
        """Extract parameter data from meta-analysis results."""
        param_data = {}

        for effect_name in results.effect_estimates.keys():
            estimate = results.effect_estimates[effect_name]
            se = results.effect_se[effect_name]
            t_statistic = estimate / se if se > 0 else 0

            param_data[effect_name] = {
                'estimate': estimate,
                'se': se,
                't_statistic': t_statistic,
                'p_value': 2 * (1 - stats.t.cdf(abs(t_statistic), df=results.n_studies - 1))
            }

        return param_data

    def _apply_multiple_comparison_correction(
        self,
        param_data: Dict[str, Dict[str, float]],
        alpha: float,
        method: str
    ) -> Dict[str, Dict[str, float]]:
        """Apply multiple comparison correction to p-values."""
        p_values = [data['p_value'] for data in param_data.values()]
        effect_names = list(param_data.keys())

        # Apply correction
        rejected, p_corrected, alpha_sidak, alpha_bonf = multipletests(
            p_values, alpha=alpha, method=method
        )

        # Update p-values
        for i, effect_name in enumerate(effect_names):
            param_data[effect_name]['p_value_corrected'] = p_corrected[i]
            param_data[effect_name]['significant_corrected'] = rejected[i]

        logger.debug(f"Applied {method} correction: {sum(rejected)} significant effects")
        return param_data

    def _interpret_parameter(
        self,
        effect_name: str,
        param_data: Dict[str, float],
        alpha: float
    ) -> ParameterInterpretation:
        """Interpret a single parameter."""
        estimate = param_data['estimate']
        se = param_data['se']
        t_stat = param_data['t_statistic']
        p_value = param_data.get('p_value_corrected', param_data['p_value'])

        # Effect size category
        abs_estimate = abs(estimate)
        if abs_estimate < self.effect_size_thresholds['small']:
            effect_size_category = "negligible"
        elif abs_estimate < self.effect_size_thresholds['medium']:
            effect_size_category = "small"
        elif abs_estimate < self.effect_size_thresholds['large']:
            effect_size_category = "medium"
        elif abs_estimate < self.effect_size_thresholds['very_large']:
            effect_size_category = "large"
        else:
            effect_size_category = "very_large"

        # Significance level
        if p_value < 0.001:
            significance_level = "***"
        elif p_value < 0.01:
            significance_level = "**"
        elif p_value < 0.05:
            significance_level = "*"
        elif p_value < 0.1:
            significance_level = "."
        else:
            significance_level = ""

        # Confidence interval
        ci_lower = estimate - 1.96 * se
        ci_upper = estimate + 1.96 * se

        # Theoretical interpretation
        theoretical_interpretation = self._get_theoretical_interpretation(effect_name, estimate)

        # Practical significance
        practical_significance = (
            abs_estimate >= self.effect_size_thresholds['small'] and
            p_value < alpha
        )

        return ParameterInterpretation(
            effect_name=effect_name,
            parameter_estimate=estimate,
            standard_error=se,
            t_statistic=t_stat,
            p_value=p_value,
            effect_size_category=effect_size_category,
            significance_level=significance_level,
            theoretical_interpretation=theoretical_interpretation,
            practical_significance=practical_significance,
            confidence_interval=(ci_lower, ci_upper)
        )

    def _get_theoretical_interpretation(self, effect_name: str, estimate: float) -> str:
        """Get theoretical interpretation for specific effects."""
        direction = "positive" if estimate > 0 else "negative"

        interpretations = {
            'density': f"Baseline tendency to form ties is {direction}",
            'recip': f"Reciprocity tendency is {direction}",
            'transTrip': f"Transitivity (clustering) tendency is {direction}",
            'sameX.ethnicity': f"Ethnic homophily is {direction}",
            'sameX.gender': f"Gender homophily is {direction}",
            'avSim.tolerance': f"Average similarity influence on tolerance is {direction}",
            'totSim.tolerance': f"Total similarity influence on tolerance is {direction}",
            'attractionRepulsionInfluence': f"Attraction-repulsion influence mechanism is {direction}",
            'complexContagionInfluence': f"Complex contagion influence is {direction}",
            'toleranceCooperationSelection': f"Tolerance → cooperation selection effect is {direction}",
            'altX.tolerance': f"Selection of tolerant others is {direction}",
            'egoX.prejudice': f"Own prejudice effect on tie formation is {direction}",
            'altX.prejudice': f"Others' prejudice effect on being selected is {direction}",
            'linear.tolerance': f"Linear tolerance shape parameter is {direction}",
            'quad.tolerance': f"Quadratic tolerance shape parameter is {direction}"
        }

        return interpretations.get(effect_name, f"Effect {effect_name} is {direction}")

    def test_research_hypotheses(
        self,
        interpretations: Dict[str, ParameterInterpretation]
    ) -> Dict[str, HypothesisTest]:
        """
        Test predefined research hypotheses.

        Args:
            interpretations: Parameter interpretations

        Returns:
            Dictionary of hypothesis test results
        """
        logger.info("Testing research hypotheses...")

        hypothesis_results = {}

        for hypothesis in self.research_hypotheses:
            # Find relevant effects
            relevant_effects = [
                effect for effect in hypothesis.target_effects
                if effect in interpretations
            ]

            if not relevant_effects:
                logger.warning(f"No relevant effects found for hypothesis {hypothesis.hypothesis_name}")
                hypothesis_results[hypothesis.hypothesis_name] = hypothesis
                continue

            # Test hypothesis
            test_result = self._test_single_hypothesis(hypothesis, interpretations, relevant_effects)
            hypothesis_results[hypothesis.hypothesis_name] = test_result

        self.hypothesis_tests = hypothesis_results
        logger.info(f"Hypothesis testing completed: {len(hypothesis_results)} hypotheses tested")

        return hypothesis_results

    def _test_single_hypothesis(
        self,
        hypothesis: HypothesisTest,
        interpretations: Dict[str, ParameterInterpretation],
        relevant_effects: List[str]
    ) -> HypothesisTest:
        """Test a single research hypothesis."""
        # Collect evidence
        supporting_effects = 0
        total_effects = len(relevant_effects)
        combined_p_values = []
        effect_sizes = []

        for effect_name in relevant_effects:
            interp = interpretations[effect_name]
            estimate = interp.parameter_estimate
            p_value = interp.p_value

            # Check direction
            direction_correct = (
                (hypothesis.expected_direction == "positive" and estimate > 0) or
                (hypothesis.expected_direction == "negative" and estimate < 0) or
                hypothesis.expected_direction == "any"
            )

            # Check magnitude
            magnitude_sufficient = abs(estimate) >= hypothesis.minimum_effect_size

            # Check significance
            is_significant = interp.practical_significance

            if direction_correct and magnitude_sufficient and is_significant:
                supporting_effects += 1

            combined_p_values.append(p_value)
            effect_sizes.append(abs(estimate))

        # Combine evidence using Fisher's method
        if combined_p_values:
            # Fisher's combined probability test
            fisher_statistic = -2 * sum(np.log(p) for p in combined_p_values if p > 0)
            df = 2 * len(combined_p_values)
            combined_p_value = 1 - stats.chi2.cdf(fisher_statistic, df)
        else:
            combined_p_value = 1.0

        # Overall test statistic (mean effect size)
        mean_effect_size = np.mean(effect_sizes) if effect_sizes else 0.0

        # Determine support
        support_rate = supporting_effects / total_effects if total_effects > 0 else 0
        supported = support_rate >= 0.5 and combined_p_value < 0.05

        # Evidence strength
        if combined_p_value < 0.001 and support_rate >= 0.8:
            evidence_strength = "strong"
        elif combined_p_value < 0.01 and support_rate >= 0.6:
            evidence_strength = "moderate"
        elif combined_p_value < 0.05 and support_rate >= 0.5:
            evidence_strength = "weak"
        else:
            evidence_strength = "none"

        # Create updated hypothesis
        updated_hypothesis = HypothesisTest(
            hypothesis_name=hypothesis.hypothesis_name,
            description=hypothesis.description,
            target_effects=hypothesis.target_effects,
            expected_direction=hypothesis.expected_direction,
            minimum_effect_size=hypothesis.minimum_effect_size,
            test_statistic=mean_effect_size,
            p_value=combined_p_value,
            supported=supported,
            evidence_strength=evidence_strength
        )

        return updated_hypothesis

    def create_parameter_summary_table(
        self,
        interpretations: Dict[str, ParameterInterpretation],
        output_path: Optional[Union[str, Path]] = None
    ) -> pd.DataFrame:
        """
        Create publication-ready parameter summary table.

        Args:
            interpretations: Parameter interpretations
            output_path: Optional path to save table

        Returns:
            Summary table DataFrame
        """
        # Prepare data for table
        table_data = []

        for effect_name, interp in interpretations.items():
            row = {
                'Effect': effect_name.replace('_', ' ').replace('.', ' → '),
                'Estimate': f"{interp.parameter_estimate:.3f}",
                'SE': f"{interp.standard_error:.3f}",
                'CI_Lower': f"{interp.confidence_interval[0]:.3f}",
                'CI_Upper': f"{interp.confidence_interval[1]:.3f}",
                't-statistic': f"{interp.t_statistic:.2f}",
                'p-value': f"{interp.p_value:.3f}" if interp.p_value >= 0.001 else "< 0.001",
                'Significance': interp.significance_level,
                'Effect_Size': interp.effect_size_category.title(),
                'Interpretation': interp.theoretical_interpretation
            }
            table_data.append(row)

        # Create DataFrame
        summary_df = pd.DataFrame(table_data)

        # Sort by effect importance (custom order)
        effect_order = [
            'density', 'recip', 'transTrip', 'attractionRepulsionInfluence',
            'toleranceCooperationSelection', 'sameX.ethnicity', 'avSim.tolerance',
            'altX.tolerance', 'egoX.prejudice', 'linear.tolerance'
        ]

        # Custom sort
        summary_df['sort_key'] = summary_df['Effect'].apply(
            lambda x: effect_order.index(x.replace(' ', '.').replace(' → ', '.'))
            if x.replace(' ', '.').replace(' → ', '.') in effect_order else 999
        )
        summary_df = summary_df.sort_values('sort_key').drop('sort_key', axis=1)

        # Save if path provided
        if output_path:
            summary_df.to_csv(output_path, index=False)
            logger.info(f"Parameter summary table saved to {output_path}")

        return summary_df

    def create_hypothesis_summary_table(
        self,
        hypothesis_results: Dict[str, HypothesisTest],
        output_path: Optional[Union[str, Path]] = None
    ) -> pd.DataFrame:
        """
        Create summary table for hypothesis test results.

        Args:
            hypothesis_results: Hypothesis test results
            output_path: Optional path to save table

        Returns:
            Hypothesis summary DataFrame
        """
        table_data = []

        for hypothesis_name, result in hypothesis_results.items():
            row = {
                'Hypothesis': hypothesis_name.replace('_', ' '),
                'Description': result.description,
                'Expected_Direction': result.expected_direction.title(),
                'Test_Statistic': f"{result.test_statistic:.3f}",
                'p_value': f"{result.p_value:.3f}" if result.p_value >= 0.001 else "< 0.001",
                'Supported': "Yes" if result.supported else "No",
                'Evidence_Strength': result.evidence_strength.title(),
                'Target_Effects': ', '.join(result.target_effects)
            }
            table_data.append(row)

        hypothesis_df = pd.DataFrame(table_data)

        # Save if path provided
        if output_path:
            hypothesis_df.to_csv(output_path, index=False)
            logger.info(f"Hypothesis summary table saved to {output_path}")

        return hypothesis_df

    def create_parameter_visualization(
        self,
        interpretations: Dict[str, ParameterInterpretation],
        output_dir: Union[str, Path]
    ):
        """Create visualization of parameter estimates."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Extract data for plotting
        effects = list(interpretations.keys())
        estimates = [interp.parameter_estimate for interp in interpretations.values()]
        ci_lower = [interp.confidence_interval[0] for interp in interpretations.values()]
        ci_upper = [interp.confidence_interval[1] for interp in interpretations.values()]
        significant = [interp.practical_significance for interp in interpretations.values()]

        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))

        # Color by significance
        colors = ['red' if sig else 'blue' for sig in significant]

        # Create error bars
        y_positions = range(len(effects))
        ax.errorbar(estimates, y_positions,
                   xerr=[np.array(estimates) - np.array(ci_lower),
                         np.array(ci_upper) - np.array(estimates)],
                   fmt='o', capsize=5, color='black', alpha=0.7)

        # Color points by significance
        ax.scatter(estimates, y_positions, c=colors, s=100, alpha=0.8, zorder=3)

        # Add vertical line at zero
        ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)

        # Formatting
        ax.set_yticks(y_positions)
        ax.set_yticklabels([eff.replace('_', ' ') for eff in effects])
        ax.set_xlabel('Parameter Estimate')
        ax.set_title('SAOM Parameter Estimates with 95% Confidence Intervals')
        ax.grid(True, alpha=0.3)

        # Legend
        red_patch = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red',
                              markersize=10, label='Significant')
        blue_patch = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue',
                               markersize=10, label='Non-significant')
        ax.legend(handles=[red_patch, blue_patch])

        plt.tight_layout()
        plt.savefig(output_dir / "parameter_estimates.png", dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Parameter visualization saved to {output_dir}")

    def export_complete_analysis(
        self,
        interpretations: Dict[str, ParameterInterpretation],
        hypothesis_results: Dict[str, HypothesisTest],
        output_dir: Union[str, Path]
    ):
        """Export complete parameter analysis results."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create summary tables
        param_table = self.create_parameter_summary_table(
            interpretations, output_dir / "parameter_summary.csv"
        )
        hypothesis_table = self.create_hypothesis_summary_table(
            hypothesis_results, output_dir / "hypothesis_summary.csv"
        )

        # Create visualizations
        self.create_parameter_visualization(interpretations, output_dir)

        # Export detailed results
        import json
        detailed_results = {
            'parameter_interpretations': {
                name: {
                    'estimate': interp.parameter_estimate,
                    'se': interp.standard_error,
                    'p_value': interp.p_value,
                    'effect_size_category': interp.effect_size_category,
                    'practical_significance': interp.practical_significance,
                    'interpretation': interp.theoretical_interpretation
                }
                for name, interp in interpretations.items()
            },
            'hypothesis_tests': {
                name: {
                    'description': result.description,
                    'p_value': result.p_value,
                    'supported': result.supported,
                    'evidence_strength': result.evidence_strength
                }
                for name, result in hypothesis_results.items()
            }
        }

        with open(output_dir / "detailed_analysis.json", 'w') as f:
            json.dump(detailed_results, f, indent=2, default=str)

        logger.info(f"Complete parameter analysis exported to {output_dir}")


if __name__ == "__main__":
    # Test parameter analyzer
    logging.basicConfig(level=logging.INFO)

    try:
        # Create test analyzer
        analyzer = ParameterAnalyzer()

        # Test with dummy results
        from ..models.tolerance_cooperation_saom import SAOMResults, SAOMModelConfig

        # Create dummy SAOM results for testing
        dummy_results = SAOMResults(
            parameters=np.array([0.2, 0.5, 0.3, 0.4, -0.2, 0.25]),
            standard_errors=np.array([0.1, 0.15, 0.12, 0.18, 0.11, 0.13]),
            t_statistics=np.array([2.0, 3.3, 2.5, 2.2, -1.8, 1.9]),
            effect_names=['density', 'recip', 'transTrip', 'sameX.ethnicity',
                         'egoX.prejudice', 'avSim.tolerance'],
            converged=True,
            max_convergence_ratio=0.15,
            iterations=3000,
            log_likelihood=-1250.5
        )

        # Test parameter analysis
        interpretations = analyzer.analyze_parameters(dummy_results)
        hypothesis_results = analyzer.test_research_hypotheses(interpretations)

        print("✓ Parameter analysis test completed successfully")
        print(f"  - Analyzed {len(interpretations)} parameters")
        print(f"  - Tested {len(hypothesis_results)} hypotheses")
        print(f"  - {sum(1 for h in hypothesis_results.values() if h.supported)} hypotheses supported")

    except Exception as e:
        logger.error(f"Parameter analysis test failed: {e}")
        import traceback
        traceback.print_exc()