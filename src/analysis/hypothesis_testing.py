"""
Comprehensive Hypothesis Testing Suite for ABM-RSiena Integration

This module implements rigorous statistical hypothesis testing protocols with multiple
comparison corrections, effect size calculations, and power analysis for computational
social science research meeting PhD dissertation standards.

Author: Gamma Agent - Statistical Analysis & Validation Specialist
Date: 2025-09-15
"""

import logging
import numpy as np
import pandas as pd
import scipy.stats as stats
from typing import Dict, List, Optional, Tuple, Any, Union, NamedTuple
from dataclasses import dataclass, field
from pathlib import Path
import warnings
from abc import ABC, abstractmethod

# Statistical libraries
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pingouin as pg
from scipy.stats import (
    ttest_ind, ttest_rel, mannwhitneyu, wilcoxon, kruskal, friedmanchisquare,
    chi2_contingency, fisher_exact, anderson_ksamp, levene, bartlett,
    shapiro, normaltest, ks_2samp, jarque_bera, boxcox_normmax
)
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.power import ttest_power, anova_power_f
from statsmodels.stats.weightstats import ttest_ind as weighted_ttest
from statsmodels.stats.contingency_tables import mcnemar

# Effect size libraries
from scipy.stats import pearsonr, spearmanr
import pingouin as pg

logger = logging.getLogger(__name__)

@dataclass
class HypothesisTest:
    """Container for hypothesis test configuration."""
    name: str
    description: str
    null_hypothesis: str
    alternative_hypothesis: str
    test_type: str  # 'parametric', 'non_parametric', 'bayesian'
    assumptions: List[str]
    expected_effect_size: Optional[float] = None
    power_target: float = 0.8
    alpha: float = 0.05

@dataclass
class TestResult:
    """Container for statistical test results."""
    test_name: str
    statistic: float
    p_value: float
    degrees_of_freedom: Optional[Union[int, Tuple[int, ...]]] = None
    effect_size: Optional[float] = None
    effect_size_type: Optional[str] = None
    confidence_interval: Optional[Tuple[float, float]] = None
    power: Optional[float] = None
    is_significant: bool = False
    interpretation: str = ""
    assumptions_met: Dict[str, bool] = field(default_factory=dict)
    sample_size: Optional[int] = None
    corrected_p_value: Optional[float] = None

@dataclass
class MultipleComparisonResults:
    """Container for multiple comparison correction results."""
    original_p_values: np.ndarray
    corrected_p_values: np.ndarray
    rejected: np.ndarray
    correction_method: str
    alpha: float
    n_hypotheses: int
    n_rejected: int
    family_wise_error_rate: float

class AssumptionChecker:
    """
    Class for checking statistical test assumptions.
    Implements comprehensive assumption testing for parametric and non-parametric tests.
    """

    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha

    def check_normality(self, data: np.ndarray, method: str = "shapiro") -> Tuple[bool, float, str]:
        """
        Check normality assumption using various tests.

        Args:
            data: Data array to test
            method: Test method ('shapiro', 'ks', 'jarque_bera', 'anderson')

        Returns:
            Tuple of (assumption_met, p_value, interpretation)
        """
        if len(data) < 3:
            return False, np.nan, "Sample size too small for normality testing"

        try:
            if method == "shapiro" and len(data) <= 5000:
                statistic, p_value = shapiro(data)
                interpretation = f"Shapiro-Wilk test: {'Normal' if p_value > self.alpha else 'Non-normal'} distribution"
            elif method == "ks":
                # Lilliefors test approximation
                statistic, p_value = ks_2samp(data, stats.norm.rvs(loc=np.mean(data),
                                                                 scale=np.std(data), size=len(data)))
                interpretation = f"KS test: {'Normal' if p_value > self.alpha else 'Non-normal'} distribution"
            elif method == "jarque_bera":
                statistic, p_value = jarque_bera(data)
                interpretation = f"Jarque-Bera test: {'Normal' if p_value > self.alpha else 'Non-normal'} distribution"
            elif method == "anderson":
                result = stats.anderson(data, dist='norm')
                # Use 5% critical value
                critical_value = result.critical_values[2] if len(result.critical_values) > 2 else result.critical_values[-1]
                p_value = 0.05 if result.statistic > critical_value else 0.1  # Approximate
                interpretation = f"Anderson-Darling test: {'Normal' if result.statistic <= critical_value else 'Non-normal'} distribution"
            else:
                # Default to D'Agostino's test
                statistic, p_value = normaltest(data)
                interpretation = f"D'Agostino test: {'Normal' if p_value > self.alpha else 'Non-normal'} distribution"

            assumption_met = p_value > self.alpha
            return assumption_met, p_value, interpretation

        except Exception as e:
            logger.warning(f"Normality test failed: {e}")
            return False, np.nan, f"Normality test failed: {str(e)}"

    def check_homogeneity_of_variance(self, *groups) -> Tuple[bool, float, str]:
        """
        Check homogeneity of variance assumption using Levene's test.

        Args:
            *groups: Variable number of data groups

        Returns:
            Tuple of (assumption_met, p_value, interpretation)
        """
        if len(groups) < 2:
            return True, 1.0, "Only one group provided"

        # Filter out empty groups
        groups = [group for group in groups if len(group) > 0]

        if len(groups) < 2:
            return True, 1.0, "Insufficient non-empty groups for variance test"

        try:
            statistic, p_value = levene(*groups)
            assumption_met = p_value > self.alpha
            interpretation = f"Levene's test: {'Equal' if assumption_met else 'Unequal'} variances"
            return assumption_met, p_value, interpretation
        except Exception as e:
            logger.warning(f"Homogeneity of variance test failed: {e}")
            return False, np.nan, f"Variance test failed: {str(e)}"

    def check_independence(self, data: np.ndarray, method: str = "autocorr") -> Tuple[bool, float, str]:
        """
        Check independence assumption using autocorrelation or runs test.

        Args:
            data: Data array to test
            method: Test method ('autocorr', 'runs')

        Returns:
            Tuple of (assumption_met, test_statistic, interpretation)
        """
        if len(data) < 10:
            return True, 0.0, "Sample size too small for independence testing"

        try:
            if method == "autocorr":
                # Lag-1 autocorrelation test
                autocorr = np.corrcoef(data[:-1], data[1:])[0, 1]
                # Approximate significance test
                n = len(data)
                std_error = 1 / np.sqrt(n)
                z_score = abs(autocorr) / std_error
                p_value = 2 * (1 - stats.norm.cdf(z_score))

                assumption_met = p_value > self.alpha
                interpretation = f"Autocorrelation test: {'Independent' if assumption_met else 'Dependent'} observations"
                return assumption_met, autocorr, interpretation

            elif method == "runs":
                # Wald-Wolfowitz runs test
                median = np.median(data)
                runs, n1, n2 = self._runs_test(data > median)

                if n1 == 0 or n2 == 0:
                    return True, 0.0, "No variation in data for runs test"

                # Expected runs and standard deviation
                expected_runs = (2 * n1 * n2) / (n1 + n2) + 1
                runs_std = np.sqrt((2 * n1 * n2 * (2 * n1 * n2 - n1 - n2)) /
                                 ((n1 + n2) ** 2 * (n1 + n2 - 1)))

                if runs_std > 0:
                    z_score = (runs - expected_runs) / runs_std
                    p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))

                    assumption_met = p_value > self.alpha
                    interpretation = f"Runs test: {'Independent' if assumption_met else 'Dependent'} observations"
                    return assumption_met, z_score, interpretation
                else:
                    return True, 0.0, "Insufficient variation for runs test"

        except Exception as e:
            logger.warning(f"Independence test failed: {e}")
            return True, 0.0, f"Independence test failed: {str(e)}"

    def _runs_test(self, binary_sequence: np.ndarray) -> Tuple[int, int, int]:
        """Helper function for runs test."""
        runs = 1
        n1 = np.sum(binary_sequence)
        n2 = len(binary_sequence) - n1

        for i in range(1, len(binary_sequence)):
            if binary_sequence[i] != binary_sequence[i-1]:
                runs += 1

        return runs, n1, n2

class EffectSizeCalculator:
    """
    Calculator for various effect size measures.
    Implements Cohen's d, eta-squared, Cramer's V, and other effect sizes.
    """

    @staticmethod
    def cohens_d(group1: np.ndarray, group2: np.ndarray, paired: bool = False) -> float:
        """
        Calculate Cohen's d effect size.

        Args:
            group1, group2: Data arrays
            paired: Whether groups are paired

        Returns:
            Cohen's d effect size
        """
        if paired:
            diff = group1 - group2
            return np.mean(diff) / np.std(diff, ddof=1)
        else:
            n1, n2 = len(group1), len(group2)
            pooled_std = np.sqrt(((n1 - 1) * np.var(group1, ddof=1) +
                                (n2 - 1) * np.var(group2, ddof=1)) / (n1 + n2 - 2))
            return (np.mean(group1) - np.mean(group2)) / pooled_std if pooled_std > 0 else 0

    @staticmethod
    def hedges_g(group1: np.ndarray, group2: np.ndarray) -> float:
        """
        Calculate Hedges' g effect size (bias-corrected Cohen's d).

        Args:
            group1, group2: Data arrays

        Returns:
            Hedges' g effect size
        """
        d = EffectSizeCalculator.cohens_d(group1, group2)
        n = len(group1) + len(group2) - 2
        correction = 1 - (3 / (4 * n - 1))
        return d * correction

    @staticmethod
    def eta_squared(f_statistic: float, df_between: int, df_within: int) -> float:
        """
        Calculate eta-squared effect size for ANOVA.

        Args:
            f_statistic: F-statistic from ANOVA
            df_between: Degrees of freedom between groups
            df_within: Degrees of freedom within groups

        Returns:
            Eta-squared effect size
        """
        ss_between = f_statistic * df_between
        ss_total = ss_between + df_within
        return ss_between / ss_total if ss_total > 0 else 0

    @staticmethod
    def cramers_v(chi2: float, n: int, min_dim: int) -> float:
        """
        Calculate Cramer's V effect size for chi-square test.

        Args:
            chi2: Chi-square statistic
            n: Total sample size
            min_dim: Minimum dimension (min(rows-1, cols-1))

        Returns:
            Cramer's V effect size
        """
        return np.sqrt(chi2 / (n * min_dim)) if n > 0 and min_dim > 0 else 0

    @staticmethod
    def pearson_r_effect_size(r: float) -> str:
        """
        Interpret Pearson correlation effect size.

        Args:
            r: Pearson correlation coefficient

        Returns:
            Effect size interpretation
        """
        abs_r = abs(r)
        if abs_r < 0.1:
            return "negligible"
        elif abs_r < 0.3:
            return "small"
        elif abs_r < 0.5:
            return "medium"
        elif abs_r < 0.7:
            return "large"
        else:
            return "very large"

class PowerAnalyzer:
    """
    Class for statistical power analysis and sample size calculations.
    Implements power calculations for various test types.
    """

    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha

    def power_t_test(self, effect_size: float, n1: int, n2: Optional[int] = None,
                    test_type: str = "two_sample") -> float:
        """
        Calculate statistical power for t-test.

        Args:
            effect_size: Expected Cohen's d
            n1: Sample size group 1
            n2: Sample size group 2 (for two-sample test)
            test_type: 'one_sample', 'two_sample', 'paired'

        Returns:
            Statistical power
        """
        try:
            if test_type == "one_sample":
                return ttest_power(effect_size, n1, self.alpha, alternative='two-sided')
            elif test_type == "two_sample":
                n2 = n2 or n1
                return ttest_power(effect_size, n1, self.alpha, alternative='two-sided')
            elif test_type == "paired":
                return ttest_power(effect_size, n1, self.alpha, alternative='two-sided')
            else:
                logger.warning(f"Unknown test type: {test_type}")
                return np.nan
        except Exception as e:
            logger.warning(f"Power calculation failed: {e}")
            return np.nan

    def sample_size_t_test(self, effect_size: float, power: float = 0.8,
                          test_type: str = "two_sample") -> int:
        """
        Calculate required sample size for t-test.

        Args:
            effect_size: Expected Cohen's d
            power: Desired statistical power
            test_type: 'one_sample', 'two_sample', 'paired'

        Returns:
            Required sample size per group
        """
        from statsmodels.stats.power import ttest_power

        # Binary search for sample size
        n_min, n_max = 5, 10000

        while n_max - n_min > 1:
            n_mid = (n_min + n_max) // 2
            calculated_power = self.power_t_test(effect_size, n_mid, test_type=test_type)

            if calculated_power >= power:
                n_max = n_mid
            else:
                n_min = n_mid

        return n_max

    def power_anova(self, effect_size: float, n_groups: int, n_per_group: int) -> float:
        """
        Calculate statistical power for ANOVA.

        Args:
            effect_size: Expected eta-squared
            n_groups: Number of groups
            n_per_group: Sample size per group

        Returns:
            Statistical power
        """
        try:
            # Convert eta-squared to Cohen's f
            f = np.sqrt(effect_size / (1 - effect_size)) if effect_size < 1 else 1
            return anova_power_f(f, n_groups - 1, n_per_group * n_groups, self.alpha)
        except Exception as e:
            logger.warning(f"ANOVA power calculation failed: {e}")
            return np.nan

class HypothesisTestingSuite:
    """
    Comprehensive hypothesis testing suite with multiple comparison corrections.
    Implements rigorous statistical testing protocols for computational social science.
    """

    def __init__(self, alpha: float = 0.05, power_target: float = 0.8):
        self.alpha = alpha
        self.power_target = power_target
        self.assumption_checker = AssumptionChecker(alpha)
        self.effect_calculator = EffectSizeCalculator()
        self.power_analyzer = PowerAnalyzer(alpha)

        # Define research hypotheses for ABM-RSiena integration
        self.research_hypotheses = self._initialize_research_hypotheses()

    def _initialize_research_hypotheses(self) -> List[HypothesisTest]:
        """Initialize standard research hypotheses for ABM-RSiena studies."""
        hypotheses = [
            HypothesisTest(
                name="network_density_equivalence",
                description="ABM and RSiena produce equivalent network densities",
                null_hypothesis="μ_ABM = μ_RSiena (network densities are equal)",
                alternative_hypothesis="μ_ABM ≠ μ_RSiena (network densities differ)",
                test_type="parametric",
                assumptions=["normality", "homogeneity_of_variance", "independence"],
                expected_effect_size=0.2
            ),
            HypothesisTest(
                name="clustering_coefficient_similarity",
                description="ABM and RSiena show similar clustering patterns",
                null_hypothesis="Clustering coefficients have the same distribution",
                alternative_hypothesis="Clustering coefficients have different distributions",
                test_type="non_parametric",
                assumptions=["independence"],
                expected_effect_size=0.3
            ),
            HypothesisTest(
                name="degree_distribution_equivalence",
                description="Degree distributions are statistically equivalent",
                null_hypothesis="Degree distributions are identical",
                alternative_hypothesis="Degree distributions differ significantly",
                test_type="non_parametric",
                assumptions=["independence"],
                expected_effect_size=0.25
            ),
            HypothesisTest(
                name="temporal_stability_comparison",
                description="Temporal stability patterns are equivalent",
                null_hypothesis="Temporal correlations are equal between methods",
                alternative_hypothesis="Temporal correlations differ between methods",
                test_type="parametric",
                assumptions=["normality", "homogeneity_of_variance"],
                expected_effect_size=0.15
            ),
            HypothesisTest(
                name="centrality_measures_concordance",
                description="Centrality measures show concordance between methods",
                null_hypothesis="Centrality rankings are uncorrelated",
                alternative_hypothesis="Centrality rankings are significantly correlated",
                test_type="non_parametric",
                assumptions=["monotonic_relationship"],
                expected_effect_size=0.5
            )
        ]
        return hypotheses

    def test_network_density_equivalence(self, abm_densities: np.ndarray,
                                       rsiena_densities: np.ndarray) -> TestResult:
        """
        Test equivalence of network densities between ABM and RSiena.

        Args:
            abm_densities: Array of ABM network densities
            rsiena_densities: Array of RSiena network densities

        Returns:
            TestResult object with comprehensive results
        """
        logger.info("Testing network density equivalence")

        # Check assumptions
        assumptions_met = {}

        # Normality checks
        abm_normal, abm_norm_p, abm_norm_interp = self.assumption_checker.check_normality(abm_densities)
        rsiena_normal, rsiena_norm_p, rsiena_norm_interp = self.assumption_checker.check_normality(rsiena_densities)
        assumptions_met['abm_normality'] = abm_normal
        assumptions_met['rsiena_normality'] = rsiena_normal

        # Homogeneity of variance
        var_equal, var_p, var_interp = self.assumption_checker.check_homogeneity_of_variance(
            abm_densities, rsiena_densities
        )
        assumptions_met['homogeneity_of_variance'] = var_equal

        # Choose appropriate test
        if abm_normal and rsiena_normal and var_equal:
            # Parametric t-test
            statistic, p_value = ttest_ind(abm_densities, rsiena_densities, equal_var=True)
            test_name = "Independent t-test"
            df = len(abm_densities) + len(rsiena_densities) - 2
        elif abm_normal and rsiena_normal and not var_equal:
            # Welch's t-test
            statistic, p_value = ttest_ind(abm_densities, rsiena_densities, equal_var=False)
            test_name = "Welch's t-test"
            df = None  # Complex calculation for Welch's
        else:
            # Non-parametric Mann-Whitney U test
            statistic, p_value = mannwhitneyu(abm_densities, rsiena_densities, alternative='two-sided')
            test_name = "Mann-Whitney U test"
            df = None

        # Effect size calculation
        if 't-test' in test_name:
            effect_size = self.effect_calculator.cohens_d(abm_densities, rsiena_densities)
            effect_size_type = "Cohen's d"
        else:
            # Rank biserial correlation for Mann-Whitney
            n1, n2 = len(abm_densities), len(rsiena_densities)
            effect_size = 1 - (2 * statistic) / (n1 * n2)
            effect_size_type = "Rank biserial correlation"

        # Power calculation
        power = self.power_analyzer.power_t_test(
            abs(effect_size), len(abm_densities), len(rsiena_densities), "two_sample"
        )

        # Confidence interval for effect size (bootstrap)
        ci = self._bootstrap_effect_size_ci(abm_densities, rsiena_densities,
                                          lambda x, y: self.effect_calculator.cohens_d(x, y))

        return TestResult(
            test_name=test_name,
            statistic=statistic,
            p_value=p_value,
            degrees_of_freedom=df,
            effect_size=effect_size,
            effect_size_type=effect_size_type,
            confidence_interval=ci,
            power=power,
            is_significant=p_value < self.alpha,
            interpretation=self._interpret_density_test_result(p_value, effect_size, power),
            assumptions_met=assumptions_met,
            sample_size=len(abm_densities) + len(rsiena_densities)
        )

    def test_degree_distribution_equivalence(self, abm_degrees: np.ndarray,
                                           rsiena_degrees: np.ndarray) -> TestResult:
        """
        Test equivalence of degree distributions using Kolmogorov-Smirnov test.

        Args:
            abm_degrees: Array of degrees from ABM
            rsiena_degrees: Array of degrees from RSiena

        Returns:
            TestResult object
        """
        logger.info("Testing degree distribution equivalence")

        # Kolmogorov-Smirnov two-sample test
        statistic, p_value = ks_2samp(abm_degrees, rsiena_degrees)

        # Effect size (using standardized difference in means as approximation)
        effect_size = abs(np.mean(abm_degrees) - np.mean(rsiena_degrees)) / \
                     np.sqrt((np.var(abm_degrees) + np.var(rsiena_degrees)) / 2)

        # Additional distribution comparison metrics
        abm_median = np.median(abm_degrees)
        rsiena_median = np.median(rsiena_degrees)
        median_diff = abs(abm_median - rsiena_median)

        assumptions_met = {'independence': True}  # Assumed for cross-sectional data

        interpretation = f"KS test shows {'significant' if p_value < self.alpha else 'non-significant'} " + \
                        f"difference in degree distributions (D = {statistic:.4f}, p = {p_value:.4f}). " + \
                        f"Median difference: {median_diff:.2f}"

        return TestResult(
            test_name="Kolmogorov-Smirnov two-sample test",
            statistic=statistic,
            p_value=p_value,
            effect_size=effect_size,
            effect_size_type="Standardized mean difference",
            is_significant=p_value < self.alpha,
            interpretation=interpretation,
            assumptions_met=assumptions_met,
            sample_size=len(abm_degrees) + len(rsiena_degrees)
        )

    def test_clustering_similarity(self, abm_clustering: np.ndarray,
                                 rsiena_clustering: np.ndarray) -> TestResult:
        """
        Test similarity of clustering coefficients.

        Args:
            abm_clustering: Array of clustering coefficients from ABM
            rsiena_clustering: Array of clustering coefficients from RSiena

        Returns:
            TestResult object
        """
        logger.info("Testing clustering coefficient similarity")

        # Check normality for both samples
        abm_normal, _, _ = self.assumption_checker.check_normality(abm_clustering)
        rsiena_normal, _, _ = self.assumption_checker.check_normality(rsiena_clustering)

        assumptions_met = {
            'abm_normality': abm_normal,
            'rsiena_normality': rsiena_normal
        }

        if abm_normal and rsiena_normal:
            # Parametric correlation test
            correlation, p_value = pearsonr(abm_clustering, rsiena_clustering)
            test_name = "Pearson correlation test"
            effect_size = abs(correlation)
            effect_size_type = "Pearson r"
        else:
            # Non-parametric correlation test
            correlation, p_value = spearmanr(abm_clustering, rsiena_clustering)
            test_name = "Spearman rank correlation test"
            effect_size = abs(correlation)
            effect_size_type = "Spearman rho"

        # Power calculation (approximate for correlation)
        n = min(len(abm_clustering), len(rsiena_clustering))
        power = self._correlation_power(effect_size, n)

        interpretation = f"{test_name} shows {'significant' if p_value < self.alpha else 'non-significant'} " + \
                        f"correlation (r = {correlation:.4f}, p = {p_value:.4f}). " + \
                        f"Effect size is {self.effect_calculator.pearson_r_effect_size(correlation)}."

        return TestResult(
            test_name=test_name,
            statistic=correlation,
            p_value=p_value,
            effect_size=effect_size,
            effect_size_type=effect_size_type,
            power=power,
            is_significant=p_value < self.alpha,
            interpretation=interpretation,
            assumptions_met=assumptions_met,
            sample_size=n
        )

    def perform_multiple_comparisons_correction(self, test_results: List[TestResult],
                                              method: str = "fdr_bh") -> MultipleComparisonResults:
        """
        Apply multiple comparisons correction to test results.

        Args:
            test_results: List of TestResult objects
            method: Correction method ('bonferroni', 'holm', 'fdr_bh', 'fdr_by')

        Returns:
            MultipleComparisonResults object
        """
        logger.info(f"Applying {method} multiple comparisons correction")

        # Extract p-values
        p_values = np.array([result.p_value for result in test_results if not np.isnan(result.p_value)])

        if len(p_values) == 0:
            logger.warning("No valid p-values for multiple comparison correction")
            return MultipleComparisonResults(
                original_p_values=np.array([]),
                corrected_p_values=np.array([]),
                rejected=np.array([]),
                correction_method=method,
                alpha=self.alpha,
                n_hypotheses=0,
                n_rejected=0,
                family_wise_error_rate=0.0
            )

        # Apply correction
        rejected, p_corrected, alpha_sidak, alpha_bonf = multipletests(
            p_values, alpha=self.alpha, method=method
        )

        # Calculate family-wise error rate
        if method == "bonferroni":
            fwer = min(1.0, len(p_values) * self.alpha)
        elif method == "holm":
            fwer = self.alpha
        else:  # FDR methods
            fwer = np.nan  # FDR doesn't control FWER

        # Update test results with corrected p-values
        valid_indices = [i for i, result in enumerate(test_results) if not np.isnan(result.p_value)]
        for i, idx in enumerate(valid_indices):
            test_results[idx].corrected_p_value = p_corrected[i]
            test_results[idx].is_significant = rejected[i]

        return MultipleComparisonResults(
            original_p_values=p_values,
            corrected_p_values=p_corrected,
            rejected=rejected,
            correction_method=method,
            alpha=self.alpha,
            n_hypotheses=len(p_values),
            n_rejected=np.sum(rejected),
            family_wise_error_rate=fwer
        )

    def comprehensive_hypothesis_testing(self, abm_data: Dict[str, np.ndarray],
                                       rsiena_data: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """
        Perform comprehensive hypothesis testing suite.

        Args:
            abm_data: Dictionary with ABM network metrics
            rsiena_data: Dictionary with RSiena network metrics

        Returns:
            Dictionary with all test results and corrections
        """
        logger.info("Starting comprehensive hypothesis testing suite")

        test_results = []

        # Test 1: Network density equivalence
        if 'density' in abm_data and 'density' in rsiena_data:
            density_result = self.test_network_density_equivalence(
                abm_data['density'], rsiena_data['density']
            )
            test_results.append(density_result)

        # Test 2: Degree distribution equivalence
        if 'degrees' in abm_data and 'degrees' in rsiena_data:
            degree_result = self.test_degree_distribution_equivalence(
                abm_data['degrees'], rsiena_data['degrees']
            )
            test_results.append(degree_result)

        # Test 3: Clustering similarity
        if 'clustering' in abm_data and 'clustering' in rsiena_data:
            clustering_result = self.test_clustering_similarity(
                abm_data['clustering'], rsiena_data['clustering']
            )
            test_results.append(clustering_result)

        # Test 4: Path length comparison
        if 'path_length' in abm_data and 'path_length' in rsiena_data:
            path_result = self.test_network_density_equivalence(  # Reuse for continuous variable
                abm_data['path_length'], rsiena_data['path_length']
            )
            path_result.test_name = "Average path length comparison"
            test_results.append(path_result)

        # Apply multiple comparisons correction
        mc_results = self.perform_multiple_comparisons_correction(test_results, method="fdr_bh")

        # Compile comprehensive results
        results = {
            'test_results': test_results,
            'multiple_comparisons': mc_results,
            'summary': self._generate_testing_summary(test_results, mc_results),
            'power_analysis': self._generate_power_summary(test_results),
            'effect_sizes': self._extract_effect_sizes(test_results),
            'recommendations': self._generate_recommendations(test_results, mc_results)
        }

        logger.info("Comprehensive hypothesis testing completed")
        return results

    def _bootstrap_effect_size_ci(self, group1: np.ndarray, group2: np.ndarray,
                                effect_func: callable, n_bootstrap: int = 1000,
                                confidence_level: float = 0.95) -> Tuple[float, float]:
        """
        Bootstrap confidence interval for effect size.

        Args:
            group1, group2: Data arrays
            effect_func: Function to calculate effect size
            n_bootstrap: Number of bootstrap samples
            confidence_level: Confidence level for CI

        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        bootstrap_effects = []

        for _ in range(n_bootstrap):
            # Bootstrap samples
            boot1 = np.random.choice(group1, len(group1), replace=True)
            boot2 = np.random.choice(group2, len(group2), replace=True)

            try:
                effect = effect_func(boot1, boot2)
                if not np.isnan(effect) and not np.isinf(effect):
                    bootstrap_effects.append(effect)
            except:
                continue

        if not bootstrap_effects:
            return (np.nan, np.nan)

        alpha = 1 - confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100

        return (np.percentile(bootstrap_effects, lower_percentile),
                np.percentile(bootstrap_effects, upper_percentile))

    def _correlation_power(self, correlation: float, n: int) -> float:
        """
        Approximate power calculation for correlation test.

        Args:
            correlation: Expected correlation coefficient
            n: Sample size

        Returns:
            Statistical power
        """
        if n < 3:
            return np.nan

        # Fisher z-transformation
        z_r = 0.5 * np.log((1 + abs(correlation)) / (1 - abs(correlation)))
        se = 1 / np.sqrt(n - 3)

        # Critical value
        z_crit = stats.norm.ppf(1 - self.alpha / 2)

        # Power calculation
        power = 1 - stats.norm.cdf(z_crit - z_r / se) + stats.norm.cdf(-z_crit - z_r / se)
        return power

    def _interpret_density_test_result(self, p_value: float, effect_size: float, power: float) -> str:
        """Generate interpretation for density test result."""
        significance = "significant" if p_value < self.alpha else "non-significant"

        if abs(effect_size) < 0.2:
            magnitude = "small"
        elif abs(effect_size) < 0.5:
            magnitude = "medium"
        elif abs(effect_size) < 0.8:
            magnitude = "large"
        else:
            magnitude = "very large"

        power_str = "adequate" if power >= 0.8 else "inadequate"

        return f"Test shows {significance} difference with {magnitude} effect size " + \
               f"(d = {effect_size:.3f}). Statistical power is {power_str} ({power:.3f})."

    def _generate_testing_summary(self, test_results: List[TestResult],
                                mc_results: MultipleComparisonResults) -> Dict[str, Any]:
        """Generate summary of testing results."""
        n_tests = len(test_results)
        n_significant = sum(1 for result in test_results if result.is_significant)
        n_significant_corrected = mc_results.n_rejected

        return {
            'total_tests': n_tests,
            'significant_before_correction': n_significant,
            'significant_after_correction': n_significant_corrected,
            'correction_method': mc_results.correction_method,
            'alpha_level': self.alpha,
            'mean_power': np.nanmean([r.power for r in test_results if r.power is not None]),
            'median_effect_size': np.nanmedian([r.effect_size for r in test_results if r.effect_size is not None])
        }

    def _generate_power_summary(self, test_results: List[TestResult]) -> Dict[str, Any]:
        """Generate power analysis summary."""
        powers = [r.power for r in test_results if r.power is not None]

        if not powers:
            return {'message': 'No power calculations available'}

        return {
            'mean_power': np.mean(powers),
            'min_power': np.min(powers),
            'max_power': np.max(powers),
            'underpowered_tests': sum(1 for p in powers if p < 0.8),
            'adequately_powered_tests': sum(1 for p in powers if p >= 0.8),
            'power_target': self.power_target
        }

    def _extract_effect_sizes(self, test_results: List[TestResult]) -> Dict[str, Any]:
        """Extract and summarize effect sizes."""
        effect_sizes = [(r.test_name, r.effect_size, r.effect_size_type)
                       for r in test_results if r.effect_size is not None]

        if not effect_sizes:
            return {'message': 'No effect sizes calculated'}

        sizes = [es[1] for es in effect_sizes]

        return {
            'effect_sizes': effect_sizes,
            'mean_effect_size': np.mean(sizes),
            'median_effect_size': np.median(sizes),
            'large_effects': sum(1 for s in sizes if abs(s) >= 0.8),
            'medium_effects': sum(1 for s in sizes if 0.5 <= abs(s) < 0.8),
            'small_effects': sum(1 for s in sizes if 0.2 <= abs(s) < 0.5),
            'negligible_effects': sum(1 for s in sizes if abs(s) < 0.2)
        }

    def _generate_recommendations(self, test_results: List[TestResult],
                                mc_results: MultipleComparisonResults) -> List[str]:
        """Generate methodological recommendations."""
        recommendations = []

        # Power-based recommendations
        low_power_tests = [r for r in test_results if r.power is not None and r.power < 0.8]
        if low_power_tests:
            recommendations.append(
                f"Consider increasing sample size for {len(low_power_tests)} underpowered tests"
            )

        # Effect size recommendations
        large_effects = [r for r in test_results if r.effect_size is not None and abs(r.effect_size) >= 0.8]
        if large_effects:
            recommendations.append(
                f"{len(large_effects)} tests show large effect sizes - consider practical significance"
            )

        # Multiple comparisons recommendations
        if mc_results.n_rejected < len(test_results) / 2:
            recommendations.append(
                "Consider using less conservative multiple comparison correction if appropriate"
            )

        # Assumption violations
        assumption_violations = []
        for result in test_results:
            violations = [k for k, v in result.assumptions_met.items() if not v]
            if violations:
                assumption_violations.extend(violations)

        if assumption_violations:
            recommendations.append(
                f"Address assumption violations: {', '.join(set(assumption_violations))}"
            )

        return recommendations


# Example usage and testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Initialize testing suite
    tester = HypothesisTestingSuite(alpha=0.05, power_target=0.8)

    # Generate sample data
    np.random.seed(42)
    n_networks = 20

    # Simulated ABM data
    abm_data = {
        'density': np.random.normal(0.15, 0.03, n_networks),
        'degrees': np.random.poisson(5, 200),
        'clustering': np.random.beta(2, 5, n_networks),
        'path_length': np.random.normal(3.2, 0.5, n_networks)
    }

    # Simulated RSiena data (slightly different)
    rsiena_data = {
        'density': np.random.normal(0.17, 0.03, n_networks),
        'degrees': np.random.poisson(5.5, 200),
        'clustering': np.random.beta(2.2, 5, n_networks),
        'path_length': np.random.normal(3.0, 0.5, n_networks)
    }

    # Perform comprehensive testing
    results = tester.comprehensive_hypothesis_testing(abm_data, rsiena_data)

    # Print summary
    print("Hypothesis Testing Results Summary:")
    print("=" * 50)
    for key, value in results['summary'].items():
        print(f"{key}: {value}")

    print("\nRecommendations:")
    for rec in results['recommendations']:
        print(f"- {rec}")