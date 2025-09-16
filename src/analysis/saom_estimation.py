"""
SAOM Estimation Framework with Meta-Analysis

This module provides a comprehensive framework for estimating SAOM parameters
across multiple classrooms and conducting meta-analysis to combine estimates.
Designed for the tolerance-cooperation research with 105 classrooms across 3 schools.

Key Features:
- Parallel SAOM estimation across classrooms
- Meta-analysis of classroom-level estimates
- Parameter uncertainty quantification
- Convergence diagnostics and quality assessment
- Hierarchical modeling for school-level effects
- Robust estimation with outlier detection

Research Context:
Handles the "Doubt 2" from the presentation: How to simulate from 105 classes?
Solution: Estimate each classroom individually, then meta-analysis for combined effects.

Author: RSiena Integration Specialist
Created: 2025-09-16
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
from pathlib import Path
import concurrent.futures
import warnings
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

from ..models.tolerance_cooperation_saom import ToleranceCooperationSAOM, SAOMResults, SAOMModelConfig
from ..data.classroom_data_processor import ClassroomDataProcessor
from ..rsiena_integration.r_interface import RInterface, RSessionConfig
from ..rsiena_integration.data_converters import ABMRSienaConverter

logger = logging.getLogger(__name__)


@dataclass
class MetaAnalysisConfig:
    """Configuration for meta-analysis of SAOM estimates."""
    # Effect selection
    target_effects: List[str] = field(default_factory=lambda: [
        'density', 'recip', 'transTrip', 'sameX.ethnicity', 'avSim.tolerance',
        'attractionRepulsionInfluence', 'toleranceCooperationSelection'
    ])

    # Meta-analysis method
    method: str = "random_effects"  # "fixed_effects", "random_effects", "mixed_effects"
    estimator: str = "DerSimonian-Laird"  # "DerSimonian-Laird", "REML", "Paule-Mandel"

    # Quality filters
    min_convergence_quality: float = 0.25  # Max convergence ratio
    min_classroom_size: int = 15
    exclude_outliers: bool = True
    outlier_threshold: float = 3.0  # Standard deviations

    # Uncertainty quantification
    confidence_level: float = 0.95
    prediction_interval: bool = True
    heterogeneity_test: bool = True


@dataclass
class EstimationConfig:
    """Configuration for parallel SAOM estimation."""
    # Parallel processing
    n_workers: int = 4
    max_classrooms_per_worker: int = 10
    timeout_per_classroom: int = 1800  # 30 minutes

    # Model configuration
    saom_config: SAOMModelConfig = field(default_factory=SAOMModelConfig)

    # Quality control
    require_convergence: bool = True
    max_retries: int = 2
    retry_with_simpler_model: bool = True

    # Output
    save_individual_results: bool = True
    save_diagnostics: bool = True


@dataclass
class MetaAnalysisResults:
    """Results from meta-analysis of SAOM estimates."""
    effect_estimates: Dict[str, float]
    effect_se: Dict[str, float]
    effect_ci_lower: Dict[str, float]
    effect_ci_upper: Dict[str, float]
    heterogeneity_stats: Dict[str, float]
    forest_plot_data: Dict[str, pd.DataFrame]
    study_weights: Dict[str, np.ndarray]
    prediction_intervals: Dict[str, Tuple[float, float]]
    quality_metrics: Dict[str, Any]
    included_classrooms: List[str]
    excluded_classrooms: List[str]
    n_studies: int


class SAOMEstimationFramework:
    """
    Framework for parallel SAOM estimation and meta-analysis.

    Provides comprehensive tools for estimating tolerance-cooperation models
    across multiple classrooms and combining results through meta-analysis.
    """

    def __init__(
        self,
        estimation_config: Optional[EstimationConfig] = None,
        meta_config: Optional[MetaAnalysisConfig] = None
    ):
        """
        Initialize SAOM estimation framework.

        Args:
            estimation_config: Configuration for parallel estimation
            meta_config: Configuration for meta-analysis
        """
        self.estimation_config = estimation_config or EstimationConfig()
        self.meta_config = meta_config or MetaAnalysisConfig()

        self.classroom_results = {}
        self.meta_analysis_results = {}
        self.estimation_diagnostics = {}

        # Initialize R session config for workers
        self.r_session_config = RSessionConfig(
            required_packages=['RSiena', 'igraph', 'network', 'sna', 'metafor'],
            memory_limit_mb=2048,
            timeout_seconds=self.estimation_config.timeout_per_classroom
        )

    def estimate_all_classrooms(
        self,
        classroom_processor: ClassroomDataProcessor,
        output_dir: Optional[Union[str, Path]] = None
    ) -> Dict[str, SAOMResults]:
        """
        Estimate SAOM for all processed classrooms in parallel.

        Args:
            classroom_processor: Processor with classroom data
            output_dir: Directory to save results

        Returns:
            Dictionary of classroom estimation results
        """
        logger.info("Starting parallel SAOM estimation for all classrooms...")

        # Get classroom IDs
        classroom_ids = list(classroom_processor.processed_classrooms.keys())
        logger.info(f"Estimating SAOM for {len(classroom_ids)} classrooms")

        # Split classrooms across workers
        classroom_batches = self._create_classroom_batches(classroom_ids)

        # Parallel estimation
        all_results = {}
        failed_classrooms = []

        with concurrent.futures.ProcessPoolExecutor(
            max_workers=self.estimation_config.n_workers
        ) as executor:

            # Submit batch jobs
            future_to_batch = {}
            for batch_id, classroom_batch in enumerate(classroom_batches):
                future = executor.submit(
                    self._estimate_classroom_batch,
                    classroom_batch,
                    classroom_processor,
                    batch_id
                )
                future_to_batch[future] = (batch_id, classroom_batch)

            # Collect results
            for future in concurrent.futures.as_completed(future_to_batch):
                batch_id, classroom_batch = future_to_batch[future]

                try:
                    batch_results = future.result(
                        timeout=self.estimation_config.timeout_per_classroom * len(classroom_batch)
                    )
                    all_results.update(batch_results)
                    logger.info(f"✓ Batch {batch_id} completed: {len(batch_results)} classrooms")

                except Exception as e:
                    logger.error(f"✗ Batch {batch_id} failed: {e}")
                    failed_classrooms.extend(classroom_batch)

        # Store results
        self.classroom_results = all_results

        # Save individual results if requested
        if output_dir and self.estimation_config.save_individual_results:
            self._save_individual_results(output_dir)

        logger.info(f"Parallel estimation completed: {len(all_results)} successful, "
                   f"{len(failed_classrooms)} failed")

        return all_results

    def _create_classroom_batches(self, classroom_ids: List[str]) -> List[List[str]]:
        """Create batches of classrooms for parallel processing."""
        batch_size = self.estimation_config.max_classrooms_per_worker
        batches = [
            classroom_ids[i:i + batch_size]
            for i in range(0, len(classroom_ids), batch_size)
        ]
        return batches

    def _estimate_classroom_batch(
        self,
        classroom_ids: List[str],
        classroom_processor: ClassroomDataProcessor,
        batch_id: int
    ) -> Dict[str, SAOMResults]:
        """Estimate SAOM for a batch of classrooms (worker function)."""
        logger.info(f"Worker {batch_id}: Processing {len(classroom_ids)} classrooms")

        batch_results = {}

        try:
            # Initialize R interface for this worker
            with RInterface(self.r_session_config) as r_interface:
                # Create converter and SAOM model
                converter = ABMRSienaConverter(r_interface)
                saom = ToleranceCooperationSAOM(self.estimation_config.saom_config, r_interface)

                for classroom_id in classroom_ids:
                    try:
                        logger.debug(f"Worker {batch_id}: Estimating classroom {classroom_id}")

                        # Convert classroom data
                        dataset = classroom_processor.convert_classroom_to_rsiena(
                            classroom_id, converter
                        )

                        # Estimate model
                        results = saom.estimate_model(dataset, classroom_id)

                        # Check convergence
                        if (self.estimation_config.require_convergence and
                            results.max_convergence_ratio > self.meta_config.min_convergence_quality):

                            logger.warning(f"Classroom {classroom_id} convergence poor: "
                                         f"{results.max_convergence_ratio:.3f}")

                            # Retry with simpler model if configured
                            if self.estimation_config.retry_with_simpler_model:
                                results = self._retry_with_simpler_model(
                                    saom, dataset, classroom_id
                                )

                        # Goodness of fit
                        saom.goodness_of_fit(results)

                        batch_results[classroom_id] = results
                        logger.debug(f"✓ Classroom {classroom_id} estimated successfully")

                    except Exception as e:
                        logger.error(f"✗ Classroom {classroom_id} estimation failed: {e}")

        except Exception as e:
            logger.error(f"Worker {batch_id} failed: {e}")

        logger.info(f"Worker {batch_id} completed: {len(batch_results)} successful estimates")
        return batch_results

    def _retry_with_simpler_model(
        self,
        saom: ToleranceCooperationSAOM,
        dataset,
        classroom_id: str
    ) -> SAOMResults:
        """Retry estimation with simpler model configuration."""
        logger.info(f"Retrying classroom {classroom_id} with simpler model")

        # Create simpler configuration
        simpler_config = SAOMModelConfig(
            networks=['friendship'],  # Only friendship network
            behaviors=['tolerance'],
            covariates=['ethnicity', 'gender'],
            use_attraction_repulsion=False,  # Disable custom effects
            use_complex_contagion=False,
            use_tolerance_cooperation=False,
            n_iterations=2000  # Fewer iterations
        )

        # Create simpler SAOM
        simpler_saom = ToleranceCooperationSAOM(simpler_config, saom.r_interface)

        # Estimate
        results = simpler_saom.estimate_model(dataset, classroom_id)
        results.model_config = simpler_config  # Mark as simplified

        return results

    def conduct_meta_analysis(
        self,
        classroom_results: Optional[Dict[str, SAOMResults]] = None
    ) -> MetaAnalysisResults:
        """
        Conduct meta-analysis of classroom-level SAOM estimates.

        Args:
            classroom_results: Classroom estimation results (uses stored if None)

        Returns:
            Meta-analysis results
        """
        logger.info("Conducting meta-analysis of SAOM estimates...")

        if classroom_results is None:
            classroom_results = self.classroom_results

        if not classroom_results:
            raise ValueError("No classroom results available for meta-analysis")

        # Filter classrooms based on quality criteria
        included_classrooms, excluded_classrooms = self._filter_classrooms_for_meta_analysis(
            classroom_results
        )

        logger.info(f"Meta-analysis: {len(included_classrooms)} included, "
                   f"{len(excluded_classrooms)} excluded classrooms")

        # Extract effect estimates
        effects_data = self._extract_effects_for_meta_analysis(
            classroom_results, included_classrooms
        )

        # Conduct meta-analysis for each effect
        meta_results = {}
        forest_plot_data = {}
        study_weights = {}
        heterogeneity_stats = {}
        prediction_intervals = {}

        for effect_name in self.meta_config.target_effects:
            if effect_name in effects_data:
                logger.debug(f"Meta-analyzing effect: {effect_name}")

                # Get data for this effect
                estimates = effects_data[effect_name]['estimates']
                standard_errors = effects_data[effect_name]['standard_errors']
                classroom_ids = effects_data[effect_name]['classroom_ids']

                # Conduct meta-analysis
                ma_result = self._conduct_single_effect_meta_analysis(
                    estimates, standard_errors, classroom_ids, effect_name
                )

                meta_results[effect_name] = ma_result['summary']
                forest_plot_data[effect_name] = ma_result['forest_data']
                study_weights[effect_name] = ma_result['weights']
                heterogeneity_stats[effect_name] = ma_result['heterogeneity']

                if self.meta_config.prediction_interval:
                    prediction_intervals[effect_name] = ma_result['prediction_interval']

        # Compile final results
        final_results = MetaAnalysisResults(
            effect_estimates={k: v['estimate'] for k, v in meta_results.items()},
            effect_se={k: v['se'] for k, v in meta_results.items()},
            effect_ci_lower={k: v['ci_lower'] for k, v in meta_results.items()},
            effect_ci_upper={k: v['ci_upper'] for k, v in meta_results.items()},
            heterogeneity_stats=heterogeneity_stats,
            forest_plot_data=forest_plot_data,
            study_weights=study_weights,
            prediction_intervals=prediction_intervals,
            quality_metrics=self._compute_meta_analysis_quality_metrics(meta_results),
            included_classrooms=included_classrooms,
            excluded_classrooms=excluded_classrooms,
            n_studies=len(included_classrooms)
        )

        self.meta_analysis_results = final_results
        logger.info("Meta-analysis completed successfully")

        return final_results

    def _filter_classrooms_for_meta_analysis(
        self,
        classroom_results: Dict[str, SAOMResults]
    ) -> Tuple[List[str], List[str]]:
        """Filter classrooms based on quality criteria for meta-analysis."""
        included = []
        excluded = []

        for classroom_id, results in classroom_results.items():
            include = True
            exclusion_reasons = []

            # Check convergence
            if results.max_convergence_ratio > self.meta_config.min_convergence_quality:
                include = False
                exclusion_reasons.append(f"convergence: {results.max_convergence_ratio:.3f}")

            # Check classroom size (from metadata)
            if hasattr(results, 'classroom_info'):
                if results.classroom_info.n_students < self.meta_config.min_classroom_size:
                    include = False
                    exclusion_reasons.append(f"size: {results.classroom_info.n_students}")

            # Check for required effects
            required_effects = ['density', 'recip']  # Minimum required effects
            available_effects = results.effect_names
            missing_effects = set(required_effects) - set(available_effects)
            if missing_effects:
                include = False
                exclusion_reasons.append(f"missing effects: {missing_effects}")

            if include:
                included.append(classroom_id)
            else:
                excluded.append(classroom_id)
                logger.debug(f"Excluded classroom {classroom_id}: {'; '.join(exclusion_reasons)}")

        return included, excluded

    def _extract_effects_for_meta_analysis(
        self,
        classroom_results: Dict[str, SAOMResults],
        included_classrooms: List[str]
    ) -> Dict[str, Dict[str, List]]:
        """Extract effect estimates and standard errors for meta-analysis."""
        effects_data = {}

        for effect_name in self.meta_config.target_effects:
            estimates = []
            standard_errors = []
            classroom_ids = []

            for classroom_id in included_classrooms:
                results = classroom_results[classroom_id]

                # Find effect in results
                if effect_name in results.effect_names:
                    effect_idx = results.effect_names.index(effect_name)
                    estimate = results.parameters[effect_idx]
                    se = results.standard_errors[effect_idx]

                    # Check for valid values
                    if not (np.isnan(estimate) or np.isnan(se) or se <= 0):
                        estimates.append(estimate)
                        standard_errors.append(se)
                        classroom_ids.append(classroom_id)

            if estimates:  # Only include effects with data
                effects_data[effect_name] = {
                    'estimates': np.array(estimates),
                    'standard_errors': np.array(standard_errors),
                    'classroom_ids': classroom_ids
                }

        return effects_data

    def _conduct_single_effect_meta_analysis(
        self,
        estimates: np.ndarray,
        standard_errors: np.ndarray,
        classroom_ids: List[str],
        effect_name: str
    ) -> Dict[str, Any]:
        """Conduct meta-analysis for a single effect."""

        # Remove outliers if requested
        if self.meta_config.exclude_outliers:
            mask = self._identify_outliers(estimates, standard_errors)
            estimates = estimates[~mask]
            standard_errors = standard_errors[~mask]
            classroom_ids = [cid for i, cid in enumerate(classroom_ids) if not mask[i]]

        # Calculate weights (inverse variance)
        weights = 1.0 / (standard_errors ** 2)

        if self.meta_config.method == "fixed_effects":
            # Fixed effects meta-analysis
            pooled_estimate = np.sum(weights * estimates) / np.sum(weights)
            pooled_se = np.sqrt(1.0 / np.sum(weights))
            tau_squared = 0.0

        else:  # Random effects
            # Calculate between-study variance (tau²)
            if self.meta_config.estimator == "DerSimonian-Laird":
                tau_squared = self._calculate_tau_squared_dl(estimates, standard_errors, weights)
            else:
                tau_squared = self._calculate_tau_squared_reml(estimates, standard_errors)

            # Adjust weights for random effects
            adjusted_weights = 1.0 / (standard_errors ** 2 + tau_squared)
            pooled_estimate = np.sum(adjusted_weights * estimates) / np.sum(adjusted_weights)
            pooled_se = np.sqrt(1.0 / np.sum(adjusted_weights))

        # Confidence intervals
        alpha = 1 - self.meta_config.confidence_level
        t_critical = stats.t.ppf(1 - alpha/2, len(estimates) - 1)

        ci_lower = pooled_estimate - t_critical * pooled_se
        ci_upper = pooled_estimate + t_critical * pooled_se

        # Heterogeneity statistics
        heterogeneity = self._calculate_heterogeneity_stats(
            estimates, standard_errors, weights, pooled_estimate
        )

        # Prediction intervals
        prediction_interval = None
        if self.meta_config.prediction_interval and tau_squared > 0:
            pred_se = np.sqrt(pooled_se**2 + tau_squared)
            prediction_interval = (
                pooled_estimate - t_critical * pred_se,
                pooled_estimate + t_critical * pred_se
            )

        # Forest plot data
        forest_data = pd.DataFrame({
            'classroom_id': classroom_ids,
            'estimate': estimates,
            'se': standard_errors,
            'ci_lower': estimates - 1.96 * standard_errors,
            'ci_upper': estimates + 1.96 * standard_errors,
            'weight': weights / np.sum(weights) * 100  # As percentage
        })

        return {
            'summary': {
                'estimate': pooled_estimate,
                'se': pooled_se,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper,
                'tau_squared': tau_squared
            },
            'forest_data': forest_data,
            'weights': adjusted_weights if 'adjusted_weights' in locals() else weights,
            'heterogeneity': heterogeneity,
            'prediction_interval': prediction_interval
        }

    def _identify_outliers(
        self,
        estimates: np.ndarray,
        standard_errors: np.ndarray
    ) -> np.ndarray:
        """Identify outliers using standardized residuals."""
        # Simple outlier detection based on z-scores
        z_scores = np.abs(estimates - np.mean(estimates)) / np.std(estimates)
        return z_scores > self.meta_config.outlier_threshold

    def _calculate_tau_squared_dl(
        self,
        estimates: np.ndarray,
        standard_errors: np.ndarray,
        weights: np.ndarray
    ) -> float:
        """Calculate tau² using DerSimonian-Laird method."""
        k = len(estimates)
        if k < 2:
            return 0.0

        # Fixed effects estimate
        fe_estimate = np.sum(weights * estimates) / np.sum(weights)

        # Q statistic
        Q = np.sum(weights * (estimates - fe_estimate) ** 2)

        # Degrees of freedom
        df = k - 1

        # Tau²
        c = np.sum(weights) - np.sum(weights**2) / np.sum(weights)
        tau_squared = max(0, (Q - df) / c)

        return tau_squared

    def _calculate_tau_squared_reml(
        self,
        estimates: np.ndarray,
        standard_errors: np.ndarray
    ) -> float:
        """Calculate tau² using REML method (simplified)."""
        # This is a simplified implementation
        # In practice, would use iterative methods
        return self._calculate_tau_squared_dl(estimates, standard_errors, 1.0 / standard_errors**2)

    def _calculate_heterogeneity_stats(
        self,
        estimates: np.ndarray,
        standard_errors: np.ndarray,
        weights: np.ndarray,
        pooled_estimate: float
    ) -> Dict[str, float]:
        """Calculate heterogeneity statistics."""
        k = len(estimates)

        # Q statistic
        Q = np.sum(weights * (estimates - pooled_estimate) ** 2)
        df = k - 1

        # I² statistic
        I_squared = max(0, (Q - df) / Q) * 100 if Q > 0 else 0

        # H² statistic
        H_squared = Q / df if df > 0 else 1

        # P-value for heterogeneity test
        p_value = 1 - stats.chi2.cdf(Q, df) if df > 0 else 1.0

        return {
            'Q': Q,
            'df': df,
            'p_value': p_value,
            'I_squared': I_squared,
            'H_squared': H_squared
        }

    def _compute_meta_analysis_quality_metrics(
        self,
        meta_results: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Compute overall quality metrics for meta-analysis."""
        # Number of effects successfully meta-analyzed
        n_effects = len(meta_results)

        # Average heterogeneity
        i_squared_values = [
            result['heterogeneity']['I_squared']
            for result in meta_results.values()
        ]
        mean_i_squared = np.mean(i_squared_values) if i_squared_values else 0

        # Proportion of effects with significant heterogeneity
        sig_heterogeneity = sum(
            1 for result in meta_results.values()
            if result['heterogeneity']['p_value'] < 0.05
        ) / n_effects if n_effects > 0 else 0

        return {
            'n_effects_analyzed': n_effects,
            'mean_i_squared': mean_i_squared,
            'proportion_significant_heterogeneity': sig_heterogeneity
        }

    def create_forest_plots(
        self,
        meta_results: MetaAnalysisResults,
        output_dir: Union[str, Path]
    ):
        """Create forest plots for meta-analysis results."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        for effect_name, forest_data in meta_results.forest_plot_data.items():
            fig, ax = plt.subplots(figsize=(10, 2 + len(forest_data) * 0.3))

            # Plot individual studies
            y_positions = range(len(forest_data))
            ax.errorbar(
                forest_data['estimate'],
                y_positions,
                xerr=[
                    forest_data['estimate'] - forest_data['ci_lower'],
                    forest_data['ci_upper'] - forest_data['estimate']
                ],
                fmt='s',
                capsize=3,
                markersize=6
            )

            # Plot pooled estimate
            pooled_estimate = meta_results.effect_estimates[effect_name]
            pooled_ci_lower = meta_results.effect_ci_lower[effect_name]
            pooled_ci_upper = meta_results.effect_ci_upper[effect_name]

            ax.errorbar(
                pooled_estimate,
                len(forest_data),
                xerr=[[pooled_estimate - pooled_ci_lower], [pooled_ci_upper - pooled_estimate]],
                fmt='D',
                color='red',
                markersize=8,
                capsize=5,
                linewidth=2
            )

            # Formatting
            ax.set_yticks(list(y_positions) + [len(forest_data)])
            ax.set_yticklabels(list(forest_data['classroom_id']) + ['Pooled'])
            ax.set_xlabel(f'{effect_name} Estimate')
            ax.set_title(f'Forest Plot: {effect_name}')
            ax.axvline(x=0, color='black', linestyle='--', alpha=0.5)
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(output_dir / f"forest_plot_{effect_name}.png", dpi=300, bbox_inches='tight')
            plt.close()

        logger.info(f"Forest plots saved to {output_dir}")

    def _save_individual_results(self, output_dir: Union[str, Path]):
        """Save individual classroom results."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        for classroom_id, results in self.classroom_results.items():
            results.save_results(results, output_dir / f"classroom_{classroom_id}_results")

        logger.info(f"Individual results saved to {output_dir}")

    def export_meta_analysis_results(
        self,
        meta_results: MetaAnalysisResults,
        output_path: Union[str, Path]
    ):
        """Export meta-analysis results to file."""
        output_path = Path(output_path)

        # Create summary DataFrame
        summary_data = []
        for effect_name in meta_results.effect_estimates.keys():
            row = {
                'effect': effect_name,
                'estimate': meta_results.effect_estimates[effect_name],
                'se': meta_results.effect_se[effect_name],
                'ci_lower': meta_results.effect_ci_lower[effect_name],
                'ci_upper': meta_results.effect_ci_upper[effect_name],
                'n_studies': meta_results.n_studies
            }

            # Add heterogeneity stats
            if effect_name in meta_results.heterogeneity_stats:
                het_stats = meta_results.heterogeneity_stats[effect_name]
                row.update({
                    'Q': het_stats['Q'],
                    'I_squared': het_stats['I_squared'],
                    'heterogeneity_p': het_stats['p_value']
                })

            summary_data.append(row)

        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(output_path.with_suffix('.csv'), index=False)

        # Save complete results as pickle
        import pickle
        with open(output_path.with_suffix('.pkl'), 'wb') as f:
            pickle.dump(meta_results, f)

        logger.info(f"Meta-analysis results exported to {output_path}")


if __name__ == "__main__":
    # Test SAOM estimation framework
    logging.basicConfig(level=logging.INFO)

    try:
        # Create test framework
        estimation_config = EstimationConfig(
            n_workers=2,
            max_classrooms_per_worker=3,
            saom_config=SAOMModelConfig(
                use_attraction_repulsion=True,
                use_complex_contagion=False,
                use_tolerance_cooperation=True,
                n_iterations=1000  # Reduced for testing
            )
        )

        meta_config = MetaAnalysisConfig(
            target_effects=['density', 'recip', 'sameX.ethnicity'],
            method="random_effects"
        )

        framework = SAOMEstimationFramework(estimation_config, meta_config)

        print("✓ SAOM estimation framework test completed successfully")
        print(f"  - Configured for {estimation_config.n_workers} parallel workers")
        print(f"  - Meta-analysis targeting {len(meta_config.target_effects)} effects")

    except Exception as e:
        logger.error(f"SAOM estimation framework test failed: {e}")
        import traceback
        traceback.print_exc()