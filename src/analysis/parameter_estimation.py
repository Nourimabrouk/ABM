"""
Bayesian Parameter Estimation for ABM-RSiena Integration

This module implements comprehensive Bayesian parameter estimation, uncertainty
quantification, and model comparison using PyMC for computational social science
research meeting PhD dissertation standards.

Author: Gamma Agent - Statistical Analysis & Validation Specialist
Date: 2025-09-15
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass, field
from pathlib import Path
import warnings
import pickle

# Bayesian analysis libraries
import pymc as pm
import arviz as az
import pytensor.tensor as pt
from scipy import stats
from scipy.optimize import minimize

# Network analysis
import networkx as nx

# Plotting
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)

@dataclass
class PriorSpecification:
    """Container for Bayesian prior specifications."""
    parameter_name: str
    distribution: str  # 'normal', 'beta', 'gamma', 'uniform', etc.
    parameters: Dict[str, float]  # Distribution parameters
    description: str = ""
    theoretical_justification: str = ""

@dataclass
class PosteriorResults:
    """Container for Bayesian posterior estimation results."""
    parameter_name: str
    mean: float
    std: float
    hdi_95: Tuple[float, float]  # 95% Highest Density Interval
    hdi_89: Tuple[float, float]  # 89% Highest Density Interval
    median: float
    mode: float
    effective_sample_size: float
    rhat: float  # Gelman-Rubin statistic
    mcse: float  # Monte Carlo Standard Error
    posterior_samples: np.ndarray

@dataclass
class ModelComparison:
    """Container for Bayesian model comparison results."""
    model_names: List[str]
    loo_scores: Dict[str, float]  # Leave-One-Out Cross-Validation
    waic_scores: Dict[str, float]  # Watanabe-Akaike Information Criterion
    model_weights: Dict[str, float]  # Model averaging weights
    best_model: str
    evidence_strength: str  # 'decisive', 'strong', 'moderate', 'weak'
    bayes_factors: Dict[str, float]

class NetworkPriors:
    """
    Class for defining theory-informed priors for network parameters.
    Based on established network science literature and empirical findings.
    """

    @staticmethod
    def get_density_prior(network_type: str = "social") -> PriorSpecification:
        """
        Get theoretically-informed prior for network density.

        Args:
            network_type: Type of network ('social', 'collaboration', 'friendship')

        Returns:
            PriorSpecification object
        """
        if network_type == "social":
            # Social networks typically have low density (Dunbar's number constraints)
            return PriorSpecification(
                parameter_name="density",
                distribution="beta",
                parameters={"alpha": 2.0, "beta": 10.0},  # Peak around 0.15
                description="Social network density prior",
                theoretical_justification="Based on Dunbar's number and empirical social network studies"
            )
        elif network_type == "friendship":
            # Friendship networks even lower density
            return PriorSpecification(
                parameter_name="density",
                distribution="beta",
                parameters={"alpha": 1.5, "beta": 15.0},  # Peak around 0.09
                description="Friendship network density prior",
                theoretical_justification="Friendship networks constrained by cognitive limits and time"
            )
        elif network_type == "collaboration":
            # Professional collaboration networks moderate density
            return PriorSpecification(
                parameter_name="density",
                distribution="beta",
                parameters={"alpha": 3.0, "beta": 8.0},  # Peak around 0.27
                description="Collaboration network density prior",
                theoretical_justification="Professional networks have moderate density due to project constraints"
            )
        else:
            # Default weakly informative prior
            return PriorSpecification(
                parameter_name="density",
                distribution="uniform",
                parameters={"lower": 0.01, "upper": 0.5},
                description="Weakly informative density prior",
                theoretical_justification="Conservative uniform prior with realistic bounds"
            )

    @staticmethod
    def get_clustering_prior(network_type: str = "social") -> PriorSpecification:
        """Get prior for clustering coefficient based on network type."""
        if network_type in ["social", "friendship"]:
            # Social networks exhibit high clustering (homophily)
            return PriorSpecification(
                parameter_name="clustering",
                distribution="beta",
                parameters={"alpha": 5.0, "beta": 2.0},  # Peak around 0.7
                description="Social network clustering prior",
                theoretical_justification="High clustering due to triadic closure and homophily"
            )
        else:
            return PriorSpecification(
                parameter_name="clustering",
                distribution="beta",
                parameters={"alpha": 2.0, "beta": 2.0},  # Uniform on [0,1]
                description="Weakly informative clustering prior",
                theoretical_justification="Neutral prior for clustering"
            )

    @staticmethod
    def get_degree_distribution_prior() -> List[PriorSpecification]:
        """Get priors for degree distribution parameters."""
        return [
            PriorSpecification(
                parameter_name="degree_power_law_alpha",
                distribution="normal",
                parameters={"mu": 2.5, "sigma": 0.5},
                description="Power law exponent for degree distribution",
                theoretical_justification="Most real networks have power law exponent between 2-3"
            ),
            PriorSpecification(
                parameter_name="degree_cutoff",
                distribution="lognormal",
                parameters={"mu": 2.0, "sigma": 0.5},
                description="Degree distribution cutoff parameter",
                theoretical_justification="Natural cutoff due to resource constraints"
            )
        ]

    @staticmethod
    def get_temporal_parameters_prior() -> List[PriorSpecification]:
        """Get priors for temporal evolution parameters."""
        return [
            PriorSpecification(
                parameter_name="edge_formation_rate",
                distribution="gamma",
                parameters={"alpha": 2.0, "beta": 10.0},
                description="Rate of new edge formation",
                theoretical_justification="Formation rates are positive with moderate variation"
            ),
            PriorSpecification(
                parameter_name="edge_dissolution_rate",
                distribution="gamma",
                parameters={"alpha": 1.5, "beta": 15.0},
                description="Rate of edge dissolution",
                theoretical_justification="Dissolution typically slower than formation"
            ),
            PriorSpecification(
                parameter_name="temporal_correlation",
                distribution="beta",
                parameters={"alpha": 8.0, "beta": 2.0},
                description="Temporal correlation in network evolution",
                theoretical_justification="Networks show strong temporal dependence"
            )
        ]

class BayesianNetworkModel:
    """
    Bayesian model for network parameter estimation and uncertainty quantification.
    """

    def __init__(self, model_name: str = "network_model"):
        self.model_name = model_name
        self.model = None
        self.trace = None
        self.prior_predictive = None
        self.posterior_predictive = None
        self.priors = NetworkPriors()

    def build_density_model(self, observed_densities: np.ndarray,
                          network_type: str = "social") -> pm.Model:
        """
        Build Bayesian model for network density estimation.

        Args:
            observed_densities: Array of observed network densities
            network_type: Type of network for prior selection

        Returns:
            PyMC model object
        """
        density_prior = self.priors.get_density_prior(network_type)

        with pm.Model() as model:
            # Prior for true density parameter
            if density_prior.distribution == "beta":
                true_density = pm.Beta("true_density",
                                     alpha=density_prior.parameters["alpha"],
                                     beta=density_prior.parameters["beta"])
            elif density_prior.distribution == "uniform":
                true_density = pm.Uniform("true_density",
                                        lower=density_prior.parameters["lower"],
                                        upper=density_prior.parameters["upper"])

            # Measurement error
            sigma = pm.HalfNormal("sigma", sigma=0.05)  # Small measurement error

            # Likelihood
            likelihood = pm.Normal("observed_densities",
                                 mu=true_density,
                                 sigma=sigma,
                                 observed=observed_densities)

        self.model = model
        return model

    def build_degree_distribution_model(self, observed_degrees: np.ndarray) -> pm.Model:
        """
        Build Bayesian model for degree distribution parameters.

        Args:
            observed_degrees: Array of observed degrees

        Returns:
            PyMC model object
        """
        with pm.Model() as model:
            # Power law parameters
            alpha = pm.Normal("power_law_alpha", mu=2.5, sigma=0.5)
            xmin = pm.Lognormal("power_law_xmin", mu=1.0, sigma=0.3)

            # Truncated power law likelihood
            # Using custom likelihood for power law
            def power_law_logp(degrees, alpha, xmin):
                valid_degrees = degrees[degrees >= xmin]
                if len(valid_degrees) == 0:
                    return -np.inf

                log_prob = (len(valid_degrees) * pt.log(alpha - 1) -
                           len(valid_degrees) * pt.log(xmin) -
                           alpha * pt.sum(pt.log(valid_degrees / xmin)))
                return log_prob

            # Custom potential for power law
            pm.Potential("power_law_potential",
                        power_law_logp(observed_degrees, alpha, xmin))

        self.model = model
        return model

    def build_clustering_model(self, observed_clustering: np.ndarray,
                             network_type: str = "social") -> pm.Model:
        """
        Build Bayesian model for clustering coefficient estimation.

        Args:
            observed_clustering: Array of observed clustering coefficients
            network_type: Type of network for prior selection

        Returns:
            PyMC model object
        """
        clustering_prior = self.priors.get_clustering_prior(network_type)

        with pm.Model() as model:
            # Prior for true clustering
            true_clustering = pm.Beta("true_clustering",
                                    alpha=clustering_prior.parameters["alpha"],
                                    beta=clustering_prior.parameters["beta"])

            # Measurement precision (higher for more reliable measurements)
            precision = pm.Gamma("precision", alpha=10.0, beta=1.0)

            # Beta likelihood (appropriate for [0,1] bounded data)
            alpha_param = true_clustering * precision
            beta_param = (1 - true_clustering) * precision

            likelihood = pm.Beta("observed_clustering",
                               alpha=alpha_param,
                               beta=beta_param,
                               observed=observed_clustering)

        self.model = model
        return model

    def build_temporal_evolution_model(self, density_time_series: np.ndarray,
                                     time_points: np.ndarray) -> pm.Model:
        """
        Build Bayesian model for temporal network evolution.

        Args:
            density_time_series: Time series of network densities
            time_points: Corresponding time points

        Returns:
            PyMC model object
        """
        with pm.Model() as model:
            # Initial density
            initial_density = pm.Beta("initial_density", alpha=2.0, beta=8.0)

            # Evolution parameters
            formation_rate = pm.Gamma("formation_rate", alpha=2.0, beta=10.0)
            dissolution_rate = pm.Gamma("dissolution_rate", alpha=1.5, beta=15.0)

            # Temporal correlation
            rho = pm.Beta("temporal_correlation", alpha=8.0, beta=2.0)

            # Observation noise
            sigma = pm.HalfNormal("observation_sigma", sigma=0.02)

            # Latent process (AR(1) for densities)
            latent_density = pm.AR("latent_density",
                                 rho=rho,
                                 sigma=sigma,
                                 init_dist=pm.Beta.dist(alpha=2.0, beta=8.0),
                                 shape=len(time_points))

            # Likelihood
            likelihood = pm.Normal("observed_densities",
                                 mu=latent_density,
                                 sigma=sigma,
                                 observed=density_time_series)

        self.model = model
        return model

    def build_comparative_model(self, abm_data: Dict[str, np.ndarray],
                              rsiena_data: Dict[str, np.ndarray]) -> pm.Model:
        """
        Build Bayesian model comparing ABM and RSiena outputs.

        Args:
            abm_data: Dictionary with ABM network metrics
            rsiena_data: Dictionary with RSiena network metrics

        Returns:
            PyMC model object
        """
        with pm.Model() as model:
            # True underlying parameters
            true_density = pm.Beta("true_density", alpha=2.0, beta=8.0)
            true_clustering = pm.Beta("true_clustering", alpha=5.0, beta=2.0)

            # Method-specific biases
            abm_density_bias = pm.Normal("abm_density_bias", mu=0, sigma=0.05)
            rsiena_density_bias = pm.Normal("rsiena_density_bias", mu=0, sigma=0.05)

            abm_clustering_bias = pm.Normal("abm_clustering_bias", mu=0, sigma=0.1)
            rsiena_clustering_bias = pm.Normal("rsiena_clustering_bias", mu=0, sigma=0.1)

            # Method-specific precisions
            abm_precision = pm.Gamma("abm_precision", alpha=10.0, beta=1.0)
            rsiena_precision = pm.Gamma("rsiena_precision", alpha=10.0, beta=1.0)

            # Likelihoods for densities
            pm.Normal("abm_density_obs",
                     mu=true_density + abm_density_bias,
                     sigma=1/pm.math.sqrt(abm_precision),
                     observed=abm_data.get('density', []))

            pm.Normal("rsiena_density_obs",
                     mu=true_density + rsiena_density_bias,
                     sigma=1/pm.math.sqrt(rsiena_precision),
                     observed=rsiena_data.get('density', []))

            # Likelihoods for clustering (if available)
            if 'clustering' in abm_data and 'clustering' in rsiena_data:
                pm.Beta("abm_clustering_obs",
                       alpha=(true_clustering + abm_clustering_bias) * abm_precision,
                       beta=(1 - true_clustering - abm_clustering_bias) * abm_precision,
                       observed=abm_data['clustering'])

                pm.Beta("rsiena_clustering_obs",
                       alpha=(true_clustering + rsiena_clustering_bias) * rsiena_precision,
                       beta=(1 - true_clustering - rsiena_clustering_bias) * rsiena_precision,
                       observed=rsiena_data['clustering'])

        self.model = model
        return model

    def sample_posterior(self, draws: int = 2000, tune: int = 1000,
                       chains: int = 4, cores: int = None,
                       target_accept: float = 0.95) -> az.InferenceData:
        """
        Sample from the posterior distribution using NUTS sampler.

        Args:
            draws: Number of posterior samples per chain
            tune: Number of tuning steps per chain
            chains: Number of MCMC chains
            cores: Number of CPU cores to use
            target_accept: Target acceptance rate

        Returns:
            ArviZ InferenceData object
        """
        if self.model is None:
            raise ValueError("Model must be built before sampling")

        logger.info(f"Sampling {draws} draws from {chains} chains")

        with self.model:
            # Sample posterior
            self.trace = pm.sample(draws=draws,
                                 tune=tune,
                                 chains=chains,
                                 cores=cores,
                                 target_accept=target_accept,
                                 return_inferencedata=True)

            # Sample prior predictive
            self.prior_predictive = pm.sample_prior_predictive(samples=1000)

            # Sample posterior predictive
            self.posterior_predictive = pm.sample_posterior_predictive(self.trace,
                                                                     samples=500)

        logger.info("Posterior sampling completed")
        return self.trace

    def analyze_convergence(self) -> Dict[str, Any]:
        """
        Analyze MCMC convergence diagnostics.

        Returns:
            Dictionary with convergence diagnostics
        """
        if self.trace is None:
            raise ValueError("Must sample posterior before analyzing convergence")

        logger.info("Analyzing MCMC convergence")

        # R-hat statistics
        rhat = az.rhat(self.trace)

        # Effective sample size
        ess_bulk = az.ess(self.trace, kind="bulk")
        ess_tail = az.ess(self.trace, kind="tail")

        # Monte Carlo Standard Error
        mcse = az.mcse(self.trace)

        # Energy diagnostics
        energy = az.bfmi(self.trace)

        # Divergences
        divergences = self.trace.sample_stats.diverging.sum().values

        convergence_summary = {
            'rhat_max': float(rhat.max()),
            'rhat_problematic': int((rhat > 1.01).sum()),
            'ess_bulk_min': float(ess_bulk.min()),
            'ess_tail_min': float(ess_tail.min()),
            'mcse_max': float(mcse.max()),
            'energy_bfmi': float(energy.mean()),
            'total_divergences': int(divergences.sum()),
            'convergence_ok': bool(rhat.max() < 1.01 and ess_bulk.min() > 400 and
                                 ess_tail.min() > 400 and divergences.sum() == 0)
        }

        if not convergence_summary['convergence_ok']:
            logger.warning("MCMC convergence issues detected!")

        return convergence_summary

    def posterior_summary(self) -> Dict[str, PosteriorResults]:
        """
        Generate comprehensive posterior summary statistics.

        Returns:
            Dictionary of PosteriorResults for each parameter
        """
        if self.trace is None:
            raise ValueError("Must sample posterior before summarizing")

        logger.info("Generating posterior summary")

        summary_stats = az.summary(self.trace, hdi_prob=0.95)
        hdi_89 = az.hdi(self.trace, hdi_prob=0.89)

        results = {}

        for param_name in summary_stats.index:
            # Extract samples for this parameter
            samples = self.trace.posterior[param_name].values.flatten()

            # Calculate mode (highest density point)
            from scipy.stats import gaussian_kde
            try:
                kde = gaussian_kde(samples)
                x_range = np.linspace(samples.min(), samples.max(), 1000)
                kde_vals = kde(x_range)
                mode = x_range[np.argmax(kde_vals)]
            except:
                mode = np.median(samples)  # Fallback to median

            results[param_name] = PosteriorResults(
                parameter_name=param_name,
                mean=float(summary_stats.loc[param_name, 'mean']),
                std=float(summary_stats.loc[param_name, 'sd']),
                hdi_95=(float(summary_stats.loc[param_name, 'hdi_2.5%']),
                       float(summary_stats.loc[param_name, 'hdi_97.5%'])),
                hdi_89=(float(hdi_89[param_name].values[0]),
                       float(hdi_89[param_name].values[1])),
                median=float(np.median(samples)),
                mode=float(mode),
                effective_sample_size=float(summary_stats.loc[param_name, 'ess_bulk']),
                rhat=float(summary_stats.loc[param_name, 'r_hat']),
                mcse=float(summary_stats.loc[param_name, 'mcse_mean']),
                posterior_samples=samples
            )

        return results

    def model_comparison(self, models: List['BayesianNetworkModel']) -> ModelComparison:
        """
        Compare multiple Bayesian models using LOO-CV and WAIC.

        Args:
            models: List of BayesianNetworkModel objects to compare

        Returns:
            ModelComparison object
        """
        logger.info(f"Comparing {len(models)} Bayesian models")

        model_names = [model.model_name for model in models]
        traces = {name: model.trace for name, model in zip(model_names, models)}

        # Calculate LOO and WAIC scores
        loo_scores = {}
        waic_scores = {}

        for name, trace in traces.items():
            try:
                loo = az.loo(trace)
                waic = az.waic(trace)

                loo_scores[name] = float(loo.elpd_loo)
                waic_scores[name] = float(waic.elpd_waic)
            except Exception as e:
                logger.warning(f"Model comparison failed for {name}: {e}")
                loo_scores[name] = np.nan
                waic_scores[name] = np.nan

        # Model comparison using LOO
        try:
            loo_compare = az.compare({name: trace for name, trace in traces.items()})
            best_model = str(loo_compare.index[0])

            # Calculate Akaike weights
            loo_ic = loo_compare['elpd_loo'].values
            loo_weights = np.exp(loo_ic - np.max(loo_ic))
            loo_weights = loo_weights / np.sum(loo_weights)

            model_weights = dict(zip(loo_compare.index, loo_weights))

            # Determine evidence strength based on weight difference
            best_weight = model_weights[best_model]
            if best_weight > 0.9:
                evidence_strength = "decisive"
            elif best_weight > 0.75:
                evidence_strength = "strong"
            elif best_weight > 0.6:
                evidence_strength = "moderate"
            else:
                evidence_strength = "weak"

        except Exception as e:
            logger.warning(f"Model comparison failed: {e}")
            best_model = model_names[0]
            model_weights = {name: 1.0/len(model_names) for name in model_names}
            evidence_strength = "inconclusive"

        # Calculate approximate Bayes factors (using LOO differences)
        bayes_factors = {}
        best_loo = max(loo_scores.values())

        for name, loo_score in loo_scores.items():
            if not np.isnan(loo_score) and not np.isnan(best_loo):
                bf = np.exp(loo_score - best_loo)
                bayes_factors[name] = float(bf)
            else:
                bayes_factors[name] = np.nan

        return ModelComparison(
            model_names=model_names,
            loo_scores=loo_scores,
            waic_scores=waic_scores,
            model_weights=model_weights,
            best_model=best_model,
            evidence_strength=evidence_strength,
            bayes_factors=bayes_factors
        )

    def posterior_predictive_checks(self) -> Dict[str, Any]:
        """
        Perform posterior predictive checks for model validation.

        Returns:
            Dictionary with predictive check results
        """
        if self.posterior_predictive is None:
            raise ValueError("Must sample posterior predictive before checks")

        logger.info("Performing posterior predictive checks")

        # This is a placeholder for comprehensive predictive checks
        # In practice, would compare observed vs predicted distributions
        checks = {
            'bayesian_p_values': {},
            'predictive_intervals': {},
            'model_adequacy': 'pending_implementation'
        }

        return checks

    def generate_plots(self, output_dir: Path = None) -> Dict[str, Path]:
        """
        Generate comprehensive Bayesian analysis plots.

        Args:
            output_dir: Directory to save plots

        Returns:
            Dictionary mapping plot types to file paths
        """
        if self.trace is None:
            raise ValueError("Must sample posterior before plotting")

        output_dir = output_dir or Path("outputs/bayesian_plots")
        output_dir.mkdir(parents=True, exist_ok=True)

        plot_files = {}

        # Posterior distributions
        fig, axes = plt.subplots(figsize=(12, 8))
        az.plot_posterior(self.trace, ax=axes)
        plt.tight_layout()
        posterior_file = output_dir / "posterior_distributions.png"
        plt.savefig(posterior_file, dpi=300, bbox_inches='tight')
        plt.close()
        plot_files['posterior'] = posterior_file

        # Trace plots
        fig = az.plot_trace(self.trace, figsize=(12, 8))
        trace_file = output_dir / "trace_plots.png"
        plt.savefig(trace_file, dpi=300, bbox_inches='tight')
        plt.close()
        plot_files['trace'] = trace_file

        # Rank plots for convergence
        fig = az.plot_rank(self.trace, figsize=(10, 6))
        rank_file = output_dir / "rank_plots.png"
        plt.savefig(rank_file, dpi=300, bbox_inches='tight')
        plt.close()
        plot_files['rank'] = rank_file

        # Energy plot
        fig = az.plot_energy(self.trace, figsize=(8, 6))
        energy_file = output_dir / "energy_plot.png"
        plt.savefig(energy_file, dpi=300, bbox_inches='tight')
        plt.close()
        plot_files['energy'] = energy_file

        logger.info(f"Plots saved to {output_dir}")
        return plot_files


class BayesianParameterEstimator:
    """
    Main class for comprehensive Bayesian parameter estimation and model comparison.
    """

    def __init__(self, random_seed: int = 42):
        np.random.seed(random_seed)
        self.models = {}
        self.results = {}

    def estimate_network_parameters(self, network_data: Dict[str, np.ndarray],
                                  network_type: str = "social") -> Dict[str, PosteriorResults]:
        """
        Comprehensive Bayesian estimation of network parameters.

        Args:
            network_data: Dictionary with network metrics (density, clustering, etc.)
            network_type: Type of network for prior selection

        Returns:
            Dictionary of posterior results for each parameter
        """
        logger.info("Starting Bayesian parameter estimation")

        all_results = {}

        # Density estimation
        if 'density' in network_data:
            density_model = BayesianNetworkModel("density_model")
            density_model.build_density_model(network_data['density'], network_type)
            density_model.sample_posterior()

            density_results = density_model.posterior_summary()
            all_results.update(density_results)
            self.models['density'] = density_model

        # Clustering estimation
        if 'clustering' in network_data:
            clustering_model = BayesianNetworkModel("clustering_model")
            clustering_model.build_clustering_model(network_data['clustering'], network_type)
            clustering_model.sample_posterior()

            clustering_results = clustering_model.posterior_summary()
            all_results.update(clustering_results)
            self.models['clustering'] = clustering_model

        # Degree distribution estimation
        if 'degrees' in network_data:
            degree_model = BayesianNetworkModel("degree_model")
            degree_model.build_degree_distribution_model(network_data['degrees'])
            degree_model.sample_posterior()

            degree_results = degree_model.posterior_summary()
            all_results.update(degree_results)
            self.models['degrees'] = degree_model

        logger.info("Bayesian parameter estimation completed")
        return all_results

    def compare_methods(self, abm_data: Dict[str, np.ndarray],
                       rsiena_data: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """
        Bayesian comparison of ABM and RSiena methods.

        Args:
            abm_data: ABM network metrics
            rsiena_data: RSiena network metrics

        Returns:
            Comprehensive comparison results
        """
        logger.info("Starting Bayesian method comparison")

        # Build comparative model
        comparative_model = BayesianNetworkModel("method_comparison")
        comparative_model.build_comparative_model(abm_data, rsiena_data)
        comparative_model.sample_posterior()

        # Get posterior results
        posterior_results = comparative_model.posterior_summary()

        # Analyze method differences
        method_comparison = {
            'posterior_results': posterior_results,
            'convergence_diagnostics': comparative_model.analyze_convergence(),
            'model_adequacy': comparative_model.posterior_predictive_checks()
        }

        # Calculate probability that methods agree
        if 'abm_density_bias' in posterior_results and 'rsiena_density_bias' in posterior_results:
            abm_bias_samples = posterior_results['abm_density_bias'].posterior_samples
            rsiena_bias_samples = posterior_results['rsiena_density_bias'].posterior_samples

            bias_diff = abm_bias_samples - rsiena_bias_samples
            prob_agree = np.mean(np.abs(bias_diff) < 0.05)  # Within 5% agreement

            method_comparison['agreement_probability'] = float(prob_agree)
            method_comparison['bias_difference_hdi'] = (
                float(np.percentile(bias_diff, 2.5)),
                float(np.percentile(bias_diff, 97.5))
            )

        self.models['method_comparison'] = comparative_model
        logger.info("Bayesian method comparison completed")
        return method_comparison

    def uncertainty_quantification(self, parameter_name: str) -> Dict[str, float]:
        """
        Comprehensive uncertainty quantification for a parameter.

        Args:
            parameter_name: Name of parameter to analyze

        Returns:
            Dictionary with uncertainty metrics
        """
        if parameter_name not in self.results:
            raise ValueError(f"Parameter {parameter_name} not found in results")

        result = self.results[parameter_name]
        samples = result.posterior_samples

        # Calculate various uncertainty metrics
        uncertainty_metrics = {
            'posterior_variance': float(np.var(samples)),
            'coefficient_of_variation': float(np.std(samples) / np.mean(samples)),
            'hdi_width_95': float(result.hdi_95[1] - result.hdi_95[0]),
            'hdi_width_89': float(result.hdi_89[1] - result.hdi_89[0]),
            'tail_probability_positive': float(np.mean(samples > 0)),
            'tail_probability_large': float(np.mean(samples > np.median(samples) + 2*np.std(samples))),
            'entropy': float(-np.sum(np.histogram(samples, bins=50)[0] * np.log(np.histogram(samples, bins=50)[0] + 1e-10)))
        }

        return uncertainty_metrics

    def generate_report(self, output_file: Path = None) -> str:
        """
        Generate comprehensive Bayesian analysis report.

        Args:
            output_file: Optional file to save report

        Returns:
            Report text
        """
        report_lines = []
        report_lines.append("# Bayesian Parameter Estimation Report")
        report_lines.append("=" * 50)
        report_lines.append("")

        # Model summaries
        report_lines.append("## Model Summaries")
        report_lines.append("")

        for model_name, model in self.models.items():
            report_lines.append(f"### {model_name}")

            # Convergence diagnostics
            if hasattr(model, 'trace') and model.trace is not None:
                conv_diag = model.analyze_convergence()
                report_lines.append(f"- Convergence Status: {'✓ Good' if conv_diag['convergence_ok'] else '✗ Issues'}")
                report_lines.append(f"- Max R̂: {conv_diag['rhat_max']:.4f}")
                report_lines.append(f"- Min ESS: {min(conv_diag['ess_bulk_min'], conv_diag['ess_tail_min']):.0f}")
                report_lines.append(f"- Divergences: {conv_diag['total_divergences']}")

            # Parameter estimates
            if hasattr(model, 'trace') and model.trace is not None:
                posterior_results = model.posterior_summary()
                for param_name, result in posterior_results.items():
                    report_lines.append(f"  - {param_name}: {result.mean:.4f} ± {result.std:.4f}")
                    report_lines.append(f"    95% HDI: [{result.hdi_95[0]:.4f}, {result.hdi_95[1]:.4f}]")

            report_lines.append("")

        report_text = "\n".join(report_lines)

        # Save report if requested
        if output_file:
            output_file.write_text(report_text)
            logger.info(f"Bayesian analysis report saved to {output_file}")

        return report_text


# Example usage and testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Initialize Bayesian estimator
    estimator = BayesianParameterEstimator()

    # Generate sample network data
    np.random.seed(42)
    network_data = {
        'density': np.random.beta(2, 8, 20),  # 20 network observations
        'clustering': np.random.beta(5, 2, 20),
        'degrees': np.random.poisson(5, 200)  # Degree sequence from networks
    }

    # Estimate parameters
    results = estimator.estimate_network_parameters(network_data, network_type="social")

    # Print summary
    print("Bayesian Parameter Estimation Results:")
    print("=" * 50)
    for param_name, result in results.items():
        print(f"{param_name}:")
        print(f"  Mean: {result.mean:.4f}")
        print(f"  95% HDI: [{result.hdi_95[0]:.4f}, {result.hdi_95[1]:.4f}]")
        print(f"  R̂: {result.rhat:.4f}")
        print()

    # Generate report
    report = estimator.generate_report()
    print(report)