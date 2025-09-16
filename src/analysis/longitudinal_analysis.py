"""
Longitudinal Network Analysis with Change Point Detection

This module implements comprehensive longitudinal network analysis including change
point detection, temporal trend analysis, and dynamic network modeling for
ABM-RSiena integration studies meeting PhD dissertation standards.

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

# Statistical libraries
from scipy import stats
from scipy.signal import find_peaks
from scipy.optimize import minimize
import sklearn.cluster as cluster
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# Time series analysis
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tsa.statespace.structural import UnobservedComponents

# Change point detection
import ruptures as rpt

# Network analysis
import networkx as nx

# Plotting
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)

@dataclass
class ChangePoint:
    """Container for change point information."""
    time_index: int
    time_value: Optional[float] = None
    confidence_score: float = 0.0
    change_type: str = ""  # 'level', 'trend', 'variance', 'structural'
    magnitude: float = 0.0
    description: str = ""

@dataclass
class TemporalTrend:
    """Container for temporal trend analysis."""
    slope: float
    intercept: float
    r_squared: float
    p_value: float
    confidence_interval: Tuple[float, float]
    trend_type: str  # 'increasing', 'decreasing', 'stable'
    seasonal_component: Optional[np.ndarray] = None
    residuals: Optional[np.ndarray] = None

@dataclass
class NetworkEvolution:
    """Container for network evolution analysis."""
    time_points: List[float]
    network_metrics: Dict[str, np.ndarray]
    change_points: List[ChangePoint]
    trends: Dict[str, TemporalTrend]
    stability_periods: List[Tuple[int, int]]  # (start_idx, end_idx)
    transition_periods: List[Tuple[int, int]]
    model_diagnostics: Dict[str, Any] = field(default_factory=dict)

class NetworkMetricsExtractor:
    """
    Extracts comprehensive network metrics from longitudinal network data.
    """

    def __init__(self):
        self.metrics_functions = {
            'density': lambda G: nx.density(G) if len(G) > 0 else 0,
            'clustering': lambda G: nx.average_clustering(G) if len(G) > 0 else 0,
            'transitivity': lambda G: nx.transitivity(G) if len(G) > 0 else 0,
            'diameter': self._safe_diameter,
            'average_path_length': self._safe_avg_path_length,
            'degree_centralization': self._degree_centralization,
            'betweenness_centralization': self._betweenness_centralization,
            'closeness_centralization': self._closeness_centralization,
            'assortativity': self._safe_assortativity,
            'n_components': lambda G: nx.number_connected_components(G),
            'largest_component_size': self._largest_component_fraction,
            'edge_count': lambda G: G.number_of_edges(),
            'node_count': lambda G: G.number_of_nodes(),
            'modularity': self._safe_modularity
        }

    def extract_metrics_time_series(self, networks: List[nx.Graph],
                                   time_points: Optional[List[float]] = None) -> Dict[str, np.ndarray]:
        """
        Extract time series of network metrics.

        Args:
            networks: List of network objects in temporal order
            time_points: Optional time points for each network

        Returns:
            Dictionary with metric time series
        """
        if time_points is None:
            time_points = list(range(len(networks)))

        metrics_series = {metric: [] for metric in self.metrics_functions.keys()}
        metrics_series['time'] = time_points

        for i, network in enumerate(networks):
            logger.debug(f"Extracting metrics for network {i+1}/{len(networks)}")

            for metric_name, metric_func in self.metrics_functions.items():
                try:
                    value = metric_func(network)
                    metrics_series[metric_name].append(value)
                except Exception as e:
                    logger.warning(f"Failed to compute {metric_name} for network {i}: {e}")
                    metrics_series[metric_name].append(np.nan)

        # Convert to numpy arrays
        for key, values in metrics_series.items():
            if key != 'time':
                metrics_series[key] = np.array(values)

        return metrics_series

    def _safe_diameter(self, G):
        """Safely compute network diameter."""
        if len(G) == 0:
            return 0
        if not nx.is_connected(G):
            # Use largest connected component
            largest_cc = max(nx.connected_components(G), key=len, default=set())
            if len(largest_cc) < 2:
                return 0
            G_cc = G.subgraph(largest_cc)
            return nx.diameter(G_cc)
        return nx.diameter(G)

    def _safe_avg_path_length(self, G):
        """Safely compute average path length."""
        if len(G) == 0:
            return 0
        if not nx.is_connected(G):
            largest_cc = max(nx.connected_components(G), key=len, default=set())
            if len(largest_cc) < 2:
                return 0
            G_cc = G.subgraph(largest_cc)
            return nx.average_shortest_path_length(G_cc)
        return nx.average_shortest_path_length(G)

    def _degree_centralization(self, G):
        """Compute degree centralization."""
        if len(G) <= 1:
            return 0
        degrees = dict(G.degree())
        max_degree = max(degrees.values()) if degrees else 0
        n = len(G)
        numerator = sum(max_degree - deg for deg in degrees.values())
        denominator = (n - 1) * (n - 2)
        return numerator / denominator if denominator > 0 else 0

    def _betweenness_centralization(self, G):
        """Compute betweenness centralization."""
        if len(G) <= 2:
            return 0
        betweenness = nx.betweenness_centrality(G)
        max_bet = max(betweenness.values()) if betweenness else 0
        n = len(G)
        numerator = sum(max_bet - bet for bet in betweenness.values())
        denominator = (n - 1) * (n - 2) * (n - 3) / 2
        return numerator / denominator if denominator > 0 else 0

    def _closeness_centralization(self, G):
        """Compute closeness centralization."""
        if len(G) <= 1:
            return 0
        if not nx.is_connected(G):
            return 0  # Closeness centralization requires connected graph
        closeness = nx.closeness_centrality(G)
        max_close = max(closeness.values()) if closeness else 0
        n = len(G)
        numerator = sum(max_close - close for close in closeness.values())
        denominator = (n - 1) * (n - 2) / (2 * n - 3)
        return numerator / denominator if denominator > 0 else 0

    def _safe_assortativity(self, G):
        """Safely compute degree assortativity."""
        try:
            if G.number_of_edges() == 0:
                return 0
            return nx.degree_assortativity_coefficient(G)
        except:
            return 0

    def _largest_component_fraction(self, G):
        """Compute fraction of nodes in largest component."""
        if len(G) == 0:
            return 0
        components = list(nx.connected_components(G))
        if not components:
            return 0
        largest_size = max(len(comp) for comp in components)
        return largest_size / len(G)

    def _safe_modularity(self, G):
        """Safely compute modularity using community detection."""
        if G.number_of_edges() == 0:
            return 0
        try:
            # Use Louvain algorithm for community detection
            import networkx.algorithms.community as nx_comm
            communities = nx_comm.greedy_modularity_communities(G)
            return nx_comm.modularity(G, communities)
        except:
            return 0

class ChangePointDetector:
    """
    Implements various change point detection algorithms for network time series.
    """

    def __init__(self):
        self.methods = {
            'pelt': self._pelt_detection,
            'binary_segmentation': self._binary_segmentation,
            'window': self._window_detection,
            'kernel': self._kernel_detection,
            'dynp': self._dynamic_programming
        }

    def detect_change_points(self, time_series: np.ndarray, method: str = 'pelt',
                           model: str = 'rbf', min_size: int = 2,
                           jump: int = 5, pen: float = None) -> List[ChangePoint]:
        """
        Detect change points in time series data.

        Args:
            time_series: Time series data
            method: Detection method ('pelt', 'binary_segmentation', 'window', etc.)
            model: Cost function model ('l2', 'rbf', 'linear', 'normal', 'autoregressive')
            min_size: Minimum segment size
            jump: Jump size for efficiency
            pen: Penalty parameter (auto-detected if None)

        Returns:
            List of ChangePoint objects
        """
        if len(time_series) < 4:
            logger.warning("Time series too short for change point detection")
            return []

        # Handle NaN values
        valid_indices = ~np.isnan(time_series)
        if not np.any(valid_indices):
            logger.warning("Time series contains only NaN values")
            return []

        clean_series = time_series[valid_indices]
        if len(clean_series) < 4:
            logger.warning("Too few valid values for change point detection")
            return []

        logger.info(f"Detecting change points using {method} method")

        try:
            detection_func = self.methods.get(method, self._pelt_detection)
            change_points = detection_func(clean_series, model, min_size, jump, pen)

            # Convert back to original indices if needed
            if np.sum(valid_indices) != len(time_series):
                original_indices = np.where(valid_indices)[0]
                for cp in change_points:
                    if cp.time_index < len(original_indices):
                        cp.time_index = original_indices[cp.time_index]

            return change_points

        except Exception as e:
            logger.error(f"Change point detection failed: {e}")
            return []

    def _pelt_detection(self, series: np.ndarray, model: str, min_size: int,
                       jump: int, pen: float) -> List[ChangePoint]:
        """PELT (Pruned Exact Linear Time) change point detection."""
        if pen is None:
            pen = np.log(len(series)) * np.var(series)

        algo = rpt.Pelt(model=model, min_size=min_size, jump=jump).fit(series)
        change_indices = algo.predict(pen=pen)

        change_points = []
        for idx in change_indices[:-1]:  # Exclude the end point
            # Estimate change magnitude
            before_segment = series[max(0, idx-min_size):idx]
            after_segment = series[idx:min(idx+min_size, len(series))]

            if len(before_segment) > 0 and len(after_segment) > 0:
                magnitude = abs(np.mean(after_segment) - np.mean(before_segment))
                change_type = self._classify_change_type(before_segment, after_segment)

                change_points.append(ChangePoint(
                    time_index=idx,
                    confidence_score=1.0,  # PELT doesn't provide confidence scores
                    change_type=change_type,
                    magnitude=magnitude,
                    description=f"PELT detected change at index {idx}"
                ))

        return change_points

    def _binary_segmentation(self, series: np.ndarray, model: str, min_size: int,
                           jump: int, pen: float) -> List[ChangePoint]:
        """Binary segmentation change point detection."""
        if pen is None:
            pen = np.log(len(series)) * np.var(series)

        algo = rpt.Binseg(model=model, min_size=min_size, jump=jump).fit(series)
        n_bkps = min(10, len(series) // (2 * min_size))  # Reasonable number of breakpoints
        change_indices = algo.predict(n_bkps=n_bkps)

        change_points = []
        for idx in change_indices[:-1]:
            # Estimate confidence and magnitude
            window_size = min(min_size * 2, len(series) // 4)
            before_segment = series[max(0, idx-window_size):idx]
            after_segment = series[idx:min(idx+window_size, len(series))]

            if len(before_segment) > 0 and len(after_segment) > 0:
                magnitude = abs(np.mean(after_segment) - np.mean(before_segment))
                change_type = self._classify_change_type(before_segment, after_segment)

                # Simple confidence based on magnitude relative to variance
                confidence = min(1.0, magnitude / (np.std(series) + 1e-10))

                change_points.append(ChangePoint(
                    time_index=idx,
                    confidence_score=confidence,
                    change_type=change_type,
                    magnitude=magnitude,
                    description=f"Binary segmentation detected change at index {idx}"
                ))

        return change_points

    def _window_detection(self, series: np.ndarray, model: str, min_size: int,
                         jump: int, pen: float) -> List[ChangePoint]:
        """Window-based change point detection."""
        if pen is None:
            pen = np.log(len(series)) * np.var(series)

        algo = rpt.Window(width=min_size*2, model=model, min_size=min_size, jump=jump).fit(series)
        change_indices = algo.predict(pen=pen)

        change_points = []
        for idx in change_indices[:-1]:
            magnitude = self._estimate_change_magnitude(series, idx, min_size)
            change_type = self._estimate_change_type(series, idx, min_size)

            change_points.append(ChangePoint(
                time_index=idx,
                confidence_score=0.8,  # Window method generally reliable
                change_type=change_type,
                magnitude=magnitude,
                description=f"Window method detected change at index {idx}"
            ))

        return change_points

    def _kernel_detection(self, series: np.ndarray, model: str, min_size: int,
                         jump: int, pen: float) -> List[ChangePoint]:
        """Kernel change point detection."""
        if pen is None:
            pen = np.log(len(series)) * np.var(series)

        width = max(min_size, len(series) // 10)
        algo = rpt.KernelCPD(kernel="rbf", min_size=min_size, jump=jump).fit(series)
        change_indices = algo.predict(pen=pen)

        change_points = []
        for idx in change_indices[:-1]:
            magnitude = self._estimate_change_magnitude(series, idx, min_size)
            change_type = self._estimate_change_type(series, idx, min_size)

            change_points.append(ChangePoint(
                time_index=idx,
                confidence_score=0.7,  # Kernel methods can be less precise
                change_type=change_type,
                magnitude=magnitude,
                description=f"Kernel method detected change at index {idx}"
            ))

        return change_points

    def _dynamic_programming(self, series: np.ndarray, model: str, min_size: int,
                           jump: int, pen: float) -> List[ChangePoint]:
        """Dynamic programming change point detection."""
        algo = rpt.Dynp(model=model, min_size=min_size, jump=jump).fit(series)
        n_bkps = min(5, len(series) // (3 * min_size))
        change_indices = algo.predict(n_bkps=n_bkps)

        change_points = []
        for idx in change_indices[:-1]:
            magnitude = self._estimate_change_magnitude(series, idx, min_size)
            change_type = self._estimate_change_type(series, idx, min_size)

            change_points.append(ChangePoint(
                time_index=idx,
                confidence_score=0.9,  # Dynamic programming is usually accurate
                change_type=change_type,
                magnitude=magnitude,
                description=f"Dynamic programming detected change at index {idx}"
            ))

        return change_points

    def _classify_change_type(self, before_segment: np.ndarray,
                            after_segment: np.ndarray) -> str:
        """Classify the type of change between segments."""
        if len(before_segment) == 0 or len(after_segment) == 0:
            return "unknown"

        # Test for level change
        before_mean = np.mean(before_segment)
        after_mean = np.mean(after_segment)
        mean_change = abs(after_mean - before_mean)

        # Test for variance change
        before_var = np.var(before_segment, ddof=1) if len(before_segment) > 1 else 0
        after_var = np.var(after_segment, ddof=1) if len(after_segment) > 1 else 0
        var_ratio = max(after_var, before_var) / (min(after_var, before_var) + 1e-10)

        # Test for trend change
        if len(before_segment) > 2 and len(after_segment) > 2:
            before_trend = np.polyfit(range(len(before_segment)), before_segment, 1)[0]
            after_trend = np.polyfit(range(len(after_segment)), after_segment, 1)[0]
            trend_change = abs(after_trend - before_trend)
        else:
            trend_change = 0

        # Classify based on dominant change
        overall_std = np.std(np.concatenate([before_segment, after_segment]))

        if var_ratio > 2.0:
            return "variance"
        elif mean_change > 0.5 * overall_std:
            return "level"
        elif trend_change > 0.1 * overall_std:
            return "trend"
        else:
            return "structural"

    def _estimate_change_magnitude(self, series: np.ndarray, change_point: int,
                                 window_size: int) -> float:
        """Estimate magnitude of change at a change point."""
        before_start = max(0, change_point - window_size)
        after_end = min(len(series), change_point + window_size)

        before_segment = series[before_start:change_point]
        after_segment = series[change_point:after_end]

        if len(before_segment) > 0 and len(after_segment) > 0:
            return abs(np.mean(after_segment) - np.mean(before_segment))
        return 0.0

    def _estimate_change_type(self, series: np.ndarray, change_point: int,
                            window_size: int) -> str:
        """Estimate type of change at a change point."""
        before_start = max(0, change_point - window_size)
        after_end = min(len(series), change_point + window_size)

        before_segment = series[before_start:change_point]
        after_segment = series[change_point:after_end]

        return self._classify_change_type(before_segment, after_segment)

class TrendAnalyzer:
    """
    Analyzes temporal trends in network metrics with statistical testing.
    """

    def __init__(self):
        self.trend_methods = {
            'linear': self._linear_trend,
            'robust': self._robust_trend,
            'seasonal': self._seasonal_trend,
            'spline': self._spline_trend
        }

    def analyze_trend(self, time_series: np.ndarray, time_points: np.ndarray = None,
                     method: str = 'linear') -> TemporalTrend:
        """
        Analyze temporal trend in time series data.

        Args:
            time_series: Time series values
            time_points: Time points (default: sequential integers)
            method: Trend analysis method

        Returns:
            TemporalTrend object
        """
        if time_points is None:
            time_points = np.arange(len(time_series))

        # Remove NaN values
        valid_mask = ~np.isnan(time_series)
        if not np.any(valid_mask):
            return self._empty_trend()

        clean_time = time_points[valid_mask]
        clean_series = time_series[valid_mask]

        if len(clean_series) < 3:
            return self._empty_trend()

        logger.info(f"Analyzing temporal trend using {method} method")

        try:
            trend_func = self.trend_methods.get(method, self._linear_trend)
            return trend_func(clean_time, clean_series)
        except Exception as e:
            logger.error(f"Trend analysis failed: {e}")
            return self._empty_trend()

    def _linear_trend(self, time_points: np.ndarray, values: np.ndarray) -> TemporalTrend:
        """Linear trend analysis using OLS regression."""
        # Add constant for intercept
        X = sm.add_constant(time_points)
        model = sm.OLS(values, X).fit()

        slope = model.params[1]
        intercept = model.params[0]
        r_squared = model.rsquared
        p_value = model.pvalues[1]  # p-value for slope
        conf_int = model.conf_int().iloc[1]  # CI for slope

        # Classify trend
        if p_value < 0.05:
            if slope > 0:
                trend_type = "increasing"
            else:
                trend_type = "decreasing"
        else:
            trend_type = "stable"

        return TemporalTrend(
            slope=slope,
            intercept=intercept,
            r_squared=r_squared,
            p_value=p_value,
            confidence_interval=(conf_int[0], conf_int[1]),
            trend_type=trend_type,
            residuals=model.resid
        )

    def _robust_trend(self, time_points: np.ndarray, values: np.ndarray) -> TemporalTrend:
        """Robust trend analysis using robust linear regression."""
        try:
            X = sm.add_constant(time_points)
            model = sm.RLM(values, X, M=sm.robust.norms.HuberT()).fit()

            slope = model.params[1]
            intercept = model.params[0]

            # Calculate R-squared manually for robust regression
            y_pred = model.predict()
            ss_res = np.sum((values - y_pred) ** 2)
            ss_tot = np.sum((values - np.mean(values)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

            p_value = model.pvalues[1] if hasattr(model, 'pvalues') else np.nan
            conf_int = model.conf_int().iloc[1] if hasattr(model, 'conf_int') else (np.nan, np.nan)

            # Classify trend
            if not np.isnan(p_value) and p_value < 0.05:
                trend_type = "increasing" if slope > 0 else "decreasing"
            else:
                trend_type = "stable"

            return TemporalTrend(
                slope=slope,
                intercept=intercept,
                r_squared=r_squared,
                p_value=p_value,
                confidence_interval=conf_int,
                trend_type=trend_type,
                residuals=model.resid
            )

        except Exception as e:
            logger.warning(f"Robust trend analysis failed, falling back to linear: {e}")
            return self._linear_trend(time_points, values)

    def _seasonal_trend(self, time_points: np.ndarray, values: np.ndarray) -> TemporalTrend:
        """Seasonal trend decomposition."""
        try:
            # Try seasonal decomposition if enough data points
            if len(values) >= 8:  # Minimum for seasonal decomposition
                # Use UnobservedComponents for trend extraction
                model = UnobservedComponents(values, 'local linear trend')
                fit = model.fit(disp=0)

                trend_component = fit.level.smoothed
                slope = np.polyfit(time_points, trend_component, 1)[0]
                intercept = np.polyfit(time_points, trend_component, 1)[1]

                # Calculate R-squared for trend component
                trend_pred = slope * time_points + intercept
                ss_res = np.sum((values - trend_pred) ** 2)
                ss_tot = np.sum((values - np.mean(values)) ** 2)
                r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

                # Statistical significance test
                X = sm.add_constant(time_points)
                ols_model = sm.OLS(trend_component, X).fit()
                p_value = ols_model.pvalues[1]
                conf_int = ols_model.conf_int().iloc[1]

                trend_type = "increasing" if slope > 0 and p_value < 0.05 else \
                           "decreasing" if slope < 0 and p_value < 0.05 else "stable"

                return TemporalTrend(
                    slope=slope,
                    intercept=intercept,
                    r_squared=r_squared,
                    p_value=p_value,
                    confidence_interval=(conf_int[0], conf_int[1]),
                    trend_type=trend_type,
                    seasonal_component=trend_component,
                    residuals=values - trend_component
                )
            else:
                # Fall back to linear trend for short series
                return self._linear_trend(time_points, values)

        except Exception as e:
            logger.warning(f"Seasonal trend analysis failed, falling back to linear: {e}")
            return self._linear_trend(time_points, values)

    def _spline_trend(self, time_points: np.ndarray, values: np.ndarray) -> TemporalTrend:
        """Non-parametric spline trend analysis."""
        try:
            from scipy.interpolate import UnivariateSpline

            # Fit smoothing spline
            spline = UnivariateSpline(time_points, values, s=None)
            trend_values = spline(time_points)

            # Estimate overall slope
            slope = (trend_values[-1] - trend_values[0]) / (time_points[-1] - time_points[0])
            intercept = trend_values[0] - slope * time_points[0]

            # Calculate R-squared
            ss_res = np.sum((values - trend_values) ** 2)
            ss_tot = np.sum((values - np.mean(values)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

            # Use Mann-Kendall test for trend significance
            try:
                mk_tau, mk_p = stats.kendalltau(time_points, values)
                p_value = mk_p
            except:
                p_value = np.nan

            trend_type = "increasing" if slope > 0 and p_value < 0.05 else \
                        "decreasing" if slope < 0 and p_value < 0.05 else "stable"

            return TemporalTrend(
                slope=slope,
                intercept=intercept,
                r_squared=r_squared,
                p_value=p_value,
                confidence_interval=(np.nan, np.nan),  # Not available for spline
                trend_type=trend_type,
                residuals=values - trend_values
            )

        except Exception as e:
            logger.warning(f"Spline trend analysis failed, falling back to linear: {e}")
            return self._linear_trend(time_points, values)

    def _empty_trend(self) -> TemporalTrend:
        """Return empty trend result for error cases."""
        return TemporalTrend(
            slope=0.0,
            intercept=0.0,
            r_squared=0.0,
            p_value=1.0,
            confidence_interval=(0.0, 0.0),
            trend_type="stable"
        )

class LongitudinalNetworkAnalyzer:
    """
    Main class for comprehensive longitudinal network analysis.
    """

    def __init__(self):
        self.metrics_extractor = NetworkMetricsExtractor()
        self.change_detector = ChangePointDetector()
        self.trend_analyzer = TrendAnalyzer()

    def analyze_network_evolution(self, networks: List[nx.Graph],
                                 time_points: Optional[List[float]] = None,
                                 change_detection_method: str = 'pelt',
                                 trend_method: str = 'linear') -> NetworkEvolution:
        """
        Comprehensive analysis of network evolution over time.

        Args:
            networks: List of network objects in temporal order
            time_points: Time points for each network
            change_detection_method: Method for change point detection
            trend_method: Method for trend analysis

        Returns:
            NetworkEvolution object with comprehensive results
        """
        logger.info("Starting comprehensive longitudinal network analysis")

        if not networks:
            raise ValueError("No networks provided for analysis")

        if time_points is None:
            time_points = list(range(len(networks)))

        # Extract network metrics time series
        logger.info("Extracting network metrics time series")
        metrics_series = self.metrics_extractor.extract_metrics_time_series(networks, time_points)

        # Detect change points for each metric
        logger.info("Detecting change points")
        all_change_points = []
        change_points_by_metric = {}

        for metric_name, metric_values in metrics_series.items():
            if metric_name == 'time':
                continue

            metric_changes = self.change_detector.detect_change_points(
                metric_values, method=change_detection_method
            )

            change_points_by_metric[metric_name] = metric_changes
            all_change_points.extend(metric_changes)

        # Consolidate change points across metrics
        consolidated_changes = self._consolidate_change_points(all_change_points)

        # Analyze trends for each metric
        logger.info("Analyzing temporal trends")
        trends = {}
        time_array = np.array(time_points)

        for metric_name, metric_values in metrics_series.items():
            if metric_name == 'time':
                continue

            trend_result = self.trend_analyzer.analyze_trend(
                metric_values, time_array, method=trend_method
            )
            trends[metric_name] = trend_result

        # Identify stability and transition periods
        stability_periods, transition_periods = self._identify_periods(
            consolidated_changes, len(networks)
        )

        # Model diagnostics
        diagnostics = self._compute_diagnostics(metrics_series, trends, consolidated_changes)

        return NetworkEvolution(
            time_points=time_points,
            network_metrics=metrics_series,
            change_points=consolidated_changes,
            trends=trends,
            stability_periods=stability_periods,
            transition_periods=transition_periods,
            model_diagnostics=diagnostics
        )

    def _consolidate_change_points(self, change_points: List[ChangePoint],
                                 tolerance: int = 2) -> List[ChangePoint]:
        """Consolidate change points that are close in time across different metrics."""
        if not change_points:
            return []

        # Sort by time index
        sorted_changes = sorted(change_points, key=lambda x: x.time_index)

        consolidated = []
        current_group = [sorted_changes[0]]

        for cp in sorted_changes[1:]:
            # Check if this change point is close to the current group
            if abs(cp.time_index - current_group[-1].time_index) <= tolerance:
                current_group.append(cp)
            else:
                # Consolidate current group and start new group
                consolidated.append(self._merge_change_points(current_group))
                current_group = [cp]

        # Don't forget the last group
        if current_group:
            consolidated.append(self._merge_change_points(current_group))

        return consolidated

    def _merge_change_points(self, change_points: List[ChangePoint]) -> ChangePoint:
        """Merge multiple change points into a single consolidated change point."""
        if len(change_points) == 1:
            return change_points[0]

        # Use median time index
        time_indices = [cp.time_index for cp in change_points]
        median_time = int(np.median(time_indices))

        # Average confidence scores
        avg_confidence = np.mean([cp.confidence_score for cp in change_points])

        # Combine change types
        change_types = [cp.change_type for cp in change_points]
        unique_types = list(set(change_types))
        combined_type = "_".join(unique_types) if len(unique_types) > 1 else unique_types[0]

        # Average magnitude
        avg_magnitude = np.mean([cp.magnitude for cp in change_points])

        # Combined description
        description = f"Consolidated change at index {median_time} affecting {len(change_points)} metrics"

        return ChangePoint(
            time_index=median_time,
            confidence_score=avg_confidence,
            change_type=combined_type,
            magnitude=avg_magnitude,
            description=description
        )

    def _identify_periods(self, change_points: List[ChangePoint],
                         total_length: int) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
        """Identify stability and transition periods based on change points."""
        if not change_points:
            return [(0, total_length - 1)], []

        # Sort change points by time
        sorted_changes = sorted(change_points, key=lambda x: x.time_index)

        stability_periods = []
        transition_periods = []

        # Define transition window around each change point
        transition_window = 3  # ±3 time points around change

        current_start = 0
        for cp in sorted_changes:
            # Stability period before transition
            transition_start = max(0, cp.time_index - transition_window)
            if current_start < transition_start:
                stability_periods.append((current_start, transition_start - 1))

            # Transition period
            transition_end = min(total_length - 1, cp.time_index + transition_window)
            transition_periods.append((transition_start, transition_end))

            current_start = transition_end + 1

        # Final stability period
        if current_start < total_length:
            stability_periods.append((current_start, total_length - 1))

        return stability_periods, transition_periods

    def _compute_diagnostics(self, metrics_series: Dict[str, np.ndarray],
                           trends: Dict[str, TemporalTrend],
                           change_points: List[ChangePoint]) -> Dict[str, Any]:
        """Compute model diagnostics and quality metrics."""
        diagnostics = {}

        # Overall trend strength
        trend_strengths = []
        for metric_name, trend in trends.items():
            if not np.isnan(trend.r_squared):
                trend_strengths.append(trend.r_squared)

        diagnostics['mean_trend_strength'] = np.mean(trend_strengths) if trend_strengths else 0
        diagnostics['trend_strength_std'] = np.std(trend_strengths) if trend_strengths else 0

        # Change point density
        total_time_points = len(metrics_series.get('time', []))
        diagnostics['change_point_density'] = len(change_points) / total_time_points if total_time_points > 0 else 0

        # Stability assessment
        if change_points:
            change_intervals = []
            sorted_changes = sorted(change_points, key=lambda x: x.time_index)
            for i in range(len(sorted_changes) - 1):
                interval = sorted_changes[i+1].time_index - sorted_changes[i].time_index
                change_intervals.append(interval)

            diagnostics['mean_stability_duration'] = np.mean(change_intervals) if change_intervals else total_time_points
            diagnostics['stability_variability'] = np.std(change_intervals) if len(change_intervals) > 1 else 0
        else:
            diagnostics['mean_stability_duration'] = total_time_points
            diagnostics['stability_variability'] = 0

        # Trend consistency across metrics
        increasing_trends = sum(1 for trend in trends.values() if trend.trend_type == "increasing")
        decreasing_trends = sum(1 for trend in trends.values() if trend.trend_type == "decreasing")
        stable_trends = sum(1 for trend in trends.values() if trend.trend_type == "stable")

        total_trends = len(trends)
        if total_trends > 0:
            diagnostics['trend_consistency'] = max(increasing_trends, decreasing_trends, stable_trends) / total_trends
        else:
            diagnostics['trend_consistency'] = 0

        return diagnostics

    def plot_longitudinal_analysis(self, evolution: NetworkEvolution,
                                 output_dir: Path = None,
                                 metrics_to_plot: List[str] = None) -> Dict[str, Path]:
        """
        Generate comprehensive plots for longitudinal network analysis.

        Args:
            evolution: NetworkEvolution object
            output_dir: Directory to save plots
            metrics_to_plot: List of metrics to plot (default: key metrics)

        Returns:
            Dictionary mapping plot types to file paths
        """
        output_dir = output_dir or Path("outputs/longitudinal_plots")
        output_dir.mkdir(parents=True, exist_ok=True)

        if metrics_to_plot is None:
            metrics_to_plot = ['density', 'clustering', 'transitivity', 'diameter', 'modularity']

        plot_files = {}
        time_points = evolution.time_points

        # Multi-metric time series plot
        fig, axes = plt.subplots(len(metrics_to_plot), 1, figsize=(12, 3*len(metrics_to_plot)))
        if len(metrics_to_plot) == 1:
            axes = [axes]

        for i, metric in enumerate(metrics_to_plot):
            if metric in evolution.network_metrics:
                values = evolution.network_metrics[metric]
                axes[i].plot(time_points, values, 'b-', linewidth=2, marker='o', markersize=4)

                # Add trend line if available
                if metric in evolution.trends:
                    trend = evolution.trends[metric]
                    trend_line = trend.slope * np.array(time_points) + trend.intercept
                    axes[i].plot(time_points, trend_line, 'r--', alpha=0.7,
                               label=f'Trend (p={trend.p_value:.3f})')

                # Mark change points
                for cp in evolution.change_points:
                    if cp.time_index < len(time_points):
                        axes[i].axvline(time_points[cp.time_index], color='red', alpha=0.5,
                                      linestyle=':', label='Change Point' if i == 0 else "")

                axes[i].set_ylabel(metric.replace('_', ' ').title())
                axes[i].grid(True, alpha=0.3)
                if i == 0:
                    axes[i].legend()

        axes[-1].set_xlabel('Time')
        plt.suptitle('Network Evolution Over Time')
        plt.tight_layout()

        timeseries_file = output_dir / "network_evolution_timeseries.png"
        plt.savefig(timeseries_file, dpi=300, bbox_inches='tight')
        plt.close()
        plot_files['timeseries'] = timeseries_file

        # Change points summary plot
        if evolution.change_points:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

            # Change point timeline
            change_times = [time_points[cp.time_index] for cp in evolution.change_points
                           if cp.time_index < len(time_points)]
            change_magnitudes = [cp.magnitude for cp in evolution.change_points
                               if cp.time_index < len(time_points)]

            ax1.scatter(change_times, change_magnitudes, s=100, alpha=0.7, c='red')
            ax1.set_xlabel('Time')
            ax1.set_ylabel('Change Magnitude')
            ax1.set_title('Change Points Over Time')
            ax1.grid(True, alpha=0.3)

            # Change type distribution
            change_types = [cp.change_type for cp in evolution.change_points]
            unique_types, type_counts = np.unique(change_types, return_counts=True)

            ax2.bar(unique_types, type_counts, alpha=0.7, color='skyblue')
            ax2.set_xlabel('Change Type')
            ax2.set_ylabel('Count')
            ax2.set_title('Distribution of Change Types')
            ax2.tick_params(axis='x', rotation=45)

            plt.tight_layout()
            changepoints_file = output_dir / "change_points_analysis.png"
            plt.savefig(changepoints_file, dpi=300, bbox_inches='tight')
            plt.close()
            plot_files['change_points'] = changepoints_file

        logger.info(f"Longitudinal analysis plots saved to {output_dir}")
        return plot_files

    def generate_longitudinal_report(self, evolution: NetworkEvolution,
                                   output_file: Path = None) -> str:
        """
        Generate comprehensive longitudinal analysis report.

        Args:
            evolution: NetworkEvolution object
            output_file: Optional file to save report

        Returns:
            Report text
        """
        report_lines = []
        report_lines.append("# Longitudinal Network Analysis Report")
        report_lines.append("=" * 50)
        report_lines.append("")

        # Executive summary
        report_lines.append("## Executive Summary")
        report_lines.append("")
        report_lines.append(f"- **Analysis Period**: {len(evolution.time_points)} time points")
        report_lines.append(f"- **Change Points Detected**: {len(evolution.change_points)}")
        report_lines.append(f"- **Stability Periods**: {len(evolution.stability_periods)}")
        report_lines.append(f"- **Transition Periods**: {len(evolution.transition_periods)}")
        report_lines.append("")

        # Network evolution overview
        if evolution.model_diagnostics:
            diag = evolution.model_diagnostics
            report_lines.append("### Evolution Characteristics:")
            report_lines.append(f"- **Mean Trend Strength**: {diag.get('mean_trend_strength', 0):.4f}")
            report_lines.append(f"- **Change Point Density**: {diag.get('change_point_density', 0):.4f} per time unit")
            report_lines.append(f"- **Mean Stability Duration**: {diag.get('mean_stability_duration', 0):.1f} time units")
            report_lines.append(f"- **Trend Consistency**: {diag.get('trend_consistency', 0):.4f}")
            report_lines.append("")

        # Trends analysis
        report_lines.append("## Temporal Trends Analysis")
        report_lines.append("")
        report_lines.append("| Metric | Trend Type | Slope | R² | p-value |")
        report_lines.append("|--------|------------|-------|----|---------| ")

        for metric_name, trend in evolution.trends.items():
            report_lines.append(f"| {metric_name} | {trend.trend_type} | "
                              f"{trend.slope:.6f} | {trend.r_squared:.4f} | "
                              f"{trend.p_value:.4f} |")

        report_lines.append("")

        # Change points analysis
        if evolution.change_points:
            report_lines.append("## Change Points Analysis")
            report_lines.append("")
            report_lines.append("| Time Index | Time Value | Type | Magnitude | Confidence |")
            report_lines.append("|------------|------------|------|-----------|------------|")

            for cp in evolution.change_points:
                time_value = evolution.time_points[cp.time_index] if cp.time_index < len(evolution.time_points) else "N/A"
                report_lines.append(f"| {cp.time_index} | {time_value} | {cp.change_type} | "
                                  f"{cp.magnitude:.4f} | {cp.confidence_score:.4f} |")

            report_lines.append("")

        # Periods analysis
        report_lines.append("## Stability and Transition Periods")
        report_lines.append("")

        if evolution.stability_periods:
            report_lines.append("### Stability Periods:")
            for i, (start, end) in enumerate(evolution.stability_periods, 1):
                duration = end - start + 1
                start_time = evolution.time_points[start] if start < len(evolution.time_points) else start
                end_time = evolution.time_points[end] if end < len(evolution.time_points) else end
                report_lines.append(f"- **Period {i}**: Time {start_time} to {end_time} ({duration} time units)")

        if evolution.transition_periods:
            report_lines.append("")
            report_lines.append("### Transition Periods:")
            for i, (start, end) in enumerate(evolution.transition_periods, 1):
                duration = end - start + 1
                start_time = evolution.time_points[start] if start < len(evolution.time_points) else start
                end_time = evolution.time_points[end] if end < len(evolution.time_points) else end
                report_lines.append(f"- **Period {i}**: Time {start_time} to {end_time} ({duration} time units)")

        report_lines.append("")

        # Conclusions and recommendations
        report_lines.append("## Conclusions and Recommendations")
        report_lines.append("")

        # Generate insights based on analysis
        increasing_metrics = [name for name, trend in evolution.trends.items()
                            if trend.trend_type == "increasing" and trend.p_value < 0.05]
        decreasing_metrics = [name for name, trend in evolution.trends.items()
                            if trend.trend_type == "decreasing" and trend.p_value < 0.05]

        if increasing_metrics:
            report_lines.append(f"- **Increasing trends detected** in: {', '.join(increasing_metrics)}")

        if decreasing_metrics:
            report_lines.append(f"- **Decreasing trends detected** in: {', '.join(decreasing_metrics)}")

        if len(evolution.change_points) > 3:
            report_lines.append("- **High volatility period**: Multiple change points suggest dynamic network evolution")
        elif len(evolution.change_points) == 0:
            report_lines.append("- **Stable evolution**: No significant change points detected")

        report_lines.append("")

        report_text = "\n".join(report_lines)

        # Save report if requested
        if output_file:
            output_file.write_text(report_text)
            logger.info(f"Longitudinal analysis report saved to {output_file}")

        return report_text


# Example usage and testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Initialize analyzer
    analyzer = LongitudinalNetworkAnalyzer()

    # Generate synthetic network time series for testing
    np.random.seed(42)
    networks = []

    # Create evolving networks
    n_nodes = 50
    base_density = 0.1

    for t in range(20):
        # Add some trend and change points
        if t < 5:
            density = base_density + 0.01 * t  # Increasing
        elif t < 10:
            density = base_density + 0.05 + np.random.normal(0, 0.005)  # Stable
        elif t < 15:
            density = base_density + 0.05 - 0.01 * (t - 10)  # Decreasing
        else:
            density = base_density + np.random.normal(0, 0.01)  # Random

        # Create random network with target density
        p = density
        G = nx.erdos_renyi_graph(n_nodes, p)
        networks.append(G)

    # Perform analysis
    time_points = list(range(len(networks)))
    evolution = analyzer.analyze_network_evolution(networks, time_points)

    print("Longitudinal Network Analysis Results:")
    print("=" * 50)
    print(f"Change points detected: {len(evolution.change_points)}")
    print(f"Stability periods: {len(evolution.stability_periods)}")
    print(f"Transition periods: {len(evolution.transition_periods)}")

    # Print trends for key metrics
    key_metrics = ['density', 'clustering', 'transitivity']
    for metric in key_metrics:
        if metric in evolution.trends:
            trend = evolution.trends[metric]
            print(f"{metric}: {trend.trend_type} trend (p={trend.p_value:.4f})")

    # Generate plots
    plot_files = analyzer.plot_longitudinal_analysis(evolution)
    print(f"\nPlots generated: {list(plot_files.keys())}")

    # Generate report
    report = analyzer.generate_longitudinal_report(evolution)
    print("\nReport Preview:")
    print("=" * 30)
    print(report[:800] + "..." if len(report) > 800 else report)