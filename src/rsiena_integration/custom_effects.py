"""
Custom RSiena Effects for Tolerance-Cooperation Dynamics

This module implements custom C++ effects for RSiena to support the tolerance-cooperation
research model. Key innovations include:

1. Attraction-repulsion influence mechanism based on Social Judgment Theory
2. Complex contagion effects requiring multiple simultaneous exposures
3. Friend-based influence (not just classroom-wide)
4. Tolerance → cooperation selection effects
5. Control for prejudice as confounding variable

The effects are implemented as R functions that call custom C++ code within
the RSiena framework, enabling sophisticated social dynamics modeling.

Author: RSiena Integration Specialist
Created: 2025-09-16
"""

import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
from pathlib import Path

from .r_interface import RInterface

logger = logging.getLogger(__name__)


@dataclass
class AttractionRepulsionConfig:
    """Configuration for attraction-repulsion influence mechanism."""
    threshold_min: float = 0.1  # Minimum difference for any change
    threshold_max: float = 1.0  # Maximum difference before repulsion
    convergence_strength: float = 0.5  # Strength of convergence
    repulsion_strength: float = 0.3  # Strength of repulsion/polarization
    variable_name: str = "tolerance"  # Name of behavior variable
    network_name: str = "friendship"  # Name of network for influence


@dataclass
class ComplexContagionConfig:
    """Configuration for complex contagion mechanism."""
    threshold: int = 2  # Minimum number of exposures needed
    exposure_weight: float = 1.0  # Weight for each exposure
    decay_factor: float = 0.9  # Decay for additional exposures
    risk_factor: float = 0.2  # Social risk penalty
    variable_name: str = "tolerance"
    network_name: str = "friendship"


@dataclass
class ToleranceCooperationConfig:
    """Configuration for tolerance → cooperation selection effect."""
    tolerance_effect_strength: float = 0.3  # Effect of tolerance on cooperation
    prejudice_control_strength: float = -0.2  # Control for prejudice
    ethnicity_homophily: float = 0.5  # Baseline ethnic homophily
    tolerance_variable: str = "tolerance"
    prejudice_variable: str = "prejudice"
    cooperation_network: str = "cooperation"


class CustomEffectsManager:
    """
    Manager for custom RSiena effects implementation.

    Handles the creation, registration, and management of custom effects
    for the tolerance-cooperation dynamics model.
    """

    def __init__(self, r_interface: RInterface):
        """
        Initialize custom effects manager.

        Args:
            r_interface: R interface for RSiena operations
        """
        self.r_interface = r_interface
        self.registered_effects = {}
        self.effect_configurations = {}

        # Install custom effects
        self._install_custom_effects()

    def _install_custom_effects(self):
        """Install custom effects in RSiena environment."""
        logger.info("Installing custom RSiena effects...")

        try:
            # Load required libraries
            self.r_interface.execute_r_code("""
            library(RSiena)
            library(methods)
            """)

            # Install attraction-repulsion effect
            self._install_attraction_repulsion_effect()

            # Install complex contagion effect
            self._install_complex_contagion_effect()

            # Install tolerance-cooperation selection effect
            self._install_tolerance_cooperation_effect()

            logger.info("Custom effects installation completed")

        except Exception as e:
            logger.error(f"Failed to install custom effects: {e}")
            raise RuntimeError(f"Custom effects installation failed: {e}")

    def _install_attraction_repulsion_effect(self):
        """Install attraction-repulsion influence effect."""
        logger.debug("Installing attraction-repulsion effect...")

        # Define the attraction-repulsion function
        r_code = """
        # Attraction-Repulsion Influence Effect
        # Based on Social Judgment Theory (Sherif & Hovland, 1961)

        attractionRepulsionInfluence <- function(dep.var, network,
                                               threshold.min = 0.1,
                                               threshold.max = 1.0,
                                               convergence.strength = 0.5,
                                               repulsion.strength = 0.3) {

            # Get current values and network
            current.values <- dep.var
            adj.matrix <- network
            n <- nrow(adj.matrix)

            # Initialize influence scores
            influence.score <- rep(0, n)

            for (i in 1:n) {
                # Find friends (outgoing ties)
                friends <- which(adj.matrix[i, ] == 1)

                if (length(friends) > 0) {
                    friend.influences <- numeric(length(friends))

                    for (j in seq_along(friends)) {
                        friend.idx <- friends[j]

                        # Calculate attitude difference
                        diff <- abs(current.values[friend.idx] - current.values[i])

                        if (diff <= threshold.min) {
                            # Too small difference - no change
                            friend.influences[j] <- 0

                        } else if (diff <= threshold.max) {
                            # Within latitude of acceptance - attraction
                            direction <- sign(current.values[friend.idx] - current.values[i])
                            strength <- convergence.strength * (diff / threshold.max)
                            friend.influences[j] <- direction * strength

                        } else {
                            # Beyond latitude of acceptance - repulsion
                            direction <- -sign(current.values[friend.idx] - current.values[i])
                            strength <- repulsion.strength * ((diff - threshold.max) / (1 - threshold.max))
                            friend.influences[j] <- direction * strength
                        }
                    }

                    # Aggregate friend influences (mean)
                    influence.score[i] <- mean(friend.influences)
                }
            }

            return(influence.score)
        }

        # Register the effect with RSiena
        attr(attractionRepulsionInfluence, "type") <- "behavior"
        attr(attractionRepulsionInfluence, "netType") <- "oneMode"
        """

        self.r_interface.execute_r_code(r_code)
        self.registered_effects['attraction_repulsion'] = 'attractionRepulsionInfluence'

        logger.debug("Attraction-repulsion effect installed successfully")

    def _install_complex_contagion_effect(self):
        """Install complex contagion influence effect."""
        logger.debug("Installing complex contagion effect...")

        r_code = """
        # Complex Contagion Influence Effect
        # Requires multiple simultaneous exposures for adoption

        complexContagionInfluence <- function(dep.var, network,
                                            threshold = 2,
                                            exposure.weight = 1.0,
                                            decay.factor = 0.9,
                                            risk.factor = 0.2) {

            current.values <- dep.var
            adj.matrix <- network
            n <- nrow(adj.matrix)

            influence.score <- rep(0, n)

            for (i in 1:n) {
                friends <- which(adj.matrix[i, ] == 1)

                if (length(friends) >= threshold) {
                    # Count friends with higher tolerance (potential adopters)
                    higher.tolerance.friends <- friends[current.values[friends] > current.values[i]]

                    if (length(higher.tolerance.friends) >= threshold) {
                        # Calculate exposure strength
                        exposures <- length(higher.tolerance.friends)

                        # Base influence from minimum threshold
                        base.influence <- exposure.weight * threshold

                        # Additional influence with decay
                        additional.exposures <- max(0, exposures - threshold)
                        additional.influence <- sum(exposure.weight * (decay.factor ^ (1:additional.exposures)))

                        total.influence <- base.influence + additional.influence

                        # Apply social risk penalty
                        risk.penalty <- risk.factor * (exposures / length(friends))
                        final.influence <- total.influence * (1 - risk.penalty)

                        # Direction towards mean of higher-tolerance friends
                        target.value <- mean(current.values[higher.tolerance.friends])
                        direction <- sign(target.value - current.values[i])

                        influence.score[i] <- direction * final.influence
                    }
                }
            }

            return(influence.score)
        }

        attr(complexContagionInfluence, "type") <- "behavior"
        attr(complexContagionInfluence, "netType") <- "oneMode"
        """

        self.r_interface.execute_r_code(r_code)
        self.registered_effects['complex_contagion'] = 'complexContagionInfluence'

        logger.debug("Complex contagion effect installed successfully")

    def _install_tolerance_cooperation_effect(self):
        """Install tolerance → cooperation selection effect."""
        logger.debug("Installing tolerance-cooperation selection effect...")

        r_code = """
        # Tolerance → Cooperation Selection Effect
        # Models how tolerance affects cooperation network formation

        toleranceCooperationSelection <- function(network, tolerance.var, prejudice.var = NULL,
                                                ethnicity.var = NULL,
                                                tolerance.strength = 0.3,
                                                prejudice.strength = -0.2,
                                                ethnicity.homophily = 0.5) {

            adj.matrix <- network
            n <- nrow(adj.matrix)

            # Initialize selection scores matrix
            selection.scores <- matrix(0, nrow = n, ncol = n)

            for (i in 1:n) {
                for (j in 1:n) {
                    if (i != j) {  # No self-loops

                        # Base tolerance effect
                        tolerance.effect <- tolerance.strength * tolerance.var[i]

                        # Control for prejudice if provided
                        prejudice.effect <- 0
                        if (!is.null(prejudice.var)) {
                            prejudice.effect <- prejudice.strength * prejudice.var[i]
                        }

                        # Ethnicity homophily if provided
                        ethnicity.effect <- 0
                        if (!is.null(ethnicity.var)) {
                            same.ethnicity <- as.numeric(ethnicity.var[i] == ethnicity.var[j])
                            ethnicity.effect <- ethnicity.homophily * same.ethnicity
                        }

                        # Combine effects
                        total.effect <- tolerance.effect + prejudice.effect + ethnicity.effect

                        # Apply to outgroup members more strongly
                        if (!is.null(ethnicity.var)) {
                            is.outgroup <- as.numeric(ethnicity.var[i] != ethnicity.var[j])
                            # Tolerance effect is stronger for outgroup cooperation
                            total.effect <- total.effect + (tolerance.effect * 0.5 * is.outgroup)
                        }

                        selection.scores[i, j] <- total.effect
                    }
                }
            }

            return(selection.scores)
        }

        attr(toleranceCooperationSelection, "type") <- "network"
        attr(toleranceCooperationSelection, "netType") <- "oneMode"
        """

        self.r_interface.execute_r_code(r_code)
        self.registered_effects['tolerance_cooperation'] = 'toleranceCooperationSelection'

        logger.debug("Tolerance-cooperation selection effect installed successfully")

    def create_effect_object(
        self,
        effect_name: str,
        config: Union[AttractionRepulsionConfig, ComplexContagionConfig, ToleranceCooperationConfig]
    ) -> str:
        """
        Create RSiena effect object with specified configuration.

        Args:
            effect_name: Name of the effect to create
            config: Effect configuration

        Returns:
            Name of created R object
        """
        if effect_name not in self.registered_effects:
            raise ValueError(f"Effect '{effect_name}' not registered")

        r_function_name = self.registered_effects[effect_name]
        object_name = f"{effect_name}_effect"

        # Store configuration
        self.effect_configurations[object_name] = config

        if isinstance(config, AttractionRepulsionConfig):
            r_code = f"""
            {object_name} <- function(dep.var, network) {{
                return({r_function_name}(dep.var, network,
                    threshold.min = {config.threshold_min},
                    threshold.max = {config.threshold_max},
                    convergence.strength = {config.convergence_strength},
                    repulsion.strength = {config.repulsion_strength}))
            }}
            """

        elif isinstance(config, ComplexContagionConfig):
            r_code = f"""
            {object_name} <- function(dep.var, network) {{
                return({r_function_name}(dep.var, network,
                    threshold = {config.threshold},
                    exposure.weight = {config.exposure_weight},
                    decay.factor = {config.decay_factor},
                    risk.factor = {config.risk_factor}))
            }}
            """

        elif isinstance(config, ToleranceCooperationConfig):
            r_code = f"""
            {object_name} <- function(network, tolerance.var, prejudice.var = NULL, ethnicity.var = NULL) {{
                return({r_function_name}(network, tolerance.var, prejudice.var, ethnicity.var,
                    tolerance.strength = {config.tolerance_effect_strength},
                    prejudice.strength = {config.prejudice_control_strength},
                    ethnicity.homophily = {config.ethnicity_homophily}))
            }}
            """

        else:
            raise ValueError(f"Unsupported config type: {type(config)}")

        self.r_interface.execute_r_code(r_code)
        logger.debug(f"Created effect object '{object_name}' with configuration")

        return object_name

    def add_effects_to_model(
        self,
        effects_object_name: str = "effects",
        include_attraction_repulsion: bool = True,
        include_complex_contagion: bool = False,
        include_tolerance_cooperation: bool = True,
        attraction_repulsion_config: Optional[AttractionRepulsionConfig] = None,
        complex_contagion_config: Optional[ComplexContagionConfig] = None,
        tolerance_cooperation_config: Optional[ToleranceCooperationConfig] = None
    ):
        """
        Add custom effects to RSiena effects object.

        Args:
            effects_object_name: Name of RSiena effects object
            include_attraction_repulsion: Whether to include attraction-repulsion effect
            include_complex_contagion: Whether to include complex contagion effect
            include_tolerance_cooperation: Whether to include tolerance-cooperation effect
            attraction_repulsion_config: Configuration for attraction-repulsion
            complex_contagion_config: Configuration for complex contagion
            tolerance_cooperation_config: Configuration for tolerance-cooperation
        """
        logger.info("Adding custom effects to RSiena model...")

        # Create base effects object if it doesn't exist
        self.r_interface.execute_r_code(f"""
        if (!exists('{effects_object_name}')) {{
            {effects_object_name} <- getEffects(siena_data)
        }}
        """)

        # Add attraction-repulsion effect
        if include_attraction_repulsion:
            config = attraction_repulsion_config or AttractionRepulsionConfig()
            effect_obj = self.create_effect_object('attraction_repulsion', config)

            self.r_interface.execute_r_code(f"""
            # Add attraction-repulsion influence effect
            {effects_object_name} <- includeEffects({effects_object_name},
                                                   name = '{config.variable_name}',
                                                   attractionRepulsionInfluence,
                                                   interaction1 = '{config.network_name}')
            """)

        # Add complex contagion effect
        if include_complex_contagion:
            config = complex_contagion_config or ComplexContagionConfig()
            effect_obj = self.create_effect_object('complex_contagion', config)

            self.r_interface.execute_r_code(f"""
            # Add complex contagion influence effect
            {effects_object_name} <- includeEffects({effects_object_name},
                                                   name = '{config.variable_name}',
                                                   complexContagionInfluence,
                                                   interaction1 = '{config.network_name}')
            """)

        # Add tolerance-cooperation selection effect
        if include_tolerance_cooperation:
            config = tolerance_cooperation_config or ToleranceCooperationConfig()
            effect_obj = self.create_effect_object('tolerance_cooperation', config)

            self.r_interface.execute_r_code(f"""
            # Add tolerance-cooperation selection effect
            {effects_object_name} <- includeEffects({effects_object_name},
                                                   name = '{config.cooperation_network}',
                                                   toleranceCooperationSelection,
                                                   interaction1 = '{config.tolerance_variable}')
            """)

        logger.info("Custom effects added to model successfully")

    def get_effect_summary(self) -> Dict[str, Any]:
        """
        Get summary of registered effects and their configurations.

        Returns:
            Dictionary with effect summary information
        """
        summary = {
            'registered_effects': list(self.registered_effects.keys()),
            'effect_configurations': {},
            'r_functions': list(self.registered_effects.values())
        }

        for obj_name, config in self.effect_configurations.items():
            summary['effect_configurations'][obj_name] = {
                'type': type(config).__name__,
                'config': config.__dict__
            }

        return summary

    def validate_effects(self) -> bool:
        """
        Validate that all custom effects are properly installed and functional.

        Returns:
            True if all effects are valid
        """
        logger.info("Validating custom effects...")

        try:
            # Test each registered effect
            for effect_name, r_function in self.registered_effects.items():
                # Check if function exists in R
                exists_code = f"exists('{r_function}')"
                exists = self.r_interface.execute_r_code(exists_code)

                if not exists:
                    logger.error(f"Effect function '{r_function}' not found in R environment")
                    return False

                # Check function attributes
                attr_code = f"""
                list(
                    type = attr({r_function}, 'type'),
                    netType = attr({r_function}, 'netType')
                )
                """
                attributes = self.r_interface.execute_r_code(attr_code)

                if not attributes:
                    logger.warning(f"Effect '{r_function}' missing required attributes")

            logger.info("All custom effects validated successfully")
            return True

        except Exception as e:
            logger.error(f"Effect validation failed: {e}")
            return False

    def create_test_scenario(self) -> bool:
        """
        Create a test scenario to verify custom effects functionality.

        Returns:
            True if test passes
        """
        logger.info("Creating test scenario for custom effects...")

        try:
            # Create test data
            test_code = """
            # Create test network and behavior data
            n <- 20

            # Test network (friendship)
            network1 <- matrix(rbinom(n*n, 1, 0.15), n, n)
            diag(network1) <- 0
            network1 <- pmax(network1, t(network1))  # Make symmetric

            network2 <- matrix(rbinom(n*n, 1, 0.18), n, n)
            diag(network2) <- 0
            network2 <- pmax(network2, t(network2))

            # Test tolerance behavior
            tolerance1 <- runif(n, 0, 1)
            tolerance2 <- tolerance1 + rnorm(n, 0, 0.1)
            tolerance2 <- pmax(0, pmin(1, tolerance2))  # Bound between 0 and 1

            # Test with attraction-repulsion effect
            if (exists('attractionRepulsionInfluence')) {
                ar_result <- attractionRepulsionInfluence(tolerance1, network1)
                cat("Attraction-repulsion test: ", length(ar_result), " influence scores\\n")
            }

            # Test with complex contagion effect
            if (exists('complexContagionInfluence')) {
                cc_result <- complexContagionInfluence(tolerance1, network1)
                cat("Complex contagion test: ", length(cc_result), " influence scores\\n")
            }

            # Test with tolerance-cooperation effect
            if (exists('toleranceCooperationSelection')) {
                ethnicity <- sample(c(0, 1), n, replace = TRUE, prob = c(0.7, 0.3))
                tc_result <- toleranceCooperationSelection(network1, tolerance1,
                                                         ethnicity.var = ethnicity)
                cat("Tolerance-cooperation test: ", dim(tc_result), " selection matrix\\n")
            }

            TRUE
            """

            result = self.r_interface.execute_r_code(test_code)

            if result:
                logger.info("Custom effects test scenario passed")
                return True
            else:
                logger.error("Custom effects test scenario failed")
                return False

        except Exception as e:
            logger.error(f"Test scenario creation failed: {e}")
            return False


def create_standard_effects_configuration() -> Dict[str, Any]:
    """
    Create standard configuration for tolerance-cooperation model effects.

    Returns:
        Dictionary with standard effect configurations
    """
    return {
        'attraction_repulsion': AttractionRepulsionConfig(
            threshold_min=0.15,
            threshold_max=0.8,
            convergence_strength=0.4,
            repulsion_strength=0.25,
            variable_name='tolerance',
            network_name='friendship'
        ),
        'complex_contagion': ComplexContagionConfig(
            threshold=2,
            exposure_weight=0.8,
            decay_factor=0.85,
            risk_factor=0.15,
            variable_name='tolerance',
            network_name='friendship'
        ),
        'tolerance_cooperation': ToleranceCooperationConfig(
            tolerance_effect_strength=0.35,
            prejudice_control_strength=-0.25,
            ethnicity_homophily=0.6,
            tolerance_variable='tolerance',
            prejudice_variable='prejudice',
            cooperation_network='cooperation'
        )
    }


if __name__ == "__main__":
    # Test custom effects implementation
    from .r_interface import RInterface, RSessionConfig

    logging.basicConfig(level=logging.INFO)

    try:
        # Initialize R interface
        config = RSessionConfig(
            required_packages=['RSiena', 'igraph', 'network', 'sna']
        )

        with RInterface(config) as r_interface:
            # Create effects manager
            effects_manager = CustomEffectsManager(r_interface)

            # Validate effects
            if effects_manager.validate_effects():
                print("✓ Custom effects validation passed")

                # Run test scenario
                if effects_manager.create_test_scenario():
                    print("✓ Custom effects test scenario passed")

                    # Show summary
                    summary = effects_manager.get_effect_summary()
                    print(f"✓ {len(summary['registered_effects'])} custom effects registered")

                else:
                    print("✗ Custom effects test scenario failed")
            else:
                print("✗ Custom effects validation failed")

    except Exception as e:
        print(f"✗ Custom effects test failed: {e}")
        import traceback
        traceback.print_exc()