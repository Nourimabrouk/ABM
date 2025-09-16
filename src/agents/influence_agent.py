"""
Influence Agent Implementation for Opinion Dynamics and Social Influence

This module implements specialized agents for modeling opinion dynamics, social influence,
and behavioral change processes within the ABM-RSiena integration framework.

Key Features:
- Bounded confidence opinion dynamics with heterogeneous confidence bounds
- Multiple influence mechanisms: conformity, compliance, identification, internalization
- Social pressure and minority/majority dynamics
- Opinion leadership and influence cascades
- Polarization and consensus formation processes

Author: Beta Agent - Implementation Specialist
Created: 2025-09-15
"""

import logging
import numpy as np
import networkx as nx
from typing import Dict, List, Optional, Set, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum

from .social_agent import SocialAgent, AgentAttributes, BehaviorState, InfluenceType

logger = logging.getLogger(__name__)


@dataclass
class OpinionDynamicsParameters:
    """Parameters governing opinion dynamics and influence processes."""
    # Bounded confidence parameters
    confidence_bound: float = 0.3  # Threshold for opinion influence
    confidence_decay: float = 0.01  # Rate of confidence bound change

    # Influence strength parameters
    base_influence_strength: float = 0.1  # Base rate of opinion change
    network_influence_multiplier: float = 1.5  # Effect of network position
    similarity_influence_multiplier: float = 2.0  # Effect of similarity

    # Social pressure parameters
    majority_pressure_threshold: float = 0.7  # Threshold for majority pressure
    minority_resistance_factor: float = 0.8  # Resistance when in minority
    extremism_amplification: float = 1.2  # Amplification for extreme opinions

    # Opinion persistence and volatility
    opinion_inertia: float = 0.85  # Resistance to opinion change
    volatility_factor: float = 0.1  # Random opinion fluctuation

    # Leadership and expertise effects
    expertise_weight: float = 2.0  # Weight given to expert opinions
    leadership_influence_bonus: float = 1.5  # Bonus for opinion leaders


@dataclass
class InfluenceEvent:
    """Record of an influence attempt between agents."""
    influencer_id: int
    target_id: int
    step: int
    influence_type: InfluenceType
    opinion_before: float
    opinion_after: float
    influence_strength: float
    success: bool
    mechanism: str = ""


class InfluenceAgent(SocialAgent):
    """
    Specialized agent for modeling opinion dynamics and social influence.

    Extends SocialAgent with sophisticated opinion dynamics including:
    - Bounded confidence with adaptive bounds
    - Multiple influence mechanisms
    - Opinion leadership dynamics
    - Social pressure and conformity effects
    - Polarization and consensus processes
    """

    def __init__(
        self,
        unique_id: int,
        model: 'ABMRSienaModel',
        attributes: AgentAttributes,
        behavior_variables: List[str] = None,
        opinion_params: Optional[OpinionDynamicsParameters] = None,
        initial_opinion: Optional[float] = None,
        is_opinion_leader: bool = False,
        expertise_domain: Optional[str] = None
    ):
        """
        Initialize influence agent with opinion dynamics capabilities.

        Args:
            unique_id: Unique agent identifier
            model: Reference to ABM model
            attributes: Agent attributes
            behavior_variables: Behavior variables to track
            opinion_params: Opinion dynamics parameters
            initial_opinion: Initial opinion value
            is_opinion_leader: Whether agent is an opinion leader
            expertise_domain: Domain of expertise (if any)
        """
        super().__init__(unique_id, model, attributes, behavior_variables)

        # Opinion dynamics parameters
        self.opinion_params = opinion_params or OpinionDynamicsParameters()

        # Opinion state
        if initial_opinion is not None:
            self.opinion = initial_opinion
        elif not hasattr(self, 'opinion'):
            self.opinion = np.random.uniform(-1, 1)

        self.behavior_state.opinion = self.opinion

        # Individual confidence bound (can vary by agent)
        base_bound = self.opinion_params.confidence_bound
        personality_adjustment = (
            self.attributes.openness * 0.3 +  # Open people more tolerant
            (1 - self.attributes.neuroticism) * 0.2 +  # Stable people more tolerant
            self.attributes.agreeableness * 0.1  # Agreeable people more tolerant
        )
        self.confidence_bound = base_bound * (1 + personality_adjustment)

        # Opinion leadership and expertise
        self.is_opinion_leader = is_opinion_leader
        self.expertise_domain = expertise_domain
        self.leadership_score = self._calculate_leadership_potential()

        # Opinion history and dynamics
        self.opinion_history: List[float] = [self.opinion]
        self.influence_events: List[InfluenceEvent] = []
        self.influence_received: List[InfluenceEvent] = []

        # Social pressure and conformity state
        self.perceived_majority_opinion: Optional[float] = None
        self.social_pressure_level: float = 0.0
        self.conformity_pressure: float = 0.0

        # Extremism and polarization tracking
        self.extremism_level = abs(self.opinion)
        self.polarization_tendency = self.attributes.conformity_tendency

        # Influence tracking
        self.influence_attempts_made = 0
        self.influence_successes = 0
        self.influence_received_count = 0
        self.opinion_leadership_score = 0.0

        logger.debug(f"Created influence agent {unique_id} with opinion {self.opinion:.3f}")

    def _calculate_leadership_potential(self) -> float:
        """Calculate agent's potential for opinion leadership."""
        leadership_score = (
            0.3 * self.attributes.extroversion +
            0.2 * self.attributes.leadership_tendency +
            0.2 * self.attributes.social_status +
            0.1 * self.attributes.academic_performance +
            0.1 * (1 - self.attributes.conformity_tendency) +  # Leaders less conformist
            0.1 * self.attributes.conscientiousness
        )
        return np.clip(leadership_score, 0, 1)

    def behavior_update_stage(self):
        """
        Enhanced behavior update with sophisticated opinion dynamics.
        """
        # Update perceived social environment
        self._update_social_perception()

        # Update opinion through various mechanisms
        self._update_opinion_bounded_confidence()
        self._apply_social_pressure()
        self._apply_extremism_dynamics()

        # Opinion leadership activities
        if self.is_opinion_leader or self.leadership_score > 0.7:
            self._attempt_influence_others()

        # Update other behaviors
        super().behavior_update_stage()

        # Update opinion-related metrics
        self._update_opinion_metrics()

    def _update_social_perception(self):
        """Update agent's perception of the social environment."""
        if not self.friends:
            return

        # Calculate perceived majority opinion among friends
        friend_opinions = []
        for friend_id in self.friends:
            friend = self.model.get_agent_by_id(friend_id)
            if friend and hasattr(friend, 'opinion'):
                friend_opinions.append(friend.opinion)

        if friend_opinions:
            self.perceived_majority_opinion = np.mean(friend_opinions)

            # Calculate social pressure based on opinion distribution
            opinion_variance = np.var(friend_opinions)
            opinion_distance = abs(self.opinion - self.perceived_majority_opinion)

            # Higher pressure when surrounded by uniform opposing opinions
            self.social_pressure_level = opinion_distance * (1 - opinion_variance)

            # Conformity pressure based on minority position
            similar_opinions = sum(1 for op in friend_opinions
                                 if abs(op - self.opinion) < self.confidence_bound)
            minority_ratio = similar_opinions / len(friend_opinions)
            self.conformity_pressure = 1 - minority_ratio

    def _update_opinion_bounded_confidence(self):
        """
        Update opinion using bounded confidence model with network effects.
        """
        if not self.friends:
            return

        # Collect friend opinions within confidence bound
        influential_opinions = []
        influence_weights = []

        for friend_id in self.friends:
            friend = self.model.get_agent_by_id(friend_id)
            if not friend or not hasattr(friend, 'opinion'):
                continue

            opinion_distance = abs(self.opinion - friend.opinion)

            # Only influenced by opinions within confidence bound
            if opinion_distance <= self.confidence_bound:
                influential_opinions.append(friend.opinion)

                # Calculate influence weight based on multiple factors
                weight = self._calculate_influence_weight(friend)
                influence_weights.append(weight)

        if not influential_opinions:
            return

        # Calculate weighted average of influential opinions
        weights = np.array(influence_weights)
        weights = weights / weights.sum()  # Normalize

        target_opinion = np.average(influential_opinions, weights=weights)

        # Calculate opinion change
        base_change = self.opinion_params.base_influence_strength * (target_opinion - self.opinion)

        # Apply personality and network modifiers
        personality_modifier = (
            (1 - self.attributes.conscientiousness) * 0.3 +  # Less stubborn
            self.attributes.agreeableness * 0.2 +  # More agreeable
            (1 - self.attributes.neuroticism) * 0.1  # More stable = more influence
        )

        network_modifier = min(2.0, len(self.friends) / 10)  # More friends = more influence

        final_change = base_change * personality_modifier * network_modifier

        # Apply inertia
        final_change *= (1 - self.opinion_params.opinion_inertia)

        # Update opinion
        old_opinion = self.opinion
        self.opinion = np.clip(self.opinion + final_change, -1, 1)

        # Record change
        if abs(self.opinion - old_opinion) > 0.01:
            self.opinion_changes += 1
            self.opinion_history.append(self.opinion)

            # Keep history manageable
            if len(self.opinion_history) > 50:
                self.opinion_history = self.opinion_history[-50:]

    def _calculate_influence_weight(self, influencer: 'SocialAgent') -> float:
        """
        Calculate how much weight to give to an influencer's opinion.

        Args:
            influencer: The agent attempting to influence

        Returns:
            Influence weight (higher = more influential)
        """
        weight = 1.0

        # Similarity effects (homophily)
        similarity = self.calculate_friendship_attraction(influencer)
        weight *= (1 + similarity * self.opinion_params.similarity_influence_multiplier)

        # Network position effects
        if hasattr(influencer, 'network_position'):
            centrality = influencer.network_position.degree / max(1, len(self.model.schedule.agents) * 0.1)
            weight *= (1 + centrality * self.opinion_params.network_influence_multiplier)

        # Opinion leadership effects
        if hasattr(influencer, 'is_opinion_leader') and influencer.is_opinion_leader:
            weight *= self.opinion_params.leadership_influence_bonus

        if hasattr(influencer, 'leadership_score'):
            weight *= (1 + influencer.leadership_score * 0.5)

        # Expertise effects
        if hasattr(influencer, 'expertise_domain') and influencer.expertise_domain:
            weight *= self.opinion_params.expertise_weight

        # Social status effects
        status_diff = influencer.attributes.social_status - self.attributes.social_status
        if status_diff > 0:  # Higher status = more influential
            weight *= (1 + status_diff * 0.2)

        return weight

    def _apply_social_pressure(self):
        """Apply social pressure effects on opinion change."""
        if self.social_pressure_level < self.opinion_params.majority_pressure_threshold:
            return

        if self.perceived_majority_opinion is None:
            return

        # Calculate pressure-induced opinion change
        pressure_direction = self.perceived_majority_opinion - self.opinion
        pressure_magnitude = self.social_pressure_level

        # Personality effects on pressure resistance
        resistance = (
            self.attributes.conscientiousness * 0.4 +  # Stubborn people resist
            (1 - self.attributes.agreeableness) * 0.3 +  # Disagreeable people resist
            self.attributes.extroversion * 0.2  # Extroverts resist more
        )

        # Apply minority resistance
        if self.conformity_pressure > 0.5:
            resistance *= self.opinion_params.minority_resistance_factor

        # Calculate final pressure effect
        pressure_effect = pressure_magnitude * (1 - resistance) * 0.1
        pressure_change = pressure_direction * pressure_effect

        # Apply change
        old_opinion = self.opinion
        self.opinion = np.clip(self.opinion + pressure_change, -1, 1)

        if abs(self.opinion - old_opinion) > 0.001:
            # Record pressure-induced change
            event = InfluenceEvent(
                influencer_id=-1,  # Social pressure (no specific influencer)
                target_id=self.unique_id,
                step=self.model.schedule.steps,
                influence_type=InfluenceType.CONFORMITY,
                opinion_before=old_opinion,
                opinion_after=self.opinion,
                influence_strength=pressure_effect,
                success=True,
                mechanism="social_pressure"
            )
            self.influence_received.append(event)

    def _apply_extremism_dynamics(self):
        """Apply extremism amplification and polarization dynamics."""
        # Update extremism level
        self.extremism_level = abs(self.opinion)

        # Extremism amplification for strong opinions
        if self.extremism_level > 0.7:
            amplification = (self.extremism_level - 0.7) * self.opinion_params.extremism_amplification

            # Amplify in direction of current opinion
            direction = 1 if self.opinion > 0 else -1
            extremism_change = amplification * direction * 0.01

            old_opinion = self.opinion
            self.opinion = np.clip(self.opinion + extremism_change, -1, 1)

            if abs(self.opinion - old_opinion) > 0.001:
                logger.debug(f"Agent {self.unique_id}: extremism amplification {old_opinion:.3f} -> {self.opinion:.3f}")

        # Random volatility
        if np.random.random() < self.opinion_params.volatility_factor:
            volatility_change = np.random.normal(0, 0.02)
            self.opinion = np.clip(self.opinion + volatility_change, -1, 1)

    def _attempt_influence_others(self):
        """Attempt to influence other agents' opinions."""
        if not self.friends:
            return

        # Opinion leaders attempt to influence more frequently
        influence_probability = 0.1
        if self.is_opinion_leader:
            influence_probability *= 2
        if self.leadership_score > 0.7:
            influence_probability *= 1.5

        if np.random.random() > influence_probability:
            return

        # Select target for influence attempt
        targets = list(self.friends)
        if not targets:
            return

        target_id = self.random.choice(targets)
        target = self.model.get_agent_by_id(target_id)

        if not target or not hasattr(target, 'opinion'):
            return

        self.influence_attempts_made += 1

        # Calculate influence success probability
        success_prob = self._calculate_influence_success_probability(target)

        if np.random.random() < success_prob:
            # Successful influence attempt
            old_opinion = target.opinion

            # Calculate opinion change
            influence_strength = self._calculate_directed_influence_strength(target)
            opinion_change = influence_strength * (self.opinion - target.opinion)

            target.opinion = np.clip(target.opinion + opinion_change, -1, 1)

            # Record successful influence
            event = InfluenceEvent(
                influencer_id=self.unique_id,
                target_id=target.unique_id,
                step=self.model.schedule.steps,
                influence_type=InfluenceType.IDENTIFICATION,
                opinion_before=old_opinion,
                opinion_after=target.opinion,
                influence_strength=influence_strength,
                success=True,
                mechanism="directed_influence"
            )

            self.influence_events.append(event)
            if hasattr(target, 'influence_received'):
                target.influence_received.append(event)
                target.influence_received_count += 1

            self.influence_successes += 1

            logger.debug(f"Agent {self.unique_id} influenced {target.unique_id}: {old_opinion:.3f} -> {target.opinion:.3f}")

    def _calculate_influence_success_probability(self, target: 'SocialAgent') -> float:
        """Calculate probability of successfully influencing target."""
        base_prob = 0.2

        # Opinion distance (closer opinions more likely to influence)
        opinion_distance = abs(self.opinion - target.opinion)
        if hasattr(target, 'confidence_bound'):
            distance_factor = max(0, 1 - opinion_distance / target.confidence_bound)
        else:
            distance_factor = max(0, 1 - opinion_distance / 0.3)
        base_prob *= distance_factor

        # Influencer characteristics
        base_prob *= (1 + self.leadership_score)
        base_prob *= (1 + self.attributes.extroversion * 0.5)
        base_prob *= (1 + self.attributes.social_status * 0.3)

        # Target susceptibility
        target_resistance = (
            target.attributes.conscientiousness * 0.4 +
            (1 - target.attributes.agreeableness) * 0.3 +
            target.attributes.extroversion * 0.2
        )
        base_prob *= (1 - target_resistance)

        # Relationship quality
        attraction = self.calculate_friendship_attraction(target)
        base_prob *= (1 + attraction * 0.5)

        return np.clip(base_prob, 0, 0.8)

    def _calculate_directed_influence_strength(self, target: 'SocialAgent') -> float:
        """Calculate strength of directed influence on target."""
        base_strength = 0.1

        # Influencer power
        base_strength *= (1 + self.leadership_score)
        if self.is_opinion_leader:
            base_strength *= 1.5

        # Network effects
        if hasattr(self, 'network_position'):
            centrality_effect = self.network_position.degree / max(1, len(self.model.schedule.agents) * 0.1)
            base_strength *= (1 + centrality_effect * 0.5)

        # Target susceptibility
        if hasattr(target, 'attributes'):
            susceptibility = (
                target.attributes.agreeableness * 0.3 +
                (1 - target.attributes.conscientiousness) * 0.3 +
                target.attributes.conformity_tendency * 0.4
            )
            base_strength *= (1 + susceptibility)

        return np.clip(base_strength, 0.01, 0.3)

    def _update_opinion_metrics(self):
        """Update opinion-related metrics and scores."""
        # Update opinion leadership score based on influence success
        if self.influence_attempts_made > 0:
            success_rate = self.influence_successes / self.influence_attempts_made
            self.opinion_leadership_score = (
                0.4 * success_rate +
                0.3 * self.leadership_score +
                0.2 * (len(self.friends) / max(1, len(self.model.schedule.agents) * 0.1)) +
                0.1 * self.extremism_level
            )

        # Update confidence bound based on experience
        if len(self.opinion_history) > 10:
            opinion_volatility = np.std(self.opinion_history[-10:])
            # More volatile agents become more tolerant
            bound_adjustment = opinion_volatility * 0.1
            self.confidence_bound = np.clip(
                self.confidence_bound + bound_adjustment * self.opinion_params.confidence_decay,
                0.1, 0.8
            )

    def get_opinion_state_summary(self) -> Dict[str, Any]:
        """Get comprehensive opinion dynamics state summary."""
        return {
            'opinion': self.opinion,
            'confidence_bound': self.confidence_bound,
            'extremism_level': self.extremism_level,
            'social_pressure': self.social_pressure_level,
            'conformity_pressure': self.conformity_pressure,
            'perceived_majority': self.perceived_majority_opinion,
            'is_leader': self.is_opinion_leader,
            'leadership_score': self.leadership_score,
            'opinion_leadership': self.opinion_leadership_score,
            'influence_attempts': self.influence_attempts_made,
            'influence_successes': self.influence_successes,
            'influence_received': self.influence_received_count,
            'opinion_changes': self.opinion_changes,
            'opinion_history': self.opinion_history[-10:] if len(self.opinion_history) > 10 else self.opinion_history
        }

    def calculate_opinion_similarity(self, other: 'InfluenceAgent') -> float:
        """Calculate opinion similarity with another agent."""
        if not hasattr(other, 'opinion'):
            return 0.0
        return 1 - abs(self.opinion - other.opinion) / 2  # Normalize to 0-1

    def is_opinion_compatible(self, other: 'InfluenceAgent') -> bool:
        """Check if opinion is compatible (within confidence bound)."""
        if not hasattr(other, 'opinion'):
            return False
        return abs(self.opinion - other.opinion) <= self.confidence_bound


def create_opinion_leaders(
    agents: List[InfluenceAgent],
    leadership_proportion: float = 0.1,
    selection_criteria: str = "hybrid"
) -> List[int]:
    """
    Select opinion leaders from agent population.

    Args:
        agents: List of influence agents
        leadership_proportion: Proportion of agents to designate as leaders
        selection_criteria: Method for selecting leaders ("random", "attributes", "hybrid")

    Returns:
        List of agent IDs designated as opinion leaders
    """
    n_leaders = int(len(agents) * leadership_proportion)

    if selection_criteria == "random":
        leader_ids = np.random.choice([a.unique_id for a in agents], n_leaders, replace=False)

    elif selection_criteria == "attributes":
        # Select based on leadership potential
        leadership_scores = [(a.unique_id, a.leadership_score) for a in agents]
        leadership_scores.sort(key=lambda x: x[1], reverse=True)
        leader_ids = [uid for uid, score in leadership_scores[:n_leaders]]

    elif selection_criteria == "hybrid":
        # Combine leadership potential with some randomness
        leadership_scores = np.array([a.leadership_score for a in agents])
        # Add noise to scores
        noisy_scores = leadership_scores + np.random.normal(0, 0.1, len(leadership_scores))
        top_indices = np.argsort(noisy_scores)[-n_leaders:]
        leader_ids = [agents[i].unique_id for i in top_indices]

    else:
        raise ValueError(f"Unknown selection criteria: {selection_criteria}")

    # Mark selected agents as opinion leaders
    for agent in agents:
        if agent.unique_id in leader_ids:
            agent.is_opinion_leader = True

    logger.info(f"Selected {len(leader_ids)} opinion leaders using {selection_criteria} criteria")
    return leader_ids


def analyze_opinion_polarization(agents: List[InfluenceAgent]) -> Dict[str, float]:
    """
    Analyze opinion polarization in agent population.

    Args:
        agents: List of influence agents

    Returns:
        Dictionary with polarization metrics
    """
    opinions = [a.opinion for a in agents if hasattr(a, 'opinion')]

    if not opinions:
        return {}

    opinions = np.array(opinions)

    # Basic statistics
    mean_opinion = np.mean(opinions)
    opinion_variance = np.var(opinions)
    opinion_std = np.std(opinions)

    # Polarization measures
    # 1. Esteban-Ray polarization index
    er_polarization = 0.0
    for i, op1 in enumerate(opinions):
        for j, op2 in enumerate(opinions):
            if i != j:
                er_polarization += abs(op1 - op2)
    er_polarization /= (len(opinions) * (len(opinions) - 1))

    # 2. Bimodality (distance from uniform distribution)
    hist, bins = np.histogram(opinions, bins=10, range=(-1, 1))
    hist_normalized = hist / hist.sum()
    uniform_prob = 1.0 / len(hist)
    bimodality = np.sum(np.abs(hist_normalized - uniform_prob))

    # 3. Extremism index (proportion with |opinion| > 0.7)
    extremism_index = np.mean(np.abs(opinions) > 0.7)

    # 4. Coverage (range of opinions)
    coverage = np.max(opinions) - np.min(opinions)

    return {
        'mean_opinion': mean_opinion,
        'opinion_variance': opinion_variance,
        'opinion_std': opinion_std,
        'er_polarization': er_polarization,
        'bimodality': bimodality,
        'extremism_index': extremism_index,
        'coverage': coverage,
        'n_agents': len(opinions)
    }


if __name__ == "__main__":
    # Example usage and testing
    logging.basicConfig(level=logging.INFO)

    # Create test agents
    from .social_agent import create_realistic_agent_attributes

    test_agents = []
    for i in range(20):
        attributes = create_realistic_agent_attributes(i)
        # Mock model for testing
        class MockModel:
            def __init__(self):
                self.schedule = type('Schedule', (), {'steps': 0})()
            def get_agent_by_id(self, agent_id):
                return None

        model = MockModel()
        agent = InfluenceAgent(i, model, attributes, ['opinion'])
        test_agents.append(agent)

    # Select opinion leaders
    leader_ids = create_opinion_leaders(test_agents, 0.2, "hybrid")
    print(f"Opinion leaders: {leader_ids}")

    # Analyze initial polarization
    polarization = analyze_opinion_polarization(test_agents)
    print(f"Initial polarization metrics: {polarization}")

    # Show opinion distribution
    opinions = [a.opinion for a in test_agents]
    print(f"Opinion range: {min(opinions):.3f} to {max(opinions):.3f}")
    print(f"Opinion mean: {np.mean(opinions):.3f}, std: {np.std(opinions):.3f}")