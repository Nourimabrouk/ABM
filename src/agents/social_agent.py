"""
Social Agent Implementation for ABM-RSiena Integration

This module implements sophisticated social agents capable of network-aware behaviors,
supporting the core ABM-RSiena integration framework. Agents are designed to model
realistic social dynamics including friendship formation, opinion evolution, and
group membership dynamics.

Key Features:
- Network-aware decision making with homophily and transitivity
- Opinion dynamics with social influence mechanisms
- Multi-stage behavioral activation (formation, update, dissolution)
- Statistical tracking for RSiena parameter estimation
- Configurable behavioral parameters and learning mechanisms

Author: Beta Agent - Implementation Specialist
Created: 2025-09-15
"""

import logging
import numpy as np
import networkx as nx
from typing import Dict, List, Optional, Set, Any, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import random

import mesa
from mesa import Agent

logger = logging.getLogger(__name__)


@dataclass
class AgentAttributes:
    """
    Comprehensive agent attributes supporting social network dynamics.

    These attributes influence network formation, opinion dynamics, and
    behavioral change in the integrated ABM-RSiena model.
    """
    # Demographic attributes
    age: float
    gender: str  # 'M' or 'F'
    socioeconomic_status: float  # Standardized (-3 to 3)
    education_level: int = 1  # 1-5 scale

    # Personality and cognitive attributes
    extroversion: float = 0.5  # 0-1 scale
    openness: float = 0.5  # 0-1 scale
    conscientiousness: float = 0.5  # 0-1 scale
    agreeableness: float = 0.5  # 0-1 scale
    neuroticism: float = 0.5  # 0-1 scale

    # Social and academic attributes
    academic_performance: float = 0.0  # Standardized
    social_status: float = 0.0  # Perceived social status
    leadership_tendency: float = 0.3  # 0-1 scale

    # Behavioral attributes
    risk_tolerance: float = 0.5  # 0-1 scale
    conformity_tendency: float = 0.5  # 0-1 scale
    innovation_adoption: float = 0.5  # 0-1 scale


class InfluenceType(Enum):
    """Types of social influence mechanisms."""
    CONFORMITY = "conformity"
    COMPLIANCE = "compliance"
    IDENTIFICATION = "identification"
    INTERNALIZATION = "internalization"


@dataclass
class NetworkPosition:
    """Container for agent's network position metrics."""
    degree: int = 0
    clustering_coefficient: float = 0.0
    betweenness_centrality: float = 0.0
    closeness_centrality: float = 0.0
    eigenvector_centrality: float = 0.0
    local_transitivity: float = 0.0
    structural_holes: float = 0.0


@dataclass
class BehaviorState:
    """Container for agent's behavioral state variables."""
    opinion: Optional[float] = None  # -1 to 1 continuous opinion
    attitude: Optional[float] = None  # General attitude measure
    behavior_adoption: Optional[bool] = None  # Binary behavior adoption
    activity_level: float = 0.5  # General activity level
    influence_susceptibility: float = 0.5  # How susceptible to influence


class SocialAgent(Agent):
    """
    Advanced social agent with network-aware behaviors and RSiena integration.

    This agent class implements sophisticated social dynamics including:
    - Multi-stage behavioral activation (network formation, behavior update, dissolution)
    - Homophily-based friendship formation with transitivity effects
    - Opinion dynamics with bounded confidence and social influence
    - Empirically-grounded parameter updates from RSiena estimation
    - Comprehensive behavioral tracking for statistical validation
    """

    def __init__(
        self,
        unique_id: int,
        model: 'ABMRSienaModel',
        attributes: AgentAttributes,
        behavior_variables: List[str] = None,
        initial_network_position: Optional[NetworkPosition] = None
    ):
        """
        Initialize sophisticated social agent.

        Args:
            unique_id: Unique agent identifier
            model: Reference to the ABM model
            attributes: Agent demographic and personality attributes
            behavior_variables: List of behavior variables to track
            initial_network_position: Optional initial network position
        """
        super().__init__(unique_id, model)

        self.attributes = attributes
        self.behavior_variables = behavior_variables or ['opinion']

        # Network relationships
        self.friends: Set[int] = set()
        self.acquaintances: Set[int] = set()
        self.family_ties: Set[int] = set()
        self.work_colleagues: Set[int] = set()

        # Network position tracking
        self.network_position = initial_network_position or NetworkPosition()

        # Behavioral state
        self.behavior_state = BehaviorState()
        self._initialize_behaviors()

        # Social dynamics tracking
        self.friendship_attempts = 0
        self.friendship_successes = 0
        self.friendship_rejections = 0
        self.influence_attempts = 0
        self.influence_successes = 0
        self.opinion_changes = 0

        # Memory and learning
        self.interaction_history: Dict[int, List[Dict]] = {}
        self.influence_memory: List[Dict] = []
        self.satisfaction_history: List[float] = []

        # Current state
        self.social_satisfaction = 0.5
        self.stress_level = 0.0
        self.energy_level = 1.0

        # Behavioral parameters (updated by RSiena estimation)
        self.behavioral_parameters = self._initialize_behavioral_parameters()

        logger.debug(f"Created social agent {unique_id} with attributes: {attributes}")

    def _initialize_behaviors(self):
        """Initialize behavioral state variables."""
        # Initialize opinion based on attributes
        if 'opinion' in self.behavior_variables:
            # Opinion influenced by personality and demographics
            base_opinion = np.random.uniform(-1, 1)

            # Adjust based on personality
            if self.attributes.openness > 0.7:
                base_opinion += np.random.normal(0, 0.3)  # More variable
            if self.attributes.conformity_tendency > 0.7:
                base_opinion *= 0.5  # More moderate

            self.behavior_state.opinion = np.clip(base_opinion, -1, 1)
            setattr(self, 'opinion', self.behavior_state.opinion)

        # Initialize other behaviors
        if 'attitude' in self.behavior_variables:
            self.behavior_state.attitude = np.random.normal(0, 1)
            setattr(self, 'attitude', self.behavior_state.attitude)

        if 'behavior_adoption' in self.behavior_variables:
            adoption_prob = 0.1 + self.attributes.innovation_adoption * 0.3
            self.behavior_state.behavior_adoption = np.random.random() < adoption_prob
            setattr(self, 'behavior_adoption', self.behavior_state.behavior_adoption)

    def _initialize_behavioral_parameters(self) -> Dict[str, float]:
        """Initialize behavioral parameters that can be updated by RSiena."""
        return {
            # Friendship formation parameters
            'friendship_formation_rate': 0.1,
            'friendship_dissolution_rate': 0.02,
            'homophily_age_effect': 0.5,
            'homophily_gender_effect': 0.3,
            'homophily_ses_effect': 0.4,
            'transitivity_effect': 0.2,
            'popularity_effect': 0.05,

            # Opinion dynamics parameters
            'opinion_confidence_bound': 0.3,
            'opinion_influence_strength': 0.1,
            'opinion_persistence': 0.9,
            'social_pressure_threshold': 0.7,

            # Behavioral parameters
            'activity_rate': 0.5,
            'conformity_strength': 0.3,
            'innovation_threshold': 0.6
        }

    def step(self):
        """
        Execute agent step with staged activation.

        This method is called by Mesa's scheduler and delegates to
        the appropriate stage method.
        """
        stage = getattr(self.model.schedule, 'stage', None)

        if stage == "network_formation":
            self.network_formation_stage()
        elif stage == "behavior_update":
            self.behavior_update_stage()
        elif stage == "network_dissolution":
            self.network_dissolution_stage()
        else:
            # Default single-stage behavior
            self.network_formation_stage()
            self.behavior_update_stage()
            self.network_dissolution_stage()

    def network_formation_stage(self):
        """
        Stage 1: Network formation decisions.

        Agents evaluate potential friendships based on homophily,
        structural effects, and individual characteristics.
        """
        # Update energy and motivation
        self._update_energy_and_motivation()

        # Decide on friendship formation attempts
        formation_probability = (
            self.behavioral_parameters['friendship_formation_rate'] *
            self.energy_level *
            (1 + self.attributes.extroversion)
        )

        if self.random.random() < formation_probability:
            self.attempt_friendship_formation()

        # Update network position metrics
        self._update_network_position()

    def behavior_update_stage(self):
        """
        Stage 2: Behavior and opinion updates.

        Agents update their behaviors based on social influence,
        network exposure, and individual characteristics.
        """
        # Opinion dynamics
        if hasattr(self, 'opinion'):
            self.update_opinion_dynamics()

        # Other behavior updates
        if hasattr(self, 'attitude'):
            self.update_attitude()

        if hasattr(self, 'behavior_adoption'):
            self.update_behavior_adoption()

        # Update social satisfaction
        self._update_social_satisfaction()

    def network_dissolution_stage(self):
        """
        Stage 3: Network dissolution decisions.

        Agents evaluate existing friendships for potential dissolution
        based on satisfaction, similarity, and structural factors.
        """
        dissolution_probability = (
            self.behavioral_parameters['friendship_dissolution_rate'] *
            (1 - self.social_satisfaction) *
            (1 + self.stress_level)
        )

        if self.random.random() < dissolution_probability and self.friends:
            self.consider_friendship_dissolution()

    def attempt_friendship_formation(self):
        """
        Attempt to form new friendships based on network opportunities.
        """
        # Get potential friends (exclude existing friends and self)
        all_agents = set(agent.unique_id for agent in self.model.schedule.agents)
        potential_friends = all_agents - self.friends - {self.unique_id}

        if not potential_friends:
            return

        # Calculate attraction scores for potential friends
        candidates = []
        scores = []

        for friend_id in potential_friends:
            friend = self.model.get_agent_by_id(friend_id)
            if friend:
                score = self.calculate_friendship_attraction(friend)
                if score > 0:
                    candidates.append(friend)
                    scores.append(score)

        if not candidates:
            return

        # Select target based on weighted probability
        scores = np.array(scores)
        probabilities = scores / scores.sum()

        target_idx = self.random.choices(range(len(candidates)), weights=probabilities)[0]
        target = candidates[target_idx]

        # Propose friendship
        self.propose_friendship(target)

    def calculate_friendship_attraction(self, other: 'SocialAgent') -> float:
        """
        Calculate attraction score for potential friendship.

        Args:
            other: Potential friend agent

        Returns:
            Attraction score (higher = more attractive)
        """
        score = 0.0

        # Homophily effects
        # Age similarity
        age_diff = abs(self.attributes.age - other.attributes.age)
        age_similarity = np.exp(-age_diff / 5.0)
        score += self.behavioral_parameters['homophily_age_effect'] * age_similarity

        # Gender homophily
        if self.attributes.gender == other.attributes.gender:
            score += self.behavioral_parameters['homophily_gender_effect']

        # SES similarity
        ses_diff = abs(self.attributes.socioeconomic_status - other.attributes.socioeconomic_status)
        ses_similarity = np.exp(-ses_diff / 2.0)
        score += self.behavioral_parameters['homophily_ses_effect'] * ses_similarity

        # Education similarity
        edu_diff = abs(self.attributes.education_level - other.attributes.education_level)
        edu_similarity = np.exp(-edu_diff / 2.0)
        score += 0.2 * edu_similarity

        # Personality compatibility
        personality_similarity = 1 - np.mean([
            abs(self.attributes.extroversion - other.attributes.extroversion),
            abs(self.attributes.openness - other.attributes.openness),
            abs(self.attributes.agreeableness - other.attributes.agreeableness)
        ])
        score += 0.3 * personality_similarity

        # Network structural effects
        # Transitivity (common friends)
        common_friends = len(self.friends.intersection(other.friends))
        score += self.behavioral_parameters['transitivity_effect'] * common_friends

        # Popularity effect (preferential attachment)
        score += self.behavioral_parameters['popularity_effect'] * len(other.friends)

        # Distance penalty (if spatial model)
        # Could add spatial distance penalty here if needed

        # Opinion similarity (if opinion variable exists)
        if hasattr(self, 'opinion') and hasattr(other, 'opinion'):
            opinion_diff = abs(self.opinion - other.opinion)
            opinion_similarity = np.exp(-opinion_diff / 0.5)
            score += 0.4 * opinion_similarity

        return max(0.0, score)

    def propose_friendship(self, target: 'SocialAgent'):
        """
        Propose friendship to target agent.

        Args:
            target: Agent to propose friendship to
        """
        self.friendship_attempts += 1

        # Record interaction
        self._record_interaction(target.unique_id, 'friendship_proposal', {
            'attraction_score': self.calculate_friendship_attraction(target),
            'step': self.model.schedule.steps
        })

        # Target evaluates proposal
        acceptance_probability = target.evaluate_friendship_proposal(self)

        if self.random.random() < acceptance_probability:
            # Friendship accepted
            self.friends.add(target.unique_id)
            target.friends.add(self.unique_id)

            # Add edge to network
            self.model.network.add_edge(self.unique_id, target.unique_id)

            self.friendship_successes += 1

            logger.debug(f"Friendship formed: {self.unique_id} <-> {target.unique_id}")
        else:
            self.friendship_rejections += 1
            logger.debug(f"Friendship rejected: {self.unique_id} -> {target.unique_id}")

    def evaluate_friendship_proposal(self, proposer: 'SocialAgent') -> float:
        """
        Evaluate incoming friendship proposal.

        Args:
            proposer: Agent proposing friendship

        Returns:
            Acceptance probability (0-1)
        """
        # Base acceptance probability
        base_prob = 0.3

        # Modify based on attraction
        attraction = proposer.calculate_friendship_attraction(self)
        base_prob += attraction * 0.15

        # Capacity constraint based on personality
        ideal_friends = int(5 + self.attributes.extroversion * 10)
        current_friends = len(self.friends)

        if current_friends >= ideal_friends:
            capacity_penalty = 1 - (current_friends - ideal_friends) / ideal_friends
            base_prob *= max(0.1, capacity_penalty)

        # Social status considerations
        status_diff = self.attributes.social_status - proposer.attributes.social_status
        if status_diff > 1:  # Proposer has much lower status
            base_prob *= 0.7
        elif status_diff < -1:  # Proposer has much higher status
            base_prob *= 1.3

        # Stress and energy effects
        base_prob *= (1 - self.stress_level) * self.energy_level

        return np.clip(base_prob, 0.0, 1.0)

    def consider_friendship_dissolution(self):
        """Consider dissolving existing friendships."""
        if not self.friends:
            return

        # Select friend to potentially dissolve relationship with
        friend_id = self.random.choice(list(self.friends))
        friend = self.model.get_agent_by_id(friend_id)

        if not friend:
            return

        # Calculate dissolution probability
        dissolution_prob = self.calculate_dissolution_probability(friend)

        if self.random.random() < dissolution_prob:
            # Dissolve friendship
            self.friends.remove(friend_id)
            friend.friends.remove(self.unique_id)

            # Remove edge from network
            if self.model.network.has_edge(self.unique_id, friend_id):
                self.model.network.remove_edge(self.unique_id, friend_id)

            logger.debug(f"Friendship dissolved: {self.unique_id} <-> {friend_id}")

    def calculate_dissolution_probability(self, friend: 'SocialAgent') -> float:
        """
        Calculate probability of dissolving friendship.

        Args:
            friend: Friend to potentially dissolve relationship with

        Returns:
            Dissolution probability (0-1)
        """
        base_prob = self.behavioral_parameters['friendship_dissolution_rate']

        # Relationship satisfaction
        attraction = self.calculate_friendship_attraction(friend)
        if attraction > 1.0:
            base_prob *= 0.3  # Strong attraction = low dissolution
        elif attraction < 0.3:
            base_prob *= 2.0  # Weak attraction = high dissolution

        # Common friends (structural stability)
        common_friends = len(self.friends.intersection(friend.friends))
        base_prob *= np.exp(-common_friends * 0.2)

        # Recent interaction quality
        recent_interactions = self.interaction_history.get(friend.unique_id, [])
        if recent_interactions:
            recent_quality = np.mean([i.get('quality', 0.5) for i in recent_interactions[-5:]])
            base_prob *= (1.5 - recent_quality)

        # Stress and life circumstances
        base_prob *= (1 + self.stress_level)

        return np.clip(base_prob, 0.0, 0.5)  # Cap at 50%

    def update_opinion_dynamics(self):
        """
        Update opinion based on social influence mechanisms.

        Implements bounded confidence model with social network effects.
        """
        if not hasattr(self, 'opinion') or not self.friends:
            return

        # Collect friend opinions
        friend_opinions = []
        for friend_id in self.friends:
            friend = self.model.get_agent_by_id(friend_id)
            if friend and hasattr(friend, 'opinion'):
                friend_opinions.append(friend.opinion)

        if not friend_opinions:
            return

        # Calculate average friend opinion
        avg_friend_opinion = np.mean(friend_opinions)

        # Bounded confidence: only influenced by similar opinions
        opinion_diff = abs(self.opinion - avg_friend_opinion)
        confidence_bound = self.behavioral_parameters['opinion_confidence_bound']

        if opinion_diff <= confidence_bound:
            # Calculate influence strength
            influence_strength = self.behavioral_parameters['opinion_influence_strength']

            # Adjust influence based on network position and personality
            if self.network_position.degree > 0:
                centrality_effect = self.network_position.degree / (self.network_position.degree + 5)
                influence_strength *= (1 + centrality_effect)

            # Personality effects
            influence_strength *= self.attributes.conformity_tendency
            influence_strength *= (1 - self.attributes.conscientiousness * 0.3)  # Independent thinkers resist

            # Calculate opinion change
            opinion_change = influence_strength * (avg_friend_opinion - self.opinion)

            # Apply persistence (inertia)
            persistence = self.behavioral_parameters['opinion_persistence']
            opinion_change *= (1 - persistence)

            # Update opinion
            old_opinion = self.opinion
            self.opinion = np.clip(self.opinion + opinion_change, -1, 1)

            # Track changes
            if abs(self.opinion - old_opinion) > 0.01:
                self.opinion_changes += 1

            # Update behavior state
            self.behavior_state.opinion = self.opinion

            logger.debug(f"Agent {self.unique_id} opinion: {old_opinion:.3f} -> {self.opinion:.3f}")

    def update_attitude(self):
        """Update general attitude based on network exposure."""
        if not hasattr(self, 'attitude') or not self.friends:
            return

        # Simple attitude diffusion
        friend_attitudes = []
        for friend_id in self.friends:
            friend = self.model.get_agent_by_id(friend_id)
            if friend and hasattr(friend, 'attitude'):
                friend_attitudes.append(friend.attitude)

        if friend_attitudes:
            avg_friend_attitude = np.mean(friend_attitudes)
            attitude_change = 0.1 * (avg_friend_attitude - self.attitude)
            self.attitude += attitude_change
            self.behavior_state.attitude = self.attitude

    def update_behavior_adoption(self):
        """Update binary behavior adoption based on network exposure."""
        if not hasattr(self, 'behavior_adoption') or not self.friends:
            return

        # Count adopting friends
        adopting_friends = 0
        for friend_id in self.friends:
            friend = self.model.get_agent_by_id(friend_id)
            if friend and hasattr(friend, 'behavior_adoption') and friend.behavior_adoption:
                adopting_friends += 1

        if len(self.friends) > 0:
            adoption_rate = adopting_friends / len(self.friends)

            # Adoption threshold based on innovation tendency
            threshold = self.behavioral_parameters['innovation_threshold']
            threshold *= (1 - self.attributes.innovation_adoption)

            if not self.behavior_adoption and adoption_rate > threshold:
                self.behavior_adoption = True
                self.behavior_state.behavior_adoption = True
                logger.debug(f"Agent {self.unique_id} adopted behavior")

    def _update_network_position(self):
        """Update network position metrics."""
        if self.model.network.has_node(self.unique_id):
            # Degree
            self.network_position.degree = self.model.network.degree(self.unique_id)

            # Local clustering coefficient
            try:
                self.network_position.clustering_coefficient = nx.clustering(
                    self.model.network, self.unique_id
                )
            except:
                self.network_position.clustering_coefficient = 0.0

            # Other centrality measures (computed periodically for performance)
            if self.model.schedule.steps % 10 == 0:
                self._update_centrality_measures()

    def _update_centrality_measures(self):
        """Update computationally expensive centrality measures."""
        try:
            # Betweenness centrality
            betweenness = nx.betweenness_centrality(self.model.network)
            self.network_position.betweenness_centrality = betweenness.get(self.unique_id, 0.0)

            # Closeness centrality
            if nx.is_connected(self.model.network):
                closeness = nx.closeness_centrality(self.model.network)
                self.network_position.closeness_centrality = closeness.get(self.unique_id, 0.0)

            # Eigenvector centrality
            eigenvector = nx.eigenvector_centrality(self.model.network, max_iter=100)
            self.network_position.eigenvector_centrality = eigenvector.get(self.unique_id, 0.0)

        except Exception as e:
            logger.debug(f"Error updating centrality measures: {e}")

    def _update_energy_and_motivation(self):
        """Update agent's energy and motivation levels."""
        # Energy recovery based on social satisfaction
        energy_recovery = 0.1 * self.social_satisfaction
        self.energy_level = np.clip(self.energy_level + energy_recovery, 0.1, 1.0)

        # Stress accumulation based on social strain
        if len(self.friends) > 0:
            ideal_friends = 5 + self.attributes.extroversion * 10
            friend_strain = abs(len(self.friends) - ideal_friends) / ideal_friends
            stress_increase = 0.05 * friend_strain
            self.stress_level = np.clip(self.stress_level + stress_increase - 0.02, 0.0, 1.0)

    def _update_social_satisfaction(self):
        """Update social satisfaction based on current network state."""
        satisfaction_components = []

        # Friendship quantity satisfaction
        ideal_friends = 5 + self.attributes.extroversion * 10
        current_friends = len(self.friends)
        quantity_satisfaction = 1 - abs(current_friends - ideal_friends) / ideal_friends
        satisfaction_components.append(quantity_satisfaction)

        # Friendship quality satisfaction
        if self.friends:
            quality_scores = []
            for friend_id in self.friends:
                friend = self.model.get_agent_by_id(friend_id)
                if friend:
                    attraction = self.calculate_friendship_attraction(friend)
                    quality_scores.append(min(1.0, attraction))

            quality_satisfaction = np.mean(quality_scores) if quality_scores else 0.5
            satisfaction_components.append(quality_satisfaction)
        else:
            satisfaction_components.append(0.3)  # Low satisfaction if no friends

        # Network position satisfaction
        if self.network_position.degree > 0:
            centrality_satisfaction = min(1.0, self.network_position.degree / 10)
            satisfaction_components.append(centrality_satisfaction)
        else:
            satisfaction_components.append(0.1)

        # Update satisfaction
        self.social_satisfaction = np.mean(satisfaction_components)
        self.satisfaction_history.append(self.social_satisfaction)

        # Keep only recent history
        if len(self.satisfaction_history) > 20:
            self.satisfaction_history = self.satisfaction_history[-20:]

    def _record_interaction(self, other_id: int, interaction_type: str, details: Dict):
        """Record interaction with another agent."""
        if other_id not in self.interaction_history:
            self.interaction_history[other_id] = []

        interaction_record = {
            'type': interaction_type,
            'step': self.model.schedule.steps,
            'details': details,
            'quality': details.get('quality', 0.5)
        }

        self.interaction_history[other_id].append(interaction_record)

        # Keep only recent interactions
        if len(self.interaction_history[other_id]) > 10:
            self.interaction_history[other_id] = self.interaction_history[other_id][-10:]

    def update_parameters(self, new_parameters: Dict[str, float]):
        """
        Update behavioral parameters from RSiena estimation.

        Args:
            new_parameters: Dictionary of parameter updates
        """
        for param_name, value in new_parameters.items():
            if param_name in self.behavioral_parameters:
                old_value = self.behavioral_parameters[param_name]
                self.behavioral_parameters[param_name] = value
                logger.debug(f"Agent {self.unique_id}: {param_name} {old_value:.3f} -> {value:.3f}")

    def local_clustering_coefficient(self) -> float:
        """Calculate local clustering coefficient."""
        return self.network_position.clustering_coefficient

    def betweenness_centrality(self) -> float:
        """Get betweenness centrality."""
        return self.network_position.betweenness_centrality

    def get_neighbor_ids(self) -> Set[int]:
        """Get set of neighbor IDs."""
        return self.friends.copy()

    def get_network_distance(self, other_id: int) -> int:
        """
        Calculate network distance to another agent.

        Args:
            other_id: ID of other agent

        Returns:
            Network distance (number of steps)
        """
        try:
            return nx.shortest_path_length(self.model.network, self.unique_id, other_id)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return float('inf')

    def get_state_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive state summary for analysis.

        Returns:
            Dictionary with current agent state
        """
        return {
            'id': self.unique_id,
            'attributes': {
                'age': self.attributes.age,
                'gender': self.attributes.gender,
                'ses': self.attributes.socioeconomic_status,
                'extroversion': self.attributes.extroversion,
                'education': self.attributes.education_level
            },
            'behaviors': {
                'opinion': getattr(self, 'opinion', None),
                'attitude': getattr(self, 'attitude', None),
                'behavior_adoption': getattr(self, 'behavior_adoption', None)
            },
            'network': {
                'friends': len(self.friends),
                'degree': self.network_position.degree,
                'clustering': self.network_position.clustering_coefficient,
                'betweenness': self.network_position.betweenness_centrality
            },
            'social': {
                'satisfaction': self.social_satisfaction,
                'stress': self.stress_level,
                'energy': self.energy_level
            },
            'activity': {
                'friendship_attempts': self.friendship_attempts,
                'friendship_successes': self.friendship_successes,
                'opinion_changes': self.opinion_changes
            }
        }


# Utility functions for agent creation and management

def create_realistic_agent_attributes(
    agent_id: int,
    population_parameters: Optional[Dict] = None,
    random_seed: Optional[int] = None
) -> AgentAttributes:
    """
    Create realistic agent attributes with correlated characteristics.

    Args:
        agent_id: Unique agent identifier
        population_parameters: Parameters for population distribution
        random_seed: Random seed for reproducibility

    Returns:
        AgentAttributes with realistic correlations
    """
    if random_seed is not None:
        np.random.seed(random_seed + agent_id)

    if population_parameters is None:
        population_parameters = {
            'mean_age': 25,
            'sd_age': 10,
            'ses_correlation': 0.3,
            'education_ses_correlation': 0.5
        }

    # Generate correlated attributes
    age = max(16, np.random.normal(population_parameters['mean_age'], population_parameters['sd_age']))
    gender = np.random.choice(['M', 'F'])

    # SES with some correlation to age (older -> higher SES)
    age_effect = (age - 20) / 50  # Normalize age effect
    ses = np.random.normal(age_effect * 0.5, 1)

    # Education correlated with SES
    ses_effect = ses * population_parameters['education_ses_correlation']
    education_base = 2.5 + ses_effect  # 1-5 scale
    education_level = int(np.clip(np.random.normal(education_base, 0.8), 1, 5))

    # Personality traits with some structure
    extroversion = np.random.beta(2, 2)
    openness = np.random.beta(2, 2)
    conscientiousness = np.random.beta(3, 2)  # Slight positive skew
    agreeableness = np.random.beta(3, 2)
    neuroticism = np.random.beta(2, 3)  # Slight negative skew

    # Derived attributes
    academic_performance = (
        0.3 * conscientiousness +
        0.2 * openness +
        0.1 * (education_level - 3) +
        np.random.normal(0, 0.5)
    )

    social_status = (
        0.4 * ses +
        0.2 * academic_performance +
        0.2 * extroversion +
        np.random.normal(0, 0.3)
    )

    leadership_tendency = (
        0.4 * extroversion +
        0.3 * conscientiousness +
        0.2 * (1 - agreeableness) +  # Leaders less agreeable
        0.1 * social_status +
        np.random.normal(0, 0.2)
    )

    # Behavioral tendencies
    risk_tolerance = (
        0.3 * openness +
        0.2 * extroversion +
        0.2 * (1 - neuroticism) +
        np.random.normal(0, 0.3)
    )

    conformity_tendency = (
        0.3 * agreeableness +
        0.2 * neuroticism +
        0.1 * (1 - openness) +
        np.random.normal(0, 0.3)
    )

    innovation_adoption = (
        0.4 * openness +
        0.2 * risk_tolerance +
        0.1 * education_level / 5 +
        np.random.normal(0, 0.3)
    )

    # Clip all values to appropriate ranges
    return AgentAttributes(
        age=age,
        gender=gender,
        socioeconomic_status=np.clip(ses, -3, 3),
        education_level=education_level,
        extroversion=np.clip(extroversion, 0, 1),
        openness=np.clip(openness, 0, 1),
        conscientiousness=np.clip(conscientiousness, 0, 1),
        agreeableness=np.clip(agreeableness, 0, 1),
        neuroticism=np.clip(neuroticism, 0, 1),
        academic_performance=np.clip(academic_performance, -3, 3),
        social_status=np.clip(social_status, -3, 3),
        leadership_tendency=np.clip(leadership_tendency, 0, 1),
        risk_tolerance=np.clip(risk_tolerance, 0, 1),
        conformity_tendency=np.clip(conformity_tendency, 0, 1),
        innovation_adoption=np.clip(innovation_adoption, 0, 1)
    )


def create_agent_population(
    n_agents: int,
    behavior_variables: List[str] = None,
    population_parameters: Optional[Dict] = None,
    random_seed: Optional[int] = None
) -> List[AgentAttributes]:
    """
    Create a realistic agent population with appropriate diversity.

    Args:
        n_agents: Number of agents to create
        behavior_variables: List of behavior variables
        population_parameters: Population distribution parameters
        random_seed: Random seed for reproducibility

    Returns:
        List of AgentAttributes
    """
    attributes_list = []

    for i in range(n_agents):
        attributes = create_realistic_agent_attributes(
            i, population_parameters, random_seed
        )
        attributes_list.append(attributes)

    logger.info(f"Created population of {n_agents} agents")
    return attributes_list


if __name__ == "__main__":
    # Example usage and testing
    logging.basicConfig(level=logging.INFO)

    # Create test attributes
    attributes = create_realistic_agent_attributes(0)
    print(f"Test agent attributes: {attributes}")

    # Create test population
    population = create_agent_population(10, ['opinion', 'attitude'])
    print(f"Created population of {len(population)} agents")

    # Display population summary
    ages = [attr.age for attr in population]
    ses_scores = [attr.socioeconomic_status for attr in population]

    print(f"Age range: {min(ages):.1f} - {max(ages):.1f}")
    print(f"SES range: {min(ses_scores):.2f} - {max(ses_scores):.2f}")