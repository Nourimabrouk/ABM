"""
Social Network Agent-Based Model with RSiena Integration

This module implements a social network ABM that can be validated against
empirical network dynamics using RSiena. The model focuses on friendship
formation and dissolution in a social setting.

Features:
- Agent-based friendship formation with homophily and transitivity
- Longitudinal network evolution tracking
- RSiena integration for empirical validation
- Statistical comparison of ABM vs. empirical dynamics
"""

import logging
import numpy as np
import networkx as nx
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

import mesa
from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector

# Import our RSiena integration
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from utils.rsiena_integration import RSienaIntegrator

logger = logging.getLogger(__name__)


@dataclass
class AgentAttributes:
    """Data class for agent attributes that might influence network formation."""
    age: float
    gender: str
    socioeconomic_status: float
    extroversion: float
    academic_performance: float


class SocialAgent(Agent):
    """
    A social agent that forms and maintains friendships based on homophily,
    transitivity, and other social mechanisms.
    """

    def __init__(self, unique_id: int, model: 'SocialNetworkModel', attributes: AgentAttributes):
        super().__init__(unique_id, model)
        self.attributes = attributes
        self.friends = set()  # Current friendship ties
        self.friendship_attempts = 0
        self.friendship_rejections = 0

    def step(self):
        """Execute one step of the agent's behavior."""
        # Decide whether to attempt new friendships
        if self.random.random() < self.model.friendship_formation_rate:
            self.attempt_friendship()

        # Decide whether to dissolve existing friendships
        if self.random.random() < self.model.friendship_dissolution_rate:
            self.consider_friendship_dissolution()

        # Update friendship satisfaction (affects future behavior)
        self.update_social_satisfaction()

    def attempt_friendship(self):
        """Attempt to form a new friendship."""
        # Get potential friends (not already friends)
        potential_friends = [agent for agent in self.model.schedule.agents
                           if agent.unique_id != self.unique_id
                           and agent.unique_id not in self.friends]

        if not potential_friends:
            return

        # Select target based on homophily and proximity
        target = self.select_friendship_target(potential_friends)
        if target:
            self.propose_friendship(target)

    def select_friendship_target(self, candidates: List['SocialAgent']) -> Optional['SocialAgent']:
        """Select friendship target based on homophily and network structure."""
        if not candidates:
            return None

        # Calculate attraction scores for each candidate
        scores = []
        for candidate in candidates:
            score = self.calculate_attraction_score(candidate)
            scores.append(score)

        # Select based on weighted random choice
        if max(scores) <= 0:
            return None

        # Normalize scores and select
        scores = np.array(scores)
        scores = np.maximum(scores, 0)  # Ensure non-negative
        if scores.sum() == 0:
            return self.random.choice(candidates)

        probabilities = scores / scores.sum()
        selected_idx = self.random.choices(range(len(candidates)), weights=probabilities)[0]
        return candidates[selected_idx]

    def calculate_attraction_score(self, other: 'SocialAgent') -> float:
        """Calculate attraction score based on homophily and network effects."""
        score = 0.0

        # Homophily effects
        # Age similarity
        age_diff = abs(self.attributes.age - other.attributes.age)
        score += self.model.age_homophily * np.exp(-age_diff / 5.0)

        # Gender homophily
        if self.attributes.gender == other.attributes.gender:
            score += self.model.gender_homophily

        # SES similarity
        ses_diff = abs(self.attributes.socioeconomic_status - other.attributes.socioeconomic_status)
        score += self.model.ses_homophily * np.exp(-ses_diff / 2.0)

        # Academic performance similarity
        academic_diff = abs(self.attributes.academic_performance - other.attributes.academic_performance)
        score += self.model.academic_homophily * np.exp(-academic_diff / 1.0)

        # Network structural effects
        # Transitivity (common friends)
        common_friends = len(self.friends.intersection(other.friends))
        score += self.model.transitivity_effect * common_friends

        # Popularity effect (preferential attachment)
        score += self.model.popularity_effect * len(other.friends)

        # Individual characteristics
        # Extroversion effect
        score += self.model.extroversion_effect * (self.attributes.extroversion + other.attributes.extroversion) / 2

        return score

    def propose_friendship(self, target: 'SocialAgent'):
        """Propose friendship to target agent."""
        self.friendship_attempts += 1

        # Target decides whether to accept
        acceptance_probability = target.evaluate_friendship_proposal(self)

        if self.random.random() < acceptance_probability:
            # Friendship formed
            self.friends.add(target.unique_id)
            target.friends.add(self.unique_id)
            self.model.network.add_edge(self.unique_id, target.unique_id)
            logger.debug(f"Friendship formed: {self.unique_id} <-> {target.unique_id}")
        else:
            self.friendship_rejections += 1
            logger.debug(f"Friendship rejected: {self.unique_id} -> {target.unique_id}")

    def evaluate_friendship_proposal(self, proposer: 'SocialAgent') -> float:
        """Evaluate incoming friendship proposal."""
        # Base acceptance rate
        acceptance_prob = 0.5

        # Modify based on attraction score
        attraction = proposer.calculate_attraction_score(self)
        acceptance_prob += attraction * 0.1

        # Modify based on current number of friends (capacity constraint)
        friend_capacity = int(self.attributes.extroversion * 10 + 5)  # 5-15 friends
        if len(self.friends) >= friend_capacity:
            acceptance_prob *= 0.1  # Much less likely to accept if at capacity

        return np.clip(acceptance_prob, 0.0, 1.0)

    def consider_friendship_dissolution(self):
        """Consider dissolving existing friendships."""
        if not self.friends:
            return

        # Select a friend to potentially dissolve relationship with
        friend_id = self.random.choice(list(self.friends))
        friend = self.model.get_agent(friend_id)

        # Calculate dissolution probability
        dissolution_prob = self.calculate_dissolution_probability(friend)

        if self.random.random() < dissolution_prob:
            # Dissolve friendship
            self.friends.remove(friend_id)
            friend.friends.remove(self.unique_id)
            self.model.network.remove_edge(self.unique_id, friend_id)
            logger.debug(f"Friendship dissolved: {self.unique_id} <-> {friend_id}")

    def calculate_dissolution_probability(self, friend: 'SocialAgent') -> float:
        """Calculate probability of dissolving friendship with specific friend."""
        base_prob = 0.05  # Base dissolution rate

        # Lower dissolution for similar agents (homophily)
        similarity_score = self.calculate_attraction_score(friend)
        if similarity_score > 1.0:
            base_prob *= 0.5

        # Lower dissolution if many common friends
        common_friends = len(self.friends.intersection(friend.friends))
        base_prob *= np.exp(-common_friends * 0.2)

        return base_prob

    def update_social_satisfaction(self):
        """Update agent's social satisfaction based on current friendships."""
        # This could affect future friendship-seeking behavior
        ideal_friends = int(self.attributes.extroversion * 10 + 5)
        current_friends = len(self.friends)

        # Satisfaction based on closeness to ideal number of friends
        self.social_satisfaction = 1.0 - abs(current_friends - ideal_friends) / ideal_friends

    def get_agent(self, agent_id: int) -> 'SocialAgent':
        """Get agent by ID."""
        for agent in self.model.schedule.agents:
            if agent.unique_id == agent_id:
                return agent
        return None


class SocialNetworkModel(Model):
    """
    A social network ABM that simulates friendship formation and dissolution
    with integration to RSiena for empirical validation.
    """

    def __init__(
        self,
        n_agents: int = 100,
        # Homophily parameters
        age_homophily: float = 0.5,
        gender_homophily: float = 0.3,
        ses_homophily: float = 0.4,
        academic_homophily: float = 0.3,
        # Network structure parameters
        transitivity_effect: float = 0.2,
        popularity_effect: float = 0.05,
        extroversion_effect: float = 0.1,
        # Behavioral parameters
        friendship_formation_rate: float = 0.1,
        friendship_dissolution_rate: float = 0.02,
        # RSiena integration
        enable_rsiena_validation: bool = False,
        rsiena_validation_interval: int = 50
    ):
        super().__init__()

        # Model parameters
        self.n_agents = n_agents
        self.age_homophily = age_homophily
        self.gender_homophily = gender_homophily
        self.ses_homophily = ses_homophily
        self.academic_homophily = academic_homophily
        self.transitivity_effect = transitivity_effect
        self.popularity_effect = popularity_effect
        self.extroversion_effect = extroversion_effect
        self.friendship_formation_rate = friendship_formation_rate
        self.friendship_dissolution_rate = friendship_dissolution_rate

        # RSiena integration
        self.enable_rsiena_validation = enable_rsiena_validation
        self.rsiena_validation_interval = rsiena_validation_interval
        self.rsiena_integrator = None
        self.network_snapshots = []

        if self.enable_rsiena_validation:
            try:
                self.rsiena_integrator = RSienaIntegrator()
            except Exception as e:
                logger.warning(f"RSiena integration disabled: {e}")
                self.enable_rsiena_validation = False

        # Initialize components
        self.schedule = RandomActivation(self)
        self.network = nx.Graph()

        # Create agents
        self._create_agents()

        # Data collection
        self.datacollector = DataCollector(
            model_reporters={
                "Network_Density": self._get_network_density,
                "Average_Degree": self._get_average_degree,
                "Clustering_Coefficient": self._get_clustering_coefficient,
                "Number_of_Components": self._get_number_of_components,
                "Average_Path_Length": self._get_average_path_length,
                "Assortativity_Age": lambda m: self._get_assortativity("age"),
                "Assortativity_Gender": lambda m: self._get_assortativity("gender"),
                "Assortativity_SES": lambda m: self._get_assortativity("socioeconomic_status")
            },
            agent_reporters={
                "Degree": lambda a: len(a.friends),
                "Age": lambda a: a.attributes.age,
                "Gender": lambda a: a.attributes.gender,
                "SES": lambda a: a.attributes.socioeconomic_status,
                "Social_Satisfaction": lambda a: getattr(a, 'social_satisfaction', 0)
            }
        )

        self.running = True
        self.datacollector.collect(self)

    def _create_agents(self):
        """Create agents with random attributes."""
        for i in range(self.n_agents):
            # Generate random attributes
            attributes = AgentAttributes(
                age=self.random.normalvariate(20, 3),  # Age around 20 ± 3 years
                gender=self.random.choice(['M', 'F']),
                socioeconomic_status=self.random.normalvariate(0, 1),  # Standardized SES
                extroversion=self.random.betavariate(2, 2),  # 0-1 scale
                academic_performance=self.random.normalvariate(0, 1)  # Standardized performance
            )

            agent = SocialAgent(i, self, attributes)
            self.schedule.add(agent)
            self.network.add_node(i)

    def step(self):
        """Advance the model by one step."""
        self.schedule.step()
        self.datacollector.collect(self)

        # Store network snapshots for RSiena validation
        if self.enable_rsiena_validation and self.schedule.steps % self.rsiena_validation_interval == 0:
            self.network_snapshots.append(self.network.copy())

    def get_agent(self, agent_id: int) -> SocialAgent:
        """Get agent by ID."""
        for agent in self.schedule.agents:
            if agent.unique_id == agent_id:
                return agent
        return None

    def validate_with_rsiena(self, empirical_networks: List[nx.Graph]) -> Dict[str, Any]:
        """
        Validate the ABM against empirical network data using RSiena.

        Args:
            empirical_networks: List of empirical network snapshots

        Returns:
            Dictionary containing validation results
        """
        if not self.enable_rsiena_validation or not self.rsiena_integrator:
            raise RuntimeError("RSiena validation not enabled or not available")

        if len(self.network_snapshots) < 2:
            raise ValueError("Need at least 2 network snapshots for validation")

        # Convert empirical networks to RSiena format
        empirical_data = self.rsiena_integrator.mesa_networks_to_rsiena(empirical_networks)

        # Create effects object
        effects = self.rsiena_integrator.create_rsiena_effects(empirical_data['siena_data'])

        # Estimate RSiena model on empirical data
        rsiena_results = self.rsiena_integrator.estimate_rsiena_model(
            empirical_data['siena_data'], effects
        )

        # Validate ABM networks against RSiena predictions
        validation_results = self.rsiena_integrator.validate_abm_with_rsiena(
            self.network_snapshots, rsiena_results
        )

        return {
            'rsiena_results': rsiena_results,
            'validation_metrics': validation_results,
            'empirical_data': empirical_data
        }

    # Network metric methods
    def _get_network_density(self):
        """Get network density."""
        return nx.density(self.network)

    def _get_average_degree(self):
        """Get average degree."""
        if self.network.number_of_nodes() == 0:
            return 0
        return sum(dict(self.network.degree()).values()) / self.network.number_of_nodes()

    def _get_clustering_coefficient(self):
        """Get global clustering coefficient."""
        try:
            return nx.transitivity(self.network)
        except:
            return 0

    def _get_number_of_components(self):
        """Get number of connected components."""
        return nx.number_connected_components(self.network)

    def _get_average_path_length(self):
        """Get average path length."""
        try:
            if nx.is_connected(self.network):
                return nx.average_shortest_path_length(self.network)
            else:
                # Calculate for largest component
                largest_cc = max(nx.connected_components(self.network), key=len)
                subgraph = self.network.subgraph(largest_cc)
                return nx.average_shortest_path_length(subgraph)
        except:
            return float('inf')

    def _get_assortativity(self, attribute: str):
        """Get assortativity by attribute."""
        try:
            node_attrs = {}
            for agent in self.schedule.agents:
                if attribute == "gender":
                    node_attrs[agent.unique_id] = 1 if agent.attributes.gender == 'M' else 0
                else:
                    node_attrs[agent.unique_id] = getattr(agent.attributes, attribute)

            if attribute == "gender":
                return nx.attribute_assortativity_coefficient(self.network, node_attrs)
            else:
                return nx.numeric_assortativity_coefficient(self.network, node_attrs)
        except:
            return 0


def run_social_network_simulation(
    steps: int = 200,
    n_agents: int = 100,
    enable_rsiena: bool = False
) -> Tuple[SocialNetworkModel, pd.DataFrame]:
    """
    Run a social network simulation.

    Args:
        steps: Number of simulation steps
        n_agents: Number of agents
        enable_rsiena: Whether to enable RSiena validation

    Returns:
        Tuple of (model, data)
    """
    # Create model
    model = SocialNetworkModel(
        n_agents=n_agents,
        enable_rsiena_validation=enable_rsiena
    )

    # Run simulation
    logger.info(f"Running social network simulation for {steps} steps...")
    for step in range(steps):
        model.step()
        if step % 50 == 0:
            logger.info(f"Step {step}: Density = {model._get_network_density():.3f}")

    # Get data
    data = model.datacollector.get_model_vars_dataframe()

    logger.info("Simulation completed!")
    logger.info(f"Final network density: {model._get_network_density():.3f}")
    logger.info(f"Final average degree: {model._get_average_degree():.2f}")

    return model, data


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    # Run simulation
    model, data = run_social_network_simulation(steps=100, n_agents=50)

    # Print summary statistics
    print("\nFinal Network Statistics:")
    print(f"Density: {model._get_network_density():.3f}")
    print(f"Average Degree: {model._get_average_degree():.2f}")
    print(f"Clustering Coefficient: {model._get_clustering_coefficient():.3f}")
    print(f"Number of Components: {model._get_number_of_components()}")

    # Show data evolution
    print("\nNetwork Evolution:")
    print(data[['Network_Density', 'Average_Degree', 'Clustering_Coefficient']].tail())