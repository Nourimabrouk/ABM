"""
Institutional Agent Implementation for Group-Level Behaviors

This module implements institutional agents representing organizations, groups,
and collective entities within the ABM-RSiena integration framework. These agents
model group-level behaviors, institutional influence, and collective decision-making.

Key Features:
- Multi-level agent modeling (individuals within institutions)
- Institutional policy and culture evolution
- Group-level network formation and dissolution
- Hierarchical influence structures
- Collective action and coordination mechanisms

Author: Beta Agent - Implementation Specialist
Created: 2025-09-15
"""

import logging
import numpy as np
import networkx as nx
from typing import Dict, List, Optional, Set, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum

from .social_agent import SocialAgent, AgentAttributes
from .influence_agent import InfluenceAgent

logger = logging.getLogger(__name__)


class InstitutionType(Enum):
    """Types of institutional entities."""
    SCHOOL = "school"
    WORKPLACE = "workplace"
    COMMUNITY_GROUP = "community_group"
    FAMILY = "family"
    CLUB_ORGANIZATION = "club_organization"
    GOVERNMENT_AGENCY = "government_agency"
    RELIGIOUS_GROUP = "religious_group"


@dataclass
class InstitutionalPolicy:
    """Represents an institutional policy or rule."""
    policy_id: str
    description: str
    enforcement_level: float  # 0-1 scale
    compliance_rate: float = 0.0
    adoption_date: int = 0
    impact_strength: float = 1.0


@dataclass
class InstitutionalCulture:
    """Represents institutional culture and norms."""
    openness_to_change: float = 0.5  # 0-1 scale
    hierarchy_strength: float = 0.5  # How hierarchical the institution is
    collective_orientation: float = 0.5  # Individual vs collective focus
    innovation_tolerance: float = 0.5  # Tolerance for innovation
    conformity_pressure: float = 0.5  # Pressure to conform
    communication_openness: float = 0.5  # How open communication is


class InstitutionalAgent(SocialAgent):
    """
    Agent representing institutional entities and group-level behaviors.

    Models organizations, schools, families, and other collective entities
    that influence individual agents and coordinate group activities.
    """

    def __init__(
        self,
        unique_id: int,
        model: 'ABMRSienaModel',
        institution_type: InstitutionType,
        institution_name: str,
        capacity: int = 100,
        initial_culture: Optional[InstitutionalCulture] = None,
        leadership_structure: Optional[Dict[str, List[int]]] = None
    ):
        """
        Initialize institutional agent.

        Args:
            unique_id: Unique agent identifier
            model: Reference to ABM model
            institution_type: Type of institution
            institution_name: Name of the institution
            capacity: Maximum number of members
            initial_culture: Initial institutional culture
            leadership_structure: Dictionary defining leadership roles
        """
        # Create institutional attributes
        attributes = self._create_institutional_attributes(institution_type)
        super().__init__(unique_id, model, attributes)

        self.institution_type = institution_type
        self.institution_name = institution_name
        self.capacity = capacity

        # Institutional culture and policies
        self.culture = initial_culture or self._create_default_culture(institution_type)
        self.policies: Dict[str, InstitutionalPolicy] = {}

        # Membership and structure
        self.members: Set[int] = set()
        self.leadership_structure = leadership_structure or {}
        self.member_roles: Dict[int, str] = {}  # member_id -> role

        # Hierarchical relationships
        self.superiors: Set[int] = set()  # Higher-level institutions
        self.subordinates: Set[int] = set()  # Lower-level institutions
        self.peer_institutions: Set[int] = set()  # Peer institutions

        # Activity and performance tracking
        self.collective_activities: List[Dict] = []
        self.performance_metrics: Dict[str, float] = {}
        self.resource_level = 1.0
        self.institutional_reputation = 0.5

        # Decision-making and coordination
        self.pending_decisions: List[Dict] = []
        self.coordination_attempts = 0
        self.successful_coordinations = 0

        # Network formation at institutional level
        self.institutional_network_formation_rate = 0.05
        self.institutional_dissolution_rate = 0.01

        logger.info(f"Created {institution_type.value} institutional agent: {institution_name}")

    def _create_institutional_attributes(self, institution_type: InstitutionType) -> AgentAttributes:
        """Create attributes appropriate for institutional agent."""
        # Institutions have different "personality" profiles
        if institution_type == InstitutionType.SCHOOL:
            return AgentAttributes(
                age=50,  # Institutional age
                gender='N',  # Not applicable
                socioeconomic_status=0.0,
                education_level=5,
                extroversion=0.6,  # Schools are socially active
                openness=0.7,  # Educational institutions are open to ideas
                conscientiousness=0.8,  # Highly organized
                agreeableness=0.6,
                neuroticism=0.3,  # Stable institutions
                academic_performance=1.0,
                social_status=1.0,
                leadership_tendency=0.8
            )
        elif institution_type == InstitutionType.WORKPLACE:
            return AgentAttributes(
                age=30,
                gender='N',
                socioeconomic_status=0.5,
                education_level=3,
                extroversion=0.5,
                openness=0.5,
                conscientiousness=0.9,  # Very organized
                agreeableness=0.4,  # Can be competitive
                neuroticism=0.4,
                academic_performance=0.5,
                social_status=0.8,
                leadership_tendency=0.7
            )
        else:
            # Default institutional attributes
            return AgentAttributes(
                age=25,
                gender='N',
                socioeconomic_status=0.0,
                education_level=3,
                extroversion=0.5,
                openness=0.5,
                conscientiousness=0.7,
                agreeableness=0.6,
                neuroticism=0.4,
                academic_performance=0.5,
                social_status=0.6,
                leadership_tendency=0.6
            )

    def _create_default_culture(self, institution_type: InstitutionType) -> InstitutionalCulture:
        """Create default culture based on institution type."""
        if institution_type == InstitutionType.SCHOOL:
            return InstitutionalCulture(
                openness_to_change=0.7,
                hierarchy_strength=0.6,
                collective_orientation=0.7,
                innovation_tolerance=0.8,
                conformity_pressure=0.5,
                communication_openness=0.7
            )
        elif institution_type == InstitutionType.WORKPLACE:
            return InstitutionalCulture(
                openness_to_change=0.4,
                hierarchy_strength=0.8,
                collective_orientation=0.6,
                innovation_tolerance=0.5,
                conformity_pressure=0.7,
                communication_openness=0.5
            )
        elif institution_type == InstitutionType.FAMILY:
            return InstitutionalCulture(
                openness_to_change=0.3,
                hierarchy_strength=0.7,
                collective_orientation=0.9,
                innovation_tolerance=0.4,
                conformity_pressure=0.6,
                communication_openness=0.8
            )
        else:
            return InstitutionalCulture()

    def step(self):
        """Execute institutional agent step."""
        # Institutional-level decision making
        self._make_institutional_decisions()

        # Influence member agents
        self._influence_members()

        # Coordinate with other institutions
        self._coordinate_with_institutions()

        # Update institutional metrics
        self._update_institutional_metrics()

        # Form/dissolve institutional networks
        self._manage_institutional_networks()

        # Evolve culture and policies
        self._evolve_culture_and_policies()

    def _make_institutional_decisions(self):
        """Make collective decisions for the institution."""
        if not self.pending_decisions:
            return

        for decision in self.pending_decisions[:]:
            decision_outcome = self._evaluate_decision(decision)

            if decision_outcome['approved']:
                self._implement_decision(decision, decision_outcome)
                logger.debug(f"Institution {self.unique_id} implemented decision: {decision['type']}")

            self.pending_decisions.remove(decision)

    def _evaluate_decision(self, decision: Dict) -> Dict[str, Any]:
        """Evaluate a pending institutional decision."""
        decision_type = decision.get('type', 'general')
        urgency = decision.get('urgency', 0.5)
        resource_requirement = decision.get('resources', 0.1)

        # Base approval probability
        approval_prob = 0.5

        # Adjust based on institutional culture
        if decision_type == 'innovation':
            approval_prob *= (1 + self.culture.innovation_tolerance)
        elif decision_type == 'policy_change':
            approval_prob *= (1 + self.culture.openness_to_change)

        # Resource constraints
        if resource_requirement > self.resource_level:
            approval_prob *= 0.3

        # Urgency effects
        approval_prob *= (1 + urgency * 0.5)

        # Member consensus (if applicable)
        if self.members and decision.get('requires_consensus', False):
            member_support = self._gauge_member_support(decision)
            approval_prob *= member_support

        approved = np.random.random() < approval_prob

        return {
            'approved': approved,
            'approval_probability': approval_prob,
            'resource_cost': resource_requirement if approved else 0,
            'implementation_time': decision.get('time_required', 1)
        }

    def _implement_decision(self, decision: Dict, outcome: Dict):
        """Implement an approved decision."""
        # Deduct resources
        self.resource_level -= outcome['resource_cost']

        # Create or update policy if applicable
        if decision['type'] == 'policy_creation':
            policy = InstitutionalPolicy(
                policy_id=decision['policy_id'],
                description=decision['description'],
                enforcement_level=decision.get('enforcement', 0.7),
                adoption_date=self.model.schedule.steps
            )
            self.policies[policy.policy_id] = policy

        # Record decision in activity log
        activity = {
            'type': 'decision_implementation',
            'decision': decision,
            'outcome': outcome,
            'step': self.model.schedule.steps
        }
        self.collective_activities.append(activity)

    def _influence_members(self):
        """Influence member agents through institutional mechanisms."""
        if not self.members:
            return

        for member_id in self.members:
            member = self.model.get_agent_by_id(member_id)
            if not member:
                continue

            # Apply institutional culture influence
            self._apply_cultural_influence(member)

            # Apply policy compliance pressure
            self._apply_policy_influence(member)

            # Apply role-based influence
            member_role = self.member_roles.get(member_id, 'member')
            self._apply_role_influence(member, member_role)

    def _apply_cultural_influence(self, member: SocialAgent):
        """Apply institutional culture influence on member."""
        if not hasattr(member, 'behavior_state'):
            return

        # Influence opinion towards institutional norm (if applicable)
        if hasattr(member, 'opinion') and hasattr(self, 'institutional_opinion'):
            opinion_pressure = self.culture.conformity_pressure * 0.1
            opinion_change = opinion_pressure * (self.institutional_opinion - member.opinion)
            member.opinion = np.clip(member.opinion + opinion_change, -1, 1)

        # Influence conformity tendency
        if hasattr(member, 'attributes'):
            conformity_influence = self.culture.conformity_pressure * 0.05
            # This would affect future behavior (simplified here)

    def _apply_policy_influence(self, member: SocialAgent):
        """Apply policy compliance influence on member."""
        for policy in self.policies.values():
            if policy.enforcement_level > 0.5:
                # High enforcement policies have stronger effect
                compliance_pressure = policy.enforcement_level * 0.1

                # Apply pressure based on policy type (simplified)
                if hasattr(member, 'conformity_tendency'):
                    member.conformity_tendency = min(1.0,
                        member.conformity_tendency + compliance_pressure * 0.1)

    def _apply_role_influence(self, member: SocialAgent, role: str):
        """Apply role-based influence on member."""
        role_effects = {
            'leader': {'leadership_tendency': 0.1, 'social_status': 0.05},
            'senior': {'social_status': 0.03, 'leadership_tendency': 0.05},
            'member': {},
            'newcomer': {'conformity_tendency': 0.1}
        }

        effects = role_effects.get(role, {})
        for attribute, change in effects.items():
            if hasattr(member.attributes, attribute):
                current_value = getattr(member.attributes, attribute)
                new_value = np.clip(current_value + change, 0, 1)
                setattr(member.attributes, attribute, new_value)

    def _coordinate_with_institutions(self):
        """Coordinate activities with other institutions."""
        if not self.peer_institutions and not self.superiors:
            return

        coordination_probability = 0.1 * (1 + self.culture.collective_orientation)

        if np.random.random() < coordination_probability:
            self.coordination_attempts += 1

            # Select coordination target
            potential_partners = list(self.peer_institutions.union(self.superiors))
            if potential_partners:
                partner_id = self.random.choice(potential_partners)
                partner = self.model.get_agent_by_id(partner_id)

                if partner and isinstance(partner, InstitutionalAgent):
                    success = self._attempt_coordination(partner)
                    if success:
                        self.successful_coordinations += 1

    def _attempt_coordination(self, partner: 'InstitutionalAgent') -> bool:
        """Attempt coordination with partner institution."""
        # Calculate coordination compatibility
        cultural_compatibility = 1 - np.mean([
            abs(self.culture.openness_to_change - partner.culture.openness_to_change),
            abs(self.culture.collective_orientation - partner.culture.collective_orientation),
            abs(self.culture.communication_openness - partner.culture.communication_openness)
        ])

        resource_compatibility = min(self.resource_level, partner.resource_level)
        coordination_prob = cultural_compatibility * resource_compatibility * 0.5

        if np.random.random() < coordination_prob:
            # Successful coordination
            coordination_activity = {
                'type': 'inter_institutional_coordination',
                'partner': partner.unique_id,
                'step': self.model.schedule.steps,
                'success': True
            }

            self.collective_activities.append(coordination_activity)
            partner.collective_activities.append(coordination_activity)

            # Add to peer network if not already connected
            self.peer_institutions.add(partner.unique_id)
            partner.peer_institutions.add(self.unique_id)

            logger.debug(f"Coordination between institutions {self.unique_id} and {partner.unique_id}")
            return True

        return False

    def _manage_institutional_networks(self):
        """Manage formation and dissolution of institutional networks."""
        # Institutional network formation
        if np.random.random() < self.institutional_network_formation_rate:
            self._attempt_institutional_network_formation()

        # Institutional network dissolution
        if np.random.random() < self.institutional_dissolution_rate and self.peer_institutions:
            self._consider_institutional_network_dissolution()

    def _attempt_institutional_network_formation(self):
        """Attempt to form new institutional relationships."""
        # Find potential partner institutions
        all_institutions = [a for a in self.model.schedule.agents
                          if isinstance(a, InstitutionalAgent) and a.unique_id != self.unique_id]

        if not all_institutions:
            return

        # Filter out existing connections
        potential_partners = [inst for inst in all_institutions
                            if inst.unique_id not in self.peer_institutions]

        if not potential_partners:
            return

        # Select partner based on compatibility
        best_partner = None
        best_compatibility = 0

        for partner in potential_partners[:10]:  # Check up to 10 candidates
            compatibility = self._calculate_institutional_compatibility(partner)
            if compatibility > best_compatibility:
                best_compatibility = compatibility
                best_partner = partner

        # Form connection if compatibility is high enough
        if best_partner and best_compatibility > 0.5:
            self.peer_institutions.add(best_partner.unique_id)
            best_partner.peer_institutions.add(self.unique_id)
            logger.debug(f"Institutional network formed: {self.unique_id} <-> {best_partner.unique_id}")

    def _calculate_institutional_compatibility(self, other: 'InstitutionalAgent') -> float:
        """Calculate compatibility with another institution."""
        compatibility = 0.0

        # Type compatibility
        if self.institution_type == other.institution_type:
            compatibility += 0.3
        elif self._are_compatible_types(self.institution_type, other.institution_type):
            compatibility += 0.1

        # Cultural compatibility
        cultural_similarity = 1 - np.mean([
            abs(self.culture.openness_to_change - other.culture.openness_to_change),
            abs(self.culture.collective_orientation - other.culture.collective_orientation),
            abs(self.culture.innovation_tolerance - other.culture.innovation_tolerance),
            abs(self.culture.communication_openness - other.culture.communication_openness)
        ])
        compatibility += 0.4 * cultural_similarity

        # Resource level compatibility
        resource_similarity = 1 - abs(self.resource_level - other.resource_level)
        compatibility += 0.2 * resource_similarity

        # Reputation compatibility
        reputation_similarity = 1 - abs(self.institutional_reputation - other.institutional_reputation)
        compatibility += 0.1 * reputation_similarity

        return np.clip(compatibility, 0, 1)

    def _are_compatible_types(self, type1: InstitutionType, type2: InstitutionType) -> bool:
        """Check if two institution types are compatible for networking."""
        compatible_pairs = {
            (InstitutionType.SCHOOL, InstitutionType.COMMUNITY_GROUP),
            (InstitutionType.WORKPLACE, InstitutionType.COMMUNITY_GROUP),
            (InstitutionType.CLUB_ORGANIZATION, InstitutionType.COMMUNITY_GROUP),
            (InstitutionType.RELIGIOUS_GROUP, InstitutionType.COMMUNITY_GROUP)
        }

        return (type1, type2) in compatible_pairs or (type2, type1) in compatible_pairs

    def _consider_institutional_network_dissolution(self):
        """Consider dissolving institutional relationships."""
        if not self.peer_institutions:
            return

        # Select a peer to potentially dissolve relationship with
        peer_id = self.random.choice(list(self.peer_institutions))
        peer = self.model.get_agent_by_id(peer_id)

        if not peer or not isinstance(peer, InstitutionalAgent):
            return

        # Calculate dissolution probability
        dissolution_prob = self.institutional_dissolution_rate

        # Increase dissolution if compatibility has decreased
        current_compatibility = self._calculate_institutional_compatibility(peer)
        if current_compatibility < 0.3:
            dissolution_prob *= 3

        # Decrease dissolution if successful coordination history
        if self.successful_coordinations > 5:
            dissolution_prob *= 0.5

        if np.random.random() < dissolution_prob:
            # Dissolve relationship
            self.peer_institutions.remove(peer_id)
            peer.peer_institutions.remove(self.unique_id)
            logger.debug(f"Institutional relationship dissolved: {self.unique_id} <-> {peer_id}")

    def _evolve_culture_and_policies(self):
        """Evolve institutional culture and policies over time."""
        # Culture evolution based on member characteristics and external pressures
        if self.members and self.model.schedule.steps % 10 == 0:  # Update every 10 steps
            self._update_culture_from_members()

        # Policy evolution based on performance and external changes
        if self.model.schedule.steps % 20 == 0:  # Update every 20 steps
            self._update_policies()

    def _update_culture_from_members(self):
        """Update institutional culture based on member characteristics."""
        if not self.members:
            return

        member_agents = [self.model.get_agent_by_id(mid) for mid in self.members]
        member_agents = [a for a in member_agents if a is not None]

        if not member_agents:
            return

        # Calculate average member characteristics
        avg_openness = np.mean([a.attributes.openness for a in member_agents])
        avg_agreeableness = np.mean([a.attributes.agreeableness for a in member_agents])
        avg_extroversion = np.mean([a.attributes.extroversion for a in member_agents])

        # Gradually adjust culture towards member averages
        adjustment_rate = 0.05

        self.culture.openness_to_change = (
            (1 - adjustment_rate) * self.culture.openness_to_change +
            adjustment_rate * avg_openness
        )

        self.culture.communication_openness = (
            (1 - adjustment_rate) * self.culture.communication_openness +
            adjustment_rate * (avg_extroversion + avg_agreeableness) / 2
        )

    def _update_policies(self):
        """Update institutional policies based on performance and needs."""
        # Update policy compliance rates
        for policy in self.policies.values():
            # Simulate policy compliance evolution
            base_compliance = 0.7
            enforcement_effect = policy.enforcement_level * 0.3
            culture_effect = self.culture.conformity_pressure * 0.2

            new_compliance = np.clip(
                base_compliance + enforcement_effect + culture_effect + np.random.normal(0, 0.1),
                0, 1
            )
            policy.compliance_rate = new_compliance

    def _gauge_member_support(self, decision: Dict) -> float:
        """Gauge member support for a decision."""
        if not self.members:
            return 0.5

        # Simplified support calculation
        decision_type = decision.get('type', 'general')
        support_scores = []

        for member_id in self.members:
            member = self.model.get_agent_by_id(member_id)
            if not member:
                continue

            # Base support based on member characteristics
            support = 0.5

            if decision_type == 'innovation':
                support += member.attributes.openness * 0.3
                support += member.attributes.innovation_adoption * 0.2

            elif decision_type == 'policy_change':
                support += (1 - member.attributes.conformity_tendency) * 0.3
                support += member.attributes.openness * 0.2

            support_scores.append(np.clip(support, 0, 1))

        return np.mean(support_scores) if support_scores else 0.5

    def _update_institutional_metrics(self):
        """Update institutional performance and reputation metrics."""
        # Performance based on member satisfaction and coordination success
        member_satisfaction = self._calculate_member_satisfaction()
        coordination_success_rate = (
            self.successful_coordinations / max(1, self.coordination_attempts)
        )

        self.performance_metrics['member_satisfaction'] = member_satisfaction
        self.performance_metrics['coordination_success'] = coordination_success_rate
        self.performance_metrics['resource_efficiency'] = self.resource_level

        # Update reputation based on performance
        performance_score = np.mean(list(self.performance_metrics.values()))
        reputation_change = (performance_score - 0.5) * 0.1
        self.institutional_reputation = np.clip(
            self.institutional_reputation + reputation_change, 0, 1
        )

        # Resource regeneration/depletion
        resource_change = np.random.normal(0.02, 0.05)  # Small random fluctuation
        self.resource_level = np.clip(self.resource_level + resource_change, 0, 2)

    def _calculate_member_satisfaction(self) -> float:
        """Calculate average member satisfaction with institution."""
        if not self.members:
            return 0.5

        satisfaction_scores = []
        for member_id in self.members:
            member = self.model.get_agent_by_id(member_id)
            if member and hasattr(member, 'social_satisfaction'):
                satisfaction_scores.append(member.social_satisfaction)

        return np.mean(satisfaction_scores) if satisfaction_scores else 0.5

    # Public methods for member management

    def add_member(self, agent_id: int, role: str = 'member') -> bool:
        """Add a member to the institution."""
        if len(self.members) >= self.capacity:
            return False

        self.members.add(agent_id)
        self.member_roles[agent_id] = role

        logger.debug(f"Added agent {agent_id} to institution {self.unique_id} as {role}")
        return True

    def remove_member(self, agent_id: int) -> bool:
        """Remove a member from the institution."""
        if agent_id in self.members:
            self.members.remove(agent_id)
            if agent_id in self.member_roles:
                del self.member_roles[agent_id]
            return True
        return False

    def promote_member(self, agent_id: int, new_role: str) -> bool:
        """Promote a member to a new role."""
        if agent_id in self.members:
            old_role = self.member_roles.get(agent_id, 'member')
            self.member_roles[agent_id] = new_role
            logger.debug(f"Promoted agent {agent_id} from {old_role} to {new_role}")
            return True
        return False

    def create_policy(self, policy_id: str, description: str, enforcement_level: float = 0.5):
        """Create a new institutional policy."""
        policy = InstitutionalPolicy(
            policy_id=policy_id,
            description=description,
            enforcement_level=enforcement_level,
            adoption_date=self.model.schedule.steps
        )
        self.policies[policy_id] = policy
        logger.debug(f"Created policy '{policy_id}' in institution {self.unique_id}")

    def propose_decision(self, decision_type: str, **kwargs):
        """Propose a decision for institutional consideration."""
        decision = {
            'type': decision_type,
            'proposal_step': self.model.schedule.steps,
            **kwargs
        }
        self.pending_decisions.append(decision)

    def get_institutional_summary(self) -> Dict[str, Any]:
        """Get comprehensive institutional state summary."""
        return {
            'id': self.unique_id,
            'name': self.institution_name,
            'type': self.institution_type.value,
            'members': len(self.members),
            'capacity': self.capacity,
            'resource_level': self.resource_level,
            'reputation': self.institutional_reputation,
            'culture': {
                'openness_to_change': self.culture.openness_to_change,
                'hierarchy_strength': self.culture.hierarchy_strength,
                'collective_orientation': self.culture.collective_orientation,
                'innovation_tolerance': self.culture.innovation_tolerance,
                'conformity_pressure': self.culture.conformity_pressure,
                'communication_openness': self.culture.communication_openness
            },
            'performance': self.performance_metrics,
            'policies': len(self.policies),
            'peer_institutions': len(self.peer_institutions),
            'coordination_success_rate': (
                self.successful_coordinations / max(1, self.coordination_attempts)
            ),
            'activities': len(self.collective_activities)
        }


def create_institutional_network(
    institutions: List[InstitutionalAgent],
    network_density: float = 0.1
) -> nx.Graph:
    """
    Create network between institutional agents.

    Args:
        institutions: List of institutional agents
        network_density: Density of institutional network

    Returns:
        NetworkX graph representing institutional relationships
    """
    n_institutions = len(institutions)
    institutional_network = nx.Graph()

    # Add nodes
    for inst in institutions:
        institutional_network.add_node(inst.unique_id, **inst.get_institutional_summary())

    # Add edges based on compatibility and density
    n_edges = int(network_density * n_institutions * (n_institutions - 1) / 2)

    edges_added = 0
    attempts = 0
    max_attempts = n_edges * 10

    while edges_added < n_edges and attempts < max_attempts:
        inst1, inst2 = np.random.choice(institutions, 2, replace=False)

        if not institutional_network.has_edge(inst1.unique_id, inst2.unique_id):
            compatibility = inst1._calculate_institutional_compatibility(inst2)

            if np.random.random() < compatibility:
                institutional_network.add_edge(inst1.unique_id, inst2.unique_id,
                                             weight=compatibility)
                inst1.peer_institutions.add(inst2.unique_id)
                inst2.peer_institutions.add(inst1.unique_id)
                edges_added += 1

        attempts += 1

    logger.info(f"Created institutional network with {edges_added} edges among {n_institutions} institutions")
    return institutional_network


if __name__ == "__main__":
    # Example usage and testing
    logging.basicConfig(level=logging.INFO)

    # Create test institutions
    class MockModel:
        def __init__(self):
            self.schedule = type('Schedule', (), {'steps': 0})()
        def get_agent_by_id(self, agent_id):
            return None

    model = MockModel()

    institutions = [
        InstitutionalAgent(
            unique_id=100,
            model=model,
            institution_type=InstitutionType.SCHOOL,
            institution_name="Central High School",
            capacity=500
        ),
        InstitutionalAgent(
            unique_id=101,
            model=model,
            institution_type=InstitutionType.WORKPLACE,
            institution_name="Tech Corp",
            capacity=200
        ),
        InstitutionalAgent(
            unique_id=102,
            model=model,
            institution_type=InstitutionType.COMMUNITY_GROUP,
            institution_name="Neighborhood Association",
            capacity=50
        )
    ]

    # Create institutional network
    inst_network = create_institutional_network(institutions, network_density=0.3)

    print(f"Created {len(institutions)} institutions")
    print(f"Institutional network has {inst_network.number_of_edges()} connections")

    # Show institutional summaries
    for inst in institutions:
        summary = inst.get_institutional_summary()
        print(f"{summary['name']}: {summary['type']}, capacity {summary['capacity']}")