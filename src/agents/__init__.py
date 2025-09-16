"""
Agent definitions and behaviors for ABM simulations.

This module contains agent classes and behavior specifications for
various social actors in agent-based models.
"""

from .social_agent import (
    SocialAgent,
    AgentAttributes,
    NetworkPosition,
    BehaviorState,
    create_realistic_agent_attributes,
    create_agent_population
)

from .influence_agent import (
    InfluenceAgent,
    OpinionDynamicsParameters,
    InfluenceEvent,
    InfluenceType,
    create_opinion_leaders,
    analyze_opinion_polarization
)

from .institutional_agent import (
    InstitutionalAgent,
    InstitutionType,
    InstitutionalPolicy,
    InstitutionalCulture,
    create_institutional_network
)

__all__ = [
    'SocialAgent',
    'AgentAttributes',
    'NetworkPosition',
    'BehaviorState',
    'create_realistic_agent_attributes',
    'create_agent_population',
    'InfluenceAgent',
    'OpinionDynamicsParameters',
    'InfluenceEvent',
    'InfluenceType',
    'create_opinion_leaders',
    'analyze_opinion_polarization',
    'InstitutionalAgent',
    'InstitutionType',
    'InstitutionalPolicy',
    'InstitutionalCulture',
    'create_institutional_network'
]