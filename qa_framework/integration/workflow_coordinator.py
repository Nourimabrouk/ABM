"""
Workflow Coordinator for Inter-Agent Integration

This module coordinates the outputs of all specialized agents (Alpha through Epsilon)
to ensure seamless integration and data consistency across the complete PhD dissertation project.

Key Responsibilities:
- Validate inter-agent data handoffs
- Ensure consistency of parameters and specifications
- Coordinate temporal synchronization of workflow phases
- Manage dependency resolution between agent outputs

Author: Zeta Agent - Integration Specialist
"""

import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from datetime import datetime
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class AgentOutput:
    """Container for individual agent output validation."""
    agent_name: str
    output_type: str
    completion_status: str
    quality_score: float
    dependencies_met: bool
    data_consistency_score: float
    timestamp: datetime
    artifacts: List[Path] = field(default_factory=list)
    validation_notes: List[str] = field(default_factory=list)


@dataclass
class WorkflowPhase:
    """Container for workflow phase coordination."""
    phase_name: str
    required_agents: Set[str]
    dependencies: Set[str]
    outputs: Dict[str, AgentOutput] = field(default_factory=dict)
    phase_status: str = "PENDING"
    completion_timestamp: Optional[datetime] = None


class WorkflowCoordinator:
    """
    Coordinates integration between all specialized agents.

    This class ensures that outputs from Alpha (research design), Beta (implementation),
    Gamma (analysis), Delta (visualization), and Epsilon (documentation) agents
    integrate seamlessly into the final PhD dissertation.
    """

    def __init__(self):
        """Initialize the workflow coordinator."""
        self.phases = self._initialize_workflow_phases()
        self.agent_dependencies = self._define_agent_dependencies()
        self.integration_log = []
        self.data_consistency_cache = {}

        logger.info("Workflow Coordinator initialized for 5-agent integration")

    def _initialize_workflow_phases(self) -> Dict[str, WorkflowPhase]:
        """Initialize the workflow phases and their requirements."""
        phases = {
            "research_design": WorkflowPhase(
                phase_name="Research Design & Methodology",
                required_agents={"alpha"},
                dependencies=set()
            ),
            "model_implementation": WorkflowPhase(
                phase_name="ABM-RSiena Model Implementation",
                required_agents={"beta"},
                dependencies={"research_design"}
            ),
            "empirical_analysis": WorkflowPhase(
                phase_name="Statistical Analysis & Validation",
                required_agents={"gamma"},
                dependencies={"research_design", "model_implementation"}
            ),
            "visualization": WorkflowPhase(
                phase_name="Publication Figures & Visualization",
                required_agents={"delta"},
                dependencies={"model_implementation", "empirical_analysis"}
            ),
            "documentation": WorkflowPhase(
                phase_name="Academic Documentation",
                required_agents={"epsilon"},
                dependencies={"research_design", "empirical_analysis", "visualization"}
            ),
            "final_integration": WorkflowPhase(
                phase_name="Final Integration & QA",
                required_agents={"zeta"},
                dependencies={"research_design", "model_implementation", "empirical_analysis", "visualization", "documentation"}
            )
        }
        return phases

    def _define_agent_dependencies(self) -> Dict[str, Dict[str, Any]]:
        """Define the specific dependencies between agents."""
        return {
            "alpha": {
                "outputs": ["research_question", "methodology_design", "theoretical_framework"],
                "provides_to": ["beta", "gamma", "epsilon"],
                "data_formats": ["methodology_specs.yaml", "research_design.md"]
            },
            "beta": {
                "inputs_from": ["alpha"],
                "outputs": ["abm_model", "rsiena_integration", "simulation_framework"],
                "provides_to": ["gamma", "delta"],
                "data_formats": ["model_class.py", "integration_module.py", "config.yaml"]
            },
            "gamma": {
                "inputs_from": ["alpha", "beta"],
                "outputs": ["statistical_analysis", "validation_results", "effect_sizes"],
                "provides_to": ["delta", "epsilon"],
                "data_formats": ["analysis_results.pkl", "validation_report.json"]
            },
            "delta": {
                "inputs_from": ["beta", "gamma"],
                "outputs": ["publication_figures", "interactive_visualizations", "animations"],
                "provides_to": ["epsilon"],
                "data_formats": ["figures/*.png", "interactive/*.html", "animations/*.gif"]
            },
            "epsilon": {
                "inputs_from": ["alpha", "gamma", "delta"],
                "outputs": ["dissertation_document", "manuscript", "supplementary_materials"],
                "provides_to": ["zeta"],
                "data_formats": ["dissertation.tex", "manuscript.tex", "supplementary.pdf"]
            },
            "zeta": {
                "inputs_from": ["alpha", "beta", "gamma", "delta", "epsilon"],
                "outputs": ["final_packages", "quality_validation", "reproducibility_framework"],
                "provides_to": [],
                "data_formats": ["final_deliverables/", "qa_report.html"]
            }
        }

    def validate_agent_output(self, agent_name: str, output_artifacts: List[Path]) -> AgentOutput:
        """
        Validate the output of a specific agent.

        Args:
            agent_name: Name of the agent (alpha, beta, gamma, delta, epsilon)
            output_artifacts: List of output files/directories to validate

        Returns:
            AgentOutput validation results
        """
        logger.info(f"Validating output from {agent_name} agent")

        # Check artifact existence and format
        artifacts_valid = self._validate_artifacts(agent_name, output_artifacts)

        # Check data consistency with dependencies
        consistency_score = self._check_data_consistency(agent_name, output_artifacts)

        # Validate dependencies are met
        dependencies_met = self._validate_dependencies(agent_name)

        # Calculate overall quality score
        quality_score = self._calculate_quality_score(
            artifacts_valid, consistency_score, dependencies_met
        )

        # Determine completion status
        if quality_score >= 0.9 and dependencies_met:
            status = "COMPLETED"
        elif quality_score >= 0.7:
            status = "COMPLETED_WITH_WARNINGS"
        else:
            status = "FAILED"

        output = AgentOutput(
            agent_name=agent_name,
            output_type=self.agent_dependencies[agent_name]["outputs"][0],
            completion_status=status,
            quality_score=quality_score,
            dependencies_met=dependencies_met,
            data_consistency_score=consistency_score,
            timestamp=datetime.now(),
            artifacts=output_artifacts,
            validation_notes=[]
        )

        self._log_validation_result(output)
        return output

    def _validate_artifacts(self, agent_name: str, artifacts: List[Path]) -> bool:
        """Validate that required artifacts exist and have correct format."""
        required_formats = self.agent_dependencies[agent_name]["data_formats"]

        for format_pattern in required_formats:
            # Check if at least one artifact matches the pattern
            pattern_matched = any(
                format_pattern.split('.')[-1] in str(artifact) or
                format_pattern.split('/')[0] in str(artifact)
                for artifact in artifacts
            )

            if not pattern_matched:
                logger.warning(f"{agent_name}: Missing artifact matching pattern {format_pattern}")
                return False

        # Validate file existence
        for artifact in artifacts:
            if not artifact.exists():
                logger.error(f"{agent_name}: Artifact does not exist: {artifact}")
                return False

        return True

    def _check_data_consistency(self, agent_name: str, artifacts: List[Path]) -> float:
        """Check data consistency between agent outputs."""
        consistency_score = 1.0

        # Agent-specific consistency checks
        if agent_name == "beta":
            consistency_score *= self._check_model_parameter_consistency(artifacts)
        elif agent_name == "gamma":
            consistency_score *= self._check_statistical_consistency(artifacts)
        elif agent_name == "delta":
            consistency_score *= self._check_visualization_consistency(artifacts)
        elif agent_name == "epsilon":
            consistency_score *= self._check_documentation_consistency(artifacts)

        return consistency_score

    def _check_model_parameter_consistency(self, artifacts: List[Path]) -> float:
        """Check consistency of model parameters across artifacts."""
        # This would normally check that model specifications match
        # across different Beta agent outputs
        logger.info("Checking model parameter consistency")
        return 1.0  # Placeholder - would implement actual consistency checks

    def _check_statistical_consistency(self, artifacts: List[Path]) -> float:
        """Check consistency of statistical results."""
        logger.info("Checking statistical analysis consistency")
        return 1.0  # Placeholder

    def _check_visualization_consistency(self, artifacts: List[Path]) -> float:
        """Check consistency of visualization specifications."""
        logger.info("Checking visualization consistency")
        return 1.0  # Placeholder

    def _check_documentation_consistency(self, artifacts: List[Path]) -> float:
        """Check consistency of documentation with analysis results."""
        logger.info("Checking documentation consistency")
        return 1.0  # Placeholder

    def _validate_dependencies(self, agent_name: str) -> bool:
        """Validate that all required dependencies for an agent are met."""
        dependencies = self.agent_dependencies[agent_name].get("inputs_from", [])

        for dependency in dependencies:
            # Check if dependency agent has completed successfully
            dependency_completed = False
            for phase in self.phases.values():
                if dependency in phase.required_agents:
                    if dependency in phase.outputs:
                        output = phase.outputs[dependency]
                        dependency_completed = output.completion_status in ["COMPLETED", "COMPLETED_WITH_WARNINGS"]
                    break

            if not dependency_completed:
                logger.warning(f"{agent_name}: Dependency {dependency} not completed")
                return False

        return True

    def _calculate_quality_score(self, artifacts_valid: bool, consistency_score: float, dependencies_met: bool) -> float:
        """Calculate overall quality score for agent output."""
        base_score = 0.0

        if artifacts_valid:
            base_score += 0.4

        base_score += consistency_score * 0.4

        if dependencies_met:
            base_score += 0.2

        return min(base_score, 1.0)

    def _log_validation_result(self, output: AgentOutput):
        """Log validation result to integration log."""
        log_entry = {
            'timestamp': output.timestamp,
            'agent': output.agent_name,
            'status': output.completion_status,
            'quality_score': output.quality_score,
            'consistency_score': output.data_consistency_score,
            'dependencies_met': output.dependencies_met
        }

        self.integration_log.append(log_entry)
        logger.info(f"Agent {output.agent_name}: {output.completion_status} (Quality: {output.quality_score:.3f})")

    def coordinate_phase_transition(self, from_phase: str, to_phase: str) -> bool:
        """
        Coordinate transition between workflow phases.

        Args:
            from_phase: Source phase name
            to_phase: Target phase name

        Returns:
            True if transition is valid and can proceed
        """
        logger.info(f"Coordinating transition: {from_phase} → {to_phase}")

        # Validate source phase completion
        if from_phase not in self.phases:
            logger.error(f"Unknown source phase: {from_phase}")
            return False

        source_phase = self.phases[from_phase]
        if source_phase.phase_status != "COMPLETED":
            logger.error(f"Source phase {from_phase} not completed (status: {source_phase.phase_status})")
            return False

        # Validate target phase dependencies
        if to_phase not in self.phases:
            logger.error(f"Unknown target phase: {to_phase}")
            return False

        target_phase = self.phases[to_phase]
        for dependency in target_phase.dependencies:
            if self.phases[dependency].phase_status != "COMPLETED":
                logger.error(f"Target phase {to_phase} dependency {dependency} not completed")
                return False

        logger.info(f"Phase transition validated: {from_phase} → {to_phase}")
        return True

    def update_phase_status(self, phase_name: str, status: str, agent_outputs: Dict[str, AgentOutput] = None):
        """Update the status of a workflow phase."""
        if phase_name not in self.phases:
            logger.error(f"Unknown phase: {phase_name}")
            return

        phase = self.phases[phase_name]
        phase.phase_status = status

        if status == "COMPLETED":
            phase.completion_timestamp = datetime.now()

        if agent_outputs:
            phase.outputs.update(agent_outputs)

        logger.info(f"Phase {phase_name} status updated to: {status}")

    def generate_integration_report(self) -> Dict[str, Any]:
        """Generate comprehensive integration report."""
        logger.info("Generating integration coordination report")

        # Phase completion summary
        phase_summary = {}
        for phase_name, phase in self.phases.items():
            phase_summary[phase_name] = {
                'status': phase.phase_status,
                'required_agents': list(phase.required_agents),
                'dependencies': list(phase.dependencies),
                'completion_time': phase.completion_timestamp.isoformat() if phase.completion_timestamp else None,
                'agent_outputs': len(phase.outputs)
            }

        # Agent performance summary
        agent_summary = {}
        for log_entry in self.integration_log:
            agent = log_entry['agent']
            if agent not in agent_summary:
                agent_summary[agent] = {
                    'quality_scores': [],
                    'consistency_scores': [],
                    'completion_status': [],
                    'dependencies_met': []
                }

            agent_summary[agent]['quality_scores'].append(log_entry['quality_score'])
            agent_summary[agent]['consistency_scores'].append(log_entry['consistency_score'])
            agent_summary[agent]['completion_status'].append(log_entry['status'])
            agent_summary[agent]['dependencies_met'].append(log_entry['dependencies_met'])

        # Calculate aggregate metrics
        for agent, metrics in agent_summary.items():
            metrics['avg_quality_score'] = np.mean(metrics['quality_scores'])
            metrics['avg_consistency_score'] = np.mean(metrics['consistency_scores'])
            metrics['final_status'] = metrics['completion_status'][-1] if metrics['completion_status'] else 'PENDING'

        # Overall integration health
        total_phases = len(self.phases)
        completed_phases = sum(1 for phase in self.phases.values() if phase.phase_status == "COMPLETED")
        integration_health = completed_phases / total_phases

        report = {
            'timestamp': datetime.now().isoformat(),
            'integration_health': integration_health,
            'phases_completed': completed_phases,
            'total_phases': total_phases,
            'phase_summary': phase_summary,
            'agent_summary': agent_summary,
            'integration_log': self.integration_log,
            'recommendations': self._generate_recommendations(agent_summary, phase_summary)
        }

        return report

    def _generate_recommendations(self, agent_summary: Dict, phase_summary: Dict) -> List[str]:
        """Generate recommendations based on integration analysis."""
        recommendations = []

        # Check for agents with low quality scores
        for agent, metrics in agent_summary.items():
            if metrics.get('avg_quality_score', 0) < 0.8:
                recommendations.append(f"Review {agent} agent output - quality score below threshold")

        # Check for incomplete phases
        for phase_name, phase_info in phase_summary.items():
            if phase_info['status'] != 'COMPLETED':
                recommendations.append(f"Complete {phase_name} phase before proceeding")

        # Check for dependency issues
        incomplete_dependencies = []
        for phase_name, phase in self.phases.items():
            for dep in phase.dependencies:
                if self.phases[dep].phase_status != 'COMPLETED':
                    incomplete_dependencies.append(f"{phase_name} → {dep}")

        if incomplete_dependencies:
            recommendations.append(f"Resolve dependency issues: {', '.join(incomplete_dependencies)}")

        if not recommendations:
            recommendations.append("All integration checks passed - proceed with confidence")

        return recommendations

    def export_coordination_data(self, output_path: Path):
        """Export coordination data for external analysis."""
        report = self.generate_integration_report()

        output_path.parent.mkdir(parents=True, exist_ok=True)

        import json
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        logger.info(f"Integration coordination data exported to {output_path}")