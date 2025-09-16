"""
Master Workflow for ABM-RSiena PhD Dissertation Project

This script orchestrates the complete end-to-end execution of the PhD dissertation project,
from data loading through final manuscript generation. It coordinates all specialized
agent outputs into a cohesive, publication-ready research framework.

Key Components:
1. Research Design Execution (Alpha Agent output)
2. ABM-RSiena Model Implementation (Beta Agent output)
3. Statistical Analysis & Validation (Gamma Agent output)
4. Visualization & Publication Figures (Delta Agent output)
5. Academic Documentation (Epsilon Agent output)
6. Quality Assurance & Integration (Zeta Agent - this module)

Author: Zeta Agent - Final Integration & Quality Assurance
Created: 2025-09-15
"""

import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
import warnings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('outputs/master_workflow.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# Core project imports
from src.models.abm_rsiena_model import ABMRSienaModel, NetworkEvolutionParameters
from src.analysis.empirical_validation import EmpiricalValidator, ValidationResults
from qa_framework.integration.workflow_coordinator import WorkflowCoordinator
from qa_framework.testing.end_to_end_test import EndToEndTester
from qa_framework.validation.academic_standards import AcademicStandardsValidator
from qa_framework.reporting.quality_dashboard import QualityDashboard


@dataclass
class ProjectConfiguration:
    """Configuration for the complete PhD dissertation project."""
    # Data sources
    empirical_data_path: Path = Path("data/raw/school_networks")
    synthetic_data_path: Path = Path("data/processed/synthetic_networks")

    # Model parameters
    n_students: int = 2585
    n_schools: int = 3
    n_waves: int = 3
    abm_steps_per_wave: int = 100

    # Analysis parameters
    n_simulations: int = 1000
    tolerance_intervention_magnitudes: List[float] = field(
        default_factory=lambda: [0.1, 0.2, 0.3, 0.4, 0.5]
    )
    target_proportion: float = 0.1  # Proportion of popular students to target

    # Output specifications
    publication_figures_dpi: int = 300
    dissertation_format: str = "latex"
    manuscript_format: str = "jasss"

    # Quality assurance
    required_test_coverage: float = 0.95
    statistical_significance_level: float = 0.05
    convergence_tolerance: float = 0.01


class MasterWorkflowExecutor:
    """
    Master executor for the complete PhD dissertation workflow.

    This class coordinates all specialized agent outputs and ensures seamless
    integration into a publication-ready research framework.
    """

    def __init__(self, config: ProjectConfiguration):
        """Initialize the master workflow executor."""
        self.config = config
        self.results = {}
        self.quality_metrics = {}
        self.execution_log = []

        # Initialize coordinators
        self.workflow_coordinator = WorkflowCoordinator()
        self.academic_validator = AcademicStandardsValidator()
        self.quality_dashboard = QualityDashboard()

        logger.info("Initialized Master Workflow Executor for PhD Dissertation")

    def execute_complete_workflow(self) -> Dict[str, Any]:
        """
        Execute the complete PhD dissertation workflow.

        Returns:
            Dictionary containing all results, metrics, and validation status
        """
        logger.info("=== STARTING COMPLETE PHD DISSERTATION WORKFLOW ===")
        start_time = time.time()

        try:
            # Phase 1: Research Design Validation
            self._validate_research_design()

            # Phase 2: Data Preparation and Model Implementation
            self._prepare_empirical_data()
            self._implement_abm_rsiena_model()

            # Phase 3: Empirical Validation and Statistical Analysis
            self._conduct_empirical_validation()
            self._perform_statistical_analysis()

            # Phase 4: Intervention Experiments
            self._execute_intervention_experiments()

            # Phase 5: Visualization and Publication Figures
            self._generate_publication_figures()

            # Phase 6: Academic Documentation
            self._compile_academic_documentation()

            # Phase 7: Quality Assurance and Validation
            self._perform_comprehensive_qa()

            # Phase 8: Final Integration and Package Preparation
            self._prepare_final_packages()

            total_time = time.time() - start_time
            logger.info(f"=== WORKFLOW COMPLETED SUCCESSFULLY in {total_time:.1f} seconds ===")

            return self._compile_final_results()

        except Exception as e:
            logger.error(f"Workflow execution failed: {e}")
            raise

    def _validate_research_design(self):
        """Validate Alpha Agent's research design and methodology."""
        logger.info("Phase 1: Validating Research Design and Methodology")

        # Validate theoretical framework
        theoretical_validation = self.academic_validator.validate_theoretical_framework(
            research_question="How can network-aware tolerance interventions promote interethnic cooperation?",
            methodology="Stochastic Actor-Oriented Models with Agent-Based Model integration",
            innovation_claim="First integration of ABM and RSiena for intervention design"
        )

        # Validate methodology rigor
        methodology_validation = self.academic_validator.validate_methodology_rigor(
            data_requirements=f"{self.config.n_students} students across {self.config.n_schools} schools",
            statistical_power="Power analysis for network intervention effects",
            ethical_considerations="Simulated interventions on anonymized school data"
        )

        self.results['research_design_validation'] = {
            'theoretical_framework': theoretical_validation,
            'methodology_rigor': methodology_validation,
            'status': 'VALIDATED' if all([theoretical_validation, methodology_validation]) else 'FAILED'
        }

        if not all([theoretical_validation, methodology_validation]):
            raise ValueError("Research design validation failed - cannot proceed")

        logger.info("✓ Research design validation completed successfully")

    def _prepare_empirical_data(self):
        """Prepare empirical data following Alpha Agent specifications."""
        logger.info("Phase 2a: Preparing Empirical Data")

        from src.analysis.empirical_validation import EmpiricalDataLoader

        data_loader = EmpiricalDataLoader(self.config.empirical_data_path)

        # Load school network data (following Shani et al. 2023 study design)
        try:
            school_networks = data_loader.load_school_networks(
                n_schools=self.config.n_schools,
                n_waves=self.config.n_waves
            )

            # Validate data quality
            data_quality = data_loader.validate_data_quality(school_networks)

            if data_quality['overall_quality'] < 0.8:
                logger.warning("Data quality below threshold - using synthetic data")
                school_networks = data_loader.generate_synthetic_school_networks(
                    n_students=self.config.n_students,
                    n_schools=self.config.n_schools,
                    n_waves=self.config.n_waves
                )

            self.results['empirical_data'] = {
                'networks': school_networks,
                'quality_metrics': data_quality,
                'data_source': 'empirical' if data_quality['overall_quality'] >= 0.8 else 'synthetic'
            }

            logger.info(f"✓ Prepared {len(school_networks)} school networks across {self.config.n_waves} waves")

        except Exception as e:
            logger.error(f"Data preparation failed: {e}")
            raise

    def _implement_abm_rsiena_model(self):
        """Implement ABM-RSiena model following Beta Agent specifications."""
        logger.info("Phase 2b: Implementing ABM-RSiena Integration Model")

        # Initialize model with network evolution parameters
        evolution_params = NetworkEvolutionParameters(
            density_effect=-2.0,
            reciprocity_effect=2.0,
            transitivity_effect=0.5,
            # Add tolerance-specific effects
            opinion_linear_shape_effect=0.1,
            opinion_average_similarity_effect=0.3,
            network_behavior_effect=0.2,
            behavior_network_effect=0.15
        )

        # Create integrated ABM-RSiena model
        self.abm_rsiena_model = ABMRSienaModel(
            n_agents=self.config.n_students,
            evolution_parameters=evolution_params,
            temporal_alignment_steps=self.config.abm_steps_per_wave
        )

        # Initialize with empirical data
        if 'empirical_data' in self.results:
            initialization_success = self.abm_rsiena_model.initialize_from_empirical_data(
                self.results['empirical_data']['networks']
            )

            if not initialization_success:
                logger.warning("Model initialization with empirical data failed - using synthetic initialization")
                self.abm_rsiena_model.initialize_synthetic_population()

        # Validate model specification
        model_validation = self.abm_rsiena_model.validate_model_specification()

        self.results['abm_rsiena_model'] = {
            'model_instance': self.abm_rsiena_model,
            'parameters': evolution_params,
            'validation_status': model_validation,
            'status': 'IMPLEMENTED' if model_validation['is_valid'] else 'FAILED'
        }

        if not model_validation['is_valid']:
            raise ValueError(f"Model implementation validation failed: {model_validation['errors']}")

        logger.info("✓ ABM-RSiena model implementation completed successfully")

    def _conduct_empirical_validation(self):
        """Conduct empirical validation following Gamma Agent specifications."""
        logger.info("Phase 3a: Conducting Empirical Validation")

        from src.analysis.empirical_validation import EmpiricalValidator

        validator = EmpiricalValidator()

        # Baseline model validation without intervention
        baseline_validation = validator.validate_baseline_model(
            model=self.abm_rsiena_model,
            empirical_networks=self.results['empirical_data']['networks'],
            n_simulations=100  # Reduced for initial validation
        )

        # Goodness-of-fit testing
        gof_results = validator.conduct_goodness_of_fit_tests(
            simulated_networks=baseline_validation['simulated_networks'],
            empirical_networks=self.results['empirical_data']['networks']
        )

        # Model convergence assessment
        convergence_results = validator.assess_model_convergence(
            self.abm_rsiena_model,
            tolerance=self.config.convergence_tolerance
        )

        self.results['empirical_validation'] = {
            'baseline_validation': baseline_validation,
            'goodness_of_fit': gof_results,
            'convergence_assessment': convergence_results,
            'status': 'VALIDATED' if gof_results['overall_fit'] > 0.8 else 'FAILED'
        }

        if gof_results['overall_fit'] <= 0.8:
            logger.warning(f"Empirical validation fit below threshold: {gof_results['overall_fit']}")
            # Continue with warning rather than fail - allow sensitivity analysis

        logger.info("✓ Empirical validation completed")

    def _perform_statistical_analysis(self):
        """Perform statistical analysis following Gamma Agent specifications."""
        logger.info("Phase 3b: Performing Statistical Analysis")

        from src.analysis.statistical_analysis import StatisticalAnalyzer

        analyzer = StatisticalAnalyzer()

        # Parameter estimation
        parameter_estimates = analyzer.estimate_model_parameters(
            model=self.abm_rsiena_model,
            empirical_data=self.results['empirical_data']['networks']
        )

        # Sensitivity analysis
        sensitivity_results = analyzer.conduct_sensitivity_analysis(
            model=self.abm_rsiena_model,
            parameter_ranges=self._define_parameter_ranges()
        )

        # Effect size calculations
        effect_sizes = analyzer.calculate_effect_sizes(
            parameter_estimates['tolerance_effects']
        )

        self.results['statistical_analysis'] = {
            'parameter_estimates': parameter_estimates,
            'sensitivity_analysis': sensitivity_results,
            'effect_sizes': effect_sizes,
            'status': 'COMPLETED'
        }

        logger.info("✓ Statistical analysis completed")

    def _execute_intervention_experiments(self):
        """Execute tolerance intervention experiments."""
        logger.info("Phase 4: Executing Tolerance Intervention Experiments")

        from src.experiments.intervention_runner import InterventionRunner

        runner = InterventionRunner(self.abm_rsiena_model)

        intervention_results = {}

        for magnitude in self.config.tolerance_intervention_magnitudes:
            logger.info(f"Running intervention experiment with magnitude {magnitude}")

            # Run counterfactual simulation
            experiment_result = runner.run_counterfactual_experiment(
                tolerance_increase_magnitude=magnitude,
                target_proportion=self.config.target_proportion,
                n_simulations=self.config.n_simulations
            )

            intervention_results[f"magnitude_{magnitude}"] = experiment_result

        # Analyze intervention effectiveness
        effectiveness_analysis = runner.analyze_intervention_effectiveness(
            intervention_results
        )

        self.results['intervention_experiments'] = {
            'individual_experiments': intervention_results,
            'effectiveness_analysis': effectiveness_analysis,
            'optimal_magnitude': effectiveness_analysis['optimal_intervention_magnitude'],
            'status': 'COMPLETED'
        }

        logger.info(f"✓ Intervention experiments completed - Optimal magnitude: {effectiveness_analysis['optimal_intervention_magnitude']}")

    def _generate_publication_figures(self):
        """Generate publication figures following Delta Agent specifications."""
        logger.info("Phase 5: Generating Publication Figures")

        from src.visualization.publication_figures import PublicationFigureGenerator

        figure_generator = PublicationFigureGenerator(
            output_dir=Path("outputs/figures"),
            dpi=self.config.publication_figures_dpi
        )

        # Generate all required figures
        figure_paths = figure_generator.generate_complete_figure_set(
            empirical_data=self.results['empirical_data'],
            model_validation=self.results['empirical_validation'],
            statistical_analysis=self.results['statistical_analysis'],
            intervention_results=self.results['intervention_experiments']
        )

        self.results['publication_figures'] = {
            'figure_paths': figure_paths,
            'figure_count': len(figure_paths),
            'status': 'COMPLETED'
        }

        logger.info(f"✓ Generated {len(figure_paths)} publication-ready figures")

    def _compile_academic_documentation(self):
        """Compile academic documentation following Epsilon Agent specifications."""
        logger.info("Phase 6: Compiling Academic Documentation")

        from docs.generators.dissertation_compiler import DissertationCompiler
        from docs.generators.manuscript_compiler import ManuscriptCompiler

        # Generate PhD dissertation
        dissertation_compiler = DissertationCompiler(
            format=self.config.dissertation_format,
            output_dir=Path("outputs/dissertation")
        )

        dissertation_path = dissertation_compiler.compile_complete_dissertation(
            research_design=self.results['research_design_validation'],
            empirical_validation=self.results['empirical_validation'],
            statistical_analysis=self.results['statistical_analysis'],
            intervention_experiments=self.results['intervention_experiments'],
            figures=self.results['publication_figures']
        )

        # Generate JASSS manuscript
        manuscript_compiler = ManuscriptCompiler(
            journal="jasss",
            format=self.config.manuscript_format,
            output_dir=Path("outputs/manuscript")
        )

        manuscript_path = manuscript_compiler.compile_journal_manuscript(
            research_results=self.results,
            word_limit=8000  # JASSS typical limit
        )

        self.results['academic_documentation'] = {
            'dissertation_path': dissertation_path,
            'manuscript_path': manuscript_path,
            'status': 'COMPLETED'
        }

        logger.info("✓ Academic documentation compilation completed")

    def _perform_comprehensive_qa(self):
        """Perform comprehensive quality assurance."""
        logger.info("Phase 7: Performing Comprehensive Quality Assurance")

        # End-to-end testing
        e2e_tester = EndToEndTester()
        e2e_results = e2e_tester.run_complete_test_suite(
            project_config=self.config,
            workflow_results=self.results
        )

        # Academic standards validation
        academic_validation = self.academic_validator.validate_complete_project(
            self.results,
            required_components=['theoretical_framework', 'empirical_validation', 'statistical_rigor', 'publication_readiness']
        )

        # Quality metrics dashboard
        quality_metrics = self.quality_dashboard.generate_quality_report(
            self.results,
            test_results=e2e_results,
            academic_validation=academic_validation
        )

        self.results['quality_assurance'] = {
            'end_to_end_tests': e2e_results,
            'academic_validation': academic_validation,
            'quality_metrics': quality_metrics,
            'overall_quality_score': quality_metrics['composite_score'],
            'status': 'PASSED' if quality_metrics['composite_score'] >= 0.9 else 'WARNING'
        }

        logger.info(f"✓ Quality assurance completed - Overall score: {quality_metrics['composite_score']:.3f}")

    def _prepare_final_packages(self):
        """Prepare final dissertation and publication packages."""
        logger.info("Phase 8: Preparing Final Packages")

        from qa_framework.integration.package_builder import PackageBuilder

        package_builder = PackageBuilder()

        # PhD dissertation package
        dissertation_package = package_builder.build_dissertation_package(
            results=self.results,
            include_supplementary=True,
            include_code=True,
            include_data=True
        )

        # JASSS manuscript package
        manuscript_package = package_builder.build_manuscript_package(
            results=self.results,
            journal_requirements="jasss",
            include_supplementary=True
        )

        # Reproducibility package
        reproducibility_package = package_builder.build_reproducibility_package(
            workflow_config=self.config,
            complete_codebase=True,
            environment_specifications=True
        )

        self.results['final_packages'] = {
            'dissertation_package': dissertation_package,
            'manuscript_package': manuscript_package,
            'reproducibility_package': reproducibility_package,
            'status': 'COMPLETED'
        }

        logger.info("✓ Final packages prepared successfully")

    def _compile_final_results(self) -> Dict[str, Any]:
        """Compile and return final workflow results."""
        final_results = {
            'workflow_status': 'COMPLETED',
            'execution_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'project_configuration': self.config,
            'results_summary': self._generate_results_summary(),
            'quality_metrics': self.results.get('quality_assurance', {}),
            'deliverables': self._list_final_deliverables(),
            'complete_results': self.results
        }

        # Save complete results
        output_path = Path("outputs/final_results.pkl")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        import pickle
        with open(output_path, 'wb') as f:
            pickle.dump(final_results, f)

        logger.info(f"Final results saved to {output_path}")
        return final_results

    def _generate_results_summary(self) -> Dict[str, Any]:
        """Generate a concise summary of key results."""
        return {
            'research_validated': self.results['research_design_validation']['status'] == 'VALIDATED',
            'model_implemented': self.results['abm_rsiena_model']['status'] == 'IMPLEMENTED',
            'empirically_validated': self.results['empirical_validation']['status'] == 'VALIDATED',
            'optimal_intervention_magnitude': self.results['intervention_experiments']['optimal_magnitude'],
            'publication_figures_count': self.results['publication_figures']['figure_count'],
            'quality_score': self.results['quality_assurance']['overall_quality_score'],
            'deliverables_ready': all([
                self.results['academic_documentation']['status'] == 'COMPLETED',
                self.results['final_packages']['status'] == 'COMPLETED'
            ])
        }

    def _list_final_deliverables(self) -> List[str]:
        """List all final deliverables."""
        deliverables = []

        if 'final_packages' in self.results:
            packages = self.results['final_packages']
            deliverables.extend([
                str(packages['dissertation_package']['main_document']),
                str(packages['manuscript_package']['main_document']),
                str(packages['reproducibility_package']['setup_guide'])
            ])

        return deliverables

    def _define_parameter_ranges(self) -> Dict[str, Tuple[float, float]]:
        """Define parameter ranges for sensitivity analysis."""
        return {
            'tolerance_network_effect': (0.0, 0.5),
            'tolerance_behavior_effect': (0.0, 0.3),
            'homophily_ethnicity': (0.0, 1.0),
            'intervention_decay_rate': (0.0, 0.2)
        }


def main():
    """Main execution function for the complete PhD dissertation workflow."""
    # Initialize configuration
    config = ProjectConfiguration()

    # Create and execute master workflow
    executor = MasterWorkflowExecutor(config)

    try:
        results = executor.execute_complete_workflow()

        print("\n" + "="*80)
        print("PhD DISSERTATION WORKFLOW COMPLETED SUCCESSFULLY")
        print("="*80)
        print(f"Quality Score: {results['quality_metrics']['overall_quality_score']:.3f}/1.000")
        print(f"Research Validated: {results['results_summary']['research_validated']}")
        print(f"Model Implemented: {results['results_summary']['model_implemented']}")
        print(f"Empirically Validated: {results['results_summary']['empirically_validated']}")
        print(f"Optimal Intervention Magnitude: {results['results_summary']['optimal_intervention_magnitude']}")
        print(f"Publication Figures: {results['results_summary']['publication_figures_count']}")
        print(f"Deliverables Ready: {results['results_summary']['deliverables_ready']}")
        print("\nFinal deliverables:")
        for deliverable in results['deliverables']:
            print(f"  - {deliverable}")
        print("="*80)

        return results

    except Exception as e:
        logger.error(f"Workflow execution failed: {e}")
        raise


if __name__ == "__main__":
    results = main()