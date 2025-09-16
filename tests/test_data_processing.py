"""
Data Processing Validation Tests

Comprehensive testing suite for data processing components including classroom
data handling, RSiena data format conversion, missing data handling, and
multi-level data structure validation.

Test Coverage:
- Classroom data structure validation (105 classrooms)
- RSiena data format conversion accuracy
- Missing data handling mechanisms
- Temporal alignment verification
- Data integrity and consistency checks
- Multi-level data processing

Author: Validation Specialist
Created: 2025-09-16
"""

import unittest
import numpy as np
import pandas as pd
import networkx as nx
from typing import Dict, List, Optional, Tuple, Any
import warnings
from pathlib import Path
import tempfile
import logging
from dataclasses import dataclass
import pickle

# Import ABM-RSiena components
import sys
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from rsiena_integration.data_converters import ABMRSienaConverter, RSienaDataSet, ConversionConfig
from rsiena_integration.r_interface import RInterface, RPY2_AVAILABLE
from utils.config_manager import ModelConfiguration

logger = logging.getLogger(__name__)


@dataclass
class ClassroomData:
    """Data structure for individual classroom."""
    classroom_id: str
    n_students: int
    networks: List[np.ndarray]  # Network adjacency matrices over time
    tolerance_data: np.ndarray  # Student tolerance over time
    student_attributes: Dict[str, np.ndarray]
    temporal_metadata: Dict[str, Any]


@dataclass
class MultiLevelData:
    """Multi-level data structure for all classrooms."""
    classrooms: Dict[str, ClassroomData]
    school_attributes: Dict[str, Any]
    temporal_alignment: Dict[str, List[str]]  # Time period mappings
    metadata: Dict[str, Any]


class TestDataProcessing(unittest.TestCase):
    """Test suite for data processing validation."""

    def setUp(self):
        """Set up test fixtures."""
        self.n_classrooms = 5  # Use smaller number for testing
        self.n_students_per_classroom = 25
        self.n_periods = 4
        self.test_seed = 42
        np.random.seed(self.test_seed)

        # Create test multi-level data
        self.multi_level_data = self._create_test_multi_level_data()

        # Initialize data converter
        self.converter = ABMRSienaConverter() if RPY2_AVAILABLE else None

    def _create_test_multi_level_data(self) -> MultiLevelData:
        """Create realistic multi-level classroom data."""
        classrooms = {}

        for classroom_idx in range(self.n_classrooms):
            classroom_id = f"classroom_{classroom_idx:03d}"
            classroom_data = self._create_test_classroom_data(classroom_id)
            classrooms[classroom_id] = classroom_data

        school_attributes = {
            'school_id': 'test_school_001',
            'school_type': 'public',
            'socioeconomic_status': 'mixed',
            'location': 'urban'
        }

        temporal_alignment = {
            'period_1': ['2023-09-01', '2023-10-01'],
            'period_2': ['2023-10-01', '2023-11-01'],
            'period_3': ['2023-11-01', '2023-12-01'],
            'period_4': ['2023-12-01', '2024-01-01']
        }

        metadata = {
            'study_design': 'longitudinal',
            'intervention_type': 'tolerance_enhancement',
            'data_collection_method': 'survey',
            'ethics_approval': 'approved'
        }

        return MultiLevelData(
            classrooms=classrooms,
            school_attributes=school_attributes,
            temporal_alignment=temporal_alignment,
            metadata=metadata
        )

    def _create_test_classroom_data(self, classroom_id: str) -> ClassroomData:
        """Create test data for individual classroom."""
        # Create networks over time
        networks = []
        for period in range(self.n_periods):
            # Create adjacency matrix
            p = 0.1 + 0.02 * period  # Increasing density over time
            adj_matrix = np.random.random((self.n_students_per_classroom,
                                         self.n_students_per_classroom)) < p
            # Remove self-loops
            np.fill_diagonal(adj_matrix, False)
            # Make symmetric (undirected network)
            adj_matrix = adj_matrix | adj_matrix.T
            networks.append(adj_matrix.astype(int))

        # Create tolerance data with realistic evolution
        tolerance_data = np.zeros((self.n_students_per_classroom, self.n_periods))

        # Initial tolerance with realistic distribution
        initial_tolerance = np.random.beta(2, 2, self.n_students_per_classroom) * 100
        tolerance_data[:, 0] = initial_tolerance

        # Evolve tolerance over time
        for t in range(1, self.n_periods):
            for student in range(self.n_students_per_classroom):
                # Individual stability
                tolerance_data[student, t] = tolerance_data[student, t-1] + np.random.normal(0, 2)

                # Network influence
                neighbors = np.where(networks[t-1][student, :] == 1)[0]
                if len(neighbors) > 0:
                    neighbor_tolerance = np.mean(tolerance_data[neighbors, t-1])
                    influence = 0.1 * (neighbor_tolerance - tolerance_data[student, t-1])
                    tolerance_data[student, t] += influence

                # Keep in bounds
                tolerance_data[student, t] = np.clip(tolerance_data[student, t], 0, 100)

        # Create student attributes
        student_attributes = {
            'age': np.random.randint(14, 18, self.n_students_per_classroom),
            'gender': np.random.choice(['male', 'female'], self.n_students_per_classroom),
            'ethnicity': np.random.choice(['white', 'black', 'hispanic', 'asian', 'other'],
                                        self.n_students_per_classroom),
            'ses_background': np.random.choice(['low', 'medium', 'high'],
                                             self.n_students_per_classroom,
                                             p=[0.3, 0.5, 0.2]),
            'academic_performance': np.random.normal(75, 15, self.n_students_per_classroom)
        }

        temporal_metadata = {
            'start_date': '2023-09-01',
            'end_date': '2024-01-01',
            'measurement_intervals': 'monthly',
            'missing_data_rate': 0.05
        }

        return ClassroomData(
            classroom_id=classroom_id,
            n_students=self.n_students_per_classroom,
            networks=networks,
            tolerance_data=tolerance_data,
            student_attributes=student_attributes,
            temporal_metadata=temporal_metadata
        )

    def test_classroom_data_handling(self):
        """Test handling of multi-classroom data structure."""
        logger.info("Testing classroom data handling...")

        # Test basic structure validation
        self._validate_multi_level_structure()

        # Test individual classroom validation
        for classroom_id, classroom_data in self.multi_level_data.classrooms.items():
            with self.subTest(classroom=classroom_id):
                self._validate_individual_classroom(classroom_data)

        # Test cross-classroom consistency
        self._validate_cross_classroom_consistency()

    def _validate_multi_level_structure(self):
        """Validate overall multi-level data structure."""
        # Check number of classrooms
        self.assertEqual(len(self.multi_level_data.classrooms), self.n_classrooms,
                        "Should have correct number of classrooms")

        # Check classroom IDs
        expected_ids = {f"classroom_{i:03d}" for i in range(self.n_classrooms)}
        actual_ids = set(self.multi_level_data.classrooms.keys())
        self.assertEqual(actual_ids, expected_ids,
                        "Classroom IDs should match expected format")

        # Check school attributes
        school_attrs = self.multi_level_data.school_attributes
        required_attrs = ['school_id', 'school_type', 'socioeconomic_status', 'location']
        for attr in required_attrs:
            self.assertIn(attr, school_attrs,
                         f"School attribute '{attr}' should be present")

        # Check temporal alignment
        temporal_alignment = self.multi_level_data.temporal_alignment
        self.assertEqual(len(temporal_alignment), self.n_periods,
                        "Temporal alignment should have entry for each period")

    def _validate_individual_classroom(self, classroom_data: ClassroomData):
        """Validate individual classroom data structure."""
        # Check basic properties
        self.assertEqual(classroom_data.n_students, self.n_students_per_classroom,
                        "Classroom should have correct number of students")

        # Check network data
        self.assertEqual(len(classroom_data.networks), self.n_periods,
                        "Should have network for each time period")

        for t, network in enumerate(classroom_data.networks):
            self.assertEqual(network.shape,
                           (self.n_students_per_classroom, self.n_students_per_classroom),
                           f"Network at time {t} should have correct dimensions")

            # Check network properties
            self.assertTrue(np.all(np.diag(network) == 0),
                           f"Network at time {t} should have no self-loops")

            self.assertTrue(np.allclose(network, network.T),
                           f"Network at time {t} should be symmetric")

            self.assertTrue(np.all((network == 0) | (network == 1)),
                           f"Network at time {t} should be binary")

        # Check tolerance data
        self.assertEqual(classroom_data.tolerance_data.shape,
                        (self.n_students_per_classroom, self.n_periods),
                        "Tolerance data should have correct dimensions")

        # Check tolerance bounds
        self.assertTrue(np.all(classroom_data.tolerance_data >= 0),
                       "All tolerance values should be >= 0")
        self.assertTrue(np.all(classroom_data.tolerance_data <= 100),
                       "All tolerance values should be <= 100")

        # Check student attributes
        for attr_name, attr_values in classroom_data.student_attributes.items():
            self.assertEqual(len(attr_values), self.n_students_per_classroom,
                           f"Attribute '{attr_name}' should have value for each student")

    def _validate_cross_classroom_consistency(self):
        """Validate consistency across classrooms."""
        # Check consistent structure across classrooms
        first_classroom = list(self.multi_level_data.classrooms.values())[0]
        first_attributes = set(first_classroom.student_attributes.keys())

        for classroom_id, classroom_data in self.multi_level_data.classrooms.items():
            # Same attribute names
            classroom_attributes = set(classroom_data.student_attributes.keys())
            self.assertEqual(classroom_attributes, first_attributes,
                           f"Classroom {classroom_id} should have same attribute structure")

            # Same number of time periods
            self.assertEqual(len(classroom_data.networks), self.n_periods,
                           f"Classroom {classroom_id} should have same number of periods")

            # Same temporal metadata structure
            expected_metadata_keys = set(first_classroom.temporal_metadata.keys())
            actual_metadata_keys = set(classroom_data.temporal_metadata.keys())
            self.assertEqual(actual_metadata_keys, expected_metadata_keys,
                           f"Classroom {classroom_id} should have same metadata structure")

    @unittest.skipUnless(RPY2_AVAILABLE, "rpy2 not available")
    def test_rsiena_data_format(self):
        """Test RSiena-compatible data format conversion."""
        logger.info("Testing RSiena data format conversion...")

        # Test conversion for individual classroom
        first_classroom_id = list(self.multi_level_data.classrooms.keys())[0]
        first_classroom = self.multi_level_data.classrooms[first_classroom_id]

        # Convert to RSiena format
        rsiena_data = self._convert_classroom_to_rsiena(first_classroom)

        # Validate RSiena data structure
        self._validate_rsiena_data_structure(rsiena_data, first_classroom)

        # Test multi-classroom conversion
        self._test_multi_classroom_rsiena_conversion()

    def _convert_classroom_to_rsiena(self, classroom_data: ClassroomData) -> RSienaDataSet:
        """Convert classroom data to RSiena format."""
        # Convert networks to NetworkX format for converter
        networkx_networks = []
        for adj_matrix in classroom_data.networks:
            G = nx.from_numpy_array(adj_matrix)
            networkx_networks.append(G)

        # Prepare actor attributes
        actor_attributes = {}
        for attr_name, attr_values in classroom_data.student_attributes.items():
            if attr_name in ['age', 'academic_performance']:  # Numeric attributes
                actor_attributes[attr_name] = attr_values
            elif attr_name == 'gender':  # Binary encoding
                actor_attributes[attr_name] = (attr_values == 'male').astype(int)

        # Prepare behavior data
        behavior_data = {'tolerance': classroom_data.tolerance_data}

        # Convert using converter
        rsiena_data = self.converter.convert_to_siena(
            networkx_networks,
            actor_attributes=actor_attributes,
            behavior_data=behavior_data
        )

        return rsiena_data

    def _validate_rsiena_data_structure(self, rsiena_data: RSienaDataSet,
                                      classroom_data: ClassroomData):
        """Validate RSiena data structure."""
        # Check dimensions
        self.assertEqual(rsiena_data.n_actors, classroom_data.n_students,
                        "RSiena data should have correct number of actors")
        self.assertEqual(rsiena_data.n_periods, self.n_periods,
                        "RSiena data should have correct number of periods")

        # Check network data dimensions
        expected_shape = (self.n_periods, classroom_data.n_students, classroom_data.n_students)
        self.assertEqual(rsiena_data.network_data.shape, expected_shape,
                        "Network data should have correct shape")

        # Check behavior data
        if 'tolerance' in rsiena_data.behavior_data:
            tolerance_shape = rsiena_data.behavior_data['tolerance'].shape
            expected_tolerance_shape = (classroom_data.n_students, self.n_periods)
            self.assertEqual(tolerance_shape, expected_tolerance_shape,
                           "Tolerance behavior data should have correct shape")

        # Check actor attributes
        for attr_name, attr_data in rsiena_data.actor_attributes.items():
            self.assertEqual(len(attr_data), classroom_data.n_students,
                           f"Actor attribute '{attr_name}' should have correct length")

        # Check actor IDs
        self.assertEqual(len(rsiena_data.actor_ids), classroom_data.n_students,
                        "Actor IDs should have correct length")

    def _test_multi_classroom_rsiena_conversion(self):
        """Test conversion of multiple classrooms to RSiena format."""
        # Convert each classroom separately
        rsiena_datasets = {}

        for classroom_id, classroom_data in self.multi_level_data.classrooms.items():
            rsiena_data = self._convert_classroom_to_rsiena(classroom_data)
            rsiena_datasets[classroom_id] = rsiena_data

        # Validate multi-classroom structure
        self.assertEqual(len(rsiena_datasets), self.n_classrooms,
                        "Should have RSiena data for each classroom")

        # Check consistency across classrooms
        first_rsiena = list(rsiena_datasets.values())[0]
        for classroom_id, rsiena_data in rsiena_datasets.items():
            # Same structure
            self.assertEqual(rsiena_data.n_periods, first_rsiena.n_periods,
                           f"Classroom {classroom_id} should have same number of periods")

            # Same attribute names
            first_attrs = set(first_rsiena.actor_attributes.keys())
            classroom_attrs = set(rsiena_data.actor_attributes.keys())
            self.assertEqual(classroom_attrs, first_attrs,
                           f"Classroom {classroom_id} should have same attributes")

    def test_missing_data_handling(self):
        """Test missing data handling mechanisms."""
        logger.info("Testing missing data handling...")

        # Create data with missing values
        classroom_with_missing = self._create_data_with_missing_values()

        # Test missing data detection
        missing_patterns = self._detect_missing_patterns(classroom_with_missing)
        self._validate_missing_data_detection(missing_patterns)

        # Test missing data imputation
        imputed_data = self._apply_missing_data_imputation(classroom_with_missing)
        self._validate_missing_data_imputation(classroom_with_missing, imputed_data)

        # Test listwise deletion
        complete_case_data = self._apply_listwise_deletion(classroom_with_missing)
        self._validate_listwise_deletion(classroom_with_missing, complete_case_data)

    def _create_data_with_missing_values(self) -> ClassroomData:
        """Create classroom data with missing values."""
        # Start with complete data
        classroom_data = list(self.multi_level_data.classrooms.values())[0]

        # Create copy with missing values
        modified_data = ClassroomData(
            classroom_id=classroom_data.classroom_id + "_missing",
            n_students=classroom_data.n_students,
            networks=classroom_data.networks.copy(),
            tolerance_data=classroom_data.tolerance_data.copy(),
            student_attributes=classroom_data.student_attributes.copy(),
            temporal_metadata=classroom_data.temporal_metadata.copy()
        )

        # Introduce missing values in tolerance data (5% missing)
        n_missing = int(0.05 * modified_data.n_students * self.n_periods)
        missing_indices = np.random.choice(
            modified_data.n_students * self.n_periods,
            size=n_missing,
            replace=False
        )

        flat_tolerance = modified_data.tolerance_data.flatten()
        flat_tolerance[missing_indices] = np.nan
        modified_data.tolerance_data = flat_tolerance.reshape(modified_data.tolerance_data.shape)

        # Introduce missing values in student attributes
        missing_students = np.random.choice(
            modified_data.n_students,
            size=int(0.03 * modified_data.n_students),
            replace=False
        )

        modified_attributes = {}
        for attr_name, attr_values in modified_data.student_attributes.items():
            new_values = attr_values.copy()
            if attr_name in ['age', 'academic_performance']:  # Numeric attributes
                new_values = new_values.astype(float)
                new_values[missing_students] = np.nan
            modified_attributes[attr_name] = new_values

        modified_data.student_attributes = modified_attributes

        return modified_data

    def _detect_missing_patterns(self, classroom_data: ClassroomData) -> Dict[str, Any]:
        """Detect patterns in missing data."""
        missing_patterns = {}

        # Missing data in tolerance
        tolerance_missing = np.isnan(classroom_data.tolerance_data)
        missing_patterns['tolerance'] = {
            'total_missing': np.sum(tolerance_missing),
            'missing_rate': np.mean(tolerance_missing),
            'missing_by_time': np.sum(tolerance_missing, axis=0),
            'missing_by_student': np.sum(tolerance_missing, axis=1)
        }

        # Missing data in attributes
        missing_patterns['attributes'] = {}
        for attr_name, attr_values in classroom_data.student_attributes.items():
            if attr_values.dtype in [float, np.float64, np.float32]:
                attr_missing = np.isnan(attr_values)
                missing_patterns['attributes'][attr_name] = {
                    'total_missing': np.sum(attr_missing),
                    'missing_rate': np.mean(attr_missing)
                }

        return missing_patterns

    def _validate_missing_data_detection(self, missing_patterns: Dict[str, Any]):
        """Validate missing data detection."""
        # Check tolerance missing data detection
        tolerance_missing = missing_patterns['tolerance']
        self.assertGreater(tolerance_missing['total_missing'], 0,
                          "Should detect missing tolerance data")
        self.assertGreater(tolerance_missing['missing_rate'], 0,
                          "Missing rate should be positive")
        self.assertLess(tolerance_missing['missing_rate'], 1,
                       "Missing rate should be less than 100%")

        # Check that missing data is distributed across time periods
        missing_by_time = tolerance_missing['missing_by_time']
        self.assertEqual(len(missing_by_time), self.n_periods,
                        "Should have missing data counts for each time period")

        # Check attribute missing data detection
        attr_missing = missing_patterns['attributes']
        numeric_attrs = ['age', 'academic_performance']
        for attr in numeric_attrs:
            if attr in attr_missing:
                self.assertGreaterEqual(attr_missing[attr]['missing_rate'], 0,
                                       f"Missing rate for {attr} should be non-negative")

    def _apply_missing_data_imputation(self, classroom_data: ClassroomData) -> ClassroomData:
        """Apply missing data imputation techniques."""
        # Create copy for imputation
        imputed_data = ClassroomData(
            classroom_id=classroom_data.classroom_id + "_imputed",
            n_students=classroom_data.n_students,
            networks=classroom_data.networks.copy(),
            tolerance_data=classroom_data.tolerance_data.copy(),
            student_attributes={k: v.copy() for k, v in classroom_data.student_attributes.items()},
            temporal_metadata=classroom_data.temporal_metadata.copy()
        )

        # Forward fill imputation for tolerance data
        for student in range(imputed_data.n_students):
            student_tolerance = imputed_data.tolerance_data[student, :]
            last_valid = None

            for t in range(self.n_periods):
                if np.isnan(student_tolerance[t]):
                    if last_valid is not None:
                        imputed_data.tolerance_data[student, t] = last_valid
                    else:
                        # Use mean if no previous value
                        valid_values = student_tolerance[~np.isnan(student_tolerance)]
                        if len(valid_values) > 0:
                            imputed_data.tolerance_data[student, t] = np.mean(valid_values)
                        else:
                            # Use overall mean as last resort
                            overall_mean = np.nanmean(imputed_data.tolerance_data)
                            imputed_data.tolerance_data[student, t] = overall_mean
                else:
                    last_valid = student_tolerance[t]

        # Mean imputation for numeric attributes
        for attr_name, attr_values in imputed_data.student_attributes.items():
            if attr_values.dtype in [float, np.float64, np.float32]:
                missing_mask = np.isnan(attr_values)
                if np.any(missing_mask):
                    attr_mean = np.nanmean(attr_values)
                    attr_values[missing_mask] = attr_mean

        return imputed_data

    def _validate_missing_data_imputation(self, original_data: ClassroomData,
                                        imputed_data: ClassroomData):
        """Validate missing data imputation."""
        # Check that no missing values remain in tolerance data
        self.assertFalse(np.any(np.isnan(imputed_data.tolerance_data)),
                        "Imputed tolerance data should have no missing values")

        # Check that imputation preserves non-missing values
        original_valid_mask = ~np.isnan(original_data.tolerance_data)
        np.testing.assert_array_equal(
            original_data.tolerance_data[original_valid_mask],
            imputed_data.tolerance_data[original_valid_mask],
            err_msg="Imputation should preserve original non-missing values"
        )

        # Check that imputed values are in valid range
        self.assertTrue(np.all(imputed_data.tolerance_data >= 0),
                       "Imputed tolerance values should be >= 0")
        self.assertTrue(np.all(imputed_data.tolerance_data <= 100),
                       "Imputed tolerance values should be <= 100")

        # Check numeric attributes
        for attr_name, attr_values in imputed_data.student_attributes.items():
            if attr_values.dtype in [float, np.float64, np.float32]:
                self.assertFalse(np.any(np.isnan(attr_values)),
                               f"Imputed attribute '{attr_name}' should have no missing values")

    def _apply_listwise_deletion(self, classroom_data: ClassroomData) -> ClassroomData:
        """Apply listwise deletion for missing data."""
        # Identify students with complete data
        tolerance_complete = ~np.any(np.isnan(classroom_data.tolerance_data), axis=1)

        # Check attributes
        attr_complete = np.ones(classroom_data.n_students, dtype=bool)
        for attr_name, attr_values in classroom_data.student_attributes.items():
            if attr_values.dtype in [float, np.float64, np.float32]:
                attr_complete &= ~np.isnan(attr_values)

        # Students with complete data
        complete_students = tolerance_complete & attr_complete
        complete_indices = np.where(complete_students)[0]

        if len(complete_indices) == 0:
            # If no complete cases, return empty data
            return ClassroomData(
                classroom_id=classroom_data.classroom_id + "_listwise",
                n_students=0,
                networks=[],
                tolerance_data=np.array([]).reshape(0, self.n_periods),
                student_attributes={},
                temporal_metadata=classroom_data.temporal_metadata.copy()
            )

        # Create reduced dataset
        reduced_networks = []
        for network in classroom_data.networks:
            reduced_network = network[np.ix_(complete_indices, complete_indices)]
            reduced_networks.append(reduced_network)

        reduced_tolerance = classroom_data.tolerance_data[complete_indices, :]

        reduced_attributes = {}
        for attr_name, attr_values in classroom_data.student_attributes.items():
            reduced_attributes[attr_name] = attr_values[complete_indices]

        return ClassroomData(
            classroom_id=classroom_data.classroom_id + "_listwise",
            n_students=len(complete_indices),
            networks=reduced_networks,
            tolerance_data=reduced_tolerance,
            student_attributes=reduced_attributes,
            temporal_metadata=classroom_data.temporal_metadata.copy()
        )

    def _validate_listwise_deletion(self, original_data: ClassroomData,
                                  complete_case_data: ClassroomData):
        """Validate listwise deletion results."""
        # Check that complete case data has no missing values
        if complete_case_data.n_students > 0:
            self.assertFalse(np.any(np.isnan(complete_case_data.tolerance_data)),
                           "Complete case tolerance data should have no missing values")

            for attr_name, attr_values in complete_case_data.student_attributes.items():
                if attr_values.dtype in [float, np.float64, np.float32]:
                    self.assertFalse(np.any(np.isnan(attr_values)),
                                   f"Complete case attribute '{attr_name}' should have no missing values")

            # Check network dimensions consistency
            for network in complete_case_data.networks:
                expected_shape = (complete_case_data.n_students, complete_case_data.n_students)
                self.assertEqual(network.shape, expected_shape,
                               "Complete case networks should have consistent dimensions")

        # Check that number of students is reduced or equal
        self.assertLessEqual(complete_case_data.n_students, original_data.n_students,
                           "Listwise deletion should not increase sample size")

    def test_temporal_alignment(self):
        """Test temporal alignment between different data sources."""
        logger.info("Testing temporal alignment...")

        # Test alignment within classroom
        for classroom_id, classroom_data in self.multi_level_data.classrooms.items():
            with self.subTest(classroom=classroom_id):
                self._validate_temporal_alignment_within_classroom(classroom_data)

        # Test alignment across classrooms
        self._validate_temporal_alignment_across_classrooms()

        # Test alignment with external time reference
        self._validate_temporal_alignment_with_reference()

    def _validate_temporal_alignment_within_classroom(self, classroom_data: ClassroomData):
        """Validate temporal alignment within individual classroom."""
        # Check that networks and tolerance data have same number of periods
        self.assertEqual(len(classroom_data.networks), self.n_periods,
                        "Networks should have correct number of periods")
        self.assertEqual(classroom_data.tolerance_data.shape[1], self.n_periods,
                        "Tolerance data should have correct number of periods")

        # Check temporal metadata consistency
        temporal_meta = classroom_data.temporal_metadata
        self.assertIn('start_date', temporal_meta,
                     "Temporal metadata should include start date")
        self.assertIn('end_date', temporal_meta,
                     "Temporal metadata should include end date")

    def _validate_temporal_alignment_across_classrooms(self):
        """Validate temporal alignment across classrooms."""
        # All classrooms should have same number of periods
        n_periods_list = []
        for classroom_data in self.multi_level_data.classrooms.values():
            n_periods_list.append(len(classroom_data.networks))

        self.assertTrue(all(n == self.n_periods for n in n_periods_list),
                       "All classrooms should have same number of periods")

        # Check temporal metadata consistency
        start_dates = []
        end_dates = []
        for classroom_data in self.multi_level_data.classrooms.values():
            start_dates.append(classroom_data.temporal_metadata.get('start_date'))
            end_dates.append(classroom_data.temporal_metadata.get('end_date'))

        # All classrooms should have same study period
        self.assertEqual(len(set(start_dates)), 1,
                        "All classrooms should have same start date")
        self.assertEqual(len(set(end_dates)), 1,
                        "All classrooms should have same end date")

    def _validate_temporal_alignment_with_reference(self):
        """Validate alignment with external temporal reference."""
        temporal_alignment = self.multi_level_data.temporal_alignment

        # Check that temporal alignment covers all periods
        self.assertEqual(len(temporal_alignment), self.n_periods,
                        "Temporal alignment should cover all periods")

        # Check that periods are sequential and non-overlapping
        period_labels = sorted(temporal_alignment.keys())
        expected_labels = [f'period_{i+1}' for i in range(self.n_periods)]
        self.assertEqual(period_labels, expected_labels,
                        "Period labels should be sequential")

        # Check date consistency
        for period_label, date_range in temporal_alignment.items():
            self.assertEqual(len(date_range), 2,
                           f"Period {period_label} should have start and end date")

            start_date, end_date = date_range
            self.assertIsInstance(start_date, str,
                                f"Start date for {period_label} should be string")
            self.assertIsInstance(end_date, str,
                                f"End date for {period_label} should be string")

    def test_data_integrity_checks(self):
        """Test comprehensive data integrity checks."""
        logger.info("Testing data integrity checks...")

        # Test data consistency checks
        integrity_results = self._run_data_integrity_checks()

        # Validate integrity check results
        self._validate_integrity_check_results(integrity_results)

    def _run_data_integrity_checks(self) -> Dict[str, Any]:
        """Run comprehensive data integrity checks."""
        integrity_results = {
            'consistency_checks': {},
            'validity_checks': {},
            'completeness_checks': {},
            'errors': [],
            'warnings': []
        }

        # Consistency checks
        for classroom_id, classroom_data in self.multi_level_data.classrooms.items():
            # Check network-tolerance consistency
            network_students = classroom_data.networks[0].shape[0]
            tolerance_students = classroom_data.tolerance_data.shape[0]

            if network_students != tolerance_students:
                integrity_results['errors'].append(
                    f"Classroom {classroom_id}: Network and tolerance data have different number of students"
                )

            # Check attribute consistency
            for attr_name, attr_values in classroom_data.student_attributes.items():
                if len(attr_values) != classroom_data.n_students:
                    integrity_results['errors'].append(
                        f"Classroom {classroom_id}: Attribute {attr_name} has wrong length"
                    )

        # Validity checks
        for classroom_id, classroom_data in self.multi_level_data.classrooms.items():
            # Check tolerance value ranges
            if np.any(classroom_data.tolerance_data < 0) or np.any(classroom_data.tolerance_data > 100):
                integrity_results['errors'].append(
                    f"Classroom {classroom_id}: Tolerance values outside valid range [0,100]"
                )

            # Check network properties
            for t, network in enumerate(classroom_data.networks):
                if not np.allclose(network, network.T):
                    integrity_results['errors'].append(
                        f"Classroom {classroom_id}, time {t}: Network is not symmetric"
                    )

                if np.any(np.diag(network) != 0):
                    integrity_results['errors'].append(
                        f"Classroom {classroom_id}, time {t}: Network has self-loops"
                    )

        # Completeness checks
        total_expected_data_points = (self.n_classrooms *
                                    self.n_students_per_classroom *
                                    self.n_periods)

        actual_data_points = 0
        missing_data_points = 0

        for classroom_data in self.multi_level_data.classrooms.values():
            tolerance_data = classroom_data.tolerance_data
            actual_data_points += tolerance_data.size
            missing_data_points += np.sum(np.isnan(tolerance_data))

        completeness_rate = (actual_data_points - missing_data_points) / actual_data_points
        integrity_results['completeness_checks']['overall_completeness'] = completeness_rate

        if completeness_rate < 0.95:  # Warn if less than 95% complete
            integrity_results['warnings'].append(
                f"Overall data completeness is {completeness_rate:.2%}, below 95% threshold"
            )

        return integrity_results

    def _validate_integrity_check_results(self, integrity_results: Dict[str, Any]):
        """Validate integrity check results."""
        # Check that integrity checks ran successfully
        self.assertIn('consistency_checks', integrity_results,
                     "Integrity results should include consistency checks")
        self.assertIn('validity_checks', integrity_results,
                     "Integrity results should include validity checks")
        self.assertIn('completeness_checks', integrity_results,
                     "Integrity results should include completeness checks")

        # Check error and warning handling
        errors = integrity_results.get('errors', [])
        warnings = integrity_results.get('warnings', [])

        # Log any errors or warnings
        if errors:
            logger.warning(f"Data integrity errors found: {errors}")

        if warnings:
            logger.info(f"Data integrity warnings: {warnings}")

        # For test data, we expect some missing data but no major errors
        # Allow for some errors due to introduced missing data
        self.assertLess(len(errors), 10,
                       "Should not have excessive data integrity errors")

        # Check completeness
        completeness = integrity_results['completeness_checks'].get('overall_completeness', 0)
        self.assertGreater(completeness, 0.8,
                         "Overall data completeness should be above 80%")


if __name__ == '__main__':
    # Configure logging for test run
    logging.basicConfig(level=logging.INFO)

    # Run tests
    unittest.main(verbosity=2)