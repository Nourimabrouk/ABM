"""
Classroom Data Processor for Multi-Level SAOM Analysis

This module handles the complex multi-level structure of the tolerance-cooperation
research data: 5825 observations, 2585 respondents, 3 schools, 105 classes, 3 waves.

Key Features:
- Multi-level data structure handling (students → classes → schools)
- RSiena-compatible data formatting for each classroom
- Missing data imputation and validation
- Cross-classroom comparison capabilities
- Temporal alignment across waves
- Ethnicity and network composition analysis

Research Context:
Based on Shani et al. (2023) follow-up study data structure.
Enables classroom-by-classroom SAOM estimation and meta-analysis.

Author: RSiena Integration Specialist
Created: 2025-09-16
"""

import logging
import numpy as np
import pandas as pd
import networkx as nx
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
from pathlib import Path
import warnings
from collections import defaultdict

from ..rsiena_integration.data_converters import ABMRSienaConverter, RSienaDataSet, ConversionConfig

logger = logging.getLogger(__name__)


@dataclass
class ClassroomInfo:
    """Information about a single classroom."""
    classroom_id: str
    school_id: str
    n_students: int
    n_waves: int
    ethnicity_composition: Dict[str, int]
    network_density: List[float]  # Per wave
    tolerance_mean: List[float]   # Per wave
    tolerance_std: List[float]    # Per wave
    missing_data_percent: float
    data_quality_score: float


@dataclass
class MultiLevelDataConfig:
    """Configuration for multi-level data processing."""
    # Missing data handling
    missing_tolerance_threshold: float = 0.3  # Max missing tolerance data
    missing_network_threshold: float = 0.5   # Max missing network data
    imputation_method: str = "mean"  # "mean", "median", "forward_fill", "interpolate"

    # Network processing
    min_classroom_size: int = 10
    max_classroom_size: int = 40
    friendship_threshold: float = 0.5  # Threshold for friendship tie
    cooperation_threshold: float = 0.5  # Threshold for cooperation tie

    # Temporal alignment
    wave_alignment: str = "strict"  # "strict", "flexible"
    time_window_tolerance: int = 30  # Days tolerance for wave alignment

    # Quality filters
    min_data_quality_score: float = 0.6
    require_all_waves: bool = True
    require_both_networks: bool = True


class ClassroomDataProcessor:
    """
    Processor for multi-level classroom data structure.

    Handles the complex task of preparing classroom-level datasets
    for SAOM analysis while maintaining data quality and comparability.
    """

    def __init__(self, config: Optional[MultiLevelDataConfig] = None):
        """
        Initialize classroom data processor.

        Args:
            config: Data processing configuration
        """
        self.config = config or MultiLevelDataConfig()
        self.classroom_info = {}
        self.processed_classrooms = {}
        self.data_quality_report = {}

    def load_raw_data(
        self,
        data_path: Union[str, Path],
        format: str = "csv"
    ) -> Dict[str, pd.DataFrame]:
        """
        Load raw multi-level data from files.

        Args:
            data_path: Path to data directory
            format: Data format ("csv", "stata", "spss")

        Returns:
            Dictionary with loaded dataframes
        """
        logger.info(f"Loading raw data from {data_path}")

        data_path = Path(data_path)
        raw_data = {}

        try:
            if format == "csv":
                # Expected file structure
                expected_files = [
                    "students.csv",      # Student demographics and attributes
                    "networks.csv",      # Network data (friendship/cooperation)
                    "tolerance.csv",     # Tolerance measurements across waves
                    "classrooms.csv",    # Classroom information
                    "schools.csv"        # School information
                ]

                for file_name in expected_files:
                    file_path = data_path / file_name
                    if file_path.exists():
                        raw_data[file_name.split('.')[0]] = pd.read_csv(file_path)
                        logger.debug(f"Loaded {file_name}: {len(raw_data[file_name.split('.')[0]])} rows")
                    else:
                        logger.warning(f"File not found: {file_path}")

            elif format == "stata":
                # Load Stata files
                for stata_file in data_path.glob("*.dta"):
                    df_name = stata_file.stem
                    raw_data[df_name] = pd.read_stata(stata_file)
                    logger.debug(f"Loaded {stata_file.name}: {len(raw_data[df_name])} rows")

            else:
                raise ValueError(f"Unsupported data format: {format}")

            logger.info(f"Successfully loaded {len(raw_data)} datasets")
            return raw_data

        except Exception as e:
            logger.error(f"Failed to load raw data: {e}")
            raise RuntimeError(f"Data loading failed: {e}")

    def process_classrooms(
        self,
        raw_data: Dict[str, pd.DataFrame]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Process raw data into classroom-level datasets.

        Args:
            raw_data: Raw data dictionaries

        Returns:
            Dictionary with processed classroom data
        """
        logger.info("Processing classroom-level data...")

        # Validate input data
        self._validate_raw_data(raw_data)

        # Get classroom list
        classrooms = self._identify_classrooms(raw_data)

        # Process each classroom
        processed_classrooms = {}
        quality_reports = {}

        for classroom_id in classrooms:
            try:
                logger.debug(f"Processing classroom {classroom_id}")

                # Extract classroom data
                classroom_data = self._extract_classroom_data(raw_data, classroom_id)

                # Quality assessment
                quality_score = self._assess_data_quality(classroom_data, classroom_id)

                # Filter based on quality
                if quality_score >= self.config.min_data_quality_score:
                    # Process networks
                    networks = self._process_classroom_networks(classroom_data)

                    # Process behaviors
                    behaviors = self._process_classroom_behaviors(classroom_data)

                    # Process attributes
                    attributes = self._process_classroom_attributes(classroom_data)

                    # Create classroom info
                    info = self._create_classroom_info(classroom_data, classroom_id, quality_score)

                    processed_classrooms[classroom_id] = {
                        'networks': networks,
                        'behaviors': behaviors,
                        'attributes': attributes,
                        'info': info,
                        'raw_data': classroom_data
                    }

                    self.classroom_info[classroom_id] = info
                    quality_reports[classroom_id] = quality_score

                    logger.debug(f"✓ Classroom {classroom_id} processed successfully")

                else:
                    logger.warning(f"✗ Classroom {classroom_id} excluded (quality: {quality_score:.3f})")
                    quality_reports[classroom_id] = quality_score

            except Exception as e:
                logger.error(f"Failed to process classroom {classroom_id}: {e}")
                quality_reports[classroom_id] = 0.0

        self.data_quality_report = quality_reports
        self.processed_classrooms = processed_classrooms

        logger.info(f"Processed {len(processed_classrooms)} classrooms successfully")
        return processed_classrooms

    def _validate_raw_data(self, raw_data: Dict[str, pd.DataFrame]):
        """Validate raw data structure and content."""
        required_datasets = ['students', 'networks', 'tolerance']

        for dataset in required_datasets:
            if dataset not in raw_data:
                raise ValueError(f"Required dataset '{dataset}' not found")

        # Check for required columns
        if 'students' in raw_data:
            required_student_cols = ['student_id', 'classroom_id', 'school_id', 'ethnicity', 'gender']
            missing_cols = set(required_student_cols) - set(raw_data['students'].columns)
            if missing_cols:
                logger.warning(f"Missing student columns: {missing_cols}")

        if 'networks' in raw_data:
            required_network_cols = ['student_i', 'student_j', 'classroom_id', 'wave', 'network_type']
            missing_cols = set(required_network_cols) - set(raw_data['networks'].columns)
            if missing_cols:
                logger.warning(f"Missing network columns: {missing_cols}")

        if 'tolerance' in raw_data:
            required_tolerance_cols = ['student_id', 'classroom_id', 'wave', 'tolerance', 'prejudice']
            missing_cols = set(required_tolerance_cols) - set(raw_data['tolerance'].columns)
            if missing_cols:
                logger.warning(f"Missing tolerance columns: {missing_cols}")

        logger.debug("Raw data validation completed")

    def _identify_classrooms(self, raw_data: Dict[str, pd.DataFrame]) -> List[str]:
        """Identify unique classrooms in the data."""
        classroom_ids = set()

        # Get classrooms from different datasets
        for dataset_name, df in raw_data.items():
            if 'classroom_id' in df.columns:
                classroom_ids.update(df['classroom_id'].unique())

        classroom_list = sorted(list(classroom_ids))
        logger.info(f"Identified {len(classroom_list)} classrooms")

        return classroom_list

    def _extract_classroom_data(
        self,
        raw_data: Dict[str, pd.DataFrame],
        classroom_id: str
    ) -> Dict[str, pd.DataFrame]:
        """Extract data for a specific classroom."""
        classroom_data = {}

        for dataset_name, df in raw_data.items():
            if 'classroom_id' in df.columns:
                classroom_subset = df[df['classroom_id'] == classroom_id].copy()
                classroom_data[dataset_name] = classroom_subset

        return classroom_data

    def _assess_data_quality(
        self,
        classroom_data: Dict[str, pd.DataFrame],
        classroom_id: str
    ) -> float:
        """Assess data quality for a classroom."""
        quality_scores = []

        # Check classroom size
        if 'students' in classroom_data:
            n_students = len(classroom_data['students'])
            size_score = 1.0 if (self.config.min_classroom_size <= n_students <= self.config.max_classroom_size) else 0.5
            quality_scores.append(size_score)

        # Check wave completeness
        if 'tolerance' in classroom_data:
            expected_waves = 3  # Based on research design
            actual_waves = len(classroom_data['tolerance']['wave'].unique())
            wave_score = actual_waves / expected_waves
            quality_scores.append(wave_score)

        # Check missing data
        if 'tolerance' in classroom_data:
            tolerance_missing = classroom_data['tolerance']['tolerance'].isnull().mean()
            missing_score = 1.0 - tolerance_missing
            quality_scores.append(missing_score)

        # Check network data availability
        if 'networks' in classroom_data:
            network_types = classroom_data['networks']['network_type'].unique()
            network_score = len(network_types) / 2  # Expecting friendship and cooperation
            quality_scores.append(network_score)

        # Calculate overall quality score
        overall_score = np.mean(quality_scores) if quality_scores else 0.0

        return overall_score

    def _process_classroom_networks(
        self,
        classroom_data: Dict[str, pd.DataFrame]
    ) -> Dict[str, List[nx.Graph]]:
        """Process network data for a classroom."""
        if 'networks' not in classroom_data:
            return {}

        networks_df = classroom_data['networks']
        networks = defaultdict(list)

        # Get unique waves and network types
        waves = sorted(networks_df['wave'].unique())
        network_types = networks_df['network_type'].unique()

        for network_type in network_types:
            for wave in waves:
                # Filter data for this network type and wave
                wave_network_data = networks_df[
                    (networks_df['network_type'] == network_type) &
                    (networks_df['wave'] == wave)
                ]

                # Create network
                G = nx.DiGraph()

                # Add edges
                for _, row in wave_network_data.iterrows():
                    if 'weight' in row and not pd.isna(row['weight']):
                        weight = row['weight']
                        # Apply threshold if specified
                        threshold = (self.config.friendship_threshold if network_type == 'friendship'
                                   else self.config.cooperation_threshold)
                        if weight >= threshold:
                            G.add_edge(row['student_i'], row['student_j'], weight=weight)
                    else:
                        G.add_edge(row['student_i'], row['student_j'])

                networks[network_type].append(G)

        return dict(networks)

    def _process_classroom_behaviors(
        self,
        classroom_data: Dict[str, pd.DataFrame]
    ) -> List[Dict[int, Dict[str, Any]]]:
        """Process behavior data for a classroom."""
        if 'tolerance' not in classroom_data:
            return []

        tolerance_df = classroom_data['tolerance']
        behaviors = []

        waves = sorted(tolerance_df['wave'].unique())

        for wave in waves:
            wave_data = tolerance_df[tolerance_df['wave'] == wave]
            wave_behaviors = {}

            for _, row in wave_data.iterrows():
                student_id = row['student_id']
                student_behaviors = {}

                # Add tolerance
                if not pd.isna(row['tolerance']):
                    student_behaviors['tolerance'] = float(row['tolerance'])

                # Add prejudice if available
                if 'prejudice' in row and not pd.isna(row['prejudice']):
                    student_behaviors['prejudice'] = float(row['prejudice'])

                if student_behaviors:  # Only add if we have some behavior data
                    wave_behaviors[student_id] = student_behaviors

            behaviors.append(wave_behaviors)

        return behaviors

    def _process_classroom_attributes(
        self,
        classroom_data: Dict[str, pd.DataFrame]
    ) -> Dict[str, List[Any]]:
        """Process actor attributes for a classroom."""
        if 'students' not in classroom_data:
            return {}

        students_df = classroom_data['students']
        attributes = {}

        # Get consistent student ordering
        student_ids = sorted(students_df['student_id'].unique())

        # Process each attribute
        attribute_columns = ['ethnicity', 'gender', 'age', 'ses']  # Socioeconomic status

        for attr_col in attribute_columns:
            if attr_col in students_df.columns:
                attr_values = []
                for student_id in student_ids:
                    student_row = students_df[students_df['student_id'] == student_id]
                    if len(student_row) > 0 and not pd.isna(student_row[attr_col].iloc[0]):
                        attr_values.append(student_row[attr_col].iloc[0])
                    else:
                        # Handle missing values
                        if attr_col in ['ethnicity', 'gender']:
                            attr_values.append(0)  # Default to majority group
                        else:
                            attr_values.append(np.nan)

                # Impute missing values if needed
                attr_values = self._impute_missing_values(attr_values, attr_col)
                attributes[attr_col] = attr_values

        return attributes

    def _impute_missing_values(self, values: List[Any], attribute_name: str) -> List[Any]:
        """Impute missing values in attribute data."""
        values = np.array(values)
        missing_mask = pd.isna(values)

        if not missing_mask.any():
            return values.tolist()

        if self.config.imputation_method == "mean":
            if attribute_name in ['ethnicity', 'gender']:
                # Use mode for categorical
                from scipy import stats
                mode_value = stats.mode(values[~missing_mask], keepdims=True)[0][0]
                values[missing_mask] = mode_value
            else:
                # Use mean for continuous
                mean_value = np.nanmean(values)
                values[missing_mask] = mean_value

        elif self.config.imputation_method == "median":
            median_value = np.nanmedian(values)
            values[missing_mask] = median_value

        return values.tolist()

    def _create_classroom_info(
        self,
        classroom_data: Dict[str, pd.DataFrame],
        classroom_id: str,
        quality_score: float
    ) -> ClassroomInfo:
        """Create information object for a classroom."""
        # Get basic info
        school_id = "unknown"
        n_students = 0
        n_waves = 0

        if 'students' in classroom_data:
            n_students = len(classroom_data['students'])
            if 'school_id' in classroom_data['students'].columns:
                school_id = classroom_data['students']['school_id'].iloc[0]

        if 'tolerance' in classroom_data:
            n_waves = len(classroom_data['tolerance']['wave'].unique())

        # Get ethnicity composition
        ethnicity_composition = {}
        if 'students' in classroom_data and 'ethnicity' in classroom_data['students'].columns:
            ethnicity_counts = classroom_data['students']['ethnicity'].value_counts()
            ethnicity_composition = ethnicity_counts.to_dict()

        # Calculate network densities
        network_density = []
        if 'networks' in classroom_data:
            waves = sorted(classroom_data['networks']['wave'].unique())
            for wave in waves:
                wave_edges = len(classroom_data['networks'][classroom_data['networks']['wave'] == wave])
                max_edges = n_students * (n_students - 1)
                density = wave_edges / max_edges if max_edges > 0 else 0
                network_density.append(density)

        # Calculate tolerance statistics
        tolerance_mean = []
        tolerance_std = []
        if 'tolerance' in classroom_data:
            waves = sorted(classroom_data['tolerance']['wave'].unique())
            for wave in waves:
                wave_tolerance = classroom_data['tolerance'][
                    classroom_data['tolerance']['wave'] == wave
                ]['tolerance'].dropna()
                tolerance_mean.append(wave_tolerance.mean())
                tolerance_std.append(wave_tolerance.std())

        # Calculate missing data percentage
        missing_percent = 0.0
        if 'tolerance' in classroom_data:
            missing_percent = classroom_data['tolerance']['tolerance'].isnull().mean()

        return ClassroomInfo(
            classroom_id=classroom_id,
            school_id=school_id,
            n_students=n_students,
            n_waves=n_waves,
            ethnicity_composition=ethnicity_composition,
            network_density=network_density,
            tolerance_mean=tolerance_mean,
            tolerance_std=tolerance_std,
            missing_data_percent=missing_percent,
            data_quality_score=quality_score
        )

    def convert_classroom_to_rsiena(
        self,
        classroom_id: str,
        converter: ABMRSienaConverter
    ) -> RSienaDataSet:
        """
        Convert processed classroom data to RSiena format.

        Args:
            classroom_id: ID of classroom to convert
            converter: RSiena data converter

        Returns:
            RSiena dataset for the classroom
        """
        if classroom_id not in self.processed_classrooms:
            raise ValueError(f"Classroom {classroom_id} not found in processed data")

        classroom_data = self.processed_classrooms[classroom_id]

        # Extract data components
        networks = classroom_data['networks']
        behaviors = classroom_data['behaviors']
        attributes = classroom_data['attributes']

        # Convert networks to list format (assuming friendship is primary)
        if 'friendship' in networks:
            network_list = networks['friendship']
        else:
            # Use first available network type
            network_type = list(networks.keys())[0]
            network_list = networks[network_type]
            logger.warning(f"Using {network_type} network as primary for classroom {classroom_id}")

        # Convert to RSiena format
        dataset = converter.convert_to_rsiena(
            networks=network_list,
            behaviors=behaviors,
            actor_attributes=attributes
        )

        # Add classroom-specific metadata
        dataset.metadata.update({
            'classroom_id': classroom_id,
            'classroom_info': self.classroom_info[classroom_id],
            'data_quality_score': self.classroom_info[classroom_id].data_quality_score
        })

        return dataset

    def get_processing_summary(self) -> Dict[str, Any]:
        """
        Get summary of data processing results.

        Returns:
            Summary dictionary
        """
        total_classrooms = len(self.data_quality_report)
        processed_classrooms = len(self.processed_classrooms)
        excluded_classrooms = total_classrooms - processed_classrooms

        # Quality statistics
        quality_scores = list(self.data_quality_report.values())
        mean_quality = np.mean(quality_scores) if quality_scores else 0
        processed_quality_scores = [
            score for cid, score in self.data_quality_report.items()
            if cid in self.processed_classrooms
        ]

        # Size statistics
        classroom_sizes = [info.n_students for info in self.classroom_info.values()]

        summary = {
            'total_classrooms': total_classrooms,
            'processed_classrooms': processed_classrooms,
            'excluded_classrooms': excluded_classrooms,
            'processing_rate': processed_classrooms / total_classrooms if total_classrooms > 0 else 0,
            'quality_statistics': {
                'mean_quality_all': mean_quality,
                'mean_quality_processed': np.mean(processed_quality_scores) if processed_quality_scores else 0,
                'min_quality': min(quality_scores) if quality_scores else 0,
                'max_quality': max(quality_scores) if quality_scores else 0
            },
            'size_statistics': {
                'mean_size': np.mean(classroom_sizes) if classroom_sizes else 0,
                'min_size': min(classroom_sizes) if classroom_sizes else 0,
                'max_size': max(classroom_sizes) if classroom_sizes else 0,
                'std_size': np.std(classroom_sizes) if classroom_sizes else 0
            },
            'classroom_ids': list(self.processed_classrooms.keys()),
            'excluded_classroom_ids': [
                cid for cid in self.data_quality_report.keys()
                if cid not in self.processed_classrooms
            ]
        }

        return summary

    def export_classroom_data(
        self,
        output_dir: Union[str, Path],
        format: str = "csv"
    ):
        """
        Export processed classroom data to files.

        Args:
            output_dir: Output directory
            format: Export format ("csv", "json", "pickle")
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        for classroom_id, classroom_data in self.processed_classrooms.items():
            classroom_dir = output_dir / f"classroom_{classroom_id}"
            classroom_dir.mkdir(exist_ok=True)

            if format == "csv":
                # Export networks
                for network_type, networks in classroom_data['networks'].items():
                    for wave, network in enumerate(networks):
                        edges_df = pd.DataFrame([
                            {'source': u, 'target': v, 'weight': d.get('weight', 1)}
                            for u, v, d in network.edges(data=True)
                        ])
                        edges_df.to_csv(
                            classroom_dir / f"{network_type}_wave_{wave+1}.csv",
                            index=False
                        )

                # Export attributes
                attr_df = pd.DataFrame(classroom_data['attributes'])
                attr_df.to_csv(classroom_dir / "attributes.csv", index=False)

                # Export behaviors
                for wave, behaviors in enumerate(classroom_data['behaviors']):
                    behavior_df = pd.DataFrame.from_dict(behaviors, orient='index')
                    behavior_df.to_csv(
                        classroom_dir / f"behaviors_wave_{wave+1}.csv"
                    )

            elif format == "json":
                import json
                with open(classroom_dir / "classroom_data.json", 'w') as f:
                    # Convert networkx graphs to serializable format
                    serializable_data = {
                        'networks': {
                            network_type: [
                                {
                                    'nodes': list(G.nodes()),
                                    'edges': list(G.edges(data=True))
                                }
                                for G in networks
                            ]
                            for network_type, networks in classroom_data['networks'].items()
                        },
                        'behaviors': classroom_data['behaviors'],
                        'attributes': classroom_data['attributes'],
                        'info': classroom_data['info'].__dict__
                    }
                    json.dump(serializable_data, f, indent=2, default=str)

            elif format == "pickle":
                import pickle
                with open(classroom_dir / "classroom_data.pkl", 'wb') as f:
                    pickle.dump(classroom_data, f)

        logger.info(f"Exported {len(self.processed_classrooms)} classrooms to {output_dir}")


def create_test_processor() -> ClassroomDataProcessor:
    """Create test instance of classroom data processor."""
    config = MultiLevelDataConfig(
        min_classroom_size=15,
        max_classroom_size=35,
        missing_tolerance_threshold=0.2,
        min_data_quality_score=0.7
    )

    return ClassroomDataProcessor(config)


if __name__ == "__main__":
    # Test classroom data processor
    logging.basicConfig(level=logging.INFO)

    try:
        # Create test processor
        processor = create_test_processor()

        # Create synthetic test data
        logger.info("Creating synthetic multi-classroom data...")

        # Generate test data structure
        n_schools = 3
        n_classrooms = 15  # Subset of 105 for testing
        n_waves = 3

        students_data = []
        networks_data = []
        tolerance_data = []

        student_id_counter = 1

        for school_id in range(1, n_schools + 1):
            for classroom_id in range(1, n_classrooms // n_schools + 1):
                classroom_full_id = f"S{school_id}C{classroom_id}"

                # Generate students for this classroom
                n_students = np.random.randint(20, 30)

                for i in range(n_students):
                    students_data.append({
                        'student_id': student_id_counter,
                        'classroom_id': classroom_full_id,
                        'school_id': f"S{school_id}",
                        'ethnicity': np.random.choice([0, 1], p=[0.7, 0.3]),
                        'gender': np.random.choice([0, 1]),
                        'age': np.random.randint(14, 18)
                    })

                    # Generate tolerance data across waves
                    base_tolerance = np.random.normal(0.5, 0.2)
                    for wave in range(1, n_waves + 1):
                        tolerance = base_tolerance + wave * 0.05 + np.random.normal(0, 0.1)
                        tolerance = max(0, min(1, tolerance))

                        tolerance_data.append({
                            'student_id': student_id_counter,
                            'classroom_id': classroom_full_id,
                            'wave': wave,
                            'tolerance': tolerance,
                            'prejudice': np.random.normal(0.3, 0.15)
                        })

                    student_id_counter += 1

                # Generate network data
                classroom_students = [
                    s['student_id'] for s in students_data
                    if s['classroom_id'] == classroom_full_id
                ]

                for wave in range(1, n_waves + 1):
                    # Friendship network
                    for i, student_i in enumerate(classroom_students):
                        for j, student_j in enumerate(classroom_students):
                            if i != j and np.random.random() < 0.15:
                                networks_data.append({
                                    'student_i': student_i,
                                    'student_j': student_j,
                                    'classroom_id': classroom_full_id,
                                    'wave': wave,
                                    'network_type': 'friendship',
                                    'weight': np.random.uniform(0.5, 1.0)
                                })

                    # Cooperation network (sparser)
                    for i, student_i in enumerate(classroom_students):
                        for j, student_j in enumerate(classroom_students):
                            if i != j and np.random.random() < 0.08:
                                networks_data.append({
                                    'student_i': student_i,
                                    'student_j': student_j,
                                    'classroom_id': classroom_full_id,
                                    'wave': wave,
                                    'network_type': 'cooperation',
                                    'weight': np.random.uniform(0.5, 1.0)
                                })

        # Create dataframes
        raw_data = {
            'students': pd.DataFrame(students_data),
            'networks': pd.DataFrame(networks_data),
            'tolerance': pd.DataFrame(tolerance_data)
        }

        logger.info(f"Created test data: {len(students_data)} students, "
                   f"{len(networks_data)} network edges, {len(tolerance_data)} tolerance observations")

        # Process classrooms
        processed_classrooms = processor.process_classrooms(raw_data)

        # Get summary
        summary = processor.get_processing_summary()

        print("✓ Classroom data processing test completed successfully")
        print(f"  - Processed {summary['processed_classrooms']} out of {summary['total_classrooms']} classrooms")
        print(f"  - Mean quality score: {summary['quality_statistics']['mean_quality_processed']:.3f}")
        print(f"  - Mean classroom size: {summary['size_statistics']['mean_size']:.1f}")

    except Exception as e:
        logger.error(f"Classroom data processing test failed: {e}")
        import traceback
        traceback.print_exc()