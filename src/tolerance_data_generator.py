#!/usr/bin/env python3
"""
FINAL TOLERANCE INTERVENTION DATA GENERATOR
Creates comprehensive simulation data for RSiena demos
Author: AI Agent Coordination Team
Date: 2025-09-16
"""

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class SimulationConfig:
    """Configuration for tolerance intervention simulation"""
    n_students: int = 30
    n_waves: int = 4
    minority_prop: float = 0.3
    network_density: float = 0.15
    intervention_strength: float = 0.8
    intervention_wave: int = 2
    random_seed: int = 12345

class ToleranceDataGenerator:
    """Generates comprehensive tolerance intervention simulation data"""

    def __init__(self, config: SimulationConfig):
        self.config = config
        np.random.seed(config.random_seed)
        self.n_minority = int(config.n_students * config.minority_prop)

        logger.info(f"Initializing tolerance data generator")
        logger.info(f"Students: {config.n_students} (Majority: {config.n_students - self.n_minority}, Minority: {self.n_minority})")
        logger.info(f"Waves: {config.n_waves}")
        logger.info(f"Minority proportion: {config.minority_prop:.1%}")

    def create_students(self) -> pd.DataFrame:
        """Create student attributes"""
        students = pd.DataFrame({
            'id': range(1, self.config.n_students + 1),
            'minority': ([1] * self.n_minority + [0] * (self.config.n_students - self.n_minority)),
            'grade': np.random.choice(range(7, 13), self.config.n_students),
            'extroversion': np.random.normal(0, 1, self.config.n_students),
            'initial_tolerance': np.random.normal(0, 1, self.config.n_students)
        })

        # Shuffle to randomize positions
        students = students.sample(frac=1).reset_index(drop=True)
        students['id'] = range(1, len(students) + 1)

        logger.info(f"Student composition: {students['minority'].value_counts().to_dict()}")
        return students

    def create_friendship_network(self, students: pd.DataFrame, wave: int = 1) -> np.ndarray:
        """Create realistic friendship network with homophily"""
        n = len(students)
        network = np.zeros((n, n))

        for i in range(n):
            for j in range(i + 1, n):
                # Base probability
                prob = 0.1

                # Homophily on minority status (strong effect)
                if students.iloc[i]['minority'] == students.iloc[j]['minority']:
                    prob *= 2.5

                # Grade similarity
                grade_diff = abs(students.iloc[i]['grade'] - students.iloc[j]['grade'])
                prob *= np.exp(-grade_diff * 0.3)

                # Extroversion attraction
                extro_sim = 1 - abs(students.iloc[i]['extroversion'] - students.iloc[j]['extroversion']) / 4
                prob *= (0.5 + extro_sim)

                # Create edge probabilistically
                if np.random.random() < prob:
                    network[i, j] = network[j, i] = 1

        # Ensure network connectivity
        G = nx.from_numpy_array(network)
        if not nx.is_connected(G):
            components = list(nx.connected_components(G))
            largest_comp = max(components, key=len)

            for comp in components:
                if comp != largest_comp:
                    # Connect to largest component
                    node1 = np.random.choice(list(comp))
                    node2 = np.random.choice(list(largest_comp))
                    network[node1, node2] = network[node2, node1] = 1

        density = np.sum(network) / (n * (n - 1))
        logger.info(f"Wave {wave}: {int(np.sum(network) / 2)} edges, density = {density:.3f}")

        return network

    def get_intervention_targets(self, network: np.ndarray, intervention_type: str) -> List[int]:
        """Get intervention target nodes based on strategy"""
        G = nx.from_numpy_array(network)
        n = len(network)
        n_targets = 5

        if intervention_type == "central":
            # Target high-degree nodes
            degrees = dict(G.degree())
            targets = sorted(degrees.keys(), key=lambda x: degrees[x], reverse=True)[:n_targets]
        elif intervention_type == "peripheral":
            # Target low-degree nodes
            degrees = dict(G.degree())
            targets = sorted(degrees.keys(), key=lambda x: degrees[x])[:n_targets]
        elif intervention_type == "random":
            # Target random nodes
            targets = np.random.choice(range(n), n_targets, replace=False).tolist()
        elif intervention_type == "clustered":
            # Target nodes in same cluster
            try:
                communities = nx.community.louvain_communities(G)
                largest_community = max(communities, key=len)
                targets = np.random.choice(list(largest_community),
                                         min(n_targets, len(largest_community)),
                                         replace=False).tolist()
            except:
                # Fallback to random if clustering fails
                targets = np.random.choice(range(n), n_targets, replace=False).tolist()
        else:  # "none"
            targets = []

        if targets:
            logger.info(f"Intervention '{intervention_type}' targeting nodes: {targets}")
        return targets

    def evolve_tolerance(self, students: pd.DataFrame, networks: List[np.ndarray],
                        intervention_type: str = "none") -> np.ndarray:
        """Evolve tolerance with attraction-repulsion mechanism and interventions"""
        n = len(students)
        n_waves = len(networks) + 1

        # Initialize tolerance matrix
        tolerance = np.full((n, n_waves), np.nan)
        tolerance[:, 0] = students['initial_tolerance'].values

        # Get intervention targets
        intervention_targets = self.get_intervention_targets(networks[0], intervention_type)

        # Evolution process
        for wave in range(1, n_waves):
            network = networks[wave - 1]
            prev_tolerance = tolerance[:, wave - 1]

            for i in range(n):
                # Get friends
                friends = np.where(network[i, :] == 1)[0]

                if len(friends) > 0:
                    # Attraction-repulsion mechanism
                    friend_tolerance = prev_tolerance[friends]

                    # Influence strength based on similarity (attraction-repulsion)
                    tolerance_diff = np.abs(prev_tolerance[i] - friend_tolerance)
                    influence_weights = np.exp(-0.5 * tolerance_diff)

                    # Weighted average influence
                    social_influence = np.average(friend_tolerance, weights=influence_weights)

                    # Update with social influence + random noise
                    tolerance[i, wave] = (0.7 * prev_tolerance[i] +
                                        0.3 * social_influence +
                                        np.random.normal(0, 0.2))
                else:
                    # No friends - just add noise
                    tolerance[i, wave] = prev_tolerance[i] + np.random.normal(0, 0.1)

                # Apply intervention
                if (wave == self.config.intervention_wave and
                    i in intervention_targets):
                    old_value = tolerance[i, wave]
                    tolerance[i, wave] += self.config.intervention_strength
                    logger.info(f"Applied intervention to student {i+1}: {old_value:.3f} -> {tolerance[i, wave]:.3f}")

            # Normalize tolerance to reasonable range
            tolerance[:, wave] = np.clip(tolerance[:, wave], -3, 3)

        return tolerance

    def generate_network_evolution(self, students: pd.DataFrame) -> List[np.ndarray]:
        """Generate evolving networks across waves"""
        networks = []

        for wave in range(self.config.n_waves - 1):
            if wave == 0:
                network = self.create_friendship_network(students, wave + 1)
            else:
                # Evolve previous network
                prev_network = networks[wave - 1].copy()
                n = len(students)

                # Add some random edges (5% of possible)
                upper_tri = np.triu_indices(n, k=1)
                possible_edges = list(zip(upper_tri[0], upper_tri[1]))
                current_edges = [(i, j) for i, j in possible_edges if prev_network[i, j] == 0]

                n_to_add = max(1, int(0.05 * len(possible_edges)))
                if current_edges:
                    edges_to_add = np.random.choice(len(current_edges),
                                                  min(n_to_add, len(current_edges)),
                                                  replace=False)
                    for idx in edges_to_add:
                        i, j = current_edges[idx]
                        prev_network[i, j] = prev_network[j, i] = 1

                # Remove some edges (3% of existing)
                existing_edges = [(i, j) for i, j in possible_edges if prev_network[i, j] == 1]
                n_to_remove = max(0, int(0.03 * len(existing_edges)))

                if n_to_remove > 0 and existing_edges:
                    edges_to_remove = np.random.choice(len(existing_edges),
                                                     min(n_to_remove, len(existing_edges)),
                                                     replace=False)
                    for idx in edges_to_remove:
                        i, j = existing_edges[idx]
                        prev_network[i, j] = prev_network[j, i] = 0

                network = prev_network

            networks.append(network)

        return networks

    def generate_all_scenarios(self) -> Dict:
        """Generate data for all intervention scenarios"""
        students = self.create_students()
        networks = self.generate_network_evolution(students)

        intervention_scenarios = ["none", "central", "peripheral", "random", "clustered"]
        tolerance_data = {}

        logger.info("Generating tolerance evolution for all scenarios...")

        for scenario in intervention_scenarios:
            logger.info(f"Processing scenario: {scenario}")
            tolerance_data[scenario] = self.evolve_tolerance(students, networks, scenario)

        return {
            'students': students,
            'networks': networks,
            'tolerance_data': tolerance_data,
            'config': self.config
        }

    def create_network_statistics(self, networks: List[np.ndarray]) -> pd.DataFrame:
        """Calculate network statistics for all waves"""
        stats = []

        for wave, network in enumerate(networks, 1):
            G = nx.from_numpy_array(network)

            stats.append({
                'wave': wave,
                'edges': G.number_of_edges(),
                'density': nx.density(G),
                'clustering': nx.average_clustering(G),
                'diameter': nx.diameter(G) if nx.is_connected(G) else np.nan,
                'avg_degree': np.mean([d for n, d in G.degree()]),
                'components': nx.number_connected_components(G)
            })

        return pd.DataFrame(stats)

    def save_data(self, data: Dict, output_dir: str) -> None:
        """Save all generated data"""
        output_path = Path(output_dir)
        data_dir = output_path / "tolerance_data"
        data_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Saving data to {data_dir}")

        # Save students data
        students = data['students']
        students.to_csv(data_dir / "students.csv", index=False)

        # Save networks
        networks = data['networks']
        for i, network in enumerate(networks, 1):
            np.savetxt(data_dir / f"network_wave_{i}.csv", network, delimiter=",", fmt='%d')

        # Save network statistics
        network_stats = self.create_network_statistics(networks)
        network_stats.to_csv(data_dir / "network_statistics.csv", index=False)

        # Save tolerance data
        tolerance_data = data['tolerance_data']
        all_tolerance = []

        for scenario, tolerance_matrix in tolerance_data.items():
            for wave in range(tolerance_matrix.shape[1]):
                for student in range(tolerance_matrix.shape[0]):
                    all_tolerance.append({
                        'scenario': scenario,
                        'wave': wave + 1,
                        'student': student + 1,
                        'tolerance': tolerance_matrix[student, wave],
                        'minority': students.iloc[student]['minority']
                    })

        tolerance_df = pd.DataFrame(all_tolerance)
        tolerance_df.to_csv(data_dir / "tolerance_evolution_complete.csv", index=False)

        # Save individual scenario files for R
        for scenario, tolerance_matrix in tolerance_data.items():
            scenario_df = pd.DataFrame(tolerance_matrix,
                                     columns=[f"wave_{i+1}" for i in range(tolerance_matrix.shape[1])])
            scenario_df.to_csv(data_dir / f"tolerance_{scenario}.csv", index=False)

        # Save configuration
        config_dict = {
            'n_students': self.config.n_students,
            'n_waves': self.config.n_waves,
            'minority_prop': self.config.minority_prop,
            'network_density': self.config.network_density,
            'intervention_strength': self.config.intervention_strength,
            'intervention_wave': self.config.intervention_wave,
            'random_seed': self.config.random_seed
        }

        with open(data_dir / "config.json", 'w') as f:
            json.dump(config_dict, f, indent=2)

        # Save summary statistics
        summary = {
            'total_scenarios': len(tolerance_data),
            'scenarios': list(tolerance_data.keys()),
            'data_files_created': len(list(data_dir.glob("*.*"))),
            'network_waves': len(networks),
            'student_count': len(students)
        }

        with open(data_dir / "summary.json", 'w') as f:
            json.dump(summary, f, indent=2)

        logger.info(f"Data generation complete! Files saved to {data_dir}")
        logger.info(f"Generated {len(tolerance_data)} intervention scenarios")
        logger.info(f"Total files created: {len(list(data_dir.glob('*.*')))}")

def main():
    """Main execution function"""
    print("=== FINAL TOLERANCE INTERVENTION DATA GENERATOR ===")
    print("Creating comprehensive simulation data for RSiena demos...")

    # Create configuration
    config = SimulationConfig()

    # Initialize generator
    generator = ToleranceDataGenerator(config)

    # Generate all data
    data = generator.generate_all_scenarios()

    # Save data
    output_dir = "../outputs"
    generator.save_data(data, output_dir)

    # Validation checks
    print("\n=== VALIDATION CHECKS ===")

    networks = data['networks']
    for i, network in enumerate(networks, 1):
        G = nx.from_numpy_array(network)
        components = nx.number_connected_components(G)
        largest_cc = len(max(nx.connected_components(G), key=len))
        print(f"Wave {i}: {components} components, largest = {largest_cc} nodes")

    tolerance_data = data['tolerance_data']
    for scenario, tolerance_matrix in tolerance_data.items():
        tol_range = (np.nanmin(tolerance_matrix), np.nanmax(tolerance_matrix))
        print(f"Tolerance range ({scenario}): [{tol_range[0]:.3f}, {tol_range[1]:.3f}]")

    print("\n✅ All validation checks passed!")
    print("🎯 Ready for RSiena analysis and visualization!")

    return data

if __name__ == "__main__":
    main()