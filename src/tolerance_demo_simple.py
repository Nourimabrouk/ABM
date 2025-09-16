#!/usr/bin/env python3
"""
TOLERANCE INTERVENTION RESEARCH - SIMPLIFIED DEMONSTRATION
Agent-Based Model for Interethnic Cooperation through Tolerance
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
import os

# Set publication-quality defaults
plt.style.use('default')
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'serif'
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['legend.fontsize'] = 9

class ToleranceDemo:
    """Simplified tolerance intervention demonstration"""

    def __init__(self, n_students=30, minority_prop=0.3, n_waves=4):
        self.n_students = n_students
        self.minority_prop = minority_prop
        self.n_waves = n_waves
        self.output_dir = 'outputs/visualizations/'
        os.makedirs(self.output_dir, exist_ok=True)

        print("TOLERANCE INTERVENTION RESEARCH DEMONSTRATION")
        print("=" * 60)
        print(f"Students: {n_students} | Minority: {minority_prop*100:.0f}% | Waves: {n_waves}")
        print("=" * 60)

    def generate_data(self):
        """Generate classroom data"""
        print("\\nGenerating Classroom Data...")

        # Student attributes
        n_minority = int(self.n_students * self.minority_prop)
        ethnicity = ['minority'] * n_minority + ['majority'] * (self.n_students - n_minority)
        np.random.shuffle(ethnicity)

        # Initial tolerance levels
        tolerance = np.zeros(self.n_students)
        for i in range(self.n_students):
            if ethnicity[i] == 'minority':
                tolerance[i] = np.random.normal(3.5, 0.8)
            else:
                tolerance[i] = np.random.normal(3.0, 0.8)
        tolerance = np.clip(tolerance, 1, 5)

        self.students = pd.DataFrame({
            'id': range(self.n_students),
            'ethnicity': ethnicity,
            'tolerance_w1': tolerance,
            'tolerance_w2': tolerance.copy(),
            'tolerance_w3': tolerance.copy(),
            'received_intervention': False
        })

        # Generate friendship network
        self.network = nx.erdos_renyi_graph(self.n_students, 0.12)

        # Add homophily
        for i in range(self.n_students):
            for j in range(i+1, self.n_students):
                if self.students.iloc[i]['ethnicity'] == self.students.iloc[j]['ethnicity']:
                    if np.random.random() < 0.3 and not self.network.has_edge(i, j):
                        if len(list(self.network.neighbors(i))) < 6:  # Degree constraint
                            self.network.add_edge(i, j)

        print(f"Generated {self.n_students} students")
        print(f"Network density: {nx.density(self.network):.3f}")

    def apply_intervention(self):
        """Apply tolerance intervention"""
        print("\\nApplying Clustered Intervention...")

        # Target 25% of students using clustered strategy
        n_target = int(self.n_students * 0.25)

        # Select connected cluster
        start = np.random.choice(self.n_students)
        targets = list(nx.single_source_shortest_path_length(
            self.network, start, cutoff=2).keys())[:n_target]

        self.students.loc[self.students['id'].isin(targets), 'received_intervention'] = True

        # Increase tolerance for targets
        for target in targets:
            current = self.students.loc[target, 'tolerance_w1']
            new_tolerance = current + 1.0  # 1 SD increase
            self.students.loc[target, 'tolerance_w2'] = np.clip(new_tolerance, 1, 5)

        print(f"Applied intervention to {len(targets)} students")

    def simulate_diffusion(self):
        """Simulate tolerance diffusion"""
        print("\\nSimulating Influence Diffusion...")

        # Wave 2 -> Wave 3 diffusion
        for i in range(self.n_students):
            friends = list(self.network.neighbors(i))

            if friends:
                my_tolerance = self.students.loc[i, 'tolerance_w2']
                friend_tolerances = self.students.loc[friends, 'tolerance_w2'].values

                total_change = 0
                for friend_tol in friend_tolerances:
                    diff = abs(my_tolerance - friend_tol)

                    # Attraction-repulsion mechanism
                    if 0.5 <= diff <= 1.5:  # Attraction
                        total_change += 0.1 * (friend_tol - my_tolerance)
                    elif diff > 1.5:  # Repulsion
                        total_change += -0.05 * np.sign(friend_tol - my_tolerance)

                new_val = my_tolerance + total_change
                self.students.loc[i, 'tolerance_w3'] = np.clip(new_val, 1, 5)

        print("Completed influence diffusion")

    def create_visualizations(self):
        """Create publication-quality visualizations"""
        print("\\nCreating Visualizations...")

        # Create comprehensive figure
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Tolerance Intervention Effects on Interethnic Cooperation\\n' +
                    'Agent-Based Model with Attraction-Repulsion Influence',
                    fontsize=16, fontweight='bold')

        # 1. Network plot
        self.plot_network(axes[0, 0])

        # 2. Tolerance evolution
        self.plot_tolerance_evolution(axes[0, 1])

        # 3. Intervention effectiveness
        self.plot_effectiveness(axes[0, 2])

        # 4. Distribution comparison
        self.plot_distributions(axes[1, 0])

        # 5. Statistical summary
        self.plot_statistics(axes[1, 1])

        # 6. Effect sizes
        self.plot_effect_sizes(axes[1, 2])

        plt.tight_layout()

        # Save figure
        output_path = f"{self.output_dir}tolerance_intervention_comprehensive.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved visualization: {output_path}")

        # Also save as PDF
        pdf_path = f"{self.output_dir}tolerance_intervention_comprehensive.pdf"
        plt.savefig(pdf_path, bbox_inches='tight', facecolor='white')
        print(f"Saved PDF: {pdf_path}")

        plt.show()
        return fig

    def plot_network(self, ax):
        """Plot network structure"""
        pos = nx.spring_layout(self.network, k=1.5, iterations=50, seed=42)

        # Node colors and sizes
        node_colors = []
        node_sizes = []

        for i in range(self.n_students):
            if self.students.iloc[i]['received_intervention']:
                node_colors.append('gold')
            elif self.students.iloc[i]['ethnicity'] == 'minority':
                node_colors.append('red')
            else:
                node_colors.append('lightblue')

            node_sizes.append(self.students.iloc[i]['tolerance_w3'] * 100)

        # Draw network
        nx.draw_networkx_nodes(self.network, pos, node_color=node_colors,
                              node_size=node_sizes, alpha=0.8, ax=ax)
        nx.draw_networkx_edges(self.network, pos, edge_color='gray',
                              alpha=0.3, ax=ax)

        ax.set_title('Social Network Structure\\n(Size=Tolerance, Color=Group/Intervention)',
                    fontweight='bold')
        ax.axis('off')

        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='gold', label='Intervention Target'),
            Patch(facecolor='red', label='Minority'),
            Patch(facecolor='lightblue', label='Majority')
        ]
        ax.legend(handles=legend_elements, loc='upper right')

    def plot_tolerance_evolution(self, ax):
        """Plot tolerance over time"""
        treated = self.students[self.students['received_intervention']]
        control = self.students[~self.students['received_intervention']]

        waves = [1, 2, 3]
        treated_means = [
            treated['tolerance_w1'].mean(),
            treated['tolerance_w2'].mean(),
            treated['tolerance_w3'].mean()
        ]
        control_means = [
            control['tolerance_w1'].mean(),
            control['tolerance_w2'].mean(),
            control['tolerance_w3'].mean()
        ]

        ax.plot(waves, treated_means, 'o-', color='red', linewidth=2,
                markersize=8, label='Treated')
        ax.plot(waves, control_means, 's-', color='blue', linewidth=2,
                markersize=8, label='Control')

        ax.set_xlabel('Wave', fontweight='bold')
        ax.set_ylabel('Mean Tolerance', fontweight='bold')
        ax.set_title('Tolerance Evolution Over Time', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim([1, 5])

    def plot_effectiveness(self, ax):
        """Plot intervention effectiveness"""
        treated = self.students[self.students['received_intervention']]
        control = self.students[~self.students['received_intervention']]

        # Calculate effect at each wave
        effects = []
        waves = ['Wave 1', 'Wave 2', 'Wave 3']

        for wave in ['w1', 'w2', 'w3']:
            col = f'tolerance_{wave}'
            effect = treated[col].mean() - control[col].mean()
            effects.append(effect)

        bars = ax.bar(waves, effects, color=['gray', 'orange', 'green'], alpha=0.8)
        ax.set_ylabel('Effect Size (Treatment - Control)', fontweight='bold')
        ax.set_title('Intervention Effectiveness by Wave', fontweight='bold')
        ax.grid(True, alpha=0.3)

        # Add value labels
        for bar, effect in zip(bars, effects):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                   f'{effect:.3f}', ha='center', va='bottom', fontweight='bold')

    def plot_distributions(self, ax):
        """Plot tolerance distributions"""
        treated = self.students[self.students['received_intervention']]
        control = self.students[~self.students['received_intervention']]

        ax.hist(control['tolerance_w3'], alpha=0.6, label='Control',
                color='blue', bins=15, density=True)
        ax.hist(treated['tolerance_w3'], alpha=0.6, label='Treated',
                color='red', bins=15, density=True)

        ax.set_xlabel('Final Tolerance Level', fontweight='bold')
        ax.set_ylabel('Density', fontweight='bold')
        ax.set_title('Final Tolerance Distributions', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

    def plot_statistics(self, ax):
        """Plot statistical summary"""
        treated = self.students[self.students['received_intervention']]
        control = self.students[~self.students['received_intervention']]

        # T-test
        t_stat, p_value = stats.ttest_ind(treated['tolerance_w3'], control['tolerance_w3'])

        # Cohen's d
        pooled_std = np.sqrt((treated['tolerance_w3'].var() + control['tolerance_w3'].var()) / 2)
        cohens_d = (treated['tolerance_w3'].mean() - control['tolerance_w3'].mean()) / pooled_std

        stats_data = {
            'Sample Size': f'{self.n_students} students',
            'Treated': f'{sum(self.students["received_intervention"])} students',
            'T-statistic': f'{t_stat:.3f}',
            'P-value': f'{p_value:.4f}',
            'Cohens_d': f'{cohens_d:.3f}',
            'Effect Size': 'Large' if abs(cohens_d) > 0.8 else 'Medium' if abs(cohens_d) > 0.5 else 'Small'
        }

        y_pos = np.arange(len(stats_data))
        ax.barh(y_pos, [1]*len(stats_data), alpha=0.1)

        for i, (key, value) in enumerate(stats_data.items()):
            ax.text(0.05, i, f'{key}: {value}', fontsize=10, va='center')

        ax.set_yticks(y_pos)
        ax.set_yticklabels([])
        ax.set_xlim([0, 1])
        ax.set_title('Statistical Summary', fontweight='bold')
        ax.set_xticks([])

    def plot_effect_sizes(self, ax):
        """Plot effect sizes comparison"""
        treated = self.students[self.students['received_intervention']]
        control = self.students[~self.students['received_intervention']]

        # Effect sizes for different measures
        measures = ['Tolerance\\nIncrease', 'Network\\nPosition', 'Influence\\nSpread']

        # Calculate effect sizes
        tolerance_effect = (treated['tolerance_w3'].mean() - treated['tolerance_w1'].mean()) / \
                          treated['tolerance_w1'].std()

        # Network position effect (approximation)
        treated_centrality = [len(list(self.network.neighbors(i)))
                             for i in treated.index]
        control_centrality = [len(list(self.network.neighbors(i)))
                             for i in control.index]
        network_effect = (np.mean(treated_centrality) - np.mean(control_centrality)) / \
                        np.std(control_centrality) if np.std(control_centrality) > 0 else 0

        # Influence spread (wave 2 to 3 change)
        treated_change = treated['tolerance_w3'] - treated['tolerance_w2']
        control_change = control['tolerance_w3'] - control['tolerance_w2']
        influence_effect = (treated_change.mean() - control_change.mean()) / \
                          control_change.std() if control_change.std() > 0 else 0

        effects = [tolerance_effect, network_effect, influence_effect]
        colors = ['green' if e > 0.5 else 'orange' if e > 0.2 else 'red' for e in effects]

        bars = ax.bar(measures, effects, color=colors, alpha=0.8)
        ax.set_ylabel('Effect Size (Cohens d)', fontweight='bold')
        ax.set_title('Multiple Effect Size Analysis', fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0.2, color='red', linestyle='--', alpha=0.5, label='Small')
        ax.axhline(y=0.5, color='orange', linestyle='--', alpha=0.5, label='Medium')
        ax.axhline(y=0.8, color='green', linestyle='--', alpha=0.5, label='Large')

        # Add value labels
        for bar, effect in zip(bars, effects):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                   f'{effect:.3f}', ha='center', va='bottom', fontweight='bold')

    def run_demo(self):
        """Run complete demo"""
        print("\\n" + "="*60)
        print("STARTING TOLERANCE INTERVENTION DEMO")
        print("="*60)

        self.generate_data()
        self.apply_intervention()
        self.simulate_diffusion()
        fig = self.create_visualizations()

        print("\\n" + "="*60)
        print("DEMO COMPLETED SUCCESSFULLY!")
        print("="*60)

        # Summary statistics
        treated = self.students[self.students['received_intervention']]
        control = self.students[~self.students['received_intervention']]
        t_stat, p_value = stats.ttest_ind(treated['tolerance_w3'], control['tolerance_w3'])
        pooled_std = np.sqrt((treated['tolerance_w3'].var() + control['tolerance_w3'].var()) / 2)
        cohens_d = (treated['tolerance_w3'].mean() - control['tolerance_w3'].mean()) / pooled_std

        print("\\nKEY FINDINGS:")
        print(f"• Tolerance increased by {(self.students['tolerance_w3'].mean() - self.students['tolerance_w1'].mean()):.3f} points")
        print(f"• Treatment effect: {cohens_d:.3f} (Cohens d)")
        print(f"• Statistical significance: p = {p_value:.4f}")
        print(f"• Network density: {nx.density(self.network):.3f}")
        print("\\nReady for PhD defense and publication!")

        return fig

if __name__ == "__main__":
    demo = ToleranceDemo(n_students=30, minority_prop=0.3, n_waves=4)
    fig = demo.run_demo()