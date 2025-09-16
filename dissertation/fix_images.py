import os
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def create_placeholder_image(filename, title):
    """Create a professional placeholder image"""
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Add border
    rect = patches.Rectangle((0.05, 0.05), 0.9, 0.9, linewidth=2,
                            edgecolor='#2E86AB', facecolor='#F0F0F0')
    ax.add_patch(rect)

    # Add text
    ax.text(0.5, 0.6, title, fontsize=20, ha='center', va='center',
            color='#2E86AB', weight='bold')
    ax.text(0.5, 0.4, 'Placeholder Image', fontsize=14, ha='center',
            va='center', color='#666666', style='italic')
    ax.text(0.5, 0.3, f'({filename})', fontsize=10, ha='center',
            va='center', color='#999999')

    # Remove axes
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    # Save figure
    plt.savefig(filename, dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f"Created: {filename}")

# Dictionary of corrupted images and their titles
images_to_fix = {
    'figures/tolerance_cooperation_plot.png': 'Tolerance-Cooperation Dynamics',
    'figures/complex_contagion_plot.png': 'Complex Contagion Effects',
    'figures/intensity_strategy_plot.png': 'Intervention Intensity Strategy',
    'figures/intervention_persistence.png': 'Intervention Persistence Over Time',
    'figures/meta_analysis_plot.png': 'Meta-Analysis Results',
    'figures/proportion_effects_plot.png': 'Proportion Effects Analysis',
    'figures/targeting_strategy_comparison.png': 'Targeting Strategy Comparison',
    'figures/temporal_dynamics_plot.png': 'Temporal Dynamics',
    'figures/methodology/utrecht_logo.png': 'Utrecht University',
    'outputs/visualizations/tolerance_intervention_comprehensive.png': 'Comprehensive Intervention Results'
}

# Create placeholder images
for filepath, title in images_to_fix.items():
    full_path = os.path.join('C:\\Users\\Nouri\\Documents\\GitHub\\ABM\\dissertation', filepath)

    # Check if file exists and is corrupted (small size)
    if os.path.exists(full_path):
        file_size = os.path.getsize(full_path)
        if file_size < 100:  # File is likely corrupted
            create_placeholder_image(full_path, title)
    else:
        # Create directory if needed
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        create_placeholder_image(full_path, title)

print("\nAll placeholder images created successfully!")