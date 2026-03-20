import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import os


# Set style
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'sans-serif'

# Color scheme
color_input = '#3B82F6'      # Blue
color_shared = '#8B5CF6'     # Purple
color_heads = ['#10B981', '#F59E0B', '#EF4444', '#EC4899', '#06B6D4', '#8B5CF6', '#F97316']  # Various colors
color_text = '#1F2937'       # Dark gray

fig, ax = plt.subplots(figsize=(14, 10))
ax.set_xlim(0, 14)
ax.set_ylim(0, 10)
ax.axis('off')

# Title
ax.text(7, 9.5, 'Multi-Head Neural Network Architecture', 
        ha='center', va='top', fontsize=18, fontweight='bold')

# ============================================================================
# INPUT LAYER
# ============================================================================
input_box = FancyBboxPatch((6, 8.5), 2, 0.6, 
                           boxstyle="round,pad=0.1", 
                           edgecolor=color_input, 
                           facecolor=color_input, 
                           alpha=0.3, 
                           linewidth=2)
ax.add_patch(input_box)
ax.text(7, 8.8, 'Input State', ha='center', va='center', 
        fontsize=12, fontweight='bold', color=color_input)
ax.text(7, 8.4, '1308 dimensions', ha='center', va='center', 
        fontsize=9, style='italic')

# ============================================================================
# SHARED HIDDEN LAYERS
# ============================================================================
layer_configs = [
    (768, 7.3, 'Hidden Layer 1\n768 units'),
    (512, 6.4, 'Hidden Layer 2\n512 units'),
    (256, 5.5, 'Hidden Layer 3\n256 units')
]

prev_y = 8.5
for units, y_pos, label in layer_configs:
    # Draw box
    width = 2.5
    height = 0.5
    x_pos = 7 - width/2
    
    box = FancyBboxPatch((x_pos, y_pos), width, height,
                         boxstyle="round,pad=0.08",
                         edgecolor=color_shared,
                         facecolor=color_shared,
                         alpha=0.3,
                         linewidth=2)
    ax.add_patch(box)
    ax.text(7, y_pos + height/2, label, 
            ha='center', va='center', fontsize=10, fontweight='bold',
            color=color_shared)
    
    # Arrow from previous layer
    arrow = FancyArrowPatch((7, prev_y), (7, y_pos + height),
                           arrowstyle='->', mutation_scale=20, 
                           linewidth=2, color=color_shared, alpha=0.7)
    ax.add_patch(arrow)
    
    prev_y = y_pos

# ============================================================================
# SPLIT TO HEADS - Fan out arrows
# ============================================================================
split_y = 5.5
fan_out_y = 4.8

# Draw split point
ax.plot(7, split_y - 0.1, 'o', markersize=12, color=color_shared, zorder=10)

# Head positions
head_positions = [1.5, 3.2, 4.9, 6.6, 8.3, 10, 11.7]
head_info = [
    ('Action Type\nHead', '4 classes', [256, 128, 64]),
    ('Card Selection\nHead', '15 classes', [256, 128, 64]),
    ('Card Reserve\nHead', '15 classes', [256, 128, 64]),
    ('Gem Take 3\nHead', '26 classes', [256, 128, 64]),
    ('Gem Take 2\nHead', '5 classes', [256, 128, 64]),
    ('Noble Select\nHead', '5 classes', [256, 128, 64]),
    ('Gems Removed\nHead', '84 classes', [256, 128, 64])
]

# ============================================================================
# HEAD-SPECIFIC LAYERS AND OUTPUTS
# ============================================================================
for i, (x_pos, (head_name, output_classes, hidden_units)) in enumerate(zip(head_positions, head_info)):
    color = color_heads[i]
    
    # Fan-out arrow from split point to head
    arrow = FancyArrowPatch((7, split_y - 0.1), (x_pos, fan_out_y),
                           arrowstyle='->', mutation_scale=15,
                           linewidth=1.5, color=color, alpha=0.6,
                           connectionstyle="arc3,rad=0.3")
    ax.add_patch(arrow)
    
    # Head hidden layer 1 (128 units)
    box1 = FancyBboxPatch((x_pos - 0.6, 4.0), 1.2, 0.4,
                         boxstyle="round,pad=0.05",
                         edgecolor=color,
                         facecolor=color,
                         alpha=0.2,
                         linewidth=1.5)
    ax.add_patch(box1)
    ax.text(x_pos, 4.2, f'{hidden_units[0]}', ha='center', va='center',
            fontsize=8, fontweight='bold', color=color)
    
    # Arrow between head layers
    arrow = FancyArrowPatch((x_pos, 4.0), (x_pos, 3.3),
                           arrowstyle='->', mutation_scale=12,
                           linewidth=1.2, color=color, alpha=0.7)
    ax.add_patch(arrow)
    
    # Head hidden layer 2 (64 units)
    box2 = FancyBboxPatch((x_pos - 0.5, 2.9), 1.0, 0.4,
                         boxstyle="round,pad=0.05",
                         edgecolor=color,
                         facecolor=color,
                         alpha=0.2,
                         linewidth=1.5)
    ax.add_patch(box2)
    ax.text(x_pos, 3.1, f'{hidden_units[1]}', ha='center', va='center',
            fontsize=8, fontweight='bold', color=color)
    
    # Arrow to output
    arrow = FancyArrowPatch((x_pos, 2.9), (x_pos, 2.1),
                           arrowstyle='->', mutation_scale=12,
                           linewidth=1.2, color=color, alpha=0.7)
    ax.add_patch(arrow)
    
    # Output layer
    output_box = FancyBboxPatch((x_pos - 0.6, 1.2), 1.2, 0.9,
                               boxstyle="round,pad=0.08",
                               edgecolor=color,
                               facecolor=color,
                               alpha=0.4,
                               linewidth=2)
    ax.add_patch(output_box)
    ax.text(x_pos, 1.85, head_name, ha='center', va='center',
            fontsize=8, fontweight='bold', color=color)
    ax.text(x_pos, 1.45, output_classes, ha='center', va='center',
            fontsize=7, style='italic', color=color)

# ============================================================================
# LEGEND / KEY INFORMATION
# ============================================================================
legend_y = 0.6
ax.text(7, legend_y, 'Total Parameters: 2,288,986 | Input: Game State (1308-dim) | Output: 7 Action Predictions',
        ha='center', va='center', fontsize=9, style='italic',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.3))

# Add layer labels
ax.text(0.3, 6.5, 'Shared\nLayers', ha='center', va='center',
        fontsize=10, fontweight='bold', color=color_shared,
        rotation=90)

ax.text(0.3, 3.0, 'Head-Specific\nLayers', ha='center', va='center',
        fontsize=10, fontweight='bold', color='gray',
        rotation=90)

ax.text(0.3, 1.6, 'Output\nLayers', ha='center', va='center',
        fontsize=10, fontweight='bold', color='gray',
        rotation=90)
plt.tight_layout()
output_dir = os.path.dirname(os.path.abspath(__file__))
os.makedirs(output_dir, exist_ok=True)
plt.savefig(os.path.join(output_dir, 'architecture_diagram_compact.png'), bbox_inches='tight', dpi=300, facecolor='white')
print(f" Saved: {os.path.join(output_dir, 'architecture_diagram_compact.png')}")
plt.close()

# ============================================================================
# ALTERNATIVE: More compact vertical version
# ============================================================================
fig2, ax2 = plt.subplots(figsize=(12, 11))
ax2.set_xlim(0, 12)
ax2.set_ylim(0, 11)
ax2.axis('off')

# Input - MASSIVE TEXT
input_box = FancyBboxPatch((4.0, 9.5), 4, 0.8,
                          boxstyle="round,pad=0.1",
                          edgecolor='black',
                          facecolor='#6B46C1',
                          alpha=0.9,
                          linewidth=2.5)
ax2.add_patch(input_box)
ax2.text(6, 10.05, 'Input: Game State Vector', ha='center', va='center',
        fontsize=19, fontweight='bold', color='white')
ax2.text(6, 9.65, '1308 dimensions', ha='center', va='center',
        fontsize=17, style='italic', color='white', fontweight='bold')

# Arrow
arrow = FancyArrowPatch((6, 9.5), (6, 8.8),
                       arrowstyle='->', mutation_scale=30,
                       linewidth=3, color='#6B46C1', alpha=0.8)
ax2.add_patch(arrow)

# Shared layers as one block with better colors - MASSIVE TEXT
shared_box = FancyBboxPatch((3.0, 6.8), 6, 2.0,
                           boxstyle="round,pad=0.12",
                           edgecolor='black',
                           facecolor='#9333EA',
                           alpha=0.9,
                           linewidth=2.5)
ax2.add_patch(shared_box)
ax2.text(6, 8.5, 'Shared Hidden Layers', ha='center', va='center',
        fontsize=20, fontweight='bold', color='white')

# Layer details with MASSIVE TEXT
ax2.text(6, 8.05, 'Layer 1: 768 units', ha='center', va='center',
        fontsize=18, color='white', fontweight='bold')
ax2.text(6, 7.65, 'Layer 2: 512 units', ha='center', va='center',
        fontsize=18, color='white', fontweight='bold')
ax2.text(6, 7.25, 'Layer 3: 256 units', ha='center', va='center',
        fontsize=18, color='white', fontweight='bold')
ax2.text(6, 6.9, 'ReLU activation', ha='center', va='center',
        fontsize=16, style='italic', color='white', fontweight='bold')

# Arrow
arrow = FancyArrowPatch((6, 6.8), (6, 6.2),
                       arrowstyle='->', mutation_scale=30,
                       linewidth=3, color='#9333EA', alpha=0.8)
ax2.add_patch(arrow)

# Split indicator - MASSIVE TEXT
ax2.text(6, 6.0, 'Split into 7 Specialized Heads', ha='center', va='center',
        fontsize=17, fontweight='bold', style='italic', color='black',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#FDE047', alpha=0.95, edgecolor='black', linewidth=2))

# Better colors for heads - matching poster style
heads_colors_new = ['#10B981', '#3B82F6', '#EF4444', '#F59E0B', '#06B6D4', '#EC4899', '#8B5CF6']

# Heads arranged in grid with better spacing and sizing
heads_config = [
    [(1.8, 4.1, 'Action Type', '4 classes', 0),
     (4.2, 4.1, 'Card Selection', '15 classes', 1),
     (6.6, 4.1, 'Card Reserve', '15 classes', 2),
     (9.0, 4.1, 'Gem Take 3', '26 classes', 3)],
    [(2.5, 1.3, 'Gem Take 2', '5 classes', 4),
     (5.5, 1.3, 'Noble Select', '5 classes', 5),
     (8.5, 1.3, 'Gems Removed', '84 classes', 6)]
]

for row in heads_config:
    for x, y, name, classes, idx in row:
        color = heads_colors_new[idx]
        
        # Draw box with maximum opacity
        box = FancyBboxPatch((x - 1.0, y), 2.0, 1.4,
                           boxstyle="round,pad=0.12",
                           edgecolor='black',
                           facecolor=color,
                           alpha=0.95,
                           linewidth=2.5)
        ax2.add_patch(box)
        
        # Head name - MASSIVE and bolder
        ax2.text(x, y + 1.1, name, ha='center', va='center',
                fontsize=17, fontweight='bold', color='white')
        
        # Number of classes - MASSIVE
        ax2.text(x, y + 0.75, classes, ha='center', va='center',
                fontsize=16, style='italic', color='white', fontweight='bold')
        
        # Arrow symbol - BIGGER
        ax2.text(x, y + 0.45, '↓', ha='center', va='center',
                fontsize=22, color='white', fontweight='bold')
        
        # Hidden layer info - MASSIVE and clearer
        ax2.text(x, y + 0.15, '256 → 128 → 64', ha='center', va='center',
                fontsize=16, color='white', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.5))
        
        # Arrow from split with better visibility
        if y > 3:  # Top row
            arrow = FancyArrowPatch((6, 5.8), (x, y + 1.4),
                                   arrowstyle='->', mutation_scale=20,
                                   linewidth=2.5, color=color, alpha=0.7,
                                   connectionstyle="arc3,rad=0.2")
        else:  # Bottom row
            arrow = FancyArrowPatch((6, 5.8), (x, y + 1.4),
                                   arrowstyle='->', mutation_scale=20,
                                   linewidth=2.5, color=color, alpha=0.7,
                                   connectionstyle="arc3,rad=0.3")
        ax2.add_patch(arrow)

# Legend with poster style - MASSIVE TEXT
ax2.text(6, 0.4, 'Each head: 3 hidden layers (256 units → 128 units → 64 units) + Output layer with Softmax',
        ha='center', va='center', fontsize=14, style='italic', fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.9, edgecolor='black', linewidth=2))

plt.tight_layout()
output_dir = os.path.dirname(os.path.abspath(__file__))
os.makedirs(output_dir, exist_ok=True)
plt.savefig(os.path.join(output_dir, 'architecture_diagram_compact.png'), bbox_inches='tight', dpi=300, facecolor='white')
print(f" Saved: {os.path.join(output_dir, 'architecture_diagram_compact.png')}")
plt.close()


print("\n" + "="*60)
print("Architecture diagrams created successfully!")
print("="*60)
print("\nGenerated files:")
print("  1. architecture_diagram.png - Horizontal layout with all details")
print("  2. architecture_diagram_compact.png - More compact vertical layout")
