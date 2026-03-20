#!/usr/bin/env python3
"""
Plot comprehensive comparison of our model against different opponent types.
Includes: winrate, victory points, and average play time across 2, 3, and 4 player games.
"""

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.gridspec as gridspec
import numpy as np

# ========== WINRATE DATA ==========
# Format: [2_players, 3_players, 4_players]
winrate_random = [0.589, 0.825, 0.888]  # Winrate against random opponents (1000 games each)
winrate_ismcts = [0.01, 0.0, 0.04]  # Winrate against ISMCTS opponents (100 games each)

# ========== VICTORY POINTS DATA ==========
# Format: [[player1_vp], [player2_vp, player3_vp], [player2_vp, player3_vp, player4_vp]]
vp_model_vs_random = [[13.5], [15.4], [15.8]]
vp_random_vs_model = [[12.4], [8.6, 8.6], [7.8, 7.8, 7.7]]

vp_model_vs_ismcts = [[5.3], [6.5], [7.2]]  # Placeholder
vp_ismcts_vs_model = [[15.8], [12.6, 12.0], [11.2, 10.5, 11.5]]  # Placeholder

# ========== PLAY TIME DATA (ms) ==========
# Format: [2_players, 3_players, 4_players]
playtime_random = [0.25, 0.30, 0.55]
playtime_model = [10.3, 16.36, 23.8]
playtime_ismcts = [1323, 2322.5, 4392]

# Configuration
player_counts = ['2 Players', '3 Players', '4 Players']
x = np.arange(len(player_counts))
width = 0.35

# Define consistent colors for each player type
COLOR_MODEL = '#2E7D32'    # Green
COLOR_RANDOM = '#1976D2'   # Blue
COLOR_ISMCTS = '#D32F2F'   # Red

# Create figure with custom grid layout
fig = plt.figure(figsize=(24, 12))
gs = gridspec.GridSpec(2, 3, figure=fig)

# Top row: per-head accuracies (spans 2 columns), winrate
ax_per_head = fig.add_subplot(gs[0, 0:2])  # Top left + top middle
ax_winrate = fig.add_subplot(gs[0, 2])      # Top right

# Bottom row: VP vs random, VP vs ISMCTS, play time
ax_vp_random = fig.add_subplot(gs[1, 0])    # Bottom left
ax_vp_ismcts = fig.add_subplot(gs[1, 1])    # Bottom middle
ax_playtime = fig.add_subplot(gs[1, 2])     # Bottom right

# ========== TOP LEFT + TOP MIDDLE: PER-HEAD ACCURACIES (spans 2 columns) ==========
try:
    per_head_img = mpimg.imread('/home/apprentyr/projects/splendor_ia/logs/per_head_accuracies_test.png')
    ax_per_head.imshow(per_head_img)
    ax_per_head.axis('off')
    ax_per_head.set_title('Per-Head Accuracies', fontsize=14, fontweight='bold')
except FileNotFoundError:
    ax_per_head.text(0.5, 0.5, 'Per-head accuracies\nimage not found',
                     ha='center', va='center', fontsize=12)
    ax_per_head.set_title('Per-Head Accuracies', fontsize=14, fontweight='bold')
    ax_per_head.axis('off')

# ========== TOP RIGHT: WINRATE COMPARISON ==========
bars1 = ax_winrate.bar(x - width/2, winrate_random, width, label='vs Random (1000 games)',
                color=COLOR_RANDOM, edgecolor='black', linewidth=1.2)
bars2 = ax_winrate.bar(x + width/2, winrate_ismcts, width, label='vs ISMCTS (100 games)',
                color=COLOR_ISMCTS, edgecolor='black', linewidth=1.2)

def add_percentage_labels(ax, bars):
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1%}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

add_percentage_labels(ax_winrate, bars1)
add_percentage_labels(ax_winrate, bars2)

ax_winrate.set_ylabel('Win Rate', fontsize=12, fontweight='bold')
ax_winrate.set_xlabel('Number of Players', fontsize=12, fontweight='bold')
ax_winrate.set_title('Model Win Rate vs Different Opponents', fontsize=14, fontweight='bold')
ax_winrate.set_xticks(x)
ax_winrate.set_xticklabels(player_counts)
ax_winrate.legend(fontsize=10)
ax_winrate.set_ylim(0, 1.0)
ax_winrate.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
ax_winrate.grid(axis='y', alpha=0.3, linestyle='--')

# ========== BOTTOM LEFT: VICTORY POINTS (Model vs Random) ==========
# Calculate positions for grouped bars
bar_positions_2p = [0]
bar_positions_3p = [1, 1.3]
bar_positions_4p = [2, 2.3, 2.6]

# Helper function for value labels
def add_value_labels(ax, x_pos, height, offset=0):
    ax.text(x_pos, height, f'{height:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

# 2 players
ax_vp_random.bar(bar_positions_2p[0] - 0.15, vp_model_vs_random[0][0], 0.3,
        color=COLOR_MODEL, edgecolor='black', linewidth=1.2, label='Model')
ax_vp_random.bar(bar_positions_2p[0] + 0.15, vp_random_vs_model[0][0], 0.3,
        color=COLOR_RANDOM, edgecolor='black', linewidth=1.2, label='Random')

# 3 players
ax_vp_random.bar(bar_positions_3p[0] - 0.15, vp_model_vs_random[1][0], 0.3, color=COLOR_MODEL, edgecolor='black', linewidth=1.2)
ax_vp_random.bar(bar_positions_3p[0] + 0.15, np.mean(vp_random_vs_model[1]), 0.3, color=COLOR_RANDOM, edgecolor='black', linewidth=1.2)

# 4 players
ax_vp_random.bar(bar_positions_4p[0] - 0.15, vp_model_vs_random[2][0], 0.3, color=COLOR_MODEL, edgecolor='black', linewidth=1.2)
ax_vp_random.bar(bar_positions_4p[0] + 0.15, np.mean(vp_random_vs_model[2]), 0.3, color=COLOR_RANDOM, edgecolor='black', linewidth=1.2)

# Add value labels
add_value_labels(ax_vp_random, bar_positions_2p[0] - 0.15, vp_model_vs_random[0][0])
add_value_labels(ax_vp_random, bar_positions_2p[0] + 0.15, vp_random_vs_model[0][0])
add_value_labels(ax_vp_random, bar_positions_3p[0] - 0.15, vp_model_vs_random[1][0])
add_value_labels(ax_vp_random, bar_positions_3p[0] + 0.15, np.mean(vp_random_vs_model[1]))
add_value_labels(ax_vp_random, bar_positions_4p[0] - 0.15, vp_model_vs_random[2][0])
add_value_labels(ax_vp_random, bar_positions_4p[0] + 0.15, np.mean(vp_random_vs_model[2]))

ax_vp_random.set_ylabel('Average Victory Points', fontsize=12, fontweight='bold')
ax_vp_random.set_xlabel('Number of Players', fontsize=12, fontweight='bold')
ax_vp_random.set_title('Victory Points: Model vs Random', fontsize=14, fontweight='bold')
ax_vp_random.set_xticks([0, 1.15, 2.3])
ax_vp_random.set_xticklabels(['2 Players', '3 Players', '4 Players'])
ax_vp_random.legend(fontsize=10)
ax_vp_random.grid(axis='y', alpha=0.3, linestyle='--')

# ========== BOTTOM MIDDLE: VICTORY POINTS (Model vs ISMCTS) ==========
# 2 players
ax_vp_ismcts.bar(bar_positions_2p[0] - 0.15, vp_model_vs_ismcts[0][0], 0.3,
        color=COLOR_MODEL, edgecolor='black', linewidth=1.2, label='Model')
ax_vp_ismcts.bar(bar_positions_2p[0] + 0.15, vp_ismcts_vs_model[0][0], 0.3,
        color=COLOR_ISMCTS, edgecolor='black', linewidth=1.2, label='ISMCTS')

# 3 players
ax_vp_ismcts.bar(bar_positions_3p[0] - 0.15, vp_model_vs_ismcts[1][0], 0.3, color=COLOR_MODEL, edgecolor='black', linewidth=1.2)
ax_vp_ismcts.bar(bar_positions_3p[0] + 0.15, np.mean(vp_ismcts_vs_model[1]), 0.3, color=COLOR_ISMCTS, edgecolor='black', linewidth=1.2)

# 4 players
ax_vp_ismcts.bar(bar_positions_4p[0] - 0.15, vp_model_vs_ismcts[2][0], 0.3, color=COLOR_MODEL, edgecolor='black', linewidth=1.2)
ax_vp_ismcts.bar(bar_positions_4p[0] + 0.15, np.mean(vp_ismcts_vs_model[2]), 0.3, color=COLOR_ISMCTS, edgecolor='black', linewidth=1.2)

# Add value labels (only if non-zero)
if vp_model_vs_ismcts[0][0] > 0:
    add_value_labels(ax_vp_ismcts, bar_positions_2p[0] - 0.15, vp_model_vs_ismcts[0][0])
    add_value_labels(ax_vp_ismcts, bar_positions_2p[0] + 0.15, vp_ismcts_vs_model[0][0])
    add_value_labels(ax_vp_ismcts, bar_positions_3p[0] - 0.15, vp_model_vs_ismcts[1][0])
    add_value_labels(ax_vp_ismcts, bar_positions_3p[0] + 0.15, np.mean(vp_ismcts_vs_model[1]))
    add_value_labels(ax_vp_ismcts, bar_positions_4p[0] - 0.15, vp_model_vs_ismcts[2][0])
    add_value_labels(ax_vp_ismcts, bar_positions_4p[0] + 0.15, np.mean(vp_ismcts_vs_model[2]))

ax_vp_ismcts.set_ylabel('Average Victory Points', fontsize=12, fontweight='bold')
ax_vp_ismcts.set_xlabel('Number of Players', fontsize=12, fontweight='bold')
ax_vp_ismcts.set_title('Victory Points: Model vs ISMCTS', fontsize=14, fontweight='bold')
ax_vp_ismcts.set_xticks([0, 1.15, 2.3])
ax_vp_ismcts.set_xticklabels(['2 Players', '3 Players', '4 Players'])
ax_vp_ismcts.legend(fontsize=10)
ax_vp_ismcts.grid(axis='y', alpha=0.3, linestyle='--')

# ========== BOTTOM RIGHT: AVERAGE PLAY TIME ==========
bar_width = 0.25
bars_time_random = ax_playtime.bar(x - bar_width, playtime_random, bar_width, label='Random',
                           color=COLOR_RANDOM, edgecolor='black', linewidth=1.2)
bars_time_model = ax_playtime.bar(x, playtime_model, bar_width, label='Model',
                          color=COLOR_MODEL, edgecolor='black', linewidth=1.2)
bars_time_ismcts = ax_playtime.bar(x + bar_width, playtime_ismcts, bar_width, label='ISMCTS',
                           color=COLOR_ISMCTS, edgecolor='black', linewidth=1.2)

def add_time_labels(ax, bars):
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}',
                ha='center', va='bottom', fontsize=9, fontweight='bold')

add_time_labels(ax_playtime, bars_time_random)
add_time_labels(ax_playtime, bars_time_model)
add_time_labels(ax_playtime, bars_time_ismcts)

ax_playtime.set_ylabel('Average Play Time (ms)', fontsize=12, fontweight='bold')
ax_playtime.set_xlabel('Number of Players', fontsize=12, fontweight='bold')
ax_playtime.set_title('Average Play Time Comparison', fontsize=14, fontweight='bold')
ax_playtime.set_xticks(x)
ax_playtime.set_xticklabels(player_counts)
ax_playtime.legend(fontsize=10)
ax_playtime.grid(axis='y', alpha=0.3, linestyle='--')

# Adjust layout and save
plt.tight_layout()
plt.savefig('/home/apprentyr/projects/splendor_ia/scripts/model_comparison.png',
            dpi=300, bbox_inches='tight')
print("Plot saved to: scripts/model_comparison.png")
plt.show()
