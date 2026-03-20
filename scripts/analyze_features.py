"""
Simple script to analyze features at each preprocessing stage.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from src.imitation_learning.parallel_processor import (
    fill_nan_values_for_row,
    compact_cards_and_add_position_for_row
)
from src.imitation_learning.feature_engineering import extract_all_features, get_all_feature_names

# Load a single CSV file
csv_path = "data/games/2_games/1.csv"
print(f"Loading {csv_path}...\n")
df = pd.read_csv(csv_path)

# Get first row as example
row = df.iloc[0]
row_dict = row.to_dict()

print("=" * 80)
print("STAGE 1: RAW CSV COLUMNS")
print("=" * 80)
all_cols = list(df.columns)
print(f"Total columns: {len(all_cols)}\n")

# Separate metadata and labels
metadata = ['game_id', 'num_players', 'turn_number', 'current_player',
            'player0_position', 'player1_position', 'player2_position', 'player3_position']
labels = ['action_type', 'card_selection', 'card_reservation', 'noble_selection',
          'gem_take3_white', 'gem_take3_blue', 'gem_take3_green', 'gem_take3_red', 'gem_take3_black',
          'gem_take2_white', 'gem_take2_blue', 'gem_take2_green', 'gem_take2_red', 'gem_take2_black',
          'gems_removed_white', 'gems_removed_blue', 'gems_removed_green', 'gems_removed_red', 'gems_removed_black', 'gems_removed_gold']

features = [col for col in all_cols if col not in metadata and col not in labels]

print(f"Metadata (excluded from model): {len(metadata)} columns")
print(f"  List: {metadata}")

print(f"\nLabels (outputs): {len(labels)} columns")
print(f"  Complete list: {labels}")

print(f"\nFeatures (inputs): {len(features)} columns")

# Find one-hot features (bonuses)
bonus_features = [col for col in features if '_bonus_' in col]
non_bonus_features = [col for col in features if '_bonus_' not in col]

print(f"One-hot encoded (card bonuses): {len(bonus_features)} columns")
print("Pattern: card{0-11}_bonus_{white,blue,green,red,black}, player{0-3}_reserved{0-2}_bonus_{white,blue,green,red,black}")
print(f"\nContinuous/other features: {len(non_bonus_features)} columns")
print("Patterns:")
print("  - gems_board_{white,blue,green,red,black,gold}")
print("  - deck_level{1,2,3}_remaining")
print("  - card{0-11}_{vp,level,cost_white,cost_blue,cost_green,cost_red,cost_black}")
print("  - noble{0-4}_{vp,req_white,req_blue,req_green,req_red,req_black}")
print("  - player{0-3}_{vp,gems_white,...,gems_gold,reduction_white,...,reduction_black}")
print("  - player{0-3}_reserved{0-2}_{vp,level,cost_white,...,cost_black}")

print("\n" + "=" * 80)
print("STAGE 2: AFTER CARD COMPACTION & ONE-HOT ENCODING (BASELINE)")
print("=" * 80)

# Fill NaN
filled_row = {}
for key, value in row_dict.items():
    if key not in labels and key != 'action_type':
        filled_row[key] = 0 if pd.isna(value) else value
    else:
        filled_row[key] = value

# Compact cards
compacted_row = compact_cards_and_add_position_for_row(filled_row)

# Convert to dataframe for easier manipulation
df_single = pd.DataFrame([compacted_row])

# Simulate one-hot encoding
df_eng = df_single.copy()
onehot_cols = []

# Current player one-hot
for i in range(4):
    col_name = f"current_player_{i}"
    df_eng[col_name] = (df_eng["current_player"] == i).astype(int)
    onehot_cols.append(col_name)

# Num players one-hot
for n in [2, 3, 4]:
    col_name = f"num_players_{n}"
    df_eng[col_name] = (df_eng["num_players"] == n).astype(int)
    onehot_cols.append(col_name)

# Player position one-hot
for player_idx in range(4):
    position_col = f"player{player_idx}_position"
    if position_col in df_eng.columns:
        for pos in range(4):
            col_name = f"{position_col}_{pos}"
            df_eng[col_name] = (df_eng[position_col] == pos).astype(int)
            onehot_cols.append(col_name)

# Get all columns
all_baseline_cols = list(df_eng.columns)

# Remove original categorical and metadata
baseline_features = [col for col in all_baseline_cols
                     if col not in metadata and col not in labels]

# Add turn_number to features
if 'turn_number' not in baseline_features:
    baseline_features.append('turn_number')

print(f"Total baseline features: {len(baseline_features)}\n")

# Categorize
explicit_onehot = onehot_cols
card_bonuses = [col for col in baseline_features if '_bonus_' in col]
position_features = [col for col in baseline_features if 'position' in col]
continuous = [col for col in baseline_features
              if col not in explicit_onehot and col not in card_bonuses and col not in position_features]

print(f"Explicitly one-hot encoded: {len(explicit_onehot)} columns")
print("  Complete list:", explicit_onehot)

print(f"\nCard bonuses (one-hot constraint): {len(card_bonuses)} columns")
print("  Pattern: card{0-11}_bonus_{white,blue,green,red,black}")
print("  Pattern: player{0-3}_reserved{0-2}_bonus_{white,blue,green,red,black}")

print(f"\nPosition indices (discrete): {len(position_features)} columns")
print("  Patterns:")
print("    - card{0-11}_position")
print("    - player{0-3}_reserved{0-2}_position")
print("    - player{0-3}_position_{0-3} (one-hot encoding of player positions)")

print(f"\nContinuous features: {len(continuous)} columns")
print("  Patterns:")
print("    - gems_board_{white,blue,green,red,black,gold}")
print("    - deck_level{1,2,3}_remaining")
print("    - card{0-11}_{vp,level,cost_white,cost_blue,cost_green,cost_red,cost_black}")
print("    - noble{0-4}_{vp,req_white,req_blue,req_green,req_red,req_black}")
print("    - player{0-3}_{vp,gems_white,gems_blue,gems_green,gems_red,gems_black,gems_gold}")
print("    - player{0-3}_{reduction_white,reduction_blue,reduction_green,reduction_red,reduction_black}")
print("    - player{0-3}_reserved{0-2}_{vp,level,cost_white,cost_blue,cost_green,cost_red,cost_black}")
print("    - turn_number")

print("\n" + "=" * 80)
print("STAGE 3: AFTER STRATEGIC FEATURE ENGINEERING (ENHANCED)")
print("=" * 80)

# Get strategic feature names (don't need actual values, just names)
strategic_feature_names = get_all_feature_names()

print(f"Strategic features added: {len(strategic_feature_names)} columns\n")

# Categorize strategic features
token_features = [f for f in strategic_feature_names if 'can_take2_' in f or 'tokens_left_if' in f or 'max_tokens_pile' in f or 'maximum_takeable' in f]
card_features = [f for f in strategic_feature_names if f.startswith('player') and '_card' in f and '_noble' not in f]
noble_features = [f for f in strategic_feature_names if '_noble' in f and '_card' not in f]
synergy_features = [f for f in strategic_feature_names if '_nobles_after_buy' in f or '_closest_noble_distance_after_buy' in f]
comparison_features = [f for f in strategic_feature_names if f.startswith('player') and ('_vp' in f or 'leaderboard' in f or 'distance_to_max' in f or 'gems_reduction' in f or 'buying_capacity' in f or 'total_gem' in f or '_num_' in f)] + [f for f in strategic_feature_names if f.startswith('max_')]
progression_features = [f for f in strategic_feature_names if 'distance_to_end' in f or 'deck_level' in f or 'total_cards' in f or '_vp_to_win' in f]

# Remove duplicates (some might be in multiple categories)
card_features = [f for f in card_features if f not in synergy_features]
noble_features = [f for f in noble_features if f not in synergy_features]
comparison_features = [f for f in comparison_features if f not in card_features and f not in noble_features]

print(f"Token features: {len(token_features)}")
print("  Patterns:")
print("    - can_take2_{white,blue,green,red,black} (5 binary)")
print("    - tokens_left_if_take2_{white,blue,green,red,black} (5)")
print("    - tokens_left_if_take1_{white,blue,green,red,black} (5)")
print("    - max_tokens_pile_{white,blue,green,red,black,gold} (6)")
print("    - maximum_takeable_this_turn_{white,blue,green,red,black} (5)")

print(f"\nCard features: {len(card_features)}")
print("  Patterns (per player 0-3, per card 0-14):")
print("    - player{i}_card{j}_can_build (binary)")
print("    - player{i}_card{j}_must_use_gold (binary)")
print("    - player{i}_card{j}_distance_{white,blue,green,red,black} (5)")
print("    - player{i}_card{j}_distance_total")
print("    - player{i}_card{j}_vp_if_buy")

print(f"\nNoble features: {len(noble_features)}")
print("  Patterns (per player 0-3, per noble 0-4):")
print("    - player{i}_noble{j}_distance_{white,blue,green,red,black} (5)")
print("    - player{i}_noble{j}_distance_total")
print("    - player{i}_noble{j}_acquirable (binary)")
print("  Per player aggregates:")
print("    - player{i}_closest_noble_distance")
print("    - player{i}_nobles_acquirable_count")

print(f"\nCard-noble synergy: {len(synergy_features)}")
print("  Patterns (per player 0-3, per card 0-14):")
print("    - player{i}_card{j}_nobles_after_buy")
print("    - player{i}_card{j}_closest_noble_distance_after_buy")

print(f"\nPlayer comparison: {len(comparison_features)}")
print("  Patterns (per player 0-3):")
print("    - player{i}_vp")
print("    - player{i}_total_gems_reduction")
print("    - player{i}_buying_capacity")
print("    - player{i}_total_gems_possessed")
print("    - player{i}_total_gem_colors_possessed")
print("    - player{i}_num_reserved_cards")
print("    - player{i}_num_nobles_acquired")
print("    - player{i}_num_cards_bought")
print("    - player{i}_distance_to_max_vp")
print("    - player{i}_leaderboard_position")
print("    - player{i}_vp_gap_to_leader")
print("    - player{i}_gems_reduction_leaderboard_position")
print("  Global:")
print("    - max_vp_among_players")
print("    - max_gems_reduction_among_players")

print(f"\nGame progression: {len(progression_features)}")
print("  Complete list:")
print("    - distance_to_end_game")
print("    - deck_level1_remaining")
print("    - deck_level2_remaining")
print("    - deck_level3_remaining")
print("    - total_cards_bought")
print("    - player{0-3}_vp_to_win (4)")

# Identify binary strategic features
binary_strategic = [f for f in strategic_feature_names if 'can_build' in f or 'must_use_gold' in f or 'acquirable' in f or 'can_take2_' in f]
continuous_strategic = [f for f in strategic_feature_names if f not in binary_strategic]

print(f"\n{'='*80}")
print("STRATEGIC FEATURES - NORMALIZATION BREAKDOWN")
print("="*80)
print(f"Binary strategic features (NOT normalized): {len(binary_strategic)}")
print("  Patterns:")
print("    - can_take2_{white,blue,green,red,black} (5)")
print("    - player{0-3}_card{0-14}_can_build (60)")
print("    - player{0-3}_card{0-14}_must_use_gold (60)")
print("    - player{0-3}_noble{0-4}_acquirable (20)")
print(f"  Count: 5 + 60 + 60 + 20 + 4 (nobles_acquirable_count) = {len(binary_strategic)}")

print(f"\nContinuous strategic features (normalized): {len(continuous_strategic)}")
print("  All other strategic features (distances, counts, VPs, capacities)")

print("\n" + "=" * 80)
print("FINAL SUMMARY")
print("=" * 80)
print(f"Baseline features: {len(baseline_features)}")
print(f"Strategic features: {len(strategic_feature_names)}")
print(f"Total enhanced features: {len(baseline_features) + len(strategic_feature_names)}")
print("\nOne-hot / Binary breakdown:")
print(f"  Explicit one-hot: {len(explicit_onehot)}")
print(f"  Card bonuses: {len(card_bonuses)}")
print(f"  Position indices: {len(position_features)}")
print(f"  Binary strategic: {len(binary_strategic)}")
print(f"  Total non-normalized: {len(explicit_onehot) + len(card_bonuses) + len(position_features) + len(binary_strategic)}")
