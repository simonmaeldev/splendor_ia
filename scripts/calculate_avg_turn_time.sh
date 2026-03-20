#!/bin/bash

# Script to calculate average playtime per turn per player from simulation_log.txt
# Formula: (duration / turns / players) averaged across all games

LOG_FILE="${1:-data/simulation_log.txt}"

if [ ! -f "$LOG_FILE" ]; then
    echo "Error: Log file not found: $LOG_FILE"
    exit 1
fi

# Parse the log file and calculate average turn time per player
awk '
BEGIN {
    sum = 0
    count = 0
    total_duration = 0
}
{
    # Extract players, turns, and duration
    # Format: "X players, Y turns, ... duration: Z.Ws, ..."
    if (match($0, /([0-9]+) players, ([0-9]+) turns.*duration: ([0-9.]+)s/, arr)) {
        players = arr[1]
        turns = arr[2]
        duration = arr[3]

        # Calculate time per turn per player for this game
        time_per_turn_per_player = duration / turns / players

        sum += time_per_turn_per_player
        total_duration += duration
        count++
    }
}
END {
    if (count > 0) {
        avg = sum / count
        printf "Total games: %d\n", count
        printf "Total playtime: %.2f seconds (%.2f minutes, %.2f hours)\n", total_duration, total_duration/60, total_duration/3600
        printf "Average playtime per turn per player: %.4f seconds\n", avg
        printf "Average playtime per turn per player: %.2f ms\n", avg * 1000
    } else {
        print "No valid data found in log file"
    }
}
' "$LOG_FILE"
