#!/bin/bash
# Easy wrapper for evaluating existing experiments
# Usage: ./eval_existing_experiment.sh <experiment_dir> <game>
#
# Examples:
#   ./eval_existing_experiment.sh experiments/Pong_seed42 pong
#   ./eval_existing_experiment.sh comparison_results/vision_symbol_seed123 breakout

set -e

# Parse arguments
EXPERIMENT_DIR="${1}"
GAME="${2}"

if [ -z "$EXPERIMENT_DIR" ] || [ -z "$GAME" ]; then
    echo "Usage: $0 <experiment_dir> <game>"
    echo ""
    echo "Examples:"
    echo "  $0 experiments/Pong_seed42 pong"
    echo "  $0 comparison_results/vision_symbol_seed123 breakout"
    exit 1
fi

echo "======================================================================="
echo "Evaluating Existing Experiment"
echo "======================================================================="
echo "Directory: $EXPERIMENT_DIR"
echo "Game: $GAME"
echo "======================================================================="
echo ""

# Check if directory exists
if [ ! -d "$EXPERIMENT_DIR" ]; then
    echo "❌ Error: Directory not found: $EXPERIMENT_DIR"
    exit 1
fi

# Check if Results directory exists
if [ ! -d "$EXPERIMENT_DIR/Results" ]; then
    echo "❌ Error: No Results directory found in $EXPERIMENT_DIR"
    exit 1
fi

# Run evaluation
python run_coordinate_evaluation_v2.py \
    --experiment_dir "$EXPERIMENT_DIR" \
    --game "$GAME"

echo ""
echo "======================================================================="
echo "✅ Evaluation Complete!"
echo "======================================================================="
echo "Results saved to: $EXPERIMENT_DIR/Results/coordinate_evaluation/"
