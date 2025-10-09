#!/bin/bash
# Run comprehensive evaluation across multiple models and games
#
# Usage:
#   ./run_full_evaluation_suite.sh [num_frames] [max_eval_frames]
#
# Examples:
#   ./run_full_evaluation_suite.sh 300 100  # Generate 300 frames, evaluate 100
#   ./run_full_evaluation_suite.sh 600 200  # Generate 600 frames, evaluate 200

set -e

NUM_FRAMES="${1:-300}"
MAX_EVAL="${2:-100}"

echo "========================================================================="
echo "VLM Detection Evaluation - Full Suite"
echo "========================================================================="
echo "Configuration:"
echo "  - Total frames to generate: $NUM_FRAMES"
echo "  - Frames to evaluate: $MAX_EVAL"
echo "  - Results will be organized in: evaluation_results/"
echo "========================================================================="
echo ""

# Games to evaluate
GAMES=("pong" "breakout" "space_invaders")

# Models to evaluate (add more as needed)
# Format: "provider|model_name"
MODELS=(
    "bedrock|claude-4-sonnet"
    # Add more models here:
    # "openrouter|anthropic/claude-sonnet-4"
    "openrouter|google/gemini-2.5-pro-preview"
    # "bedrock|claude-3-7-sonnet"
)

SEED=42

total_runs=$((${#GAMES[@]} * ${#MODELS[@]}))
current_run=0

echo "📊 Will run $total_runs evaluations"
echo ""

for model_spec in "${MODELS[@]}"; do
    IFS='|' read -r provider model <<< "$model_spec"

    for game in "${GAMES[@]}"; do
        current_run=$((current_run + 1))

        echo ""
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "[$current_run/$total_runs] Evaluating: $provider/$model on $game"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo ""

        # Run evaluation
        ./eval_detection_only.sh "$provider" "$model" "$game" "$SEED" "$NUM_FRAMES" "$MAX_EVAL"

        if [ $? -eq 0 ]; then
            echo "✅ Completed: $provider/$model on $game"
        else
            echo "❌ Failed: $provider/$model on $game"
        fi

        # Brief pause between runs
        sleep 2
    done
done

echo ""
echo "========================================================================="
echo "✅ ALL EVALUATIONS COMPLETE"
echo "========================================================================="
echo ""
echo "Results saved to: evaluation_results/"
echo ""
echo "Next steps:"
echo "  1. Run analysis: python create_evaluation_graphs.py"
echo "  2. Check graphs: evaluation_results/analysis/"
echo "  3. Check tables: evaluation_results/analysis/*.tex"
echo "  4. Read summary: evaluation_results/analysis/EVALUATION_SUMMARY.md"
echo "========================================================================="
echo ""
