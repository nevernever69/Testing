#!/bin/bash
# Batch evaluate multiple models on random agent gameplay
# Usage: ./batch_eval_models.sh <game> <seed> <frames>
#
# Example: ./batch_eval_models.sh pong 42 300

set -e

GAME="${1:-pong}"
SEED="${2:-42}"
FRAMES="${3:-300}"

echo "======================================================================="
echo "Batch Model Evaluation"
echo "======================================================================="
echo "Game: $GAME"
echo "Seed: $SEED"
echo "Frames: $FRAMES"
echo "======================================================================="
echo ""

# Check for API key
if [ -z "$OPENROUTER_API_KEY" ]; then
    if [ -f "OPENROUTER_API_KEY.txt" ]; then
        export OPENROUTER_API_KEY=$(cat OPENROUTER_API_KEY.txt)
        echo "✅ Loaded API key from OPENROUTER_API_KEY.txt"
    else
        echo "❌ Error: OPENROUTER_API_KEY not set!"
        exit 1
    fi
fi

# Define models to test
MODELS=(
    "bedrock|claude-4.5-sonnet"
    "openrouter|openai/gpt-5"
    # "openrouter|google/gemini-2.5-pro-preview"
)

echo "Will test ${#MODELS[@]} models:"
for model_spec in "${MODELS[@]}"; do
    IFS='|' read -r provider model <<< "$model_spec"
    echo "  - $provider / $model"
done
echo ""

# Run each model
for idx in "${!MODELS[@]}"; do
    model_spec="${MODELS[$idx]}"
    IFS='|' read -r provider model <<< "$model_spec"

    echo ""
    echo "======================================================================="
    echo "[$((idx+1))/${#MODELS[@]}] Testing: $provider / $model"
    echo "======================================================================="
    echo ""

    ./eval_random_agent.sh "$provider" "$model" "$GAME" "$SEED" "$FRAMES"

    echo ""
    echo "✅ Completed: $provider / $model"
    echo ""
done

echo ""
echo "======================================================================="
echo "✅ All Models Evaluated!"
echo "======================================================================="
echo ""
echo "To compare results, run:"
echo "  python compare_model_results.py --game $GAME --seed $SEED"
