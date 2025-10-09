#!/bin/bash
# Easy wrapper for random agent coordinate evaluation
# Usage: ./eval_random_agent.sh <provider> <model> [game] [seed] [frames]
#
# Examples:
#   ./eval_random_agent.sh openrouter "anthropic/claude-sonnet-4"
#   ./eval_random_agent.sh bedrock claude-4.5-sonnet pong 42 300
#   ./eval_random_agent.sh openrouter "google/gemini-2.0-flash-exp:free" breakout 123 300

set -e

# Parse arguments with defaults
PROVIDER="${1:-openrouter}"
MODEL="${2:-anthropic/claude-sonnet-4}"
GAME="${3:-pong}"
SEED="${4:-42}"
NUM_FRAMES="${5:-300}"

echo "======================================================================="
echo "Random Agent Coordinate Evaluation"
echo "======================================================================="
echo "Provider: $PROVIDER"
echo "Model: $MODEL"
echo "Game: $GAME"
echo "Seed: $SEED"
echo "Frames: $NUM_FRAMES"
echo "Sample: Every 10th frame"
echo "======================================================================="
echo ""

# Check for API key if needed
if [ "$PROVIDER" == "openrouter" ]; then
    if [ -z "$OPENROUTER_API_KEY" ]; then
        if [ -f "OPENROUTER_API_KEY.txt" ]; then
            export OPENROUTER_API_KEY=$(cat OPENROUTER_API_KEY.txt)
            echo "✅ Loaded API key from OPENROUTER_API_KEY.txt"
        else
            echo "❌ Error: OPENROUTER_API_KEY not set!"
            echo "Please set: export OPENROUTER_API_KEY='your-key'"
            echo "Or create: OPENROUTER_API_KEY.txt"
            exit 1
        fi
    else
        echo "✅ OPENROUTER_API_KEY found in environment"
    fi
    echo ""
fi

# Run evaluation
python run_coordinate_evaluation_v2.py \
    --game "$GAME" \
    --seed "$SEED" \
    --num_frames "$NUM_FRAMES" \
    --random_agent \
    --provider "$PROVIDER" \
    --model "$MODEL" \
    --sample_every 10

echo ""
echo "======================================================================="
echo "✅ Evaluation Complete!"
echo "======================================================================="
echo "Results saved to: coordinate_eval_${GAME}_seed${SEED}_${PROVIDER}_*/"
