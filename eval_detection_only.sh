#!/bin/bash
# V3 Detection Only: Fast evaluation (no action reasoning)
# 2x faster, 50% cheaper - only does object detection
#
# Usage: ./eval_detection_only.sh <provider> <model> [game] [seed] [total_frames] [max_eval_frames] [start_frame]
# Examples:
#   ./eval_detection_only.sh bedrock claude-4-sonnet pong 42 300 10
#   ./eval_detection_only.sh bedrock claude-4-sonnet spaceinvaders 42 300 10  # Auto uses start_frame=45
#   ./eval_detection_only.sh bedrock claude-4-sonnet pong 42 300 10 20  # Override start_frame to 20
#   ./eval_detection_only.sh openrouter "anthropic/claude-sonnet-4" breakout 123 600 20

set -e

PROVIDER="${1:-bedrock}"
MODEL="${2:-claude-4-sonnet}"
GAME="${3:-pong}"
SEED="${4:-42}"
NUM_FRAMES="${5:-300}"
MAX_EVAL_FRAMES="${6:-}"  # Optional: limit number of frames to evaluate

# Game-specific default start frames (can be overridden by parameter 7)
if [ -z "${7}" ]; then
    if [ "$GAME" == "spaceinvaders" ] || [ "$GAME" == "space_invaders" ]; then
        START_FRAME=50  # SpaceInvaders: skip frames where player isn't controllable yet
    else
        START_FRAME=10  # Pong, Breakout, etc.
    fi
else
    START_FRAME="${7}"
fi

echo "======================================================================="
echo "V3: Detection Only - Fast Coordinate Evaluation"
echo "======================================================================="
echo "Provider: $PROVIDER"
echo "Model: $MODEL"
echo "Game: $GAME"
echo "Seed: $SEED"
echo "Total Frames: $NUM_FRAMES"
echo "Start Frame: $START_FRAME (skipping first $START_FRAME frames)"
if [ -n "$MAX_EVAL_FRAMES" ]; then
    echo "Max Eval Frames: $MAX_EVAL_FRAMES (will evaluate first $MAX_EVAL_FRAMES frames)"
fi
echo ""
echo "Mode: DETECTION ONLY (no action reasoning)"
echo "  → 2x faster (one API call per frame)"
echo "  → 50% cheaper (half the tokens)"
echo "  → Pure coordinate evaluation"
echo "======================================================================="
echo ""

# Check for API key if needed
if [ "$PROVIDER" == "openrouter" ]; then
    if [ -z "$OPENROUTER_API_KEY" ]; then
        if [ -f "OPENROUTER_API_KEY.txt" ]; then
            export OPENROUTER_API_KEY=$(cat OPENROUTER_API_KEY.txt)
            echo "✅ Loaded API key"
        else
            echo "❌ Error: OPENROUTER_API_KEY not set!"
            exit 1
        fi
    fi
fi

chmod +x run_coordinate_evaluation_v3_detection_only.py 2>/dev/null || true

# Build command with optional parameters
CMD="python run_coordinate_evaluation_v3_detection_only.py \
    --game \"$GAME\" \
    --seed \"$SEED\" \
    --num_frames \"$NUM_FRAMES\" \
    --random_agent \
    --provider \"$PROVIDER\" \
    --model \"$MODEL\" \
    --sample_every 10 \
    --start_frame \"$START_FRAME\" \
    --save_all_frames \
    --create_visualizations"

# Add max_eval_frames if specified
if [ -n "$MAX_EVAL_FRAMES" ]; then
    CMD="$CMD --max_eval_frames $MAX_EVAL_FRAMES"
fi

# Execute
eval $CMD

echo ""
echo "======================================================================="
echo "✅ Detection-Only Evaluation Complete!"
echo "======================================================================="
