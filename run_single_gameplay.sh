#!/bin/bash
# Quick script to run 600 frame gameplay with metrics
# Usage: ./run_single_gameplay.sh [provider] [model_id] [game] [num_frames] [seed]
# Examples:
#   ./run_single_gameplay.sh
#   ./run_single_gameplay.sh openrouter anthropic/claude-sonnet-4
#   ./run_single_gameplay.sh bedrock claude-4-sonnet pong 600 42

set -e  # Exit on error

# Configuration (with defaults and command-line overrides)
PROVIDER="${1:-openrouter}"
MODEL_ID="${2:-anthropic/claude-sonnet-4}"
GAME_TYPE="${3:-pong}"
NUM_FRAMES="${4:-600}"
SEED="${5:-42}"

# Set environment based on game type
case "$GAME_TYPE" in
    pong)
        ENV_NAME="ALE/Pong-v5"
        ;;
    breakout)
        ENV_NAME="ALE/Breakout-v5"
        ;;
    space_invaders|spaceinvaders)
        ENV_NAME="ALE/SpaceInvaders-v5"
        GAME_TYPE="space_invaders"
        ;;
    *)
        echo "Unknown game type: $GAME_TYPE"
        echo "Supported: pong, breakout, space_invaders"
        exit 1
        ;;
esac

# Check for API key (provider-specific)
if [ "$PROVIDER" == "openrouter" ]; then
    if [ -z "$OPENROUTER_API_KEY" ]; then
        echo "ERROR: OPENROUTER_API_KEY environment variable not set!"
        echo "Please run: export OPENROUTER_API_KEY='your-key-here'"
        exit 1
    fi
    API_KEY="$OPENROUTER_API_KEY"
elif [ "$PROVIDER" == "bedrock" ]; then
    # Bedrock uses AWS credentials, no API key needed
    API_KEY=""
    if [ -z "$AWS_REGION" ]; then
        AWS_REGION="us-east-1"
    fi
else
    echo "ERROR: Unsupported provider: $PROVIDER"
    echo "Supported: openrouter, bedrock"
    exit 1
fi

echo "======================================================================="
echo "Running 600 Frame Gameplay"
echo "======================================================================="
echo "Environment: $ENV_NAME"
echo "Game Type: $GAME_TYPE"
echo "Provider: $PROVIDER"
echo "Model: $MODEL_ID"
echo "Frames: $NUM_FRAMES"
echo "Seed: $SEED"
echo "======================================================================="
echo ""

# Run Vision-Only
echo "Starting Vision-Only gameplay..."

# Save API key to file for direct_frame_runner
if [ "$PROVIDER" == "openrouter" ]; then
    echo "$API_KEY" > API_KEY.txt
fi

# Build command based on provider
if [ "$PROVIDER" == "openrouter" ]; then
    python direct_frame_runner.py \
      --game "$ENV_NAME" \
      --provider "$PROVIDER" \
      --model_name "$MODEL_ID" \
      --api_key_file API_KEY.txt \
      --game_type "$GAME_TYPE" \
      --num_frames $NUM_FRAMES \
      --seed $SEED \
      --output_dir ./experiments/
elif [ "$PROVIDER" == "bedrock" ]; then
    python direct_frame_runner.py \
      --game "$ENV_NAME" \
      --provider "$PROVIDER" \
      --model_name "$MODEL_ID" \
      --aws_region "${AWS_REGION:-us-east-1}" \
      --game_type "$GAME_TYPE" \
      --num_frames $NUM_FRAMES \
      --seed $SEED \
      --output_dir ./experiments/
fi

echo ""
echo "======================================================================="
echo "✅ COMPLETE!"
echo "======================================================================="
echo ""
echo "Results saved in: experiments/Pong_anthropic_claude-sonnet-4_direct_frame/"
echo ""
echo "Check metrics:"
echo "  cat experiments/Pong_*/Results/actions_rewards.csv | tail -5"
echo ""
echo "Final score:"
echo "  tail -1 experiments/Pong_*/Results/actions_rewards.csv | cut -d',' -f2"
echo ""
