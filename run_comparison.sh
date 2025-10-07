#!/bin/bash
# Run Vision-Only vs Vision+Symbol comparison with multiple seeds
# Usage: ./run_comparison.sh [provider] [model_id] [game] [num_frames] [seeds...]
# Examples:
#   ./run_comparison.sh
#   ./run_comparison.sh openrouter anthropic/claude-sonnet-4
#   ./run_comparison.sh bedrock claude-4-sonnet pong 600 42 123 456
#   ./run_comparison.sh openrouter "google/gemini-2.0-flash-exp:free" breakout

set -e

# Configuration (with defaults and command-line overrides)
PROVIDER="${1:-openrouter}"
MODEL_ID="${2:-anthropic/claude-sonnet-4}"
GAME_TYPE="${3:-pong}"
NUM_FRAMES="${4:-600}"

# Parse seeds (defaults to 42 123 456 if not provided)
# Shift past the first 4 arguments to get seeds
SEEDS=(42 123 456)  # Default seeds
if [ $# -gt 4 ]; then
    # User provided seeds after the first 4 arguments
    SEEDS=("${@:5}")  # Get all arguments from position 5 onwards
fi

# Set environment based on game type
case "$GAME_TYPE" in
    pong)
        GAME="Pong"
        ENV_NAME="ALE/Pong-v5"
        ;;
    breakout)
        GAME="Breakout"
        ENV_NAME="ALE/Breakout-v5"
        ;;
    space_invaders|spaceinvaders)
        GAME="SpaceInvaders"
        ENV_NAME="SpaceInvaders-v4"
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
    # Save API key to file for advance_game_runner
    echo "$OPENROUTER_API_KEY" > OPENROUTER_API_KEY.txt
    API_KEY="$OPENROUTER_API_KEY"
elif [ "$PROVIDER" == "bedrock" ]; then
    # Bedrock uses AWS credentials, no API key needed
    API_KEY=""
    AWS_REGION="${AWS_REGION:-us-east-1}"
    # Create empty file for advance_game_runner (it checks for file existence)
    touch OPENROUTER_API_KEY.txt
else
    echo "ERROR: Unsupported provider: $PROVIDER"
    echo "Supported: openrouter, bedrock"
    exit 1
fi

echo "======================================================================="
echo "GAMEPLAY COMPARISON: Vision-Only vs Vision+Symbol"
echo "======================================================================="
echo "Game: $GAME"
echo "Model: $MODEL_ID"
echo "Frames per run: $NUM_FRAMES"
echo "Seeds: ${SEEDS[@]}"
echo "======================================================================="
echo ""

# Create output directory
mkdir -p ./comparison_results

# Run Vision-Only for each seed
echo "======================================================================="
echo "PHASE 1: Running Vision-Only"
echo "======================================================================="
for seed in "${SEEDS[@]}"; do
    echo ""
    echo "--- Vision-Only Seed $seed ---"

    if [ "$PROVIDER" == "openrouter" ]; then
        python direct_frame_runner.py \
          --game "$ENV_NAME" \
          --provider "$PROVIDER" \
          --model_name "$MODEL_ID" \
          --api_key_file OPENROUTER_API_KEY.txt \
          --game_type "$GAME_TYPE" \
          --num_frames $NUM_FRAMES \
          --seed $seed \
          --output_dir "./comparison_results/vision_only_seed${seed}/"
    elif [ "$PROVIDER" == "bedrock" ]; then
        python direct_frame_runner.py \
          --game "$ENV_NAME" \
          --provider "$PROVIDER" \
          --model_name "$MODEL_ID" \
          --aws_region "$AWS_REGION" \
          --game_type "$GAME_TYPE" \
          --num_frames $NUM_FRAMES \
          --seed $seed \
          --output_dir "./comparison_results/vision_only_seed${seed}/"
    fi

    # Quick result
    result=$(tail -1 "./comparison_results/vision_only_seed${seed}/${GAME}_"*/Results/actions_rewards.csv 2>/dev/null | cut -d',' -f2 || echo "N/A")
    echo "✅ Seed $seed complete - Final Score: $result"
done

echo ""
echo "======================================================================="
echo "PHASE 2: Running Vision+Symbol"
echo "======================================================================="
for seed in "${SEEDS[@]}"; do
    echo ""
    echo "--- Vision+Symbol Seed $seed ---"

    if [ "$PROVIDER" == "openrouter" ]; then
        python advance_game_runner.py \
          --game "$ENV_NAME" \
          --provider "$PROVIDER" \
          --model_name "$MODEL_ID" \
          --openrouter_key_file OPENROUTER_API_KEY.txt \
          --detection_model "$MODEL_ID" \
          --game_type "$GAME_TYPE" \
          --num_frames $NUM_FRAMES \
          --seed $seed \
          --output_dir "./comparison_results/vision_symbol_seed${seed}/"
    elif [ "$PROVIDER" == "bedrock" ]; then
        python advance_game_runner.py \
          --game "$ENV_NAME" \
          --provider "$PROVIDER" \
          --model_name "$MODEL_ID" \
          --aws_region "$AWS_REGION" \
          --openrouter_key_file OPENROUTER_API_KEY.txt \
          --detection_model "$MODEL_ID" \
          --game_type "$GAME_TYPE" \
          --num_frames $NUM_FRAMES \
          --seed $seed \
          --output_dir "./comparison_results/vision_symbol_seed${seed}/"
    fi

    # Quick result
    result=$(tail -1 "./comparison_results/vision_symbol_seed${seed}/${GAME}_"*/Results/actions_rewards.csv 2>/dev/null | cut -d',' -f2 || echo "N/A")
    echo "✅ Seed $seed complete - Final Score: $result"
done

echo ""
echo "======================================================================="
echo "RESULTS SUMMARY"
echo "======================================================================="

echo ""
echo "Vision-Only Scores:"
for seed in "${SEEDS[@]}"; do
    result=$(tail -1 "./comparison_results/vision_only_seed${seed}/${GAME}_"*/Results/actions_rewards.csv 2>/dev/null | cut -d',' -f2 || echo "N/A")
    echo "  Seed $seed: $result"
done

echo ""
echo "Vision+Symbol Scores:"
for seed in "${SEEDS[@]}"; do
    result=$(tail -1 "./comparison_results/vision_symbol_seed${seed}/${GAME}_"*/Results/actions_rewards.csv 2>/dev/null | cut -d',' -f2 || echo "N/A")
    echo "  Seed $seed: $result"
done

echo ""
echo "======================================================================="
echo "✅ ALL RUNS COMPLETE!"
echo "======================================================================="
echo ""
echo "Results saved in: ./comparison_results/"
echo ""
echo "To analyze results, run:"
echo "  python analyze_comparison.py"
echo ""
