#!/bin/bash
# Run Vision-Only vs Vision+Symbol comparison with proper resumption
# Usage: ./run_comparison_v2.sh <provider> <model> [game] [num_frames] <seed1> <seed2> <seed3>...
#
# IMPORTANT: Seeds are MANDATORY! You must specify at least one seed.
#
# Examples:
#   ./run_comparison_v2.sh bedrock claude-4.5-sonnet pong 600 42 123 456
#   ./run_comparison_v2.sh openrouter "anthropic/claude-sonnet-4" breakout 600 42 123
#   ./run_comparison_v2.sh bedrock claude-4.5-sonnet pong 600 42

set -e  # Exit on error (will stop if API fails)

# Check minimum arguments
if [ $# -lt 3 ]; then
    echo "ERROR: Not enough arguments!"
    echo ""
    echo "Usage: $0 <provider> <model> [game] [num_frames] <seed1> [seed2] [seed3]..."
    echo ""
    echo "Examples:"
    echo "  $0 bedrock claude-4.5-sonnet pong 600 42 123 456"
    echo "  $0 openrouter \"anthropic/claude-sonnet-4\" breakout 600 42 123"
    echo "  $0 bedrock claude-4.5-sonnet pong 600 42"
    echo ""
    echo "IMPORTANT: At least one seed is REQUIRED!"
    exit 1
fi

# Required arguments
PROVIDER="$1"
MODEL_ID="$2"

# Parse optional game and frames, then seeds
shift 2

# Default game and frames if not a number (seed)
GAME_TYPE="pong"
NUM_FRAMES=600

# Check if next arg is a valid game name
if [[ "$1" =~ ^(pong|breakout|space_invaders|spaceinvaders)$ ]]; then
    GAME_TYPE="$1"
    shift
fi

# Check if next arg is a number (num_frames)
if [[ "$1" =~ ^[0-9]+$ ]]; then
    NUM_FRAMES="$1"
    shift
fi

# Remaining arguments are seeds (REQUIRED)
if [ $# -eq 0 ]; then
    echo "ERROR: No seeds specified!"
    echo "You must specify at least one seed."
    echo ""
    echo "Example: $0 bedrock claude-4.5-sonnet pong 600 42 123 456"
    exit 1
fi

SEEDS=("$@")

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
        echo "ERROR: Unknown game type: $GAME_TYPE"
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
    # Save API key to file
    echo "$OPENROUTER_API_KEY" > OPENROUTER_API_KEY.txt
    API_KEY="$OPENROUTER_API_KEY"
elif [ "$PROVIDER" == "bedrock" ]; then
    # Bedrock uses AWS credentials
    API_KEY=""
    AWS_REGION="${AWS_REGION:-us-east-1}"
    # Create empty file (some scripts check for existence)
    touch OPENROUTER_API_KEY.txt
else
    echo "ERROR: Unsupported provider: $PROVIDER"
    echo "Supported: openrouter, bedrock"
    exit 1
fi

echo "======================================================================="
echo "GAMEPLAY COMPARISON: Vision-Only vs Vision+Symbol"
echo "======================================================================="
echo "Provider: $PROVIDER"
echo "Model: $MODEL_ID"
echo "Game: $GAME ($GAME_TYPE)"
echo "Frames per run: $NUM_FRAMES"
echo "Seeds: ${SEEDS[@]}"
echo "======================================================================="
echo ""

# Create output directory
mkdir -p ./comparison_results

# Create state tracking directory
STATE_DIR="./comparison_results/.state"
mkdir -p "$STATE_DIR"

# Function to check if a run is complete
is_complete() {
    local pipeline=$1
    local seed=$2
    local marker="$STATE_DIR/${pipeline}_seed${seed}.done"
    [ -f "$marker" ]
}

# Function to mark a run as complete
mark_complete() {
    local pipeline=$1
    local seed=$2
    local marker="$STATE_DIR/${pipeline}_seed${seed}.done"
    touch "$marker"
    echo "$(date): Completed $pipeline seed $seed" >> "$STATE_DIR/progress.log"
}

# Function to run Vision-Only
run_vision_only() {
    local seed=$1
    local results_csv="./comparison_results/vision_only_seed${seed}/${GAME}_"*"/Results/actions_rewards.csv"

    if is_complete "vision_only" $seed && [ -f $results_csv ] 2>/dev/null; then
        echo "⏭️  Vision-Only Seed $seed already complete (skipping)"
        return 0
    fi

    # Remove incomplete markers and state files
    rm -f "$STATE_DIR/vision_only_seed${seed}.done"
    find "./comparison_results/vision_only_seed${seed}" -name "*_state.pkl" -delete 2>/dev/null || true

    echo ""
    echo "======================================================================="
    echo "Running: Vision-Only - Seed $seed"
    echo "======================================================================="

    if [ "$PROVIDER" == "openrouter" ]; then
        python direct_frame_runner.py \
          --game "$ENV_NAME" \
          --provider "$PROVIDER" \
          --model_name "$MODEL_ID" \
          --api_key_file OPENROUTER_API_KEY.txt \
          --game_type "$GAME_TYPE" \
          --num_frames "$NUM_FRAMES" \
          --seed "$seed" \
          --output_dir "./comparison_results/vision_only_seed${seed}/"
    elif [ "$PROVIDER" == "bedrock" ]; then
        python direct_frame_runner.py \
          --game "$ENV_NAME" \
          --provider "$PROVIDER" \
          --model_name "$MODEL_ID" \
          --aws_region "$AWS_REGION" \
          --game_type "$GAME_TYPE" \
          --num_frames "$NUM_FRAMES" \
          --seed "$seed" \
          --output_dir "./comparison_results/vision_only_seed${seed}/"
    fi

    # Check if run actually completed
    if [ -f $results_csv ] 2>/dev/null; then
        mark_complete "vision_only" $seed
        result=$(tail -1 $results_csv 2>/dev/null | cut -d',' -f2 || echo "N/A")
        echo "✅ Vision-Only Seed $seed complete - Final Score: $result"
    else
        echo "⚠️  Vision-Only Seed $seed did not complete"
        return 1
    fi
}

# Function to run Vision+Symbol
run_vision_symbol() {
    local seed=$1
    local results_csv="./comparison_results/vision_symbol_seed${seed}/${GAME}_"*"/Results/actions_rewards.csv"

    if is_complete "vision_symbol" $seed && [ -f $results_csv ] 2>/dev/null; then
        echo "⏭️  Vision+Symbol Seed $seed already complete (skipping)"
        return 0
    fi

    # Remove incomplete marker if exists
    rm -f "$STATE_DIR/vision_symbol_seed${seed}.done"

    echo ""
    echo "======================================================================="
    echo "Running: Vision+Symbol - Seed $seed"
    echo "======================================================================="

    if [ "$PROVIDER" == "openrouter" ]; then
        python advance_game_runner.py \
          --game "$ENV_NAME" \
          --provider "$PROVIDER" \
          --model_name "$MODEL_ID" \
          --openrouter_key_file OPENROUTER_API_KEY.txt \
          --detection_model "$MODEL_ID" \
          --game_type "$GAME_TYPE" \
          --num_frames "$NUM_FRAMES" \
          --seed "$seed" \
          --disable_history \
          --resume \
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
          --num_frames "$NUM_FRAMES" \
          --seed "$seed" \
          --disable_history \
          --resume \
          --output_dir "./comparison_results/vision_symbol_seed${seed}/"
    fi

    # Check if run actually completed
    if [ -f $results_csv ] 2>/dev/null; then
        mark_complete "vision_symbol" $seed
        result=$(tail -1 $results_csv 2>/dev/null | cut -d',' -f2 || echo "N/A")
        echo "✅ Vision+Symbol Seed $seed complete - Final Score: $result"
    else
        echo "⚠️  Vision+Symbol Seed $seed did not complete"
        return 1
    fi
}

# Main loop: Process each seed completely (both pipelines) before moving to next
for seed in "${SEEDS[@]}"; do
    echo ""
    echo "======================================================================="
    echo "PROCESSING SEED: $seed"
    echo "======================================================================="

    # Run Vision-Only for this seed
    run_vision_only $seed

    # Run Vision+Symbol for this seed
    run_vision_symbol $seed

    echo ""
    echo "✅ Seed $seed COMPLETE (both pipelines finished)"
    echo ""
done

echo ""
echo "======================================================================="
echo "ALL SEEDS COMPLETE!"
echo "======================================================================="
echo ""

# Show summary
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
echo "✅ COMPARISON COMPLETE!"
echo "======================================================================="
echo ""
echo "Results saved in: ./comparison_results/"
echo ""
echo "To analyze results, run:"
echo "  python analyze_comparison.py"
echo ""
echo "Progress log: $STATE_DIR/progress.log"
echo ""
