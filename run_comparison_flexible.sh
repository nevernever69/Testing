#!/bin/bash
# Run Vision-Only vs Vision+Symbol comparison with flexible argument parsing
# Usage: ./run_comparison_flexible.sh <provider> <model> [game] <seed1> [seed2] [seed3]...
#
# Examples:
#   ./run_comparison_flexible.sh bedrock claude-4.5-sonnet 42
#   ./run_comparison_flexible.sh bedrock claude-4.5-sonnet pong 42
#   ./run_comparison_flexible.sh bedrock claude-4.5-sonnet 42 123 456
#   ./run_comparison_flexible.sh openrouter "google/gemini-2.5-pro" pong 42 123
#   NUM_FRAMES=20 ./run_comparison_flexible.sh openrouter "google/gemini-2.5-pro" pong 42

set -e  # Exit on error

# Usage check
if [ $# -lt 3 ]; then
    echo "ERROR: Not enough arguments!"
    echo ""
    echo "Usage: $0 <provider> <model> [game] <seed1> [seed2] [seed3]..."
    echo ""
    echo "Examples:"
    echo "  $0 bedrock claude-4.5-sonnet 42"
    echo "  $0 bedrock claude-4.5-sonnet pong 42"
    echo "  $0 bedrock claude-4.5-sonnet pong 42 123 456"
    echo "  $0 openrouter \"anthropic/claude-sonnet-4\" breakout 42 123"
    echo ""
    echo "Optional environment variables:"
    echo "  NUM_FRAMES (default: 600)"
    echo ""
    exit 1
fi

# Parse arguments
PROVIDER="$1"
MODEL_ID="$2"
shift 2

# Check if third argument is a game name or seed
KNOWN_GAMES=("pong" "breakout" "space_invaders" "spaceinvaders")
GAME_TYPE="${GAME_TYPE:-pong}"  # Default

if [ $# -gt 0 ]; then
    # Check if first remaining arg is a known game
    if [[ " ${KNOWN_GAMES[@]} " =~ " $1 " ]]; then
        GAME_TYPE="$1"
        shift
    fi
fi

# All remaining arguments are seeds
SEEDS=("$@")

# Validate we have at least one seed
if [ ${#SEEDS[@]} -eq 0 ]; then
    echo "ERROR: At least one seed is required!"
    echo ""
    echo "Examples:"
    echo "  $0 $PROVIDER $MODEL_ID 42"
    echo "  $0 $PROVIDER $MODEL_ID $GAME_TYPE 42 123"
    exit 1
fi

# Optional environment variables with defaults
NUM_FRAMES="${NUM_FRAMES:-600}"

echo "======================================================================="
echo "GAMEPLAY COMPARISON SCRIPT"
echo "======================================================================="
echo "Provider:   $PROVIDER"
echo "Model:      $MODEL_ID"
echo "Game:       $GAME_TYPE"
echo "Frames:     $NUM_FRAMES"
echo "Seeds:      ${SEEDS[@]}"
echo "======================================================================="
echo ""

# Validate seeds
for seed in "${SEEDS[@]}"; do
    if ! [[ "$seed" =~ ^[0-9]+$ ]]; then
        echo "ERROR: Invalid seed '$seed' - must be a number!"
        exit 1
    fi
done

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
        ENV_NAME="ALE/SpaceInvaders-v5"
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
    echo "$OPENROUTER_API_KEY" > OPENROUTER_API_KEY.txt
    API_KEY="$OPENROUTER_API_KEY"
elif [ "$PROVIDER" == "bedrock" ]; then
    API_KEY=""
    AWS_REGION="${AWS_REGION:-us-east-1}"
    touch OPENROUTER_API_KEY.txt
    echo "Using AWS Bedrock (Region: $AWS_REGION)"
else
    echo "ERROR: Unsupported provider: $PROVIDER"
    echo "Supported: openrouter, bedrock"
    exit 1
fi

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

    echo ""
    echo "======================================================================="
    echo "VISION-ONLY - Seed $seed"
    echo "======================================================================="

    if is_complete "vision_only" "$seed"; then
        echo "⏭️  Already complete (skipping)"
        local result=$(tail -1 "./comparison_results/vision_only_seed${seed}/${GAME}_"*/actions_rewards.csv 2>/dev/null | cut -d',' -f2 || echo "N/A")
        echo "   Final Score: $result"
        return 0
    fi

    echo "Starting Vision-Only gameplay for seed $seed..."
    echo ""

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

    # Mark as complete
    mark_complete "vision_only" "$seed"

    # Show result
    local result=$(tail -1 "./comparison_results/vision_only_seed${seed}/${GAME}_"*/actions_rewards.csv 2>/dev/null | cut -d',' -f2 || echo "N/A")
    echo ""
    echo "✅ Vision-Only Seed $seed complete - Final Score: $result"
}

# Function to run Vision+Symbol
run_vision_symbol() {
    local seed=$1

    echo ""
    echo "======================================================================="
    echo "VISION+SYMBOL - Seed $seed"
    echo "======================================================================="

    if is_complete "vision_symbol" "$seed"; then
        echo "⏭️  Already complete (skipping)"
        local result=$(tail -1 "./comparison_results/vision_symbol_seed${seed}/${GAME}_"*/actions_rewards.csv 2>/dev/null | cut -d',' -f2 || echo "N/A")
        echo "   Final Score: $result"
        return 0
    fi

    echo "Starting Vision+Symbol gameplay for seed $seed..."
    echo ""

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
          --output_dir "./comparison_results/vision_symbol_seed${seed}/"
    fi

    # Mark as complete
    mark_complete "vision_symbol" "$seed"

    # Show result
    local result=$(tail -1 "./comparison_results/vision_symbol_seed${seed}/${GAME}_"*/actions_rewards.csv 2>/dev/null | cut -d',' -f2 || echo "N/A")
    echo ""
    echo "✅ Vision+Symbol Seed $seed complete - Final Score: $result"
}

# Main loop: Process each seed completely before moving to next
for seed in "${SEEDS[@]}"; do
    echo ""
    echo "======================================================================="
    echo "PROCESSING SEED: $seed"
    echo "======================================================================="

    # Run Vision-Only for this seed
    run_vision_only "$seed"

    # Run Vision+Symbol for this seed
    run_vision_symbol "$seed"

    echo ""
    echo "✅✅ SEED $seed COMPLETE (both pipelines finished) ✅✅"
    echo ""
done

# Final summary
echo ""
echo "======================================================================="
echo "ALL SEEDS COMPLETE!"
echo "======================================================================="
echo ""

echo "RESULTS SUMMARY"
echo "======================================================================="
echo ""
echo "Vision-Only Scores:"
for seed in "${SEEDS[@]}"; do
    result=$(tail -1 "./comparison_results/vision_only_seed${seed}/${GAME}_"*/actions_rewards.csv 2>/dev/null | cut -d',' -f2 || echo "N/A")
    echo "  Seed $seed: $result"
done

echo ""
echo "Vision+Symbol Scores:"
for seed in "${SEEDS[@]}"; do
    result=$(tail -1 "./comparison_results/vision_symbol_seed${seed}/${GAME}_"*/actions_rewards.csv 2>/dev/null | cut -d',' -f2 || echo "N/A")
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
