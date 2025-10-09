#!/bin/bash
# V3 Wrapper: Save Everything (all frames, ground truth viz, VLM viz, comparisons)
# Usage: ./eval_save_everything.sh <provider> <model> [game] [seed] [frames]
#
# Examples:
#   ./eval_save_everything.sh openrouter "anthropic/claude-sonnet-4"
#   ./eval_save_everything.sh bedrock claude-4.5-sonnet breakout 123 300

set -e

# Parse arguments with defaults
PROVIDER="${1:-openrouter}"
MODEL="${2:-anthropic/claude-sonnet-4}"
GAME="${3:-pong}"
SEED="${4:-42}"
NUM_FRAMES="${5:-300}"

echo "======================================================================="
echo "V3: Save Everything - Comprehensive Evaluation"
echo "======================================================================="
echo "Provider: $PROVIDER"
echo "Model: $MODEL"
echo "Game: $GAME"
echo "Seed: $SEED"
echo "Frames: $NUM_FRAMES"
echo ""
echo "Will save:"
echo "  ✅ All $NUM_FRAMES RGB frames"
echo "  ✅ OCAtari ground truth with bounding boxes"
echo "  ✅ VLM detections with bounding boxes"
echo "  ✅ Side-by-side comparisons"
echo "  ✅ VLM raw responses with coordinates"
echo "  ✅ Detailed frame-by-frame results"
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

# Make script executable
chmod +x run_coordinate_evaluation_v3.py 2>/dev/null || true

# Run V3 evaluation with all saving features
python run_coordinate_evaluation_v3.py \
    --game "$GAME" \
    --seed "$SEED" \
    --num_frames "$NUM_FRAMES" \
    --random_agent \
    --provider "$PROVIDER" \
    --model "$MODEL" \
    --sample_every 10 \
    --save_all_frames \
    --create_visualizations

echo ""
echo "======================================================================="
echo "✅ V3 Evaluation Complete - Everything Saved!"
echo "======================================================================="
echo "Output directory: coordinate_eval_v3_${GAME}_seed${SEED}_${PROVIDER}_*/"
echo ""
echo "What was saved:"
echo "  📁 frames/                           - All $NUM_FRAMES RGB frames"
echo "  📁 ground_truth_visualizations/      - OCAtari with bounding boxes"
echo "  📁 vlm_visualizations/               - VLM detections with boxes"
echo "  📁 side_by_side_comparisons/         - GT vs VLM comparisons"
echo "  📁 frame_*_detection/                - Per-frame VLM data"
echo "  📄 ground_truth.json                 - All ground truth data"
echo "  📄 evaluation_summary.json           - Overall metrics"
echo "  📄 detailed_frame_results.json       - Per-frame metrics"
echo "======================================================================="
