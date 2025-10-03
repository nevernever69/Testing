#!/bin/bash
################################################################################
# Full Evaluation Pipeline for Research Paper
# Runs complete benchmark and generates all visualizations
################################################################################

set -e  # Exit on error

echo "================================================================================"
echo "  FULL EVALUATION PIPELINE"
echo "================================================================================"
echo ""

# Configuration
DATASET_DIR="./benchmark_v2.0_scenarios"
RESULTS_DIR="./benchmark_results"
FIGURES_DIR="./paper_figures"
PROVIDER="bedrock"
AWS_REGION="us-east-1"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset)
            DATASET_DIR="$2"
            shift 2
            ;;
        --results)
            RESULTS_DIR="$2"
            shift 2
            ;;
        --provider)
            PROVIDER="$2"
            shift 2
            ;;
        --openrouter_key)
            OPENROUTER_KEY="$2"
            shift 2
            ;;
        --aws_region)
            AWS_REGION="$2"
            shift 2
            ;;
        --use_llm_judge)
            USE_LLM_JUDGE="--use_llm_judge"
            shift
            ;;
        --skip_evaluation)
            SKIP_EVAL=1
            shift
            ;;
        --skip_visualization)
            SKIP_VIZ=1
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "Configuration:"
echo "  Dataset: $DATASET_DIR"
echo "  Results: $RESULTS_DIR"
echo "  Figures: $FIGURES_DIR"
echo "  Provider: $PROVIDER"
echo "  AWS Region: $AWS_REGION"
echo ""

# Check if dataset exists
if [ ! -d "$DATASET_DIR" ]; then
    echo "❌ Error: Dataset not found at $DATASET_DIR"
    echo ""
    echo "Generate dataset first with:"
    echo "  python generate_scenario_dataset.py"
    echo "  OR"
    echo "  python generate_scenario_dataset_human.py"
    exit 1
fi

# Count frames in dataset
FRAME_COUNT=$(find "$DATASET_DIR/frames" -name "*.png" | wc -l)
echo "Dataset frames: $FRAME_COUNT"
echo ""

# Step 1: Run benchmark evaluation
if [ -z "$SKIP_EVAL" ]; then
    echo "================================================================================"
    echo "  STEP 1: Running Benchmark Evaluation"
    echo "================================================================================"
    echo ""

    if [ -n "$OPENROUTER_KEY" ]; then
        python run_benchmark.py \
            --pipeline both \
            --dataset "$DATASET_DIR" \
            --output "$RESULTS_DIR" \
            --provider "$PROVIDER" \
            --aws_region "$AWS_REGION" \
            --openrouter_key "$OPENROUTER_KEY" \
            $USE_LLM_JUDGE
    else
        python run_benchmark.py \
            --pipeline both \
            --dataset "$DATASET_DIR" \
            --output "$RESULTS_DIR" \
            --provider "$PROVIDER" \
            --aws_region "$AWS_REGION" \
            $USE_LLM_JUDGE
    fi

    echo ""
    echo "✅ Evaluation complete!"
    echo ""
else
    echo "⏭  Skipping evaluation (using existing results)"
    echo ""
fi

# Step 2: Generate visualizations
if [ -z "$SKIP_VIZ" ]; then
    echo "================================================================================"
    echo "  STEP 2: Generating Paper Visualizations"
    echo "================================================================================"
    echo ""

    python generate_paper_visualizations.py \
        --results "$RESULTS_DIR" \
        --output "$FIGURES_DIR"

    echo ""
    echo "✅ Visualizations complete!"
    echo ""
else
    echo "⏭  Skipping visualization"
    echo ""
fi

# Summary
echo "================================================================================"
echo "  PIPELINE COMPLETE!"
echo "================================================================================"
echo ""
echo "Results saved to:"
echo "  📊 Benchmark results: $RESULTS_DIR/"
echo "  📈 Paper figures: $FIGURES_DIR/"
echo ""
echo "Generated files:"
echo ""
echo "Benchmark Results:"
echo "  - vision_only_results.json"
echo "  - vision_symbol_results.json"
echo "  - comparison_results.json"
echo ""
echo "Paper Figures:"
echo "  - figure1_overall_comparison.png/.pdf"
echo "  - figure2_per_task_breakdown.png/.pdf"
echo "  - figure3_per_game_breakdown.png/.pdf"
echo "  - figure4_improvement_heatmap.png/.pdf"
echo "  - figure5_score_distributions.png/.pdf"
echo "  - figure6_task_difficulty.png/.pdf"
echo "  - figure7_statistical_tests.png/.pdf"
echo "  - figure8_confusion_matrix_*.png/.pdf"
echo "  - figure9_error_analysis.png/.pdf"
echo "  - figure10_radar_comparison.png/.pdf"
echo "  - figure11_task_boxplots.png/.pdf"
echo "  - figure12_complexity_analysis.png/.pdf"
echo "  - table_results.tex (LaTeX table)"
echo "  - statistical_tests.txt (detailed statistics)"
echo ""
echo "================================================================================"
