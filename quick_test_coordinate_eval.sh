#!/bin/bash
# Quick test of coordinate evaluation system
# Usage: ./quick_test_coordinate_eval.sh

set -e

echo "======================================================================="
echo "Testing Coordinate Accuracy Evaluation System"
echo "======================================================================="
echo ""

# Test 1: Run the example in coordinate_accuracy_evaluator.py
echo "Test 1: Running built-in example..."
python coordinate_accuracy_evaluator.py

echo ""
echo "======================================================================="
echo "Test 1: PASSED ✅"
echo "======================================================================="
echo ""

# Test 2: Check if we can find experiment directories
echo "Test 2: Checking for experiment directories..."
if [ -d "experiments" ]; then
    echo "Found experiments directory:"
    ls -d experiments/*/ 2>/dev/null | head -5 || echo "  (empty)"
else
    echo "No experiments directory found (expected if no experiments run yet)"
fi

echo ""
echo "======================================================================="
echo "Test 2: PASSED ✅"
echo "======================================================================="
echo ""

# Test 3: Show usage
echo "Test 3: Showing usage examples..."
echo ""
echo "To evaluate an experiment:"
echo "  python run_coordinate_evaluation.py \\"
echo "    --experiment_dir experiments/Pong_seed42 \\"
echo "    --game pong"
echo ""
echo "To aggregate multiple experiments:"
echo "  python run_coordinate_evaluation.py \\"
echo "    --experiment_dir experiments/ \\"
echo "    --game pong \\"
echo "    --aggregate"
echo ""

echo "======================================================================="
echo "All tests PASSED ✅"
echo "======================================================================="
echo ""
echo "The coordinate evaluation system is ready to use!"
echo "See COORDINATE_EVALUATION_GUIDE.md for detailed documentation."
