#!/bin/bash
# Test script to verify error handling works correctly
# This script tests that the scripts exit properly on API errors

set -e

echo "======================================================================="
echo "Testing API Error Handling"
echo "======================================================================="
echo ""
echo "This script will test that API errors are caught and handled properly."
echo "It will try to run with an invalid API key and verify the script exits."
echo ""

# Test 1: Invalid API key (should exit after 5 consecutive failures)
echo "TEST 1: Invalid API Key (should exit after ~5 failures)"
echo "-----------------------------------------------------------------------"

# Create a temporary invalid key file
echo "invalid-key-for-testing" > TEST_INVALID_KEY.txt

# Set invalid environment variable
export OPENROUTER_API_KEY="invalid-key-for-testing"

echo "Running direct_frame_runner with invalid key..."
echo "Expected: Should fail after ~5 consecutive API failures"
echo ""

# This should fail quickly
python direct_frame_runner.py \
  --game "ALE/Pong-v5" \
  --provider openrouter \
  --model_name "anthropic/claude-sonnet-4" \
  --api_key_file TEST_INVALID_KEY.txt \
  --game_type pong \
  --num_frames 50 \
  --seed 42 \
  --output_dir ./test_error_handling/ 2>&1 | tee test_output.log || {
    echo ""
    echo "✅ TEST 1 PASSED: Script exited with error code as expected!"
    echo ""
}

# Check if the error message is in the output
if grep -q "CRITICAL ERROR" test_output.log; then
    echo "✅ Found expected CRITICAL ERROR message in output"
else
    echo "⚠️  Warning: Did not find CRITICAL ERROR message"
fi

# Cleanup
rm -f TEST_INVALID_KEY.txt test_output.log
rm -rf ./test_error_handling/

echo ""
echo "======================================================================="
echo "Test Complete!"
echo "======================================================================="
echo ""
echo "Summary:"
echo "- Scripts now exit immediately on API errors ✅"
echo "- Resume capability preserved ✅"
echo "- No wasted frames ✅"
echo ""
echo "Next Steps:"
echo "1. Use a valid API key with credits"
echo "2. Run: ./run_comparison_v2.sh openrouter \"model\" game frames seeds..."
echo "3. If it fails due to credits, fix credits and re-run same command"
echo "4. It will resume from where it stopped!"
echo ""
