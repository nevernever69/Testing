"""
Simple debug test to see where scores are getting lost
"""

import os
import tempfile
from PIL import Image
import numpy as np

# Create a simple test
def simple_test():
    print("🧪 Simple Debug Test")

    # Create a small test image
    img = Image.fromarray(np.zeros((210, 160, 3), dtype=np.uint8))

    # Create temp directory for test
    test_dir = "./simple_debug_test"
    os.makedirs(test_dir, exist_ok=True)

    # Save test image
    img_path = os.path.join(test_dir, "test_frame.png")
    img.save(img_path)

    print(f"Created test image: {img_path}")

    # Initialize diagnostic evaluator
    from atari_gpt_diagnostic import AtariGPTDiagnostic
    evaluator = AtariGPTDiagnostic(enable_detailed_logging=True, log_output_dir=test_dir)

    # Mock pipeline function that returns a simple response
    def mock_vision_pipeline(frame, prompt):
        return "I see a player paddle and a ball in this Pong game"

    # Add pipeline type attribute
    mock_vision_pipeline._pipeline_type = "vision_only"

    # Ground truth data (OCAtari format)
    ground_truth = {
        'objects': [
            {'id': 0, 'category': 'Player', 'position': (20, 100), 'velocity': (0.0, 0.0), 'size': (4, 15)},
            {'id': 1, 'category': 'Ball', 'position': (80, 120), 'velocity': (1.0, -1.0), 'size': (2, 4)}
        ]
    }

    print("🎯 Testing single frame evaluation...")

    # Test single frame evaluation
    results = evaluator.evaluate_single_frame(
        frame=np.array(img),
        game_name="Pong",
        pipeline_func=mock_vision_pipeline,
        ground_truth=ground_truth
    )

    print(f"\n📊 Results Summary:")
    for result in results:
        print(f"   {result.category}: {result.score}")

    # Check files created
    log_dir = os.path.join(test_dir, "detailed_logs")
    if os.path.exists(log_dir):
        files = os.listdir(log_dir)
        print(f"\n📁 Files created: {len(files)}")
        for f in files:
            print(f"   📄 {f}")

    return results

if __name__ == "__main__":
    results = simple_test()