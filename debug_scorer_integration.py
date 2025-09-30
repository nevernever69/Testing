"""
Debug the scorer integration issue
"""

import os
from atari_gpt_diagnostic import AtariGPTDiagnostic
from automated_scoring_rubric import AutomatedScoringRubric

def test_scorer_integration():
    print("🧪 Testing scorer integration...")

    # Set up test directory
    test_dir = "./debug_integration"
    os.makedirs(test_dir, exist_ok=True)

    # Initialize diagnostic evaluator with detailed logging
    evaluator = AtariGPTDiagnostic(enable_detailed_logging=True, log_output_dir=test_dir)

    print(f"Evaluator initialized: {evaluator.enable_detailed_logging}")
    print(f"Scorer available: {evaluator.scorer is not None}")
    print(f"Comparator available: {evaluator.ground_truth_comparator is not None}")

    # Test data that matches what we see in the benchmark
    test_response = """This appears to be a retro-style video game screenshot with pixelated graphics. Key elements include:

- Score display showing "20" in orange/brown pixelated text in the upper left
- Green rectangular object in the upper right corner
- Orange/brown rectangular player character or object on the left side
- White projectile or bullet in the upper right area
- Small green rectangular object in the bottom right corner"""

    # Ground truth in OCAtari format (what we actually get)
    ocatari_ground_truth = {
        'objects': [
            {'id': 0, 'category': 'Player', 'position': (140, 178), 'velocity': (0.0, 0.0), 'size': (4, 15)},
            {'id': 1, 'category': 'Ball', 'position': (140, 175), 'velocity': (0.0, 0.0), 'size': (2, 4)},
            {'id': 2, 'category': 'Enemy', 'position': (16, 175), 'velocity': (0.0, 0.0), 'size': (4, 15)}
        ]
    }

    # Test direct scorer call
    print("\n1️⃣ Testing direct AutomatedScoringRubric...")
    direct_scorer = AutomatedScoringRubric(enable_detailed_logging=True, log_output_dir=test_dir)

    # Convert to expected format manually
    converted_gt = {
        'frame_id': 'test_001',
        'pipeline_type': 'vision_only',
        'objects': [
            {'label': 'player', 'coordinates': [140, 178, 144, 193], 'description': 'Player at (140, 178)'},
            {'label': 'ball', 'coordinates': [140, 175, 142, 179], 'description': 'Ball at (140, 175)'},
            {'label': 'enemy', 'coordinates': [16, 175, 20, 190], 'description': 'Enemy at (16, 175)'}
        ]
    }

    direct_result = direct_scorer.score_response(test_response, 'visual', converted_gt, 'pong')
    print(f"Direct scorer result: {direct_result.score}")

    # Test the diagnostic evaluator's scorer (simulate what happens in benchmark)
    print("\n2️⃣ Testing diagnostic evaluator scorer...")

    # Simulate what happens in the evaluate_single_frame method
    if evaluator.enable_detailed_logging and evaluator.scorer:
        print("✅ Enhanced scoring path should be used")

        # Simulate the enhanced_ground_truth construction
        enhanced_ground_truth = ocatari_ground_truth.copy()

        # Apply the conversion logic from our fix
        if 'objects' in enhanced_ground_truth and enhanced_ground_truth['objects']:
            converted_objects = []
            for obj in enhanced_ground_truth['objects']:
                if isinstance(obj, dict) and 'category' in obj and 'position' in obj:
                    converted_obj = {
                        'label': obj['category'].lower(),
                        'coordinates': [obj['position'][0], obj['position'][1],
                                      obj['position'][0] + obj.get('size', [4, 4])[0],
                                      obj['position'][1] + obj.get('size', [4, 4])[1]],
                        'description': f"{obj['category']} at {obj['position']}"
                    }
                    converted_objects.append(converted_obj)
            enhanced_ground_truth['objects'] = converted_objects

        enhanced_ground_truth.update({
            'frame_id': 'pong_test',
            'pipeline_type': 'vision_only'
        })

        print(f"Enhanced ground truth objects: {len(enhanced_ground_truth.get('objects', []))}")
        for obj in enhanced_ground_truth.get('objects', []):
            print(f"  - {obj['label']}: {obj['coordinates']}")

        try:
            evaluator_result = evaluator.scorer.score_response(
                response=test_response,
                task_type='visual',
                ground_truth=enhanced_ground_truth,
                game_name='pong'
            )
            print(f"Evaluator scorer result: {evaluator_result.score}")
            print(f"Components: {evaluator_result.components}")
            print(f"Issues: {evaluator_result.issues}")
        except Exception as e:
            print(f"❌ Evaluator scorer error: {e}")
            import traceback
            traceback.print_exc()

    # Check if files were created
    print(f"\n3️⃣ Checking log files...")
    log_dir = os.path.join(test_dir, "detailed_logs")
    if os.path.exists(log_dir):
        files = os.listdir(log_dir)
        print(f"Files in {log_dir}: {files}")
        for f in files:
            print(f"  📄 {f}")
    else:
        print(f"❌ Log directory {log_dir} does not exist")

if __name__ == "__main__":
    test_scorer_integration()