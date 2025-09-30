"""
Test Enhanced Logging System
This script demonstrates the detailed match logging capabilities with a quick test.
"""

import tempfile
import os
from detailed_match_logger import DetailedMatchLogger
from automated_scoring_rubric import AutomatedScoringRubric, ScoringResult

def test_detailed_logging():
    """Test the detailed logging system with sample data."""

    # Create temporary output directory
    with tempfile.TemporaryDirectory(prefix="test_logging_") as temp_dir:
        print(f"🧪 Testing Detailed Logging System")
        print(f"📂 Output directory: {temp_dir}")
        print("=" * 60)

        # Initialize logger
        logger = DetailedMatchLogger(temp_dir)

        # Initialize scorer with logging enabled
        scorer = AutomatedScoringRubric(enable_detailed_logging=True, log_output_dir=temp_dir)

        # Test data - simulating actual VLM responses
        test_cases = [
            {
                'response': "I can see a white rectangular paddle on the left side at position (20, 100) and a small round white ball near the center at coordinates (80, 120). The ball appears to be moving towards the right paddle.",
                'task_type': 'visual',
                'ground_truth': {
                    'frame_id': 'pong_frame_001',
                    'pipeline_type': 'vision_only',
                    'objects': [
                        {'label': 'paddle', 'x': 20, 'y': 100, 'category': 'player'},
                        {'label': 'ball', 'x': 80, 'y': 120, 'category': 'ball'}
                    ]
                },
                'game_name': 'pong'
            },
            {
                'response': "The ball is positioned to the left of the right paddle, approximately 30 pixels away horizontally. It's located in the upper portion of the screen, above the center line.",
                'task_type': 'spatial',
                'ground_truth': {
                    'frame_id': 'pong_frame_002',
                    'pipeline_type': 'vision_symbol',
                    'objects': [
                        {'label': 'ball', 'x': 120, 'y': 80, 'category': 'ball'},
                        {'label': 'paddle', 'x': 150, 'y': 95, 'category': 'player'}
                    ]
                },
                'game_name': 'pong'
            },
            {
                'response': "This is the classic game Pong, a table tennis simulation.",
                'task_type': 'identification',
                'ground_truth': {
                    'frame_id': 'pong_frame_003',
                    'pipeline_type': 'vision_only'
                },
                'game_name': 'pong'
            },
            {
                'response': "The optimal strategy is to move the paddle vertically to intercept the ball's trajectory. Based on the ball's current angle and speed, I should position the paddle slightly above its current position to make contact.",
                'task_type': 'strategy',
                'ground_truth': {
                    'frame_id': 'pong_frame_004',
                    'pipeline_type': 'vision_symbol'
                },
                'game_name': 'pong'
            }
        ]

        print("📋 Processing test cases...")

        # Process each test case
        for i, case in enumerate(test_cases, 1):
            print(f"  {i}. Testing {case['task_type']} task ({case['ground_truth']['pipeline_type']})")

            # Score the response (this will automatically log details)
            result = scorer.score_response(
                response=case['response'],
                task_type=case['task_type'],
                ground_truth=case['ground_truth'],
                game_name=case['game_name']
            )

            print(f"     Score: {result.score:.2f}, Confidence: {result.confidence:.2f}")

        # Test trajectory prediction logging
        print("\n🎯 Testing trajectory prediction logging...")

        trajectory_response = "The ball will move to (90, 110) in the next frame, then (100, 100), then (110, 90)."
        predicted_positions = [(90, 110), (100, 100), (110, 90)]
        actual_positions = [(92, 108), (102, 98), (112, 88)]

        log_id = logger.log_trajectory_prediction(
            response=trajectory_response,
            predicted_positions=predicted_positions,
            actual_positions=actual_positions,
            frame_id="pong_trajectory_001",
            pipeline_type="vision_symbol"
        )

        print(f"     Trajectory logged with ID: {log_id}")

        # Test pipeline comparison
        print("\n📊 Testing pipeline comparison logging...")

        baseline_results = {
            'overall_statistics': {
                'mean_score': 0.45,
                'category_breakdown': {
                    'visual': {'mean_score': 0.6, 'std_score': 0.2},
                    'spatial': {'mean_score': 0.3, 'std_score': 0.15},
                    'strategy': {'mean_score': 0.4, 'std_score': 0.1},
                    'identification': {'mean_score': 0.5, 'std_score': 0.0}
                }
            }
        }

        comparison_results = {
            'overall_statistics': {
                'mean_score': 0.75,
                'category_breakdown': {
                    'visual': {'mean_score': 0.9, 'std_score': 0.1},
                    'spatial': {'mean_score': 0.8, 'std_score': 0.12},
                    'strategy': {'mean_score': 0.6, 'std_score': 0.08},
                    'identification': {'mean_score': 1.0, 'std_score': 0.0}
                }
            }
        }

        comparison_id = logger.log_pipeline_comparison(
            baseline_results=baseline_results,
            comparison_results=comparison_results,
            baseline_name="Vision-only",
            comparison_name="Vision+Symbol"
        )

        print(f"     Comparison logged with ID: {comparison_id}")

        # Generate summary report
        print("\n📑 Generating summary report...")
        report_file = logger.generate_summary_report()
        print(f"     Summary report: {report_file}")

        # Show what files were created
        print("\n📁 Files created:")
        log_files = [f for f in os.listdir(logger.log_dir) if f.endswith('.json') or f.endswith('.txt')]
        for file in sorted(log_files):
            file_path = os.path.join(logger.log_dir, file)
            size = os.path.getsize(file_path)
            print(f"     {file} ({size} bytes)")

        print(f"\n✨ Test completed! {len(log_files)} log files generated.")
        print(f"📂 All files saved in: {logger.log_dir}")

        # Show a sample of the summary report
        print("\n📖 Summary Report Preview:")
        with open(report_file, 'r') as f:
            lines = f.readlines()[:20]  # First 20 lines
            for line in lines:
                print(f"     {line.rstrip()}")
            if len(f.readlines()) > 20:
                print("     ... (truncated)")

if __name__ == "__main__":
    test_detailed_logging()