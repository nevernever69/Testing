"""
Spatial Reasoning Benchmark Demo Runner

This script demonstrates how to run the complete spatial reasoning benchmark
system with mock pipelines to verify everything works correctly.

Usage:
    python run_spatial_benchmark_demo.py
"""

import numpy as np
from typing import Tuple

# Import our benchmark components
from intelligent_frame_selector import IntelligentFrameSelector
from atari_gpt_diagnostic import AtariGPTDiagnostic, create_mock_pipeline_func
from trajectory_prediction_benchmark import TrajectoryPredictionBenchmark, create_mock_trajectory_pipeline
from automated_scoring_rubric import AutomatedScoringRubric


def create_vision_symbol_pipeline():
    """
    Mock Vision+Symbol pipeline that simulates better spatial reasoning.
    Replace this with your actual pipeline.
    """
    def pipeline_func(frame: np.ndarray, prompt: str) -> str:
        """Enhanced pipeline with symbolic grounding."""
        prompt_lower = prompt.lower()

        if "identify all key elements" in prompt_lower:
            return "I can see a player paddle at coordinates (140, 120), a ball at position (75, 100), and an opponent paddle at (20, 115). The ball appears to be moving diagonally."

        elif "where are the key elements located" in prompt_lower:
            return "The player paddle is positioned on the right side at x=140, y=120. The ball is located in the center-left area at coordinates (75, 100). The opponent paddle is on the far left at x=20, y=115. The ball is approximately 65 pixels from the player paddle."

        elif "ideal next move" in prompt_lower:
            return "Move the paddle upward by approximately 20 pixels to intercept the ball's trajectory at coordinates (140, 100). This will allow for an optimal return shot toward the opponent's weak side."

        elif "identify the game" in prompt_lower:
            return "This is Pong, the classic paddle-based game."

        elif "predict" in prompt_lower and "trajectory" in prompt_lower:
            if "next frame" in prompt_lower:
                return "The ball will be at position (77, 98)"
            elif "next 3 frames" in prompt_lower:
                return "Ball trajectory: (77, 98), (79, 96), (81, 94)"
            elif "next 5 frames" in prompt_lower:
                return "Ball positions: (77, 98), (79, 96), (81, 94), (83, 92), (85, 90)"

        return "Processing request with symbolic spatial grounding enabled."

    return pipeline_func


def create_vision_only_pipeline():
    """
    Mock Vision-only pipeline that simulates poor spatial reasoning.
    This represents the baseline that struggles with spatial tasks.
    """
    def pipeline_func(frame: np.ndarray, prompt: str) -> str:
        """Basic vision-only pipeline."""
        prompt_lower = prompt.lower()

        if "identify all key elements" in prompt_lower:
            return "I can see some game objects including what appears to be paddles and a moving element."

        elif "where are the key elements located" in prompt_lower:
            return "The game elements are positioned in various locations across the screen. There are objects on the left and right sides."

        elif "ideal next move" in prompt_lower:
            return "Move the paddle in a strategic direction to intercept the game object."

        elif "identify the game" in prompt_lower:
            return "This appears to be a classic arcade game."

        elif "predict" in prompt_lower and "trajectory" in prompt_lower:
            return "The object will continue moving in its current direction."

        return "Unable to provide detailed spatial analysis."

    return pipeline_func


def run_demo():
    """Run complete spatial reasoning benchmark demo."""
    print("🎮 Spatial Reasoning Benchmark System Demo")
    print("=" * 60)

    # Step 1: Select challenging frames
    print("\n1️⃣ Selecting Challenging Frames...")
    selector = IntelligentFrameSelector(['Pong'], seed=42)
    selected_frames = selector.select_challenging_frames(num_frames_per_game=5)

    print(f"   ✅ Selected {len(selected_frames)} challenging frames")
    print(f"   📊 Mean complexity score: {np.mean([f.complexity_score for f in selected_frames]):.3f}")

    # Convert to format needed for benchmarks
    test_frames = []
    for frame_complexity in selected_frames:
        if frame_complexity.frame_data:
            test_frame = {
                'frame_id': frame_complexity.frame_id,
                'game': frame_complexity.game,
                'frame': np.array(frame_complexity.frame_data['frame']),
                'objects': frame_complexity.frame_data['objects']
            }
            test_frames.append(test_frame)

    print(f"   📁 Prepared {len(test_frames)} frames for evaluation")

    # Step 2: Run Atari-GPT Diagnostic Evaluation
    print("\n2️⃣ Running Atari-GPT Diagnostic Evaluation...")

    evaluator = AtariGPTDiagnostic()

    # Create mock frame data for diagnostic evaluation
    diagnostic_frames = []
    for i, frame_data in enumerate(test_frames[:3]):  # Use first 3 frames
        diagnostic_frames.append((
            frame_data['frame'],
            frame_data['game'],
            {'objects': frame_data['objects']}
        ))

    # Test Vision-only pipeline
    vision_only_func = create_vision_only_pipeline()
    print("   🔍 Evaluating Vision-only pipeline...")
    vision_only_results = evaluator.evaluate_frame_set(diagnostic_frames, vision_only_func)

    # Test Vision+Symbol pipeline
    vision_symbol_func = create_vision_symbol_pipeline()
    print("   🧠 Evaluating Vision+Symbol pipeline...")
    vision_symbol_results = evaluator.evaluate_frame_set(diagnostic_frames, vision_symbol_func)

    # Compare results
    comparison = evaluator.compare_pipelines(vision_only_results, vision_symbol_results)

    print("\n   📈 Diagnostic Results Comparison:")
    for category, comp in comparison['category_comparison'].items():
        improvement = comp['percent_improvement']
        print(f"      {category.capitalize():15}: {comp['baseline_score']:.1%} → {comp['comparison_score']:.1%} ({improvement:+.1f}%)")

    overall = comparison['overall_comparison']
    print(f"      {'Overall':15}: {overall['baseline_score']:.1%} → {overall['comparison_score']:.1%} ({overall['percent_improvement']:+.1f}%)")

    # Step 3: Run Trajectory Prediction Benchmark
    print("\n3️⃣ Running Trajectory Prediction Benchmark...")

    traj_benchmark = TrajectoryPredictionBenchmark()

    # Test Vision+Symbol pipeline on trajectory prediction
    print("   🎯 Testing trajectory prediction capabilities...")
    traj_results = traj_benchmark.evaluate_pipeline(vision_symbol_func, test_frames[:2])  # Use subset for demo

    print(f"   📊 Trajectory Results:")
    print(f"      Mean L2 Error: {traj_results.aggregate_metrics['mean_l2_error']:.1f} pixels")
    print(f"      Success Rate: {traj_results.aggregate_metrics['success_rate']:.1%}")
    print(f"      Direction Accuracy: {traj_results.aggregate_metrics['mean_direction_accuracy']:.1%}")

    # Step 4: Export Results
    print("\n4️⃣ Exporting Results...")

    # Export diagnostic comparison
    evaluator.export_results(vision_only_results, "demo_vision_only_results.json")
    evaluator.export_results(vision_symbol_results, "demo_vision_symbol_results.json")

    import json
    with open("demo_diagnostic_comparison.json", 'w') as f:
        json.dump(comparison, f, indent=2)

    # Export trajectory results
    traj_benchmark.generate_analysis_report(traj_results, "demo_trajectory_results.json")

    # Export frame selection results
    selector.export_selected_frames("demo_selected_frames.json")

    print("   💾 Results exported to:")
    print("      - demo_vision_only_results.json")
    print("      - demo_vision_symbol_results.json")
    print("      - demo_diagnostic_comparison.json")
    print("      - demo_trajectory_results.json")
    print("      - demo_selected_frames.json")

    # Step 5: Generate Summary Report
    print("\n5️⃣ Summary Report")
    print("=" * 40)

    print("\n🎯 Key Findings:")
    print(f"   • Vision+Symbol shows {overall['percent_improvement']:+.1f}% overall improvement")
    print(f"   • Spatial reasoning improved from {comparison['category_comparison']['spatial']['baseline_score']:.1%} to {comparison['category_comparison']['spatial']['comparison_score']:.1%}")
    print(f"   • Trajectory prediction achieves {traj_results.aggregate_metrics['success_rate']:.1%} success rate")
    print(f"   • {len(selected_frames)} challenging scenarios identified for comprehensive evaluation")

    print("\n📝 Next Steps:")
    print("   1. Replace mock pipelines with your actual Vision-only and Vision+Symbol implementations")
    print("   2. Run full evaluation with all games (Pong, Breakout, SpaceInvaders)")
    print("   3. Increase frame count for more robust statistical analysis")
    print("   4. Add Privileged-Symbol (OCAtari) pipeline for upper bound comparison")

    print("\n✨ Demo completed successfully! The framework is ready for your actual pipelines.")


if __name__ == "__main__":
    run_demo()