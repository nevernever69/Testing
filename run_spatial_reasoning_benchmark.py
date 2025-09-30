"""
Spatial Reasoning Benchmark Integration

This script integrates the spatial reasoning benchmark with your existing VLM pipeline
system (advance_game_runner.py and advanced_zero_shot_pipeline.py).

Usage:
    python run_spatial_reasoning_benchmark.py --provider bedrock --model claude-4-sonnet --games Pong Breakout
"""

import os
import json
import argparse
import numpy as np
import tempfile
import base64
from PIL import Image
from io import BytesIO
from typing import Dict, List, Any, Optional

# Import your existing VLM infrastructure
from advanced_zero_shot_pipeline import AdvancedSymbolicDetector

# Import our benchmark components
from intelligent_frame_selector import IntelligentFrameSelector
from atari_gpt_diagnostic import AtariGPTDiagnostic
from trajectory_prediction_benchmark import TrajectoryPredictionBenchmark
from ocatari_ground_truth import OCAtariGroundTruth


class SpatialReasoningBenchmark:
    """
    Integrates spatial reasoning evaluation with your existing VLM pipeline system.

    Supports both Vision-only and Vision+Symbol evaluation modes using your
    actual VLM infrastructure.
    """

    def __init__(self,
                 provider: str = "bedrock",
                 model_name: str = "anthropic/claude-sonnet-4",
                 api_key: str = None,
                 aws_region: str = "us-east-1"):
        """
        Initialize spatial reasoning benchmark with VLM integration.

        Args:
            provider: VLM provider ('bedrock', 'openrouter', etc.)
            model_name: Model name to use
            api_key: API key (if required)
            aws_region: AWS region for Bedrock
        """
        self.provider = provider
        self.model_name = model_name
        self.api_key = api_key or os.getenv('OPENROUTER_API_KEY', 'dummy_key')
        self.aws_region = aws_region

        # Initialize benchmark components with detailed logging
        self.frame_selector = IntelligentFrameSelector()
        self.diagnostic_evaluator = None  # Will be initialized with proper logging

        # Enable detailed logging for all benchmark components
        self.enable_detailed_logging = True
        self.log_output_dir = None  # Will be set when run_benchmark is called
        self.trajectory_benchmark = None  # Will be initialized with proper logging

        # Work directly in output directory - no temp files
        self.temp_dir = None  # Will be set to output directory

        # Add comprehensive logging
        self.save_all_data = True  # Always save everything for inspection

    def create_vision_only_pipeline(self):
        """
        Create Vision-only pipeline that bypasses symbolic detection.

        This simulates the Atari-GPT baseline where the VLM only sees raw frames
        without any symbolic grounding.
        """
        # Initialize detector but we'll bypass symbolic detection
        detector = AdvancedSymbolicDetector(
            self.api_key,
            self.model_name,
            detection_mode="generic",
            provider=self.provider,
            aws_region=self.aws_region,
            disable_history=True  # Disable history for clean evaluation
        )
        detector.prompts_dir = os.path.join(self.temp_dir, "vision_only_prompts")
        os.makedirs(detector.prompts_dir, exist_ok=True)

        def vision_only_func(frame: np.ndarray, prompt: str) -> str:
            """Vision-only pipeline function."""
            # Generate unique ID for this call
            import time
            call_id = f"vision_only_{int(time.time()*1000)}"

            # Save frame as image with descriptive name
            frame_path = os.path.join(self.temp_dir, f"{call_id}_frame.png")
            frame_image = Image.fromarray(frame.astype('uint8'))
            frame_image.save(frame_path)

            # Scale image to 1280x720 for fair comparison (like your existing pipeline)
            scaled_frame_path = os.path.join(self.temp_dir, f"{call_id}_frame_scaled.png")
            detector.scale_image(frame_path, scaled_frame_path)

            # Convert to base64 using scaled image
            base64_image = detector.encode_image_to_base64(scaled_frame_path)

            # Create messages for VLM call
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ]

            # Save prompt for inspection
            prompt_file = os.path.join(self.temp_dir, f"{call_id}_prompt.txt")
            with open(prompt_file, 'w') as f:
                f.write(f"VISION-ONLY PIPELINE\n")
                f.write(f"Call ID: {call_id}\n")
                f.write(f"Original frame: {frame_path}\n")
                f.write(f"Scaled frame (1280x720): {scaled_frame_path}\n")
                f.write(f"Prompt: {prompt}\n")

            # Make VLM API call with error handling
            try:
                response = detector._make_api_call(messages, max_tokens=500, call_id=f"vision_only_{call_id}")
            except Exception as api_error:
                response = None
                # Save API error
                error_file = os.path.join(self.temp_dir, f"{call_id}_api_error.txt")
                with open(error_file, 'w') as f:
                    f.write(f"VISION-ONLY API ERROR\n")
                    f.write(f"Call ID: {call_id}\n")
                    f.write(f"Error: {str(api_error)}\n")

            # Save response for inspection
            response_file = os.path.join(self.temp_dir, f"{call_id}_response.txt")
            with open(response_file, 'w') as f:
                f.write(f"VISION-ONLY RESPONSE\n")
                f.write(f"Call ID: {call_id}\n")
                f.write(f"Response: {response or 'No response received'}\n")

            return response or "Unable to analyze the image."

        # Add pipeline type attribution
        vision_only_func._pipeline_type = "vision_only"
        return vision_only_func

    def create_vision_symbol_pipeline(self, game_name: str):
        """
        Create Vision+Symbol pipeline using your existing symbolic detection.

        This uses your full pipeline with symbolic grounding.
        """
        # Initialize appropriate detector for the game
        detector_class_map = {
            'Pong': 'PongAdvancedDetector',
            'Breakout': 'BreakoutAdvancedDetector',
            'SpaceInvaders': 'SpaceInvadersAdvancedDetector'
        }

        # Import and initialize the appropriate detector
        from advanced_zero_shot_pipeline import (
            PongAdvancedDetector, BreakoutAdvancedDetector, SpaceInvadersAdvancedDetector
        )

        detector_classes = {
            'PongAdvancedDetector': PongAdvancedDetector,
            'BreakoutAdvancedDetector': BreakoutAdvancedDetector,
            'SpaceInvadersAdvancedDetector': SpaceInvadersAdvancedDetector
        }

        detector_class_name = detector_class_map.get(game_name, 'PongAdvancedDetector')
        DetectorClass = detector_classes[detector_class_name]

        detector = DetectorClass(
            self.api_key,
            self.model_name,
            detection_mode="specific",
            provider=self.provider,
            aws_region=self.aws_region,
            disable_history=True
        )
        detector.prompts_dir = os.path.join(self.temp_dir, "vision_symbol_prompts")
        os.makedirs(detector.prompts_dir, exist_ok=True)

        def vision_symbol_func(frame: np.ndarray, prompt: str) -> str:
            """Vision+Symbol pipeline function."""
            # Generate unique ID for this call
            import time
            call_id = f"vision_symbol_{int(time.time()*1000)}"

            # Save frame as image with descriptive name
            frame_path = os.path.join(self.temp_dir, f"{call_id}_frame.png")
            frame_image = Image.fromarray(frame.astype('uint8'))
            frame_image.save(frame_path)

            # Scale image to 1280x720 for fair comparison (like your existing pipeline)
            scaled_frame_path = os.path.join(self.temp_dir, f"{call_id}_frame_scaled.png")
            detector.scale_image(frame_path, scaled_frame_path)

            try:
                # First get symbolic detection - need to pass game_name (use scaled image)
                symbols = detector.detect_objects(scaled_frame_path, game_name.lower())

                # Save symbolic detection results
                symbols_file = os.path.join(self.temp_dir, f"{call_id}_symbols.json")
                with open(symbols_file, 'w') as f:
                    json.dump(symbols or {}, f, indent=2)

                # Create enhanced prompt with symbolic information
                if symbols and 'objects' in symbols:
                    # Extract object information for symbolic context
                    object_descriptions = []
                    for obj in symbols.get('objects', []):
                        obj_desc = f"{obj.get('label', 'object')} at {obj.get('coordinates', [0,0,0,0])}"
                        object_descriptions.append(obj_desc)

                    symbol_info = f"Detected objects: {'; '.join(object_descriptions)}"
                    enhanced_prompt = f"{prompt}\n\nSymbolic context: {symbol_info}"
                else:
                    enhanced_prompt = prompt

                # Save prompt for inspection
                prompt_file = os.path.join(self.temp_dir, f"{call_id}_prompt.txt")
                with open(prompt_file, 'w') as f:
                    f.write(f"VISION+SYMBOL PIPELINE\n")
                    f.write(f"Call ID: {call_id}\n")
                    f.write(f"Game: {game_name}\n")
                    f.write(f"Original frame: {frame_path}\n")
                    f.write(f"Scaled frame (1280x720): {scaled_frame_path}\n")
                    f.write(f"Symbols saved as: {symbols_file}\n")
                    f.write(f"Original prompt: {prompt}\n")
                    f.write(f"Enhanced prompt: {enhanced_prompt}\n")
                    f.write(f"Symbols detected: {len(symbols.get('objects', []))} objects\n")

                # Convert to base64 using scaled image
                base64_image = detector.encode_image_to_base64(scaled_frame_path)

                # Create messages for VLM call with symbolic context
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": enhanced_prompt
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ]

                # Make VLM API call
                response = detector._make_api_call(messages, max_tokens=500, call_id=f"vision_symbol_{call_id}")

                # Save response for inspection
                response_file = os.path.join(self.temp_dir, f"{call_id}_response.txt")
                with open(response_file, 'w') as f:
                    f.write(f"VISION+SYMBOL RESPONSE\n")
                    f.write(f"Call ID: {call_id}\n")
                    f.write(f"Response: {response or 'No response received'}\n")

                return response or "Unable to analyze the image with symbolic grounding."

            except Exception as e:
                print(f"Error in vision+symbol pipeline: {e}")

                # Save error information
                error_file = os.path.join(self.temp_dir, f"{call_id}_error.txt")
                with open(error_file, 'w') as f:
                    f.write(f"VISION+SYMBOL ERROR\n")
                    f.write(f"Call ID: {call_id}\n")
                    f.write(f"Error: {str(e)}\n")
                    f.write(f"Falling back to vision-only mode\n")

                # Fallback to vision-only if symbolic detection fails (use scaled image)
                base64_image = detector.encode_image_to_base64(scaled_frame_path)
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
                        ]
                    }
                ]
                response = detector._make_api_call(messages, max_tokens=500, call_id=f"vision_fallback_{call_id}")

                # Save fallback response
                fallback_file = os.path.join(self.temp_dir, f"{call_id}_fallback_response.txt")
                with open(fallback_file, 'w') as f:
                    f.write(f"FALLBACK RESPONSE\n")
                    f.write(f"Call ID: {call_id}\n")
                    f.write(f"Response: {response or 'No response received'}\n")

                return response or "Unable to analyze the image."

        # Add pipeline type attribution
        vision_symbol_func._pipeline_type = "vision_symbol"
        return vision_symbol_func

    def create_privileged_symbol_pipeline(self):
        """
        Create Privileged-Symbol pipeline using perfect OCAtari ground truth.

        This represents the upper bound performance using perfect symbolic information.
        """
        detector = AdvancedSymbolicDetector(
            self.api_key,
            self.model_name,
            detection_mode="specific",
            provider=self.provider,
            aws_region=self.aws_region,
            disable_history=True
        )
        detector.prompts_dir = os.path.join(self.temp_dir, "privileged_symbol_prompts")
        os.makedirs(detector.prompts_dir, exist_ok=True)

        def privileged_symbol_func(frame: np.ndarray, prompt: str, ground_truth: Dict[str, Any] = None) -> str:
            """Privileged-Symbol pipeline with perfect OCAtari information."""
            # Save frame as image
            frame_path = os.path.join(self.temp_dir, "temp_frame.png")
            frame_image = Image.fromarray(frame.astype('uint8'))
            frame_image.save(frame_path)

            # Create enhanced prompt with perfect symbolic information
            if ground_truth and 'objects' in ground_truth:
                perfect_symbols = []
                for obj in ground_truth['objects']:
                    obj_info = f"{obj['category']} at position {obj['position']}, velocity {obj['velocity']}"
                    perfect_symbols.append(obj_info)

                symbol_context = f"Perfect object information: {'; '.join(perfect_symbols)}"
                enhanced_prompt = f"{prompt}\n\nPerfect symbolic context: {symbol_context}"
            else:
                enhanced_prompt = prompt

            # Convert to base64
            base64_image = detector.encode_image_to_base64(frame_path)

            # Create messages for VLM call
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": enhanced_prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ]

            # Make VLM API call
            response = detector._make_api_call(messages, max_tokens=500, call_id="privileged_symbol")
            return response or "Unable to analyze with perfect symbolic information."

        return privileged_symbol_func

    def run_benchmark(self,
                     games: List[str] = None,
                     num_frames_per_game: int = 15,
                     output_dir: str = "./spatial_benchmark_results") -> Dict[str, Any]:
        """
        Run complete spatial reasoning benchmark.

        Args:
            games: List of games to evaluate ['Pong', 'Breakout', 'SpaceInvaders']
            num_frames_per_game: Number of challenging frames to evaluate per game
            output_dir: Directory to save results

        Returns:
            Complete benchmark results
        """
        if games is None:
            games = ['Pong', 'Breakout', 'SpaceInvaders']

        os.makedirs(output_dir, exist_ok=True)

        # Set up detailed logging - work directly in output directory
        self.log_output_dir = output_dir
        self.temp_dir = os.path.join(output_dir, "inspection_data")  # Work directly in output
        os.makedirs(self.temp_dir, exist_ok=True)

        self.diagnostic_evaluator = AtariGPTDiagnostic(
            enable_detailed_logging=self.enable_detailed_logging,
            log_output_dir=output_dir
        )
        self.trajectory_benchmark = TrajectoryPredictionBenchmark(
            enable_detailed_logging=self.enable_detailed_logging,
            log_output_dir=output_dir
        )

        print(f"   🗂️  Working directory: {self.temp_dir}")
        print(f"   📊 Detailed logs will be saved to: {output_dir}/detailed_logs/")

        print("🎮 Starting Spatial Reasoning Benchmark with Real VLM Pipeline")
        print("=" * 70)

        # Step 1: Select challenging frames across all games
        print("1️⃣ Selecting challenging frames...")
        selected_frames = self.frame_selector.select_challenging_frames(
            num_frames_per_game=num_frames_per_game
        )

        # Convert to evaluation format and add ground truth
        evaluation_frames = self._prepare_evaluation_frames(selected_frames)

        print(f"   ✅ Prepared {len(evaluation_frames)} frames for evaluation")

        # Step 2: Run three-way pipeline comparison
        results = {}

        for game in games:
            print(f"\n2️⃣ Evaluating {game}...")

            # Filter frames for this game
            game_frames = [f for f in evaluation_frames if f['game'] == game]

            if not game_frames:
                print(f"   ⚠️ No frames found for {game}, skipping...")
                continue

            print(f"   📊 Evaluating {len(game_frames)} frames...")

            # Create pipelines
            vision_only = self.create_vision_only_pipeline()
            vision_symbol = self.create_vision_symbol_pipeline(game)
            privileged_symbol = self.create_privileged_symbol_pipeline()

            # Run Atari-GPT diagnostic evaluation
            print("   🔍 Running Atari-GPT diagnostic evaluation...")
            diagnostic_frames = [(f['frame'], f['game'], f['ground_truth']) for f in game_frames[:5]]  # Limit for speed

            vision_only_diag = self.diagnostic_evaluator.evaluate_frame_set(diagnostic_frames, vision_only)
            vision_symbol_diag = self.diagnostic_evaluator.evaluate_frame_set(diagnostic_frames, vision_symbol)

            # Compare diagnostic results
            diagnostic_comparison = self.diagnostic_evaluator.compare_pipelines(
                vision_only_diag, vision_symbol_diag, "Vision-only", "Vision+Symbol"
            )

            # Run trajectory prediction benchmark
            print("   🎯 Running trajectory prediction benchmark...")
            trajectory_frames = [f for f in game_frames if len(f.get('objects', [])) > 0][:3]  # Frames with moving objects

            if trajectory_frames:
                traj_results_vs = self.trajectory_benchmark.evaluate_pipeline(vision_symbol, trajectory_frames)
                traj_results_vo = self.trajectory_benchmark.evaluate_pipeline(vision_only, trajectory_frames)
            else:
                traj_results_vs = traj_results_vo = None

            # Store game results
            results[game] = {
                'diagnostic_comparison': diagnostic_comparison,
                'vision_only_diagnostic': vision_only_diag,
                'vision_symbol_diagnostic': vision_symbol_diag,
                'trajectory_vision_symbol': traj_results_vs,
                'trajectory_vision_only': traj_results_vo,
                'frames_evaluated': len(game_frames)
            }

            # Print game summary
            self._print_game_summary(game, results[game])

        # Step 3: Export comprehensive results
        print(f"\n3️⃣ Exporting results to {output_dir}...")
        self._export_results(results, output_dir, selected_frames)

        print("\n✨ Spatial reasoning benchmark completed!")
        return results

    def _prepare_evaluation_frames(self, selected_frames: List) -> List[Dict[str, Any]]:
        """Prepare frames with ground truth for evaluation."""
        evaluation_frames = []

        for frame_complexity in selected_frames:
            if frame_complexity.frame_data:
                eval_frame = {
                    'frame_id': frame_complexity.frame_id,
                    'game': frame_complexity.game,
                    'frame': np.array(frame_complexity.frame_data['frame']),
                    'objects': frame_complexity.frame_data['objects'],
                    'ground_truth': {
                        'objects': frame_complexity.frame_data['objects'],
                        'spatial_relationships': frame_complexity.frame_data.get('spatial_relationships', {}),
                        'predicted_collisions': frame_complexity.frame_data.get('predicted_collisions', [])
                    },
                    'complexity_score': frame_complexity.complexity_score
                }
                evaluation_frames.append(eval_frame)

        return evaluation_frames

    def _print_game_summary(self, game: str, game_results: Dict[str, Any]):
        """Print summary results for a game."""
        print(f"   📈 {game} Results Summary:")

        if 'diagnostic_comparison' in game_results:
            comp = game_results['diagnostic_comparison']
            overall = comp['overall_comparison']
            spatial = comp['category_comparison'].get('spatial', {})

            print(f"      Overall Improvement: {overall.get('percent_improvement', 0):+.1f}%")
            print(f"      Spatial Improvement: {spatial.get('percent_improvement', 0):+.1f}%")

        if game_results.get('trajectory_vision_symbol'):
            traj = game_results['trajectory_vision_symbol']
            if traj.aggregate_metrics.get('mean_l2_error') != float('inf'):
                print(f"      Trajectory Error: {traj.aggregate_metrics['mean_l2_error']:.1f} pixels")
                print(f"      Trajectory Success: {traj.aggregate_metrics['success_rate']:.1%}")

    def _export_results(self, results: Dict[str, Any], output_dir: str, selected_frames: List):
        """Export comprehensive results to files."""
        # Export main results
        with open(os.path.join(output_dir, "spatial_benchmark_results.json"), 'w') as f:
            # Convert results to JSON-serializable format
            json_results = {}
            for game, game_results in results.items():
                json_results[game] = {}

                if 'diagnostic_comparison' in game_results:
                    json_results[game]['diagnostic_comparison'] = game_results['diagnostic_comparison']

                # Export diagnostic results (without full response texts for size)
                for diag_key in ['vision_only_diagnostic', 'vision_symbol_diagnostic']:
                    if diag_key in game_results:
                        diag_result = game_results[diag_key]
                        json_results[game][diag_key] = {
                            'overall_statistics': diag_result['overall_statistics'],
                            'evaluation_summary': diag_result['evaluation_summary']
                        }

                # Export trajectory results
                for traj_key in ['trajectory_vision_symbol', 'trajectory_vision_only']:
                    if traj_key in game_results and game_results[traj_key]:
                        traj_result = game_results[traj_key]
                        json_results[game][traj_key] = {
                            'aggregate_metrics': traj_result.aggregate_metrics,
                            'game_breakdown': traj_result.game_breakdown,
                            'horizon_analysis': traj_result.horizon_analysis
                        }

                json_results[game]['frames_evaluated'] = game_results.get('frames_evaluated', 0)

            json.dump(json_results, f, indent=2)

        # Export frame selection details
        self.frame_selector.export_selected_frames(
            os.path.join(output_dir, "selected_frames.json")
        )

        # Export summary report
        with open(os.path.join(output_dir, "benchmark_summary.txt"), 'w') as f:
            f.write("Spatial Reasoning Benchmark Summary\n")
            f.write("=" * 40 + "\n\n")

            for game, game_results in results.items():
                f.write(f"{game}:\n")
                if 'diagnostic_comparison' in game_results:
                    comp = game_results['diagnostic_comparison']
                    overall = comp['overall_comparison']
                    f.write(f"  Overall improvement: {overall.get('percent_improvement', 0):+.1f}%\n")

                    for category, cat_comp in comp['category_comparison'].items():
                        f.write(f"  {category}: {cat_comp.get('percent_improvement', 0):+.1f}%\n")
                f.write(f"  Frames evaluated: {game_results.get('frames_evaluated', 0)}\n\n")

        print(f"   📁 Results exported to {output_dir}")

        # Copy all frames, prompts, and responses to output directory for inspection
        self._copy_inspection_data(output_dir)

    def _copy_inspection_data(self, output_dir: str):
        """Copy all saved frames, prompts, and responses to output directory."""
        import shutil
        import glob

        # Create inspection subdirectory
        inspection_dir = os.path.join(output_dir, "inspection_data")
        os.makedirs(inspection_dir, exist_ok=True)

        # Copy all files from temp directory (only files, not directories)
        temp_files = glob.glob(os.path.join(self.temp_dir, "*"))
        files_to_copy = [f for f in temp_files if os.path.isfile(f)]

        print(f"   📋 Copying {len(files_to_copy)} inspection files...")

        for file_path in files_to_copy:
            try:
                filename = os.path.basename(file_path)
                dest_path = os.path.join(inspection_dir, filename)
                shutil.copy2(file_path, dest_path)
            except Exception as e:
                print(f"   ⚠️ Warning: Could not copy {file_path}: {e}")

        # Create index file for easy navigation
        index_file = os.path.join(inspection_dir, "README_INSPECTION.txt")
        with open(index_file, 'w') as f:
            f.write("SPATIAL REASONING BENCHMARK - INSPECTION DATA\n")
            f.write("=" * 50 + "\n\n")
            f.write("This directory contains all frames, prompts, and responses from the benchmark.\n\n")

            # Categorize files
            vision_only_files = [f for f in os.listdir(inspection_dir) if f.startswith("vision_only_")]
            vision_symbol_files = [f for f in os.listdir(inspection_dir) if f.startswith("vision_symbol_")]

            f.write(f"VISION-ONLY FILES ({len(vision_only_files)}):\n")
            for file in sorted(vision_only_files):
                f.write(f"  - {file}\n")

            f.write(f"\nVISION+SYMBOL FILES ({len(vision_symbol_files)}):\n")
            for file in sorted(vision_symbol_files):
                f.write(f"  - {file}\n")

            f.write("\nFILE PATTERNS:\n")
            f.write("  - *_frame.png: Game frame images sent to VLM\n")
            f.write("  - *_prompt.txt: Text prompts sent to VLM\n")
            f.write("  - *_response.txt: VLM responses received\n")
            f.write("  - *_symbols.json: Symbolic detection results (Vision+Symbol only)\n")
            f.write("  - *_error.txt: Error logs (if any)\n")
            f.write("  - *_fallback_response.txt: Fallback responses (if symbolic detection failed)\n")

        print(f"   📂 All inspection data copied to: {inspection_dir}")
        print(f"   📖 See {index_file} for file organization")


def main():
    parser = argparse.ArgumentParser(description="Run Spatial Reasoning Benchmark with VLM Integration")
    parser.add_argument("--provider", default="bedrock", choices=["bedrock", "openrouter"],
                       help="VLM provider to use")
    parser.add_argument("--model", default="anthropic/claude-sonnet-4",
                       help="Model name to use")
    parser.add_argument("--games", nargs="+", default=["Pong", "Breakout"],
                       help="Games to evaluate")
    parser.add_argument("--frames", type=int, default=10,
                       help="Number of frames per game to evaluate")
    parser.add_argument("--output", default="./spatial_benchmark_results",
                       help="Output directory for results")
    parser.add_argument("--aws-region", default="us-east-1",
                       help="AWS region for Bedrock")

    args = parser.parse_args()

    # Initialize benchmark
    benchmark = SpatialReasoningBenchmark(
        provider=args.provider,
        model_name=args.model,
        aws_region=args.aws_region
    )

    # Run benchmark
    results = benchmark.run_benchmark(
        games=args.games,
        num_frames_per_game=args.frames,
        output_dir=args.output
    )

    print(f"\n🎯 Benchmark completed! Results saved to {args.output}")


if __name__ == "__main__":
    main()