#!/usr/bin/env python3
"""
Enhanced Coordinate Accuracy Evaluation with Random Agent Support

This version can:
1. Evaluate existing experiments (like v1)
2. Generate new ground truth with random agent (no VLM needed)
3. Run VLM detection on random agent gameplay
4. Support provider/model switching

Usage:
    # Evaluate existing experiment
    python run_coordinate_evaluation_v2.py \
        --experiment_dir experiments/Pong_seed42 \
        --game pong

    # Generate random agent ground truth (300 frames) and evaluate
    python run_coordinate_evaluation_v2.py \
        --game pong \
        --seed 42 \
        --num_frames 300 \
        --random_agent \
        --provider openrouter \
        --model "anthropic/claude-sonnet-4"

    # Evaluate random agent run (frame sampling)
    python run_coordinate_evaluation_v2.py \
        --game pong \
        --seed 42 \
        --num_frames 300 \
        --random_agent \
        --provider bedrock \
        --model claude-4.5-sonnet \
        --sample_every 10
"""

import argparse
import json
import sys
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
from coordinate_accuracy_evaluator import CoordinateAccuracyEvaluator, FrameEvaluationResult
from ocatari_ground_truth import OCAtariGroundTruth


def generate_random_agent_ground_truth(game: str, seed: int, num_frames: int) -> tuple[List[Dict], List[int]]:
    """Generate ground truth using random agent.

    Args:
        game: Game name
        seed: Random seed
        num_frames: Number of frames to generate

    Returns:
        Tuple of (ground_truth_data, actions)
    """
    print(f"Generating random agent ground truth for {game} (seed={seed}, frames={num_frames})")

    # Map game names to OCAtari format
    game_map = {
        'pong': 'Pong',
        'breakout': 'Breakout',
        'spaceinvaders': 'SpaceInvaders',
        'space_invaders': 'SpaceInvaders'
    }

    ocatari_game_name = game_map.get(game.lower(), game)

    # Initialize OCAtari
    extractor = OCAtariGroundTruth(ocatari_game_name)
    extractor.reset(seed=seed)

    ground_truth_data = []
    actions = []

    # Set random seed for actions
    np.random.seed(seed)

    # Generate ground truth for each frame
    for frame_idx in range(num_frames):
        # Get current state
        frame, objects = extractor.get_frame_and_objects()

        # Store ground truth
        gt_data = {
            'frame': frame_idx,
            'objects': [obj.to_dict() for obj in objects]
        }
        ground_truth_data.append(gt_data)

        # Random action
        action = extractor.env.action_space.sample()
        actions.append(action)

        # Take action
        extractor.step(action)

        if (frame_idx + 1) % 50 == 0:
            print(f"  Generated {frame_idx + 1}/{num_frames} frames...")

    extractor.close()

    print(f"✅ Generated {len(ground_truth_data)} frames of ground truth")
    return ground_truth_data, actions


def run_vlm_detection_on_frames(
    game: str,
    seed: int,
    actions: List[int],
    frames_to_detect: List[int],
    provider: str,
    model: str,
    output_dir: Path
) -> Dict[int, Dict[str, Any]]:
    """Run VLM detection on specific frames.

    Args:
        game: Game name
        seed: Random seed
        actions: Action sequence
        frames_to_detect: Frame numbers to run detection on
        provider: Provider (openrouter, bedrock)
        model: Model name
        output_dir: Output directory for detections

    Returns:
        Dictionary mapping frame_number -> detection_results
    """
    print(f"\n{'='*80}")
    print(f"Running VLM detection with {provider}/{model}")
    print(f"Frames to detect: {len(frames_to_detect)}")
    print(f"{'='*80}\n")

    # Import the detector
    from advanced_zero_shot_pipeline import AdvancedSymbolicDetector

    # Map game to detector
    game_map = {
        'pong': ('Pong', {
            0: "NOOP", 1: "FIRE", 2: "RIGHT/UP", 3: "LEFT/DOWN",
            4: "RIGHTFIRE/UPFIRE", 5: "LEFTFIRE/DOWNFIRE"
        }),
        'breakout': ('Breakout', {
            0: "NOOP", 1: "FIRE", 2: "RIGHT", 3: "LEFT",
            4: "RIGHTFIRE", 5: "LEFTFIRE"
        }),
        'spaceinvaders': ('Space Invaders', {
            0: "NOOP", 1: "FIRE", 2: "RIGHT", 3: "LEFT",
            4: "RIGHTFIRE", 5: "LEFTFIRE"
        }),
        'space_invaders': ('Space Invaders', {
            0: "NOOP", 1: "FIRE", 2: "RIGHT", 3: "LEFT",
            4: "RIGHTFIRE", 5: "LEFTFIRE"
        })
    }

    game_name, game_controls = game_map.get(game.lower(), (game, {}))

    # Get API key
    if provider == 'openrouter':
        api_key = os.environ.get('OPENROUTER_API_KEY', '')
        if not api_key:
            # Try to load from file
            key_file = Path('OPENROUTER_API_KEY.txt')
            if key_file.exists():
                api_key = key_file.read_text().strip()
            else:
                print("Error: OPENROUTER_API_KEY not found in environment or OPENROUTER_API_KEY.txt")
                return {}
    else:
        api_key = ""  # Bedrock uses AWS credentials

    # Initialize detector
    detector = AdvancedSymbolicDetector(
        openrouter_api_key=api_key,
        model_name=model,
        provider=provider,
        disable_history=True  # Disable history for isolated frame detection
    )

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize OCAtari to replay and save frames
    game_map_ocatari = {
        'pong': 'Pong',
        'breakout': 'Breakout',
        'spaceinvaders': 'SpaceInvaders',
        'space_invaders': 'SpaceInvaders'
    }
    ocatari_game = game_map_ocatari.get(game.lower(), game)
    extractor = OCAtariGroundTruth(ocatari_game)
    extractor.reset(seed=seed)

    # Replay actions up to max frame needed
    max_frame = max(frames_to_detect)
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(exist_ok=True)

    print("Replaying game to capture frames...")
    for frame_idx in range(max_frame + 1):
        if frame_idx in frames_to_detect:
            # Save frame
            frame_rgb = extractor.env.render()
            import cv2
            frame_path = frames_dir / f"frame_{frame_idx:06d}.png"
            cv2.imwrite(str(frame_path), cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))

        # Take action
        if frame_idx < len(actions):
            extractor.step(actions[frame_idx])
        else:
            extractor.step(0)

        if (frame_idx + 1) % 50 == 0:
            print(f"  Replayed {frame_idx + 1}/{max_frame + 1} frames...")

    extractor.close()
    print("✅ Frames captured")

    # Run detection on each frame
    detections = {}
    for idx, frame_num in enumerate(frames_to_detect):
        print(f"\n[{idx+1}/{len(frames_to_detect)}] Detecting frame {frame_num}...")

        frame_path = frames_dir / f"frame_{frame_num:06d}.png"
        frame_output_dir = output_dir / f"frame_{frame_num:06d}_detection"

        if not frame_path.exists():
            print(f"  Warning: Frame {frame_num} not found, skipping")
            continue

        try:
            # Run detection
            result = detector.process_single_frame(
                str(frame_path),
                str(frame_output_dir),
                game_name,
                game_controls
            )

            # Extract detections
            if 'symbolic_state' in result:
                detections[frame_num] = {'objects': result['symbolic_state'].get('objects', [])}
            else:
                detections[frame_num] = {'objects': []}

            print(f"  ✅ Detected {len(detections[frame_num]['objects'])} objects")

        except Exception as e:
            print(f"  ❌ Error detecting frame {frame_num}: {e}")
            detections[frame_num] = {'objects': []}

    return detections


def evaluate_with_detections(
    game: str,
    seed: int,
    ground_truth_data: List[Dict],
    detections: Dict[int, Dict],
    frames_to_eval: List[int],
    output_dir: Path,
    provider: str = "random",
    model: str = "random_agent"
) -> Dict[str, Any]:
    """Evaluate VLM detections against ground truth.

    Args:
        game: Game name
        seed: Random seed
        ground_truth_data: Ground truth data for all frames
        detections: VLM detections {frame_num: detection_dict}
        frames_to_eval: Frame numbers to evaluate
        output_dir: Output directory
        provider: Provider name (for metadata)
        model: Model name (for metadata)

    Returns:
        Evaluation results dictionary
    """
    print(f"\n{'='*80}")
    print(f"Evaluating Detection Accuracy")
    print(f"{'='*80}\n")

    # Initialize evaluator
    evaluator = CoordinateAccuracyEvaluator(game)

    # Evaluate each frame
    results = []
    for frame_num in frames_to_eval:
        if frame_num >= len(ground_truth_data):
            print(f"  Warning: Frame {frame_num} beyond ground truth, skipping")
            continue

        if frame_num not in detections:
            print(f"  Warning: No detections for frame {frame_num}, skipping")
            continue

        # Get data
        gt_data = ground_truth_data[frame_num]
        vlm_data = detections[frame_num]

        # Evaluate
        result = evaluator.evaluate_frame(gt_data, vlm_data, frame_id=frame_num)
        results.append(result)

        print(f"  Frame {frame_num}: "
              f"Precision={result.precision:.3f}, "
              f"Recall={result.recall:.3f}, "
              f"F1={result.f1_score:.3f}, "
              f"Important F1={result.important_f1:.3f}")

    # Aggregate results
    if results:
        aggregate = {
            'experiment': f"{game}_seed{seed}_{provider}_{model.replace('/', '_')}",
            'game': game,
            'seed': seed,
            'provider': provider,
            'model': model,
            'frames_evaluated': len(results),
            'avg_precision': float(np.mean([r.precision for r in results])),
            'avg_recall': float(np.mean([r.recall for r in results])),
            'avg_f1': float(np.mean([r.f1_score for r in results])),
            'avg_important_precision': float(np.mean([r.important_precision for r in results])),
            'avg_important_recall': float(np.mean([r.important_recall for r in results])),
            'avg_important_f1': float(np.mean([r.important_f1 for r in results])),
            'avg_iou': float(np.mean([r.avg_iou for r in results])),
            'avg_center_distance': float(np.mean([r.avg_center_distance for r in results])),
            'frame_results': [
                {
                    'frame': r.frame_id,
                    'precision': r.precision,
                    'recall': r.recall,
                    'f1': r.f1_score,
                    'important_f1': r.important_f1,
                    'iou': r.avg_iou,
                    'center_distance': r.avg_center_distance
                }
                for r in results
            ]
        }

        # Save results
        output_file = output_dir / "evaluation_summary.json"
        with open(output_file, 'w') as f:
            json.dump(aggregate, f, indent=2)

        print(f"\n✅ Evaluation complete! Results saved to: {output_file}")
        print(f"\nSUMMARY:")
        print(f"  Average Precision: {aggregate['avg_precision']:.3f}")
        print(f"  Average Recall: {aggregate['avg_recall']:.3f}")
        print(f"  Average F1: {aggregate['avg_f1']:.3f}")
        print(f"  Important Objects F1: {aggregate['avg_important_f1']:.3f}")
        print(f"  Average IoU: {aggregate['avg_iou']:.3f}")
        print(f"  Average Center Distance: {aggregate['avg_center_distance']:.1f}px")

        return aggregate

    return None


def main():
    parser = argparse.ArgumentParser(
        description="Enhanced coordinate evaluation with random agent support",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate existing experiment
  python run_coordinate_evaluation_v2.py \\
      --experiment_dir experiments/Pong_seed42 --game pong

  # Generate 300 frames with random agent and evaluate with VLM
  python run_coordinate_evaluation_v2.py \\
      --game pong --seed 42 --num_frames 300 --random_agent \\
      --provider openrouter --model "anthropic/claude-sonnet-4" \\
      --sample_every 10

  # Just generate ground truth (no VLM detection)
  python run_coordinate_evaluation_v2.py \\
      --game breakout --seed 123 --num_frames 300 --random_agent \\
      --ground_truth_only --output_dir evaluation_data/
        """
    )

    # Mode selection
    parser.add_argument('--experiment_dir', type=str, default=None,
                       help='Path to existing experiment directory (mode 1: evaluate existing)')
    parser.add_argument('--random_agent', action='store_true',
                       help='Use random agent mode (mode 2: generate new data)')

    # Game parameters
    parser.add_argument('--game', type=str, required=True,
                       choices=['pong', 'breakout', 'spaceinvaders', 'space_invaders'],
                       help='Game name')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    parser.add_argument('--num_frames', type=int, default=300,
                       help='Number of frames for random agent (default: 300)')

    # VLM parameters
    parser.add_argument('--provider', type=str, default='openrouter',
                       choices=['openrouter', 'bedrock'],
                       help='VLM provider (default: openrouter)')
    parser.add_argument('--model', type=str, default='anthropic/claude-sonnet-4',
                       help='Model name (default: anthropic/claude-sonnet-4)')

    # Evaluation parameters
    parser.add_argument('--frames', type=str, default=None,
                       help='Comma-separated list of frame numbers to evaluate')
    parser.add_argument('--sample_every', type=int, default=10,
                       help='Sample every N frames for evaluation (default: 10)')
    parser.add_argument('--ground_truth_only', action='store_true',
                       help='Only generate ground truth, skip VLM detection')

    # Output
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory (default: auto-generated)')

    args = parser.parse_args()

    # Determine mode
    if args.experiment_dir:
        # Mode 1: Evaluate existing experiment
        print("Mode: Evaluating existing experiment")
        from run_coordinate_evaluation import evaluate_experiment

        experiment_dir = Path(args.experiment_dir)
        if not experiment_dir.exists():
            print(f"Error: Directory {experiment_dir} does not exist")
            sys.exit(1)

        # Parse frames
        frames_to_eval = None
        if args.frames:
            try:
                frames_to_eval = [int(f.strip()) for f in args.frames.split(',')]
            except ValueError:
                print(f"Error: Invalid frame numbers: {args.frames}")
                sys.exit(1)

        evaluate_experiment(experiment_dir, args.game, frames_to_eval)

    elif args.random_agent:
        # Mode 2: Random agent mode
        print("Mode: Random agent evaluation")

        # Set output directory
        if args.output_dir:
            output_dir = Path(args.output_dir)
        else:
            output_dir = Path(f"coordinate_eval_{args.game}_seed{args.seed}_{args.provider}_{args.model.replace('/', '_')}")

        output_dir.mkdir(parents=True, exist_ok=True)

        # Generate ground truth
        ground_truth_data, actions = generate_random_agent_ground_truth(
            args.game, args.seed, args.num_frames
        )

        # Save ground truth
        gt_file = output_dir / "ground_truth.json"
        with open(gt_file, 'w') as f:
            json.dump({
                'game': args.game,
                'seed': args.seed,
                'num_frames': args.num_frames,
                'actions': actions,
                'ground_truth': ground_truth_data
            }, f, indent=2)
        print(f"✅ Ground truth saved to: {gt_file}")

        if args.ground_truth_only:
            print("Ground truth only mode - skipping VLM detection")
            return

        # Determine frames to evaluate
        if args.frames:
            frames_to_eval = [int(f.strip()) for f in args.frames.split(',')]
        else:
            frames_to_eval = list(range(0, args.num_frames, args.sample_every))

        print(f"\nWill evaluate {len(frames_to_eval)} frames")

        # Run VLM detection
        detections = run_vlm_detection_on_frames(
            args.game, args.seed, actions, frames_to_eval,
            args.provider, args.model, output_dir
        )

        # Evaluate
        result = evaluate_with_detections(
            args.game, args.seed, ground_truth_data, detections,
            frames_to_eval, output_dir, args.provider, args.model
        )

    else:
        print("Error: Must specify either --experiment_dir or --random_agent")
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
