#!/usr/bin/env python3
"""
V3: Enhanced Coordinate Evaluation - SAVES EVERYTHING

New features in V3:
- Saves ALL frames (not just sampled)
- Saves OCAtari ground truth visualizations
- Saves VLM detection visualizations
- Saves VLM raw responses with coordinates
- Creates side-by-side comparison frames
- Comprehensive data export for analysis

Usage:
    python run_coordinate_evaluation_v3.py \
        --game pong \
        --seed 42 \
        --num_frames 300 \
        --random_agent \
        --provider openrouter \
        --model "anthropic/claude-sonnet-4" \
        --save_all_frames
"""

import argparse
import json
import sys
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
import cv2
from coordinate_accuracy_evaluator import CoordinateAccuracyEvaluator, FrameEvaluationResult
from ocatari_ground_truth import OCAtariGroundTruth
from visualize_ground_truth import GroundTruthVisualizer


def convert_numpy_types(obj):
    """Convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(item) for item in obj)
    return obj


def generate_random_agent_ground_truth_v3(
    game: str,
    seed: int,
    num_frames: int,
    save_all_frames: bool = True,
    frames_dir: Path = None
) -> tuple[List[Dict], List[int]]:
    """Generate ground truth with optional frame saving.

    Args:
        game: Game name
        seed: Random seed
        num_frames: Number of frames
        save_all_frames: If True, save all RGB frames
        frames_dir: Directory to save frames

    Returns:
        Tuple of (ground_truth_data, actions)
    """
    print(f"Generating random agent ground truth for {game} (seed={seed}, frames={num_frames})")
    if save_all_frames:
        print(f"  Saving all {num_frames} frames to {frames_dir}")

    # Map game names
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

    # Create frames directory if saving
    if save_all_frames and frames_dir:
        frames_dir.mkdir(parents=True, exist_ok=True)

    # Set random seed
    np.random.seed(seed)

    # Generate ground truth
    for frame_idx in range(num_frames):
        # Get current state
        frame_rgb, objects = extractor.get_frame_and_objects()

        # Save frame if requested
        if save_all_frames and frames_dir:
            frame_path = frames_dir / f"frame_{frame_idx:06d}.png"
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(frame_path), frame_bgr)

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
    if save_all_frames:
        print(f"✅ Saved {num_frames} RGB frames")

    return ground_truth_data, actions


def run_vlm_detection_on_frames_v3(
    game: str,
    seed: int,
    actions: List[int],
    frames_to_detect: List[int],
    provider: str,
    model: str,
    output_dir: Path,
    frames_dir: Path
) -> Dict[int, Dict[str, Any]]:
    """Run VLM detection and save everything.

    Args:
        game: Game name
        seed: Random seed
        actions: Action sequence
        frames_to_detect: Frame numbers to detect
        provider: Provider name
        model: Model name
        output_dir: Output directory
        frames_dir: Directory where frames are saved

    Returns:
        Dictionary mapping frame_number -> detection_results
    """
    print(f"\n{'='*80}")
    print(f"Running VLM detection with {provider}/{model}")
    print(f"Frames to detect: {len(frames_to_detect)}")
    print(f"{'='*80}\n")

    # Import detector
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
            key_file = Path('OPENROUTER_API_KEY.txt')
            if key_file.exists():
                api_key = key_file.read_text().strip()
            else:
                print("Error: OPENROUTER_API_KEY not found")
                return {}
    else:
        api_key = ""

    # Initialize detector
    detector = AdvancedSymbolicDetector(
        openrouter_api_key=api_key,
        model_name=model,
        provider=provider,
        disable_history=True
    )

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

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

            # Extract detections with full response
            if 'symbolic_state' in result:
                detections[frame_num] = {
                    'objects': result['symbolic_state'].get('objects', []),
                    'full_response': result.get('full_api_response', ''),
                    'reasoning': result.get('action_decision', {}).get('reasoning', ''),
                    'raw_detections': result['symbolic_state']
                }
            else:
                detections[frame_num] = {'objects': []}

            # Save comprehensive detection data
            detection_file = frame_output_dir / "detection_full.json"
            with open(detection_file, 'w') as f:
                json.dump({
                    'frame': frame_num,
                    'game': game,
                    'provider': provider,
                    'model': model,
                    'detections': detections[frame_num]
                }, f, indent=2)

            print(f"  ✅ Detected {len(detections[frame_num]['objects'])} objects")
            print(f"  💾 Saved full detection data to {detection_file.name}")

        except Exception as e:
            print(f"  ❌ Error detecting frame {frame_num}: {e}")
            detections[frame_num] = {'objects': []}

    return detections


def create_visualizations_v3(
    ground_truth_data: List[Dict],
    detections: Dict[int, Dict],
    frames_to_eval: List[int],
    frames_dir: Path,
    output_dir: Path,
    game: str
) -> None:
    """Create comprehensive visualizations.

    Args:
        ground_truth_data: Ground truth for all frames
        detections: VLM detections
        frames_to_eval: Frames to visualize
        frames_dir: Directory with RGB frames
        output_dir: Output directory
        game: Game name
    """
    print(f"\n{'='*80}")
    print(f"Creating Visualizations")
    print(f"{'='*80}\n")

    visualizer = GroundTruthVisualizer()
    evaluator = CoordinateAccuracyEvaluator(game)

    # Create subdirectories
    gt_viz_dir = output_dir / "ground_truth_visualizations"
    vlm_viz_dir = output_dir / "vlm_visualizations"
    comparison_dir = output_dir / "side_by_side_comparisons"

    gt_viz_dir.mkdir(parents=True, exist_ok=True)
    vlm_viz_dir.mkdir(parents=True, exist_ok=True)
    comparison_dir.mkdir(parents=True, exist_ok=True)

    for frame_num in frames_to_eval:
        if frame_num >= len(ground_truth_data):
            continue

        frame_path = frames_dir / f"frame_{frame_num:06d}.png"
        if not frame_path.exists():
            continue

        # Load frame
        frame_bgr = cv2.imread(str(frame_path))
        if frame_bgr is None:
            continue

        # Scale frame to 1280x720
        frame_bgr_scaled = cv2.resize(frame_bgr, (1280, 720), interpolation=cv2.INTER_LINEAR)

        # 1. Visualize ground truth
        gt_data = ground_truth_data[frame_num]
        gt_objects = evaluator.prepare_ocatari_objects(gt_data)

        gt_output_path = gt_viz_dir / f"frame_{frame_num:06d}_ground_truth.png"
        visualizer.draw_ground_truth(
            frame_bgr_scaled,
            gt_objects,
            gt_output_path,
            title=f"Frame {frame_num}: OCAtari Ground Truth ({len(gt_objects)} objects)"
        )

        # 2. Check if VLM detection exists
        vlm_annotated_path = output_dir / f"frame_{frame_num:06d}_detection" / "annotated_frame.jpg"
        if vlm_annotated_path.exists():
            # Copy VLM visualization
            vlm_output_path = vlm_viz_dir / f"frame_{frame_num:06d}_vlm_detection.png"
            vlm_frame = cv2.imread(str(vlm_annotated_path))
            if vlm_frame is not None:
                cv2.imwrite(str(vlm_output_path), vlm_frame)

                # 3. Create side-by-side comparison
                comparison_path = comparison_dir / f"frame_{frame_num:06d}_comparison.png"
                visualizer.create_side_by_side(
                    gt_output_path,
                    vlm_output_path,
                    comparison_path,
                    frame_num
                )

        print(f"  ✅ Visualized frame {frame_num}")

    print(f"\n✅ Visualizations saved:")
    print(f"  Ground Truth: {gt_viz_dir}")
    print(f"  VLM Detection: {vlm_viz_dir}")
    print(f"  Comparisons: {comparison_dir}")


def evaluate_with_full_export(
    game: str,
    seed: int,
    ground_truth_data: List[Dict],
    detections: Dict[int, Dict],
    frames_to_eval: List[int],
    output_dir: Path,
    provider: str,
    model: str
) -> Dict[str, Any]:
    """Evaluate and export comprehensive results."""

    print(f"\n{'='*80}")
    print(f"Evaluating Detection Accuracy")
    print(f"{'='*80}\n")

    evaluator = CoordinateAccuracyEvaluator(game)

    # Evaluate each frame
    results = []
    detailed_results = []

    for frame_num in frames_to_eval:
        if frame_num >= len(ground_truth_data):
            continue

        if frame_num not in detections:
            continue

        # Get data
        gt_data = ground_truth_data[frame_num]
        vlm_data = detections[frame_num]

        # Evaluate
        result = evaluator.evaluate_frame(gt_data, vlm_data, frame_id=frame_num)
        results.append(result)

        # Detailed result
        detailed_results.append({
            'frame': frame_num,
            'ground_truth_objects': len(gt_data['objects']),
            'vlm_detections': len(vlm_data['objects']),
            'matched_objects': result.matched_objects,
            'precision': result.precision,
            'recall': result.recall,
            'f1': result.f1_score,
            'important_f1': result.important_f1,
            'iou': result.avg_iou,
            'center_distance': result.avg_center_distance,
            'vlm_response': vlm_data.get('full_response', '')[:200] + '...',  # Truncated
            'vlm_reasoning': vlm_data.get('reasoning', '')[:200] + '...'
        })

        print(f"  Frame {frame_num}: "
              f"P={result.precision:.3f}, "
              f"R={result.recall:.3f}, "
              f"F1={result.f1_score:.3f}, "
              f"ImpF1={result.important_f1:.3f}")

    # Aggregate
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
            'detailed_frame_results': detailed_results
        }

        # Save main summary
        output_file = output_dir / "evaluation_summary.json"
        with open(output_file, 'w') as f:
            json.dump(aggregate, f, indent=2)

        # Save detailed per-frame results
        detailed_file = output_dir / "detailed_frame_results.json"
        with open(detailed_file, 'w') as f:
            json.dump(detailed_results, f, indent=2)

        print(f"\n✅ Evaluation complete!")
        print(f"  Main summary: {output_file}")
        print(f"  Detailed results: {detailed_file}")
        print(f"\nSUMMARY:")
        print(f"  Precision: {aggregate['avg_precision']:.3f}")
        print(f"  Recall: {aggregate['avg_recall']:.3f}")
        print(f"  F1: {aggregate['avg_f1']:.3f}")
        print(f"  Important F1: {aggregate['avg_important_f1']:.3f}")
        print(f"  IoU: {aggregate['avg_iou']:.3f}")
        print(f"  Center Distance: {aggregate['avg_center_distance']:.1f}px")

        return aggregate

    return None


def main():
    parser = argparse.ArgumentParser(
        description="V3: Enhanced coordinate evaluation - saves everything",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Game parameters
    parser.add_argument('--game', type=str, required=True,
                       choices=['pong', 'breakout', 'spaceinvaders', 'space_invaders'])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_frames', type=int, default=300)

    # Mode
    parser.add_argument('--random_agent', action='store_true', required=True,
                       help='Random agent mode (V3 only supports this)')

    # VLM parameters
    parser.add_argument('--provider', type=str, default='openrouter',
                       choices=['openrouter', 'bedrock'])
    parser.add_argument('--model', type=str, default='anthropic/claude-sonnet-4')

    # Evaluation parameters
    parser.add_argument('--sample_every', type=int, default=10)
    parser.add_argument('--frames', type=str, default=None)

    # NEW: Save options
    parser.add_argument('--save_all_frames', action='store_true',
                       help='Save all frames (not just sampled)')
    parser.add_argument('--create_visualizations', action='store_true',
                       help='Create ground truth and comparison visualizations')

    # Output
    parser.add_argument('--output_dir', type=str, default=None)

    args = parser.parse_args()

    # Set output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(f"coordinate_eval_v3_{args.game}_seed{args.seed}_{args.provider}_{args.model.replace('/', '_')}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Frames directory
    frames_dir = output_dir / "frames"

    # Generate ground truth
    ground_truth_data, actions = generate_random_agent_ground_truth_v3(
        args.game, args.seed, args.num_frames,
        save_all_frames=args.save_all_frames or args.create_visualizations,
        frames_dir=frames_dir
    )

    # Save ground truth (convert numpy types)
    gt_file = output_dir / "ground_truth.json"
    with open(gt_file, 'w') as f:
        gt_data_to_save = {
            'game': args.game,
            'seed': args.seed,
            'num_frames': args.num_frames,
            'actions': convert_numpy_types(actions),
            'ground_truth': convert_numpy_types(ground_truth_data)
        }
        json.dump(gt_data_to_save, f, indent=2)
    print(f"✅ Ground truth saved to: {gt_file}")

    # Determine frames to evaluate
    if args.frames:
        frames_to_eval = [int(f.strip()) for f in args.frames.split(',')]
    else:
        frames_to_eval = list(range(0, args.num_frames, args.sample_every))

    print(f"\nWill evaluate {len(frames_to_eval)} frames")

    # Run VLM detection
    detections = run_vlm_detection_on_frames_v3(
        args.game, args.seed, actions, frames_to_eval,
        args.provider, args.model, output_dir, frames_dir
    )

    # Create visualizations if requested
    if args.create_visualizations:
        create_visualizations_v3(
            ground_truth_data, detections, frames_to_eval,
            frames_dir, output_dir, args.game
        )

    # Evaluate
    result = evaluate_with_full_export(
        args.game, args.seed, ground_truth_data, detections,
        frames_to_eval, output_dir, args.provider, args.model
    )

    print(f"\n{'='*80}")
    print(f"✅ V3 EVALUATION COMPLETE - Everything Saved!")
    print(f"{'='*80}")
    print(f"Output directory: {output_dir}")
    print(f"  - All frames: {frames_dir}")
    print(f"  - Ground truth: ground_truth.json")
    print(f"  - VLM detections: frame_*_detection/")
    print(f"  - Visualizations: */")
    print(f"  - Results: evaluation_summary.json")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
