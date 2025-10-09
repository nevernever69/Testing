#!/usr/bin/env python3
"""
V3 Detection Only: Fast coordinate evaluation without action reasoning

Only runs object detection (no action decision), making it:
- 2x faster (one API call per frame instead of two)
- 50% cheaper (half the API costs)
- Focused on what matters: coordinate accuracy

Usage:
    python run_coordinate_evaluation_v3_detection_only.py \
        --game pong \
        --seed 42 \
        --num_frames 300 \
        --random_agent \
        --provider bedrock \
        --model claude-4-sonnet \
        --save_all_frames \
        --create_visualizations
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
    frames_dir: Path = None,
    start_frame: int = 0
) -> tuple[List[Dict], List[int]]:
    """Generate ground truth with optional frame saving.

    Args:
        start_frame: Skip this many frames at the start (useful to skip initialization frames)
                    The saved frames will be renumbered starting from 0.
    """
    total_frames = num_frames + start_frame
    print(f"Generating random agent ground truth for {game} (seed={seed}, total frames={total_frames})")
    if start_frame > 0:
        print(f"  ⚠️  Skipping first {start_frame} frames (initialization)")
        print(f"  → Will save {num_frames} frames starting from frame {start_frame}")
    if save_all_frames:
        print(f"  Saving all {num_frames} frames to {frames_dir}")

    game_map = {
        'pong': 'Pong',
        'breakout': 'Breakout',
        'spaceinvaders': 'SpaceInvaders',
        'space_invaders': 'SpaceInvaders'
    }

    ocatari_game_name = game_map.get(game.lower(), game)
    extractor = OCAtariGroundTruth(ocatari_game_name)
    extractor.reset(seed=seed)

    ground_truth_data = []
    actions = []

    if save_all_frames and frames_dir:
        frames_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(seed)

    for frame_idx in range(total_frames):
        frame_rgb, objects = extractor.get_frame_and_objects()

        # Only save frames after start_frame, but renumber them starting from 0
        if frame_idx >= start_frame:
            saved_frame_idx = frame_idx - start_frame

            if save_all_frames and frames_dir:
                frame_path = frames_dir / f"frame_{saved_frame_idx:06d}.png"
                frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(frame_path), frame_bgr)

            gt_data = {
                'frame': saved_frame_idx,
                'objects': [obj.to_dict() for obj in objects]
            }
            ground_truth_data.append(gt_data)

        action = extractor.env.action_space.sample()
        actions.append(action)
        extractor.step(action)

        if (frame_idx + 1) % 50 == 0:
            print(f"  Generated {frame_idx + 1}/{total_frames} frames...")

    extractor.close()

    print(f"✅ Generated {len(ground_truth_data)} frames of ground truth")
    if save_all_frames:
        print(f"✅ Saved {num_frames} RGB frames")

    return ground_truth_data, actions


def run_detection_only(
    game: str,
    frames_to_detect: List[int],
    provider: str,
    model: str,
    output_dir: Path,
    frames_dir: Path
) -> Dict[int, Dict[str, Any]]:
    """Run VLM detection only (no action reasoning).

    This is 2x faster and 50% cheaper than full pipeline.
    """
    print(f"\n{'='*80}")
    print(f"Running DETECTION ONLY with {provider}/{model}")
    print(f"Frames to detect: {len(frames_to_detect)}")
    print(f"Note: Skipping action reasoning (only doing object detection)")
    print(f"{'='*80}\n")

    from advanced_zero_shot_pipeline import AdvancedSymbolicDetector

    game_map = {
        'pong': 'Pong',
        'breakout': 'Breakout',
        'spaceinvaders': 'Space Invaders',
        'space_invaders': 'Space Invaders'
    }

    game_name = game_map.get(game.lower(), game)

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

    output_dir.mkdir(parents=True, exist_ok=True)

    detections = {}
    for idx, frame_num in enumerate(frames_to_detect):
        print(f"\n[{idx+1}/{len(frames_to_detect)}] Detecting frame {frame_num}...")

        frame_path = frames_dir / f"frame_{frame_num:06d}.png"
        if not frame_path.exists():
            print(f"  Warning: Frame {frame_num} not found, skipping")
            continue

        try:
            # Scale image
            frame_bgr = cv2.imread(str(frame_path))
            if frame_bgr is None:
                print(f"  Error: Could not load frame")
                continue

            frame_scaled = cv2.resize(frame_bgr, (1280, 720), interpolation=cv2.INTER_LINEAR)

            # Create output directory
            frame_output_dir = output_dir / f"frame_{frame_num:06d}_detection"
            frame_output_dir.mkdir(parents=True, exist_ok=True)

            # Create prompts directory for this frame
            prompts_dir = frame_output_dir / "prompts"
            prompts_dir.mkdir(parents=True, exist_ok=True)

            # Set prompts directory on detector to enable prompt/response saving
            detector.prompts_dir = str(prompts_dir)

            # Save scaled frame
            scaled_path = frame_output_dir / "scaled_frame.jpg"
            cv2.imwrite(str(scaled_path), frame_scaled)

            # Run DETECTION ONLY
            print(f"  → Running object detection...")
            detection_result = detector.detect_objects(str(scaled_path), game_name)

            if "error" in detection_result:
                print(f"  ❌ Detection failed: {detection_result['error']}")
                detections[frame_num] = {'objects': []}
                continue

            # Draw bounding boxes
            annotated_path = frame_output_dir / "annotated_frame.jpg"
            if detection_result.get("objects"):
                detector.draw_bounding_boxes(str(scaled_path), detection_result, str(annotated_path))
            else:
                cv2.imwrite(str(annotated_path), frame_scaled)

            # Extract objects
            detections[frame_num] = {
                'objects': detector.generate_symbolic_state(detection_result).get('objects', []),
                'full_response': detection_result.get('full_api_response', ''),
                'raw_detections': detection_result
            }

            # Read the saved prompt file if it exists
            prompt_file = prompts_dir / "api_prompt_detection.txt"
            prompt_text = ""
            if prompt_file.exists():
                prompt_text = prompt_file.read_text()

            # Save comprehensive detection data with prompt and response
            detection_file = frame_output_dir / "detection_full.json"
            with open(detection_file, 'w') as f:
                json.dump({
                    'frame': frame_num,
                    'game': game,
                    'provider': provider,
                    'model': model,
                    'prompt': prompt_text,
                    'response': detection_result.get('full_api_response', ''),
                    'detections': detections[frame_num]
                }, f, indent=2)

            print(f"  ✅ Detected {len(detections[frame_num]['objects'])} objects")

        except Exception as e:
            print(f"  ❌ Error: {e}")
            import traceback
            traceback.print_exc()
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
    """Create comprehensive visualizations."""
    print(f"\n{'='*80}")
    print(f"Creating Visualizations")
    print(f"{'='*80}\n")

    visualizer = GroundTruthVisualizer()
    evaluator = CoordinateAccuracyEvaluator(game)

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

        frame_bgr = cv2.imread(str(frame_path))
        if frame_bgr is None:
            continue

        frame_bgr_scaled = cv2.resize(frame_bgr, (1280, 720), interpolation=cv2.INTER_LINEAR)

        # Ground truth visualization
        gt_data = ground_truth_data[frame_num]
        gt_objects = evaluator.prepare_ocatari_objects(gt_data)

        gt_output_path = gt_viz_dir / f"frame_{frame_num:06d}_ground_truth.png"
        visualizer.draw_ground_truth(
            frame_bgr_scaled,
            gt_objects,
            gt_output_path,
            title=f"Frame {frame_num}: OCAtari Ground Truth ({len(gt_objects)} objects)"
        )

        # VLM visualization
        vlm_annotated_path = output_dir / f"frame_{frame_num:06d}_detection" / "annotated_frame.jpg"
        if vlm_annotated_path.exists():
            vlm_output_path = vlm_viz_dir / f"frame_{frame_num:06d}_vlm_detection.png"
            vlm_frame = cv2.imread(str(vlm_annotated_path))
            if vlm_frame is not None:
                cv2.imwrite(str(vlm_output_path), vlm_frame)

                # Side-by-side comparison
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

    results = []
    detailed_results = []
    comparison_data = []  # NEW: For detailed matching comparison

    for frame_num in frames_to_eval:
        if frame_num >= len(ground_truth_data):
            continue

        if frame_num not in detections:
            continue

        gt_data = ground_truth_data[frame_num]
        vlm_data = detections[frame_num]

        result = evaluator.evaluate_frame(gt_data, vlm_data, frame_id=frame_num)
        results.append(result)

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
            'vlm_response': vlm_data.get('full_response', '')[:200] + '...',
            'prompt_file': f"frame_{frame_num:06d}_detection/prompts/api_prompt_detection.txt",
            'response_file': f"frame_{frame_num:06d}_detection/detection_full.json"
        })

        # NEW: Create detailed comparison data for this frame
        frame_comparison = {
            'frame': frame_num,
            'metrics': {
                'precision': result.precision,
                'recall': result.recall,
                'f1': result.f1_score,
                'important_f1': result.important_f1,
                'iou': result.avg_iou,
                'center_distance': result.avg_center_distance
            },
            'matches': []
        }

        # Add all match details
        for match in result.matches:
            match_detail = {
                'matched': match.matched,
                'match_quality': match.match_quality,
                'iou': float(match.iou),
                'center_distance': float(match.center_distance)
            }

            # Ground truth info
            if match.gt_obj:
                gt_bbox = match.gt_obj['bbox']
                match_detail['ground_truth'] = {
                    'category': match.gt_obj.get('category', 'Unknown'),
                    'bbox': [float(x) for x in gt_bbox],
                    'center': [float(c) for c in match.gt_obj['center']],
                    'is_important': match.gt_obj.get('is_important', False)
                }
            else:
                match_detail['ground_truth'] = None

            # VLM detection info
            if match.vlm_obj and match.vlm_obj != {}:
                vlm_bbox = match.vlm_obj['bbox']
                match_detail['vlm_detection'] = {
                    'label': match.vlm_obj.get('label', 'unknown'),
                    'normalized_label': match.vlm_obj.get('normalized_label', 'unknown'),
                    'bbox': [float(x) for x in vlm_bbox],
                    'center': [float(c) for c in match.vlm_obj['center']],
                    'confidence': float(match.vlm_obj.get('confidence', 0.0)),
                    'is_important': match.vlm_obj.get('is_important', False)
                }

                # Calculate semantic similarity score
                if match.gt_obj:
                    from coordinate_accuracy_evaluator import GameSpecificMatcher
                    semantic_score = GameSpecificMatcher.fuzzy_match_objects(
                        match.gt_obj, match.vlm_obj, game
                    )
                    match_detail['semantic_similarity'] = float(semantic_score)
            else:
                match_detail['vlm_detection'] = None

            frame_comparison['matches'].append(match_detail)

        comparison_data.append(frame_comparison)

        print(f"  Frame {frame_num}: "
              f"P={result.precision:.3f}, "
              f"R={result.recall:.3f}, "
              f"F1={result.f1_score:.3f}, "
              f"ImpF1={result.important_f1:.3f}")

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

        output_file = output_dir / "evaluation_summary.json"
        with open(output_file, 'w') as f:
            json.dump(aggregate, f, indent=2)

        detailed_file = output_dir / "detailed_frame_results.json"
        with open(detailed_file, 'w') as f:
            json.dump(detailed_results, f, indent=2)

        # NEW: Save detailed comparison file
        comparison_file = output_dir / "matching_comparison.json"
        with open(comparison_file, 'w') as f:
            json.dump(comparison_data, f, indent=2)

        print(f"\n✅ Evaluation complete!")
        print(f"  Main summary: {output_file}")
        print(f"  Detailed results: {detailed_file}")
        print(f"  Matching comparison: {comparison_file}")
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
        description="V3 Detection Only: Fast coordinate evaluation (no action reasoning)"
    )

    parser.add_argument('--game', type=str, required=True,
                       choices=['pong', 'breakout', 'spaceinvaders', 'space_invaders'])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_frames', type=int, default=300)
    parser.add_argument('--random_agent', action='store_true', required=True)
    parser.add_argument('--provider', type=str, default='openrouter',
                       choices=['openrouter', 'bedrock'])
    parser.add_argument('--model', type=str, default='anthropic/claude-sonnet-4')
    parser.add_argument('--sample_every', type=int, default=10,
                       help='Sample frames every N frames (default: 10)')
    parser.add_argument('--frames', type=str, default=None,
                       help='Specific frames to evaluate (comma-separated, e.g., "0,10,20")')
    parser.add_argument('--max_eval_frames', type=int, default=None,
                       help='Maximum number of frames to evaluate (limits total evaluations)')
    parser.add_argument('--start_frame', type=int, default=0,
                       help='Skip first N frames (e.g., 10 to skip initialization). Saved frames will be renumbered from 0.')
    parser.add_argument('--save_all_frames', action='store_true')
    parser.add_argument('--create_visualizations', action='store_true')
    parser.add_argument('--output_dir', type=str, default=None)

    args = parser.parse_args()

    # Create centralized results directory structure
    results_base = Path("evaluation_results")
    results_base.mkdir(exist_ok=True)

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        # Organize: evaluation_results/<provider>_<model>/<game>_seed<seed>
        model_safe = args.model.replace('/', '_').replace(':', '_')
        provider_model_dir = results_base / f"{args.provider}_{model_safe}"
        provider_model_dir.mkdir(parents=True, exist_ok=True)

        output_dir = provider_model_dir / f"{args.game}_seed{args.seed}"

    output_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = output_dir / "frames"

    # Generate ground truth
    ground_truth_data, actions = generate_random_agent_ground_truth_v3(
        args.game, args.seed, args.num_frames,
        save_all_frames=args.save_all_frames or args.create_visualizations,
        frames_dir=frames_dir,
        start_frame=args.start_frame
    )

    # Save ground truth
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
        # User specified exact frames
        frames_to_eval = [int(f.strip()) for f in args.frames.split(',')]
    else:
        # Sample frames evenly
        frames_to_eval = list(range(0, args.num_frames, args.sample_every))

    # Apply max_eval_frames limit if specified
    if args.max_eval_frames and len(frames_to_eval) > args.max_eval_frames:
        frames_to_eval = frames_to_eval[:args.max_eval_frames]
        print(f"\n⚠️  Limited to first {args.max_eval_frames} frames (--max_eval_frames)")

    print(f"\nWill evaluate {len(frames_to_eval)} frames: {frames_to_eval[:5]}{'...' if len(frames_to_eval) > 5 else ''}")

    # Run DETECTION ONLY (no action reasoning)
    detections = run_detection_only(
        args.game, frames_to_eval,
        args.provider, args.model, output_dir, frames_dir
    )

    # Create visualizations
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

    # Save lightweight summary to centralized index
    if result:
        summary_index = results_base / "all_results_index.json"
        index_data = []
        if summary_index.exists():
            with open(summary_index, 'r') as f:
                index_data = json.load(f)

        # Add/update this run
        run_summary = {
            'provider': args.provider,
            'model': args.model,
            'game': args.game,
            'seed': args.seed,
            'frames_evaluated': len(frames_to_eval),
            'precision': result['avg_precision'],
            'recall': result['avg_recall'],
            'f1': result['avg_f1'],
            'important_f1': result['avg_important_f1'],
            'iou': result['avg_iou'],
            'center_distance': result['avg_center_distance'],
            'output_dir': str(output_dir),
            'timestamp': str(Path(output_dir / "evaluation_summary.json").stat().st_mtime)
        }

        # Remove any duplicate entry for same provider/model/game/seed
        index_data = [x for x in index_data if not (
            x['provider'] == args.provider and
            x['model'] == args.model and
            x['game'] == args.game and
            x['seed'] == args.seed
        )]
        index_data.append(run_summary)

        with open(summary_index, 'w') as f:
            json.dump(index_data, f, indent=2)

        print(f"✅ Added to centralized index: {summary_index}")

    print(f"\n{'='*80}")
    print(f"✅ V3 DETECTION-ONLY EVALUATION COMPLETE")
    print(f"{'='*80}")
    print(f"Output directory: {output_dir}")
    print(f"  - All frames: {frames_dir}")
    print(f"  - Visualizations: */")
    print(f"  - Results: evaluation_summary.json")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
