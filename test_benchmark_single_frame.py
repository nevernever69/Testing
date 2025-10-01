#!/usr/bin/env python3
"""
Quick test of benchmark system with a single frame.
Use this to verify everything works before running the full benchmark.

Usage:
    python test_benchmark_single_frame.py --provider bedrock --openrouter_key YOUR_KEY
"""

import argparse
import json
from pathlib import Path

from automatic_benchmark import BenchmarkDatasetLoader, AutomaticEvaluator
from automatic_benchmark.pipeline_adapters import DirectFrameAdapter, AdvancedGameAdapter
from automatic_benchmark.utils import get_all_prompts
from automatic_benchmark.utils.detailed_logger import DetailedBenchmarkLogger
from automatic_benchmark.utils.coordinate_scaler import scale_ground_truth_coordinates


def test_single_frame(
    dataset_path: str,
    frame_id: str = None,
    provider: str = 'bedrock',
    openrouter_key: str = None,
    aws_region: str = 'us-east-1',
    use_llm_judge: bool = False,
    force_llm_judge: bool = False,
    llm_judge_only: bool = False
):
    """
    Test benchmark with a single frame.

    Args:
        dataset_path: Path to benchmark dataset
        frame_id: Specific frame to test (or None for first frame)
        provider: VLM provider
        openrouter_key: OpenRouter API key for detection
        aws_region: AWS region for Bedrock
        use_llm_judge: Enable LLM-as-judge scoring
        force_llm_judge: Force LLM judge on ALL evaluations
        llm_judge_only: Use ONLY LLM judge (disable rule-based and semantic)
    """
    print(f"\n{'='*70}")
    print(f"Testing Benchmark System with Single Frame")
    print(f"{'='*70}\n")

    # Initialize detailed logger
    logger = DetailedBenchmarkLogger("./benchmark_logs")
    print(f"📝 Detailed logging enabled: {logger.get_session_dir()}\n")

    # Load dataset
    print("📂 Loading dataset...")
    loader = BenchmarkDatasetLoader(dataset_path)
    dataset = loader.load()

    # Get first frame if not specified
    if frame_id is None:
        frame_id = dataset['frames'][0]['frame_id']

    print(f"✅ Dataset loaded: {len(dataset['frames'])} frames")
    print(f"🎯 Testing with frame: {frame_id}\n")

    # Load frame and ground truth
    frame = loader.load_frame(frame_id)
    ground_truth = loader.load_ground_truth(frame_id)

    if frame is None or ground_truth is None:
        print(f"❌ Error: Could not load frame {frame_id}")
        return

    game = ground_truth['game']
    print(f"Game: {game}")
    print(f"Frame shape: {frame.shape}")
    print(f"Complexity: {ground_truth.get('complexity', {}).get('complexity_category', 'unknown')}")
    print(f"Objects: {len(ground_truth.get('ocatari_data', {}).get('objects', []))}\n")

    # Scale frame and coordinates to match VLM input (160x210 -> 1280x720)
    print("📐 Scaling frame and coordinates from 160x210 to 1280x720...")
    import cv2
    frame_scaled = cv2.resize(frame, (1280, 720), interpolation=cv2.INTER_LINEAR)
    ground_truth = scale_ground_truth_coordinates(ground_truth)
    print("✅ Frame and coordinates scaled\n")

    # Initialize evaluator
    print("🧠 Initializing evaluator...")
    evaluator = AutomaticEvaluator(
        use_semantic=True,
        use_llm_judge=use_llm_judge,
        force_llm_judge=force_llm_judge,
        llm_judge_only=llm_judge_only,
        llm_provider='bedrock' if use_llm_judge or llm_judge_only else None
    )

    # Debug info
    if llm_judge_only:
        print(f"✅ Evaluator ready (LLM judge ONLY mode - no rule-based/semantic)")
        if evaluator.llm_judge:
            print(f"   LLM Judge available: {evaluator.llm_judge.available}")
            print(f"   LLM Judge model: {evaluator.llm_judge.model}")
        else:
            print(f"   ⚠️  LLM Judge is None!")
        print()
    else:
        print(f"✅ Evaluator ready (LLM judge: {use_llm_judge})\n")

    # Get prompts
    prompts = get_all_prompts()

    # Test Vision-Only Pipeline
    print(f"\n{'='*70}")
    print(f"Testing VISION-ONLY Pipeline")
    print(f"{'='*70}\n")

    try:
        vision_only = DirectFrameAdapter(
            provider=provider,
            model_id=None,
            aws_region=aws_region
        )

        vision_only_results = {}

        for task_type, prompt in prompts.items():
            print(f"📝 Task: {task_type}")
            print(f"   Prompt: {prompt[:80]}...")

            # Get VLM response
            response = vision_only.process(frame_scaled, prompt)
            print(f"   Response: {response[:120]}...")

            # Evaluate
            eval_result = evaluator.evaluate(
                response=response,
                task_type=task_type,
                ground_truth=ground_truth,
                game_name=game
            )

            print(f"   Score: {eval_result.final_score:.3f} (confidence: {eval_result.confidence:.3f})")

            # Only show breakdown if not in LLM judge only mode
            if eval_result.rule_based_result and eval_result.semantic_result:
                print(f"   Breakdown: rule={eval_result.rule_based_result['score']:.3f}, semantic={eval_result.semantic_result['score']:.3f}")

            if eval_result.llm_judge_result:
                print(f"   LLM Judge: {eval_result.llm_judge_result['score']:.3f}")

            if eval_result.two_tier_result:
                tt = eval_result.two_tier_result
                print(f"   Two-Tier: core={tt['core_score']:.3f} ({tt['breakdown']['core_detected']}/{tt['breakdown']['core_objects']}), "
                      f"secondary={tt['secondary_score']:.3f} ({tt['breakdown']['secondary_detected']}/{tt['breakdown']['secondary_objects']}), "
                      f"final={tt['final_score']:.3f}")
            print()

            # Log detailed evaluation
            llm_judge_prompt = None
            llm_judge_response = None
            if eval_result.llm_judge_result and 'details' in eval_result.llm_judge_result:
                llm_judge_prompt = eval_result.llm_judge_result['details'].get('judge_prompt')
                judge_raw = eval_result.llm_judge_result['details'].get('judge_raw_response')
                if judge_raw:
                    llm_judge_response = json.dumps(judge_raw, indent=2)

            # Get actual prompt that was sent to VLM (may include image + text)
            actual_prompt = vision_only.get_actual_prompt()

            logger.log_evaluation(
                frame_id=frame_id,
                pipeline="Vision-Only",
                task_type=task_type,
                task_prompt=actual_prompt,  # Use actual prompt instead of original
                vlm_response=response,
                ground_truth=ground_truth,
                eval_result=eval_result,
                llm_judge_prompt=llm_judge_prompt,
                llm_judge_response=llm_judge_response,
                frame=frame_scaled,  # Use scaled frame
                detection_results=None  # No detection for vision-only
            )

            vision_only_results[task_type] = {
                'response': response,
                'score': eval_result.final_score,
                'confidence': eval_result.confidence,
                'reasoning': eval_result.reasoning
            }

        # Summary
        avg_score = sum(r['score'] for r in vision_only_results.values()) / len(vision_only_results)
        print(f"✅ Vision-Only Average: {avg_score:.3f}\n")

    except Exception as e:
        print(f"❌ Error testing vision-only: {e}")
        import traceback
        traceback.print_exc()

    # Test Vision+Symbol Pipeline (works with Bedrock or OpenRouter)
    print(f"\n{'='*70}")
    print(f"Testing VISION+SYMBOL Pipeline")
    print(f"{'='*70}\n")

    try:
        # Use appropriate detection model based on provider
        if provider == 'bedrock':
            detection_model = 'claude-4-sonnet'  # Bedrock Claude 4 Sonnet
        else:
            detection_model = openrouter_key and 'anthropic/claude-sonnet-4' or None

        vision_symbol = AdvancedGameAdapter(
            provider=provider,
            model_id=None,
            openrouter_api_key=openrouter_key,
            detection_model=detection_model,
            aws_region=aws_region,
            game_type=game.lower()
        )

        vision_symbol_results = {}

        for task_type, prompt in prompts.items():
            print(f"📝 Task: {task_type}")
            print(f"   Prompt: {prompt[:80]}...")

            # Get VLM response (with symbolic info)
            response = vision_symbol.process(frame_scaled, prompt)
            print(f"   Response: {response[:120]}...")

            # Evaluate
            eval_result = evaluator.evaluate(
                response=response,
                task_type=task_type,
                ground_truth=ground_truth,
                game_name=game
            )

            print(f"   Score: {eval_result.final_score:.3f} (confidence: {eval_result.confidence:.3f})")

            # Only show breakdown if not in LLM judge only mode
            if eval_result.rule_based_result and eval_result.semantic_result:
                print(f"   Breakdown: rule={eval_result.rule_based_result['score']:.3f}, semantic={eval_result.semantic_result['score']:.3f}")

            if eval_result.llm_judge_result:
                print(f"   LLM Judge: {eval_result.llm_judge_result['score']:.3f}")

            if eval_result.two_tier_result:
                tt = eval_result.two_tier_result
                print(f"   Two-Tier: core={tt['core_score']:.3f} ({tt['breakdown']['core_detected']}/{tt['breakdown']['core_objects']}), "
                      f"secondary={tt['secondary_score']:.3f} ({tt['breakdown']['secondary_detected']}/{tt['breakdown']['secondary_objects']}), "
                      f"final={tt['final_score']:.3f}")
            print()

            # Log detailed evaluation
            llm_judge_prompt = None
            llm_judge_response = None
            if eval_result.llm_judge_result and 'details' in eval_result.llm_judge_result:
                llm_judge_prompt = eval_result.llm_judge_result['details'].get('judge_prompt')
                judge_raw = eval_result.llm_judge_result['details'].get('judge_raw_response')
                if judge_raw:
                    llm_judge_response = json.dumps(judge_raw, indent=2)

            # Get actual prompt with symbolic information
            actual_prompt = vision_symbol.get_actual_prompt()
            detection_results = vision_symbol.get_detection_results()

            logger.log_evaluation(
                frame_id=frame_id,
                pipeline="Vision+Symbol",
                task_type=task_type,
                task_prompt=actual_prompt,  # Use actual prompt with symbolic info
                vlm_response=response,
                ground_truth=ground_truth,
                eval_result=eval_result,
                llm_judge_prompt=llm_judge_prompt,
                llm_judge_response=llm_judge_response,
                frame=frame_scaled,  # Use scaled frame
                detection_results=detection_results  # Pass detection results for annotation
            )

            vision_symbol_results[task_type] = {
                'response': response,
                'score': eval_result.final_score,
                'confidence': eval_result.confidence,
                'reasoning': eval_result.reasoning
            }

        # Summary
        avg_score = sum(r['score'] for r in vision_symbol_results.values()) / len(vision_symbol_results)
        print(f"✅ Vision+Symbol Average: {avg_score:.3f}\n")

        # Comparison
        print(f"\n{'='*70}")
        print(f"COMPARISON")
        print(f"{'='*70}\n")

        for task_type in prompts.keys():
            vo_score = vision_only_results[task_type]['score']
            vs_score = vision_symbol_results[task_type]['score']
            improvement = vs_score - vo_score
            improvement_pct = (improvement / vo_score * 100) if vo_score > 0 else 0

            print(f"{task_type:15}: {vo_score:.3f} → {vs_score:.3f} ({improvement_pct:+.1f}%)")

        vo_avg = sum(r['score'] for r in vision_only_results.values()) / len(vision_only_results)
        vs_avg = sum(r['score'] for r in vision_symbol_results.values()) / len(vision_symbol_results)
        total_improvement = vs_avg - vo_avg
        total_improvement_pct = (total_improvement / vo_avg * 100) if vo_avg > 0 else 0

        print(f"\n{'Overall':15}: {vo_avg:.3f} → {vs_avg:.3f} ({total_improvement_pct:+.1f}%)")

    except Exception as e:
        print(f"❌ Error testing vision+symbol: {e}")
        import traceback
        traceback.print_exc()

    # Evaluator stats
    print(f"\n{'='*70}")
    print(f"Evaluator Statistics")
    print(f"{'='*70}\n")

    stats = evaluator.get_statistics()
    print(f"Total evaluations: {stats['total_evaluations']}")
    print(f"LLM judge usage: {stats['llm_judge_usage_rate']:.1%}")
    print(f"Estimated cost: ${stats['llm_judge_cost_estimate']:.2f}")

    # Save detailed logs
    logger.save_summary()

    print(f"\n{'='*70}")
    print(f"✅ Test Complete!")
    print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(description='Test benchmark with single frame')
    parser.add_argument('--dataset', type=str, default='./benchmark_v2.1',
                       help='Path to benchmark dataset')
    parser.add_argument('--frame_id', type=str, default=None,
                       help='Specific frame to test (default: first frame)')
    parser.add_argument('--provider', type=str, default='bedrock',
                       help='VLM provider (bedrock, openai, anthropic)')
    parser.add_argument('--openrouter_key', type=str, default=None,
                       help='OpenRouter API key for detection')
    parser.add_argument('--aws_region', type=str, default='us-east-1',
                       help='AWS region for Bedrock')
    parser.add_argument('--use_llm_judge', action='store_true',
                       help='Enable LLM-as-judge scoring')
    parser.add_argument('--force_llm_judge', action='store_true',
                       help='Force LLM judge on ALL evaluations (expensive, use for detailed analysis)')
    parser.add_argument('--llm_judge_only', action='store_true',
                       help='Use ONLY LLM judge (disable rule-based and semantic scoring)')

    args = parser.parse_args()

    test_single_frame(
        dataset_path=args.dataset,
        frame_id=args.frame_id,
        provider=args.provider,
        openrouter_key=args.openrouter_key,
        aws_region=args.aws_region,
        use_llm_judge=args.use_llm_judge,
        force_llm_judge=args.force_llm_judge,
        llm_judge_only=args.llm_judge_only
    )


if __name__ == "__main__":
    main()
