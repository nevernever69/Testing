"""
Detailed logger for benchmark evaluation.
Saves prompts, responses, LLM judge reasoning, and scores.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional


class DetailedBenchmarkLogger:
    """
    Logs detailed benchmark evaluation data including:
    - Task prompts
    - VLM responses
    - Ground truth references
    - Scoring breakdowns
    - LLM judge reasoning
    """

    def __init__(self, output_dir: str):
        """
        Initialize logger.

        Args:
            output_dir: Directory to save logs
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Session ID for this benchmark run
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = self.output_dir / self.session_id
        self.session_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        self.prompts_dir = self.session_dir / "prompts"
        self.responses_dir = self.session_dir / "responses"
        self.evaluations_dir = self.session_dir / "evaluations"
        self.frames_dir = self.session_dir / "frames"
        self.annotated_frames_dir = self.session_dir / "annotated_frames"

        for d in [self.prompts_dir, self.responses_dir, self.evaluations_dir,
                  self.frames_dir, self.annotated_frames_dir]:
            d.mkdir(parents=True, exist_ok=True)

        # Summary data
        self.summary = {
            'session_id': self.session_id,
            'timestamp': datetime.now().isoformat(),
            'evaluations': []
        }

    def log_evaluation(
        self,
        frame_id: str,
        pipeline: str,
        task_type: str,
        task_prompt: str,
        vlm_response: str,
        ground_truth: Dict[str, Any],
        eval_result: Any,
        llm_judge_prompt: Optional[str] = None,
        llm_judge_response: Optional[str] = None,
        frame: Optional[Any] = None,
        detection_results: Optional[Dict[str, Any]] = None
    ):
        """
        Log a complete evaluation.

        Args:
            frame_id: Frame identifier
            pipeline: Pipeline name (Vision-Only or Vision+Symbol)
            task_type: Task type (visual, spatial, strategy, identification)
            task_prompt: The prompt sent to VLM
            vlm_response: VLM's response
            ground_truth: Ground truth data
            eval_result: Evaluation result object
            llm_judge_prompt: Prompt sent to LLM judge (if used)
            llm_judge_response: Raw response from LLM judge
            frame: Original frame image (numpy array)
            detection_results: Detection results with bounding boxes (for Vision+Symbol)
        """
        # Create unique identifier
        eval_id = f"{frame_id}_{pipeline}_{task_type}".replace(" ", "_").replace("+", "")

        # Save task prompt
        prompt_file = self.prompts_dir / f"{eval_id}_task_prompt.txt"
        with open(prompt_file, 'w') as f:
            f.write(f"Frame: {frame_id}\n")
            f.write(f"Pipeline: {pipeline}\n")
            f.write(f"Task: {task_type}\n")
            f.write(f"\n{'='*70}\n")
            f.write(f"TASK PROMPT SENT TO VLM:\n")
            f.write(f"{'='*70}\n\n")
            f.write(task_prompt)

        # Save VLM response
        response_file = self.responses_dir / f"{eval_id}_vlm_response.txt"
        with open(response_file, 'w') as f:
            f.write(f"Frame: {frame_id}\n")
            f.write(f"Pipeline: {pipeline}\n")
            f.write(f"Task: {task_type}\n")
            f.write(f"\n{'='*70}\n")
            f.write(f"VLM RESPONSE:\n")
            f.write(f"{'='*70}\n\n")
            f.write(vlm_response)

        # Save LLM judge prompt and response if available
        if llm_judge_prompt:
            judge_prompt_file = self.prompts_dir / f"{eval_id}_llm_judge_prompt.txt"
            with open(judge_prompt_file, 'w') as f:
                f.write(f"Frame: {frame_id}\n")
                f.write(f"Pipeline: {pipeline}\n")
                f.write(f"Task: {task_type}\n")
                f.write(f"\n{'='*70}\n")
                f.write(f"PROMPT SENT TO LLM JUDGE:\n")
                f.write(f"{'='*70}\n\n")
                f.write(llm_judge_prompt)

        if llm_judge_response:
            judge_response_file = self.responses_dir / f"{eval_id}_llm_judge_response.txt"
            with open(judge_response_file, 'w') as f:
                f.write(f"Frame: {frame_id}\n")
                f.write(f"Pipeline: {pipeline}\n")
                f.write(f"Task: {task_type}\n")
                f.write(f"\n{'='*70}\n")
                f.write(f"LLM JUDGE RAW RESPONSE:\n")
                f.write(f"{'='*70}\n\n")
                f.write(llm_judge_response)

        # Save complete evaluation with ground truth and scoring
        eval_file = self.evaluations_dir / f"{eval_id}_complete_evaluation.json"

        # Extract reference answers
        ref_answers = ground_truth.get('reference_answers', {}).get(task_type, {})
        if isinstance(ref_answers, dict):
            ref_qualitative = ref_answers.get('qualitative', 'N/A')
            ref_quantitative = ref_answers.get('quantitative', 'N/A')
        else:
            ref_qualitative = 'N/A'
            ref_quantitative = ref_answers

        evaluation_data = {
            'metadata': {
                'frame_id': frame_id,
                'pipeline': pipeline,
                'task_type': task_type,
                'game': ground_truth.get('game', 'Unknown'),
                'timestamp': datetime.now().isoformat()
            },
            'task_prompt': task_prompt,
            'vlm_response': vlm_response,
            'ground_truth': {
                'objects': ground_truth.get('ocatari_data', {}).get('objects', []),
                'reference_answer_qualitative': ref_qualitative,
                'reference_answer_quantitative': ref_quantitative
            },
            'scoring': {
                'final_score': eval_result.final_score,
                'confidence': eval_result.confidence,
                'rule_based': {
                    'score': eval_result.rule_based_result.get('score', 0.0) if eval_result.rule_based_result else None,
                    'reasoning': eval_result.rule_based_result.get('reasoning', '') if eval_result.rule_based_result else None
                } if eval_result.rule_based_result else None,
                'semantic': {
                    'score': eval_result.semantic_result.get('score', 0.0) if eval_result.semantic_result else None,
                    'reasoning': eval_result.semantic_result.get('reasoning', '') if eval_result.semantic_result else None
                } if eval_result.semantic_result else None,
                'llm_judge': {
                    'score': eval_result.llm_judge_result.get('score', None) if eval_result.llm_judge_result else None,
                    'reasoning': eval_result.llm_judge_result.get('reasoning', '') if eval_result.llm_judge_result else '',
                    'identified_issues': eval_result.llm_judge_result.get('details', {}).get('identified_issues', []) if eval_result.llm_judge_result else [],
                    'strengths': eval_result.llm_judge_result.get('details', {}).get('strengths', []) if eval_result.llm_judge_result else []
                } if eval_result.llm_judge_result else None,
                'two_tier': eval_result.two_tier_result if eval_result.two_tier_result else None
            },
            'reasoning': eval_result.reasoning
        }

        with open(eval_file, 'w') as f:
            json.dump(evaluation_data, f, indent=2)

        # Save frame images if provided
        if frame is not None:
            self._save_frame_image(frame_id, pipeline, task_type, frame, detection_results)

        # Add to summary
        self.summary['evaluations'].append({
            'eval_id': eval_id,
            'frame_id': frame_id,
            'pipeline': pipeline,
            'task_type': task_type,
            'final_score': eval_result.final_score,
            'llm_judge_score': eval_result.llm_judge_result.get('score', None) if eval_result.llm_judge_result else None
        })

    def save_summary(self):
        """Save session summary."""
        summary_file = self.session_dir / "session_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(self.summary, f, indent=2)

        # Also create a readable report
        report_file = self.session_dir / "session_report.txt"
        with open(report_file, 'w') as f:
            f.write(f"{'='*70}\n")
            f.write(f"BENCHMARK EVALUATION SESSION REPORT\n")
            f.write(f"{'='*70}\n\n")
            f.write(f"Session ID: {self.session_id}\n")
            f.write(f"Timestamp: {self.summary['timestamp']}\n")
            f.write(f"Total Evaluations: {len(self.summary['evaluations'])}\n\n")

            # Group by pipeline
            pipelines = {}
            for eval_data in self.summary['evaluations']:
                pipeline = eval_data['pipeline']
                if pipeline not in pipelines:
                    pipelines[pipeline] = []
                pipelines[pipeline].append(eval_data)

            for pipeline, evals in pipelines.items():
                f.write(f"\n{'='*70}\n")
                f.write(f"{pipeline} Pipeline\n")
                f.write(f"{'='*70}\n\n")

                avg_score = sum(e['final_score'] for e in evals) / len(evals)
                f.write(f"Average Score: {avg_score:.3f}\n\n")

                for eval_data in evals:
                    f.write(f"  {eval_data['task_type']:15} | Score: {eval_data['final_score']:.3f}")
                    if eval_data['llm_judge_score'] is not None:
                        f.write(f" | LLM Judge: {eval_data['llm_judge_score']:.3f}")
                    f.write(f"\n")

        print(f"\n✅ Detailed logs saved to: {self.session_dir}")
        print(f"   - Prompts: {self.prompts_dir}")
        print(f"   - Responses: {self.responses_dir}")
        print(f"   - Evaluations: {self.evaluations_dir}")
        print(f"   - Frames: {self.frames_dir}")
        print(f"   - Annotated Frames: {self.annotated_frames_dir}")
        print(f"   - Summary: {summary_file}")

    def _save_frame_image(
        self,
        frame_id: str,
        pipeline: str,
        task_type: str,
        frame: Any,
        detection_results: Optional[Dict[str, Any]] = None
    ):
        """
        Save frame image and optionally annotated frame with bounding boxes.

        Args:
            frame_id: Frame identifier
            pipeline: Pipeline name
            task_type: Task type
            frame: Frame image (numpy array)
            detection_results: Detection results with bounding boxes
        """
        import cv2
        import numpy as np
        from PIL import Image

        eval_id = f"{frame_id}_{pipeline}_{task_type}".replace(" ", "_").replace("+", "")

        # Save original frame
        frame_file = self.frames_dir / f"{eval_id}.png"
        if isinstance(frame, np.ndarray):
            Image.fromarray(frame).save(frame_file)

        # For Vision+Symbol, save annotated frame with bounding boxes
        if detection_results and "Vision+Symbol" in pipeline:
            annotated_frame = frame.copy()
            objects = detection_results.get('objects', [])

            # Draw bounding boxes
            for obj in objects:
                label = obj.get('label', 'Unknown')
                coords = obj.get('coordinates', [])

                if len(coords) == 4:
                    x1, y1, x2, y2 = map(int, coords)

                    # Draw rectangle
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    # Draw label with background
                    label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    cv2.rectangle(annotated_frame, (x1, y1 - label_size[1] - 5),
                                  (x1 + label_size[0], y1), (0, 255, 0), -1)
                    cv2.putText(annotated_frame, label, (x1, y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

            # Save annotated frame
            annotated_file = self.annotated_frames_dir / f"{eval_id}_annotated.png"
            Image.fromarray(annotated_frame).save(annotated_file)

    def get_session_dir(self) -> Path:
        """Get the session directory path."""
        return self.session_dir
