#!/usr/bin/env python3
"""
Detection Accuracy Comparison Tool

Compares our object detection coordinates with OCatari ground truth
to measure spatial reasoning accuracy.
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict
import sys


class DetectionAccuracyAnalyzer:
    def __init__(self, results_dir, position_tolerance=20):
        """
        Initialize detection accuracy analyzer.

        Args:
            results_dir: Path to results directory with detections
            position_tolerance: Maximum pixel distance to consider a match (default: 20)
        """
        self.results_dir = Path(results_dir)
        self.position_tolerance = position_tolerance
        self.detections_dir = self.results_dir / 'Results' / 'detections'

    def load_step_data(self, step_number):
        """Load both our detection and OCatari ground truth for a step"""
        step_dir = self.detections_dir / f'step_{step_number:04d}'

        if not step_dir.exists():
            return None, None

        # Load our detection results
        our_detection = None
        detection_file = step_dir / 'detections.json'
        if detection_file.exists():
            with open(detection_file, 'r') as f:
                our_detection = json.load(f)

        # Load OCatari ground truth
        ocatari_ground_truth = None
        ocatari_file = step_dir / 'ocatari_ground_truth.json'
        if ocatari_file.exists():
            with open(ocatari_file, 'r') as f:
                ocatari_ground_truth = json.load(f)

        return our_detection, ocatari_ground_truth

    def calculate_position_error(self, pos1, pos2):
        """Calculate Euclidean distance between two positions"""
        if not pos1 or not pos2:
            return float('inf')

        # Handle different position formats
        if isinstance(pos1, dict):
            x1, y1 = pos1.get('x', 0), pos1.get('y', 0)
        elif isinstance(pos1, (list, tuple)) and len(pos1) >= 2:
            x1, y1 = pos1[0], pos1[1]
        else:
            return float('inf')

        if isinstance(pos2, dict):
            x2, y2 = pos2.get('x', 0), pos2.get('y', 0)
        elif isinstance(pos2, (list, tuple)) and len(pos2) >= 2:
            x2, y2 = pos2[0], pos2[1]
        else:
            return float('inf')

        return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)

    def match_objects(self, our_objects, ocatari_objects):
        """
        Match our detected objects with OCatari ground truth objects.

        Returns:
            matches: List of (our_obj, ocatari_obj, distance) tuples
            unmatched_ours: Objects we detected but OCatari didn't
            unmatched_ocatari: Objects OCatari detected but we missed
        """
        matches = []
        used_ocatari = set()
        unmatched_ours = []

        # Try to match each of our detections
        for our_obj in our_objects:
            best_match = None
            best_distance = float('inf')
            best_idx = -1

            our_pos = our_obj.get('position', our_obj.get('bbox', {}).get('center', None))
            if not our_pos and 'x' in our_obj:
                our_pos = {'x': our_obj['x'], 'y': our_obj['y']}

            for idx, ocatari_obj in enumerate(ocatari_objects):
                if idx in used_ocatari:
                    continue

                ocatari_pos = ocatari_obj.get('position')
                if not ocatari_pos:
                    continue

                distance = self.calculate_position_error(our_pos, ocatari_pos)

                if distance < best_distance and distance < self.position_tolerance:
                    best_match = ocatari_obj
                    best_distance = distance
                    best_idx = idx

            if best_match:
                matches.append((our_obj, best_match, best_distance))
                used_ocatari.add(best_idx)
            else:
                unmatched_ours.append(our_obj)

        # Unmatched OCatari objects (we missed these)
        unmatched_ocatari = [obj for idx, obj in enumerate(ocatari_objects)
                            if idx not in used_ocatari]

        return matches, unmatched_ours, unmatched_ocatari

    def analyze_step(self, step_number):
        """Analyze accuracy for a single step"""
        our_detection, ocatari_gt = self.load_step_data(step_number)

        if not our_detection or not ocatari_gt:
            return None

        # Extract object lists
        our_objects = our_detection.get('detected_objects', [])
        if isinstance(our_detection.get('symbolic_state'), dict):
            # Convert symbolic_state format to list
            our_objects = []
            for key, value in our_detection.get('symbolic_state', {}).items():
                if isinstance(value, dict) and ('x' in value or 'position' in value):
                    obj = {'label': key, **value}
                    our_objects.append(obj)
                elif isinstance(value, list):
                    for item in value:
                        if isinstance(item, dict):
                            obj = {'label': key, **item}
                            our_objects.append(obj)

        ocatari_objects = ocatari_gt.get('objects', [])

        # Match objects
        matches, unmatched_ours, unmatched_ocatari = self.match_objects(
            our_objects, ocatari_objects
        )

        # Calculate metrics
        total_gt = len(ocatari_objects)
        total_detected = len(our_objects)
        true_positives = len(matches)
        false_positives = len(unmatched_ours)
        false_negatives = len(unmatched_ocatari)

        precision = true_positives / total_detected if total_detected > 0 else 0
        recall = true_positives / total_gt if total_gt > 0 else 0
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) > 0 else 0)

        # Average position error for matches
        avg_position_error = (np.mean([m[2] for m in matches])
                             if matches else 0)

        return {
            'step': step_number,
            'total_ground_truth': total_gt,
            'total_detected': total_detected,
            'true_positives': true_positives,
            'false_positives': false_positives,
            'false_negatives': false_negatives,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'avg_position_error': avg_position_error,
            'matches': matches,
            'unmatched_ours': unmatched_ours,
            'unmatched_ocatari': unmatched_ocatari
        }

    def analyze_all_steps(self):
        """Analyze all available steps"""
        if not self.detections_dir.exists():
            print(f"Detections directory not found: {self.detections_dir}")
            return []

        results = []
        step_dirs = sorted(self.detections_dir.glob('step_*'))

        print(f"Analyzing {len(step_dirs)} steps...")

        for step_dir in step_dirs:
            step_num = int(step_dir.name.replace('step_', ''))
            result = self.analyze_step(step_num)
            if result:
                results.append(result)

        return results

    def generate_report(self, output_file='detection_accuracy_report.txt'):
        """Generate comprehensive detection accuracy report"""
        results = self.analyze_all_steps()

        if not results:
            print("No results to analyze!")
            return

        # Aggregate statistics
        total_steps = len(results)
        avg_precision = np.mean([r['precision'] for r in results])
        avg_recall = np.mean([r['recall'] for r in results])
        avg_f1 = np.mean([r['f1_score'] for r in results])
        avg_position_error = np.mean([r['avg_position_error'] for r in results if r['avg_position_error'] > 0])

        total_true_positives = sum(r['true_positives'] for r in results)
        total_false_positives = sum(r['false_positives'] for r in results)
        total_false_negatives = sum(r['false_negatives'] for r in results)

        # Generate report
        report = []
        report.append("=" * 100)
        report.append("DETECTION ACCURACY REPORT (vs OCatari Ground Truth)")
        report.append("=" * 100)
        report.append(f"Results Directory: {self.results_dir}")
        report.append(f"Total Steps Analyzed: {total_steps}")
        report.append(f"Position Tolerance: {self.position_tolerance} pixels")
        report.append("")

        report.append("-" * 100)
        report.append("OVERALL ACCURACY METRICS")
        report.append("-" * 100)
        report.append(f"Average Precision:        {avg_precision*100:6.2f}%  (How many of our detections were correct)")
        report.append(f"Average Recall:           {avg_recall*100:6.2f}%  (How many ground truth objects we found)")
        report.append(f"Average F1 Score:         {avg_f1*100:6.2f}%  (Harmonic mean of precision and recall)")
        report.append(f"Average Position Error:   {avg_position_error:6.2f} pixels (For correctly matched objects)")
        report.append("")

        report.append("-" * 100)
        report.append("CONFUSION MATRIX (TOTAL ACROSS ALL FRAMES)")
        report.append("-" * 100)
        report.append(f"True Positives:           {total_true_positives:6d}  (Correctly detected objects)")
        report.append(f"False Positives:          {total_false_positives:6d}  (Incorrectly detected objects)")
        report.append(f"False Negatives:          {total_false_negatives:6d}  (Missed objects)")
        report.append("")

        # Per-step breakdown (first 10 and last 10)
        report.append("-" * 100)
        report.append("SAMPLE PER-STEP BREAKDOWN (First 10 steps)")
        report.append("-" * 100)
        report.append(f"{'Step':<8} {'GT Objs':<10} {'Detected':<10} {'Precision':<12} {'Recall':<12} {'F1':<12} {'Pos Error':<12}")
        report.append("-" * 100)

        for result in results[:10]:
            report.append(
                f"{result['step']:<8} "
                f"{result['total_ground_truth']:<10} "
                f"{result['total_detected']:<10} "
                f"{result['precision']*100:>10.1f}% "
                f"{result['recall']*100:>10.1f}% "
                f"{result['f1_score']*100:>10.1f}% "
                f"{result['avg_position_error']:>10.2f}px"
            )

        if len(results) > 20:
            report.append("...")
            report.append("\nLast 10 steps:")
            report.append("-" * 100)
            for result in results[-10:]:
                report.append(
                    f"{result['step']:<8} "
                    f"{result['total_ground_truth']:<10} "
                    f"{result['total_detected']:<10} "
                    f"{result['precision']*100:>10.1f}% "
                    f"{result['recall']*100:>10.1f}% "
                    f"{result['f1_score']*100:>10.1f}% "
                    f"{result['avg_position_error']:>10.2f}px"
                )

        report.append("")

        # Find best and worst performing steps
        best_step = max(results, key=lambda r: r['f1_score'])
        worst_step = min(results, key=lambda r: r['f1_score'])

        report.append("-" * 100)
        report.append("BEST AND WORST PERFORMING STEPS")
        report.append("-" * 100)
        report.append(f"Best Step: {best_step['step']} - F1: {best_step['f1_score']*100:.1f}%, "
                     f"Precision: {best_step['precision']*100:.1f}%, "
                     f"Recall: {best_step['recall']*100:.1f}%")
        report.append(f"Worst Step: {worst_step['step']} - F1: {worst_step['f1_score']*100:.1f}%, "
                     f"Precision: {worst_step['precision']*100:.1f}%, "
                     f"Recall: {worst_step['recall']*100:.1f}%")
        report.append("")

        # Write report
        report_text = "\n".join(report)
        output_path = self.results_dir / output_file
        with open(output_path, 'w') as f:
            f.write(report_text)

        print(report_text)
        print(f"\n📊 Detection accuracy report saved to: {output_path}")

        return report_text


def main():
    if len(sys.argv) < 2:
        print("Usage: python compare_detection_accuracy.py <results_directory> [position_tolerance]")
        print("\nExample:")
        print("  python compare_detection_accuracy.py ./comparison_results/pong_bedrock_results/vision_symbol_seed456/Pong_bedrock_symbolic_only/")
        print("  python compare_detection_accuracy.py ./results/ 15  # With custom 15px tolerance")
        sys.exit(1)

    results_dir = sys.argv[1]
    tolerance = int(sys.argv[2]) if len(sys.argv) > 2 else 20

    analyzer = DetectionAccuracyAnalyzer(results_dir, position_tolerance=tolerance)
    analyzer.generate_report()


if __name__ == "__main__":
    main()
