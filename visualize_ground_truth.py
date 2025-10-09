"""
Visualize OCAtari Ground Truth with Bounding Boxes

This module draws bounding boxes on OCAtari ground truth frames
to visualize object detections in the same style as VLM detections.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Tuple
from coordinate_accuracy_evaluator import CoordinateScaler


class GroundTruthVisualizer:
    """Visualize OCAtari ground truth with bounding boxes."""

    def __init__(self):
        self.scaler = CoordinateScaler()
        self.colors = [
            (255, 0, 0),      # Red
            (0, 255, 0),      # Green
            (0, 0, 255),      # Blue
            (255, 255, 0),    # Cyan
            (255, 0, 255),    # Magenta
            (0, 255, 255),    # Yellow
            (255, 128, 0),    # Orange
            (128, 0, 255),    # Purple
            (0, 255, 128),    # Spring Green
            (255, 0, 128),    # Rose
        ]

    def draw_ground_truth(
        self,
        frame_bgr: np.ndarray,
        gt_objects: List[Dict[str, Any]],
        output_path: Path,
        title: str = "OCAtari Ground Truth"
    ) -> None:
        """Draw ground truth bounding boxes on frame.

        Args:
            frame_bgr: Frame in BGR format (210x160 or 1280x720)
            gt_objects: List of ground truth objects with scaled coordinates
            output_path: Path to save annotated frame
            title: Title to display on frame
        """
        # Check frame size and scale if needed
        h, w = frame_bgr.shape[:2]
        if w == 160 and h == 210:
            # Scale OCAtari frame to 1280x720
            frame_bgr = cv2.resize(frame_bgr, (1280, 720), interpolation=cv2.INTER_LINEAR)

        annotated = frame_bgr.copy()

        # Draw each object
        for idx, obj in enumerate(gt_objects):
            bbox = obj.get('bbox', [])
            if len(bbox) != 4:
                continue

            x1, y1, x2, y2 = map(int, bbox)
            color = self.colors[idx % len(self.colors)]

            # Draw bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            # Prepare label
            category = obj.get('category', 'Unknown')
            label = f"{category}"

            # Draw label background
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            label_w, label_h = label_size

            # Position label above box
            label_y = max(y1 - 5, label_h + 5)
            cv2.rectangle(
                annotated,
                (x1, label_y - label_h - 5),
                (x1 + label_w + 5, label_y + 5),
                color,
                -1
            )

            # Draw label text
            cv2.putText(
                annotated,
                label,
                (x1 + 2, label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2
            )

        # Add title
        cv2.putText(
            annotated,
            title,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2
        )

        # Add object count
        count_text = f"Objects: {len(gt_objects)}"
        cv2.putText(
            annotated,
            count_text,
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        # Save
        cv2.imwrite(str(output_path), annotated)

    def create_side_by_side(
        self,
        gt_frame_path: Path,
        vlm_frame_path: Path,
        output_path: Path,
        frame_number: int
    ) -> None:
        """Create side-by-side comparison of ground truth and VLM detection.

        Args:
            gt_frame_path: Path to ground truth annotated frame
            vlm_frame_path: Path to VLM annotated frame
            output_path: Path to save comparison
            frame_number: Frame number for title
        """
        # Load both frames
        gt_frame = cv2.imread(str(gt_frame_path))
        vlm_frame = cv2.imread(str(vlm_frame_path))

        if gt_frame is None or vlm_frame is None:
            print(f"Warning: Could not load frames for comparison")
            return

        # Ensure same size
        h, w = gt_frame.shape[:2]
        vlm_frame = cv2.resize(vlm_frame, (w, h))

        # Create side-by-side
        combined = np.hstack([gt_frame, vlm_frame])

        # Add labels
        label_h = 50
        labeled = np.zeros((combined.shape[0] + label_h, combined.shape[1], 3), dtype=np.uint8)
        labeled[label_h:, :] = combined

        # Add text labels
        cv2.putText(
            labeled,
            f"Frame {frame_number}: OCAtari Ground Truth",
            (10, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

        cv2.putText(
            labeled,
            "VLM Detection",
            (w + 10, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

        # Save
        cv2.imwrite(str(output_path), labeled)


def visualize_ground_truth_frames(
    ground_truth_data: List[Dict],
    frames_dir: Path,
    output_dir: Path,
    frames_to_visualize: List[int] = None
) -> None:
    """Visualize ground truth for multiple frames.

    Args:
        ground_truth_data: List of ground truth data for each frame
        frames_dir: Directory containing RGB frames
        output_dir: Output directory for visualizations
        frames_to_visualize: List of frame numbers (None = all)
    """
    from coordinate_accuracy_evaluator import CoordinateAccuracyEvaluator

    output_dir.mkdir(parents=True, exist_ok=True)
    visualizer = GroundTruthVisualizer()

    # Determine which frames to visualize
    if frames_to_visualize is None:
        frames_to_visualize = list(range(len(ground_truth_data)))

    # Prepare ground truth objects
    evaluator = CoordinateAccuracyEvaluator('pong')  # Game doesn't matter for visualization

    for frame_num in frames_to_visualize:
        if frame_num >= len(ground_truth_data):
            continue

        frame_path = frames_dir / f"frame_{frame_num:06d}.png"
        if not frame_path.exists():
            continue

        # Load frame
        frame_bgr = cv2.imread(str(frame_path))
        if frame_bgr is None:
            continue

        # Get ground truth
        gt_data = ground_truth_data[frame_num]
        gt_objects = evaluator.prepare_ocatari_objects(gt_data)

        # Draw and save
        output_path = output_dir / f"frame_{frame_num:06d}_ground_truth.png"
        visualizer.draw_ground_truth(
            frame_bgr,
            gt_objects,
            output_path,
            title=f"OCAtari Ground Truth - Frame {frame_num}"
        )

        print(f"  Visualized frame {frame_num} -> {output_path.name}")


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Visualize OCAtari ground truth")
    parser.add_argument('--ground_truth', type=str, required=True,
                       help='Path to ground_truth.json')
    parser.add_argument('--frames_dir', type=str, required=True,
                       help='Directory containing frames')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for visualizations')
    parser.add_argument('--frames', type=str, default=None,
                       help='Comma-separated list of frame numbers')

    args = parser.parse_args()

    # Load ground truth
    with open(args.ground_truth, 'r') as f:
        data = json.load(f)
        ground_truth_data = data['ground_truth']

    # Parse frames
    frames_to_viz = None
    if args.frames:
        frames_to_viz = [int(f.strip()) for f in args.frames.split(',')]

    # Visualize
    visualize_ground_truth_frames(
        ground_truth_data,
        Path(args.frames_dir),
        Path(args.output_dir),
        frames_to_viz
    )

    print(f"\n✅ Visualization complete!")
