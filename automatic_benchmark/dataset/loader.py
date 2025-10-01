"""
Benchmark dataset loader.
Loads fixed benchmark datasets for evaluation.
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, Optional
from PIL import Image
import numpy as np


class BenchmarkDatasetLoader:
    """
    Loads benchmark dataset for evaluation.
    """

    def __init__(self, dataset_path: str):
        """
        Initialize dataset loader.

        Args:
            dataset_path: Path to dataset directory
        """
        self.dataset_path = Path(dataset_path)

        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {self.dataset_path}")

        self.frames_dir = self.dataset_path / "frames"
        self.ground_truth_dir = self.dataset_path / "ground_truth"
        self.metadata_dir = self.dataset_path / "metadata"

        # Load dataset index
        index_path = self.metadata_dir / "dataset_index.json"
        if not index_path.exists():
            raise FileNotFoundError(f"Dataset index not found: {index_path}")

        with open(index_path, 'r') as f:
            self.dataset_index = json.load(f)

    def load(self) -> Dict[str, Any]:
        """
        Load complete dataset index.

        Returns:
            Dataset dictionary with metadata and frame list
        """
        return self.dataset_index

    def get_metadata(self) -> Dict[str, Any]:
        """Get dataset metadata."""
        return self.dataset_index['metadata']

    def get_frames(self) -> list:
        """Get list of all frames."""
        return self.dataset_index['frames']

    def load_frame(self, frame_id: str) -> Optional[np.ndarray]:
        """
        Load a frame image by ID.

        Args:
            frame_id: Frame identifier

        Returns:
            Frame as numpy array (210, 160, 3) or None if not found
        """
        frame_path = self.frames_dir / f"{frame_id}.png"

        if not frame_path.exists():
            print(f"Warning: Frame not found: {frame_path}")
            return None

        image = Image.open(frame_path)
        return np.array(image)

    def load_ground_truth(self, frame_id: str) -> Optional[Dict[str, Any]]:
        """
        Load ground truth for a frame.

        Args:
            frame_id: Frame identifier

        Returns:
            Ground truth dictionary or None if not found
        """
        gt_path = self.ground_truth_dir / f"{frame_id}.json"

        if not gt_path.exists():
            print(f"Warning: Ground truth not found: {gt_path}")
            return None

        with open(gt_path, 'r') as f:
            return json.load(f)

    def get_frames_by_game(self, game: str) -> list:
        """
        Get all frames for a specific game.

        Args:
            game: Game name

        Returns:
            List of frame metadata dictionaries
        """
        return [f for f in self.dataset_index['frames'] if f['game'] == game]

    def get_frames_by_complexity(self, complexity: str) -> list:
        """
        Get all frames of a specific complexity.

        Args:
            complexity: 'easy', 'medium', or 'hard'

        Returns:
            List of frame metadata dictionaries
        """
        return [f for f in self.dataset_index['frames']
                if f['complexity']['complexity_category'] == complexity]

    def get_frame_data(self, frame_id: str) -> Optional[Dict[str, Any]]:
        """
        Get complete data for a frame (metadata, image, ground truth).

        Args:
            frame_id: Frame identifier

        Returns:
            Dictionary with 'metadata', 'frame', 'ground_truth' or None
        """
        # Find metadata
        metadata = None
        for frame in self.dataset_index['frames']:
            if frame['frame_id'] == frame_id:
                metadata = frame
                break

        if metadata is None:
            return None

        # Load frame and ground truth
        frame = self.load_frame(frame_id)
        ground_truth = self.load_ground_truth(frame_id)

        return {
            'metadata': metadata,
            'frame': frame,
            'ground_truth': ground_truth
        }

    def __len__(self) -> int:
        """Get total number of frames in dataset."""
        return self.dataset_index['metadata']['total_frames']

    def __repr__(self) -> str:
        """String representation of dataset."""
        metadata = self.dataset_index['metadata']
        return (f"BenchmarkDataset(version={metadata['version']}, "
                f"frames={metadata['total_frames']}, "
                f"games={metadata['games']})")
