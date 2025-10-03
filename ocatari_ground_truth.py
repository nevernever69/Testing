"""
OCAtari Ground Truth Extraction Module

This module provides a wrapper around OCAtari for extracting precise object positions
and properties from Atari game RAM states. It serves as the ground truth generator
for spatial reasoning benchmarks.

Usage:
    extractor = OCAtariGroundTruth('Pong')
    frame, objects = extractor.get_frame_and_objects()
    trajectories = extractor.predict_trajectory(object_id, steps=3)
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from ocatari import OCAtari
import json


@dataclass
class ObjectInfo:
    """Standardized object information extracted from OCAtari."""
    id: int
    category: str
    position: Tuple[int, int]  # (x, y)
    velocity: Tuple[float, float]  # (dx, dy)
    size: Tuple[int, int]  # (width, height)
    properties: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'id': self.id,
            'category': self.category,
            'position': self.position,
            'velocity': self.velocity,
            'size': self.size,
            'properties': self.properties
        }


class OCAtariGroundTruth:
    """
    OCAtari wrapper for ground truth extraction and spatial reasoning evaluation.

    Provides standardized interface for:
    - Object position and property extraction
    - Trajectory prediction using physics
    - Collision detection
    - Spatial relationship calculations
    """

    def __init__(self, game_name: str, render_mode: str = 'rgb_array'):
        """
        Initialize OCAtari environment for ground truth extraction.

        Args:
            game_name: Name of the Atari game ('Pong', 'Breakout', 'SpaceInvaders')
            render_mode: Rendering mode ('rgb_array' for frames)
        """
        self.game_name = game_name
        self.env = OCAtari(game_name, mode='ram', render_mode=render_mode)
        self.reset()

        # Track object history for velocity calculation
        self.object_history = []
        self.frame_count = 0

    def reset(self, seed=None, **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment and return initial observation."""
        if seed is not None:
            obs, info = self.env.reset(seed=seed, **kwargs)
        else:
            obs, info = self.env.reset(**kwargs)
        self.object_history = []
        self.frame_count = 0
        return obs, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Take environment step and update object history."""
        obs, reward, done, truncated, info = self.env.step(action)
        self.frame_count += 1

        # Store object positions for velocity calculation
        current_objects = self._extract_objects()
        self.object_history.append({
            'frame': self.frame_count,
            'objects': current_objects
        })

        # Keep only last few frames for velocity calculation
        if len(self.object_history) > 5:
            self.object_history.pop(0)

        return obs, reward, done, truncated, info

    def get_frame_and_objects(self) -> Tuple[np.ndarray, List[ObjectInfo]]:
        """
        Get current frame and extracted object information.

        Returns:
            frame: RGB frame (210, 160, 3)
            objects: List of ObjectInfo with positions, velocities, etc.
        """
        frame = self.env.render()
        objects = self._extract_objects()
        return frame, objects

    def _extract_objects(self) -> List[ObjectInfo]:
        """Extract standardized object information from OCAtari."""
        objects = []

        if not hasattr(self.env, 'objects') or not self.env.objects:
            return objects

        for i, obj in enumerate(self.env.objects):
            # Get basic properties
            position = obj.xy if hasattr(obj, 'xy') else (0, 0)
            size = obj.wh if hasattr(obj, 'wh') else (0, 0)
            category = type(obj).__name__

            # Calculate velocity from history
            velocity = self._calculate_velocity(i, position)

            # Extract additional properties
            properties = {
                'center': obj.center if hasattr(obj, 'center') else position,
                'rgb': obj.rgb if hasattr(obj, 'rgb') else None,
                'orientation': getattr(obj, 'orientation', None),
            }

            # Add game-specific properties
            if hasattr(obj, 'player_num'):
                properties['player_num'] = obj.player_num

            objects.append(ObjectInfo(
                id=i,
                category=category,
                position=position,
                velocity=velocity,
                size=size,
                properties=properties
            ))

        return objects

    def _calculate_velocity(self, object_id: int, current_pos: Tuple[int, int]) -> Tuple[float, float]:
        """Calculate object velocity from position history."""
        if len(self.object_history) < 2:
            return (0.0, 0.0)

        try:
            # Get previous position
            prev_frame = self.object_history[-1]
            if object_id < len(prev_frame['objects']):
                prev_pos = prev_frame['objects'][object_id].position

                # Calculate velocity (pixels per frame)
                dx = current_pos[0] - prev_pos[0]
                dy = current_pos[1] - prev_pos[1]

                return (float(dx), float(dy))
        except (IndexError, KeyError):
            pass

        return (0.0, 0.0)

    def predict_trajectory(self, object_id: int, steps: int = 3) -> List[Tuple[int, int]]:
        """
        Predict object trajectory using current position and velocity.

        Args:
            object_id: ID of object to predict
            steps: Number of future frames to predict

        Returns:
            List of predicted (x, y) positions
        """
        _, objects = self.get_frame_and_objects()

        if object_id >= len(objects):
            return []

        obj = objects[object_id]
        predictions = []

        x, y = obj.position
        dx, dy = obj.velocity

        for step in range(1, steps + 1):
            # Simple linear prediction (can be enhanced with physics)
            pred_x = x + dx * step
            pred_y = y + dy * step

            # Clamp to screen boundaries
            pred_x = max(0, min(159, int(pred_x)))  # OCAtari frame width
            pred_y = max(0, min(209, int(pred_y)))  # OCAtari frame height

            predictions.append((pred_x, pred_y))

        return predictions

    def detect_collisions(self, steps: int = 5) -> List[Dict[str, Any]]:
        """
        Detect potential collisions in the next few frames.

        Args:
            steps: Number of frames to look ahead

        Returns:
            List of collision predictions with timing and positions
        """
        _, objects = self.get_frame_and_objects()
        collisions = []

        for i, obj1 in enumerate(objects):
            for j, obj2 in enumerate(objects):
                if i >= j:  # Avoid duplicates and self-collision
                    continue

                collision = self._predict_collision(obj1, obj2, steps)
                if collision:
                    collisions.append({
                        'object1_id': i,
                        'object2_id': j,
                        'object1_category': obj1.category,
                        'object2_category': obj2.category,
                        'collision_frame': collision['frame'],
                        'collision_position': collision['position'],
                        'confidence': collision['confidence']
                    })

        return collisions

    def _predict_collision(self, obj1: ObjectInfo, obj2: ObjectInfo, steps: int) -> Optional[Dict[str, Any]]:
        """Predict collision between two objects."""
        # Simple bounding box collision detection
        for step in range(1, steps + 1):
            # Predict positions
            x1 = obj1.position[0] + obj1.velocity[0] * step
            y1 = obj1.position[1] + obj1.velocity[1] * step
            x2 = obj2.position[0] + obj2.velocity[0] * step
            y2 = obj2.position[1] + obj2.velocity[1] * step

            # Check bounding box overlap
            w1, h1 = obj1.size if obj1.size != (0, 0) else (4, 4)  # Default size
            w2, h2 = obj2.size if obj2.size != (0, 0) else (4, 4)

            if (abs(x1 - x2) < (w1 + w2) / 2 and
                abs(y1 - y2) < (h1 + h2) / 2):

                return {
                    'frame': step,
                    'position': (int((x1 + x2) / 2), int((y1 + y2) / 2)),
                    'confidence': 1.0 - (step / steps)  # Higher confidence for sooner collisions
                }

        return None

    def calculate_distances(self) -> Dict[Tuple[int, int], float]:
        """
        Calculate distances between all object pairs.

        Returns:
            Dictionary mapping (obj1_id, obj2_id) to distance in pixels
        """
        _, objects = self.get_frame_and_objects()
        distances = {}

        for i, obj1 in enumerate(objects):
            for j, obj2 in enumerate(objects):
                if i != j:
                    dist = np.sqrt(
                        (obj1.position[0] - obj2.position[0])**2 +
                        (obj1.position[1] - obj2.position[1])**2
                    )
                    distances[(i, j)] = float(dist)

        return distances

    def get_spatial_relationships(self) -> Dict[str, Any]:
        """
        Get comprehensive spatial relationship information.

        Returns:
            Dictionary with distances, relative positions, and orderings
        """
        _, objects = self.get_frame_and_objects()

        if not objects:
            return {}

        distances = self.calculate_distances()
        # Convert tuple keys to strings for JSON serialization
        distances_serializable = {f"{k[0]}_{k[1]}": v for k, v in distances.items()}

        relationships = {
            'distances': distances_serializable,
            'object_positions': [obj.position for obj in objects],
            'object_categories': [obj.category for obj in objects],
            'leftmost_object': min(range(len(objects)), key=lambda i: objects[i].position[0]),
            'rightmost_object': max(range(len(objects)), key=lambda i: objects[i].position[0]),
            'topmost_object': min(range(len(objects)), key=lambda i: objects[i].position[1]),
            'bottommost_object': max(range(len(objects)), key=lambda i: objects[i].position[1]),
        }

        return relationships

    def export_frame_data(self, filename: str):
        """Export current frame data to JSON for analysis."""
        frame, objects = self.get_frame_and_objects()

        data = {
            'game': self.game_name,
            'frame_count': self.frame_count,
            'frame_shape': frame.shape,
            'objects': [obj.to_dict() for obj in objects],
            'spatial_relationships': self.get_spatial_relationships(),
            'predicted_collisions': self.detect_collisions()
        }

        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)

    def close(self):
        """Close the OCAtari environment."""
        self.env.close()


# Utility functions for common spatial reasoning tasks

def calculate_trajectory_error(predicted: List[Tuple[int, int]],
                             actual: List[Tuple[int, int]]) -> float:
    """Calculate L2 distance error between predicted and actual trajectories."""
    if len(predicted) != len(actual):
        return float('inf')

    total_error = 0.0
    for pred, act in zip(predicted, actual):
        error = np.sqrt((pred[0] - act[0])**2 + (pred[1] - act[1])**2)
        total_error += error

    return total_error / len(predicted)


def calculate_collision_accuracy(predicted_collisions: List[Dict[str, Any]],
                               actual_collisions: List[Dict[str, Any]],
                               frame_tolerance: int = 2,
                               position_tolerance: int = 10) -> Dict[str, float]:
    """Calculate collision detection accuracy metrics."""
    if not actual_collisions:
        return {'precision': 1.0 if not predicted_collisions else 0.0,
                'recall': 1.0,
                'f1': 1.0 if not predicted_collisions else 0.0}

    true_positives = 0

    for pred in predicted_collisions:
        for actual in actual_collisions:
            frame_match = abs(pred['collision_frame'] - actual['collision_frame']) <= frame_tolerance
            pos_match = (abs(pred['collision_position'][0] - actual['collision_position'][0]) <= position_tolerance and
                        abs(pred['collision_position'][1] - actual['collision_position'][1]) <= position_tolerance)

            if frame_match and pos_match:
                true_positives += 1
                break

    precision = true_positives / len(predicted_collisions) if predicted_collisions else 0.0
    recall = true_positives / len(actual_collisions) if actual_collisions else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {'precision': precision, 'recall': recall, 'f1': f1}


if __name__ == "__main__":
    # Test the ground truth extractor
    extractor = OCAtariGroundTruth('Pong')

    # Take a few steps
    for i in range(10):
        action = extractor.env.action_space.sample()
        extractor.step(action)

    # Get current state
    frame, objects = extractor.get_frame_and_objects()
    print(f"Frame shape: {frame.shape}")
    print(f"Objects found: {len(objects)}")

    for obj in objects:
        print(f"  {obj.category}: pos={obj.position}, vel={obj.velocity}")

    # Test trajectory prediction
    if objects:
        trajectory = extractor.predict_trajectory(0, steps=3)
        print(f"Predicted trajectory for object 0: {trajectory}")

    # Test collision detection
    collisions = extractor.detect_collisions()
    print(f"Predicted collisions: {len(collisions)}")

    extractor.close()