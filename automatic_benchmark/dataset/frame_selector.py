"""
Frame selection logic for benchmark dataset creation.
Handles complexity calculation and stratified sampling.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Any
from ..config import COMPLEXITY_THRESHOLDS, COMPLEXITY_DISTRIBUTION
from ..utils.helpers import calculate_distance, categorize_complexity


@dataclass
class FrameComplexity:
    """Complexity metrics for a single frame."""
    frame_id: str
    game: str
    object_count: int
    spatial_density: float
    occlusion_score: float
    symmetry_score: float
    velocity_variance: float
    complexity_category: str  # 'easy', 'medium', 'hard'


class FrameSelector:
    """
    Selects frames for benchmark using stratified sampling based on complexity.
    """

    def __init__(self, seed: int = 42):
        """
        Initialize frame selector.

        Args:
            seed: Random seed for reproducibility
        """
        self.rng = np.random.RandomState(seed)

    def calculate_complexity(
        self,
        frame: np.ndarray,
        objects: List[Any],
        game: str,
        frame_number: int = 0
    ) -> FrameComplexity:
        """
        Calculate complexity metrics for a frame.

        Args:
            frame: RGB frame array (210, 160, 3)
            objects: List of OCAtari objects
            game: Name of the game
            frame_number: Frame number for ID generation

        Returns:
            FrameComplexity object with all metrics
        """
        object_count = len(objects)

        # Calculate spatial density (objects per unit area)
        frame_area = frame.shape[0] * frame.shape[1]
        spatial_density = object_count / frame_area if frame_area > 0 else 0

        # Calculate occlusion score (based on object proximity)
        occlusion_score = self._calculate_occlusion(objects)

        # Calculate spatial symmetry
        symmetry_score = self._calculate_symmetry(objects, frame.shape[1])

        # Calculate velocity variance
        velocity_variance = self._calculate_velocity_variance(objects)

        # Categorize complexity
        complexity_category = categorize_complexity(object_count, spatial_density)

        # Generate unique frame ID
        import hashlib
        frame_hash = hashlib.md5(frame.tobytes()).hexdigest()[:8]
        frame_id = f"{game.lower()}_{frame_number:04d}_{frame_hash}"

        return FrameComplexity(
            frame_id=frame_id,
            game=game,
            object_count=object_count,
            spatial_density=spatial_density,
            occlusion_score=occlusion_score,
            symmetry_score=symmetry_score,
            velocity_variance=velocity_variance,
            complexity_category=complexity_category
        )

    def _calculate_occlusion(self, objects: List[Any]) -> float:
        """
        Estimate occlusion based on object proximity.

        Closer objects → higher occlusion score
        """
        if len(objects) < 2:
            return 0.0

        positions = [obj.position for obj in objects if hasattr(obj, 'position')]
        if len(positions) < 2:
            return 0.0

        # Calculate minimum distances between all pairs
        min_distances = []
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                dist = calculate_distance(positions[i], positions[j])
                min_distances.append(dist)

        if not min_distances:
            return 0.0

        # Lower average minimum distance → higher occlusion
        avg_min_dist = np.mean(min_distances)
        occlusion_score = max(0.0, 1.0 - (avg_min_dist / 50.0))  # Normalize by 50 pixels

        return min(1.0, occlusion_score)

    def _calculate_symmetry(self, objects: List[Any], frame_width: int) -> float:
        """
        Calculate spatial symmetry across the frame.

        Perfect left-right balance → symmetry = 1.0
        """
        if not objects:
            return 1.0

        positions = [obj.position for obj in objects if hasattr(obj, 'position')]
        if not positions:
            return 1.0

        center_x = frame_width / 2
        left_objects = sum(1 for pos in positions if pos[0] < center_x)
        right_objects = sum(1 for pos in positions if pos[0] >= center_x)

        total = len(positions)
        symmetry = 1.0 - abs(left_objects - right_objects) / total

        return symmetry

    def _calculate_velocity_variance(self, objects: List[Any]) -> float:
        """
        Calculate variance in object velocities.

        High variance → more dynamic scene
        """
        if not objects:
            return 0.0

        velocities = [obj.velocity for obj in objects if hasattr(obj, 'velocity')]
        if not velocities:
            return 0.0

        # Calculate speeds (magnitude of velocity vectors)
        speeds = [np.sqrt(vx**2 + vy**2) for vx, vy in velocities]

        if len(speeds) < 2:
            return 0.0

        return float(np.var(speeds))

    def stratified_sample(
        self,
        candidates: List[Tuple[np.ndarray, List[Any], FrameComplexity]],
        target_count: int
    ) -> List[Tuple[np.ndarray, List[Any], FrameComplexity]]:
        """
        Perform stratified sampling to ensure balanced complexity distribution.

        Args:
            candidates: List of (frame, objects, complexity) tuples
            target_count: Number of frames to select

        Returns:
            Selected frames with balanced complexity
        """
        # Re-categorize based on COMPOSITE complexity score (more robust than just object count)
        # Calculate composite complexity: object_count + spatial_density + occlusion + velocity_variance
        composite_scores = []
        for frame, objects, complexity in candidates:
            # Normalize each metric to [0, 1] and combine
            obj_norm = min(complexity.object_count / 20.0, 1.0)  # Normalize to max 20 objects
            density_norm = min(complexity.spatial_density / 0.002, 1.0)  # Normalize to max density
            occlusion_norm = complexity.occlusion_score  # Already [0, 1]
            velocity_norm = min(complexity.velocity_variance / 10.0, 1.0)  # Normalize to max variance

            # Weighted composite score
            composite = (0.4 * obj_norm + 0.3 * density_norm +
                        0.2 * occlusion_norm + 0.1 * velocity_norm)
            composite_scores.append(composite)

        if composite_scores:
            # Use percentiles on composite score: 0-33rd = easy, 33-66 = medium, 66-100 = hard
            p33 = np.percentile(composite_scores, 33)
            p66 = np.percentile(composite_scores, 66)

            # Re-assign complexity categories based on composite score
            for (frame, objects, complexity), comp_score in zip(candidates, composite_scores):
                if comp_score <= p33:
                    complexity.complexity_category = 'easy'
                elif comp_score <= p66:
                    complexity.complexity_category = 'medium'
                else:
                    complexity.complexity_category = 'hard'

        # Group by complexity category
        by_complexity = {
            'easy': [],
            'medium': [],
            'hard': []
        }

        for candidate in candidates:
            complexity = candidate[2]
            by_complexity[complexity.complexity_category].append(candidate)

        # Calculate target per category
        target_per_category = {
            category: int(target_count * COMPLEXITY_DISTRIBUTION[category])
            for category in ['easy', 'medium', 'hard']
        }

        # Adjust for rounding
        total_allocated = sum(target_per_category.values())
        if total_allocated < target_count:
            # Add remainder to medium category
            target_per_category['medium'] += target_count - total_allocated

        selected = []

        for category in ['easy', 'medium', 'hard']:
            available = by_complexity[category]
            target = target_per_category[category]

            if len(available) >= target:
                # Random sample without replacement
                indices = self.rng.choice(len(available), target, replace=False)
                selected.extend([available[i] for i in indices])
            else:
                # Use all available if not enough
                selected.extend(available)
                print(f"  ⚠️  Warning: Only {len(available)} {category} frames available, "
                      f"target was {target}")

        # If we still need more frames, sample randomly from remainder
        if len(selected) < target_count:
            remaining_needed = target_count - len(selected)

            # Use frame IDs to track which frames are already selected
            selected_ids = set(c[2].frame_id for c in selected)
            all_remaining = [c for c in candidates if c[2].frame_id not in selected_ids]

            if len(all_remaining) >= remaining_needed:
                indices = self.rng.choice(len(all_remaining), remaining_needed, replace=False)
                selected.extend([all_remaining[i] for i in indices])

        return selected[:target_count]  # Ensure we don't exceed target

    def get_complexity_distribution(
        self,
        frames: List[Tuple[np.ndarray, List[Any], FrameComplexity]]
    ) -> dict:
        """
        Get distribution of complexity categories in a frame set.

        Args:
            frames: List of (frame, objects, complexity) tuples

        Returns:
            Dictionary with counts per category
        """
        distribution = {'easy': 0, 'medium': 0, 'hard': 0}

        for _, _, complexity in frames:
            distribution[complexity.complexity_category] += 1

        return distribution
