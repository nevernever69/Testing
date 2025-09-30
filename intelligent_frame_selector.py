"""
Intelligent Frame Selection System

This module automatically identifies the most challenging and informative frames
for spatial reasoning benchmarks. Uses OCAtari ground truth to detect scenarios
with high spatial complexity, imminent collisions, and dynamic object interactions.

Usage:
    selector = IntelligentFrameSelector(['Pong', 'Breakout', 'SpaceInvaders'])
    challenging_frames = selector.select_challenging_frames(num_frames=25)
    selector.export_selected_frames('challenging_frames_dataset.json')
"""

import numpy as np
import json
import random
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from collections import defaultdict
import math

from ocatari_ground_truth import OCAtariGroundTruth, ObjectInfo


@dataclass
class FrameComplexity:
    """Complexity scoring for a frame."""
    frame_id: str
    game: str
    frame_count: int
    complexity_score: float
    spatial_dynamics: float
    collision_proximity: float
    object_diversity: float
    trajectory_complexity: float
    reasoning: str
    frame_data: Optional[Dict[str, Any]] = None


class IntelligentFrameSelector:
    """
    Automatically selects challenging frames for spatial reasoning evaluation.

    Analyzes game frames using multiple complexity metrics:
    - Spatial dynamics (object movement and interactions)
    - Collision proximity (near-collision scenarios)
    - Object diversity (variety of object types and configurations)
    - Trajectory complexity (non-linear paths, bounces)
    """

    def __init__(self, games: List[str] = None, seed: int = 42):
        """
        Initialize frame selector.

        Args:
            games: List of games to analyze ['Pong', 'Breakout', 'SpaceInvaders']
            seed: Random seed for reproducible frame selection
        """
        self.games = games or ['Pong', 'Breakout', 'SpaceInvaders']
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)

        # Complexity thresholds for different scenarios
        self.complexity_thresholds = {
            'trajectory_scenarios': 0.6,    # Frames good for trajectory prediction
            'collision_scenarios': 0.7,     # Frames with imminent collisions
            'spatial_scenarios': 0.5,       # Frames with complex spatial relationships
            'diagnostic_scenarios': 0.4     # Frames suitable for Atari-GPT diagnostics
        }

        # Game-specific complexity weights
        self.game_weights = {
            'Pong': {
                'spatial_dynamics': 0.3,
                'collision_proximity': 0.4,
                'object_diversity': 0.1,
                'trajectory_complexity': 0.2
            },
            'Breakout': {
                'spatial_dynamics': 0.25,
                'collision_proximity': 0.35,
                'object_diversity': 0.2,
                'trajectory_complexity': 0.2
            },
            'SpaceInvaders': {
                'spatial_dynamics': 0.35,
                'collision_proximity': 0.2,
                'object_diversity': 0.3,
                'trajectory_complexity': 0.15
            }
        }

        self.selected_frames = []

    def select_challenging_frames(self,
                                num_frames_per_game: int = 25,
                                scenario_distribution: Optional[Dict[str, float]] = None) -> List[FrameComplexity]:
        """
        Select challenging frames across all games.

        Args:
            num_frames_per_game: Number of frames to select per game
            scenario_distribution: Distribution of scenario types to select

        Returns:
            List of selected frames with complexity analysis
        """
        if scenario_distribution is None:
            scenario_distribution = {
                'trajectory_scenarios': 0.3,
                'collision_scenarios': 0.25,
                'spatial_scenarios': 0.25,
                'diagnostic_scenarios': 0.2
            }

        all_selected_frames = []

        for game in self.games:
            print(f"Selecting frames for {game}...")

            # Analyze frames for this game
            frame_complexities = self._analyze_game_frames(game, num_samples=100)

            # Select frames by scenario type
            game_frames = self._select_frames_by_scenario(
                frame_complexities, num_frames_per_game, scenario_distribution
            )

            all_selected_frames.extend(game_frames)

        self.selected_frames = all_selected_frames
        return all_selected_frames

    def _analyze_game_frames(self, game: str, num_samples: int = 100) -> List[FrameComplexity]:
        """Analyze frames from a specific game for complexity."""
        print(f"  Analyzing {num_samples} frames from {game}...")

        extractor = OCAtariGroundTruth(game)
        frame_complexities = []

        try:
            for sample in range(num_samples):
                # Generate varied gameplay scenarios
                steps_to_take = np.random.randint(10, 50)

                for step in range(steps_to_take):
                    action = extractor.env.action_space.sample()
                    extractor.step(action)

                # Analyze current frame
                frame, objects = extractor.get_frame_and_objects()

                if objects:  # Only analyze frames with objects
                    complexity = self._calculate_frame_complexity(
                        frame, objects, extractor, game, sample
                    )

                    if complexity.complexity_score > 0.2:  # Filter out trivial frames
                        frame_complexities.append(complexity)

        finally:
            extractor.close()

        # Sort by complexity score
        frame_complexities.sort(key=lambda x: x.complexity_score, reverse=True)

        print(f"  Found {len(frame_complexities)} complex frames")
        return frame_complexities

    def _calculate_frame_complexity(self,
                                  frame: np.ndarray,
                                  objects: List[ObjectInfo],
                                  extractor: OCAtariGroundTruth,
                                  game: str,
                                  sample_id: int) -> FrameComplexity:
        """Calculate complexity metrics for a single frame."""

        # Calculate individual complexity components
        spatial_dynamics = self._calculate_spatial_dynamics(objects)
        collision_proximity = self._calculate_collision_proximity(objects, extractor)
        object_diversity = self._calculate_object_diversity(objects, game)
        trajectory_complexity = self._calculate_trajectory_complexity(objects, extractor)

        # Weighted combination based on game type
        weights = self.game_weights.get(game, self.game_weights['Pong'])

        complexity_score = (
            spatial_dynamics * weights['spatial_dynamics'] +
            collision_proximity * weights['collision_proximity'] +
            object_diversity * weights['object_diversity'] +
            trajectory_complexity * weights['trajectory_complexity']
        )

        # Generate reasoning
        reasoning = self._generate_complexity_reasoning(
            spatial_dynamics, collision_proximity, object_diversity, trajectory_complexity
        )

        # Store frame data for later use
        frame_data = {
            'frame': frame.tolist(),  # Convert numpy array for JSON serialization
            'objects': [obj.to_dict() for obj in objects],
            'spatial_relationships': extractor.get_spatial_relationships(),
            'predicted_collisions': extractor.detect_collisions()
        }

        return FrameComplexity(
            frame_id=f"{game}_{sample_id}_{extractor.frame_count}",
            game=game,
            frame_count=extractor.frame_count,
            complexity_score=complexity_score,
            spatial_dynamics=spatial_dynamics,
            collision_proximity=collision_proximity,
            object_diversity=object_diversity,
            trajectory_complexity=trajectory_complexity,
            reasoning=reasoning,
            frame_data=frame_data
        )

    def _calculate_spatial_dynamics(self, objects: List[ObjectInfo]) -> float:
        """Calculate spatial dynamics score based on object movement and distribution."""
        if len(objects) < 2:
            return 0.0

        # Velocity variance - more varied movement = higher complexity
        velocities = [math.sqrt(obj.velocity[0]**2 + obj.velocity[1]**2) for obj in objects]
        velocity_variance = np.var(velocities) if velocities else 0.0

        # Spatial distribution - objects spread across frame = higher complexity
        positions = [obj.position for obj in objects]
        x_coords = [pos[0] for pos in positions]
        y_coords = [pos[1] for pos in positions]

        x_spread = (max(x_coords) - min(x_coords)) / 160.0 if len(x_coords) > 1 else 0.0  # Normalize to frame width
        y_spread = (max(y_coords) - min(y_coords)) / 210.0 if len(y_coords) > 1 else 0.0  # Normalize to frame height

        spatial_spread = (x_spread + y_spread) / 2

        # Combine metrics
        dynamics_score = min(1.0, (velocity_variance / 100.0) + spatial_spread)
        return dynamics_score

    def _calculate_collision_proximity(self, objects: List[ObjectInfo], extractor: OCAtariGroundTruth) -> float:
        """Calculate collision proximity score - higher for frames with imminent collisions."""
        if len(objects) < 2:
            return 0.0

        predicted_collisions = extractor.detect_collisions(steps=5)

        if not predicted_collisions:
            return 0.0

        # Score based on collision timing and confidence
        proximity_score = 0.0

        for collision in predicted_collisions:
            # Sooner collisions = higher score
            time_factor = 1.0 - (collision['collision_frame'] / 5.0)
            confidence_factor = collision.get('confidence', 0.5)

            collision_score = time_factor * confidence_factor
            proximity_score = max(proximity_score, collision_score)

        return min(1.0, proximity_score)

    def _calculate_object_diversity(self, objects: List[ObjectInfo], game: str) -> float:
        """Calculate object diversity score based on variety and configuration."""
        if not objects:
            return 0.0

        # Count unique object categories
        categories = set(obj.category for obj in objects)
        category_diversity = len(categories) / self._get_max_categories(game)

        # Bonus for multiple objects of same type (formations, multiple bricks, etc.)
        category_counts = defaultdict(int)
        for obj in objects:
            category_counts[obj.category] += 1

        formation_bonus = 0.0
        for category, count in category_counts.items():
            if count > 1:  # Multiple objects of same type
                formation_bonus += min(0.3, count / 10.0)  # Cap bonus

        diversity_score = min(1.0, category_diversity + formation_bonus)
        return diversity_score

    def _get_max_categories(self, game: str) -> int:
        """Get maximum expected object categories for normalization."""
        max_categories = {
            'Pong': 3,          # Player, Ball, Enemy
            'Breakout': 8,      # Player, Ball, BlockRows (multiple types)
            'SpaceInvaders': 10 # Player, Aliens, Bullets, Shields (multiple types)
        }
        return max_categories.get(game, 5)

    def _calculate_trajectory_complexity(self, objects: List[ObjectInfo], extractor: OCAtariGroundTruth) -> float:
        """Calculate trajectory complexity based on predicted paths and physics."""
        if not objects:
            return 0.0

        complexity_score = 0.0

        for i, obj in enumerate(objects):
            # Only consider moving objects
            velocity = math.sqrt(obj.velocity[0]**2 + obj.velocity[1]**2)
            if velocity < 0.5:  # Stationary objects
                continue

            # Predict trajectory
            predicted_positions = extractor.predict_trajectory(i, steps=5)

            if len(predicted_positions) >= 2:
                # Calculate trajectory curvature (how non-linear the path is)
                curvature = self._calculate_trajectory_curvature(predicted_positions)
                complexity_score = max(complexity_score, curvature)

        return min(1.0, complexity_score)

    def _calculate_trajectory_curvature(self, positions: List[Tuple[int, int]]) -> float:
        """Calculate curvature of trajectory (0=straight, 1=highly curved)."""
        if len(positions) < 3:
            return 0.0

        # Calculate angles between consecutive segments
        angles = []
        for i in range(len(positions) - 2):
            p1, p2, p3 = positions[i], positions[i+1], positions[i+2]

            # Calculate vectors
            v1 = (p2[0] - p1[0], p2[1] - p1[1])
            v2 = (p3[0] - p2[0], p3[1] - p2[1])

            # Calculate angle between vectors
            if v1 != (0, 0) and v2 != (0, 0):
                dot_product = v1[0]*v2[0] + v1[1]*v2[1]
                mag1 = math.sqrt(v1[0]**2 + v1[1]**2)
                mag2 = math.sqrt(v2[0]**2 + v2[1]**2)

                cos_angle = dot_product / (mag1 * mag2)
                cos_angle = max(-1, min(1, cos_angle))  # Clamp to valid range
                angle = math.acos(cos_angle)

                angles.append(angle)

        if not angles:
            return 0.0

        # Average deviation from straight line (pi radians)
        curvature = sum(abs(angle - math.pi) for angle in angles) / len(angles)
        normalized_curvature = curvature / math.pi

        return normalized_curvature

    def _generate_complexity_reasoning(self,
                                     spatial_dynamics: float,
                                     collision_proximity: float,
                                     object_diversity: float,
                                     trajectory_complexity: float) -> str:
        """Generate human-readable reasoning for complexity score."""
        components = []

        if spatial_dynamics > 0.6:
            components.append("high spatial dynamics")
        elif spatial_dynamics > 0.3:
            components.append("moderate spatial dynamics")

        if collision_proximity > 0.7:
            components.append("imminent collision scenario")
        elif collision_proximity > 0.4:
            components.append("potential collision scenario")

        if object_diversity > 0.6:
            components.append("diverse object configuration")

        if trajectory_complexity > 0.5:
            components.append("complex trajectory patterns")

        if not components:
            return "basic spatial scenario"

        return f"Complex frame with {', '.join(components)}"

    def _select_frames_by_scenario(self,
                                  frame_complexities: List[FrameComplexity],
                                  num_frames: int,
                                  scenario_distribution: Dict[str, float]) -> List[FrameComplexity]:
        """Select frames according to desired scenario distribution."""
        selected = []

        for scenario_type, proportion in scenario_distribution.items():
            num_for_scenario = int(num_frames * proportion)

            # Filter frames suitable for this scenario type
            threshold = self.complexity_thresholds[scenario_type]

            if scenario_type == 'trajectory_scenarios':
                suitable_frames = [f for f in frame_complexities if f.trajectory_complexity > 0.4]
            elif scenario_type == 'collision_scenarios':
                suitable_frames = [f for f in frame_complexities if f.collision_proximity > 0.5]
            elif scenario_type == 'spatial_scenarios':
                suitable_frames = [f for f in frame_complexities if f.object_diversity > 0.4]
            else:  # diagnostic_scenarios
                suitable_frames = [f for f in frame_complexities if f.complexity_score > threshold]

            # Select top frames for this scenario
            scenario_frames = suitable_frames[:num_for_scenario]
            selected.extend(scenario_frames)

        # Fill remaining slots with highest complexity frames not yet selected
        selected_ids = set(f.frame_id for f in selected)
        remaining_frames = [f for f in frame_complexities if f.frame_id not in selected_ids]
        remaining_slots = num_frames - len(selected)

        if remaining_slots > 0:
            selected.extend(remaining_frames[:remaining_slots])

        return selected

    def get_frame_statistics(self) -> Dict[str, Any]:
        """Get statistics about selected frames."""
        if not self.selected_frames:
            return {}

        # Overall statistics
        complexity_scores = [f.complexity_score for f in self.selected_frames]
        games = [f.game for f in self.selected_frames]

        stats = {
            'total_frames': len(self.selected_frames),
            'complexity_distribution': {
                'mean': np.mean(complexity_scores),
                'std': np.std(complexity_scores),
                'min': np.min(complexity_scores),
                'max': np.max(complexity_scores),
                'median': np.median(complexity_scores)
            },
            'games_distribution': {game: games.count(game) for game in set(games)},
            'scenario_analysis': self._analyze_scenario_distribution()
        }

        return stats

    def _analyze_scenario_distribution(self) -> Dict[str, Any]:
        """Analyze distribution of scenario types in selected frames."""
        trajectory_frames = sum(1 for f in self.selected_frames if f.trajectory_complexity > 0.4)
        collision_frames = sum(1 for f in self.selected_frames if f.collision_proximity > 0.5)
        spatial_frames = sum(1 for f in self.selected_frames if f.object_diversity > 0.4)
        high_complexity = sum(1 for f in self.selected_frames if f.complexity_score > 0.6)

        return {
            'trajectory_suitable': trajectory_frames,
            'collision_suitable': collision_frames,
            'spatial_suitable': spatial_frames,
            'high_complexity': high_complexity,
            'average_components': {
                'spatial_dynamics': np.mean([f.spatial_dynamics for f in self.selected_frames]),
                'collision_proximity': np.mean([f.collision_proximity for f in self.selected_frames]),
                'object_diversity': np.mean([f.object_diversity for f in self.selected_frames]),
                'trajectory_complexity': np.mean([f.trajectory_complexity for f in self.selected_frames])
            }
        }

    def export_selected_frames(self, filename: str, include_frame_data: bool = False):
        """Export selected frames to JSON file."""
        if not self.selected_frames:
            print("No frames selected. Run select_challenging_frames() first.")
            return

        export_data = {
            'metadata': {
                'total_frames': len(self.selected_frames),
                'games': self.games,
                'selection_seed': self.seed,
                'statistics': self.get_frame_statistics()
            },
            'frames': []
        }

        for frame in self.selected_frames:
            frame_export = {
                'frame_id': frame.frame_id,
                'game': frame.game,
                'frame_count': frame.frame_count,
                'complexity_score': frame.complexity_score,
                'components': {
                    'spatial_dynamics': frame.spatial_dynamics,
                    'collision_proximity': frame.collision_proximity,
                    'object_diversity': frame.object_diversity,
                    'trajectory_complexity': frame.trajectory_complexity
                },
                'reasoning': frame.reasoning
            }

            if include_frame_data and frame.frame_data:
                # Note: This will create very large files due to frame data
                frame_export['frame_data'] = frame.frame_data

            export_data['frames'].append(frame_export)

        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2)

        print(f"Exported {len(self.selected_frames)} frames to {filename}")

    def create_selection_report(self) -> str:
        """Create a detailed report of frame selection results."""
        if not self.selected_frames:
            return "No frames selected."

        stats = self.get_frame_statistics()

        report = f"""
Intelligent Frame Selection Report
================================

Total Frames Selected: {stats['total_frames']}

Complexity Distribution:
  Mean Score: {stats['complexity_distribution']['mean']:.3f}
  Std Dev:    {stats['complexity_distribution']['std']:.3f}
  Range:      {stats['complexity_distribution']['min']:.3f} - {stats['complexity_distribution']['max']:.3f}

Games Distribution:
"""
        for game, count in stats['games_distribution'].items():
            percentage = count / stats['total_frames'] * 100
            report += f"  {game}: {count} frames ({percentage:.1f}%)\n"

        report += f"""
Scenario Suitability:
  Trajectory Prediction: {stats['scenario_analysis']['trajectory_suitable']} frames
  Collision Detection:   {stats['scenario_analysis']['collision_suitable']} frames
  Spatial Relationships: {stats['scenario_analysis']['spatial_suitable']} frames
  High Complexity:       {stats['scenario_analysis']['high_complexity']} frames

Average Component Scores:
  Spatial Dynamics:      {stats['scenario_analysis']['average_components']['spatial_dynamics']:.3f}
  Collision Proximity:   {stats['scenario_analysis']['average_components']['collision_proximity']:.3f}
  Object Diversity:      {stats['scenario_analysis']['average_components']['object_diversity']:.3f}
  Trajectory Complexity: {stats['scenario_analysis']['average_components']['trajectory_complexity']:.3f}
"""

        return report


if __name__ == "__main__":
    # Test the intelligent frame selector
    print("Testing Intelligent Frame Selection System")
    print("=" * 50)

    selector = IntelligentFrameSelector(['Pong', 'Breakout'])  # Test with 2 games for speed

    # Select challenging frames
    selected_frames = selector.select_challenging_frames(num_frames_per_game=10)

    # Print results
    print(f"\nSelected {len(selected_frames)} challenging frames")

    # Show top 5 most complex frames
    print("\nTop 5 Most Complex Frames:")
    for i, frame in enumerate(selected_frames[:5]):
        print(f"  {i+1}. {frame.frame_id} (Score: {frame.complexity_score:.3f})")
        print(f"     {frame.reasoning}")

    # Generate report
    report = selector.create_selection_report()
    print(report)

    # Export results
    selector.export_selected_frames('test_challenging_frames.json')

    print("\nFrame selection test completed successfully!")