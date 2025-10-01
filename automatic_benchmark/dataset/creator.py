"""
Benchmark dataset creator.
Generates fixed, reproducible test sets with OCAtari ground truth.
"""

import numpy as np
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Any
from dataclasses import asdict
from datetime import datetime
from PIL import Image
from tqdm import tqdm

from ocatari_ground_truth import OCAtariGroundTruth
from .frame_selector import FrameSelector, FrameComplexity
from ..config import (
    GAMES, FRAMES_PER_GAME, RANDOM_SEED,
    CANDIDATE_FRAMES_PER_GAME
)
from ..utils.helpers import calculate_spatial_relationships, generate_spatial_description


# Game-specific important objects (filter out noise from OCAtari)
IMPORTANT_OBJECTS = {
    'Pong': {
        'keywords': ['player', 'paddle', 'ball', 'enemy'],
        'exclude': ['score', 'display']
    },
    'Breakout': {
        'keywords': ['player', 'paddle', 'ball', 'block', 'brick'],
        'exclude': ['score', 'display', 'lives']
    },
    'SpaceInvaders': {
        'keywords': ['player', 'alien', 'enemy', 'bullet', 'projectile', 'shield'],
        'exclude': ['score', 'display', 'lives']
    }
}


class BenchmarkDatasetCreator:
    """
    Creates reproducible benchmark dataset with fixed frames and ground truth.
    """

    # Coordinate scaling from original Atari (160x210) to benchmark (1280x720)
    ORIGINAL_WIDTH = 160
    ORIGINAL_HEIGHT = 210
    SCALED_WIDTH = 1280
    SCALED_HEIGHT = 720
    WIDTH_SCALE = SCALED_WIDTH / ORIGINAL_WIDTH   # 8.0
    HEIGHT_SCALE = SCALED_HEIGHT / ORIGINAL_HEIGHT  # 3.43

    def __init__(self, output_dir: str, seed: int = RANDOM_SEED):
        """
        Initialize dataset creator.

        Args:
            output_dir: Directory to save dataset
            seed: Random seed for reproducibility
        """
        self.output_dir = Path(output_dir)
        self.seed = seed
        self.frame_selector = FrameSelector(seed=seed)

        # Create directory structure
        self.frames_dir = self.output_dir / "frames"
        self.ground_truth_dir = self.output_dir / "ground_truth"
        self.metadata_dir = self.output_dir / "metadata"

        for dir in [self.frames_dir, self.ground_truth_dir, self.metadata_dir]:
            dir.mkdir(parents=True, exist_ok=True)

        # Initialize metadata
        self.dataset_metadata = {
            'version': '1.0.0',
            'created': datetime.now().isoformat(),
            'seed': seed,
            'games': [],
            'total_frames': 0,
            'frames_per_game': {},
            'complexity_distribution': {}
        }

    def create_dataset(
        self,
        games: List[str] = None,
        frames_per_game: int = FRAMES_PER_GAME
    ) -> Dict[str, Any]:
        """
        Create complete benchmark dataset.

        Args:
            games: List of game names (default: Pong, Breakout, SpaceInvaders)
            frames_per_game: Number of frames per game

        Returns:
            Dataset index dictionary
        """
        if games is None:
            games = GAMES

        print(f"\n{'='*70}")
        print(f"Creating Automatic Benchmark Dataset")
        print(f"{'='*70}")
        print(f"Output directory: {self.output_dir}")
        print(f"Games: {', '.join(games)}")
        print(f"Frames per game: {frames_per_game}")
        print(f"Random seed: {self.seed}")
        print(f"{'='*70}\n")

        all_frames = []

        for game in games:
            print(f"\n{'='*60}")
            print(f"Processing: {game}")
            print(f"{'='*60}")

            # Generate candidates
            candidates = self._generate_candidate_frames(game, CANDIDATE_FRAMES_PER_GAME)

            # Stratified sampling
            selected = self.frame_selector.stratified_sample(candidates, frames_per_game)

            print(f"\n💾 Saving {game} frames...")
            game_frames = []

            for idx, (frame, objects, complexity) in enumerate(tqdm(selected, desc=f"Saving {game}")):
                frame_data = self._save_frame(frame, objects, complexity, game, idx)
                game_frames.append(frame_data)
                all_frames.append(frame_data)

            # Update metadata
            complexity_dist = self.frame_selector.get_complexity_distribution(selected)
            self.dataset_metadata['frames_per_game'][game] = {
                'total': len(game_frames),
                'complexity_distribution': complexity_dist
            }

            print(f"✅ Saved {len(game_frames)} frames for {game}")
            print(f"   Distribution: {complexity_dist}")

        # Update overall metadata
        self.dataset_metadata['games'] = games
        self.dataset_metadata['total_frames'] = len(all_frames)

        # Aggregate complexity distribution
        all_complexity = {'easy': 0, 'medium': 0, 'hard': 0}
        for game_meta in self.dataset_metadata['frames_per_game'].values():
            for category, count in game_meta['complexity_distribution'].items():
                all_complexity[category] += count
        self.dataset_metadata['complexity_distribution'] = all_complexity

        # Create dataset index
        dataset_index = {
            'metadata': self.dataset_metadata,
            'frames': all_frames
        }

        # Save dataset index (with custom encoder)
        index_path = self.metadata_dir / "dataset_index.json"

        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                return super(NumpyEncoder, self).default(obj)

        with open(index_path, 'w') as f:
            json.dump(dataset_index, f, indent=2, cls=NumpyEncoder)

        # Create README
        self._create_readme()

        print(f"\n{'='*70}")
        print(f"✅ Dataset Created Successfully!")
        print(f"{'='*70}")
        print(f"Total frames: {len(all_frames)}")
        print(f"Complexity distribution: {all_complexity}")
        print(f"Dataset index: {index_path}")
        print(f"{'='*70}\n")

        return dataset_index

    def _generate_candidate_frames(
        self,
        game: str,
        num_candidates: int
    ) -> List[Tuple[np.ndarray, List[Any], FrameComplexity]]:
        """
        Generate candidate frames with complexity metrics.

        Args:
            game: Game name
            num_candidates: Number of candidates to generate

        Returns:
            List of (frame, objects, complexity) tuples
        """
        print(f"\n🎮 Generating {num_candidates} candidate frames for {game}...")

        extractor = OCAtariGroundTruth(game)
        candidates = []

        # Use tqdm for progress
        for i in tqdm(range(num_candidates), desc=f"Generating {game} candidates"):
            # Random gameplay to get diverse states
            # Longer gameplay → more variety (bricks destroyed, aliens shot, etc.)
            steps = np.random.randint(20, 150)  # Increased from (5, 50) to (20, 150)
            for _ in range(steps):
                action = extractor.env.action_space.sample()
                extractor.step(action)

            frame, objects = extractor.get_frame_and_objects()

            # Calculate complexity
            complexity = self.frame_selector.calculate_complexity(frame, objects, game, i)

            candidates.append((frame, objects, complexity))

        extractor.close()

        print(f"✅ Generated {len(candidates)} candidate frames")

        return candidates

    def _save_frame(
        self,
        frame: np.ndarray,
        objects: List[Any],
        complexity: FrameComplexity,
        game: str,
        frame_idx: int
    ) -> Dict[str, Any]:
        """
        Save a frame and its ground truth.

        Args:
            frame: RGB frame array
            objects: OCAtari objects
            complexity: Complexity metrics
            game: Game name
            frame_idx: Frame index

        Returns:
            Frame metadata dictionary
        """
        # Generate frame ID
        frame_id = f"{game.lower()}_{frame_idx:04d}_{complexity.complexity_category}"

        # Save frame image
        frame_path = self.frames_dir / f"{frame_id}.png"
        Image.fromarray(frame).save(frame_path)

        # Create ground truth data
        ground_truth = self._create_ground_truth(frame_id, game, objects, complexity)

        # Save ground truth (with custom encoder for numpy types)
        gt_path = self.ground_truth_dir / f"{frame_id}.json"

        # Custom JSON encoder for numpy types
        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                return super(NumpyEncoder, self).default(obj)

        with open(gt_path, 'w') as f:
            json.dump(ground_truth, f, indent=2, cls=NumpyEncoder)

        # Return frame metadata (ensure complexity dict has Python types)
        complexity_dict_for_metadata = asdict(complexity)
        for key, value in complexity_dict_for_metadata.items():
            if isinstance(value, np.integer):
                complexity_dict_for_metadata[key] = int(value)
            elif isinstance(value, np.floating):
                complexity_dict_for_metadata[key] = float(value)

        return {
            'frame_id': frame_id,
            'game': game,
            'frame_number': frame_idx,
            'complexity': complexity_dict_for_metadata,
            'frame_path': f"frames/{frame_id}.png",
            'ground_truth_path': f"ground_truth/{frame_id}.json"
        }

    def _create_ground_truth(
        self,
        frame_id: str,
        game: str,
        objects: List[Any],
        complexity: FrameComplexity
    ) -> Dict[str, Any]:
        """
        Create ground truth data from OCAtari objects.

        Args:
            frame_id: Unique frame identifier
            game: Game name
            objects: OCAtari objects
            complexity: Complexity metrics

        Returns:
            Ground truth dictionary
        """
        # Convert OCAtari objects to serializable format (convert numpy types to Python types)
        def to_python_types(obj):
            """Convert numpy types to native Python types for JSON serialization."""
            import numpy as np
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (list, tuple)):
                return [to_python_types(item) for item in obj]
            else:
                return obj

        serialized_objects = []
        for obj in objects:
            obj_dict = {
                'id': to_python_types(getattr(obj, 'id', None)),
                'category': str(obj.category),
                'position': to_python_types(list(obj.position)) if hasattr(obj, 'position') else [0, 0],
                'size': to_python_types(list(obj.size)) if hasattr(obj, 'size') else [0, 0],
                'velocity': to_python_types(list(obj.velocity)) if hasattr(obj, 'velocity') else [0.0, 0.0],
                'center': to_python_types(list(obj.properties.get('center', obj.position))) if hasattr(obj, 'properties') else to_python_types(list(obj.position)),
                'rgb': to_python_types(list(obj.properties.get('rgb', [0, 0, 0]))) if hasattr(obj, 'properties') else [0, 0, 0]
            }
            serialized_objects.append(obj_dict)

        # Calculate spatial relationships
        spatial_relationships = calculate_spatial_relationships(serialized_objects)

        # Get action space info
        from ocatari_ground_truth import OCAtariGroundTruth
        temp_extractor = OCAtariGroundTruth(game)
        action_space_info = {
            'type': 'Discrete',
            'n': temp_extractor.env.action_space.n,
            'actions': temp_extractor.env.get_action_meanings() if hasattr(temp_extractor.env, 'get_action_meanings') else []
        }
        temp_extractor.close()

        # Convert complexity dataclass to dict and ensure all values are JSON-serializable
        complexity_dict = asdict(complexity)
        for key, value in complexity_dict.items():
            if isinstance(value, np.integer):
                complexity_dict[key] = int(value)
            elif isinstance(value, np.floating):
                complexity_dict[key] = float(value)

        # Annotate all objects with importance tiers
        annotated_objects = self._annotate_object_tiers(serialized_objects, game)

        # Space Invaders: Add formation analysis
        formation_data = None
        if 'space' in game.lower() or 'invader' in game.lower():
            from ..utils.spaceinvaders_formation import analyze_spaceinvaders_frame
            formation_data = analyze_spaceinvaders_frame(serialized_objects)

        # Create ground truth structure
        ground_truth = {
            'frame_id': frame_id,
            'game': game,
            'complexity': complexity_dict,
            'ocatari_data': {
                'objects': annotated_objects,  # ALL objects with tier annotations
                'spatial_relationships': spatial_relationships,
                'action_space': action_space_info
            },
            'formation_analysis': formation_data,  # Space Invaders formation tracking (None for other games)
            'reference_answers': {
                'visual': {
                    'qualitative': self._generate_visual_reference_qualitative(serialized_objects, game, formation_data),
                    'quantitative': self._generate_visual_reference_quantitative(serialized_objects, game, formation_data)
                },
                'spatial': {
                    'qualitative': self._generate_spatial_reference_qualitative(serialized_objects, game, formation_data),
                    'quantitative': self._generate_spatial_reference_quantitative(serialized_objects, spatial_relationships, game)
                },
                'strategy': {
                    'qualitative': self._generate_strategy_reference_qualitative(serialized_objects, game, formation_data),
                    'quantitative': self._generate_strategy_reference_quantitative(serialized_objects, game)
                },
                'identification': {
                    'qualitative': game,  # Game name is same for both formats
                    'quantitative': game
                }
            }
        }

        return ground_truth

    def _filter_important_objects(self, objects: List[Dict[str, Any]], game: str) -> List[Dict[str, Any]]:
        """
        Filter objects to keep only CORE objects for reference generation.
        (Note: Full object list is kept in ground truth, this is just for reference text)

        Args:
            objects: All objects from OCAtari
            game: Game name

        Returns:
            Filtered list of core objects for reference generation
        """
        from automatic_benchmark.config.object_importance import get_object_tier

        # For reference generation, focus on core objects only
        # But ground truth will contain ALL objects with tier annotations
        core_objects = []
        for obj in objects:
            tier = get_object_tier(game, obj['category'])
            if tier == 'core':
                core_objects.append(obj)

        # If no core objects found, fall back to all objects
        return core_objects if core_objects else objects

    def _annotate_object_tiers(self, objects: List[Dict[str, Any]], game: str) -> List[Dict[str, Any]]:
        """
        Annotate each object with its importance tier (core/secondary/unknown).

        Args:
            objects: All objects from OCAtari
            game: Game name

        Returns:
            Objects with 'tier' field added
        """
        from automatic_benchmark.config.object_importance import get_object_tier

        annotated = []
        for obj in objects:
            obj_copy = obj.copy()
            obj_copy['tier'] = get_object_tier(game, obj['category'])
            annotated.append(obj_copy)

        return annotated

    def _generate_visual_reference_qualitative(self, objects: List[Dict[str, Any]], game: str, formation_data: Dict = None) -> str:
        """Generate qualitative visual reference (Vision-Only friendly)."""
        # Space Invaders: Use formation-based description
        if formation_data and ('space' in game.lower() or 'invader' in game.lower()):
            return self._generate_spaceinvaders_visual_qualitative(formation_data)

        # Filter to important objects only
        important_objs = self._filter_important_objects(objects, game)

        if not important_objs:
            return f"This is a {game} game frame with no clearly visible key game objects."

        descriptions = []

        for obj in important_objs:
            category = obj['category']
            position = obj['position']

            # Scale position to match benchmark frame (1280x720)
            x_orig, y_orig = position[0], position[1]
            x = x_orig * self.WIDTH_SCALE
            y = y_orig * self.HEIGHT_SCALE

            # Horizontal position (0-1280 for scaled frame)
            if x < 400:
                h_pos = "left side"
            elif x < 880:
                h_pos = "center area"
            else:
                h_pos = "right side"

            # Vertical position (0-720 for scaled frame)
            if y < 240:
                v_pos = "upper"
            elif y < 480:
                v_pos = "middle"
            else:
                v_pos = "lower"

            descriptions.append(f"{category} in the {v_pos} {h_pos}")

        return f"Key elements: {', '.join(descriptions)}."

    def _generate_visual_reference_quantitative(self, objects: List[Dict[str, Any]], game: str, formation_data: Dict = None) -> str:
        """Generate quantitative visual reference (Vision+Symbol format) with scaled coordinates."""
        # Space Invaders: Use formation-based description
        if formation_data and ('space' in game.lower() or 'invader' in game.lower()):
            return self._generate_spaceinvaders_visual_quantitative(formation_data)

        # Filter to important objects only
        important_objs = self._filter_important_objects(objects, game)

        if not important_objs:
            return f"This is a {game} game frame with no clearly visible key game objects."

        obj_details = []
        for obj in important_objs:
            category = obj['category']
            position = obj['position']
            size = obj.get('size', [0, 0])

            # Scale coordinates to 1280x720
            x_scaled = int(position[0] * self.WIDTH_SCALE)
            y_scaled = int(position[1] * self.HEIGHT_SCALE)
            w_scaled = int(size[0] * self.WIDTH_SCALE)
            h_scaled = int(size[1] * self.HEIGHT_SCALE)

            obj_details.append(f"{category} at ({x_scaled}, {y_scaled}), size {w_scaled}x{h_scaled}")

        description = f"Detected {len(important_objs)} key objects: "
        description += "; ".join(obj_details) + "."
        return description

    def _generate_spatial_reference_qualitative(self, objects: List[Dict[str, Any]], game: str, formation_data: Dict = None) -> str:
        """Generate qualitative spatial reference (Vision-Only friendly)."""
        # Space Invaders: Use formation-based description
        if formation_data and ('space' in game.lower() or 'invader' in game.lower()):
            return self._generate_spaceinvaders_spatial_qualitative(formation_data)

        # Filter to important objects only
        important_objs = self._filter_important_objects(objects, game)

        if not important_objs or len(important_objs) < 2:
            return "Not enough key objects for spatial analysis."

        # Only describe key relationships (not all pairs)
        spatial_desc = []

        for i, obj1 in enumerate(important_objs[:3]):  # Max 3 objects to avoid verbosity
            for obj2 in important_objs[i+1:i+2]:  # Only 1 pair per object
                cat1, cat2 = obj1['category'], obj2['category']
                pos1, pos2 = obj1['position'], obj2['position']

                # Horizontal
                if abs(pos1[0] - pos2[0]) > 20:  # Significant difference
                    h_rel = "left of" if pos1[0] < pos2[0] else "right of"
                    spatial_desc.append(f"The {cat1} is {h_rel} the {cat2}")

                # Vertical
                if abs(pos1[1] - pos2[1]) > 20:
                    v_rel = "above" if pos1[1] < pos2[1] else "below"
                    spatial_desc.append(f"The {cat1} is {v_rel} the {cat2}")

                # Proximity
                distance = ((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)**0.5
                if distance < 50:
                    proximity = "very close to"
                elif distance < 100:
                    proximity = "near"
                else:
                    proximity = "far from"
                spatial_desc.append(f"The {cat1} is {proximity} the {cat2}")

        return "Spatial layout: " + "; ".join(spatial_desc) + "."

    def _generate_spatial_reference_quantitative(self, objects: List[Dict[str, Any]], relationships: Dict, game: str) -> str:
        """Generate quantitative spatial reference (Vision+Symbol format) with scaled coordinates."""
        # Filter to important objects only
        important_objs = self._filter_important_objects(objects, game)

        if not important_objs or len(important_objs) < 2:
            return "Not enough key objects for spatial analysis."

        spatial_desc = []

        for i, obj1 in enumerate(important_objs[:3]):
            for obj2 in important_objs[i+1:i+2]:
                cat1, cat2 = obj1['category'], obj2['category']
                pos1, pos2 = obj1['position'], obj2['position']

                # Scale positions to 1280x720
                x1 = int(pos1[0] * self.WIDTH_SCALE)
                y1 = int(pos1[1] * self.HEIGHT_SCALE)
                x2 = int(pos2[0] * self.WIDTH_SCALE)
                y2 = int(pos2[1] * self.HEIGHT_SCALE)

                # Horizontal
                h_rel = "LEFT" if x1 < x2 else "RIGHT"
                spatial_desc.append(f"{cat1} at x={x1} is {h_rel} of {cat2} at x={x2}")

                # Vertical
                v_rel = "ABOVE" if y1 < y2 else "BELOW"
                spatial_desc.append(f"{cat1} at y={y1} is {v_rel} {cat2} at y={y2}")

                # Distance (scaled)
                distance = ((x1 - x2)**2 + (y1 - y2)**2)**0.5
                spatial_desc.append(f"Distance: {distance:.1f} pixels")

        return "Spatial relationships: " + "; ".join(spatial_desc) + "."

    def _generate_strategy_reference_qualitative(self, objects: List[Dict[str, Any]], game: str, formation_data: Dict = None) -> str:
        """Generate qualitative strategy reference (Vision-Only friendly)."""
        if not objects:
            return "Unable to determine optimal strategy without visible game elements."

        # Game-specific strategy - qualitative descriptions
        if game == 'Pong':
            return self._generate_pong_strategy_qualitative(objects)
        elif game == 'Breakout':
            return self._generate_breakout_strategy_qualitative(objects)
        elif game == 'SpaceInvaders':
            # Use formation-based strategy if available
            if formation_data:
                return self._generate_spaceinvaders_strategy_qualitative_formation(formation_data)
            return self._generate_spaceinvaders_strategy_qualitative(objects)

        return f"Analyze {game} game state and determine optimal action."

    def _generate_strategy_reference_quantitative(self, objects: List[Dict[str, Any]], game: str) -> str:
        """Generate quantitative strategy reference (Vision+Symbol format)."""
        if not objects:
            return "Unable to determine optimal strategy without visible game elements."

        # Game-specific strategy with coordinates and measurements
        if game == 'Pong':
            return self._generate_pong_strategy_quantitative(objects)
        elif game == 'Breakout':
            return self._generate_breakout_strategy_quantitative(objects)
        elif game == 'SpaceInvaders':
            return self._generate_spaceinvaders_strategy_quantitative(objects)

        return f"Analyze {game} game state and determine optimal action based on object positions."

    def _generate_pong_strategy_qualitative(self, objects: List[Dict[str, Any]]) -> str:
        """Generate Pong strategy without exact coordinates (Vision-Only friendly)."""
        player_paddle = next((o for o in objects if 'player' in o['category'].lower()), None)
        ball = next((o for o in objects if 'ball' in o['category'].lower()), None)

        if not (player_paddle and ball):
            return "HOLD position - Cannot determine game state."

        paddle_y = player_paddle['position'][1]
        ball_y = ball['position'][1]

        # Determine relative position
        diff = ball_y - paddle_y
        if abs(diff) < 10:
            return "HOLD - Paddle well-aligned with ball."
        elif diff < -10:
            return "MOVE UP - Ball is above paddle position."
        else:
            return "MOVE DOWN - Ball is below paddle position."

    def _generate_pong_strategy_quantitative(self, objects: List[Dict[str, Any]]) -> str:
        """Generate Pong strategy with scaled coordinates and measurements."""
        player_paddle = next((o for o in objects if 'player' in o['category'].lower()), None)
        ball = next((o for o in objects if 'ball' in o['category'].lower()), None)

        if not (player_paddle and ball):
            return "HOLD position - Unable to locate paddle or ball."

        # Scale coordinates to 1280x720
        paddle_y = int(player_paddle['position'][1] * self.HEIGHT_SCALE)
        ball_y = int(ball['position'][1] * self.HEIGHT_SCALE)
        ball_vel = ball.get('velocity', [0.0, 0.0])

        # Trajectory prediction (velocity stays the same per frame)
        trajectory = ""
        if abs(ball_vel[1]) > 0.5:  # Ball is moving vertically
            direction = "downward" if ball_vel[1] > 0 else "upward"
            trajectory = f" Ball moving {direction} with velocity y={ball_vel[1]:.1f}."

        # Action recommendation (scale threshold too)
        diff = ball_y - paddle_y
        threshold = int(10 * self.HEIGHT_SCALE)  # ~34 pixels in scaled space

        if abs(diff) < threshold:
            action = "HOLD"
            reasoning = f"Paddle at y={paddle_y} well-aligned with ball at y={ball_y}."
        elif diff < -threshold:
            action = "MOVE UP"
            reasoning = f"Ball at y={ball_y} is above paddle at y={paddle_y}. Gap of {abs(diff):.0f} pixels."
        else:
            action = "MOVE DOWN"
            reasoning = f"Ball at y={ball_y} is below paddle at y={paddle_y}. Gap of {abs(diff):.0f} pixels."

        return f"{action} - {reasoning}{trajectory}"

    def _generate_breakout_strategy_qualitative(self, objects: List[Dict[str, Any]]) -> str:
        """Generate Breakout strategy without exact coordinates."""
        player = next((o for o in objects if 'player' in o['category'].lower()), None)
        ball = next((o for o in objects if 'ball' in o['category'].lower()), None)

        if not ball:
            return "HOLD position - No ball visible."

        if not player:
            return "Move paddle to intercept ball."

        paddle_x = player['position'][0]
        ball_x = ball['position'][0]

        # Determine relative position
        diff = ball_x - paddle_x
        if abs(diff) < 10:
            return "HOLD - Paddle aligned with ball."
        elif diff < -10:
            return "MOVE LEFT - Ball is left of paddle."
        else:
            return "MOVE RIGHT - Ball is right of paddle."

    def _generate_breakout_strategy_quantitative(self, objects: List[Dict[str, Any]]) -> str:
        """Generate Breakout strategy with scaled coordinates and measurements."""
        player = next((o for o in objects if 'player' in o['category'].lower()), None)
        ball = next((o for o in objects if 'ball' in o['category'].lower()), None)

        if not ball:
            return "HOLD position - No ball visible."

        if not player:
            return "Move paddle to intercept ball."

        # Scale coordinates to 1280x720
        paddle_x = int(player['position'][0] * self.WIDTH_SCALE)
        ball_x = int(ball['position'][0] * self.WIDTH_SCALE)
        ball_vel = ball.get('velocity', [0.0, 0.0])

        # Trajectory prediction
        trajectory = ""
        if abs(ball_vel[0]) > 0.5:
            direction = "right" if ball_vel[0] > 0 else "left"
            trajectory = f" Ball moving {direction} with velocity x={ball_vel[0]:.1f}."

        # Action (scale threshold)
        diff = ball_x - paddle_x
        threshold = int(10 * self.WIDTH_SCALE)  # ~80 pixels in scaled space

        if abs(diff) < threshold:
            action = "HOLD"
            reasoning = f"Paddle at x={paddle_x} aligned with ball at x={ball_x}."
        elif diff < -threshold:
            action = "MOVE LEFT"
            reasoning = f"Ball at x={ball_x} is left of paddle at x={paddle_x}."
        else:
            action = "MOVE RIGHT"
            reasoning = f"Ball at x={ball_x} is right of paddle at x={paddle_x}."

        return f"{action} - {reasoning}{trajectory}"

    def _generate_spaceinvaders_strategy_qualitative(self, objects: List[Dict[str, Any]]) -> str:
        """Generate Space Invaders strategy without exact coordinates."""
        player = next((o for o in objects if 'player' in o['category'].lower()), None)
        aliens = [o for o in objects if 'alien' in o['category'].lower() or 'enemy' in o['category'].lower()]
        bullets = [o for o in objects if 'bullet' in o['category'].lower() or 'projectile' in o['category'].lower()]

        if not player:
            return "Unable to assess strategy - player ship not visible."

        player_x = player['position'][0]
        player_y = player['position'][1]

        # Check for immediate threats
        has_threat = False
        for bullet in bullets:
            bullet_y = bullet['position'][1]
            bullet_x = bullet['position'][0]
            # Bullet above player (potential threat)
            if bullet_y < player_y and abs(bullet_x - player_x) < 20:
                has_threat = True
                break

        if has_threat:
            return "EVADE - Immediate threat from enemy fire detected."

        # Find target
        if aliens:
            closest_alien = min(aliens, key=lambda a: abs(a['position'][0] - player_x))
            target_x = closest_alien['position'][0]

            if abs(target_x - player_x) < 10:
                return "FIRE - Alien directly above, shoot now."
            elif target_x < player_x:
                return "MOVE LEFT and FIRE - Target alien on left side."
            else:
                return "MOVE RIGHT and FIRE - Target alien on right side."
        else:
            return "HOLD - No immediate targets or threats."

    def _generate_spaceinvaders_strategy_quantitative(self, objects: List[Dict[str, Any]]) -> str:
        """Generate Space Invaders strategy with scaled coordinates and threat assessment."""
        player = next((o for o in objects if 'player' in o['category'].lower()), None)
        aliens = [o for o in objects if 'alien' in o['category'].lower() or 'enemy' in o['category'].lower()]
        bullets = [o for o in objects if 'bullet' in o['category'].lower() or 'projectile' in o['category'].lower()]

        if not player:
            return "Unable to assess strategy without player ship."

        # Scale coordinates to 1280x720
        player_x = int(player['position'][0] * self.WIDTH_SCALE)
        player_y = int(player['position'][1] * self.HEIGHT_SCALE)

        # Threat assessment
        threats = []
        for bullet in bullets:
            bullet_y = int(bullet['position'][1] * self.HEIGHT_SCALE)
            bullet_x = int(bullet['position'][0] * self.WIDTH_SCALE)

            # Check if bullet is above player (potential threat)
            if bullet_y < player_y:
                distance = abs(bullet_x - player_x)
                close_threshold = int(20 * self.WIDTH_SCALE)  # ~160 pixels
                moderate_threshold = int(50 * self.WIDTH_SCALE)  # ~400 pixels

                if distance < close_threshold:
                    threats.append(f"bullet at x={bullet_x}, close danger")
                elif distance < moderate_threshold:
                    threats.append(f"bullet at x={bullet_x}, moderate threat")

        # Target assessment
        if aliens:
            # Find closest alien (in scaled space)
            alien_distances = [(a, abs(int(a['position'][0] * self.WIDTH_SCALE) - player_x)) for a in aliens]
            closest_alien, _ = min(alien_distances, key=lambda x: x[1])
            target_x = int(closest_alien['position'][0] * self.WIDTH_SCALE)

            alignment_threshold = int(10 * self.WIDTH_SCALE)  # ~80 pixels

            if threats:
                action = "EVADE"
                reasoning = f"Immediate threats detected: {'; '.join(threats)}. Move to avoid."
            elif abs(target_x - player_x) < alignment_threshold:
                action = "FIRE"
                reasoning = f"Alien at x={target_x} directly above player at x={player_x}. Shoot!"
            elif target_x < player_x:
                action = "MOVE LEFT and FIRE"
                reasoning = f"Target alien at x={target_x}, player at x={player_x}."
            else:
                action = "MOVE RIGHT and FIRE"
                reasoning = f"Target alien at x={target_x}, player at x={player_x}."
        else:
            action = "HOLD"
            reasoning = "No immediate targets or threats."

        return f"{action} - {reasoning}"

    # Space Invaders Formation-Based Reference Generation
    def _generate_spaceinvaders_visual_qualitative(self, formation_data: Dict) -> str:
        """Generate visual reference using formation analysis."""
        formation = formation_data['alien_formation']
        total_aliens = formation['total_aliens']
        player_count = formation_data.get('player_count', 0)
        shield_count = formation_data.get('shield_count', 0)
        bullet_count = formation_data.get('bullet_count', 0)

        parts = []

        # Player ship
        if player_count > 0:
            parts.append("a player ship at the bottom")

        # Alien formation
        if total_aliens > 0:
            num_cols = formation['num_columns']
            num_rows = formation['num_rows']
            destroyed = formation['destroyed_count']

            if destroyed == 0:
                parts.append(f"an alien formation with {total_aliens} aliens arranged in {num_cols} columns and {num_rows} rows (formation intact)")
            elif destroyed < 10:
                parts.append(f"an alien formation with approximately {total_aliens} aliens in {num_cols} columns and {num_rows} rows")
            else:
                parts.append(f"an alien formation with approximately {total_aliens} aliens in {num_cols} columns and {num_rows} rows ({formation['destruction_description']})")
        else:
            parts.append("no remaining aliens (all destroyed)")

        # Projectiles/bullets
        if bullet_count > 0:
            parts.append(f"{bullet_count} projectile{'s' if bullet_count > 1 else ''}")

        # Shields
        if shield_count > 0:
            parts.append(f"{shield_count} protective shield{'s' if shield_count > 1 else ''}")

        return f"Key elements: {', '.join(parts)}."

    def _generate_spaceinvaders_visual_quantitative(self, formation_data: Dict) -> str:
        """Generate quantitative visual reference with formation metrics."""
        formation = formation_data['alien_formation']
        total_aliens = formation['total_aliens']
        player_count = formation_data.get('player_count', 0)
        shield_count = formation_data.get('shield_count', 0)

        details = []
        details.append(f"Player ship count: {player_count}")
        details.append(f"Alien formation: {total_aliens} aliens total")
        details.append(f"Formation structure: {formation['num_columns']} columns × {formation['num_rows']} rows")
        details.append(f"Aliens destroyed: {formation['destroyed_count']} ({int(formation['destroyed_ratio']*100)}%)")
        details.append(f"Shield count: {shield_count}")

        if formation['formation_bbox']:
            bbox = formation['formation_bbox']
            details.append(f"Formation center: ({bbox['center_x']}, {bbox['center_y']})")

        return "Detected objects with formation analysis: " + "; ".join(details) + "."

    def _generate_spaceinvaders_spatial_qualitative(self, formation_data: Dict) -> str:
        """Generate spatial reference using formation position."""
        formation = formation_data['alien_formation']

        if formation['total_aliens'] == 0:
            return "Spatial layout: Player ship at bottom, no alien formation remaining."

        bbox = formation['formation_bbox']
        offset = formation['formation_offset']

        # Describe formation position
        parts = []

        # Vertical position
        center_y = bbox['center_y']
        if center_y < 240:
            v_pos = "upper portion"
        elif center_y < 480:
            v_pos = "middle area"
        else:
            v_pos = "lower portion (dangerously close)"

        parts.append(f"The alien formation is positioned in the {v_pos} of the screen")

        # Horizontal bias
        center_x = bbox['center_x']
        if center_x < 400:
            h_bias = "shifted toward the left side"
        elif center_x < 880:
            h_bias = "centered horizontally"
        else:
            h_bias = "shifted toward the right side"

        parts.append(h_bias)

        # Movement description
        dx, dy = offset['dx'], offset['dy']
        if abs(dx) > 20 or abs(dy) > 20:
            parts.append(f"Formation has {formation['movement_description'].lower()}")

        # Destruction pattern
        if formation['destroyed_count'] > 5:
            num_cols = formation['num_columns']
            if num_cols < 9:
                parts.append(f"with edge columns destroyed (narrowed to {num_cols} columns)")

        # Shields
        shield_count = formation_data.get('shield_count', 0)
        if shield_count > 0:
            parts.append(f"{shield_count} protective shields positioned between the formation and player")

        # Bullet positions (critical for spatial verification)
        bullet_analysis = formation_data.get('bullet_analysis', [])
        if bullet_analysis:
            bullet_descriptions = []
            for bullet_info in bullet_analysis:
                desc = bullet_info['relative_to_player']['description']
                bullet_descriptions.append(desc)
            if bullet_descriptions:
                parts.append(f"Projectiles: {', '.join(bullet_descriptions)}")

        return ". ".join(parts) + ". Player ship is at the bottom of the screen."

    def _generate_spaceinvaders_strategy_qualitative_formation(self, formation_data: Dict) -> str:
        """Generate strategy reference using formation state."""
        formation = formation_data['alien_formation']
        total_aliens = formation['total_aliens']

        if total_aliens == 0:
            return "HOLD - All aliens destroyed, level complete."

        bbox = formation['formation_bbox']
        center_y = bbox['center_y']
        center_x = bbox['center_x']
        shield_count = formation_data.get('shield_count', 0)

        # Assess urgency based on formation height
        if center_y > 480:
            urgency = "CRITICAL"
            action = "FIRE RAPIDLY"
            reasoning = f"Alien formation has descended dangerously close (y={center_y}). Shoot immediately to prevent game over."
        elif center_y > 360:
            urgency = "HIGH"
            action = "MOVE and FIRE"
            reasoning = f"Alien formation at y={center_y} is advancing. Position under threats and shoot."
        else:
            urgency = "MODERATE"
            action = "POSITION and FIRE"
            reasoning = f"Alien formation at y={center_y}. Find optimal firing position."

        # Consider destruction pattern
        num_cols = formation['num_columns']
        if num_cols <= 5:
            reasoning += f" Formation narrowed to {num_cols} columns - focus fire on remaining threats."

        # Shield strategy
        if shield_count > 0:
            reasoning += f" Use {shield_count} shield(s) for cover while firing."

        return f"{action} - {reasoning}"

    def _create_readme(self):
        """Create dataset README."""
        readme_content = f"""# Automatic Atari Benchmark Dataset v{self.dataset_metadata['version']}

## Dataset Information

**Created:** {self.dataset_metadata['created']}
**Total Frames:** {self.dataset_metadata['total_frames']}
**Games:** {', '.join(self.dataset_metadata['games'])}
**Random Seed:** {self.seed}

## Complexity Distribution

```json
{json.dumps(self.dataset_metadata['complexity_distribution'], indent=2)}
```

## Directory Structure

```
{self.output_dir.name}/
├── frames/           # Frame images (PNG)
├── ground_truth/     # OCAtari ground truth (JSON)
├── metadata/         # Dataset metadata and index
└── README.md         # This file
```

## Per-Game Breakdown

"""
        for game, stats in self.dataset_metadata['frames_per_game'].items():
            readme_content += f"### {game}\n"
            readme_content += f"- Total frames: {stats['total']}\n"
            readme_content += f"- Complexity distribution: {stats['complexity_distribution']}\n\n"

        readme_content += """
## Tasks

This benchmark evaluates 4 types of reasoning:

1. **Visual Understanding**: Object detection and identification
2. **Spatial Reasoning**: Relative positions and spatial relationships
3. **Strategic Planning**: Game-playing strategy and next actions
4. **Game Identification**: Recognizing the game from visual cues

## Ground Truth Format

Each frame has associated ground truth in JSON format:

```json
{
  "frame_id": "pong_0001_medium",
  "game": "Pong",
  "ocatari_data": {
    "objects": [...],
    "spatial_relationships": {...},
    "action_space": {...}
  }
}
```

## Usage

```python
from automatic_benchmark import BenchmarkDatasetLoader

loader = BenchmarkDatasetLoader('path/to/dataset')
dataset = loader.load()

for frame_data in dataset['frames']:
    frame_id = frame_data['frame_id']
    # Load frame and ground truth
    # Run evaluation
```

## Citation

If you use this benchmark, please cite:

```bibtex
@dataset{atari_automatic_benchmark,
  title={Automatic Atari Spatial-Visual Reasoning Benchmark},
  version={""" + self.dataset_metadata['version'] + """},
  year={2025}
}
```
"""
        readme_path = self.output_dir / "README.md"
        with open(readme_path, 'w') as f:
            f.write(readme_content)
