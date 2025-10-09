"""
Coordinate Accuracy Evaluator for VLM Detection vs OCAtari Ground Truth

This module evaluates the accuracy of VLM object detection against OCAtari ground truth
with fair, game-specific matching strategies and relaxation for partial detections.

Key Features:
- Scales OCAtari coordinates (210 width x 160 height) to VLM coordinates (1280x720)
- Game-specific object matching (Pong, Breakout, SpaceInvaders)
- Fair relaxation for partial detections:
  * Pong: Focus on 2 paddles + ball
  * Breakout: Focus on paddle + ball (relaxed block matching)
  * SpaceInvaders: Focus on player + bullets (relaxed alien matching)
- IoU-based coordinate matching with position tolerance
- Fuzzy semantic matching with position-based disambiguation
"""

import numpy as np
import json
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from pathlib import Path


@dataclass
class DetectionMatch:
    """Result of matching a VLM detection to ground truth."""
    vlm_obj: Dict[str, Any]
    gt_obj: Optional[Dict[str, Any]]
    iou: float
    center_distance: float
    matched: bool
    match_quality: str  # 'exact', 'good', 'fair', 'poor', 'unmatched'


@dataclass
class FrameEvaluationResult:
    """Evaluation result for a single frame."""
    frame_id: int
    game: str

    # Core metrics
    total_gt_objects: int
    total_vlm_detections: int
    matched_objects: int

    # Important objects (game-specific)
    important_gt_count: int
    important_vlm_count: int
    important_matched: int

    # Detailed matches
    matches: List[DetectionMatch]

    # Aggregate metrics
    precision: float
    recall: float
    f1_score: float

    # Important object metrics
    important_precision: float
    important_recall: float
    important_f1: float

    # Position accuracy
    avg_iou: float
    avg_center_distance: float


class CoordinateScaler:
    """Scale coordinates between OCAtari and VLM spaces.

    OCAtari frames are PORTRAIT: 160 width × 210 height
    VLM frames are LANDSCAPE: 1280 width × 720 height

    Both are then scaled/stretched to fit VLM's 1280×720 space.
    """

    OCATARI_WIDTH = 160   # OCAtari is portrait (narrow)
    OCATARI_HEIGHT = 210  # OCAtari is portrait (tall)
    VLM_WIDTH = 1280
    VLM_HEIGHT = 720

    @classmethod
    def ocatari_to_vlm(cls, x: float, y: float) -> Tuple[float, float]:
        """Scale OCAtari coordinates to VLM space."""
        vlm_x = (x / cls.OCATARI_WIDTH) * cls.VLM_WIDTH
        vlm_y = (y / cls.OCATARI_HEIGHT) * cls.VLM_HEIGHT
        return vlm_x, vlm_y

    @classmethod
    def ocatari_bbox_to_vlm(cls, x: float, y: float, w: float, h: float) -> Tuple[float, float, float, float]:
        """Scale OCAtari bounding box to VLM space.

        Args:
            x, y: Top-left corner in OCAtari space
            w, h: Width and height in OCAtari space

        Returns:
            x1, y1, x2, y2 in VLM space
        """
        x1_vlm, y1_vlm = cls.ocatari_to_vlm(x, y)
        x2_vlm, y2_vlm = cls.ocatari_to_vlm(x + w, y + h)
        return x1_vlm, y1_vlm, x2_vlm, y2_vlm


class GameSpecificMatcher:
    """Game-specific object matching and importance classification."""

    # Define important objects for each game
    IMPORTANT_OBJECTS = {
        'pong': {
            'required': ['Player', 'PlayerPaddle', 'EnemyPaddle', 'Ball'],
            'aliases': {
                'player': ['player', 'paddle', 'player paddle', 'green paddle', 'right paddle'],
                'enemy': ['enemy', 'opponent', 'enemy paddle', 'left paddle', 'orange paddle'],
                'ball': ['ball', 'pong ball']
            },
            'exclude_keywords': ['score', 'display', 'digit', 'number', 'lives']  # Exclude score displays
        },
        'breakout': {
            'required': ['Player', 'Ball'],
            'optional': ['Brick', 'BlockRow'],  # Bricks are important but not all need to be detected
            'aliases': {
                'player': ['player', 'paddle'],
                'ball': ['ball'],
                'brick': ['brick', 'block', 'barrier', 'wall', 'blockrow', 'row', 'layer']
            },
            'exclude_keywords': ['score', 'display', 'digit', 'number', 'lives', 'counter', 'indicator', 'interface']
        },
        'spaceinvaders': {
            'required': ['Player'],  # Only player is absolutely critical
            'optional': ['Alien', 'PlayerBullet', 'EnemyBullet', 'Shelter'],  # Others are important but may vary
            'aliases': {
                'player': ['player', 'ship', 'spaceship', 'cannon', 'shooter'],
                'bullet': ['bullet', 'player bullet', 'projectile', 'missile', 'shot'],
                'alien': ['alien', 'enemy', 'invader', 'space invader', 'ufo'],
                'enemy_bullet': ['enemy bullet', 'alien bullet', 'bomb', 'enemy projectile'],
                'shelter': ['shelter', 'shield', 'barrier', 'bunker', 'defense']
            },
            'exclude_keywords': ['score', 'display', 'digit', 'number', 'lives', 'counter', 'hi-score', 'credit']
        }
    }

    @classmethod
    def is_important_object(cls, obj_category: str, game: str) -> bool:
        """Check if an object category is important for the game.

        Uses fuzzy matching with aliases to handle variations in object naming.
        """
        game_lower = game.lower()
        if game_lower not in cls.IMPORTANT_OBJECTS:
            return True  # If game not defined, consider all objects important

        config = cls.IMPORTANT_OBJECTS[game_lower]
        obj_lower = obj_category.lower().strip()

        # First, check exclusion keywords (e.g., "score", "display")
        exclude_keywords = config.get('exclude_keywords', [])
        if any(exclude_kw in obj_lower for exclude_kw in exclude_keywords):
            return False  # Excluded → not important!

        # Check aliases (fuzzy matching)
        aliases = config.get('aliases', {})
        for canonical, alias_list in aliases.items():
            # Check if any alias keyword appears in the object label
            if any(alias in obj_lower for alias in alias_list):
                return True  # Found in aliases → important!

        # Fallback: check raw required/optional lists
        required = config.get('required', [])
        optional = config.get('optional', [])

        # Check with case-insensitive substring matching
        return any(req.lower() in obj_lower for req in required) or \
               any(opt.lower() in obj_lower for opt in optional)

    @classmethod
    def normalize_object_label(cls, label: str, game: str) -> str:
        """Normalize VLM label to match OCAtari categories."""
        game_lower = game.lower()
        label_lower = label.lower().strip()

        if game_lower not in cls.IMPORTANT_OBJECTS:
            return label

        aliases = cls.IMPORTANT_OBJECTS[game_lower].get('aliases', {})

        # Check each alias mapping
        for canonical, alias_list in aliases.items():
            if any(alias in label_lower for alias in alias_list):
                return canonical

        return label_lower

    @classmethod
    def infer_paddle_side(cls, bbox: List[float], game: str) -> str:
        """Infer which side a paddle is on based on x-coordinate.

        Args:
            bbox: Bounding box [x1, y1, x2, y2] in VLM space (1280x720)
            game: Game name

        Returns:
            'left', 'right', 'top', 'bottom', or 'unknown'
        """
        if game.lower() not in ['pong', 'breakout']:
            return 'unknown'

        center_x = (bbox[0] + bbox[2]) / 2
        center_y = (bbox[1] + bbox[3]) / 2

        # For Pong (paddles on left/right sides)
        if game.lower() == 'pong':
            # Pong paddles: left is < 25% width, right is > 75% width
            if center_x < 320:  # Left 25% of screen (1280 * 0.25)
                return 'left'
            elif center_x > 960:  # Right 25% of screen (1280 * 0.75)
                return 'right'
            else:
                return 'center'  # Might be ball or other object

        # For Breakout (paddle at bottom)
        elif game.lower() == 'breakout':
            if center_y > 540:  # Bottom 25% of screen (720 * 0.75)
                return 'bottom'
            else:
                return 'top'

        return 'unknown'

    @classmethod
    def fuzzy_match_objects(cls, gt_obj: Dict, vlm_obj: Dict, game: str) -> float:
        """Calculate semantic similarity between ground truth and VLM object.

        Returns a score from 0.0 to 1.0 indicating how well the objects match
        semantically (by type/category), considering spatial position.

        Args:
            gt_obj: Ground truth object with 'category' and 'bbox'
            vlm_obj: VLM detection with 'label', 'normalized_label' and 'bbox'
            game: Game name for context

        Returns:
            Similarity score 0.0-1.0 (1.0 = perfect match)
        """
        gt_category = gt_obj.get('category', '').lower()
        vlm_label = vlm_obj.get('label', '').lower()
        vlm_normalized = vlm_obj.get('normalized_label', '').lower()

        # Perfect match on normalized label
        if vlm_normalized == gt_category.lower():
            return 1.0

        # For Pong: Handle paddle disambiguation by position
        if game.lower() == 'pong' and gt_category == 'player':
            # OCAtari calls both paddles "Player", we need to use position
            gt_side = cls.infer_paddle_side(gt_obj['bbox'], game)
            vlm_side_hints = {
                'left': ['left', 'opponent', 'enemy', 'orange'],
                'right': ['right', 'player', 'green'],
            }

            # Check if VLM label hints at the same side as GT position
            for side, keywords in vlm_side_hints.items():
                if gt_side == side and any(kw in vlm_label for kw in keywords):
                    # Position and label agree → good match!
                    if any(kw in vlm_label for kw in ['paddle', 'player']):
                        return 0.9  # Strong match
                    return 0.7  # Moderate match

            # Generic paddle match (position unknown or mismatch)
            if any(kw in vlm_label for kw in ['paddle', 'player']) and \
               any(kw not in vlm_label for kw in ['score', 'display']):
                return 0.5  # Weak match (paddle but wrong side?)

        # Ball matching
        if 'ball' in gt_category and 'ball' in vlm_label:
            return 0.95

        # Generic paddle/player matching
        if 'player' in gt_category or 'paddle' in gt_category:
            if any(kw in vlm_label for kw in ['paddle', 'player']) and \
               'score' not in vlm_label:
                return 0.8  # Increased from 0.6 to prioritize paddle matching

        # Brick/block matching for Breakout
        if game.lower() == 'breakout':
            # Paddle matching for Breakout (critical!)
            if 'player' in gt_category and 'paddle' in vlm_label:
                # Check if it's at the bottom of the screen (y > 600)
                gt_y_center = (gt_obj['bbox'][1] + gt_obj['bbox'][3]) / 2
                vlm_y_center = (vlm_obj['bbox'][1] + vlm_obj['bbox'][3]) / 2
                if gt_y_center > 600 and vlm_y_center > 600:
                    return 0.95  # Strong match for bottom paddle
                return 0.8  # Good match for paddle anywhere

            # Brick/block matching
            if any(kw in gt_category.lower() for kw in ['brick', 'block', 'blockrow']):
                if any(kw in vlm_label for kw in ['brick', 'block', 'barrier', 'row', 'layer']):
                    return 0.7  # Good match for bricks

        # SpaceInvaders matching
        if game.lower() == 'spaceinvaders':
            # Player ship matching
            if 'player' in gt_category:
                if any(kw in vlm_label for kw in ['player', 'ship', 'cannon', 'shooter']):
                    return 0.9

            # Alien matching
            if 'alien' in gt_category.lower():
                if any(kw in vlm_label for kw in ['alien', 'invader', 'enemy', 'ufo']):
                    return 0.8

            # Bullet matching
            if 'bullet' in gt_category.lower():
                if any(kw in vlm_label for kw in ['bullet', 'projectile', 'shot', 'missile']):
                    return 0.85

        # No match
        return 0.0

    @classmethod
    def get_relaxation_config(cls, game: str) -> Dict[str, Any]:
        """Get relaxation configuration for game-specific matching."""
        game_lower = game.lower()

        if game_lower == 'pong':
            return {
                'focus_on_important': True,
                'allow_partial_optional': False,
                'min_important_recall': 0.66,  # At least 2 out of 3 (2 paddles + ball)
                'iou_threshold': 0.3,
                'position_tolerance': 30  # pixels
            }
        elif game_lower == 'breakout':
            return {
                'focus_on_important': True,
                'allow_partial_optional': True,  # Don't require all bricks
                'min_important_recall': 1.0,  # Paddle + ball are CRITICAL (must detect both!)
                'brick_relaxation': 0.5,  # Only need 50% of bricks detected
                'iou_threshold': 0.25,  # More lenient for paddle (can have slight misalignment)
                'position_tolerance': 50  # More lenient with paddle position (50px tolerance)
            }
        elif game_lower == 'spaceinvaders':
            return {
                'focus_on_important': True,
                'allow_partial_optional': True,  # Don't require all aliens/bullets
                'min_important_recall': 0.5,  # Player is critical, bullets/aliens optional
                'alien_relaxation': 0.3,  # Only need 30% of aliens detected
                'bullet_relaxation': 0.2,  # Only need 20% of bullets detected
                'iou_threshold': 0.2,  # Very relaxed for small objects (aliens, bullets)
                'position_tolerance': 50  # Relaxed tolerance for small fast-moving objects
            }
        else:
            return {
                'focus_on_important': False,
                'allow_partial_optional': False,
                'min_important_recall': 0.8,
                'iou_threshold': 0.3,
                'position_tolerance': 30
            }


def calculate_iou(bbox1: Tuple[float, float, float, float],
                  bbox2: Tuple[float, float, float, float]) -> float:
    """Calculate Intersection over Union for two bounding boxes.

    Args:
        bbox1, bbox2: Bounding boxes in format (x1, y1, x2, y2)

    Returns:
        IoU score between 0 and 1
    """
    x1_1, y1_1, x2_1, y2_1 = bbox1
    x1_2, y1_2, x2_2, y2_2 = bbox2

    # Calculate intersection
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)

    if x2_i < x1_i or y2_i < y1_i:
        return 0.0

    intersection = (x2_i - x1_i) * (y2_i - y1_i)

    # Calculate union
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection

    if union == 0:
        return 0.0

    return intersection / union


def calculate_center_distance(bbox1: Tuple[float, float, float, float],
                              bbox2: Tuple[float, float, float, float]) -> float:
    """Calculate Euclidean distance between bounding box centers."""
    cx1 = (bbox1[0] + bbox1[2]) / 2
    cy1 = (bbox1[1] + bbox1[3]) / 2
    cx2 = (bbox2[0] + bbox2[2]) / 2
    cy2 = (bbox2[1] + bbox2[3]) / 2

    return np.sqrt((cx1 - cx2)**2 + (cy1 - cy2)**2)


class CoordinateAccuracyEvaluator:
    """Main evaluator for comparing VLM detections with OCAtari ground truth."""

    def __init__(self, game: str):
        """Initialize evaluator for a specific game.

        Args:
            game: Game name ('Pong', 'Breakout', 'SpaceInvaders')
        """
        self.game = game
        self.scaler = CoordinateScaler()
        self.matcher = GameSpecificMatcher()
        self.config = self.matcher.get_relaxation_config(game)

    def prepare_ocatari_objects(self, ocatari_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Convert OCAtari objects to standardized format with scaled coordinates.

        Args:
            ocatari_data: OCAtari data with objects in original (210x160) space

        Returns:
            List of objects with coordinates scaled to VLM space (1280x720)
        """
        objects = []

        for obj in ocatari_data.get('objects', []):
            # Get position and size
            x, y = obj.get('position', (0, 0))
            w, h = obj.get('size', (0, 0))

            # If size is 0, use a small default (OCAtari sometimes doesn't give size)
            if w == 0 or h == 0:
                w, h = 4, 4  # Small default size

            # Scale to VLM coordinates
            x1_vlm, y1_vlm, x2_vlm, y2_vlm = self.scaler.ocatari_bbox_to_vlm(x, y, w, h)

            objects.append({
                'id': obj.get('id', len(objects)),
                'category': obj.get('category', 'Unknown'),
                'bbox': [x1_vlm, y1_vlm, x2_vlm, y2_vlm],
                'center': ((x1_vlm + x2_vlm) / 2, (y1_vlm + y2_vlm) / 2),
                'is_important': self.matcher.is_important_object(obj.get('category', ''), self.game)
            })

        return objects

    def prepare_vlm_objects(self, vlm_detections: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Convert VLM detections to standardized format.

        Args:
            vlm_detections: VLM detection results (already in 1280x720 space)

        Returns:
            List of standardized detection objects
        """
        objects = []

        for obj in vlm_detections.get('objects', []):
            bbox = obj.get('bbox', obj.get('coordinates', []))

            if len(bbox) != 4:
                continue

            # Normalize label
            label = obj.get('label', 'unknown')
            normalized_label = self.matcher.normalize_object_label(label, self.game)

            objects.append({
                'id': obj.get('id', len(objects)),
                'label': label,
                'normalized_label': normalized_label,
                'bbox': bbox,
                'center': ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2),
                'confidence': obj.get('confidence', 1.0),
                'is_important': self.matcher.is_important_object(label, self.game)
            })

        return objects

    def match_objects(self, gt_objects: List[Dict], vlm_objects: List[Dict]) -> List[DetectionMatch]:
        """Match VLM detections to ground truth objects using Hungarian algorithm.

        Args:
            gt_objects: Ground truth objects (OCAtari, scaled to VLM space)
            vlm_objects: VLM detections

        Returns:
            List of DetectionMatch objects
        """
        if not gt_objects or not vlm_objects:
            # Handle empty cases
            matches = []
            for gt_obj in gt_objects:
                matches.append(DetectionMatch(
                    vlm_obj={},
                    gt_obj=gt_obj,
                    iou=0.0,
                    center_distance=float('inf'),
                    matched=False,
                    match_quality='unmatched'
                ))
            for vlm_obj in vlm_objects:
                matches.append(DetectionMatch(
                    vlm_obj=vlm_obj,
                    gt_obj=None,
                    iou=0.0,
                    center_distance=float('inf'),
                    matched=False,
                    match_quality='unmatched'
                ))
            return matches

        # Build cost matrix (negative IoU + center distance penalty)
        n_gt = len(gt_objects)
        n_vlm = len(vlm_objects)
        cost_matrix = np.zeros((n_gt, n_vlm))

        for i, gt_obj in enumerate(gt_objects):
            for j, vlm_obj in enumerate(vlm_objects):
                iou = calculate_iou(gt_obj['bbox'], vlm_obj['bbox'])
                center_dist = calculate_center_distance(gt_obj['bbox'], vlm_obj['bbox'])

                # NEW: Fuzzy semantic matching bonus
                semantic_score = self.matcher.fuzzy_match_objects(gt_obj, vlm_obj, self.game)

                # Cost is negative IoU plus normalized distance penalty, minus semantic bonus
                # Lower cost = better match
                # Semantic bonus helps match paddles even when position is slightly off
                cost_matrix[i, j] = -iou + (center_dist / self.config['position_tolerance']) * 0.3 - semantic_score * 0.5

        # Use greedy matching (can be replaced with Hungarian algorithm for optimal)
        from scipy.optimize import linear_sum_assignment
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        matches = []
        matched_gt = set()
        matched_vlm = set()

        # Process matched pairs
        for i, j in zip(row_ind, col_ind):
            gt_obj = gt_objects[i]
            vlm_obj = vlm_objects[j]

            iou = calculate_iou(gt_obj['bbox'], vlm_obj['bbox'])
            center_dist = calculate_center_distance(gt_obj['bbox'], vlm_obj['bbox'])

            # Determine if match is valid based on thresholds
            is_match = (iou >= self.config['iou_threshold'] or
                       center_dist <= self.config['position_tolerance'])

            if is_match:
                # Determine match quality
                if iou >= 0.7 and center_dist <= 10:
                    quality = 'exact'
                elif iou >= 0.5 and center_dist <= 20:
                    quality = 'good'
                elif iou >= 0.3 and center_dist <= 30:
                    quality = 'fair'
                else:
                    quality = 'poor'

                matches.append(DetectionMatch(
                    vlm_obj=vlm_obj,
                    gt_obj=gt_obj,
                    iou=iou,
                    center_distance=center_dist,
                    matched=True,
                    match_quality=quality
                ))
                matched_gt.add(i)
                matched_vlm.add(j)

        # Add unmatched ground truth objects
        for i, gt_obj in enumerate(gt_objects):
            if i not in matched_gt:
                matches.append(DetectionMatch(
                    vlm_obj={},
                    gt_obj=gt_obj,
                    iou=0.0,
                    center_distance=float('inf'),
                    matched=False,
                    match_quality='unmatched'
                ))

        # Add unmatched VLM detections (false positives)
        for j, vlm_obj in enumerate(vlm_objects):
            if j not in matched_vlm:
                matches.append(DetectionMatch(
                    vlm_obj=vlm_obj,
                    gt_obj=None,
                    iou=0.0,
                    center_distance=float('inf'),
                    matched=False,
                    match_quality='unmatched'
                ))

        return matches

    def evaluate_frame(self, ocatari_data: Dict, vlm_detections: Dict, frame_id: int = 0) -> FrameEvaluationResult:
        """Evaluate VLM detections against OCAtari ground truth for a single frame.

        Args:
            ocatari_data: OCAtari ground truth data
            vlm_detections: VLM detection results
            frame_id: Frame identifier

        Returns:
            FrameEvaluationResult with all metrics
        """
        # Prepare objects
        gt_objects = self.prepare_ocatari_objects(ocatari_data)
        vlm_objects = self.prepare_vlm_objects(vlm_detections)

        # Match objects
        matches = self.match_objects(gt_objects, vlm_objects)

        # Calculate metrics
        total_gt = len(gt_objects)
        total_vlm = len(vlm_objects)
        matched = sum(1 for m in matches if m.matched)

        # Important object metrics
        important_gt = [obj for obj in gt_objects if obj['is_important']]
        important_vlm = [obj for obj in vlm_objects if obj['is_important']]
        important_matches = [m for m in matches if m.matched and m.gt_obj and m.gt_obj['is_important']]

        # Calculate precision, recall, F1 (using IMPORTANT objects only)
        # Don't penalize for detecting non-important objects that GT missed (like score displays)
        precision = len(important_matches) / len(important_vlm) if important_vlm else 0.0
        recall = len(important_matches) / len(important_gt) if important_gt else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        # Important object metrics (same as above now)
        imp_precision = precision
        imp_recall = recall
        imp_f1 = f1

        # Position accuracy
        matched_with_gt = [m for m in matches if m.matched and m.gt_obj]
        avg_iou = np.mean([m.iou for m in matched_with_gt]) if matched_with_gt else 0.0
        avg_center_dist = np.mean([m.center_distance for m in matched_with_gt]) if matched_with_gt else 0.0

        return FrameEvaluationResult(
            frame_id=frame_id,
            game=self.game,
            total_gt_objects=total_gt,
            total_vlm_detections=total_vlm,
            matched_objects=matched,
            important_gt_count=len(important_gt),
            important_vlm_count=len(important_vlm),
            important_matched=len(important_matches),
            matches=matches,
            precision=precision,
            recall=recall,
            f1_score=f1,
            important_precision=imp_precision,
            important_recall=imp_recall,
            important_f1=imp_f1,
            avg_iou=avg_iou,
            avg_center_distance=avg_center_dist
        )

    def generate_report(self, result: FrameEvaluationResult) -> str:
        """Generate human-readable evaluation report."""
        report = f"""
{'='*80}
COORDINATE ACCURACY EVALUATION REPORT
{'='*80}
Game: {result.game}
Frame: {result.frame_id}

OVERALL DETECTION METRICS:
  Ground Truth Objects: {result.total_gt_objects}
  VLM Detections: {result.total_vlm_detections}
  Matched Objects: {result.matched_objects}

  Precision: {result.precision:.3f}
  Recall: {result.recall:.3f}
  F1 Score: {result.f1_score:.3f}

IMPORTANT OBJECTS ONLY (Game-Specific):
  Important GT Objects: {result.important_gt_count}
  Important VLM Detections: {result.important_vlm_count}
  Important Matched: {result.important_matched}

  Important Precision: {result.important_precision:.3f}
  Important Recall: {result.important_recall:.3f}
  Important F1: {result.important_f1:.3f}

POSITION ACCURACY:
  Average IoU: {result.avg_iou:.3f}
  Average Center Distance: {result.avg_center_distance:.1f} pixels

MATCH QUALITY BREAKDOWN:
"""
        # Count match qualities
        quality_counts = {}
        for match in result.matches:
            quality_counts[match.match_quality] = quality_counts.get(match.match_quality, 0) + 1

        for quality, count in sorted(quality_counts.items()):
            report += f"  {quality.capitalize()}: {count}\n"

        report += f"\n{'='*80}\n"

        return report


if __name__ == "__main__":
    # Example usage
    print("Coordinate Accuracy Evaluator")
    print("=" * 80)

    # Test with sample data
    sample_ocatari = {
        'objects': [
            {'id': 0, 'category': 'Player', 'position': (80, 190), 'size': (8, 12)},
            {'id': 1, 'category': 'Ball', 'position': (75, 100), 'size': (4, 4)},
            {'id': 2, 'category': 'EnemyPaddle', 'position': (10, 90), 'size': (8, 12)}
        ]
    }

    sample_vlm = {
        'objects': [
            {'label': 'player paddle', 'coordinates': [640, 684, 704, 738], 'confidence': 0.95},
            {'label': 'ball', 'coordinates': [595, 338, 619, 362], 'confidence': 0.92},
            {'label': 'opponent paddle', 'coordinates': [70, 300, 134, 354], 'confidence': 0.88}
        ]
    }

    evaluator = CoordinateAccuracyEvaluator('Pong')
    result = evaluator.evaluate_frame(sample_ocatari, sample_vlm, frame_id=1)

    print(evaluator.generate_report(result))
