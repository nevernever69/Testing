"""
Space Invaders formation tracking and analysis.

Instead of treating 30+ aliens as independent objects, we track:
- Alien formation as a structured grid
- Column-wise alien counts
- Formation movement from initial position
- Destruction patterns
"""

import numpy as np
from typing import Dict, List, Tuple, Any


class AlienFormation:
    """
    Track Space Invaders alien formation as a structured grid.

    Initial formation (typical):
        Column: 0  1  2  3  4  5  6  7  8  9  10
        Row 0:  A  A  A  A  A  A  A  A  A  A  A   (11 aliens)
        Row 1:  A  A  A  A  A  A  A  A  A  A  A   (11 aliens)
        Row 2:  A  A  A  A  A  A  A  A  A  A  A   (11 aliens)
        Row 3:  A  A  A  A  A  A  A  A  A  A  A   (11 aliens)
        Row 4:  A  A  A  A  A  A  A  A  A  A  A   (11 aliens)
    """

    # Initial formation parameters (from OCAtari)
    INITIAL_COLUMNS = 11
    INITIAL_ROWS = 5
    INITIAL_TOTAL = 55  # 11 * 5

    # Typical initial position (approximate)
    INITIAL_X_START = 200  # Left edge of formation
    INITIAL_Y_START = 100  # Top edge of formation

    def __init__(self, aliens: List[Dict[str, Any]]):
        """
        Initialize formation from OCAtari alien detections.

        Args:
            aliens: List of alien objects from OCAtari with x, y coordinates
        """
        self.aliens = aliens
        self.alien_count = len(aliens)

        if self.alien_count == 0:
            self.formation_bbox = None
            self.column_counts = {}
            self.row_counts = {}
            self.formation_offset = (0, 0)
            self.destroyed_ratio = 1.0
            return

        # Calculate formation bounding box
        self.formation_bbox = self._calculate_bbox()

        # Analyze formation structure
        self.column_counts = self._count_by_column()
        self.row_counts = self._count_by_row()

        # Calculate movement from initial position
        self.formation_offset = self._calculate_offset()

        # Calculate destruction ratio
        self.destroyed_ratio = 1.0 - (self.alien_count / self.INITIAL_TOTAL)

    def _calculate_bbox(self) -> Dict[str, int]:
        """Calculate bounding box of entire formation."""
        x_coords = [a['x'] for a in self.aliens]
        y_coords = [a['y'] for a in self.aliens]

        return {
            'x_min': min(x_coords),
            'x_max': max(x_coords),
            'y_min': min(y_coords),
            'y_max': max(y_coords),
            'center_x': (min(x_coords) + max(x_coords)) // 2,
            'center_y': (min(y_coords) + max(y_coords)) // 2,
            'width': max(x_coords) - min(x_coords),
            'height': max(y_coords) - min(y_coords)
        }

    def _count_by_column(self) -> Dict[int, int]:
        """
        Count aliens in each vertical column.
        Returns dict: {column_index: alien_count}
        """
        if not self.aliens:
            return {}

        # Group aliens by x-coordinate (with tolerance)
        x_tolerance = 15  # Pixels
        columns = {}

        # Sort aliens by x position
        sorted_aliens = sorted(self.aliens, key=lambda a: a['x'])

        current_column = 0
        last_x = sorted_aliens[0]['x']
        columns[current_column] = 1

        for alien in sorted_aliens[1:]:
            if abs(alien['x'] - last_x) > x_tolerance:
                # New column
                current_column += 1
                columns[current_column] = 1
            else:
                # Same column
                columns[current_column] += 1
            last_x = alien['x']

        return columns

    def _count_by_row(self) -> Dict[int, int]:
        """
        Count aliens in each horizontal row.
        Returns dict: {row_index: alien_count}
        """
        if not self.aliens:
            return {}

        # Group aliens by y-coordinate (with tolerance)
        y_tolerance = 15  # Pixels
        rows = {}

        # Sort aliens by y position
        sorted_aliens = sorted(self.aliens, key=lambda a: a['y'])

        current_row = 0
        last_y = sorted_aliens[0]['y']
        rows[current_row] = 1

        for alien in sorted_aliens[1:]:
            if abs(alien['y'] - last_y) > y_tolerance:
                # New row
                current_row += 1
                rows[current_row] = 1
            else:
                # Same row
                rows[current_row] += 1
            last_y = alien['y']

        return rows

    def _calculate_offset(self) -> Tuple[int, int]:
        """
        Calculate formation movement from initial position.
        Returns (dx, dy) where:
            dx > 0: moved right
            dx < 0: moved left
            dy > 0: moved down
            dy < 0: moved up (rare)
        """
        if not self.formation_bbox:
            return (0, 0)

        dx = self.formation_bbox['x_min'] - self.INITIAL_X_START
        dy = self.formation_bbox['y_min'] - self.INITIAL_Y_START

        return (dx, dy)

    def get_destruction_description(self) -> str:
        """Get human-readable description of which parts are destroyed."""
        if self.alien_count == 0:
            return "All aliens destroyed"

        if self.alien_count == self.INITIAL_TOTAL:
            return "Formation intact (no aliens destroyed)"

        # Analyze column destruction
        num_columns = len(self.column_counts)
        max_aliens_per_column = max(self.column_counts.values())
        min_aliens_per_column = min(self.column_counts.values())

        descriptions = []

        # Check if edges are destroyed
        if num_columns < self.INITIAL_COLUMNS:
            descriptions.append(f"Formation narrowed to {num_columns} columns (from {self.INITIAL_COLUMNS})")

        # Check if bottom rows are destroyed
        num_rows = len(self.row_counts)
        if num_rows < self.INITIAL_ROWS:
            descriptions.append(f"Bottom rows destroyed ({num_rows} rows remain)")

        # Overall destruction
        descriptions.append(f"{self.alien_count}/{self.INITIAL_TOTAL} aliens remain ({int((1-self.destroyed_ratio)*100)}%)")

        return "; ".join(descriptions)

    def get_movement_description(self) -> str:
        """Get human-readable description of formation movement."""
        if not self.formation_bbox:
            return "No formation detected"

        dx, dy = self.formation_offset

        horizontal = ""
        if abs(dx) < 10:
            horizontal = "centered horizontally"
        elif dx > 0:
            horizontal = f"shifted {dx}px right"
        else:
            horizontal = f"shifted {abs(dx)}px left"

        vertical = ""
        if abs(dy) < 10:
            vertical = "at initial height"
        elif dy > 0:
            vertical = f"descended {dy}px"
        else:
            vertical = f"ascended {abs(dy)}px"

        return f"Formation {horizontal}, {vertical}"

    def to_dict(self) -> Dict[str, Any]:
        """Export formation analysis as dict for ground truth."""
        return {
            'total_aliens': self.alien_count,
            'destroyed_count': self.INITIAL_TOTAL - self.alien_count,
            'destroyed_ratio': self.destroyed_ratio,
            'column_counts': self.column_counts,
            'row_counts': self.row_counts,
            'num_columns': len(self.column_counts),
            'num_rows': len(self.row_counts),
            'formation_bbox': self.formation_bbox,
            'formation_offset': {
                'dx': self.formation_offset[0],
                'dy': self.formation_offset[1]
            },
            'destruction_description': self.get_destruction_description(),
            'movement_description': self.get_movement_description()
        }


def analyze_spaceinvaders_frame(ocatari_objects: List[Dict]) -> Dict[str, Any]:
    """
    Analyze Space Invaders frame with formation tracking.

    Args:
        ocatari_objects: List of objects from OCAtari detection

    Returns:
        Enhanced ground truth with formation analysis
    """
    # Separate aliens from other objects
    # Convert position format for AlienFormation class
    aliens_raw = [obj for obj in ocatari_objects if 'alien' in obj.get('category', '').lower()]
    aliens = []
    for obj in aliens_raw:
        pos = obj.get('position', [0, 0])
        aliens.append({
            'x': pos[0] if isinstance(pos, list) else getattr(pos, 'x', 0),
            'y': pos[1] if isinstance(pos, list) else getattr(pos, 'y', 0),
            'category': obj.get('category')
        })

    player = [obj for obj in ocatari_objects if 'player' in obj.get('category', '').lower()]
    shields = [obj for obj in ocatari_objects if 'shield' in obj.get('category', '').lower() or 'bunker' in obj.get('category', '').lower()]
    bullets = [obj for obj in ocatari_objects if 'bullet' in obj.get('category', '').lower()]

    # Analyze alien formation
    formation = AlienFormation(aliens)

    # Analyze bullet positions relative to player and formation
    bullet_analysis = []
    if bullets and player:
        player_pos = player[0].get('position', [0, 0])
        player_x = player_pos[0] if isinstance(player_pos, list) else getattr(player_pos, 'x', 0)
        player_y = player_pos[1] if isinstance(player_pos, list) else getattr(player_pos, 'y', 0)

        formation_bbox = formation.formation_bbox if formation.formation_bbox else None

        for i, bullet in enumerate(bullets):
            bullet_pos = bullet.get('position', [0, 0])
            bullet_x = bullet_pos[0] if isinstance(bullet_pos, list) else getattr(bullet_pos, 'x', 0)
            bullet_y = bullet_pos[1] if isinstance(bullet_pos, list) else getattr(bullet_pos, 'y', 0)

            analysis = {
                'bullet_index': i,
                'position': {'x': bullet_x, 'y': bullet_y},
                'relative_to_player': {
                    'above': bullet_y < player_y,
                    'horizontal_offset': bullet_x - player_x,
                    'description': 'above player' if bullet_y < player_y else 'below player'
                }
            }

            # Check if bullet is near formation
            if formation_bbox:
                near_formation = (
                    formation_bbox['x_min'] - 50 < bullet_x < formation_bbox['x_max'] + 50 and
                    bullet_y < formation_bbox['y_max'] + 50
                )
                analysis['near_formation'] = near_formation
                if near_formation:
                    analysis['relative_to_player']['description'] += ', near alien formation'

            bullet_analysis.append(analysis)

    return {
        'alien_formation': formation.to_dict(),
        'player_count': len(player),
        'shield_count': len(shields),
        'bullet_count': len(bullets),
        'bullet_analysis': bullet_analysis,
        'objects': {
            'aliens': aliens,
            'player': player,
            'shields': shields,
            'bullets': bullets
        }
    }


def compare_spaceinvaders_response(
    vlm_response: str,
    formation_data: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Compare VLM response against formation-based ground truth.

    Instead of checking "did you find all 30 aliens?", we check:
    - Did VLM notice formation destruction?
    - Did VLM describe formation movement?
    - Did VLM count columns/rows reasonably?
    """
    response_lower = vlm_response.lower()

    formation = formation_data['alien_formation']
    total_aliens = formation['total_aliens']
    destroyed_count = formation['destroyed_count']

    scores = {
        'formation_awareness': 0.0,
        'destruction_awareness': 0.0,
        'movement_awareness': 0.0,
        'count_accuracy': 0.0
    }

    # Check formation awareness
    if 'formation' in response_lower or 'grid' in response_lower or 'rows' in response_lower:
        scores['formation_awareness'] = 1.0
    elif 'aliens' in response_lower or 'invaders' in response_lower:
        scores['formation_awareness'] = 0.5

    # Check destruction awareness
    if destroyed_count > 10:  # Significant destruction
        if 'destroyed' in response_lower or 'missing' in response_lower or 'fewer' in response_lower:
            scores['destruction_awareness'] = 1.0
        elif total_aliens < 30 and any(str(total_aliens + i) in response_lower for i in range(-5, 6)):
            # VLM gave approximate count
            scores['destruction_awareness'] = 0.7
    else:
        # Not much destruction, shouldn't mention it
        if 'destroyed' not in response_lower:
            scores['destruction_awareness'] = 1.0

    # Check movement awareness (for spatial/strategy tasks)
    dx, dy = formation['formation_offset']['dx'], formation['formation_offset']['dy']
    if abs(dx) > 50 or abs(dy) > 50:
        # Significant movement
        if ('moved' in response_lower or 'shifted' in response_lower or
            'descended' in response_lower or 'lower' in response_lower):
            scores['movement_awareness'] = 1.0
    else:
        scores['movement_awareness'] = 0.5  # Neutral

    # Count accuracy - be lenient!
    # Don't expect VLM to count 30+ individual aliens
    if total_aliens > 20:
        # Many aliens - accept "many", "lots", or rough count
        if any(word in response_lower for word in ['many', 'multiple', 'numerous', 'several']):
            scores['count_accuracy'] = 1.0
        elif any(str(total_aliens + i) in response_lower for i in range(-10, 11)):
            scores['count_accuracy'] = 0.8
    else:
        # Fewer aliens - expect more accuracy
        if any(str(total_aliens + i) in response_lower for i in range(-3, 4)):
            scores['count_accuracy'] = 1.0
        elif any(word in response_lower for word in ['few', 'several']):
            scores['count_accuracy'] = 0.7

    return {
        'scores': scores,
        'average_score': sum(scores.values()) / len(scores),
        'formation_context': formation
    }
