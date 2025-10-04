"""
Scenario-based dataset creator.
Generates distinct strategic scenarios per game for comprehensive benchmarking.
Pong: 10 scenarios, Breakout: 10 scenarios, SpaceInvaders: 10 scenarios.
Total: 30 frames (simplified & more relaxed for higher success rate).
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import sys
from pathlib import Path
import json
import cv2

# Add parent directory to path to import ocatari_ground_truth
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ocatari_ground_truth import OCAtariGroundTruth
from ..config import GAMES, RANDOM_SEED


@dataclass
class Scenario:
    """Defines a strategic scenario to test."""
    name: str
    description: str
    validator: callable  # Function that checks if frame matches scenario
    priority: int = 1  # Higher priority scenarios checked first


class ScenarioBasedDatasetCreator:
    """
    Creates dataset with distinct scenarios per game.
    Pong: 10 scenarios, Breakout: 10 scenarios, SpaceInvaders: 10 scenarios.
    Total: 30 frames. Simplified with relaxed validators for high success rate.
    """

    def __init__(self, output_dir: str = "./benchmark_v2.0"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Directories
        self.frames_dir = self.output_dir / "frames"
        self.ground_truth_dir = self.output_dir / "ground_truth"
        self.metadata_dir = self.output_dir / "metadata"
        self.frames_dir.mkdir(exist_ok=True)
        self.ground_truth_dir.mkdir(exist_ok=True)
        self.metadata_dir.mkdir(exist_ok=True)

        np.random.seed(RANDOM_SEED)

        # Define scenarios for each game
        self.scenarios = {
            'Pong': self._get_pong_scenarios(),
            'Breakout': self._get_breakout_scenarios(),
            'SpaceInvaders': self._get_spaceinvaders_scenarios()
        }

    def _get_pong_scenarios(self) -> List[Scenario]:
        """Define 10 relaxed scenarios for Pong."""
        return [
            Scenario(
                name="ball_right_side",
                description="Ball in right area",
                validator=lambda objs: self._pong_ball_zone(objs, 'right')
            ),
            Scenario(
                name="ball_left_side",
                description="Ball in left area",
                validator=lambda objs: self._pong_ball_zone(objs, 'left')
            ),
            Scenario(
                name="ball_center_area",
                description="Ball in center area",
                validator=lambda objs: self._pong_ball_zone(objs, 'center')
            ),
            Scenario(
                name="ball_top_half",
                description="Ball in top half",
                validator=lambda objs: self._pong_ball_vertical(objs, 'top')
            ),
            Scenario(
                name="ball_bottom_half",
                description="Ball in bottom half",
                validator=lambda objs: self._pong_ball_vertical(objs, 'bottom')
            ),
            Scenario(
                name="paddle_top_position",
                description="Player paddle in top area",
                validator=lambda objs: self._pong_paddle_zone(objs, 'top')
            ),
            Scenario(
                name="paddle_bottom_position",
                description="Player paddle in bottom area",
                validator=lambda objs: self._pong_paddle_zone(objs, 'bottom')
            ),
            Scenario(
                name="paddle_center_position",
                description="Player paddle in center area",
                validator=lambda objs: self._pong_paddle_zone(objs, 'center')
            ),
            Scenario(
                name="game_active_state1",
                description="Active game state variant 1",
                validator=lambda objs: self._pong_active_game(objs, 1)
            ),
            Scenario(
                name="game_active_state2",
                description="Active game state variant 2",
                validator=lambda objs: self._pong_active_game(objs, 2)
            )
        ]

    def _get_breakout_scenarios(self) -> List[Scenario]:
        """Define 10 relaxed scenarios for Breakout."""
        return [
            Scenario(
                name="ball_lower_area",
                description="Ball in lower half",
                validator=lambda objs: self._breakout_ball_zone(objs, 'lower')
            ),
            Scenario(
                name="ball_upper_area",
                description="Ball in upper half",
                validator=lambda objs: self._breakout_ball_zone(objs, 'upper')
            ),
            Scenario(
                name="ball_left_side",
                description="Ball on left side",
                validator=lambda objs: self._breakout_ball_horizontal(objs, 'left')
            ),
            Scenario(
                name="ball_right_side",
                description="Ball on right side",
                validator=lambda objs: self._breakout_ball_horizontal(objs, 'right')
            ),
            Scenario(
                name="ball_center_horizontal",
                description="Ball in center horizontally",
                validator=lambda objs: self._breakout_ball_horizontal(objs, 'center')
            ),
            Scenario(
                name="paddle_left_area",
                description="Paddle on left side",
                validator=lambda objs: self._breakout_paddle_zone(objs, 'left')
            ),
            Scenario(
                name="paddle_right_area",
                description="Paddle on right side",
                validator=lambda objs: self._breakout_paddle_zone(objs, 'right')
            ),
            Scenario(
                name="paddle_center_area",
                description="Paddle in center",
                validator=lambda objs: self._breakout_paddle_zone(objs, 'center')
            ),
            Scenario(
                name="many_bricks",
                description="Many bricks remaining",
                validator=lambda objs: self._breakout_brick_count(objs, 'many')
            ),
            Scenario(
                name="few_bricks",
                description="Fewer bricks remaining",
                validator=lambda objs: self._breakout_brick_count(objs, 'few')
            )
        ]

    def _get_spaceinvaders_scenarios(self) -> List[Scenario]:
        """Define 10 relaxed scenarios for Space Invaders."""
        return [
            Scenario(
                name="player_shooting",
                description="Player with active bullet",
                validator=lambda objs: self._si_player_shooting(objs)
            ),
            Scenario(
                name="enemy_shooting",
                description="Enemy with active bullet",
                validator=lambda objs: self._si_enemy_shooting(objs)
            ),
            Scenario(
                name="many_aliens",
                description="Many aliens remaining (30+)",
                validator=lambda objs: self._si_many_aliens(objs)
            ),
            Scenario(
                name="few_aliens",
                description="Few aliens remaining (5-25)",
                validator=lambda objs: self._si_few_aliens(objs)
            ),
            Scenario(
                name="player_bullet_near_aliens",
                description="Player bullet near alien formation",
                validator=lambda objs: self._si_player_bullet_near_aliens(objs)
            ),
            Scenario(
                name="player_bullet_far",
                description="Player bullet far from aliens",
                validator=lambda objs: self._si_player_bullet_far(objs)
            ),
            Scenario(
                name="enemy_bullet_near_player",
                description="Enemy bullet threatening player",
                validator=lambda objs: self._si_enemy_bullet_near_player(objs)
            ),
            Scenario(
                name="full_formation",
                description="Full alien formation early game",
                validator=lambda objs: self._si_full_formation_no_bullets(objs)
            ),
            Scenario(
                name="aliens_high_position",
                description="Aliens at high Y position (safe)",
                validator=lambda objs: self._si_aliens_high_position(objs)
            ),
            Scenario(
                name="player_near_shield",
                description="Player positioned near shield",
                validator=lambda objs: self._si_player_near_shield(objs)
            )
        ]

    # ========================================
    # Pong Scenario Validators (Simplified & Relaxed)
    # ========================================

    def _pong_ball_zone(self, objects: List[Dict], zone: str) -> bool:
        """Ball in specific horizontal zone (left/center/right)."""
        ball = next((o for o in objects if o['category'] == 'Ball'), None)
        if not ball:
            return False

        x = ball['position'][0]
        # OCAtari frame width: 160 pixels, scaled to 1280 in saved frames
        # Use original OCAtari coordinates for detection
        if zone == 'left':
            return x < 53  # Left third
        elif zone == 'center':
            return 53 <= x <= 107  # Center third
        elif zone == 'right':
            return x > 107  # Right third
        return False

    def _pong_ball_vertical(self, objects: List[Dict], zone: str) -> bool:
        """Ball in specific vertical zone (top/bottom)."""
        ball = next((o for o in objects if o['category'] == 'Ball'), None)
        if not ball:
            return False

        y = ball['position'][1]
        # OCAtari frame height: 210 pixels
        if zone == 'top':
            return y < 105  # Top half
        elif zone == 'bottom':
            return y >= 105  # Bottom half
        return False

    def _pong_paddle_zone(self, objects: List[Dict], zone: str) -> bool:
        """Player paddle in specific vertical zone (top/center/bottom)."""
        player = next((o for o in objects if 'player' in o['category'].lower()), None)
        if not player:
            return False

        y = player['position'][1]
        # OCAtari frame height: 210 pixels
        if zone == 'top':
            return y < 70  # Top third
        elif zone == 'center':
            return 70 <= y <= 140  # Center third
        elif zone == 'bottom':
            return y > 140  # Bottom third
        return False

    def _pong_active_game(self, objects: List[Dict], variant: int) -> bool:
        """Active game state with all elements present (variant for diversity)."""
        player = next((o for o in objects if 'player' in o['category'].lower()), None)
        enemy = next((o for o in objects if 'enemy' in o['category'].lower()), None)
        ball = next((o for o in objects if o['category'] == 'Ball'), None)

        if not (player and enemy and ball):
            return False

        # Variant 1: Ball in play, any position
        if variant == 1:
            return True

        # Variant 2: Ball moving (has velocity or is not at edge)
        if variant == 2:
            x = ball['position'][0]
            return 10 < x < 150  # Not at edge, so likely in motion

        return True

    def _pong_ball_near_paddle(self, objects: List[Dict]) -> bool:
        """Ball near player paddle (very wide)."""
        player = next((o for o in objects if 'player' in o['category'].lower()), None)
        ball = next((o for o in objects if o['category'] == 'Ball'), None)

        if not player or not ball:
            return False

        # Very wide: ball in right half of screen (near player paddle)
        return ball['position'][0] > 640  # Right half

    def _pong_ball_center(self, objects: List[Dict]) -> bool:
        """Ball in center area."""
        ball = next((o for o in objects if o['category'] == 'Ball'), None)

        if not ball:
            return False

        # Center third of screen
        return 400 < ball['position'][0] < 880

    def _pong_ball_near_enemy(self, objects: List[Dict]) -> bool:
        """Ball near enemy paddle (left side)."""
        ball = next((o for o in objects if o['category'] == 'Ball'), None)

        if not ball:
            return False

        # Left half of screen (near enemy)
        return ball['position'][0] < 640

    def _pong_ball_near_paddle_aligned(self, objects: List[Dict]) -> bool:
        """Ball near player paddle, paddle aligned with ball."""
        player = next((o for o in objects if o['category'] == 'Player'), None)
        ball = next((o for o in objects if o['category'] == 'Ball'), None)

        if not player or not ball:
            return False

        distance = abs(ball['position'][0] - player['position'][0])
        y_alignment = abs(ball['position'][1] - player['position'][1])

        return distance < 100 and y_alignment < 20

    def _pong_ball_near_paddle_misaligned(self, objects: List[Dict]) -> bool:
        """Ball near player paddle, paddle NOT aligned."""
        player = next((o for o in objects if o['category'] == 'Player'), None)
        ball = next((o for o in objects if o['category'] == 'Ball'), None)

        if not player or not ball:
            return False

        distance = abs(ball['position'][0] - player['position'][0])
        y_alignment = abs(ball['position'][1] - player['position'][1])

        return distance < 100 and y_alignment > 40

    def _pong_ball_far_approaching(self, objects: List[Dict]) -> bool:
        """Ball far from player."""
        player = next((o for o in objects if o['category'] == 'Player'), None)
        ball = next((o for o in objects if o['category'] == 'Ball'), None)

        if not player or not ball:
            return False

        # Relaxed: just check ball is on left side (far from right paddle)
        return ball['position'][0] < 100

    def _pong_ball_at_edge(self, objects: List[Dict], edge: str) -> bool:
        """Ball at top or bottom edge."""
        ball = next((o for o in objects if o['category'] == 'Ball'), None)
        if not ball:
            return False

        y = ball['position'][1]
        if edge == 'top':
            return y < 40  # More strict for top
        else:  # bottom
            return y > 170  # Relaxed for bottom (OCAtari frame is ~210 height)

    def _pong_paddle_at_position(self, objects: List[Dict], position: str) -> bool:
        """Paddle at top or bottom position."""
        player = next((o for o in objects if o['category'] == 'Player'), None)
        if not player:
            return False

        y = player['position'][1]
        if position == 'top':
            return y < 70  # Relaxed for top (OCAtari frame ~210 height)
        else:  # bottom
            return y > 140  # Relaxed for bottom

    def _pong_ball_center_paddle_edge(self, objects: List[Dict]) -> bool:
        """Ball in center, paddle at edge."""
        player = next((o for o in objects if o['category'] == 'Player'), None)
        ball = next((o for o in objects if o['category'] == 'Ball'), None)

        if not player or not ball:
            return False

        # Ball in vertical center (OCAtari: ~105 is center of 210 height)
        ball_center = 80 < ball['position'][1] < 130
        # Paddle at top or bottom edge
        paddle_edge = player['position'][1] < 70 or player['position'][1] > 140

        return ball_center and paddle_edge

    def _pong_ball_moving_away(self, objects: List[Dict]) -> bool:
        """Ball on enemy side (moving away from player)."""
        ball = next((o for o in objects if o['category'] == 'Ball'), None)
        if not ball:
            return False

        # Ball on left side of screen (enemy side)
        return ball['position'][0] < 400

    def _pong_critical_moment(self, objects: List[Dict]) -> bool:
        """Ball very close to paddle edge."""
        player = next((o for o in objects if o['category'] == 'Player'), None)
        ball = next((o for o in objects if o['category'] == 'Ball'), None)

        if not player or not ball:
            return False

        # Ball very close horizontally
        x_distance = abs(ball['position'][0] - player['position'][0])
        return x_distance < 50

    # ========================================
    # Breakout Scenario Validators (Simplified & Relaxed)
    # ========================================

    def _breakout_ball_zone(self, objects: List[Dict], zone: str) -> bool:
        """Ball in specific vertical zone (lower/upper)."""
        ball = next((o for o in objects if o['category'] == 'Ball'), None)
        if not ball:
            return False

        y = ball['position'][1]
        # OCAtari frame height: 210 pixels
        if zone == 'lower':
            return y > 105  # Lower half (near paddle)
        elif zone == 'upper':
            return y <= 105  # Upper half (near bricks)
        return False

    def _breakout_ball_horizontal(self, objects: List[Dict], zone: str) -> bool:
        """Ball in specific horizontal zone (left/center/right)."""
        ball = next((o for o in objects if o['category'] == 'Ball'), None)
        if not ball:
            return False

        x = ball['position'][0]
        # OCAtari frame width: 160 pixels
        if zone == 'left':
            return x < 53  # Left third
        elif zone == 'center':
            return 53 <= x <= 107  # Center third
        elif zone == 'right':
            return x > 107  # Right third
        return False

    def _breakout_paddle_zone(self, objects: List[Dict], zone: str) -> bool:
        """Paddle in specific horizontal zone (left/center/right)."""
        player = next((o for o in objects if 'player' in o['category'].lower()), None)
        ball = next((o for o in objects if o['category'] == 'Ball'), None)

        # IMPORTANT: Require ball to be present in ALL Breakout frames
        if not player or not ball:
            return False

        x = player['position'][0]
        # OCAtari frame width: 160 pixels
        if zone == 'left':
            return x < 53  # Left third
        elif zone == 'center':
            return 53 <= x <= 107  # Center third
        elif zone == 'right':
            return x > 107  # Right third
        return False

    def _breakout_brick_count(self, objects: List[Dict], count_type: str) -> bool:
        """Check brick count (many/few)."""
        bricks = [o for o in objects if 'brick' in o['category'].lower() or 'block' in o['category'].lower()]
        ball = next((o for o in objects if o['category'] == 'Ball'), None)

        # IMPORTANT: Require ball to be present in ALL Breakout frames
        if not ball:
            return False

        if count_type == 'many':
            # OCAtari detects BlockRows, not individual bricks
            # Many = 5+ BlockRows (most/all rows intact)
            return len(bricks) >= 5
        elif count_type == 'few':
            # Few = 1-4 BlockRows (some destroyed)
            return 1 <= len(bricks) <= 4
        return False

    def _breakout_ball_lower(self, objects: List[Dict]) -> bool:
        """Ball in lower area (near paddle)."""
        ball = next((o for o in objects if o['category'] == 'Ball'), None)
        player = next((o for o in objects if 'player' in o['category'].lower()), None)

        if not ball or not player:
            return False

        # Ball in lower half of screen
        return ball['position'][1] > 360

    def _breakout_ball_upper(self, objects: List[Dict]) -> bool:
        """Ball in upper area (near bricks)."""
        ball = next((o for o in objects if o['category'] == 'Ball'), None)

        if not ball:
            return False

        # Ball in upper half of screen
        return ball['position'][1] < 360

    def _breakout_game_active(self, objects: List[Dict]) -> bool:
        """Active game with paddle, ball, and some bricks."""
        ball = next((o for o in objects if o['category'] == 'Ball'), None)
        player = next((o for o in objects if 'player' in o['category'].lower()), None)
        bricks = [o for o in objects if 'brick' in o['category'].lower() or 'block' in o['category'].lower()]

        # Just need all elements present
        return ball is not None and player is not None and len(bricks) > 0

    def _breakout_ball_near_paddle_aligned(self, objects: List[Dict]) -> bool:
        """Ball near paddle, aligned."""
        player = next((o for o in objects if 'player' in o['category'].lower()), None)
        ball = next((o for o in objects if o['category'] == 'Ball'), None)

        if not player or not ball:
            return False

        y_distance = abs(ball['position'][1] - player['position'][1])
        x_alignment = abs(ball['position'][0] - player['position'][0])

        return y_distance < 150 and x_alignment < 50

    def _breakout_ball_near_paddle_edge(self, objects: List[Dict]) -> bool:
        """Ball near paddle at left/right edge."""
        player = next((o for o in objects if 'player' in o['category'].lower()), None)
        ball = next((o for o in objects if o['category'] == 'Ball'), None)

        if not player or not ball:
            return False

        y_distance = abs(ball['position'][1] - player['position'][1])
        ball_at_edge = ball['position'][0] < 200 or ball['position'][0] > 1080

        return y_distance < 150 and ball_at_edge

    def _breakout_ball_far_top(self, objects: List[Dict]) -> bool:
        """Ball far from paddle (near bricks at top)."""
        ball = next((o for o in objects if o['category'] == 'Ball'), None)
        if not ball:
            return False

        return ball['position'][1] < 300

    def _breakout_many_bricks(self, objects: List[Dict]) -> bool:
        """Many bricks remaining."""
        # OCAtari detects BlockRows, not individual bricks (typically 6-8 BlockRows)
        bricks = [o for o in objects if 'brick' in o['category'].lower() or 'block' in o['category'].lower()]
        return len(bricks) >= 5

    def _breakout_few_bricks(self, objects: List[Dict]) -> bool:
        """Few bricks remaining."""
        # OCAtari detects BlockRows (typically 6-8), so "few" means 1-5 BlockRows
        # Multiple fallback conditions to ensure we find SOMETHING
        bricks = [o for o in objects if 'brick' in o['category'].lower() or 'block' in o['category'].lower()]
        player = next((o for o in objects if 'player' in o['category'].lower()), None)
        ball = next((o for o in objects if o['category'] == 'Ball'), None)

        # Condition 1: 1-5 bricks with ball present (ideal)
        if 1 <= len(bricks) <= 5 and ball:
            return True

        # Condition 2: Even fewer bricks (1-3) without ball requirement
        if 1 <= len(bricks) <= 3:
            return True

        # Condition 3: 1-6 bricks with player at any position (very relaxed)
        if 1 <= len(bricks) <= 6 and player:
            return True

        return False

    def _breakout_paddle_edge(self, objects: List[Dict], side: str) -> bool:
        """Paddle at left or right edge."""
        player = next((o for o in objects if 'player' in o['category'].lower()), None)
        if not player:
            return False

        x = player['position'][0]
        if side == 'left':
            return x < 250
        else:  # right
            return x > 1030

    def _breakout_paddle_center(self, objects: List[Dict]) -> bool:
        """Paddle centered."""
        player = next((o for o in objects if 'player' in o['category'].lower()), None)
        ball = next((o for o in objects if o['category'] == 'Ball'), None)

        if not player:
            return False

        # Multiple fallback conditions
        x = player['position'][0]

        # Condition 1: Paddle in middle third with ball (ideal)
        if 400 < x < 880 and ball:
            return True

        # Condition 2: Paddle in wider center area (300-980) without ball
        if 300 < x < 980:
            return True

        # Condition 3: Paddle anywhere in middle half (320-960) - very relaxed
        if 320 < x < 960:
            return True

        return False

    def _breakout_ball_bouncing(self, objects: List[Dict]) -> bool:
        """Ball near brick layer."""
        ball = next((o for o in objects if o['category'] == 'Ball'), None)
        bricks = [o for o in objects if 'brick' in o['category'].lower() or 'block' in o['category'].lower()]

        if not ball:
            return False

        # Relaxed: Ball in upper half of screen (near bricks)
        return ball['position'][1] < 360

    def _breakout_paddle_with_ball(self, objects: List[Dict]) -> bool:
        """Paddle at any position with ball active (very easy to find)."""
        player = next((o for o in objects if 'player' in o['category'].lower()), None)
        ball = next((o for o in objects if o['category'] == 'Ball'), None)

        # Just need paddle and ball present
        return player is not None and ball is not None

    def _breakout_critical_miss(self, objects: List[Dict]) -> bool:
        """Ball about to miss paddle."""
        player = next((o for o in objects if 'player' in o['category'].lower()), None)
        ball = next((o for o in objects if o['category'] == 'Ball'), None)

        if not player or not ball:
            return False

        # Multiple fallback conditions
        ball_y = ball['position'][1]
        x_misalignment = abs(ball['position'][0] - player['position'][0])

        # Condition 1: Ball low and misaligned (ideal)
        if ball_y > 500 and x_misalignment > 80:
            return True

        # Condition 2: Ball very low with moderate misalignment
        if ball_y > 550 and x_misalignment > 60:
            return True

        # Condition 3: Ball in lower third and misaligned (very relaxed)
        if ball_y > 480 and x_misalignment > 100:
            return True

        # Condition 4: Ball anywhere in lower half with large misalignment
        if ball_y > 400 and x_misalignment > 150:
            return True

        return False

    # ========================================
    # Space Invaders Scenario Validators (Simplified & Relaxed)
    # ========================================

    def _si_player_shooting(self, objects: List[Dict]) -> bool:
        """Player with active bullet (any position)."""
        bullets = [o for o in objects if 'bullet' in o['category'].lower() or 'projectile' in o['category'].lower()]
        aliens = [o for o in objects if 'alien' in o['category'].lower()]
        player = next((o for o in objects if 'player' in o['category'].lower() or 'ship' in o['category'].lower()), None)

        if len(bullets) != 1 or not aliens or not player:
            return False

        bullet = bullets[0]
        lowest_alien_y = max(a['position'][1] for a in aliens)

        # Player bullet: below aliens, shooting up
        return bullet['position'][1] > lowest_alien_y

    def _si_enemy_shooting(self, objects: List[Dict]) -> bool:
        """Enemy with active bullet (any position)."""
        bullets = [o for o in objects if 'bullet' in o['category'].lower() or 'projectile' in o['category'].lower()]
        player = next((o for o in objects if 'player' in o['category'].lower() or 'ship' in o['category'].lower()), None)

        if not (1 <= len(bullets) <= 2) or not player:
            return False

        # Check if any bullet is enemy bullet (above player, shooting down)
        for bullet in bullets:
            if bullet['position'][1] < player['position'][1]:
                return True

        return False

    def _si_many_aliens(self, objects: List[Dict]) -> bool:
        """Many aliens remaining (30+)."""
        aliens = [o for o in objects if 'alien' in o['category'].lower()]
        return len(aliens) >= 30

    def _si_few_aliens(self, objects: List[Dict]) -> bool:
        """Few aliens remaining (5-25)."""
        aliens = [o for o in objects if 'alien' in o['category'].lower()]
        return 5 <= len(aliens) <= 25

    def _si_player_bullet_near_aliens(self, objects: List[Dict]) -> bool:
        """Exactly 1 player bullet, near bottom-row aliens."""
        bullets = [o for o in objects if 'bullet' in o['category'].lower() or 'projectile' in o['category'].lower()]
        aliens = [o for o in objects if 'alien' in o['category'].lower()]

        if len(bullets) != 1 or not aliens:
            return False

        # Find lowest alien (bottom row)
        lowest_alien_y = max(a['position'][1] for a in aliens)

        # Check if bullet is player bullet (below aliens, shooting up)
        bullet = bullets[0]
        if bullet['position'][1] <= lowest_alien_y:
            return False  # Not a player bullet

        # Check if player bullet is near aliens
        distance_to_aliens = abs(bullet['position'][1] - lowest_alien_y)
        return distance_to_aliens < 150

    def _si_player_bullet_far(self, objects: List[Dict]) -> bool:
        """Exactly 1 player bullet, far from aliens."""
        bullets = [o for o in objects if 'bullet' in o['category'].lower() or 'projectile' in o['category'].lower()]
        aliens = [o for o in objects if 'alien' in o['category'].lower()]
        player = next((o for o in objects if 'player' in o['category'].lower() or 'ship' in o['category'].lower()), None)

        if len(bullets) != 1 or not aliens or not player:
            return False

        lowest_alien_y = max(a['position'][1] for a in aliens)
        bullet = bullets[0]

        # Check if bullet is player bullet (below aliens, shooting up)
        if bullet['position'][1] <= lowest_alien_y:
            return False  # Not a player bullet

        # Check if player bullet is far from aliens and between player and aliens
        bullet_between = player['position'][1] > bullet['position'][1] > lowest_alien_y
        distance_to_aliens = bullet['position'][1] - lowest_alien_y

        return bullet_between and distance_to_aliens > 100

    def _si_player_bullet_about_to_hit(self, objects: List[Dict]) -> bool:
        """Exactly 1 player bullet, about to hit alien."""
        bullets = [o for o in objects if 'bullet' in o['category'].lower() or 'projectile' in o['category'].lower()]
        aliens = [o for o in objects if 'alien' in o['category'].lower()]

        if len(bullets) != 1 or not aliens:
            return False

        # Find lowest alien to identify player bullet
        lowest_alien_y = max(a['position'][1] for a in aliens)
        bullet = bullets[0]

        # Check if bullet is player bullet (below aliens, shooting up)
        if bullet['position'][1] <= lowest_alien_y:
            return False  # Not a player bullet

        # Check if player bullet is very close to any alien
        for alien in aliens:
            x_distance = abs(bullet['position'][0] - alien['position'][0])
            y_distance = abs(bullet['position'][1] - alien['position'][1])

            if x_distance < 30 and y_distance < 50:
                return True

        return False

    def _si_enemy_bullet_near_player(self, objects: List[Dict]) -> bool:
        """1-2 enemy bullets, near player ship."""
        bullets = [o for o in objects if 'bullet' in o['category'].lower() or 'projectile' in o['category'].lower()]
        player = next((o for o in objects if 'player' in o['category'].lower() or 'ship' in o['category'].lower()), None)

        if not (1 <= len(bullets) <= 2) or not player:
            return False

        # Filter enemy bullets (above player, shooting down)
        enemy_bullets = [b for b in bullets if b['position'][1] < player['position'][1]]

        if len(enemy_bullets) == 0:
            return False

        # Check if at least one enemy bullet is near player
        for bullet in enemy_bullets:
            distance_to_player = abs(bullet['position'][1] - player['position'][1])
            if distance_to_player < 150:
                return True

        return False

    def _si_full_formation_no_bullets(self, objects: List[Dict]) -> bool:
        """Full alien formation, no bullets."""
        bullets = [o for o in objects if 'bullet' in o['category'].lower() or 'projectile' in o['category'].lower()]
        aliens = [o for o in objects if 'alien' in o['category'].lower()]

        # Relaxed: Allow 0-1 bullets, require >= 40 aliens (almost full formation)
        return len(bullets) <= 1 and len(aliens) >= 40

    def _si_aliens_mostly_destroyed(self, objects: List[Dict]) -> bool:
        """Most aliens destroyed (5-15 remaining), 1 bullet."""
        bullets = [o for o in objects if 'bullet' in o['category'].lower() or 'projectile' in o['category'].lower()]
        aliens = [o for o in objects if 'alien' in o['category'].lower()]

        # Relaxed: Allow 0-2 bullets, require 5-20 aliens
        return len(bullets) <= 2 and 5 <= len(aliens) <= 20

    def _si_aliens_low_position(self, objects: List[Dict]) -> bool:
        """Aliens at low Y position (threatening), 1 bullet."""
        bullets = [o for o in objects if 'bullet' in o['category'].lower() or 'projectile' in o['category'].lower()]
        aliens = [o for o in objects if 'alien' in o['category'].lower()]

        if not aliens:
            return False

        # Relaxed: Allow any number of bullets, just check alien position
        lowest_alien_y = max(a['position'][1] for a in aliens)
        return lowest_alien_y > 450 and len(bullets) <= 2

    def _si_aliens_high_position(self, objects: List[Dict]) -> bool:
        """Aliens at high Y position (safe), 1 bullet."""
        bullets = [o for o in objects if 'bullet' in o['category'].lower() or 'projectile' in o['category'].lower()]
        aliens = [o for o in objects if 'alien' in o['category'].lower()]

        if len(bullets) != 1 or not aliens:
            return False

        highest_alien_y = min(a['position'][1] for a in aliens)
        return highest_alien_y < 250

    def _si_player_near_shield(self, objects: List[Dict]) -> bool:
        """Player near shield, 1 bullet present."""
        bullets = [o for o in objects if 'bullet' in o['category'].lower() or 'projectile' in o['category'].lower()]
        player = next((o for o in objects if 'player' in o['category'].lower() or 'ship' in o['category'].lower()), None)
        shields = [o for o in objects if 'shield' in o['category'].lower() or 'barrier' in o['category'].lower()]

        if len(bullets) != 1 or not player or not shields:
            return False

        # Check if player is close to any shield
        for shield in shields:
            distance = abs(player['position'][0] - shield['position'][0])
            if distance < 100:
                return True

        return False

    # ========================================
    # Dataset Generation
    # ========================================

    def generate_scenario_based_dataset(self, max_attempts_per_scenario: int = 5000):
        """
        Generate dataset with distinct scenarios per game.
        Total: 30 frames (Pong: 10, Breakout: 10, SpaceInvaders: 10)

        Args:
            max_attempts_per_scenario: Maximum frames to check for each scenario
        """
        print("=" * 80)
        print("SCENARIO-BASED DATASET GENERATION (SIMPLIFIED)")
        print("=" * 80)
        print(f"Generating scenarios per game: Pong (10), Breakout (10), SpaceInvaders (10)")
        print(f"Total frames: 30 = 10 (Pong) + 10 (Breakout) + 10 (SpaceInvaders)")
        print(f"Using wide/relaxed validators for high success rate")
        print()

        dataset_metadata = {
            'total_frames': 0,
            'games': {},
            'generation_method': 'scenario_based_simplified',
            'scenarios_per_game': 'Pong:10, Breakout:10, SpaceInvaders:10'
        }

        for game in GAMES:
            print(f"\n{'=' * 80}")
            print(f"GAME: {game}")
            print(f"{'=' * 80}")

            scenarios = self.scenarios[game]
            collected_frames = []

            # Generate frames for each scenario
            for scenario in scenarios:
                print(f"\nScenario: {scenario.name}")
                print(f"Description: {scenario.description}")

                frame_data = self._find_frame_for_scenario(
                    game, scenario, max_attempts_per_scenario
                )

                if frame_data:
                    collected_frames.append(frame_data)
                    print(f"  ✓ Found matching frame after {frame_data['attempts']} attempts")
                else:
                    print(f"  ✗ No matching frame found after {max_attempts_per_scenario} attempts")

            # Save collected frames
            expected = len(scenarios)
            print(f"\n{game}: Collected {len(collected_frames)}/{expected} scenario frames")

            for i, frame_data in enumerate(collected_frames):
                self._save_frame(game, i, frame_data)

            dataset_metadata['games'][game] = {
                'frames_collected': len(collected_frames),
                'scenarios': [f['scenario_name'] for f in collected_frames]
            }
            dataset_metadata['total_frames'] += len(collected_frames)

        # Save old-style metadata (for backwards compatibility)
        metadata_path_old = self.output_dir / "dataset_metadata.json"
        with open(metadata_path_old, 'w') as f:
            json.dump(dataset_metadata, f, indent=2)

        # Create proper dataset_index.json for loader
        self._create_dataset_index(dataset_metadata)

        print("\n" + "=" * 80)
        print(f"Dataset generation complete!")
        print(f"Total frames generated: {dataset_metadata['total_frames']}")
        print(f"Metadata saved to: {self.metadata_dir / 'dataset_index.json'}")
        print("=" * 80)

    def _create_dataset_index(self, dataset_metadata: Dict) -> None:
        """Create dataset_index.json for loader compatibility."""
        from datetime import datetime

        # Build frames list from ground truth files
        frames_list = []
        for gt_file in sorted(self.ground_truth_dir.glob("*.json")):
            with open(gt_file, 'r') as f:
                gt_data = json.load(f)

            frame_entry = {
                'frame_id': gt_data['frame_id'],
                'game': gt_data['game'],
                'scenario': gt_data.get('scenario', {}),
                'object_count': gt_data['ocatari_data']['object_count']
            }
            frames_list.append(frame_entry)

        # Calculate distribution
        game_distribution = {}
        for game, game_data in dataset_metadata['games'].items():
            game_distribution[game] = {
                'total': game_data['frames_collected'],
                'scenarios': game_data['scenarios']
            }

        # Create index structure
        dataset_index = {
            'metadata': {
                'version': '2.0.0',
                'created': datetime.now().isoformat(),
                'seed': RANDOM_SEED,
                'games': list(GAMES),
                'total_frames': dataset_metadata['total_frames'],
                'frames_per_game': game_distribution,
                'generation_method': 'scenario_based_simplified',
                'scenarios_per_game': 'Pong:10, Breakout:10, SpaceInvaders:10'
            },
            'frames': frames_list
        }

        # Save dataset index
        index_path = self.metadata_dir / "dataset_index.json"
        with open(index_path, 'w') as f:
            json.dump(dataset_index, f, indent=2)

        print(f"✅ Created dataset index: {index_path}")

    def _find_frame_for_scenario(
        self,
        game: str,
        scenario: Scenario,
        max_attempts: int
    ) -> Optional[Dict]:
        """Find a frame that matches the scenario."""
        # Initialize OCAtari environment
        import random
        import time

        ocatari = OCAtariGroundTruth(game)

        # Use time-based seed for true randomness
        base_seed = int(time.time() * 1000) % (2**31)

        # Determine reset interval based on scenario needs
        # Some scenarios need game progression, others need early game states
        needs_progression = self._scenario_needs_progression(scenario.name)

        # For Breakout progression, use MUCH longer intervals since destroying bricks takes time
        if 'breakout' in game.lower() and needs_progression:
            reset_interval = 5000  # Let Breakout run much longer
        elif needs_progression:
            reset_interval = 2000
        else:
            reset_interval = 300

        candidates = []  # Store multiple candidates to pick the most different one
        last_log_attempt = 0

        for attempt in range(max_attempts):
            # Reset periodically with NEW RANDOM SEED each time
            if attempt % reset_interval == 0:
                # Use different seed for each reset to get different gameplay
                new_seed = (base_seed + attempt + np.random.randint(0, 1000)) % (2**31)

                # Use Gymnasium's reset with seed parameter
                ocatari.reset(seed=new_seed)

                # Add random warm-up steps to vary starting state
                warmup_steps = np.random.randint(10, 50)
                for _ in range(warmup_steps):
                    random_action = ocatari.env.action_space.sample()
                    ocatari.step(random_action)

                # For progression scenarios, fast-forward the game with aggressive actions
                if needs_progression:
                    self._fast_forward_game(ocatari, game, scenario.name)

            # Progress logging every 5000 attempts
            if attempt - last_log_attempt >= 5000:
                print(f"    ... attempt {attempt}/{max_attempts} ...")
                last_log_attempt = attempt

            # Use smarter actions based on game and scenario
            action = self._get_strategic_action(ocatari, game, scenario.name)

            # Take the step first
            obs, reward, terminated, truncated, info = ocatari.step(action)

            # Check for game over and reset if needed
            if terminated or truncated:
                ocatari.reset()
                if needs_progression:
                    self._fast_forward_game(ocatari, game, scenario.name)
                continue

            # Get frame and objects
            frame, objects_info = ocatari.get_frame_and_objects()

            # Skip if no objects
            if not objects_info:
                continue

            # Convert ObjectInfo to dict format for validators
            objects = []
            for obj_info in objects_info:
                objects.append({
                    'category': obj_info.category,
                    'position': obj_info.position,
                    'velocity': obj_info.velocity,
                    'size': obj_info.size
                })

            # Check if frame matches scenario
            if scenario.validator(objects):
                return {
                    'frame': frame,
                    'objects': objects,
                    'scenario_name': scenario.name,
                    'scenario_description': scenario.description,
                    'attempts': attempt + 1
                }

        return None

    def _is_duplicate_frame(self, new_frame, existing_frames: List) -> bool:
        """Check if frame is too similar to existing frames."""
        if not existing_frames:
            return False

        import cv2

        # Simple duplicate detection: check if frame is very similar to any existing frame
        for existing_frame_data in existing_frames:
            if 'frame' not in existing_frame_data:
                continue

            existing_frame = existing_frame_data['frame']

            # Resize both to same size for comparison
            if new_frame.shape != existing_frame.shape:
                continue

            # Calculate absolute difference
            diff = cv2.absdiff(new_frame, existing_frame)
            mean_diff = diff.mean()

            # Relaxed threshold: only reject if frames are EXTREMELY similar
            # Mean diff < 2.0 means nearly identical (same frame or 1 pixel moved)
            if mean_diff < 2.0:  # Only reject extremely similar frames
                return True

        return False

    def _scenario_needs_progression(self, scenario_name: str) -> bool:
        """Check if scenario requires game progression."""
        progression_scenarios = [
            'few_bricks_remaining',
            'aliens_mostly_destroyed',
            'critical_miss',
            'paddle_center',
            'player_bullet_far'
        ]
        return any(s in scenario_name for s in progression_scenarios)

    def _fast_forward_game(self, ocatari, game: str, scenario_name: str):
        """Fast-forward game to reach desired state."""
        if 'breakout' in game.lower():
            # For Breakout, actively play to destroy bricks
            if 'few_bricks' in scenario_name:
                # VERY aggressively destroy bricks for 1000-2000 steps
                for _ in range(np.random.randint(1000, 2000)):
                    # Fire action (1) or track ball with paddle
                    action = 1 if np.random.random() > 0.4 else np.random.randint(2, 4)
                    ocatari.step(action)
            elif 'critical_miss' in scenario_name:
                # Let game progress to get challenging situations
                for _ in range(np.random.randint(200, 500)):
                    action = ocatari.env.action_space.sample()
                    ocatari.step(action)
            elif 'paddle_center' in scenario_name:
                # Let game progress naturally
                for _ in range(np.random.randint(100, 300)):
                    action = ocatari.env.action_space.sample()
                    ocatari.step(action)

        elif 'space' in game.lower() or 'invader' in game.lower():
            # For Space Invaders
            if 'aliens_mostly_destroyed' in scenario_name:
                # Aggressively shoot aliens for 1000-1500 steps
                for _ in range(np.random.randint(1000, 1500)):
                    action = 1 if np.random.random() > 0.3 else np.random.randint(2, 4)  # Fire frequently
                    ocatari.step(action)
            elif 'player_bullet_far' in scenario_name:
                # Progress game a bit then try to get bullet timing right
                for _ in range(np.random.randint(50, 150)):
                    action = np.random.choice([0, 2, 3])  # Move but don't shoot initially
                    ocatari.step(action)

    def _get_strategic_action(self, ocatari, game: str, scenario_name: str) -> int:
        """Get strategic action based on game and scenario."""
        # For most scenarios, use random actions
        # For specific scenarios, use targeted actions
        if 'breakout' in game.lower():
            if 'paddle_center' in scenario_name:
                # Occasionally move to center
                if np.random.random() > 0.7:
                    return 0  # NOOP to stay in place
                return np.random.choice([2, 3])  # Left or right
            elif 'few_bricks' in scenario_name or 'critical_miss' in scenario_name:
                # Active play
                return 1 if np.random.random() > 0.5 else np.random.randint(2, 4)

        elif 'space' in game.lower() or 'invader' in game.lower():
            if 'full_formation' in scenario_name:
                # Early game, don't shoot much
                return 0 if np.random.random() > 0.2 else np.random.randint(2, 4)
            elif 'aliens_mostly_destroyed' in scenario_name:
                # Shoot frequently
                return 1 if np.random.random() > 0.3 else np.random.randint(2, 4)
            elif 'player_bullet_far' in scenario_name:
                # Shoot occasionally
                return 1 if np.random.random() > 0.7 else np.random.randint(0, 4)

        # Default: random action
        return ocatari.env.action_space.sample()

    def _save_frame(self, game: str, index: int, frame_data: Dict):
        """Save frame and ground truth."""
        from ..utils.helpers import calculate_spatial_relationships
        from ..dataset.creator import BenchmarkDatasetCreator

        # Create frame filename
        scenario_name = frame_data['scenario_name']
        frame_filename = f"{game.lower()}_{index:04d}_{scenario_name}.png"
        frame_path = self.frames_dir / frame_filename

        # Resize frame to target resolution (1280x720)
        # OCAtari returns frames in (210, 160, 3), we need to upscale
        frame = frame_data['frame']

        # Validate frame before resizing
        if frame is None:
            raise ValueError(f"Frame is None for scenario {scenario_name}")
        if frame.size == 0 or frame.shape[0] == 0 or frame.shape[1] == 0:
            raise ValueError(f"Frame has invalid shape {frame.shape} for scenario {scenario_name}")

        target_resolution = (1280, 720)
        frame_resized = cv2.resize(frame, target_resolution, interpolation=cv2.INTER_NEAREST)

        # Save frame (convert RGB to BGR for OpenCV)
        frame_bgr = cv2.cvtColor(frame_resized, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(frame_path), frame_bgr)

        # Scale factor for coordinates
        scale_x = target_resolution[0] / frame.shape[1]  # 1280 / 160
        scale_y = target_resolution[1] / frame.shape[0]  # 720 / 210

        # Prepare objects for ground truth with scaled coordinates
        serialized_objects = []
        for obj in frame_data['objects']:
            # Scale position to match resized frame
            scaled_position = [
                int(obj['position'][0] * scale_x),
                int(obj['position'][1] * scale_y)
            ]

            # Determine tier (core objects for gameplay)
            tier = self._get_object_tier(obj['category'], game)

            serialized_objects.append({
                'category': obj['category'],
                'position': scaled_position,
                'tier': tier
            })

        # Calculate spatial relationships
        spatial_relationships = calculate_spatial_relationships(serialized_objects)

        # Create ground truth
        ground_truth = {
            'frame_id': f"{game.lower()}_{index:04d}_{scenario_name}",
            'game': game,
            'scenario': {
                'name': scenario_name,
                'description': frame_data['scenario_description']
            },
            'ocatari_data': {
                'objects': serialized_objects,
                'object_count': len(serialized_objects),
                'spatial_relationships': spatial_relationships
            },
            'reference_answers': self._generate_reference_answers(
                serialized_objects, game, scenario_name
            )
        }

        # Save ground truth
        gt_filename = f"{game.lower()}_{index:04d}_{scenario_name}.json"
        gt_path = self.ground_truth_dir / gt_filename

        with open(gt_path, 'w') as f:
            json.dump(ground_truth, f, indent=2)

    def _get_object_tier(self, category: str, game: str) -> str:
        """Determine if object is core or secondary for gameplay."""
        # Core objects are critical for gameplay
        core_objects = {
            'Pong': ['Player', 'Ball', 'Enemy'],
            'Breakout': ['Player', 'Ball', 'Brick'],
            'SpaceInvaders': ['Player', 'PlayerShip', 'Ship', 'Alien', 'Bullet', 'Projectile']
        }

        game_core = core_objects.get(game, [])

        # Check if category matches any core object
        for core in game_core:
            if core.lower() in category.lower():
                return 'core'

        return 'secondary'

    def _generate_reference_answers(
        self,
        objects: List[Dict],
        game: str,
        scenario_name: str
    ) -> Dict[str, Dict[str, str]]:
        """Generate reference answers for each task type (matching old creator quality)."""
        # Filter to core objects for reference generation
        core_objects = [obj for obj in objects if obj.get('tier') == 'core']
        if not core_objects:
            core_objects = objects  # Fallback to all objects

        return {
            'visual': {
                'qualitative': self._generate_visual_reference_qualitative(core_objects, game),
                'quantitative': self._generate_visual_reference_quantitative(core_objects, game)
            },
            'spatial': {
                'qualitative': self._generate_spatial_reference_qualitative(core_objects, game),
                'quantitative': self._generate_spatial_reference_quantitative(core_objects, game)
            },
            'strategy': {
                'qualitative': self._generate_strategy_reference_qualitative(objects, game),
                'quantitative': self._generate_strategy_reference_quantitative(objects, game)
            },
            'identification': {
                'qualitative': game,
                'quantitative': game
            }
        }

    def _generate_visual_reference_qualitative(self, objects: List[Dict], game: str) -> str:
        """Generate qualitative visual reference (Vision-Only friendly)."""
        if not objects:
            return f"This is a {game} game frame with no clearly visible key game objects."

        descriptions = []
        for obj in objects:
            category = obj['category']
            x, y = obj['position']

            # Horizontal position (scaled coordinates: 0-1280)
            if x < 400:
                h_pos = "left side"
            elif x < 880:
                h_pos = "center area"
            else:
                h_pos = "right side"

            # Vertical position (scaled coordinates: 0-720)
            if y < 240:
                v_pos = "upper"
            elif y < 480:
                v_pos = "middle"
            else:
                v_pos = "lower"

            descriptions.append(f"{category} in the {v_pos} {h_pos}")

        return f"Key elements: {', '.join(descriptions)}."

    def _generate_visual_reference_quantitative(self, objects: List[Dict], game: str) -> str:
        """Generate quantitative visual reference (Vision+Symbol format)."""
        if not objects:
            return f"This is a {game} game frame with no clearly visible key game objects."

        obj_details = []
        for obj in objects:
            category = obj['category']
            x, y = obj['position']
            # Size is not always available in the object dict from scenario creator
            size_info = ""
            if 'size' in obj and obj['size'] and len(obj['size']) == 2:
                w, h = obj['size']
                size_info = f", size {w}x{h}"
            else:
                size_info = ""

            obj_details.append(f"{category} at ({x}, {y}){size_info}")

        description = f"Detected {len(objects)} key objects: "
        description += "; ".join(obj_details) + "."
        return description

    def _generate_spatial_reference_qualitative(self, objects: List[Dict], game: str) -> str:
        """Generate qualitative spatial reference (Vision-Only friendly)."""
        if not objects or len(objects) < 2:
            return "Not enough key objects for spatial analysis."

        spatial_desc = []

        # Describe relationships between object pairs
        for i, obj1 in enumerate(objects[:3]):  # Max 3 objects
            for obj2 in objects[i+1:i+2]:  # Only 1 pair per object
                cat1, cat2 = obj1['category'], obj2['category']
                x1, y1 = obj1['position']
                x2, y2 = obj2['position']

                # Horizontal relationship
                if abs(x1 - x2) > 20:
                    h_rel = "left of" if x1 < x2 else "right of"
                    spatial_desc.append(f"The {cat1} is {h_rel} the {cat2}")

                # Vertical relationship
                if abs(y1 - y2) > 20:
                    v_rel = "above" if y1 < y2 else "below"
                    spatial_desc.append(f"The {cat1} is {v_rel} the {cat2}")

                # Proximity
                distance = ((x1 - x2)**2 + (y1 - y2)**2)**0.5
                if distance < 100:
                    proximity = "near"
                else:
                    proximity = "far from"
                spatial_desc.append(f"The {cat1} is {proximity} the {cat2}")

        return "Spatial layout: " + "; ".join(spatial_desc) + "."

    def _generate_spatial_reference_quantitative(self, objects: List[Dict], game: str) -> str:
        """Generate quantitative spatial reference (Vision+Symbol format)."""
        if not objects or len(objects) < 2:
            return "Not enough key objects for spatial analysis."

        spatial_desc = []

        for i, obj1 in enumerate(objects[:3]):
            for obj2 in objects[i+1:i+2]:
                cat1, cat2 = obj1['category'], obj2['category']
                x1, y1 = obj1['position']
                x2, y2 = obj2['position']

                # Horizontal relationship
                h_rel = "LEFT" if x1 < x2 else "RIGHT"
                spatial_desc.append(f"{cat1} at x={x1} is {h_rel} of {cat2} at x={x2}")

                # Vertical relationship
                v_rel = "ABOVE" if y1 < y2 else "BELOW"
                spatial_desc.append(f"{cat1} at y={y1} is {v_rel} {cat2} at y={y2}")

                # Distance
                distance = ((x1 - x2)**2 + (y1 - y2)**2)**0.5
                spatial_desc.append(f"Distance: {distance:.1f} pixels")

        return "Spatial relationships: " + "; ".join(spatial_desc) + "."

    def _generate_strategy_reference_qualitative(self, objects: List[Dict], game: str) -> str:
        """Generate qualitative strategy reference (Vision-Only friendly)."""
        if not objects:
            return "Unable to determine optimal strategy without visible game elements."

        # Game-specific strategy
        if game == 'Pong':
            return self._generate_pong_strategy_qualitative(objects)
        elif game == 'Breakout':
            return self._generate_breakout_strategy_qualitative(objects)
        elif game == 'SpaceInvaders':
            return self._generate_spaceinvaders_strategy_qualitative(objects)

        return f"Analyze {game} game state and determine optimal action."

    def _generate_strategy_reference_quantitative(self, objects: List[Dict], game: str) -> str:
        """Generate quantitative strategy reference (Vision+Symbol format)."""
        if not objects:
            return "Unable to determine optimal strategy without visible game elements."

        # Game-specific strategy with coordinates
        if game == 'Pong':
            return self._generate_pong_strategy_quantitative(objects)
        elif game == 'Breakout':
            return self._generate_breakout_strategy_quantitative(objects)
        elif game == 'SpaceInvaders':
            return self._generate_spaceinvaders_strategy_quantitative(objects)

        return f"Analyze {game} game state and determine optimal action based on object positions."

    # === Pong Strategy ===
    def _generate_pong_strategy_qualitative(self, objects: List[Dict]) -> str:
        """Generate Pong strategy without exact coordinates."""
        player_paddle = next((o for o in objects if 'player' in o['category'].lower()), None)
        ball = next((o for o in objects if 'ball' in o['category'].lower()), None)

        if not (player_paddle and ball):
            return "HOLD position - Cannot determine game state."

        paddle_y = player_paddle['position'][1]
        ball_y = ball['position'][1]

        # Determine relative position
        diff = ball_y - paddle_y
        if abs(diff) < 30:
            return "HOLD - Paddle well-aligned with ball."
        elif diff < -30:
            return "MOVE UP - Ball is above paddle position."
        else:
            return "MOVE DOWN - Ball is below paddle position."

    def _generate_pong_strategy_quantitative(self, objects: List[Dict]) -> str:
        """Generate Pong strategy with coordinates and measurements."""
        player_paddle = next((o for o in objects if 'player' in o['category'].lower()), None)
        ball = next((o for o in objects if 'ball' in o['category'].lower()), None)

        if not (player_paddle and ball):
            return "HOLD position - Unable to locate paddle or ball."

        paddle_y = player_paddle['position'][1]
        ball_y = ball['position'][1]

        # Action recommendation
        diff = ball_y - paddle_y

        if abs(diff) < 30:
            action = "HOLD"
            reasoning = f"Paddle at y={paddle_y} well-aligned with ball at y={ball_y}."
        elif diff < -30:
            action = "MOVE UP"
            reasoning = f"Ball at y={ball_y} is above paddle at y={paddle_y}. Gap of {abs(diff):.0f} pixels."
        else:
            action = "MOVE DOWN"
            reasoning = f"Ball at y={ball_y} is below paddle at y={paddle_y}. Gap of {abs(diff):.0f} pixels."

        return f"{action} - {reasoning}"

    # === Breakout Strategy ===
    def _generate_breakout_strategy_qualitative(self, objects: List[Dict]) -> str:
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
        if abs(diff) < 30:
            return "HOLD - Paddle aligned with ball."
        elif diff < -30:
            return "MOVE LEFT - Ball is left of paddle."
        else:
            return "MOVE RIGHT - Ball is right of paddle."

    def _generate_breakout_strategy_quantitative(self, objects: List[Dict]) -> str:
        """Generate Breakout strategy with coordinates and measurements."""
        player = next((o for o in objects if 'player' in o['category'].lower()), None)
        ball = next((o for o in objects if 'ball' in o['category'].lower()), None)

        if not ball:
            return "HOLD position - No ball visible."

        if not player:
            return "Move paddle to intercept ball."

        paddle_x = player['position'][0]
        ball_x = ball['position'][0]

        # Action recommendation
        diff = ball_x - paddle_x

        if abs(diff) < 30:
            action = "HOLD"
            reasoning = f"Paddle at x={paddle_x} aligned with ball at x={ball_x}."
        elif diff < -30:
            action = "MOVE LEFT"
            reasoning = f"Ball at x={ball_x} is left of paddle at x={paddle_x}. Gap of {abs(diff):.0f} pixels."
        else:
            action = "MOVE RIGHT"
            reasoning = f"Ball at x={ball_x} is right of paddle at x={paddle_x}. Gap of {abs(diff):.0f} pixels."

        return f"{action} - {reasoning}"

    # === Space Invaders Strategy ===
    def _generate_spaceinvaders_strategy_qualitative(self, objects: List[Dict]) -> str:
        """Generate Space Invaders strategy without exact coordinates."""
        player = next((o for o in objects if 'player' in o['category'].lower() or 'ship' in o['category'].lower()), None)
        aliens = [o for o in objects if 'alien' in o['category'].lower()]
        bullets = [o for o in objects if 'bullet' in o['category'].lower() or 'projectile' in o['category'].lower()]

        if not player:
            return "HOLD position - Cannot determine player location."

        # Check for enemy bullets near player
        for bullet in bullets:
            if bullet['position'][1] > player['position'][1] - 100:  # Bullet close to player
                # Determine dodge direction
                if bullet['position'][0] < player['position'][0]:
                    return "MOVE RIGHT - Dodge enemy bullet coming from left."
                else:
                    return "MOVE LEFT - Dodge enemy bullet coming from right."

        # No immediate threat, target aliens
        if aliens:
            # Find nearest alien
            nearest = min(aliens, key=lambda a: abs(a['position'][0] - player['position'][0]))
            if abs(nearest['position'][0] - player['position'][0]) < 50:
                return "FIRE - Alien directly above, shoot now."
            elif nearest['position'][0] < player['position'][0]:
                return "MOVE LEFT and FIRE - Position under alien."
            else:
                return "MOVE RIGHT and FIRE - Position under alien."

        return "HOLD and FIRE - Eliminate threats."

    def _generate_spaceinvaders_strategy_quantitative(self, objects: List[Dict]) -> str:
        """Generate Space Invaders strategy with coordinates."""
        player = next((o for o in objects if 'player' in o['category'].lower() or 'ship' in o['category'].lower()), None)
        aliens = [o for o in objects if 'alien' in o['category'].lower()]
        bullets = [o for o in objects if 'bullet' in o['category'].lower() or 'projectile' in o['category'].lower()]

        if not player:
            return "HOLD position - Cannot determine player location."

        player_x = player['position'][0]
        player_y = player['position'][1]

        # Check for enemy bullets near player
        for bullet in bullets:
            bullet_x, bullet_y = bullet['position']
            if bullet_y > player_y - 100:
                distance = abs(bullet_x - player_x)
                if bullet_x < player_x:
                    return f"MOVE RIGHT - Enemy bullet at x={bullet_x} approaching from left. Distance: {distance:.0f} pixels."
                else:
                    return f"MOVE LEFT - Enemy bullet at x={bullet_x} approaching from right. Distance: {distance:.0f} pixels."

        # No immediate threat, target aliens
        if aliens:
            nearest = min(aliens, key=lambda a: abs(a['position'][0] - player_x))
            alien_x = nearest['position'][0]
            x_diff = alien_x - player_x

            if abs(x_diff) < 50:
                return f"FIRE - Alien at x={alien_x} directly above player at x={player_x}."
            elif x_diff < 0:
                return f"MOVE LEFT and FIRE - Alien at x={alien_x}, player at x={player_x}. Gap: {abs(x_diff):.0f} pixels."
            else:
                return f"MOVE RIGHT and FIRE - Alien at x={alien_x}, player at x={player_x}. Gap: {abs(x_diff):.0f} pixels."

        return f"HOLD and FIRE - Player at x={player_x}, eliminate threats."


def main():
    """Main entry point for scenario-based dataset generation."""
    creator = ScenarioBasedDatasetCreator(output_dir="./benchmark_v2.0_scenarios")
    creator.generate_scenario_based_dataset(max_attempts_per_scenario=5000)


if __name__ == '__main__':
    main()
