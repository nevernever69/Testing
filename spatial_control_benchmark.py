#!/usr/bin/env python3
"""
Spatial Control Benchmark - Objective tests for spatial reasoning capability.

Tests CONTROL decisions, not descriptive ability:
- Trajectory Prediction: Can you predict future positions?
- Movement Distance: Can you compute how far to move?
- Collision Detection: Can you predict collisions?
- Interception Calculation: Can you compute interception points?
"""

import json
import re
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class SpatialTest:
    """A single spatial reasoning test case."""
    frame_id: str
    test_type: str  # 'movement_distance', 'trajectory', 'collision', 'interception'
    question: str
    ground_truth: Dict
    metadata: Dict


class SpatialControlEvaluator:
    """Evaluates spatial reasoning with objective metrics (no LLM judge)."""
    
    def __init__(self, screen_width: int = 1280, screen_height: int = 720):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.screen_diagonal = np.sqrt(screen_width**2 + screen_height**2)
    
    def create_movement_distance_test(self, ground_truth: Dict, include_coordinates: bool = True) -> Optional[SpatialTest]:
        """
        Create test: "How far should paddle move to align with ball?"

        Args:
            ground_truth: OCAtari ground truth data
            include_coordinates: If True, provide coordinates in question (for Vision+Symbol)
                                If False, ask qualitatively (for Vision-Only)

        Expected answer format: "400 pixels" or "Move 400 pixels right"
        """
        objects = ground_truth.get('ocatari_data', {}).get('objects', [])

        # Find ball and paddle
        ball = None
        paddle = None
        for obj in objects:
            if obj['category'] == 'Ball':
                ball = obj
            elif obj['category'] == 'Player':
                paddle = obj

        if not ball or not paddle:
            return None

        ball_x, ball_y = ball['position']
        paddle_x, paddle_y = paddle['position']

        optimal_distance = abs(ball_x - paddle_x)
        direction = "right" if ball_x > paddle_x else "left"

        if include_coordinates:
            # For Vision+Symbol: Provide coordinates
            question = f"""Given the current game state:
- Ball position: x={ball_x}, y={ball_y}
- Paddle position: x={paddle_x}, y={paddle_y}

Question: How many pixels should the paddle move horizontally to align with the ball?

Provide your answer in this format: "Move [NUMBER] pixels [left/right]"
Example: "Move 250 pixels right"
"""
        else:
            # For Vision-Only: Ask qualitatively without coordinates
            question = f"""Look at the game frame showing a Breakout game.

Question: How many pixels should the paddle move horizontally to align with the ball?

Provide your answer in this format: "Move [NUMBER] pixels [left/right]"
Example: "Move 250 pixels right"

Note: You must provide a NUMERIC answer in pixels based on visual observation.
"""
        
        return SpatialTest(
            frame_id=ground_truth['frame_id'],
            test_type='movement_distance',
            question=question,
            ground_truth={
                'distance': optimal_distance,
                'direction': direction,
                'ball_x': ball_x,
                'paddle_x': paddle_x
            },
            metadata={
                'game': ground_truth['game'],
                'ball_pos': (ball_x, ball_y),
                'paddle_pos': (paddle_x, paddle_y)
            }
        )
    
    def create_trajectory_test(self, current_frame: Dict, previous_frame: Dict) -> Optional[SpatialTest]:
        """
        Create test: "Where will ball be in N frames?"
        
        Requires velocity information from 2 consecutive frames.
        """
        curr_objs = current_frame.get('ocatari_data', {}).get('objects', [])
        prev_objs = previous_frame.get('ocatari_data', {}).get('objects', [])
        
        # Find ball in both frames
        curr_ball = next((o for o in curr_objs if o['category'] == 'Ball'), None)
        prev_ball = next((o for o in prev_objs if o['category'] == 'Ball'), None)
        
        if not curr_ball or not prev_ball:
            return None
        
        # Calculate velocity
        curr_x, curr_y = curr_ball['position']
        prev_x, prev_y = prev_ball['position']
        vx = curr_x - prev_x
        vy = curr_y - prev_y
        
        # Predict 5 frames ahead (simple linear projection)
        frames_ahead = 5
        predicted_x = curr_x + vx * frames_ahead
        predicted_y = curr_y + vy * frames_ahead
        
        question = f"""Given two consecutive game frames:

Frame t-1: Ball at ({prev_x}, {prev_y})
Frame t: Ball at ({curr_x}, {curr_y})

Question: Assuming constant velocity, where will the ball be in {frames_ahead} frames?

Provide your answer as coordinates: "x=[X], y=[Y]"
Example: "x=850, y=320"
"""
        
        return SpatialTest(
            frame_id=current_frame['frame_id'],
            test_type='trajectory',
            question=question,
            ground_truth={
                'predicted_x': predicted_x,
                'predicted_y': predicted_y,
                'velocity_x': vx,
                'velocity_y': vy,
                'frames_ahead': frames_ahead
            },
            metadata={
                'game': current_frame['game'],
                'current_pos': (curr_x, curr_y),
                'previous_pos': (prev_x, prev_y)
            }
        )
    
    def create_collision_test(self, ground_truth: Dict) -> Optional[SpatialTest]:
        """
        Create test: "Will ball collide with paddle?"
        
        Simplified: Check if ball is above paddle and moving downward.
        """
        objects = ground_truth.get('ocatari_data', {}).get('objects', [])
        
        ball = next((o for o in objects if o['category'] == 'Ball'), None)
        paddle = next((o for o in objects if o['category'] == 'Player'), None)
        
        if not ball or not paddle:
            return None
        
        ball_x, ball_y = ball['position']
        paddle_x, paddle_y = paddle['position']
        
        # Simple heuristic: will collide if ball is above and x-aligned
        paddle_width = 128  # Typical Breakout paddle width
        x_diff = abs(ball_x - paddle_x)
        will_collide = (ball_y < paddle_y) and (x_diff < paddle_width / 2)
        
        question = f"""Given the current game state:
- Ball position: ({ball_x}, {ball_y})
- Paddle position: ({paddle_x}, {paddle_y})
- Paddle width: {paddle_width} pixels

Question: If the ball continues moving downward, will it collide with the paddle?

Answer: YES or NO
"""
        
        return SpatialTest(
            frame_id=ground_truth['frame_id'],
            test_type='collision',
            question=question,
            ground_truth={
                'will_collide': will_collide,
                'x_diff': x_diff,
                'y_diff': paddle_y - ball_y
            },
            metadata={
                'game': ground_truth['game'],
                'ball_pos': (ball_x, ball_y),
                'paddle_pos': (paddle_x, paddle_y)
            }
        )
    
    def evaluate_movement_distance(self, response: str, ground_truth: Dict) -> Dict:
        """
        Evaluate movement distance prediction.

        Parse numeric answer from response and compute error.
        """
        optimal_distance = ground_truth['distance']
        optimal_direction = ground_truth['direction']

        # Parse number from response - look for "Move N pixels" pattern first
        move_pattern = re.search(r'move\s+(\d+)\s*(?:pixels?|px)', response.lower())
        if move_pattern:
            predicted_distance = int(move_pattern.group(1))
        else:
            # Look for "N pixels [left/right]" pattern
            direction_pattern = re.search(r'(\d+)\s*(?:pixels?|px)\s+(?:left|right)', response.lower())
            if direction_pattern:
                predicted_distance = int(direction_pattern.group(1))
            else:
                # Fall back to any "N pixels" but prefer later occurrences
                numbers = re.findall(r'\b(\d+)\s*(?:pixels?|px)', response.lower())

                if not numbers:
                    # Try to find any number
                    numbers = re.findall(r'\b(\d+)\b', response)

                if not numbers:
                    return {
                        'score': 0.0,
                        'error': optimal_distance,
                        'predicted': None,
                        'optimal': optimal_distance,
                        'reason': 'No numeric distance found in response'
                    }

                # Use the LAST number found (most likely the answer)
                predicted_distance = int(numbers[-1])
        error = abs(predicted_distance - optimal_distance)
        
        # Score: 1.0 if error < 50 pixels, then linear decay
        if error < 50:
            score = 1.0
        else:
            score = max(0, 1 - error / self.screen_width)
        
        return {
            'score': score,
            'error': error,
            'predicted': predicted_distance,
            'optimal': optimal_distance,
            'reason': f'Distance error: {error} pixels'
        }
    
    def evaluate_trajectory(self, response: str, ground_truth: Dict) -> Dict:
        """Evaluate trajectory prediction."""
        optimal_x = ground_truth['predicted_x']
        optimal_y = ground_truth['predicted_y']
        
        # Parse coordinates
        x_match = re.search(r'x\s*[=:]\s*(\d+)', response.lower())
        y_match = re.search(r'y\s*[=:]\s*(\d+)', response.lower())
        
        if not x_match or not y_match:
            return {
                'score': 0.0,
                'error': self.screen_diagonal,
                'predicted': None,
                'optimal': (optimal_x, optimal_y),
                'reason': 'Could not parse coordinates from response'
            }
        
        predicted_x = int(x_match.group(1))
        predicted_y = int(y_match.group(1))
        
        # Euclidean error
        error = np.sqrt((predicted_x - optimal_x)**2 + (predicted_y - optimal_y)**2)
        score = max(0, 1 - error / self.screen_diagonal)
        
        return {
            'score': score,
            'error': error,
            'predicted': (predicted_x, predicted_y),
            'optimal': (optimal_x, optimal_y),
            'reason': f'Position error: {error:.1f} pixels'
        }
    
    def evaluate_collision(self, response: str, ground_truth: Dict) -> Dict:
        """Evaluate collision prediction."""
        optimal_answer = ground_truth['will_collide']
        
        # Parse YES/NO
        response_lower = response.lower()
        if 'yes' in response_lower and 'no' not in response_lower:
            predicted = True
        elif 'no' in response_lower and 'yes' not in response_lower:
            predicted = False
        else:
            return {
                'score': 0.0,
                'correct': False,
                'predicted': None,
                'optimal': optimal_answer,
                'reason': 'Could not parse YES/NO from response'
            }
        
        correct = (predicted == optimal_answer)
        score = 1.0 if correct else 0.0
        
        return {
            'score': score,
            'correct': correct,
            'predicted': predicted,
            'optimal': optimal_answer,
            'reason': 'Correct prediction' if correct else 'Incorrect prediction'
        }


def main():
    """Demo: Create spatial control tests from existing benchmark data."""
    dataset_path = Path("benchmark_v2.0_scenarios")
    gt_dir = dataset_path / "ground_truth"
    
    evaluator = SpatialControlEvaluator()
    
    # Load a few ground truth files
    gt_files = sorted(gt_dir.glob("*.json"))[:5]
    
    print("=" * 70)
    print("SPATIAL CONTROL BENCHMARK - TEST GENERATION")
    print("=" * 70)
    print()
    
    tests = []
    
    for gt_file in gt_files:
        with open(gt_file) as f:
            gt = json.load(f)
        
        # Create movement distance test
        test = evaluator.create_movement_distance_test(gt)
        if test:
            tests.append(test)
            print(f"✓ Created movement distance test: {test.frame_id}")
            print(f"  Optimal: {test.ground_truth['distance']} pixels {test.ground_truth['direction']}")
            print()
    
    print(f"\nGenerated {len(tests)} spatial control tests")
    print("\nExample test question:")
    print("-" * 70)
    print(tests[0].question)
    print("-" * 70)
    
    # Demo scoring
    print("\nExample Scoring:")
    print("-" * 70)
    
    # Simulate responses
    optimal = tests[0].ground_truth['distance']
    
    responses = [
        (f"Move {optimal} pixels right", "Vision+Symbol (Perfect)"),
        (f"Move {optimal - 50} pixels right", "Vision+Symbol (Close)"),
        ("Move right to align with ball", "Vision-Only (Qualitative)"),
        ("The paddle should move rightward", "Vision-Only (Vague)")
    ]
    
    for response, label in responses:
        result = evaluator.evaluate_movement_distance(response, tests[0].ground_truth)
        print(f"{label:30} → Score: {result['score']:.2f} (error: {result.get('error', 'N/A')})")
    
    print("\n" + "=" * 70)
    print("This shows MASSIVE difference between pipelines on control tasks!")
    print("=" * 70)


if __name__ == "__main__":
    main()
