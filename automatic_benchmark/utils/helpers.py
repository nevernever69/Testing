"""Helper functions for the benchmark system."""

import numpy as np
import re
from typing import List, Tuple, Optional, Dict, Any


def calculate_distance(pos1: Tuple[float, float], pos2: Tuple[float, float]) -> float:
    """
    Calculate Euclidean distance between two positions.

    Args:
        pos1: (x, y) coordinates of first position
        pos2: (x, y) coordinates of second position

    Returns:
        Distance in pixels
    """
    return np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)


def normalize_score(score: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
    """
    Normalize a score to [0, 1] range.

    Args:
        score: Raw score
        min_val: Minimum possible value
        max_val: Maximum possible value

    Returns:
        Normalized score in [0, 1]
    """
    normalized = (score - min_val) / (max_val - min_val) if max_val > min_val else 0.0
    return max(0.0, min(1.0, normalized))


def extract_coordinates(text: str) -> List[Tuple[int, int]]:
    """
    Extract coordinate mentions from text.

    Recognizes patterns like:
    - "x=140", "y=25"
    - "x≈140", "y≈25"
    - "position (140, 25)"
    - "at (140, 25)"

    Args:
        text: Text to search for coordinates

    Returns:
        List of (x, y) coordinate tuples found
    """
    coordinates = []

    # Pattern 1: (x, y) tuples
    tuple_pattern = r'\((\d+),\s*(\d+)\)'
    for match in re.finditer(tuple_pattern, text):
        x, y = int(match.group(1)), int(match.group(2))
        coordinates.append((x, y))

    # Pattern 2: Separate x= and y= mentions
    x_pattern = r'x\s*[=≈:]\s*(\d+)'
    y_pattern = r'y\s*[=≈:]\s*(\d+)'

    x_matches = [int(m.group(1)) for m in re.finditer(x_pattern, text)]
    y_matches = [int(m.group(1)) for m in re.finditer(y_pattern, text)]

    # Pair x and y values if counts match
    if len(x_matches) == len(y_matches):
        for x, y in zip(x_matches, y_matches):
            if (x, y) not in coordinates:
                coordinates.append((x, y))

    return coordinates


def extract_numbers(text: str) -> List[int]:
    """
    Extract all numbers from text.

    Args:
        text: Text to search

    Returns:
        List of integers found
    """
    return [int(n) for n in re.findall(r'\d+', text)]


def check_coordinate_accuracy(
    mentioned_coords: List[Tuple[int, int]],
    ground_truth_objects: List[Dict[str, Any]],
    tolerance: int = 20
) -> float:
    """
    Check how accurate mentioned coordinates are against ground truth.

    Args:
        mentioned_coords: Coordinates mentioned in response
        ground_truth_objects: Objects with true positions
        tolerance: Allowed pixel error

    Returns:
        Accuracy score [0, 1]
    """
    if not mentioned_coords:
        return 0.0

    correct = 0
    for coord in mentioned_coords:
        for obj in ground_truth_objects:
            obj_pos = obj.get('position', obj.get('center', (0, 0)))
            if isinstance(obj_pos, (list, tuple)) and len(obj_pos) >= 2:
                dist = calculate_distance(coord, (obj_pos[0], obj_pos[1]))
                if dist <= tolerance:
                    correct += 1
                    break

    return correct / len(mentioned_coords)


def categorize_complexity(
    object_count: int,
    spatial_density: float
) -> str:
    """
    Categorize frame complexity.

    Args:
        object_count: Number of objects detected
        spatial_density: Objects per unit area

    Returns:
        'easy', 'medium', or 'hard'
    """
    from ..config import COMPLEXITY_THRESHOLDS

    if object_count <= COMPLEXITY_THRESHOLDS['object_count']['easy']:
        return 'easy'
    elif object_count <= COMPLEXITY_THRESHOLDS['object_count']['medium']:
        return 'medium'
    else:
        return 'hard'


def calculate_spatial_relationships(objects: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Calculate spatial relationships between objects.

    Args:
        objects: List of objects with positions

    Returns:
        Dictionary with distances, relative positions, etc.
    """
    if not objects:
        return {
            'distances': {},
            'object_positions': [],
            'object_categories': [],
            'leftmost_object': None,
            'rightmost_object': None,
            'topmost_object': None,
            'bottommost_object': None
        }

    # Extract positions and categories
    positions = []
    categories = []

    for obj in objects:
        pos = obj.get('position', obj.get('center'))
        if pos:
            positions.append(tuple(pos[:2]))  # (x, y)
            categories.append(obj.get('category', 'Unknown'))

    # Calculate pairwise distances
    distances = {}
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            dist = calculate_distance(positions[i], positions[j])
            distances[f'{i}_{j}'] = dist
            distances[f'{j}_{i}'] = dist

    # Find extreme positions
    x_coords = [pos[0] for pos in positions]
    y_coords = [pos[1] for pos in positions]

    leftmost = np.argmin(x_coords) if x_coords else None
    rightmost = np.argmax(x_coords) if x_coords else None
    topmost = np.argmin(y_coords) if y_coords else None
    bottommost = np.argmax(y_coords) if y_coords else None

    # Convert numpy types to Python types for JSON serialization
    def convert_to_python(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_python(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_to_python(item) for item in obj]
        else:
            return obj

    return {
        'distances': convert_to_python(distances),
        'object_positions': positions,
        'object_categories': categories,
        'leftmost_object': convert_to_python(leftmost),
        'rightmost_object': convert_to_python(rightmost),
        'topmost_object': convert_to_python(topmost),
        'bottommost_object': convert_to_python(bottommost)
    }


def generate_spatial_description(ground_truth: Dict[str, Any]) -> str:
    """
    Generate natural language spatial description from ground truth.

    Args:
        ground_truth: Ground truth data with objects and relationships

    Returns:
        Natural language description
    """
    objects = ground_truth.get('ocatari_data', {}).get('objects', [])
    relationships = ground_truth.get('ocatari_data', {}).get('spatial_relationships', {})

    if not objects:
        return "No objects detected in frame."

    description_parts = []

    # Describe objects with positions
    for obj in objects:
        category = obj.get('category', 'Unknown')
        position = obj.get('position', (0, 0))
        description_parts.append(f"{category} at position ({position[0]}, {position[1]})")

    # Add relative position information
    categories = relationships.get('object_categories', [])
    if categories and relationships.get('leftmost_object') is not None:
        leftmost_idx = relationships['leftmost_object']
        rightmost_idx = relationships['rightmost_object']

        leftmost = categories[leftmost_idx]
        rightmost = categories[rightmost_idx]

        description_parts.append(f"{leftmost} is leftmost, {rightmost} is rightmost")

    return ". ".join(description_parts) + "."


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Safely divide two numbers, returning default if denominator is zero.

    Args:
        numerator: Numerator
        denominator: Denominator
        default: Value to return if division by zero

    Returns:
        Result of division or default
    """
    return numerator / denominator if denominator != 0 else default
