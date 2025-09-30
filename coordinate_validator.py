"""
Coordinate Validation System

This module validates that OCAtari RAM-extracted coordinates correctly correspond
to visual positions in the rendered frames. This is crucial for ensuring our
ground truth data is accurate for spatial reasoning benchmarks.

Usage:
    validator = CoordinateValidator('Pong')
    validation_results = validator.validate_coordinates()
    validator.create_validation_visualization('pong_validation.png')
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import Dict, List, Tuple, Any
import cv2
from ocatari_ground_truth import OCAtariGroundTruth, ObjectInfo


class CoordinateValidator:
    """
    Validates OCAtari coordinates against visual frame content.

    Provides methods to:
    - Visual validation of object positions
    - Accuracy measurement of coordinate extraction
    - Detection of coordinate system inconsistencies
    """

    def __init__(self, game_name: str):
        """
        Initialize coordinate validator.

        Args:
            game_name: Name of Atari game to validate
        """
        self.game_name = game_name
        self.extractor = OCAtariGroundTruth(game_name)
        self.validation_results = []

    def validate_coordinates(self, num_samples: int = 20) -> Dict[str, Any]:
        """
        Validate coordinates across multiple frames.

        Args:
            num_samples: Number of frames to sample for validation

        Returns:
            Dictionary with validation statistics and results
        """
        print(f"Validating coordinates for {self.game_name} across {num_samples} samples...")

        validation_data = []

        for sample in range(num_samples):
            # Take some steps to get varied scenarios
            for _ in range(np.random.randint(5, 15)):
                action = self.extractor.env.action_space.sample()
                self.extractor.step(action)

            # Get frame and objects
            frame, objects = self.extractor.get_frame_and_objects()

            # Validate each object
            frame_validation = {
                'sample': sample,
                'frame_count': self.extractor.frame_count,
                'objects': [],
                'frame_shape': frame.shape
            }

            for obj in objects:
                obj_validation = self._validate_single_object(frame, obj)
                frame_validation['objects'].append(obj_validation)

            validation_data.append(frame_validation)

        self.validation_results = validation_data
        return self._analyze_validation_results()

    def _validate_single_object(self, frame: np.ndarray, obj: ObjectInfo) -> Dict[str, Any]:
        """Validate coordinates for a single object."""
        x, y = obj.position
        w, h = obj.size if obj.size != (0, 0) else (4, 4)  # Default size

        validation = {
            'object_id': obj.id,
            'category': obj.category,
            'reported_position': obj.position,
            'reported_size': obj.size,
            'validation_status': 'unknown',
            'issues': []
        }

        # Check bounds
        frame_height, frame_width = frame.shape[:2]

        # Validate position is within frame bounds
        if x < 0 or x >= frame_width or y < 0 or y >= frame_height:
            validation['validation_status'] = 'out_of_bounds'
            validation['issues'].append(f"Position ({x}, {y}) outside frame bounds ({frame_width}, {frame_height})")
            return validation

        # Extract region around object for visual validation
        margin = max(w, h, 10)  # Add margin for analysis
        x_start = max(0, x - margin)
        x_end = min(frame_width, x + margin)
        y_start = max(0, y - margin)
        y_end = min(frame_height, y + margin)

        region = frame[y_start:y_end, x_start:x_end]

        # Analyze region for object presence
        object_detected = self._detect_object_in_region(region, obj)

        if object_detected:
            validation['validation_status'] = 'valid'
        else:
            validation['validation_status'] = 'questionable'
            validation['issues'].append("Visual analysis suggests no clear object at reported position")

        validation['analyzed_region'] = {
            'bounds': (x_start, y_start, x_end, y_end),
            'region_shape': region.shape
        }

        return validation

    def _detect_object_in_region(self, region: np.ndarray, obj: ObjectInfo) -> bool:
        """
        Simple object detection in region using color analysis.

        This is a heuristic approach - we look for non-background colors
        and sufficient color variation that would indicate an object.
        """
        if region.size == 0:
            return False

        # Expected object color from OCAtari
        expected_color = obj.properties.get('rgb')

        if expected_color is not None:
            # Look for pixels similar to expected color
            color_diff = np.linalg.norm(region - np.array(expected_color), axis=2)
            similar_pixels = np.sum(color_diff < 30)  # Tolerance for color matching

            # If we find enough similar pixels, consider object detected
            total_pixels = region.shape[0] * region.shape[1]
            if similar_pixels > total_pixels * 0.01:  # At least 1% of pixels match
                return True

        # Fallback: Look for any non-background activity
        # Background in Atari games is typically black (0,0,0)
        non_black_pixels = np.sum(np.any(region > 10, axis=2))
        total_pixels = region.shape[0] * region.shape[1]

        # If more than 5% of pixels are non-black, assume object present
        return non_black_pixels > total_pixels * 0.05

    def _analyze_validation_results(self) -> Dict[str, Any]:
        """Analyze validation results and compute statistics."""
        if not self.validation_results:
            return {}

        total_objects = 0
        valid_objects = 0
        out_of_bounds = 0
        questionable = 0

        categories = {}

        for frame_data in self.validation_results:
            for obj_validation in frame_data['objects']:
                total_objects += 1

                status = obj_validation['validation_status']
                category = obj_validation['category']

                # Count by status
                if status == 'valid':
                    valid_objects += 1
                elif status == 'out_of_bounds':
                    out_of_bounds += 1
                elif status == 'questionable':
                    questionable += 1

                # Count by category
                if category not in categories:
                    categories[category] = {'total': 0, 'valid': 0, 'questionable': 0, 'out_of_bounds': 0}

                categories[category]['total'] += 1
                categories[category][status] += 1

        analysis = {
            'total_samples': len(self.validation_results),
            'total_objects_analyzed': total_objects,
            'validation_summary': {
                'valid': valid_objects,
                'questionable': questionable,
                'out_of_bounds': out_of_bounds,
                'valid_percentage': (valid_objects / total_objects * 100) if total_objects > 0 else 0
            },
            'by_category': categories,
            'recommendations': self._generate_recommendations(valid_objects, total_objects, categories)
        }

        return analysis

    def _generate_recommendations(self, valid_objects: int, total_objects: int,
                                categories: Dict[str, Dict[str, int]]) -> List[str]:
        """Generate recommendations based on validation results."""
        recommendations = []

        if total_objects == 0:
            recommendations.append("No objects detected - check OCAtari configuration")
            return recommendations

        valid_percentage = valid_objects / total_objects * 100

        if valid_percentage < 70:
            recommendations.append("Low validation rate - consider coordinate system adjustments")
        elif valid_percentage < 85:
            recommendations.append("Moderate validation rate - some coordinate discrepancies detected")
        else:
            recommendations.append("High validation rate - coordinates appear accurate")

        # Category-specific recommendations
        for category, stats in categories.items():
            category_valid_rate = stats['valid'] / stats['total'] * 100 if stats['total'] > 0 else 0

            if category_valid_rate < 50:
                recommendations.append(f"{category} objects have low validation rate ({category_valid_rate:.1f}%)")
            elif stats['out_of_bounds'] > 0:
                recommendations.append(f"{category} objects sometimes appear out of bounds - check coordinate scaling")

        return recommendations

    def create_validation_visualization(self, output_path: str, max_frames: int = 6):
        """
        Create visual validation plots showing objects and their coordinates.

        Args:
            output_path: Path to save validation visualization
            max_frames: Maximum number of frames to include in visualization
        """
        if not self.validation_results:
            print("No validation data available. Run validate_coordinates() first.")
            return

        # Select frames to visualize
        frames_to_show = min(max_frames, len(self.validation_results))
        selected_frames = np.linspace(0, len(self.validation_results)-1, frames_to_show, dtype=int)

        fig, axes = plt.subplots(2, frames_to_show//2 if frames_to_show > 1 else 1,
                               figsize=(15, 10))
        if frames_to_show == 1:
            axes = [axes]
        else:
            axes = axes.flatten()

        for idx, frame_idx in enumerate(selected_frames):
            if idx >= len(axes):
                break

            frame_data = self.validation_results[frame_idx]

            # Get the actual frame
            self.extractor.frame_count = frame_data['frame_count']
            frame, objects = self.extractor.get_frame_and_objects()

            ax = axes[idx]
            ax.imshow(frame)
            ax.set_title(f'Frame {frame_data["frame_count"]} - {len(objects)} objects')

            # Draw object positions and validation results
            for obj_val in frame_data['objects']:
                x, y = obj_val['reported_position']
                status = obj_val['validation_status']
                category = obj_val['category']

                # Color code by validation status
                if status == 'valid':
                    color = 'green'
                elif status == 'questionable':
                    color = 'yellow'
                else:  # out_of_bounds
                    color = 'red'

                # Draw bounding box
                if 'analyzed_region' in obj_val:
                    bounds = obj_val['analyzed_region']['bounds']
                    x_start, y_start, x_end, y_end = bounds
                    rect = patches.Rectangle((x_start, y_start), x_end-x_start, y_end-y_start,
                                           linewidth=2, edgecolor=color, facecolor='none', alpha=0.7)
                    ax.add_patch(rect)

                # Mark object center
                ax.plot(x, y, 'o', color=color, markersize=8)
                ax.text(x, y-5, f'{category[:3]}', color=color, fontsize=8,
                       ha='center', weight='bold')

            ax.set_xlim(0, frame.shape[1])
            ax.set_ylim(frame.shape[0], 0)  # Flip y-axis for image coordinates

        # Remove empty subplots
        for idx in range(frames_to_show, len(axes)):
            fig.delaxes(axes[idx])

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"Validation visualization saved to {output_path}")

    def export_validation_report(self, report_path: str):
        """Export detailed validation report to JSON."""
        if not self.validation_results:
            print("No validation data available. Run validate_coordinates() first.")
            return

        analysis = self._analyze_validation_results()

        report = {
            'game': self.game_name,
            'validation_analysis': analysis,
            'detailed_results': self.validation_results
        }

        import json
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"Validation report exported to {report_path}")

    def close(self):
        """Close the underlying OCAtari extractor."""
        self.extractor.close()


def validate_all_games(games: List[str] = None) -> Dict[str, Dict[str, Any]]:
    """
    Validate coordinates for all specified games.

    Args:
        games: List of game names to validate. Defaults to ['Pong', 'Breakout', 'SpaceInvaders']

    Returns:
        Dictionary mapping game names to validation results
    """
    if games is None:
        games = ['Pong', 'Breakout', 'SpaceInvaders']

    all_results = {}

    for game in games:
        print(f"\n{'='*50}")
        print(f"Validating {game}")
        print(f"{'='*50}")

        validator = CoordinateValidator(game)

        try:
            results = validator.validate_coordinates(num_samples=10)
            all_results[game] = results

            # Create visualization
            validator.create_validation_visualization(f'{game.lower()}_validation.png')

            # Export detailed report
            validator.export_validation_report(f'{game.lower()}_validation_report.json')

            # Print summary
            print(f"\nValidation Summary for {game}:")
            print(f"  Total objects analyzed: {results['total_objects_analyzed']}")
            print(f"  Valid: {results['validation_summary']['valid']} ({results['validation_summary']['valid_percentage']:.1f}%)")
            print(f"  Questionable: {results['validation_summary']['questionable']}")
            print(f"  Out of bounds: {results['validation_summary']['out_of_bounds']}")
            print("\nRecommendations:")
            for rec in results['recommendations']:
                print(f"  - {rec}")

        except Exception as e:
            print(f"Error validating {game}: {e}")
            all_results[game] = {'error': str(e)}

        finally:
            validator.close()

    return all_results


if __name__ == "__main__":
    # Run validation for all games
    results = validate_all_games()

    print(f"\n{'='*60}")
    print("OVERALL VALIDATION SUMMARY")
    print(f"{'='*60}")

    for game, result in results.items():
        if 'error' not in result:
            valid_pct = result['validation_summary']['valid_percentage']
            print(f"{game}: {valid_pct:.1f}% valid coordinates")
        else:
            print(f"{game}: Validation failed - {result['error']}")