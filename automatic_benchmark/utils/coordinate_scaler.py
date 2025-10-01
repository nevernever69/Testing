"""
Coordinate scaling utilities for benchmark evaluation.
Handles transformation of OCAtari coordinates from original frame size (160x210)
to scaled frame size (1280x720).
"""

from typing import Dict, Any, List, Tuple


class CoordinateScaler:
    """
    Handles coordinate transformations between original Atari frames
    and scaled frames used in benchmark evaluation.
    """

    # Original Atari frame dimensions
    ORIGINAL_WIDTH = 160
    ORIGINAL_HEIGHT = 210

    # Scaled frame dimensions for benchmark
    SCALED_WIDTH = 1280
    SCALED_HEIGHT = 720

    def __init__(self):
        """Initialize coordinate scaler."""
        self.width_scale = self.SCALED_WIDTH / self.ORIGINAL_WIDTH   # 8.0
        self.height_scale = self.SCALED_HEIGHT / self.ORIGINAL_HEIGHT  # 3.43

    def scale_coordinates(self, ground_truth: Dict[str, Any]) -> Dict[str, Any]:
        """
        Scale all coordinates in ground truth from original to scaled dimensions.

        Args:
            ground_truth: Ground truth dict with OCAtari data

        Returns:
            Ground truth with scaled coordinates added
        """
        if 'ocatari_data' not in ground_truth:
            return ground_truth

        ocatari_data = ground_truth['ocatari_data']
        objects = ocatari_data.get('objects', [])

        scaled_objects = []
        for obj in objects:
            scaled_obj = obj.copy()

            # Scale position
            if 'position' in obj:
                x_orig, y_orig = obj['position']
                scaled_obj['position_scaled'] = [
                    int(x_orig * self.width_scale),
                    int(y_orig * self.height_scale)
                ]

            # Scale center
            if 'center' in obj:
                x_orig, y_orig = obj['center']
                scaled_obj['center_scaled'] = [
                    x_orig * self.width_scale,
                    y_orig * self.height_scale
                ]

            # Scale size
            if 'size' in obj:
                w_orig, h_orig = obj['size']
                scaled_obj['size_scaled'] = [
                    int(w_orig * self.width_scale),
                    int(h_orig * self.height_scale)
                ]

            # Velocity doesn't need scaling (it's per-frame)
            # RGB doesn't need scaling

            scaled_objects.append(scaled_obj)

        # Create scaled version of ocatari_data
        ground_truth['ocatari_data_scaled'] = {
            'objects': scaled_objects,
            'frame_dimensions': {
                'original': [self.ORIGINAL_WIDTH, self.ORIGINAL_HEIGHT],
                'scaled': [self.SCALED_WIDTH, self.SCALED_HEIGHT]
            }
        }

        return ground_truth

    def scale_detection_to_original(self, detection_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Scale detection results from scaled coordinates back to original coordinates.
        Useful for comparing detections against original OCAtari ground truth.

        Args:
            detection_results: Detection results with coordinates in scaled space

        Returns:
            Detection results with original coordinates
        """
        if 'objects' not in detection_results:
            return detection_results

        scaled_results = detection_results.copy()
        objects = detection_results['objects']
        original_objects = []

        for obj in objects:
            orig_obj = obj.copy()

            # Scale bounding box coordinates back
            if 'coordinates' in obj:
                x1, y1, x2, y2 = obj['coordinates']
                orig_obj['coordinates_original'] = [
                    int(x1 / self.width_scale),
                    int(y1 / self.height_scale),
                    int(x2 / self.width_scale),
                    int(y2 / self.height_scale)
                ]

            original_objects.append(orig_obj)

        scaled_results['objects'] = original_objects
        return scaled_results

    def get_scaling_info(self) -> Dict[str, Any]:
        """Get information about coordinate scaling."""
        return {
            'original_dimensions': [self.ORIGINAL_WIDTH, self.ORIGINAL_HEIGHT],
            'scaled_dimensions': [self.SCALED_WIDTH, self.SCALED_HEIGHT],
            'width_scale_factor': self.width_scale,
            'height_scale_factor': self.height_scale,
            'aspect_ratio_change': {
                'original': self.ORIGINAL_WIDTH / self.ORIGINAL_HEIGHT,
                'scaled': self.SCALED_WIDTH / self.SCALED_HEIGHT,
                'distortion': (self.SCALED_WIDTH / self.SCALED_HEIGHT) / (self.ORIGINAL_WIDTH / self.ORIGINAL_HEIGHT)
            }
        }


def scale_ground_truth_coordinates(ground_truth: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convenience function to scale ground truth coordinates.

    Args:
        ground_truth: Ground truth dict with OCAtari data

    Returns:
        Ground truth with scaled coordinates
    """
    scaler = CoordinateScaler()
    return scaler.scale_coordinates(ground_truth)
