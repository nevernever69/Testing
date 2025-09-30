"""
Ground Truth Comparison System
This module provides detailed comparison between VLM responses and actual ground truth,
showing exactly what matched, what was missed, and why scores were assigned.
"""

import json
import re
import os
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

class GroundTruthComparator:
    """
    Detailed comparison system that analyzes VLM responses against ground truth
    and provides comprehensive match/miss analysis.
    """

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.comparison_dir = os.path.join(output_dir, "ground_truth_comparisons")
        os.makedirs(self.comparison_dir, exist_ok=True)

        # Tracking all comparisons for summary
        self.all_comparisons = []

    def compare_visual_task(self,
                           response: str,
                           ground_truth: Dict,
                           pipeline_type: str,
                           frame_id: str) -> Dict:
        """
        Compare visual understanding response against ground truth objects.
        """
        comparison = {
            'task_type': 'visual',
            'pipeline_type': pipeline_type,
            'frame_id': frame_id,
            'timestamp': datetime.now().isoformat(),
            'response': response,
            'ground_truth': ground_truth,
            'analysis': {},
            'matches': {},
            'misses': {},
            'score_breakdown': {}
        }

        response_lower = response.lower()

        # Extract objects from ground truth (either OCAtari or symbolic detection)
        gt_objects = []
        if 'objects' in ground_truth and ground_truth['objects']:
            for obj in ground_truth['objects']:
                if isinstance(obj, dict):
                    # Symbolic detection format
                    gt_objects.append({
                        'label': obj.get('label', 'unknown'),
                        'coordinates': obj.get('coordinates', [0, 0, 0, 0]),
                        'description': obj.get('description', '')
                    })
                else:
                    # OCAtari format
                    gt_objects.append({
                        'label': getattr(obj, 'category', 'unknown'),
                        'coordinates': [getattr(obj, 'x', 0), getattr(obj, 'y', 0)],
                        'description': f"{getattr(obj, 'category', 'unknown')} at ({getattr(obj, 'x', 0)}, {getattr(obj, 'y', 0)})"
                    })

        # Object detection analysis
        object_matches = {}
        object_misses = {}

        expected_objects = ['paddle', 'ball', 'brick', 'block', 'player', 'score']
        for expected in expected_objects:
            found_in_response = expected in response_lower
            found_in_gt = any(expected in obj['label'].lower() for obj in gt_objects)

            if found_in_gt:  # This object should be detected
                if found_in_response:
                    object_matches[expected] = {
                        'status': 'CORRECTLY_IDENTIFIED',
                        'in_response': self._find_mentions(response_lower, expected),
                        'in_ground_truth': [obj for obj in gt_objects if expected in obj['label'].lower()]
                    }
                else:
                    object_misses[expected] = {
                        'status': 'MISSED_DETECTION',
                        'should_have_found': [obj for obj in gt_objects if expected in obj['label'].lower()],
                        'penalty': -0.2
                    }

        # Color analysis
        color_matches = {}
        expected_colors = ['white', 'black', 'red', 'green', 'blue', 'yellow', 'orange', 'brown']
        for color in expected_colors:
            if color in response_lower:
                color_matches[color] = {
                    'mentions': self._find_mentions(response_lower, color),
                    'contexts': self._extract_contexts(response_lower, color)
                }

        # Coordinate analysis
        coordinate_analysis = {}
        coord_patterns = [r'\((\d+),\s*(\d+)\)', r'(\d+),\s*(\d+)', r'at\s+(\d+)', r'position\s+(\d+)']
        extracted_coords = []

        for pattern in coord_patterns:
            matches = re.findall(pattern, response_lower)
            if matches:
                extracted_coords.extend(matches)

        if extracted_coords:
            coordinate_analysis = {
                'found_coordinates': extracted_coords,
                'ground_truth_coords': [[obj['coordinates'][0], obj['coordinates'][1]] for obj in gt_objects if len(obj['coordinates']) >= 2],
                'accuracy_analysis': self._analyze_coordinate_accuracy(extracted_coords, gt_objects)
            }

        # Calculate scores
        object_score = len(object_matches) / max(1, len(object_matches) + len(object_misses))
        color_score = min(1.0, len(color_matches) / 3)  # Normalize to max 3 colors
        coord_score = 1.0 if extracted_coords else 0.0

        total_score = (object_score * 0.6 + color_score * 0.2 + coord_score * 0.2)

        comparison.update({
            'analysis': {
                'total_expected_objects': len([obj for obj in gt_objects if any(exp in obj['label'].lower() for exp in expected_objects)]),
                'objects_correctly_identified': len(object_matches),
                'objects_missed': len(object_misses),
                'colors_mentioned': len(color_matches),
                'coordinates_found': len(extracted_coords)
            },
            'matches': {
                'objects': object_matches,
                'colors': color_matches,
                'coordinates': coordinate_analysis
            },
            'misses': {
                'objects': object_misses
            },
            'score_breakdown': {
                'object_detection': object_score,
                'color_detection': color_score,
                'coordinate_precision': coord_score,
                'total_score': total_score
            }
        })

        return comparison

    def compare_spatial_task(self,
                            response: str,
                            ground_truth: Dict,
                            pipeline_type: str,
                            frame_id: str) -> Dict:
        """
        Compare spatial reasoning response against ground truth relationships.
        """
        comparison = {
            'task_type': 'spatial',
            'pipeline_type': pipeline_type,
            'frame_id': frame_id,
            'timestamp': datetime.now().isoformat(),
            'response': response,
            'ground_truth': ground_truth,
            'analysis': {},
            'matches': {},
            'misses': {},
            'score_breakdown': {}
        }

        response_lower = response.lower()

        # Extract spatial terms
        spatial_terms = ['left', 'right', 'top', 'bottom', 'center', 'middle', 'above', 'below',
                        'near', 'far', 'close', 'distant', 'between', 'corner', 'edge']

        found_spatial_terms = {}
        for term in spatial_terms:
            if term in response_lower:
                found_spatial_terms[term] = {
                    'mentions': response_lower.count(term),
                    'contexts': self._extract_contexts(response_lower, term)
                }

        # Analyze actual spatial relationships from ground truth
        gt_relationships = self._calculate_spatial_relationships(ground_truth)

        # Check if spatial descriptions match reality
        relationship_matches = {}
        relationship_misses = {}

        for rel_type, relationships in gt_relationships.items():
            for rel in relationships:
                description = rel['description']
                # Check if this relationship is mentioned in the response
                if any(word in response_lower for word in rel['keywords']):
                    relationship_matches[description] = {
                        'status': 'CORRECTLY_DESCRIBED',
                        'ground_truth': rel,
                        'evidence_in_response': [word for word in rel['keywords'] if word in response_lower]
                    }
                else:
                    relationship_misses[description] = {
                        'status': 'MISSED_RELATIONSHIP',
                        'should_have_mentioned': rel['keywords'],
                        'actual_relationship': rel
                    }

        # Calculate spatial accuracy score
        spatial_term_score = min(1.0, len(found_spatial_terms) / 5)  # Expect ~5 spatial terms
        relationship_score = len(relationship_matches) / max(1, len(relationship_matches) + len(relationship_misses))
        total_score = (spatial_term_score * 0.4 + relationship_score * 0.6)

        comparison.update({
            'analysis': {
                'spatial_terms_found': len(found_spatial_terms),
                'relationships_correctly_described': len(relationship_matches),
                'relationships_missed': len(relationship_misses),
                'total_gt_relationships': len(relationship_matches) + len(relationship_misses)
            },
            'matches': {
                'spatial_terms': found_spatial_terms,
                'relationships': relationship_matches
            },
            'misses': {
                'relationships': relationship_misses
            },
            'score_breakdown': {
                'spatial_vocabulary': spatial_term_score,
                'relationship_accuracy': relationship_score,
                'total_score': total_score
            }
        })

        return comparison

    def save_comparison(self, comparison: Dict) -> str:
        """Save comparison to file and add to summary."""
        filename = f"{comparison['task_type']}_{comparison['pipeline_type']}_{comparison['frame_id']}_{int(datetime.now().timestamp()*1000)}.json"
        filepath = os.path.join(self.comparison_dir, filename)

        with open(filepath, 'w') as f:
            json.dump(comparison, f, indent=2, default=str)

        self.all_comparisons.append(comparison)
        return filepath

    def generate_summary_report(self) -> str:
        """Generate comprehensive summary of all comparisons."""
        summary_file = os.path.join(self.comparison_dir, "GROUND_TRUTH_COMPARISON_SUMMARY.txt")

        with open(summary_file, 'w') as f:
            f.write("GROUND TRUTH COMPARISON SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Generated: {datetime.now().isoformat()}\n")
            f.write(f"Total Comparisons: {len(self.all_comparisons)}\n\n")

            # Group by task type and pipeline
            by_task = {}
            by_pipeline = {}

            for comp in self.all_comparisons:
                task = comp['task_type']
                pipeline = comp['pipeline_type']

                if task not in by_task:
                    by_task[task] = []
                by_task[task].append(comp)

                if pipeline not in by_pipeline:
                    by_pipeline[pipeline] = []
                by_pipeline[pipeline].append(comp)

            # Task-wise analysis
            f.write("ANALYSIS BY TASK TYPE\n")
            f.write("-" * 30 + "\n")

            for task_type, comparisons in by_task.items():
                f.write(f"\n{task_type.upper()} TASK:\n")
                f.write(f"  Total evaluations: {len(comparisons)}\n")

                avg_score = sum(c['score_breakdown']['total_score'] for c in comparisons) / len(comparisons)
                f.write(f"  Average score: {avg_score:.3f}\n")

                if task_type == 'visual':
                    total_objects = sum(c['analysis']['total_expected_objects'] for c in comparisons)
                    found_objects = sum(c['analysis']['objects_correctly_identified'] for c in comparisons)
                    f.write(f"  Object detection rate: {found_objects}/{total_objects} ({found_objects/max(1,total_objects)*100:.1f}%)\n")

                elif task_type == 'spatial':
                    total_relationships = sum(c['analysis']['total_gt_relationships'] for c in comparisons)
                    found_relationships = sum(c['analysis']['relationships_correctly_described'] for c in comparisons)
                    f.write(f"  Relationship description rate: {found_relationships}/{total_relationships} ({found_relationships/max(1,total_relationships)*100:.1f}%)\n")

            # Pipeline comparison
            f.write("\n\nANALYSIS BY PIPELINE\n")
            f.write("-" * 30 + "\n")

            for pipeline_type, comparisons in by_pipeline.items():
                f.write(f"\n{pipeline_type.upper()} PIPELINE:\n")
                f.write(f"  Total evaluations: {len(comparisons)}\n")

                avg_score = sum(c['score_breakdown']['total_score'] for c in comparisons) / len(comparisons)
                f.write(f"  Average score: {avg_score:.3f}\n")

            f.write(f"\n\nDETAILED FILES LOCATION: {self.comparison_dir}\n")
            f.write("Individual comparison files contain complete match/miss analysis.\n")

        return summary_file

    def _find_mentions(self, text: str, word: str) -> List[Dict]:
        """Find all mentions of a word with positions and contexts."""
        mentions = []
        start = 0
        while True:
            pos = text.find(word, start)
            if pos == -1:
                break
            mentions.append({
                'position': pos,
                'context': self._extract_contexts(text, word, pos)[0] if self._extract_contexts(text, word, pos) else ''
            })
            start = pos + 1
        return mentions

    def _extract_contexts(self, text: str, word: str, position: int = None) -> List[str]:
        """Extract contexts around word mentions."""
        if position is not None:
            start = max(0, position - 20)
            end = min(len(text), position + len(word) + 20)
            return [text[start:end]]

        contexts = []
        words = text.split()
        for i, w in enumerate(words):
            if word in w:
                start_idx = max(0, i - 3)
                end_idx = min(len(words), i + 4)
                context = ' '.join(words[start_idx:end_idx])
                contexts.append(context)
        return contexts

    def _analyze_coordinate_accuracy(self, extracted_coords: List, gt_objects: List) -> Dict:
        """Analyze accuracy of extracted coordinates against ground truth."""
        if not extracted_coords or not gt_objects:
            return {'accuracy': 0.0, 'analysis': 'No coordinates to compare'}

        # Convert extracted coords to numbers
        numeric_coords = []
        for coord in extracted_coords:
            if isinstance(coord, tuple):
                try:
                    numeric_coords.append([int(coord[0]), int(coord[1])])
                except:
                    pass
            elif isinstance(coord, str) and coord.isdigit():
                numeric_coords.append([int(coord), 0])  # Single coordinate

        if not numeric_coords:
            return {'accuracy': 0.0, 'analysis': 'Could not parse coordinates'}

        # Compare with ground truth
        gt_coords = []
        for obj in gt_objects:
            if len(obj['coordinates']) >= 2:
                # Take center of bounding box if available
                if len(obj['coordinates']) == 4:  # [x1, y1, x2, y2]
                    x = (obj['coordinates'][0] + obj['coordinates'][2]) // 2
                    y = (obj['coordinates'][1] + obj['coordinates'][3]) // 2
                    gt_coords.append([x, y])
                else:
                    gt_coords.append([obj['coordinates'][0], obj['coordinates'][1]])

        # Calculate minimum distances
        accuracies = []
        for extracted in numeric_coords:
            min_dist = float('inf')
            closest_gt = None
            for gt in gt_coords:
                dist = ((extracted[0] - gt[0])**2 + (extracted[1] - gt[1])**2)**0.5
                if dist < min_dist:
                    min_dist = dist
                    closest_gt = gt

            accuracies.append({
                'extracted': extracted,
                'closest_ground_truth': closest_gt,
                'distance_pixels': min_dist,
                'accurate_within_20px': min_dist <= 20
            })

        overall_accuracy = sum(1 for acc in accuracies if acc['accurate_within_20px']) / len(accuracies)

        return {
            'accuracy': overall_accuracy,
            'individual_accuracies': accuracies,
            'mean_distance': sum(acc['distance_pixels'] for acc in accuracies) / len(accuracies)
        }

    def _calculate_spatial_relationships(self, ground_truth: Dict) -> Dict:
        """Calculate actual spatial relationships from ground truth."""
        relationships = {
            'position_relationships': [],
            'distance_relationships': []
        }

        if 'objects' not in ground_truth or len(ground_truth['objects']) < 2:
            return relationships

        objects = ground_truth['objects']

        # Calculate relationships between all object pairs
        for i in range(len(objects)):
            for j in range(i + 1, len(objects)):
                obj1 = objects[i]
                obj2 = objects[j]

                # Get positions (handle different formats)
                pos1 = self._get_object_position(obj1)
                pos2 = self._get_object_position(obj2)

                if pos1 and pos2:
                    # Position relationships
                    if pos1[0] < pos2[0]:
                        relationships['position_relationships'].append({
                            'description': f"{self._get_object_label(obj1)} is left of {self._get_object_label(obj2)}",
                            'keywords': ['left', 'left of'],
                            'type': 'horizontal_position',
                            'objects': [obj1, obj2]
                        })
                    elif pos1[0] > pos2[0]:
                        relationships['position_relationships'].append({
                            'description': f"{self._get_object_label(obj1)} is right of {self._get_object_label(obj2)}",
                            'keywords': ['right', 'right of'],
                            'type': 'horizontal_position',
                            'objects': [obj1, obj2]
                        })

                    if pos1[1] < pos2[1]:
                        relationships['position_relationships'].append({
                            'description': f"{self._get_object_label(obj1)} is above {self._get_object_label(obj2)}",
                            'keywords': ['above', 'top', 'upper'],
                            'type': 'vertical_position',
                            'objects': [obj1, obj2]
                        })
                    elif pos1[1] > pos2[1]:
                        relationships['position_relationships'].append({
                            'description': f"{self._get_object_label(obj1)} is below {self._get_object_label(obj2)}",
                            'keywords': ['below', 'bottom', 'lower'],
                            'type': 'vertical_position',
                            'objects': [obj1, obj2]
                        })

                    # Distance relationships
                    distance = ((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)**0.5
                    if distance < 50:
                        relationships['distance_relationships'].append({
                            'description': f"{self._get_object_label(obj1)} is near {self._get_object_label(obj2)}",
                            'keywords': ['near', 'close', 'close to'],
                            'distance_pixels': distance,
                            'objects': [obj1, obj2]
                        })
                    elif distance > 200:
                        relationships['distance_relationships'].append({
                            'description': f"{self._get_object_label(obj1)} is far from {self._get_object_label(obj2)}",
                            'keywords': ['far', 'distant', 'away from'],
                            'distance_pixels': distance,
                            'objects': [obj1, obj2]
                        })

        return relationships

    def _get_object_position(self, obj: Dict) -> Optional[List[int]]:
        """Extract object position handling different formats."""
        if isinstance(obj, dict):
            if 'coordinates' in obj:
                coords = obj['coordinates']
                if len(coords) >= 2:
                    # If bounding box [x1, y1, x2, y2], take center
                    if len(coords) == 4:
                        return [(coords[0] + coords[2]) // 2, (coords[1] + coords[3]) // 2]
                    else:
                        return [coords[0], coords[1]]
            elif 'x' in obj and 'y' in obj:
                return [obj['x'], obj['y']]
        else:
            # OCAtari object
            if hasattr(obj, 'x') and hasattr(obj, 'y'):
                return [obj.x, obj.y]
            elif hasattr(obj, 'position'):
                return [obj.position[0], obj.position[1]]
        return None

    def _get_object_label(self, obj: Dict) -> str:
        """Extract object label handling different formats."""
        if isinstance(obj, dict):
            return obj.get('label', obj.get('category', 'unknown'))
        else:
            return getattr(obj, 'category', getattr(obj, 'label', 'unknown'))