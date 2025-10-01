"""
Base scorer abstract class.
All scorers inherit from this to ensure consistent interface.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, Optional, List


@dataclass
class ScoringResult:
    """Result from a scorer."""
    score: float  # 0.0 to 1.0
    confidence: float  # 0.0 to 1.0
    reasoning: str
    details: Optional[Dict[str, Any]] = None


class BaseScorer(ABC):
    """
    Abstract base class for all scorers.

    All scorers must implement the score() method.
    """

    def __init__(self, **kwargs):
        """Initialize scorer with optional configuration."""
        self.config = kwargs

    @abstractmethod
    def score(
        self,
        response: str,
        task_type: str,
        ground_truth: Dict[str, Any],
        game_name: str
    ) -> ScoringResult:
        """
        Score a VLM response against ground truth.

        Args:
            response: VLM response text
            task_type: 'visual', 'spatial', 'strategy', or 'identification'
            ground_truth: Ground truth data from OCAtari
            game_name: Name of the game

        Returns:
            ScoringResult with score, confidence, and reasoning
        """
        pass

    def _clamp_score(self, score: float) -> float:
        """Ensure score is in [0, 1] range."""
        return max(0.0, min(1.0, score))

    def _validate_task_type(self, task_type: str):
        """Validate task type."""
        valid_types = ['visual', 'spatial', 'strategy', 'identification']
        if task_type not in valid_types:
            raise ValueError(f"Invalid task type: {task_type}. "
                           f"Must be one of {valid_types}")

    def _extract_objects(self, ground_truth: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract objects from ground truth."""
        return ground_truth.get('ocatari_data', {}).get('objects', [])

    def _extract_spatial_relationships(self, ground_truth: Dict[str, Any]) -> Dict[str, Any]:
        """Extract spatial relationships from ground truth."""
        return ground_truth.get('ocatari_data', {}).get('spatial_relationships', {})

    def _extract_action_space(self, ground_truth: Dict[str, Any]) -> Dict[str, Any]:
        """Extract action space from ground truth."""
        return ground_truth.get('ocatari_data', {}).get('action_space', {})
