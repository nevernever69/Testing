"""
Semantic similarity scorer.
Uses sentence embeddings to measure semantic similarity between response and ground truth.
"""

import numpy as np
from typing import Dict, Any
from .base_scorer import BaseScorer, ScoringResult
from ..config import SEMANTIC_MODEL_NAME, SEMANTIC_SIMILARITY_THRESHOLD
from ..utils.helpers import generate_spatial_description


class SemanticScorer(BaseScorer):
    """
    Semantic similarity scorer using sentence transformers.
    Provides more nuanced scoring than simple keyword matching.
    """

    def __init__(self, model_name: str = SEMANTIC_MODEL_NAME, **kwargs):
        """
        Initialize semantic scorer.

        Args:
            model_name: Name of sentence transformer model
        """
        super().__init__(**kwargs)

        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name)
            self.available = True
        except ImportError as e:
            print(f"Warning: sentence-transformers not installed. Semantic scoring disabled.")
            print(f"Install with: pip install sentence-transformers")
            print(f"Error: {e}")
            self.model = None
            self.available = False
        except Exception as e:
            print(f"Warning: Failed to load sentence transformer model '{model_name}'")
            print(f"Error: {e}")
            self.model = None
            self.available = False

    def score(
        self,
        response: str,
        task_type: str,
        ground_truth: Dict[str, Any],
        game_name: str
    ) -> ScoringResult:
        """Score response using semantic similarity."""
        self._validate_task_type(task_type)

        if not self.available:
            return ScoringResult(
                score=0.5,
                confidence=0.0,
                reasoning="Semantic scoring not available (model not loaded)"
            )

        if not response or response.startswith("ERROR"):
            return ScoringResult(
                score=0.0,
                confidence=1.0,
                reasoning="No valid response provided"
            )

        # Generate reference description from ground truth
        reference = self._generate_reference_description(task_type, ground_truth, game_name)

        # Calculate semantic similarity
        similarity = self._calculate_similarity(response, reference)

        # Interpret similarity score
        if similarity >= SEMANTIC_SIMILARITY_THRESHOLD:
            interpretation = "High semantic similarity"
            confidence = 0.9
        elif similarity >= 0.5:
            interpretation = "Moderate semantic similarity"
            confidence = 0.7
        elif similarity >= 0.3:
            interpretation = "Low semantic similarity"
            confidence = 0.6
        else:
            interpretation = "Very low semantic similarity"
            confidence = 0.8

        return ScoringResult(
            score=self._clamp_score(similarity),
            confidence=confidence,
            reasoning=f"{interpretation} (similarity={similarity:.3f})",
            details={
                'similarity': similarity,
                'reference_description': reference,
                'threshold': SEMANTIC_SIMILARITY_THRESHOLD
            }
        )

    def _generate_reference_description(
        self,
        task_type: str,
        ground_truth: Dict[str, Any],
        game_name: str
    ) -> str:
        """Generate reference description from ground truth."""
        if task_type == 'visual':
            return self._generate_visual_reference(ground_truth)
        elif task_type == 'spatial':
            return self._generate_spatial_reference(ground_truth)
        elif task_type == 'strategy':
            return self._generate_strategy_reference(ground_truth, game_name)
        elif task_type == 'identification':
            return f"This is the game {game_name}."
        else:
            return ""

    def _generate_visual_reference(self, ground_truth: Dict[str, Any]) -> str:
        """Generate visual description from ground truth objects."""
        objects = self._extract_objects(ground_truth)

        if not objects:
            return "No objects detected in the frame."

        # Use pre-generated description if available
        # Handle new format (dict with qualitative/quantitative) or old format (string)
        ref_answers = ground_truth.get('reference_answers', {}).get('visual')
        if isinstance(ref_answers, dict):
            # New format - use both qualitative and quantitative for semantic matching
            generated_desc = ref_answers.get('qualitative', '') + " " + ref_answers.get('quantitative', '')
        elif ref_answers:
            # Old format - string
            generated_desc = ref_answers
        else:
            # Fallback to old generated_descriptions
            generated_desc = ground_truth.get('generated_descriptions', {}).get('visual')

        if generated_desc:
            return generated_desc

        # Generate description
        descriptions = []
        for obj in objects:
            category = obj.get('category', 'Unknown')
            position = obj.get('position', [0, 0])
            size = obj.get('size', [0, 0])

            desc = f"{category}"
            if position:
                desc += f" at position ({position[0]}, {position[1]})"
            if size and size[0] > 0:
                desc += f" with size ({size[0]}, {size[1]})"

            descriptions.append(desc)

        return "The frame contains: " + ", ".join(descriptions) + "."

    def _generate_spatial_reference(self, ground_truth: Dict[str, Any]) -> str:
        """Generate spatial description from ground truth."""
        # Use pre-generated description if available
        # Handle new format (dict with qualitative/quantitative) or old format (string)
        ref_answers = ground_truth.get('reference_answers', {}).get('spatial')
        if isinstance(ref_answers, dict):
            # New format - use both for semantic matching
            generated_desc = ref_answers.get('qualitative', '') + " " + ref_answers.get('quantitative', '')
        elif ref_answers:
            # Old format - string
            generated_desc = ref_answers
        else:
            # Fallback to old generated_descriptions
            generated_desc = ground_truth.get('generated_descriptions', {}).get('spatial')

        if generated_desc:
            return generated_desc

        # Generate using helper function
        return generate_spatial_description(ground_truth)

    def _generate_strategy_reference(
        self,
        ground_truth: Dict[str, Any],
        game_name: str
    ) -> str:
        """Generate strategy description (generic)."""
        action_space = self._extract_action_space(ground_truth)
        actions = action_space.get('actions', [])

        if actions:
            return f"The player should take an action to optimize game performance. Valid actions include: {', '.join(actions)}."
        else:
            return "The player should take appropriate action based on the current game state."

    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate cosine similarity between two texts.

        Args:
            text1: First text
            text2: Second text

        Returns:
            Similarity score [0, 1]
        """
        if not self.model:
            return 0.5

        # Encode texts
        emb1 = self.model.encode(text1, convert_to_numpy=True)
        emb2 = self.model.encode(text2, convert_to_numpy=True)

        # Calculate cosine similarity
        similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))

        # Normalize to [0, 1]
        # Cosine similarity ranges from -1 to 1, we map to [0, 1]
        normalized_similarity = (similarity + 1.0) / 2.0

        return float(normalized_similarity)

    def batch_score(
        self,
        responses: list,
        task_types: list,
        ground_truths: list,
        game_names: list
    ) -> list:
        """
        Batch scoring for efficiency (can encode multiple texts at once).

        Args:
            responses: List of VLM responses
            task_types: List of task types
            ground_truths: List of ground truth dicts
            game_names: List of game names

        Returns:
            List of ScoringResults
        """
        if not self.available:
            return [
                ScoringResult(score=0.5, confidence=0.0, reasoning="Semantic scoring not available")
                for _ in responses
            ]

        results = []
        for resp, task, gt, game in zip(responses, task_types, ground_truths, game_names):
            result = self.score(resp, task, gt, game)
            results.append(result)

        return results
