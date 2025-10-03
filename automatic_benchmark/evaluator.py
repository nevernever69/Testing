"""
Automatic evaluator that combines multiple scoring methods.
Orchestrates rule-based, semantic, and LLM-judge scorers intelligently.
"""

import numpy as np
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict

from .scorers import RuleBasedScorer, SemanticScorer, LLMJudge, ScoringResult
from .scorers.two_tier_scorer import TwoTierScorer
from .config import (
    SCORE_WEIGHTS,
    LLM_JUDGE_THRESHOLD,
    LLM_JUDGE_BORDERLINE_MIN,
    LLM_JUDGE_BORDERLINE_MAX,
    LLM_JUDGE_RANDOM_SAMPLE_RATE
)


@dataclass
class EvaluationResult:
    """Complete evaluation result combining all scoring methods."""
    final_score: float
    confidence: float
    reasoning: str
    rule_based_result: Optional[Dict[str, Any]] = None
    semantic_result: Optional[Dict[str, Any]] = None
    llm_judge_result: Optional[Dict[str, Any]] = None
    two_tier_result: Optional[Dict[str, Any]] = None
    combination_method: str = 'weighted_average'


class AutomaticEvaluator:
    """
    Automatic evaluator combining three scoring tiers:
    1. Rule-based (fast, deterministic)
    2. Semantic similarity (embedding-based)
    3. LLM-as-judge (nuanced, expensive)

    Intelligently decides when to use LLM judge based on agreement between
    rule-based and semantic scorers.
    """

    def __init__(
        self,
        use_semantic: bool = True,
        use_llm_judge: bool = True,
        llm_provider: str = 'openai',
        llm_model: str = None,
        llm_api_key: str = None,
        aws_region: str = None,
        semantic_model: str = None,
        force_llm_judge: bool = False,
        llm_judge_only: bool = False
    ):
        """
        Initialize automatic evaluator.

        Args:
            use_semantic: Enable semantic similarity scoring
            use_llm_judge: Enable LLM-as-judge scoring
            llm_provider: LLM provider ('openai', 'anthropic', 'bedrock')
            llm_model: LLM model name (optional, uses default if not specified)
            llm_api_key: API key for LLM (optional, reads from env if not specified)
            aws_region: AWS region for Bedrock (optional, defaults to us-east-1)
            semantic_model: Sentence transformer model name (optional)
            force_llm_judge: Always use LLM judge (for validation/testing)
            llm_judge_only: Use ONLY LLM judge, disable rule-based and semantic (LLM score = final score)
        """
        self.use_semantic = use_semantic
        self.use_llm_judge = use_llm_judge
        self.force_llm_judge = force_llm_judge
        self.llm_judge_only = llm_judge_only

        # If llm_judge_only is enabled, force LLM judge on and disable others
        if llm_judge_only:
            self.use_llm_judge = True
            self.force_llm_judge = True
            self.use_semantic = False

        # Initialize scorers
        self.rule_based_scorer = RuleBasedScorer()
        self.two_tier_scorer = TwoTierScorer()

        if self.use_semantic:
            if semantic_model:
                self.semantic_scorer = SemanticScorer(model_name=semantic_model)
            else:
                self.semantic_scorer = SemanticScorer()  # Use default model
        else:
            self.semantic_scorer = None

        if self.use_llm_judge:
            self.llm_judge = LLMJudge(
                provider=llm_provider,
                model=llm_model,
                api_key=llm_api_key,
                aws_region=aws_region
            )
        else:
            self.llm_judge = None

        # Statistics
        self.stats = {
            'total_evaluations': 0,
            'rule_based_calls': 0,
            'semantic_calls': 0,
            'llm_judge_calls': 0,
            'llm_judge_cost_estimate': 0.0
        }

    def evaluate(
        self,
        response: str,
        task_type: str,
        ground_truth: Dict[str, Any],
        game_name: str
    ) -> EvaluationResult:
        """
        Evaluate a VLM response using appropriate scoring methods.

        Args:
            response: VLM response text
            task_type: 'visual', 'spatial', 'strategy', or 'identification'
            ground_truth: Ground truth data from OCAtari
            game_name: Name of the game

        Returns:
            EvaluationResult with combined score and details
        """
        self.stats['total_evaluations'] += 1

        # LLM Judge Only Mode: Skip all other scorers
        if self.llm_judge_only:
            if self.llm_judge and self.llm_judge.available:
                llm_result = self.llm_judge.score(response, task_type, ground_truth, game_name)
                self.stats['llm_judge_calls'] += 1
                self.stats['llm_judge_cost_estimate'] += 0.03

                return EvaluationResult(
                    final_score=llm_result.score,
                    confidence=llm_result.confidence,
                    reasoning=llm_result.reasoning,
                    rule_based_result=None,
                    semantic_result=None,
                    llm_judge_result=llm_result.__dict__,
                    two_tier_result=None,
                    combination_method='llm_judge_only'
                )
            else:
                # Fallback if LLM judge not available
                return EvaluationResult(
                    final_score=0.5,
                    confidence=0.0,
                    reasoning="LLM judge only mode enabled but LLM judge not available",
                    combination_method='error'
                )

        # Normal multi-scorer mode
        # Tier 0: Two-tier object evaluation (always run for visual/spatial tasks)
        two_tier_result = None
        if task_type in ['visual', 'spatial']:
            two_tier_result = self.two_tier_scorer.evaluate(
                response, ground_truth, game_name, task_type
            )

        # Tier 1: Rule-based scoring (always run - fast and free)
        rule_result = self.rule_based_scorer.score(response, task_type, ground_truth, game_name)
        self.stats['rule_based_calls'] += 1

        # Tier 2: Semantic scoring (if enabled)
        semantic_result = None
        if self.use_semantic and self.semantic_scorer and self.semantic_scorer.available:
            semantic_result = self.semantic_scorer.score(response, task_type, ground_truth, game_name)
            self.stats['semantic_calls'] += 1

        # Tier 3: LLM judge (selective use)
        llm_result = None
        should_use_llm = self._should_use_llm_judge(
            rule_result.score,
            semantic_result.score if semantic_result else None
        )

        if self.use_llm_judge and self.llm_judge and self.llm_judge.available and should_use_llm:
            llm_result = self.llm_judge.score(response, task_type, ground_truth, game_name)
            self.stats['llm_judge_calls'] += 1
            # Update cost estimate
            if self.llm_judge.provider in ['openai', 'anthropic', 'bedrock']:
                self.stats['llm_judge_cost_estimate'] += 0.03  # Rough estimate

        # Combine scores
        final_score, confidence, reasoning, method = self._combine_scores(
            rule_result,
            semantic_result,
            llm_result
        )

        return EvaluationResult(
            final_score=final_score,
            confidence=confidence,
            reasoning=reasoning,
            rule_based_result=asdict(rule_result),
            semantic_result=asdict(semantic_result) if semantic_result else None,
            llm_judge_result=asdict(llm_result) if llm_result else None,
            two_tier_result=two_tier_result,
            combination_method=method
        )

    def _should_use_llm_judge(
        self,
        rule_score: float,
        semantic_score: Optional[float]
    ) -> bool:
        """
        Decide whether to use LLM judge based on score agreement.

        Use LLM judge if:
        1. Forced (for validation)
        2. Scores disagree significantly (> threshold)
        3. Score is borderline (0.4-0.6 range)
        4. Random sampling (10% of cases)
        """
        if self.force_llm_judge:
            return True

        if semantic_score is None:
            # No semantic score, use LLM for borderline or random sampling
            if LLM_JUDGE_BORDERLINE_MIN <= rule_score <= LLM_JUDGE_BORDERLINE_MAX:
                return True
            return np.random.random() < LLM_JUDGE_RANDOM_SAMPLE_RATE

        # Check disagreement between rule and semantic
        disagreement = abs(rule_score - semantic_score)
        if disagreement > LLM_JUDGE_THRESHOLD:
            return True

        # Check if either score is borderline
        if (LLM_JUDGE_BORDERLINE_MIN <= rule_score <= LLM_JUDGE_BORDERLINE_MAX or
            LLM_JUDGE_BORDERLINE_MIN <= semantic_score <= LLM_JUDGE_BORDERLINE_MAX):
            return True

        # Random sampling for validation
        return np.random.random() < LLM_JUDGE_RANDOM_SAMPLE_RATE

    def _combine_scores(
        self,
        rule_result: ScoringResult,
        semantic_result: Optional[ScoringResult],
        llm_result: Optional[ScoringResult]
    ) -> tuple:
        """
        Combine scores from multiple scorers using intelligent weighting.

        Returns:
            (final_score, confidence, reasoning, method)
        """
        scores = []
        weights = []
        scorers_used = []

        # Rule-based (always available)
        if rule_result:
            scores.append(rule_result.score)
            weights.append(SCORE_WEIGHTS['rule_based'])
            scorers_used.append('rule_based')

        # Semantic (if available)
        if semantic_result:
            scores.append(semantic_result.score)
            weights.append(SCORE_WEIGHTS['semantic'])
            scorers_used.append('semantic')

        # LLM judge (if available)
        if llm_result:
            scores.append(llm_result.score)
            # If rule and semantic disagree, give LLM more weight
            if semantic_result and abs(rule_result.score - semantic_result.score) > LLM_JUDGE_THRESHOLD:
                weights.append(0.5)  # Higher weight for LLM in disagreement
                # Reduce other weights
                weights[0] = 0.2  # rule_based
                weights[1] = 0.3  # semantic
            else:
                weights.append(SCORE_WEIGHTS['llm_judge'])
            scorers_used.append('llm_judge')

        # Normalize weights
        total_weight = sum(weights)
        normalized_weights = [w / total_weight for w in weights]

        # Weighted average
        final_score = sum(s * w for s, w in zip(scores, normalized_weights))

        # Calculate confidence based on score agreement
        if len(scores) > 1:
            score_variance = np.var(scores)
            confidence = max(0.5, 1.0 - min(score_variance, 0.5))
        else:
            confidence = rule_result.confidence

        # Generate combined reasoning
        reasoning_parts = []
        for scorer, score in zip(scorers_used, scores):
            reasoning_parts.append(f"{scorer}={score:.2f}")

        reasoning = f"Combined: {', '.join(reasoning_parts)} → {final_score:.2f}"

        method = f"weighted_average_{'+'.join(scorers_used)}"

        return final_score, confidence, reasoning, method

    def get_statistics(self) -> Dict[str, Any]:
        """Get evaluation statistics."""
        stats = self.stats.copy()

        if stats['total_evaluations'] > 0:
            stats['llm_judge_usage_rate'] = stats['llm_judge_calls'] / stats['total_evaluations']
        else:
            stats['llm_judge_usage_rate'] = 0.0

        return stats

    def estimate_total_cost(self) -> float:
        """Estimate total cost so far."""
        return self.stats['llm_judge_cost_estimate']

    def reset_statistics(self):
        """Reset evaluation statistics."""
        self.stats = {
            'total_evaluations': 0,
            'rule_based_calls': 0,
            'semantic_calls': 0,
            'llm_judge_calls': 0,
            'llm_judge_cost_estimate': 0.0
        }
