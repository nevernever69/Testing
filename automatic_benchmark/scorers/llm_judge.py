"""
LLM-as-Judge scorer.
Uses GPT-4, Claude, or other LLMs to evaluate responses with nuanced judgment.
"""

import json
import re
from typing import Dict, Any, Optional
from .base_scorer import BaseScorer, ScoringResult
from ..config import LLM_JUDGE_PROVIDERS
from ..utils.prompts import create_llm_judge_prompt


class LLMJudge(BaseScorer):
    """
    LLM-as-Judge scorer.
    Uses powerful LLMs (GPT-4, Claude) to evaluate responses with human-like judgment.
    """

    def __init__(
        self,
        provider: str = 'openai',
        model: str = None,
        api_key: str = None,
        aws_region: str = None,
        **kwargs
    ):
        """
        Initialize LLM judge.

        Args:
            provider: 'openai', 'anthropic', or 'bedrock'
            model: Model name (defaults from config if not provided)
            api_key: API key (optional, reads from environment if not provided)
            aws_region: AWS region for Bedrock (optional, defaults to us-east-1)
        """
        super().__init__(**kwargs)

        self.provider = provider
        self.model = model or LLM_JUDGE_PROVIDERS[provider]['model']
        self.api_key = api_key
        self.aws_region = aws_region or 'us-east-1'

        # Initialize client based on provider
        self.client = None
        self.available = False

        try:
            if provider == 'openai':
                self._init_openai()
            elif provider == 'anthropic':
                self._init_anthropic()
            elif provider == 'bedrock':
                self._init_bedrock()
            else:
                raise ValueError(f"Unknown provider: {provider}")

            self.available = True

        except Exception as e:
            print(f"Warning: Failed to initialize LLM judge ({provider}): {e}")
            print("LLM judge scoring will be disabled.")

    def _init_openai(self):
        """Initialize OpenAI client."""
        try:
            import openai
            if self.api_key:
                self.client = openai.OpenAI(api_key=self.api_key)
            else:
                self.client = openai.OpenAI()  # Uses OPENAI_API_KEY env var
        except ImportError:
            raise ImportError("openai package not installed. Install with: pip install openai")

    def _init_anthropic(self):
        """Initialize Anthropic client."""
        try:
            import anthropic
            if self.api_key:
                self.client = anthropic.Anthropic(api_key=self.api_key)
            else:
                self.client = anthropic.Anthropic()  # Uses ANTHROPIC_API_KEY env var
        except ImportError:
            raise ImportError("anthropic package not installed. Install with: pip install anthropic")

    def _init_bedrock(self):
        """Initialize AWS Bedrock client."""
        try:
            from aws_model import BedrockUnifiedClient
            self.client = BedrockUnifiedClient(region_name=self.aws_region)
        except ImportError:
            raise ImportError("BedrockUnifiedClient not available. Make sure aws_model.py is in the path.")

    def score(
        self,
        response: str,
        task_type: str,
        ground_truth: Dict[str, Any],
        game_name: str
    ) -> ScoringResult:
        """Score response using LLM judge."""
        self._validate_task_type(task_type)

        if not self.available:
            return ScoringResult(
                score=0.5,
                confidence=0.0,
                reasoning="LLM judge not available"
            )

        if not response or response.startswith("ERROR"):
            return ScoringResult(
                score=0.0,
                confidence=1.0,
                reasoning="No valid response provided"
            )

        # Create judge prompt
        prompt = create_llm_judge_prompt(response, task_type, ground_truth, game_name)

        # Call LLM
        try:
            if self.provider == 'openai':
                result = self._call_openai(prompt)
            elif self.provider == 'anthropic':
                result = self._call_anthropic(prompt)
            elif self.provider == 'bedrock':
                result = self._call_bedrock(prompt)
            else:
                raise ValueError(f"Unknown provider: {self.provider}")

            # Parse result
            score = float(result.get('score', 0.5))
            reasoning = result.get('reasoning', 'No reasoning provided')
            issues = result.get('identified_issues', [])
            strengths = result.get('strengths', [])
            score_breakdown = result.get('score_breakdown', {})

            return ScoringResult(
                score=self._clamp_score(score),
                confidence=0.95,  # LLM judge has high confidence
                reasoning=reasoning,
                details={
                    'identified_issues': issues,
                    'strengths': strengths,
                    'score_breakdown': score_breakdown,  # Detailed breakdown for manual verification
                    'llm_provider': self.provider,
                    'llm_model': self.model,
                    'judge_prompt': prompt,  # Save for logging
                    'judge_raw_response': result  # Save raw response
                }
            )

        except Exception as e:
            print(f"Warning: LLM judge call failed: {e}")
            return ScoringResult(
                score=0.5,
                confidence=0.0,
                reasoning=f"LLM judge error: {str(e)}"
            )

    def _call_openai(self, prompt: str) -> Dict[str, Any]:
        """Call OpenAI API."""
        config = LLM_JUDGE_PROVIDERS['openai']

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert evaluator for Vision-Language Model benchmarks. "
                              "Be precise, fair, and consistent in your evaluations."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=config['temperature'],
            max_tokens=config['max_tokens'],
            response_format={"type": "json_object"}
        )

        content = response.choices[0].message.content
        return json.loads(content)

    def _call_anthropic(self, prompt: str) -> Dict[str, Any]:
        """Call Anthropic API."""
        config = LLM_JUDGE_PROVIDERS['anthropic']

        response = self.client.messages.create(
            model=self.model,
            max_tokens=config['max_tokens'],
            temperature=config['temperature'],
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        content = response.content[0].text

        # Extract JSON from response (may be in markdown code block)
        json_match = re.search(r'```json\s*(.*?)\s*```', content, re.DOTALL)
        if json_match:
            content = json_match.group(1)
        elif '```' in content:
            # Try to extract any code block
            json_match = re.search(r'```\s*(.*?)\s*```', content, re.DOTALL)
            if json_match:
                content = json_match.group(1)

        return json.loads(content)

    def _call_bedrock(self, prompt: str) -> Dict[str, Any]:
        """Call AWS Bedrock API using BedrockUnifiedClient."""
        config = LLM_JUDGE_PROVIDERS['bedrock']

        # Use BedrockUnifiedClient's chat_completion method
        response = self.client.chat_completion(
            model=self.model,
            messages=[{
                "role": "user",
                "content": [{"text": prompt}]
            }],
            temperature=config['temperature'],
            max_tokens=config['max_tokens']
        )

        content = response['choices'][0]['message']['content']

        # Extract JSON from response
        json_match = re.search(r'```json\s*(.*?)\s*```', content, re.DOTALL)
        if json_match:
            content = json_match.group(1)
        elif '```' in content:
            json_match = re.search(r'```\s*(.*?)\s*```', content, re.DOTALL)
            if json_match:
                content = json_match.group(1)

        return json.loads(content)

    def estimate_cost(self, num_evaluations: int) -> float:
        """
        Estimate cost for a number of evaluations.

        Args:
            num_evaluations: Number of evaluations to estimate

        Returns:
            Estimated cost in USD
        """
        # Rough estimates (may vary)
        cost_per_call = {
            'openai': 0.03,  # GPT-4 turbo
            'anthropic': 0.02,  # Claude 3.5 Sonnet
            'bedrock': 0.02  # Claude on Bedrock
        }

        return num_evaluations * cost_per_call.get(self.provider, 0.02)
