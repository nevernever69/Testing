"""
Configuration for Automatic Benchmark System
Centralizes all settings and constants.
"""

from dataclasses import dataclass
from typing import Dict, List, Any


# Benchmark Dataset Configuration
GAMES = ['Pong', 'Breakout', 'SpaceInvaders']
FRAMES_PER_GAME = 50
TOTAL_FRAMES = len(GAMES) * FRAMES_PER_GAME  # 150
RANDOM_SEED = 42

# Frame Selection Configuration
CANDIDATE_FRAMES_PER_GAME = 3000  # Generate this many candidates before sampling (more = better variety)
COMPLEXITY_CATEGORIES = ['easy', 'medium', 'hard']
COMPLEXITY_DISTRIBUTION = {
    'easy': 0.33,
    'medium': 0.34,
    'hard': 0.33
}

# Complexity Thresholds
# Note: These are adjusted to work with typical Atari games
# Pong has ~3 objects, Breakout ~10-30, SpaceInvaders ~20-50
COMPLEXITY_THRESHOLDS = {
    'object_count': {
        'easy': 5,     # <= 5 objects
        'medium': 15,  # 6-15 objects
        'hard': 15     # > 15 objects
    },
    'spatial_density': {
        'easy': 0.0005,   # Low density
        'medium': 0.001,  # Medium density
        'hard': 0.002     # High density
    }
}

# Task Prompts (from Atari-GPT paper)
TASK_PROMPTS = {
    'visual': "Identify all key elements in this image. Be specific. Use at most 100 words.",
    'spatial': "Where are the key elements located relative to each other? Be specific with respect to their position in the image. Use at most 100 words.",
    'strategy': "Describe the ideal next move if you were playing this game. Be specific. Use at most 100 words.",
    'identification': "Identify the game name. Be specific."
}

# Object Category Synonyms (for rule-based scoring)
OBJECT_SYNONYMS = {
    'Player': ['player', 'my paddle', 'right paddle', 'paddle on right', 'player paddle'],
    'Ball': ['ball', 'projectile', 'white ball', 'square', 'white square'],
    'Enemy': ['enemy', 'opponent', 'left paddle', 'ai', 'computer', 'opponent paddle'],
    'Brick': ['brick', 'block', 'tile'],
    'Alien': ['alien', 'invader', 'enemy ship'],
    'Ship': ['ship', 'player ship', 'spaceship'],
    'Bullet': ['bullet', 'shot', 'projectile', 'laser'],
    'Shield': ['shield', 'barrier', 'bunker', 'defense']
}

# Game-specific invalid actions (for strategy validation)
INVALID_ACTIONS = {
    'Pong': ['move left', 'move right', 'shoot', 'jump', 'fire left', 'fire right'],
    'Breakout': ['jump', 'move up', 'move down'],
    'SpaceInvaders': ['move up', 'move down', 'jump']
}

# Semantic Similarity Configuration
SEMANTIC_MODEL_NAME = 'all-MiniLM-L6-v2'
SEMANTIC_SIMILARITY_THRESHOLD = 0.7  # Above this is considered good match

# LLM Judge Configuration
LLM_JUDGE_PROVIDERS = {
    'openai': {
        'model': 'gpt-4-turbo-preview',
        'temperature': 0.0,
        'max_tokens': 1024
    },
    'anthropic': {
        'model': 'claude-3-5-sonnet-20241022',
        'temperature': 0.0,
        'max_tokens': 1024
    },
    'bedrock': {
        'model': 'claude-4-sonnet',
        'temperature': 0.0,
        'max_tokens': 1024
    }
}

# Score Combination Weights
SCORE_WEIGHTS = {
    'rule_based': 0.3,
    'semantic': 0.4,
    'llm_judge': 0.3
}

# When to use LLM judge
LLM_JUDGE_THRESHOLD = 0.2  # Use if rule and semantic disagree by more than this
LLM_JUDGE_BORDERLINE_MIN = 0.4  # Use for scores in borderline range
LLM_JUDGE_BORDERLINE_MAX = 0.6
LLM_JUDGE_RANDOM_SAMPLE_RATE = 0.1  # Use on 10% of cases for validation

# Statistical Analysis Configuration
SIGNIFICANCE_LEVEL = 0.05  # p-value threshold
CONFIDENCE_INTERVAL = 0.95

# Coordinate Validation
COORDINATE_TOLERANCE_PIXELS = 20  # Allow this much error in coordinate mentions

# Scoring Weights (within rule-based scorer)
RULE_BASED_WEIGHTS = {
    'visual': {
        'object_detection': 0.7,
        'false_positive_penalty': 0.5,
        'specificity_bonus': 0.3
    },
    'spatial': {
        'relative_position': 0.4,
        'coordinate_accuracy': 0.3,
        'distance_awareness': 0.3
    },
    'strategy': {
        'action_validity': 0.3,
        'action_optimality': 0.4,
        'justification': 0.3
    }
}


@dataclass
class BenchmarkConfig:
    """Configuration object for benchmark runs."""

    # Dataset
    games: List[str] = None
    frames_per_game: int = FRAMES_PER_GAME
    random_seed: int = RANDOM_SEED

    # Scoring
    use_semantic: bool = True
    use_llm_judge: bool = True
    llm_provider: str = 'openai'
    llm_model: str = None

    # Execution
    output_dir: str = './benchmark_results'
    save_detailed_logs: bool = True
    verbose: bool = True

    def __post_init__(self):
        if self.games is None:
            self.games = GAMES

        if self.llm_model is None and self.use_llm_judge:
            self.llm_model = LLM_JUDGE_PROVIDERS[self.llm_provider]['model']

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for serialization."""
        return {
            'games': self.games,
            'frames_per_game': self.frames_per_game,
            'random_seed': self.random_seed,
            'use_semantic': self.use_semantic,
            'use_llm_judge': self.use_llm_judge,
            'llm_provider': self.llm_provider,
            'llm_model': self.llm_model,
            'output_dir': self.output_dir
        }
