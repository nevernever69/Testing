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

# Task Prompts (Multi-Part for Comprehensive Evaluation)
TASK_PROMPTS = {
    'visual': """Analyze this game frame and provide:
1. OBJECTS: List all objects present in the frame
2. COUNTS: For each object type, state how many you see
3. PROPERTIES: Describe key visual properties (colors, sizes, states)

Format your answer clearly addressing all three parts.

INSTRUCTIONS:
- Identify what type of object each is (paddle, ball, blocks, score, etc.)
- Describe visual characteristics (colors, sizes, shapes, states)
- Describe WHERE each object is located using detailed qualitative descriptions:
  * Use combinations like: "upper left", "lower right", "middle right", "slightly below center"
  * Be precise: "just below the middle", "near the top-left corner", "in the upper-right portion"
  * Avoid simple "top/bottom/left/right" - be more nuanced
- Be specific about what you see in the frame""",

    'spatial': """Describe the spatial layout of this game frame:
1. ABSOLUTE POSITIONS: Where is each object located? (top/middle/bottom, left/center/right)
2. RELATIVE POSITIONS: What is the position of each object relative to others? (above/below, left/right)
3. DISTANCES: Which objects are close together? Which are far apart?
4. ALIGNMENT: Are any objects vertically or horizontally aligned?

Be specific and address all four aspects.

INSTRUCTIONS:
- For EACH PAIR of game objects (paddles, ball, enemies, blocks), describe:
  * Horizontal relationship: "far to the left of", "slightly left of", "directly left of", "to the right of"
  * Vertical relationship: "well above", "just above", "at same height as", "below"
  * Distance/proximity: "very close to", "moderately separated from", "far apart from"
- Compare ALL game object pairs systematically (ball vs paddle, paddle vs paddle, etc.)
- Express relationships in natural, nuanced language
- Stay focused on RELATIONSHIPS between objects (not individual object appearance)
- Be precise with directional and distance descriptions""",

    'strategy': """Analyze the game state and recommend an action:
1. SITUATION: What is the current game situation? Any threats or opportunities?
2. ACTION: What specific action should the player take next?
3. JUSTIFICATION: Why is this action optimal? What are you trying to achieve?

Provide a clear answer for each part.

INSTRUCTIONS:
- Based on object positions, what should the player do next?
- Which direction should paddles/ships move?
- Should the player shoot, move up, move down, or wait?
- Explain WHY this is the optimal move based on the game state
- Provide specific, actionable advice tied to the current situation""",

    'identification': """Identify the game name. Be specific.

INSTRUCTIONS:
- Based on the objects and layout, what game is this?
- What specific features identify this game?
- State the game name clearly in the first sentence
- Justify your answer with observable game elements"""
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

# Stricter Penalties for Multi-Part Evaluation
STRICTER_PENALTIES = {
    'visual': {
        'missing_core_object': -0.4,      # Missing a core gameplay object
        'wrong_count': -0.3,               # Per object type with wrong count
        'hallucinated_object': -0.5,      # Mentioned object that doesn't exist
        'missing_properties': -0.25,       # Didn't describe visual properties
        'incomplete_answer': -0.3          # Skipped entire section (objects/counts/properties)
    },
    'spatial': {
        'wrong_direction': -0.35,          # Said "left" when "right", "above" when "below"
        'wrong_distance': -0.25,           # Said "near" when "far" or vice versa
        'missed_relationship': -0.4,       # Completely missed key object relationship
        'no_absolute_positions': -0.3,     # Didn't describe where objects are
        'no_relative_positions': -0.35,    # Didn't describe object relationships
        'incomplete_answer': -0.3          # Skipped entire section
    },
    'strategy': {
        'invalid_action': -0.6,            # Suggested impossible action (auto-fail)
        'suboptimal_action': -0.3,         # Valid but wrong action
        'no_justification': -0.35,         # Didn't explain why
        'no_situation_analysis': -0.3,     # Didn't analyze game state
        'hedging': -0.4,                   # "Move up OR down" - indecisive
        'incomplete_answer': -0.35         # Skipped entire section
    }
}

# Multi-Part Scoring Weights
MULTIPART_WEIGHTS = {
    'visual': {
        'objects': 0.40,      # 40% - Identifying what's present
        'counts': 0.35,       # 35% - Counting accurately
        'properties': 0.25    # 25% - Describing visual details
    },
    'spatial': {
        'absolute_positions': 0.25,   # 25% - Where objects are on screen
        'relative_positions': 0.35,   # 35% - Relationships between objects
        'distances': 0.20,            # 20% - Near/far assessments
        'alignment': 0.20             # 20% - Aligned objects
    },
    'strategy': {
        'situation_analysis': 0.30,   # 30% - Understanding game state
        'action': 0.40,               # 40% - Correct action choice
        'justification': 0.30         # 30% - Reasoning quality
    }
}

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
