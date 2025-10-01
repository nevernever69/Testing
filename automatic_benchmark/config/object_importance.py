"""
Object importance configuration for two-tier evaluation.

Core objects: Critical for gameplay, must be detected
Secondary objects: Useful but not essential (score displays, decorations)
"""

# Core objects by game (80% weight)
# These are essential for understanding game state and strategy
CORE_OBJECTS = {
    'Pong': [
        'Player',
        'Enemy',
        'Ball'
    ],

    'Breakout': [
        'Player',
        'Ball',
        'Brick'
    ],

    'SpaceInvaders': [
        'Player',
        'Alien',
        'Enemy',
        'Bullet',
        'Projectile'
    ],

    'Tennis': [
        'Player',
        'Enemy',
        'Ball'
    ],

    'MsPacman': [
        'Player',
        'Ghost',
        'Pellet',
        'PowerPellet'
    ]
}

# Secondary objects by game (20% weight)
# These provide additional context but aren't critical
# NOTE: OCAtari has limited object detection - many games don't detect score/lives
SECONDARY_OBJECTS = {
    'Pong': [
        # OCAtari doesn't detect score, walls, or center line in Pong
        # Only Player, Enemy, Ball are detected
    ],

    'Breakout': [
        # OCAtari may detect some secondary objects
        'Score',
        'Lives',
        'Wall'
    ],

    'SpaceInvaders': [
        'Score',
        'Lives',
        'Shield',
        'Barrier',
        'UFO',
        'Bunker'
    ],

    'Tennis': [
        'Score',
        'Net'
    ],

    'MsPacman': [
        'Score',
        'Lives',
        'Fruit',
        'Cherry',
        'Strawberry'
    ]
}


def get_object_tier(game: str, object_category: str) -> str:
    """
    Get the importance tier of an object with fuzzy matching.

    Args:
        game: Game name (e.g., 'Pong', 'Breakout', 'SpaceInvaders')
        object_category: Object category from OCAtari

    Returns:
        'core', 'secondary', or 'unknown'
    """
    # Normalize game name for 3 games only
    game_lower = game.lower()
    if 'pong' in game_lower:
        game = 'Pong'
    elif 'breakout' in game_lower:
        game = 'Breakout'
    elif 'space' in game_lower or 'invaders' in game_lower:
        game = 'SpaceInvaders'

    category_lower = object_category.lower()

    # Fuzzy matching rules for common variations
    category_mappings = {
        'brick': ['brick', 'block', 'blockrow', 'tile'],
        'player': ['player', 'paddle', 'ship'],
        'enemy': ['enemy', 'opponent', 'alien'],
        'bullet': ['bullet', 'projectile', 'shot', 'laser'],
        'score': ['score', 'points'],
        'life': ['life', 'lives']
    }

    # Check core objects with fuzzy matching
    core_list = CORE_OBJECTS.get(game, [])
    for core_obj in core_list:
        core_lower = core_obj.lower()

        # Direct match
        if core_lower in category_lower or category_lower in core_lower:
            return 'core'

        # Fuzzy match using mappings
        for canonical, variations in category_mappings.items():
            if canonical in core_lower:
                if any(var in category_lower for var in variations):
                    return 'core'

    # Check secondary objects with fuzzy matching
    secondary_list = SECONDARY_OBJECTS.get(game, [])
    for sec_obj in secondary_list:
        sec_lower = sec_obj.lower()

        # Direct match
        if sec_lower in category_lower or category_lower in sec_lower:
            return 'secondary'

        # Fuzzy match using mappings
        for canonical, variations in category_mappings.items():
            if canonical in sec_lower:
                if any(var in category_lower for var in variations):
                    return 'secondary'

    # Unknown - treat as secondary by default (bonus credit)
    return 'unknown'


def get_core_objects(game: str) -> list:
    """Get list of core objects for a game."""
    return CORE_OBJECTS.get(game.title(), [])


def get_secondary_objects(game: str) -> list:
    """Get list of secondary objects for a game."""
    return SECONDARY_OBJECTS.get(game.title(), [])


def get_all_objects(game: str) -> dict:
    """
    Get all objects categorized by tier.

    Returns:
        {
            'core': [...],
            'secondary': [...]
        }
    """
    return {
        'core': get_core_objects(game),
        'secondary': get_secondary_objects(game)
    }


def calculate_tiered_score(core_score: float, secondary_score: float,
                          core_weight: float = 0.7, secondary_weight: float = 0.3) -> float:
    """
    Calculate weighted score from core and secondary evaluations.

    Args:
        core_score: Score for core objects (0.0 to 1.0)
        secondary_score: Score for secondary objects (0.0 to 1.0)
        core_weight: Weight for core objects (default 0.7 = 70%)
        secondary_weight: Weight for secondary objects (default 0.3 = 30%)

    Returns:
        Weighted final score (0.0 to 1.0)
    """
    return (core_score * core_weight) + (secondary_score * secondary_weight)
