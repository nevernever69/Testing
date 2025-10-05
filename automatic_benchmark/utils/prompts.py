"""
Standard prompts for benchmark tasks.
Based on Atari-GPT paper evaluation framework.
"""

from typing import Dict
from ..config import TASK_PROMPTS
from .game_specific_prompts import (
    get_game_specific_visual_criteria,
    get_game_specific_spatial_criteria,
    get_game_specific_strategy_criteria
)


def get_task_prompt(task_type: str) -> str:
    """
    Get the standard prompt for a task type.

    Args:
        task_type: One of 'visual', 'spatial', 'strategy', 'identification'

    Returns:
        Prompt string

    Raises:
        ValueError: If task_type is not recognized
    """
    if task_type not in TASK_PROMPTS:
        raise ValueError(f"Unknown task type: {task_type}. "
                        f"Must be one of {list(TASK_PROMPTS.keys())}")

    return TASK_PROMPTS[task_type]


def get_all_prompts() -> Dict[str, str]:
    """Get all task prompts as a dictionary."""
    return TASK_PROMPTS.copy()


def _create_compact_object_summary(objects: list) -> str:
    """Create a compact summary of objects by category and tier."""
    from collections import Counter

    if not objects:
        return "No objects detected"

    # Count by category
    category_counts = Counter(obj['category'] for obj in objects)

    # Count by tier
    tier_counts = Counter(obj.get('tier', 'unknown') for obj in objects)

    # Group by category and tier
    summary_lines = []
    summary_lines.append(f"Total: {len(objects)} objects")
    summary_lines.append(f"  - Core: {tier_counts.get('core', 0)} objects (critical for gameplay)")
    summary_lines.append(f"  - Secondary: {tier_counts.get('secondary', 0)} objects (bonus context)")
    if tier_counts.get('unknown', 0) > 0:
        summary_lines.append(f"  - Unknown: {tier_counts.get('unknown', 0)} objects")
    summary_lines.append("")
    summary_lines.append("Breakdown by category:")

    for category, count in sorted(category_counts.items(), key=lambda x: -x[1]):
        # Get tier for this category
        objs_of_category = [obj for obj in objects if obj['category'] == category]
        tier_for_category = Counter(obj.get('tier', 'unknown') for obj in objs_of_category).most_common(1)[0][0]
        summary_lines.append(f"  - {category}: {count} ({tier_for_category})")

    return "\n".join(summary_lines)


def create_llm_judge_prompt(
    response: str,
    task_type: str,
    ground_truth: Dict,
    game_name: str
) -> str:
    """
    Create a prompt for LLM-as-judge evaluation.

    Args:
        response: VLM response to evaluate
        task_type: Type of task being evaluated
        ground_truth: Ground truth data from OCAtari
        game_name: Name of the game

    Returns:
        Formatted prompt for LLM judge
    """
    import json

    # Extract relevant ground truth data
    # Use scaled coordinates if available, otherwise fall back to original
    ocatari_data = ground_truth.get('ocatari_data_scaled', ground_truth.get('ocatari_data', {}))
    reference_answers = ground_truth.get('reference_answers', {}).get(task_type, {})

    # Handle both old format (string) and new format (dict with qualitative/quantitative)
    if isinstance(reference_answers, dict):
        ref_qualitative = reference_answers.get('qualitative', 'Not available')
    else:
        # Old format - use string directly
        ref_qualitative = reference_answers if reference_answers else "Not available"

    # Create compact object summary
    objects = ocatari_data.get('objects', [])
    object_summary = _create_compact_object_summary(objects)

    base_prompt = f"""You are evaluating a Vision-Language Model's response to a {task_type} task for the Atari game {game_name}.

GROUND TRUTH SUMMARY (from OCAtari detection on 1280x720 frame):
{object_summary}

NOTE: The VLM saw the game frame at 1280x720 resolution.

EXPECTED RESPONSE (acceptable format):
{ref_qualitative}

TASK PROMPT: {get_task_prompt(task_type)}

VLM RESPONSE TO EVALUATE:
{response}

**IMPORTANT EVALUATION PRINCIPLES:**
- Focus on CORRECTNESS and CONTENT QUALITY only
- DO NOT penalize for verbosity, word count, or response length
- DO NOT criticize responses for being "slightly verbose" or going over word limits
- DO NOT mention word count in your reasoning or identified issues
- Judge ONLY on accuracy, completeness, and correctness of information

EVALUATION CRITERIA:
"""

    if task_type == 'visual':
        base_prompt += """
**VISUAL TASK EVALUATION:**
The VLM was asked to provide THREE parts:
1. OBJECTS: List all objects present
2. COUNTS: How many of each object
3. PROPERTIES: Visual properties (colors, sizes, states)

**SCORING STRUCTURE (Total: 1.0 points max):**

**Strengths - Assign points within these ranges:**
- **Object Identification (0.0-0.40):** How well did they identify game objects?
  - All core objects identified correctly: 0.35-0.40
  - Most core objects, some missing: 0.20-0.30
  - Few objects identified: 0.05-0.15
  - No meaningful identification: 0.0

- **Count Accuracy (0.0-0.35):** How accurate were the counts?
  - All counts correct (±20% for large groups): 0.30-0.35
  - Most counts correct, 1-2 errors: 0.15-0.25
  - Many count errors: 0.05-0.10
  - No counts or all wrong: 0.0

- **Property Description (0.0-0.25):** Did they describe visual properties?
  - Detailed properties (colors, sizes, states): 0.20-0.25
  - Some properties mentioned: 0.10-0.15
  - Minimal property description: 0.05
  - No properties: 0.0

**Penalties - Deduct points within these ranges:**
- **Missing Core Object:** -0.15 to -0.25 per missing core gameplay object (paddle, ball, player)
- **Hallucinated Object:** -0.30 to -0.50 per object mentioned that doesn't exist
- **Wrong Count (Critical Objects):** -0.15 to -0.30 per wrong count for player/ball/paddle
- **Completely Skipped Section:** -0.20 to -0.30 per section (objects/counts/properties)

**What to IGNORE (not issues, not strengths):**
- Spatial descriptions ("above", "below") - different task
- UI elements, score displays, lives counters - irrelevant
- **Coordinates or measurements ("at x=500", "200 pixels wide")** - ACCEPTABLE, never penalize
- **Quantitative data** - Vision+Symbol has coordinates, using them is CORRECT
- Response length or verbosity

{game_specific_visual}
"""

    elif task_type == 'spatial':
        base_prompt += """
**SPATIAL TASK EVALUATION:**
The VLM was asked to provide FOUR parts:
1. ABSOLUTE POSITIONS: Where is each object located on screen?
2. RELATIVE POSITIONS: What is the position of each object relative to others?
3. DISTANCES: Which objects are close together? Which are far apart?
4. ALIGNMENT: Are any objects vertically or horizontally aligned?

**SCORING STRUCTURE (Total: 1.0 points max):**

**Strengths - Assign points within these ranges:**
- **Absolute Positioning (0.0-0.25):** Did they describe where objects are on screen?
  - All objects positioned correctly (top/middle/bottom, left/center/right): 0.20-0.25
  - Most objects positioned correctly: 0.12-0.18
  - Some positioning mentioned: 0.05-0.10
  - No absolute positioning: 0.0

- **Relative Positioning (0.0-0.35):** Did they describe object relationships correctly?
  - All key relationships correct: 0.30-0.35
  - Most relationships correct: 0.18-0.28
  - Some relationships correct: 0.08-0.15
  - No relationships or all wrong: 0.0

- **Distance Assessment (0.0-0.20):** Did they assess near/far correctly?
  - Accurate distance descriptions: 0.15-0.20
  - Some distance awareness: 0.08-0.12
  - Minimal distance info: 0.03-0.05
  - No distance assessment: 0.0

- **Alignment (0.0-0.20):** Did they identify aligned objects?
  - Identified alignments correctly: 0.15-0.20
  - Some alignment mentioned: 0.08-0.12
  - Minimal/no alignment: 0.0-0.05

**Penalties - Deduct points within these ranges:**
- **Wrong Directional Claim:** -0.20 to -0.35 per major error (saying "left" when "right", "above" when "below")
- **Missed Key Relationship:** -0.15 to -0.25 per missing critical relationship (player-ball, player-enemy)
- **Wrong Distance Assessment:** -0.10 to -0.20 per error (saying "near" when far apart)
- **Completely Skipped Section:** -0.15 to -0.25 per section

**IMPORTANT - Vision+Symbol Mode:**
- If VLM provides coordinates/measurements, they have symbolic data access
- Focus on whether relationships are CORRECT, not precision
- **NEVER penalize for mentioning pixel distances or coordinates** - this is CORRECT usage

**What to IGNORE (not issues, not strengths):**
- UI elements unless they ONLY described UI
- Colors/appearance - different task
- Response length or verbosity
- **Coordinates/measurements ("separated by 200 pixels")** - ACCEPTABLE, never an issue
- **Quantitative spatial data** - Vision+Symbol has coordinates, using them is CORRECT

{game_specific_spatial}
"""

    elif task_type == 'strategy':
        base_prompt += """
**STRATEGY TASK EVALUATION:**
The VLM was asked to provide THREE parts:
1. SITUATION: What is the current game situation? Any threats or opportunities?
2. ACTION: What specific action should the player take **NEXT**?
3. JUSTIFICATION: Why is this action optimal? What are you trying to achieve?

**SCORING STRUCTURE (Total: 1.0 points max):**

**Strengths - Assign points within these ranges:**
- **Situation Analysis (0.0-0.30):** Did they analyze the current game state?
  - Detailed analysis of positions/threats/opportunities: 0.25-0.30
  - Good analysis with some elements: 0.15-0.22
  - Basic situation description: 0.08-0.12
  - No meaningful analysis: 0.0

- **Action (0.0-0.40):** Did they provide a specific, valid action?
  - Specific optimal action for this game state: 0.35-0.40
  - Specific valid but suboptimal action: 0.20-0.30
  - Valid action with unclear optimality: 0.12-0.18
  - Vague/hedging action ("move left OR right"): 0.03-0.08
  - No action or invalid action: 0.0

- **Justification (0.0-0.30):** Did they explain WHY this action is optimal?
  - Clear justification tied to game state: 0.25-0.30
  - Reasonable justification: 0.15-0.22
  - Generic justification ("to win", "to score"): 0.05-0.10
  - No justification: 0.0

**Penalties - Deduct points within these ranges:**
- **Invalid Action:** -0.50 to -0.70 (action impossible in this game: jump in Pong, move up in Breakout)
- **Hedging with "OR":** -0.25 to -0.40 (up OR down, fire OR wait - shows indecision)
- **Only Object Description:** -0.30 to -0.50 (described game but no strategic advice)
- **Completely Skipped Section:** -0.20 to -0.30 per section

**IMPORTANT - ONE next action is sufficient:**
- "Move left" = complete answer if optimal
- DO NOT expect multi-step plans
- DO NOT penalize for "only mentioning movement" - movement IS an action

**What to IGNORE (not issues, not strengths):**
- Game identification ("this is Pong") - minimal value
- UI elements, score displays - irrelevant
- Optional elements (shields, future actions) - bonus not required
- Response length or verbosity
- **Coordinates/measurements ("200 pixels", "at x=500")** - NEVER penalize for precision
- **Quantitative data** - Vision+Symbol pipeline has coordinates, using them is CORRECT

{game_specific_strategy}
"""

    elif task_type == 'identification':
        base_prompt += """
**STRICT REQUIREMENTS (to score >0.9):**
- MUST state game name in first 20 words
- MUST be exact match: "Pong", "Breakout", or "Space Invaders"
- Generic terms ("paddle game") are NOT acceptable

**Scoring:**
- Correct game name in first sentence: 1.0
- Correct but buried in explanation: 0.6
- Generic category only ("arcade game"): 0.2
- Wrong game: 0.0

**This task is fair for both pipelines:**
Both Vision-Only and Vision+Symbol have access to the same visual information to identify the game.

**Acceptable answers:**
- "This is Pong"
- "The game is Pong"
- "Pong - identified by the two paddles and ball"

**NOT acceptable:**
- "This is a paddle game"
- "This appears to be some kind of Pong-like game"
- "Based on analysis, this could be Pong"
"""

    base_prompt += """

**IMPORTANT: FAIR AND STRICT EVALUATION**
- Judge CORRECTNESS of content, not presence of coordinates
- Vision-Only models describe qualitatively (valid)
- Vision+Symbol models provide quantitative data (also valid)
- Both are acceptable IF the information is CORRECT
- Score LOW only for: wrong information, hallucinations, missing major elements
- Score HIGH when: correct objects/relationships/actions identified, regardless of precision level

**CRITICAL: NEVER PENALIZE COORDINATES OR MEASUREMENTS**
- If response mentions "200 pixels", "at x=500", "distance of 300", this is CORRECT
- Vision+Symbol pipeline HAS coordinate data - using it is GOOD, not an issue
- DO NOT list coordinate usage in "identified_issues"
- DO NOT deduct points for quantitative precision
- Only judge whether the CONTENT is factually correct

**CRITICAL - For Visual Task:**
- If VLM includes coordinates in visual task, DO NOT list as an issue
- DO NOT say "Included coordinate data which wasn't requested"
- Coordinates should be IGNORED, not penalized

**CRITICAL - NEVER PENALIZE UI MENTIONS:**
- Score displays, lives counters, timers, UI elements are NEVER issues
- DO NOT list them in "identified_issues" under ANY circumstances
- DO NOT deduct points for mentioning them
- COMPLETELY IGNORE any UI mentions when scoring
- Only penalize if VLM describes ONLY UI and ignores ALL game objects

**SCORING REQUIREMENTS:**
1. List STRENGTHS with point values - assign points within the specified ranges for each category
2. List ISSUES with penalty values - deduct points within the specified ranges
3. DO NOT list the same thing as both a strength and a penalty
4. Calculate the final score by summing all points (strengths + penalties)

**CRITICAL RULES:**
- Use PARTIAL CREDIT for count accuracy: if 2/3 counts correct, give ~0.20 points (not 0)
- Use PENALTIES only for critical errors: hallucinations, missing core objects, completely wrong counts
- DO NOT double-penalize: either give partial credit OR apply penalty, not both
- For minor count errors (off by 1): give partial credit ~0.20, no penalty
- For major count errors (off by >50%): give low partial credit ~0.10 AND small penalty
- Be consistent: strengths and penalties should not contradict each other

Return ONLY a JSON object with this exact format:
{{
  "score": <float between 0.0 and 1.0>,
  "reasoning": "<2-3 sentences explaining your evaluation>",
  "strengths": [
    "<Strength 1 description> [+0.XX points]",
    "<Strength 2 description> [+0.XX points]"
  ],
  "identified_issues": [
    "<Issue 1 description> [-0.XX points]",
    "<Issue 2 description> [-0.XX points]"
  ]
}}

**Example:**
{{
  "score": 0.0,
  "reasoning": "Response identified bricks correctly but had critical errors: wrong brick count and hallucinated a non-existent second paddle.",
  "strengths": [
    "Object Identification: Correctly identified bricks/BlockRows as core gameplay objects [+0.30 points]",
    "Property Description: Described brick colors (red, orange, yellow, green, blue) and dimensions [+0.20 points]"
  ],
  "identified_issues": [
    "Count Accuracy: Wrong BlockRow count - said 5 rows but ground truth shows 6 BlockRows [-0.25 points]",
    "Hallucinated Object: Claimed 2 paddles exist when ground truth shows only 1 Player paddle [-0.40 points]",
    "Missing Object: Did not identify the player paddle at all [-0.20 points]"
  ]
}}

**IMPORTANT:**
- Assign points WITHIN the specified ranges based on response quality
- Sum all strength and penalty points to get the final score
- Clamp final score between 0.0 and 1.0 (0.30 + 0.20 - 0.25 - 0.40 - 0.20 = -0.35, clamped to 0.0)
- DO NOT give high points in a category if you're also penalizing that same category

Be FAIR, strict about correctness, and lenient about precision. Base evaluation ONLY on ground truth data.
"""

    # Inject game-specific criteria
    # Use simple string replacement instead of .format() to avoid issues with JSON braces
    game_specific_visual = get_game_specific_visual_criteria(game_name)
    game_specific_spatial = get_game_specific_spatial_criteria(game_name)
    game_specific_strategy = get_game_specific_strategy_criteria(game_name)

    base_prompt = base_prompt.replace('{game_specific_visual}', game_specific_visual)
    base_prompt = base_prompt.replace('{game_specific_spatial}', game_specific_spatial)
    base_prompt = base_prompt.replace('{game_specific_strategy}', game_specific_strategy)

    return base_prompt
