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
**MULTI-PART VISUAL EVALUATION:**
The VLM was asked to provide THREE parts:
1. OBJECTS: List all objects present
2. COUNTS: How many of each object
3. PROPERTIES: Visual properties (colors, sizes, states)

**SCORING BREAKDOWN (Stricter):**

**Part 1: Objects (40%)** - Did they identify all objects?
- Identified all core objects: +0.40
- Missing 1 core object: -0.40 (STRICT - core objects are critical!)
- Hallucinated object (mentioned but doesn't exist): -0.50 per fake object
- Skipped this section entirely: -0.30

**Part 2: Counts (35%)** - Are counts accurate?
- All counts correct: +0.35
- Wrong count for an object type: -0.30 per wrong count
- No counts provided: -0.30
- Approximate counts (e.g., "~10" when 8): Accept if within ±20% for objects >10
- For critical objects (player, ball): Exact count required

**Part 3: Properties (25%)** - Did they describe visual details?
- Described properties (colors, sizes, states): +0.25
- No properties described: -0.25
- Partial properties: proportional credit

**CRITICAL PENALTIES (Applied to final score):**
- Incomplete answer (skipped entire section): -0.30 per section
- Only described UI without game objects: Score capped at 0.15
- Spatial descriptions: NO PENALTY (evaluated in spatial task)

**IMPORTANT:**
- DO NOT penalize for spatial descriptions in visual task
- DO NOT penalize for saying "projectile above player" - that's spatial, not visual
- DO NOT penalize for mentioning score displays, UI elements, lives counter
- Mentioning score/UI: Completely IGNORE (no penalty, no credit)
- DO NOT penalize for including coordinates (e.g., "at x=300, y=400") - just IGNORE them
- If VLM mentions coordinates, IGNORE them when evaluating visual reasoning
- FOCUS ONLY on: What objects? How many? What properties?

{game_specific_visual}
"""

    elif task_type == 'spatial':
        base_prompt += """
**MULTI-PART SPATIAL EVALUATION:**
The VLM was asked to provide FOUR parts:
1. ABSOLUTE POSITIONS: Where is each object located on screen?
2. RELATIVE POSITIONS: What is the position of each object relative to others?
3. DISTANCES: Which objects are close together? Which are far apart?
4. ALIGNMENT: Are any objects vertically or horizontally aligned?

**IMPORTANT - Vision+Symbol Mode:**
- If VLM provides specific coordinates/measurements, they have access to symbolic data
- DO NOT penalize for "unverifiable measurements" - they have the coordinates!
- DO NOT criticize spacing/distance precision - they calculated it from coordinates
- Focus on whether relationships are CORRECT, not precision of measurements

**SCORING BREAKDOWN (Stricter):**

**Part 1: Absolute Positions (25%)** - Did they describe where objects are on screen?
- Correctly described screen positions (top/middle/bottom, left/center/right): +0.25
- Partial positions described: proportional credit
- No absolute positions: -0.30
- Skipped this section entirely: -0.30

**Part 2: Relative Positions (35%)** - Did they describe object relationships?
- All key relationships correct: +0.35
- Each WRONG directional claim: -0.35 per error (STRICT)
  Examples of WRONG claims:
  * Says "left" when object is on right
  * Says "above" when object is below
  * Says "directly above player" when horizontally offset
  * Says "near formation" when far away
- Missed key relationship (Player-Ball, Player-Enemy): -0.40 per missing pair
- No relative positions: -0.35
- Vague but correct ("near each other"): acceptable, no penalty

**Part 3: Distances (20%)** - Did they assess near/far relationships?
- Correct distance assessments (near/far/between): +0.20
- Wrong distance assessment ("near" when far): -0.25 per error
- No distance assessment: -0.25
- Partial distance descriptions: proportional credit

**Part 4: Alignment (20%)** - Did they identify aligned objects?
- Identified vertical/horizontal alignments: +0.20
- Wrong alignment claim: -0.15 per error
- No alignment described: No penalty (optional detail)
- Partial alignment: proportional credit

**CRITICAL PENALTIES (Applied to final score):**
- Incomplete answer (skipped entire section): -0.30 per section
- ONLY described UI/scores without game objects: Score capped at 0.15
- Talking about colors/appearance instead of spatial relationships: -0.25

**IMPORTANT - DO NOT penalize for mentioning UI elements:**
- Mentioning score displays, lives, UI elements: Completely IGNORE (no penalty, no credit)
- Including "scores at top" or "lives at bottom" is PERFECTLY FINE
- Only penalize if they ONLY described UI and completely ignored game objects
- Example: "Paddles on left and right, scores at top" = PERFECTLY ACCEPTABLE

**CRITICAL - What counts as answering the task:**
✅ GOOD: "Ball is at top-center (absolute), to the right of left paddle (relative), close to left paddle (distance), vertically aligned with left paddle (alignment)"
✅ GOOD: "Player paddle on right side, ball in middle, enemy on left. Ball closer to player than enemy."
✅ ACCEPTABLE: "Paddles on left and right, ball in middle, scores at top" (mentions UI but covers game objects)
❌ BAD: "There's a number 20 in orange, and a green vertical line" (ONLY describes UI)
❌ BAD: "I see colorful elements arranged across the frame" (no object relationships)

**This is a SPATIAL RELATIONSHIP task - judge ability to describe positions and relationships.**

{game_specific_spatial}
"""

    elif task_type == 'strategy':
        base_prompt += """
**MULTI-PART STRATEGY EVALUATION:**
The VLM was asked to provide THREE parts:
1. SITUATION: What is the current game situation? Any threats or opportunities?
2. ACTION: What specific action should the player take **NEXT**?
3. JUSTIFICATION: Why is this action optimal? What are you trying to achieve?

**CRITICAL - The prompt asks for THE NEXT ACTION, not a comprehensive strategy:**
- If VLM says "Move left", that is a COMPLETE answer to "what action next?"
- DO NOT penalize for not mentioning other actions (shooting, jumping, etc.)
- DO NOT expect a multi-step plan - just the immediate next action
- One specific action = CORRECT. Multiple actions = bonus but not required.

**SCORING BREAKDOWN (Stricter):**

**Part 1: Situation Analysis (30%)** - Did they analyze the CURRENT game state?
- Analyzed positions/threats/opportunities correctly: +0.30
- Partial analysis (mentioned some elements): +0.15
- Generic description without analysis: +0.05 maximum
- No situation analysis: -0.30
- Skipped this section entirely: -0.30

**Part 2: Action (40%)** - Did they commit to ONE SPECIFIC action?
- Specific optimal action ("MOVE UP", "MOVE LEFT", "FIRE"): +0.40
- Specific suboptimal but valid action: +0.15
- Vague/hedging ("up or down", "align paddle"): +0.05 maximum (VERY LOW)
- Invalid action (impossible for this game): -0.60 (AUTO-FAIL)
- No action given: -0.40

**IMPORTANT - ONE action is sufficient:**
✅ "Move left" = COMPLETE answer (0.40 points if optimal)
✅ "Move left and prepare to fire" = COMPLETE answer (bonus for detail)
❌ DO NOT criticize "only mentions movement" - movement IS an action!
❌ DO NOT expect "should also mention shooting" - that's a FUTURE action, not NEXT action

**Part 3: Justification (30%)** - Did they explain WHY this action is optimal?
- Clear justification tied to game state: +0.30
- Generic justification ("to win", "to score"): +0.10
- No justification: -0.35
- Skipped this section entirely: -0.35

**CRITICAL PENALTIES (Applied to final score):**
- Invalid action for this game: -0.60 (Examples: "jump" in Pong, "move up" in Breakout)
- Hedging with "or" (up OR down, fire OR wait): -0.40
- Incomplete answer (skipped entire section): -0.35 per section
- Only described objects without strategy: Score capped at 0.15

**IMPORTANT - Game/object identification worth almost nothing:**
- Identifying "this is Pong" or "I see paddle/ball" adds MAX +0.05
- Strategy task tests STRATEGIC REASONING, not object detection
- Saying "I see a paddle and ball" without strategic advice: 0.15 maximum

**IMPORTANT - DO NOT penalize for scope:**
- Mentioning barriers/shields is BONUS, not required
- Mentioning future actions is BONUS, not required
- Mentioning score displays/UI: Completely ignore (no penalty, no credit)
- FOCUS ONLY on: Did they answer situation/action/justification for the NEXT move?

**IMPORTANT:**
- DO NOT penalize for verbosity or response length
- DO NOT criticize for "not mentioning X" unless X was explicitly required
- FOCUS ONLY on: situation analysis quality, action specificity, justification clarity

**Examples:**
✅ EXCELLENT (0.9+): "SITUATION: Ball is above paddle. ACTION: Move up. JUSTIFICATION: To intercept ball before it passes" (complete 3-part answer)
✅ EXCELLENT (0.9+): "SITUATION: 36 invaders, projectile at x=310. ACTION: Move left. JUSTIFICATION: Avoid projectile trajectory" (complete answer)
✅ GOOD (0.8+): "Ball is above the paddle. Move up to intercept it." (clear situation + action + implicit justification)
✅ ACCEPTABLE (0.6): "Move paddle up. Ball is coming toward paddle." (some analysis, specific action)
❌ VAGUE (0.2): "Move paddle up or down to align with ball" (hedging with "or")
❌ DESCRIPTION ONLY (0.15): "I see Pong with paddle and ball, need to align them" (no specific action)
❌ INVALID (0.0): "Jump over the ball" (invalid action for Pong)

**This is a STRATEGY task - judge STRATEGIC THINKING, not object recognition.**

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

**CRITICAL - For Visual Task:**
- If VLM includes coordinates in visual task, DO NOT list as an issue
- DO NOT say "Included coordinate data which wasn't requested"
- Coordinates should be IGNORED, not penalized

Return ONLY a JSON object with this exact format:
{{
  "score": <float between 0.0 and 1.0>,
  "reasoning": "<2-3 sentence explanation focusing on correctness, not precision>",
  "identified_issues": ["<specific issue1>", "<specific issue2>"],
  "strengths": ["<specific strength1>", "<specific strength2>"]
}}

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
