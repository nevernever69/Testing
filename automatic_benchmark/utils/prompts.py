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
**FAIR EVALUATION PRINCIPLE:**
This is a VISUAL IDENTIFICATION task, NOT a spatial relationship task.
Judge on: Did they identify the CORRECT objects? (not their spatial relationships)

**DO NOT penalize for:**
- Incorrect spatial descriptions (e.g., "projectile above player" when it's below)
- Vague position descriptions (e.g., "in upper area")
- Approximate positions
→ These are spatial issues, penalize them in SPATIAL task only!

**DO penalize for:**
- Hallucinating objects that don't exist
- Missing core gameplay objects
- Misidentifying object types

**Scoring:**
1. **Core Game Objects** (70%): Did they identify the MAIN gameplay objects?
   - Ball mentioned correctly: +0.25
   - Player paddle mentioned correctly: +0.25
   - Enemy paddle mentioned correctly: +0.25
   - (Core objects are critical for gameplay understanding)

2. **Secondary Objects** (10%): Bonus for noticing additional elements
   - Score displays mentioned: +0.05 (bonus, not required)
   - Other UI elements: +0.05 (bonus, not required)
   - Note: Mentioning score displays is fine, but shouldn't be the main focus

3. **Completeness** (20%): Did they find all CORE objects?
   - All core objects found: +0.2
   - 66% core objects found: +0.13
   - 33% core objects found: +0.07

4. **No Hallucinations** (deduction): -0.3 per fake object

**DO NOT deduct points for:**
- Not mentioning exact coordinates
- Imprecise color/size descriptions
- Using qualitative vs quantitative descriptions
- Mentioning score displays (they're visible in frame, perfectly fine to mention)
- Not mentioning secondary elements like scores/UI

**DO deduct points for:**
- Missing CORE game objects (paddle, ball, enemies): -0.25 per missing core object
- Hallucinated objects that don't exist: -0.3 per fake object
- ONLY mentioning score displays without identifying game objects: major penalty
- Completely wrong object identification: -0.4

**Example scoring:**
- "I see two paddles, a ball, and score displays showing 0-0": 0.95+ (perfect)
- "I see two paddles and a ball": 0.85+ (good, didn't mention scores but that's fine)
- "I see score displays showing 0-0": 0.1 (failed - only mentioned UI, not game objects)

{game_specific_visual}
"""

    elif task_type == 'spatial':
        base_prompt += """
**STRICT EVALUATION PRINCIPLE:**
Spatial task is about describing RELATIONSHIPS BETWEEN GAME OBJECTS.
Describing colors, visual artifacts, or score displays is NOT relevant.

**IMPORTANT - Vision+Symbol Mode:**
- If VLM provides specific coordinates/measurements, they have access to symbolic data
- DO NOT penalize for "unverifiable measurements" - they have the coordinates!
- DO NOT criticize spacing/distance precision - they calculated it from coordinates
- Focus on whether relationships are CORRECT, not precision of measurements

**Scoring (STRICT):**
1. **Core Object Relationships** (70%): Did they describe relationships between KEY objects?
   - For each correct object pair relationship (Player-Ball, Player-Enemy, Ball-Enemy):
     * Correct horizontal relationship (left/right): +0.17
     * Correct vertical relationship (above/below): +0.17
     * Correct distance assessment (near/far): +0.1
   - Maximum 3 pairs expected, total 0.7 possible

2. **Relationship Accuracy** (20%): Are the spatial statements CORRECT?
   - Each correct directional claim: maintains score
   - Each WRONG directional claim: -0.15 per error (fair penalty)
     Examples of WRONG claims:
     * Says "left" when object is on right
     * Says "above" when object is below
     * Says "directly above player" when horizontally offset
     * Says "near formation" when far away
   - Vague but correct ("near each other"): acceptable, no penalty
   - Note: Small positional errors are okay, only penalize clear mistakes

3. **Focus on Game Objects** (10%): Did they describe game object relationships?
   - Described core game object relationships: +0.1
   - Also mentioned score displays/UI elements: fine, NO penalty (they're visible in frame)
   - ONLY described score positions without game objects: 0.0 (failed task)

**IMPORTANT - DO NOT penalize for mentioning UI elements:**
- Mentioning score displays, lives, UI elements is PERFECTLY FINE
- Only penalize if they ONLY described UI and completely ignored game objects
- Including UI descriptions alongside game object relationships is acceptable

**CRITICAL - What counts as answering the task:**
✅ GOOD: "The ball is to the right of the left paddle and above it"
✅ GOOD: "Player paddle is far to the right, enemy paddle is on the left"
✅ GOOD: "Ball is between the two paddles. Score displays are at the top showing 0-0"
✅ ACCEPTABLE: "Ball positioned between paddles, closer to the right one"
❌ BAD: "There's a number 20 in orange, and a green vertical line" (ONLY describes UI)
❌ BAD: "I see colorful elements arranged across the frame" (no object relationships)

**DO NOT penalize for:**
- Briefly mentioning score displays (they ARE visible in the frame)
- Describing score positions along with game objects
- Example: "Paddles on left and right, ball in middle, scores at top" is perfectly fine

**DO penalize heavily for:**
- ONLY describing score displays/UI without game object relationships
- No comparison between game objects provided
- Talking about colors/appearance instead of spatial relationships

**Automatic very low score (< 0.2) if:**
- Response ONLY describes UI/scores without mentioning game object relationships
- No comparison between game objects provided
- Completely ignores spatial relationships task

**This is a SPATIAL RELATIONSHIP task - judge ability to compare object positions.**

{game_specific_spatial}
"""

    elif task_type == 'strategy':
        base_prompt += """
**STRICT EVALUATION PRINCIPLE:**
Strategy is about ANALYZING CURRENT STATE and giving SPECIFIC OPTIMAL ACTION.
Game/object identification is NOT the main task - strategic reasoning is.

**Scoring (STRICT):**
1. **Specific Action Given** (50%): Did they commit to a SPECIFIC direction/move?
   - Specific optimal action ("MOVE UP"): +0.5
   - Specific suboptimal action ("MOVE DOWN"): +0.2
   - Vague/hedging ("up or down", "align paddle"): +0.1 maximum
   - Invalid action: 0.0

2. **State Analysis** (40%): Did they analyze the CURRENT game state?
   - Analyzed positions to determine correct move: +0.4
   - Generic advice without state analysis: +0.05 maximum
   - No analysis: 0.0

3. **Action Optimality** (10%): Is the action correct for THIS situation?
   - Correct action for current state: +0.1
   - Wrong action: 0.0

**IMPORTANT - Game/object identification worth almost nothing:**
- Identifying "this is Pong" or "I see paddle/ball" adds MAX +0.05
- Strategy task tests STRATEGIC REASONING, not object detection
- Saying "I see a paddle and ball" without strategic advice: 0.1 maximum

**Examples:**
- "Move paddle up to intercept ball at y=126": 0.9+ (specific + analyzed)
- "Move up - ball is above paddle": 0.85+ (specific + some analysis)
- "Move paddle up or down to align": 0.2 (vague, no commitment)
- "I see Pong with paddle and ball, align them": 0.15 (just description)

**DO NOT give credit for:**
- Generic game advice that applies to any frame
- Hedging with "or" (up OR down, fire OR wait)
- Just identifying objects without strategic recommendation
- Vague advice like "position optimally", "be ready"

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
