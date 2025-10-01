"""
Game-specific evaluation criteria for LLM judge.
Each game has different object counts, mechanics, and strategic considerations.
"""

def get_game_specific_visual_criteria(game_name: str) -> str:
    """Get game-specific criteria for visual task evaluation."""

    game = game_name.lower()

    if 'pong' in game:
        return """
**PONG-SPECIFIC CRITERIA:**
- Expected objects: 2 paddles + 1 ball = **3 objects total**
- Core objects (critical): Player paddle, Enemy paddle, Ball
- Secondary objects: Score displays, center line (if VLM mentions them, that's fine)
- Simple game: Should be easy to identify all 3 core objects

**Scoring:**
- All 3 core objects identified: 0.75+ base score
- Missing 1 core object: 0.5
- Missing 2+ core objects: 0.25 or less
- Bonus for mentioning score displays: +0.05
"""

    elif 'breakout' in game:
        return """
**BREAKOUT-SPECIFIC CRITERIA:**
- Expected objects: 1 paddle + **6-8 BlockRows** + 0-1 ball = **7-9 objects total**
- IMPORTANT: OCAtari detects **BlockRows** (horizontal rows of bricks), NOT individual bricks
- Core objects (critical): Player paddle, BlockRows/bricks, Ball (if active)
- VLM might say "bricks" (correct) or "rows of bricks" (more accurate)
- Accept both "bricks" and "BlockRows" as correct

**Scoring:**
- Identified paddle + bricks/blocks: 0.6+ base score
- Identified ball (if present): +0.15
- Mentioned multiple brick rows/colors: +0.1 (shows detail)
- Missing paddle OR bricks entirely: 0.3 or less
- If VLM says "individual bricks" vs "rows": Both acceptable

**Common VLM responses:**
- Good: "Paddle at bottom, multiple rows of colored bricks at top, ball bouncing"
- Acceptable: "Paddle, bricks, ball"
- Bad: "Only score displays visible" (missed main objects)
"""

    elif 'space' in game or 'invader' in game:
        return """
**SPACE INVADERS-SPECIFIC CRITERIA:**
- Expected objects: 1 player ship + **20-55 aliens in formation** + 0-4 shields + 0-3 bullets
- Alien formation: Typically 11 columns × 5 rows = 55 aliens initially
- **Formation-based evaluation**: Aliens move as a grid, columns get destroyed
- This is the MOST COMPLEX game (many objects, dynamic formation)

**CRITICAL: Use Formation-Level Understanding**
Instead of counting individual aliens, evaluate on FORMATION AWARENESS:

✅ What to look for:
- "Alien formation" / "grid of aliens" / "rows and columns" (EXCELLENT - formation-aware)
- "Multiple rows of aliens" / "alien grid" (GOOD - structure understanding)
- "Approximately 30-40 aliens" / "many aliens" (ACCEPTABLE - reasonable estimate)
- "Left columns destroyed" / "formation narrowed" (EXCELLENT - destruction awareness)
- "Formation descended" / "aliens moved down" (GOOD - movement awareness)

❌ What NOT to require:
- Exact alien count (30 vs 35 vs 40 is fine - there are too many to count!)
- Individual alien positions (formation-level is better)
- Counting each alien separately

**Scoring:**
Core Objects (75%):
- Player ship identified: +0.25
- Alien formation/group identified: +0.35 (as a collective)
- Shields/barriers identified: +0.15

Formation Awareness Bonus (15%):
- Describes as formation/grid/rows: +0.05
- Notices destruction pattern: +0.05
- Describes formation movement: +0.05

Secondary (10%):
- Bullets/projectiles: +0.05
- Score displays: +0.05

**Scoring examples:**
- "Ship at bottom, alien formation in 4-5 rows with left column destroyed, 3 shields": 1.0 ✅ PERFECT
- "Player ship, grid of aliens (approx 30-35), shields": 0.95+ ✅ EXCELLENT
- "Spaceship, many aliens arranged in rows, protective barriers": 0.85+ ✅ GOOD
- "Ship and aliens": 0.60 (too vague but identifies main objects)
- "Only ship visible": 0.25 ❌ (missed alien formation)
- "Ship, 47 aliens, 3 shields": 0.95+ ✅ (impressive if close count!)
"""

    else:
        return ""

def get_game_specific_spatial_criteria(game_name: str) -> str:
    """Get game-specific criteria for spatial task evaluation."""

    game = game_name.lower()

    if 'pong' in game:
        return """
**PONG-SPECIFIC SPATIAL:**
- Key relationships: Player-Ball, Enemy-Ball, Player-Enemy
- Expected: 3 object pairs, simple layout
- Typical: "Ball between two paddles, closer to right/left"

**Scoring:**
- Describes all 3 relationships: 0.8+
- Describes 2 relationships: 0.6
- Describes 1 relationship: 0.3
"""

    elif 'breakout' in game:
        return """
**BREAKOUT-SPECIFIC SPATIAL:**
- Key relationships: Paddle-Ball, Paddle-Bricks, Ball-Bricks
- Typical layout: Bricks at top, paddle at bottom, ball between
- Accept "ball approaching paddle" vs "ball near top bricks"

**Scoring:**
- Describes paddle position relative to ball: +0.3
- Describes brick layout (top of screen): +0.25
- Describes ball trajectory/position: +0.25
- Missing paddle-ball relationship: -0.3
"""

    elif 'space' in game or 'invader' in game:
        return """
**SPACE INVADERS-SPECIFIC SPATIAL:**
- Key relationships: Ship-Aliens, Ship-Shields, Aliens-Shields, Alien formation
- **Many aliens** - don't expect individual positions for each
- Typical: "Aliens in formation at top, ship at bottom, shields between"

**Fair Evaluation:**
- Describes ship position (bottom): +0.2
- Describes alien formation/grid: +0.3 (excellent detail)
- Describes shields as protection between: +0.2
- Mentions aliens descending/advancing: +0.15 (strategic awareness)

**DO NOT penalize for:**
- Not describing position of every alien (there are 20-36!)
- Saying "aliens above ship" instead of each alien position
- Focusing on formation rather than individuals

**Scoring examples:**
- "Ship at bottom, grid of aliens at top, shields in middle as barriers": 0.85+
- "Player below alien formation, shields provide cover": 0.75+
- "Ship and aliens with some barriers": 0.50 (vague)
"""

    else:
        return ""

def get_game_specific_strategy_criteria(game_name: str) -> str:
    """Get game-specific criteria for strategy task evaluation."""

    game = game_name.lower()

    if 'pong' in game:
        return """
**PONG-SPECIFIC STRATEGY:**
- Valid actions: MOVE UP, MOVE DOWN (vertical only!)
- Invalid actions: Move left, move right, shoot, jump
- Strategy: Align paddle vertically with ball

**Scoring:**
- Says "MOVE UP" or "MOVE DOWN" specifically: +0.5
- Says "move paddle up/down" (hedging): +0.1 maximum
- Mentions left/right movement: 0.0 (invalid for Pong)
- Analyzes ball position relative to paddle: +0.4
"""

    elif 'breakout' in game:
        return """
**BREAKOUT-SPECIFIC STRATEGY:**
- Valid actions: MOVE LEFT, MOVE RIGHT (horizontal only!)
- Invalid actions: Move up, move down, shoot, jump
- Strategy: Position paddle under ball trajectory

**Key differences from Pong:**
- Breakout = HORIZONTAL movement (left/right)
- Pong = VERTICAL movement (up/down)
- **This is critical - many VLMs confuse these!**

**Scoring:**
- Says "MOVE LEFT" or "MOVE RIGHT" specifically: +0.5
- Says "move paddle left/right" (hedging): +0.1 maximum
- Says "move up/down": 0.0 (INVALID for Breakout)
- Analyzes ball trajectory and paddle position: +0.4
- Mentions "position under ball": +0.1 (good understanding)

**Auto-fail conditions:**
- Suggests vertical movement (up/down) in Breakout: 0.1 maximum
"""

    elif 'space' in game or 'invader' in game:
        return """
**SPACE INVADERS-SPECIFIC STRATEGY:**
- Valid actions: MOVE LEFT, MOVE RIGHT, SHOOT/FIRE
- Invalid actions: Move up, move down, jump
- Strategy: Position ship, shoot aliens, use shields

**Complexity:**
- More strategic than Pong/Breakout (2 action dimensions: move + shoot)
- Can be defensive (hide behind shields) or offensive (shoot aliens)
- Should consider alien proximity and shield positions

**Scoring:**
- Specific movement direction (LEFT/RIGHT): +0.25
- Mentions shooting/firing: +0.25
- Analyzes alien threat level: +0.2
- Mentions shield usage: +0.15
- Vague ("move and shoot"): +0.2 maximum

**Good responses:**
- "Move left to shield, shoot approaching aliens": 0.85+
- "Fire at nearest aliens while repositioning right": 0.80+
- "Shoot aliens": 0.40 (too vague, no positioning)
- "Move up to avoid aliens": 0.0 (invalid action)
"""

    else:
        return ""
