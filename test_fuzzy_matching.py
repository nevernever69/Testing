#!/usr/bin/env python3
"""
Quick test to verify fuzzy matching and position-based disambiguation works correctly.
"""

from coordinate_accuracy_evaluator import GameSpecificMatcher

def test_is_important_object():
    """Test the is_important_object() function with various labels."""
    print("="*80)
    print("Testing is_important_object() with Fuzzy Matching")
    print("="*80)

    test_cases = [
        # (label, game, expected_result, reason)
        ("right_paddle", "pong", True, "Has 'paddle' keyword → important"),
        ("left_paddle", "pong", True, "Has 'paddle' keyword → important"),
        ("ball", "pong", True, "Ball is always important"),
        ("Player", "pong", True, "OCAtari player label → important"),
        ("left_player_score", "pong", False, "Has 'score' keyword → excluded!"),
        ("right_player_score", "pong", False, "Has 'score' keyword → excluded!"),
        ("score_display", "pong", False, "Has 'score' keyword → excluded!"),
        ("paddle_right", "pong", True, "Has 'paddle' keyword → important"),
        ("green_paddle", "pong", True, "Has 'paddle' keyword → important"),
        ("pong_ball", "pong", True, "Has 'ball' keyword → important"),
    ]

    passed = 0
    failed = 0

    for label, game, expected, reason in test_cases:
        result = GameSpecificMatcher.is_important_object(label, game)
        status = "✅ PASS" if result == expected else "❌ FAIL"

        if result == expected:
            passed += 1
        else:
            failed += 1

        print(f"{status}: is_important_object('{label}', '{game}')")
        print(f"        Expected: {expected}, Got: {result}")
        print(f"        Reason: {reason}")
        print()

    print("="*80)
    print(f"Results: {passed} passed, {failed} failed")
    print("="*80)


def test_infer_paddle_side():
    """Test paddle side inference from position."""
    print("\n" + "="*80)
    print("Testing infer_paddle_side() - Position-Based Disambiguation")
    print("="*80)

    test_cases = [
        # (bbox, game, expected_side, description)
        ([100, 300, 150, 400], "pong", "left", "Left side paddle (x=125)"),
        ([1100, 300, 1150, 400], "pong", "right", "Right side paddle (x=1125)"),
        ([600, 300, 650, 400], "pong", "center", "Center object (x=625, might be ball)"),
        ([500, 650, 600, 680], "breakout", "bottom", "Bottom paddle for Breakout"),
    ]

    passed = 0
    failed = 0

    for bbox, game, expected, description in test_cases:
        result = GameSpecificMatcher.infer_paddle_side(bbox, game)
        status = "✅ PASS" if result == expected else "❌ FAIL"

        if result == expected:
            passed += 1
        else:
            failed += 1

        center_x = (bbox[0] + bbox[2]) / 2
        print(f"{status}: {description}")
        print(f"        bbox: {bbox}, center_x: {center_x}")
        print(f"        Expected: '{expected}', Got: '{result}'")
        print()

    print("="*80)
    print(f"Results: {passed} passed, {failed} failed")
    print("="*80)


def test_fuzzy_match_objects():
    """Test semantic matching between GT and VLM objects."""
    print("\n" + "="*80)
    print("Testing fuzzy_match_objects() - Semantic Similarity")
    print("="*80)

    test_cases = [
        # (gt_obj, vlm_obj, game, min_expected_score, description)
        (
            {'category': 'Player', 'bbox': [1100, 300, 1150, 400]},
            {'label': 'right_paddle', 'normalized_label': 'player', 'bbox': [1100, 300, 1150, 400]},
            'pong',
            0.8,
            "Right paddle match (position + label agree)"
        ),
        (
            {'category': 'Player', 'bbox': [100, 300, 150, 400]},
            {'label': 'left_paddle', 'normalized_label': 'enemy', 'bbox': [100, 300, 150, 400]},
            'pong',
            0.8,
            "Left paddle match (position + label agree)"
        ),
        (
            {'category': 'Ball', 'bbox': [600, 300, 620, 320]},
            {'label': 'ball', 'normalized_label': 'ball', 'bbox': [600, 300, 620, 320]},
            'pong',
            0.9,
            "Ball match (perfect semantic match)"
        ),
        (
            {'category': 'Player', 'bbox': [1100, 300, 1150, 400]},
            {'label': 'left_paddle', 'normalized_label': 'enemy', 'bbox': [1100, 300, 1150, 400]},
            'pong',
            0.3,
            "Paddle mismatch (position says right, label says left)"
        ),
    ]

    passed = 0
    failed = 0

    for gt_obj, vlm_obj, game, min_expected, description in test_cases:
        score = GameSpecificMatcher.fuzzy_match_objects(gt_obj, vlm_obj, game)
        status = "✅ PASS" if score >= min_expected else "❌ FAIL"

        if score >= min_expected:
            passed += 1
        else:
            failed += 1

        print(f"{status}: {description}")
        print(f"        GT: {gt_obj['category']} at {gt_obj['bbox']}")
        print(f"        VLM: {vlm_obj['label']} at {vlm_obj['bbox']}")
        print(f"        Semantic score: {score:.2f} (expected >= {min_expected})")
        print()

    print("="*80)
    print(f"Results: {passed} passed, {failed} failed")
    print("="*80)


if __name__ == "__main__":
    test_is_important_object()
    test_infer_paddle_side()
    test_fuzzy_match_objects()

    print("\n" + "="*80)
    print("✅ ALL TESTS COMPLETE!")
    print("="*80)
    print("\nIf all tests pass, the fuzzy matching system is working correctly.")
    print("Re-run your evaluation to see improved Important F1 scores!")
