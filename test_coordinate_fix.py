#!/usr/bin/env python3
"""
Test coordinate scaling fix
"""
import json
import glob
from pathlib import Path


def check_coordinates(analysis_file):
    """Check if coordinates are in correct range"""
    with open(analysis_file) as f:
        data = json.load(f)

    objects = data.get("symbolic_state", {}).get("objects", [])
    if not objects:
        return None, "No objects detected"

    issues = []
    for obj in objects:
        x = obj.get("x", 0)
        y = obj.get("y", 0)
        label = obj.get("label", "unknown")

        # Check if coordinates are in expected range for 1280x720 image
        if x > 1280 or y > 720:
            issues.append(f"❌ OUT OF BOUNDS: {label} at ({x}, {y}) - exceeds 1280x720")
        elif x < 300 and y < 300 and x > 10 and y > 10:
            # Likely in original Atari scale (not scaled)
            issues.append(f"⚠️  SUSPICIOUS: {label} at ({x}, {y}) - seems like Atari scale, not scaled")
        elif x <= 10 or y <= 10:
            # Very small coordinates - might be edge objects
            issues.append(f"ℹ️  EDGE: {label} at ({x}, {y}) - at screen edge")
        else:
            issues.append(f"✅ OK: {label} at ({x}, {y})")

    return objects, issues


def main():
    print("="*70)
    print("COORDINATE SCALING FIX - VERIFICATION TEST")
    print("="*70)
    print()

    # Find all analysis.json files
    pattern = "comparison_results/**/analysis.json"
    files = list(glob.glob(pattern, recursive=True))

    if not files:
        print("❌ No analysis.json files found in comparison_results/")
        print()
        print("Run a test first:")
        print("  GAME_TYPE=pong NUM_FRAMES=20 ./run_comparison_final.sh openrouter google/gemini-2.5-pro 42")
        return

    print(f"Found {len(files)} analysis files")
    print()

    total_objects = 0
    ok_count = 0
    suspicious_count = 0
    error_count = 0

    for file_path in sorted(files):
        relative_path = Path(file_path).relative_to("comparison_results")
        print(f"Checking: {relative_path}")

        objects, issues = check_coordinates(file_path)

        if objects is None:
            print(f"  {issues}")
            print()
            continue

        total_objects += len(objects)

        for issue in issues:
            print(f"  {issue}")
            if "✅" in issue:
                ok_count += 1
            elif "⚠️" in issue:
                suspicious_count += 1
            elif "❌" in issue:
                error_count += 1

        print()

    # Summary
    print("="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Total objects checked: {total_objects}")
    print(f"✅ Correct coordinates: {ok_count} ({ok_count/total_objects*100:.1f}%)")
    print(f"⚠️  Suspicious (needs scaling?): {suspicious_count} ({suspicious_count/total_objects*100:.1f}%)")
    print(f"❌ Errors (out of bounds): {error_count} ({error_count/total_objects*100:.1f}%)")
    print()

    if suspicious_count > 0 or error_count > 0:
        print("❌ ISSUES DETECTED - Coordinates may need fixing!")
        print()
        print("Possible causes:")
        print("1. Old results from before the fix")
        print("2. Model still returning wrong scale despite prompt")
        print("3. Auto-detection threshold needs tuning")
        print()
        print("Solutions:")
        print("1. Delete old results: rm -rf comparison_results/")
        print("2. Re-run with the fixed code")
        print("3. Check COORDINATE_SCALING_FIX.md for details")
    else:
        print("✅ ALL COORDINATES LOOK CORRECT!")
        print()
        print("The fix is working properly.")

    print("="*70)


if __name__ == '__main__':
    main()
