#!/usr/bin/env python3
"""
Generate Breakout scenarios only (human-played).
Overwrites existing Breakout data in benchmark_v2.0_scenarios.
"""

from generate_scenario_dataset_human import HumanPlayedDatasetGenerator

def main():
    print("\n" + "=" * 80)
    print("  BREAKOUT SCENARIO GENERATOR (Human-Played)")
    print("=" * 80)
    print()
    print("This will generate Breakout scenarios only.")
    print("Output: ./benchmark_v2.0_scenarios (will override existing Breakout data)")
    print()
    print("Controls:")
    print("  - LEFT/RIGHT arrow keys: Move paddle")
    print("  - SPACE: Launch ball")
    print("  - +/- keys: Adjust game speed (default 15 FPS)")
    print()
    print("You need to collect 10 scenarios:")
    print("  1. ball_lower_area")
    print("  2. ball_upper_area")
    print("  3. ball_left_side")
    print("  4. ball_right_side")
    print("  5. ball_center_horizontal")
    print("  6. paddle_left_area")
    print("  7. paddle_right_area")
    print("  8. paddle_center_area")
    print("  9. many_bricks")
    print("  10. few_bricks")
    print()

    ready = input("Ready to play Breakout? (y/n): ").lower()

    if ready != 'y':
        print("Exiting...")
        return

    # Initialize generator with same output directory
    generator = HumanPlayedDatasetGenerator(output_dir="./benchmark_v2.0_scenarios")

    # Play only Breakout
    generator.play_game_human('Breakout')

    # Finalize dataset (updates metadata)
    generator.finalize_dataset()

    print("\n✓ Breakout scenario generation complete!")
    print(f"   Output: ./benchmark_v2.0_scenarios")


if __name__ == '__main__':
    main()
