#!/usr/bin/env python3
"""
Generate scenario-based dataset with distinct frames per game.
Simplified version with wide/relaxed validators for high success rate.

Usage:
    python generate_scenario_dataset.py
"""

from automatic_benchmark.dataset.scenario_based_creator import ScenarioBasedDatasetCreator


def main():
    print("Starting scenario-based dataset generation...")
    print()
    print("This will generate (SIMPLIFIED & RELAXED):")
    print("  - 10 distinct scenarios for Pong")
    print("  - 10 distinct scenarios for Breakout")
    print("  - 10 distinct scenarios for Space Invaders")
    print("  - Total: 30 frames")
    print()
    print("Using wide validators for high success rate and variety.")
    print()

    creator = ScenarioBasedDatasetCreator(output_dir="./benchmark_v2.0_scenarios")
    creator.generate_scenario_based_dataset(max_attempts_per_scenario=20000)

    print()
    print("Done! Check ./benchmark_v2.0_scenarios/ for results.")


if __name__ == '__main__':
    main()
