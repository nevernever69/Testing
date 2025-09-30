#!/usr/bin/env python3
"""
Create benchmark dataset.

Usage:
    python create_benchmark_dataset.py --output ./benchmark_v1.0 --frames_per_game 50
"""

import argparse
from automatic_benchmark.dataset import BenchmarkDatasetCreator
from automatic_benchmark.config import GAMES, FRAMES_PER_GAME, RANDOM_SEED


def main():
    parser = argparse.ArgumentParser(description='Create automatic benchmark dataset')
    parser.add_argument(
        '--output',
        type=str,
        default='./automatic_benchmark_dataset_v1.0',
        help='Output directory for dataset'
    )
    parser.add_argument(
        '--frames_per_game',
        type=int,
        default=FRAMES_PER_GAME,
        help=f'Number of frames per game (default: {FRAMES_PER_GAME})'
    )
    parser.add_argument(
        '--games',
        nargs='+',
        default=GAMES,
        help=f'Games to include (default: {GAMES})'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=RANDOM_SEED,
        help=f'Random seed for reproducibility (default: {RANDOM_SEED})'
    )

    args = parser.parse_args()

    print(f"\n{'='*70}")
    print("Automatic Benchmark Dataset Creator")
    print(f"{'='*70}\n")

    # Create dataset
    creator = BenchmarkDatasetCreator(
        output_dir=args.output,
        seed=args.seed
    )

    dataset = creator.create_dataset(
        games=args.games,
        frames_per_game=args.frames_per_game
    )

    print(f"\n{'='*70}")
    print("Dataset creation complete!")
    print(f"{'='*70}")
    print(f"Total frames: {dataset['metadata']['total_frames']}")
    print(f"Location: {args.output}")
    print(f"\nNext steps:")
    print(f"  1. Review dataset: {args.output}/README.md")
    print(f"  2. Run benchmark: python run_automatic_benchmark.py --dataset {args.output}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
