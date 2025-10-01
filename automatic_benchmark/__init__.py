"""
Automatic Atari Benchmark System

A modular, reproducible benchmark for evaluating Vision-Language Models
on Atari game understanding using OCAtari ground truth.

Key Features:
- Fixed, reproducible test sets (50 frames per game)
- Three-tier scoring: Rule-based + Semantic + LLM-judge
- Automatic validation using OCAtari ground truth
- Statistical significance testing
- Easy integration with existing pipelines

Usage:
    from automatic_benchmark import BenchmarkRunner, AutomaticEvaluator

    evaluator = AutomaticEvaluator(use_llm_judge=True)
    runner = BenchmarkRunner(dataset_path='./benchmark_dataset', evaluator=evaluator)
    results = runner.run_benchmark(your_pipeline, 'YourPipelineName')
"""

__version__ = '1.0.0'

from .evaluator import AutomaticEvaluator
from .dataset.creator import BenchmarkDatasetCreator
from .dataset.loader import BenchmarkDatasetLoader

__all__ = [
    'AutomaticEvaluator',
    'BenchmarkDatasetCreator',
    'BenchmarkDatasetLoader',
]
