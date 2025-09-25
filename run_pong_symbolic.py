import argparse
from pong_symbolic_runner import PongRunWithSymbolicDetection
import os

def main():
    parser = argparse.ArgumentParser(description="Run Pong with Symbolic Detection")
    parser.add_argument("--game", type=str, default="ALE/Pong-v5", help="Game name (default: ALE/Pong-v5)")
    parser.add_argument("--provider", type=str, required=True, help="Model provider (openai, gemini, claude, openrouter, bedrock)")
    parser.add_argument("--model_name", type=str, required=True, help="Specific model name")
    parser.add_argument("--output_dir", default="./experiments", help="Output directory")
    parser.add_argument("--openrouter_key_file", type=str, default="OPENROUTER_API_KEY.txt", help="Path to OpenRouter API key file")
    parser.add_argument("--detection_model", default="anthropic/claude-sonnet-4", help="Model for symbolic detection")
    parser.add_argument("--use_symbolic_override", action="store_true", help="Enable symbolic information to override LLM decisions")
    parser.add_argument("--aws_region", type=str, default="us-east-1", help="AWS region for Bedrock (default: us-east-1)")
    
    args = parser.parse_args()
    
    # Read OpenRouter API key from file if it exists
    openrouter_api_key = None
    if os.path.exists(args.openrouter_key_file):
        with open(args.openrouter_key_file, 'r') as f:
            openrouter_api_key = f.read().strip()
        print(f"Loaded OpenRouter API key from {args.openrouter_key_file}")
    else:
        print(f"Warning: OpenRouter API key file {args.openrouter_key_file} not found")

    # Use the same model for detection if using Bedrock, otherwise use the specified detection model
    detection_model = args.model_name if args.provider == 'bedrock' else args.detection_model

    # Run Pong with symbolic detection
    PongRunWithSymbolicDetection(
        env_name=args.game,
        provider=args.provider,
        model_id=args.model_name,
        output_dir=args.output_dir,
        openrouter_api_key=openrouter_api_key,
        detection_model=detection_model,
        aws_region=args.aws_region
    )

if __name__ == "__main__":
    main()