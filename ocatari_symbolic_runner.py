#!/usr/bin/env python3
"""
OCAtari Symbolic Runner - Uses OCAtari ground truth coordinates for action decisions.
Similar to advance_game_runner.py but feeds OCAtari coordinates instead of VLM detections.
"""

import os
import csv
import pickle
import json
import cv2
import numpy as np
from tqdm import tqdm
from ocatari_ground_truth import OCAtariGroundTruth
from advanced_zero_shot_pipeline import AdvancedSymbolicDetector
import argparse
from datetime import datetime
import logging
import re


class OCAtariSymbolicRunner:
    def __init__(self, game_type, provider, model_name, output_dir="./ocatari_experiments/",
                 openrouter_api_key=None, num_frames=600, aws_region="us-east-1", seed=None):
        self.game_type = game_type.lower()
        self.provider = provider.lower()
        self.model_name = model_name
        self.num_frames = num_frames
        self.seed = seed
        self.aws_region = aws_region

        # Results directory structure (like advance_game_runner)
        self.base_dir = output_dir
        model_safe = model_name.replace('/', '_').replace(':', '_')
        self.new_dir = os.path.join(self.base_dir, f"{game_type}_{provider}_{model_safe}_ocatari_symbolic")
        self.results_dir = os.path.join(self.new_dir, "Results")
        self.frames_dir = os.path.join(self.results_dir, "frames")
        self.detections_dir = os.path.join(self.results_dir, "detections")
        self.responses_dir = os.path.join(self.results_dir, "responses")
        self.prompts_dir = os.path.join(self.results_dir, "prompts")
        self.videos_dir = os.path.join(self.results_dir, "videos")
        self.logs_dir = os.path.join(self.results_dir, "logs")

        # Create directories
        for directory in [self.new_dir, self.results_dir, self.frames_dir,
                         self.detections_dir, self.responses_dir, self.prompts_dir,
                         self.videos_dir, self.logs_dir]:
            os.makedirs(directory, exist_ok=True)

        # Stats
        self.action_list = []
        self.cum_rewards = []
        self.rewards = 0
        self.steps_taken = 0
        self.video_frames = []
        self.checkpoint_interval = 50
        self.skip_initial_frames = self._get_skip_frames()

        # Setup logging
        self.setup_logging()

        # Initialize OCAtari
        self.ocatari_env = OCAtariGroundTruth(self.get_ocatari_name(), render_mode='rgb_array')
        if seed is not None:
            self.ocatari_env.reset(seed=seed)
        else:
            self.ocatari_env.reset()
        self.logger.info(f"✓ OCAtari initialized for {self.get_ocatari_name()}")

        # Initialize VLM for action decisions
        self.detector = AdvancedSymbolicDetector(
            openrouter_api_key=openrouter_api_key or "",
            model_name=model_name,
            detection_mode="specific",
            provider=provider,
            aws_region=aws_region,
            disable_history=True
        )
        self.logger.info(f"✓ VLM initialized: {provider}/{model_name}")

        # Run
        self.run_ocatari_symbolic()

        # Save summary
        self.save_summary()

    def setup_logging(self):
        self.logger = logging.getLogger(f"OCAtariSymbolic_{self.provider}_{self.game_type}")
        self.logger.setLevel(logging.DEBUG)
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)

        log_file = os.path.join(self.logs_dir, f"ocatari_symbolic_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

        self.logger.info("="*50 + " NEW OCATARI SYMBOLIC SESSION " + "="*50)
        self.logger.info(f"Game: {self.game_type}")
        self.logger.info(f"Provider: {self.provider}")
        self.logger.info(f"Model: {self.model_name}")
        self.logger.info(f"Output directory: {self.new_dir}")
        self.logger.info(f"Number of frames: {self.num_frames}")
        self.logger.info(f"Skip initial frames: {self.skip_initial_frames}")
        self.logger.info(f"Seed: {self.seed if self.seed is not None else 'None (random)'}")

    def _get_skip_frames(self):
        """Get number of initial frames to skip based on game type"""
        skip_frames = {
            "breakout": 15,
            "space_invaders": 45,
            "pong": 15,
            "tennis": 15,
            "assault": 30,
            "pacman": 40,
            "mspacman": 96
        }
        return skip_frames.get(self.game_type, 15)

    def get_ocatari_name(self):
        game_map = {
            'pong': 'Pong',
            'breakout': 'Breakout',
            'space_invaders': 'SpaceInvaders',
            'spaceinvaders': 'SpaceInvaders',
            'assault': 'Assault',
            'tennis': 'Tennis',
            'pacman': 'Pacman',
            'mspacman': 'MsPacman'
        }
        return game_map.get(self.game_type, self.game_type.title())

    def get_game_controls(self):
        controls = {
            "breakout": {0: "NOOP", 1: "FIRE", 2: "RIGHT", 3: "LEFT"},
            "space_invaders": {0: "NOOP", 1: "FIRE", 2: "RIGHT", 3: "LEFT", 4: "RIGHTFIRE", 5: "LEFTFIRE"},
            "pong": {0: "NOOP", 1: "FIRE", 2: "UP", 3: "DOWN", 4: "UPFIRE", 5: "DOWNFIRE"},
            "tennis": {0: "NOOP", 1: "FIRE", 2: "UP", 3: "RIGHT", 4: "LEFT", 5: "DOWN"},
            "assault": {0: "NOOP", 1: "FIRE", 2: "UP", 3: "RIGHT", 4: "LEFT", 5: "RIGHTFIRE", 6: "LEFTFIRE"},
            "pacman": {0: "NOOP", 1: "UP", 2: "RIGHT", 3: "LEFT", 4: "DOWN"},
            "mspacman": {0: "NOOP", 1: "UP", 2: "RIGHT", 3: "LEFT", 4: "DOWN", 5: "UPRIGHT", 6: "UPLEFT", 7: "DOWNRIGHT", 8: "DOWNLEFT"}
        }
        return controls.get(self.game_type, {0: "NOOP", 1: "FIRE", 2: "RIGHT", 3: "LEFT"})

    def scale_ocatari_to_vlm(self, x, y, w, h):
        """Scale OCAtari coordinates (160×210) to VLM resolution (1280×720)"""
        x_scale = 1280 / 160
        y_scale = 720 / 210
        return int(x * x_scale), int(y * y_scale), int(w * x_scale), int(h * y_scale)

    def create_prompt_from_ocatari(self, ocatari_objects):
        """Create prompt with OCAtari ground truth coordinates scaled to 1280×720"""
        game_controls = self.get_game_controls()
        controls_text = "\n".join([f"- Action {k}: {v}" for k, v in game_controls.items()])

        objects_text = ""
        for obj in ocatari_objects:
            obj_dict = obj.to_dict()
            label = obj_dict.get('category', 'unknown')

            # OCAtari stores position as tuple (x, y) and size as tuple (w, h)
            position = obj_dict.get('position', (0, 0))
            size = obj_dict.get('size', (0, 0))
            x_ocatari, y_ocatari = position
            w_ocatari, h_ocatari = size

            # Scale to VLM resolution
            x, y, w, h = self.scale_ocatari_to_vlm(x_ocatari, y_ocatari, w_ocatari, h_ocatari)
            objects_text += f"- {label}: x={x}, y={y}, size={w}x{h}\n"

        prompt = f"""You are an expert {self.game_type.replace('_', ' ').title()} player.

Game controls:
{controls_text}

Current game state (OCAtari ground truth coordinates):
Total objects: {len(ocatari_objects)}

Detected objects:
{objects_text}

Analyze the game state and choose the optimal action.

Think step by step:
1. Identify key object positions
2. Predict trajectories
3. Consider strategy
4. Choose optimal action

Return ONLY JSON:
{{
    "reasoning": "your analysis",
    "action": integer_action_code
}}"""
        return prompt

    def get_action_from_vlm(self, prompt):
        """Get action decision from VLM using OCAtari coordinates"""
        try:
            # Create message in the format expected by _make_api_call
            messages = [{
                "role": "user",
                "content": prompt
            }]

            # Use detector's _make_api_call method
            response = self.detector._make_api_call(messages, max_tokens=1000, call_id="action_decision")

            if not response:
                self.logger.warning("Empty response from VLM")
                return 0, "Empty response", ""

            # Parse JSON response
            json_match = re.search(r'\{[^}]*"action"\s*:\s*(\d+)[^}]*\}', response, re.DOTALL)
            if json_match:
                action = int(json_match.group(1))
                reasoning_match = re.search(r'"reasoning"\s*:\s*"([^"]*)"', response, re.DOTALL)
                reasoning = reasoning_match.group(1) if reasoning_match else "No reasoning"
                return action, reasoning, response
            else:
                self.logger.warning("Failed to parse action from response")
                return 0, "Parse error", response

        except Exception as e:
            self.logger.error(f"VLM error: {e}")
            return 0, f"Error: {e}", ""

    def save_checkpoint(self, checkpoint_name):
        """Save checkpoint (like advance_game_runner)"""
        checkpoint_dir = os.path.join(self.results_dir, f"checkpoint_{checkpoint_name}")
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Save actions and rewards
        with open(os.path.join(checkpoint_dir, "actions_rewards.csv"), "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["action", "cumulative_reward"])
            for a, r in zip(self.action_list, self.cum_rewards):
                writer.writerow([a, r])

        # Save reward
        with open(os.path.join(checkpoint_dir, "reward.txt"), "w") as f:
            f.write(f"Checkpoint {checkpoint_name}\n")
            f.write(f"Steps taken: {self.steps_taken}\n")
            f.write(f"Current reward: {self.rewards}\n")

        # Save video segment
        if self.video_frames:
            video_path = os.path.join(self.videos_dir, f"gameplay_segment_{checkpoint_name}.mp4")
            if len(self.video_frames) > 0:
                height, width, _ = self.video_frames[0].shape
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                video = cv2.VideoWriter(video_path, fourcc, 30.0, (width, height))

                for frame in self.video_frames:
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    for _ in range(15):  # 30 FPS from 2 FPS
                        video.write(frame_bgr)

                video.release()
                self.logger.info(f"Video segment saved: {video_path}")

        self.logger.info(f"Checkpoint {checkpoint_name} saved at step {self.steps_taken} with reward {self.rewards}")

    def save_step_data(self, step, frame, ocatari_objects, prompt, action, reasoning, response):
        """Save comprehensive step data (like advance_game_runner)"""
        # Scale frame to 1280×720
        frame_scaled = cv2.resize(frame, (1280, 720), interpolation=cv2.INTER_LINEAR)

        # Save scaled frame
        frame_path = os.path.join(self.frames_dir, f"frame_{step:04d}.jpg")
        frame_bgr = cv2.cvtColor(frame_scaled, cv2.COLOR_RGB2BGR)
        cv2.imwrite(frame_path, frame_bgr)

        # Save response
        response_data = {
            "step": int(step),
            "action": int(action),
            "reasoning": reasoning,
            "analysis_type": "ocatari_symbolic",
            "timestamp": datetime.now().isoformat(),
            "cumulative_reward": float(self.rewards),
            "frame_path": frame_path
        }
        response_path = os.path.join(self.responses_dir, f"response_{step:04d}.json")
        with open(response_path, 'w') as f:
            json.dump(response_data, f, indent=2)

        # Save prompt
        prompt_info = {
            "step": step,
            "prompt": prompt,
            "full_response": response,
            "timestamp": datetime.now().isoformat()
        }
        prompt_file = os.path.join(self.prompts_dir, f"prompt_{step:04d}.json")
        with open(prompt_file, 'w') as f:
            json.dump(prompt_info, f, indent=2)

        # Save OCAtari detection data
        detection_dir = os.path.join(self.detections_dir, f"step_{step:04d}")
        os.makedirs(detection_dir, exist_ok=True)

        # Save OCAtari ground truth with original coordinates
        ocatari_data = {
            'objects': [obj.to_dict() for obj in ocatari_objects],
            'frame': step,
            'num_objects': len(ocatari_objects),
            'resolution': '160x210 (OCAtari native)'
        }
        ocatari_file = os.path.join(detection_dir, "ocatari_ground_truth.json")
        with open(ocatari_file, 'w') as f:
            json.dump(ocatari_data, f, indent=2)

        # Save scaled coordinates and visualization
        if ocatari_objects:
            # Create visualization with bounding boxes (RGB frame)
            vis_frame = frame_scaled.copy()
            for obj in ocatari_objects:
                obj_dict = obj.to_dict()
                label = obj_dict.get('category', 'unknown')

                # OCAtari stores position as tuple (x, y) and size as tuple (w, h)
                position = obj_dict.get('position', (0, 0))
                size = obj_dict.get('size', (0, 0))
                x_ocatari, y_ocatari = position
                w_ocatari, h_ocatari = size

                # Scale to VLM resolution
                x, y, w, h = self.scale_ocatari_to_vlm(x_ocatari, y_ocatari, w_ocatari, h_ocatari)

                # Draw bounding box on RGB frame
                cv2.rectangle(vis_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(vis_frame, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Convert to BGR and save visualization
            vis_frame_bgr = cv2.cvtColor(vis_frame, cv2.COLOR_RGB2BGR)
            vis_path = os.path.join(detection_dir, "ocatari_visualization.jpg")
            cv2.imwrite(vis_path, vis_frame_bgr)

            # Save scaled coordinates
            scaled_objects = []
            for obj in ocatari_objects:
                obj_dict = obj.to_dict()

                # OCAtari stores position as tuple (x, y) and size as tuple (w, h)
                position = obj_dict.get('position', (0, 0))
                size = obj_dict.get('size', (0, 0))
                x_ocatari, y_ocatari = position
                w_ocatari, h_ocatari = size

                x, y, w, h = self.scale_ocatari_to_vlm(x_ocatari, y_ocatari, w_ocatari, h_ocatari)

                scaled_objects.append({
                    'category': obj_dict.get('category', 'unknown'),
                    'x': x,
                    'y': y,
                    'w': w,
                    'h': h,
                    'original_x': x_ocatari,
                    'original_y': y_ocatari,
                    'original_w': w_ocatari,
                    'original_h': h_ocatari
                })

            scaled_data = {
                'objects': scaled_objects,
                'frame': step,
                'num_objects': len(scaled_objects),
                'resolution': '1280x720 (scaled for VLM)',
                'original_resolution': '160x210 (OCAtari native)'
            }
            scaled_file = os.path.join(detection_dir, "scaled_coordinates.json")
            with open(scaled_file, 'w') as f:
                json.dump(scaled_data, f, indent=2)

        self.logger.debug(f"STEP {step:04d}: Action={action}, Reward={self.rewards}")
        self.logger.debug(f"REASONING: {reasoning}")
        self.logger.info(f"Step {step}: Action {action} - Reward: {self.rewards}")

    def run_ocatari_symbolic(self):
        """Main rollout loop using OCAtari coordinates"""
        self.logger.info("Starting OCAtari symbolic rollout")
        self.logger.info(f"Skip first {self.skip_initial_frames} frames, then process every frame")

        pbar = tqdm(total=self.num_frames, desc=f"OCAtari Symbolic {self.game_type}", unit="step")

        for step in range(self.num_frames):
            # Get OCAtari state
            frame_rgb, ocatari_objects = self.ocatari_env.get_frame_and_objects()

            # Skip initial frames with NOOP
            if step < self.skip_initial_frames:
                action = 0
                reasoning = f"NOOP for initial frame {step} (skipping first {self.skip_initial_frames})"
                full_response = ""
                prompt = ""  # No prompt during skip
                self.logger.debug(f"Step {step}: NOOP (skip initial frame)")
            else:
                # Create prompt from OCAtari coordinates
                prompt = self.create_prompt_from_ocatari(ocatari_objects)

                # Get action from VLM
                action, reasoning, full_response = self.get_action_from_vlm(prompt)

            # Validate action
            max_actions = {
                'breakout': 3, 'space_invaders': 5, 'pong': 5,
                'tennis': 5, 'assault': 6, 'pacman': 4, 'mspacman': 8
            }
            max_action = max_actions.get(self.game_type, 5)
            if action < 0 or action > max_action:
                self.logger.warning(f"Invalid action {action}, clipping to 0")
                action = 0

            # Take action
            _, reward, terminated, truncated, _ = self.ocatari_env.step(action)

            # Update stats
            self.action_list.append(action)
            old_reward = self.rewards
            self.rewards += reward
            if reward != 0:
                self.logger.info(f"Step {step}: REWARD CHANGE! +{reward} (total: {old_reward} -> {self.rewards})")
            self.cum_rewards.append(self.rewards)

            # Store video frame
            self.video_frames.append(frame_rgb)

            # Save step data
            self.save_step_data(step, frame_rgb, ocatari_objects, prompt, action, reasoning, full_response)

            # Checkpoint
            if self.steps_taken > 0 and self.steps_taken % self.checkpoint_interval == 0:
                self.save_checkpoint(f"step_{self.steps_taken}")

            self.steps_taken += 1

            # Reset if needed
            if terminated or truncated:
                self.logger.info(f"Step {step}: Episode ended, resetting")
                if self.seed is not None:
                    self.ocatari_env.reset(seed=self.seed)
                else:
                    self.ocatari_env.reset()

            pbar.update(1)
            pbar.set_postfix({"reward": self.rewards})

        pbar.close()

        # Final checkpoint
        self.save_checkpoint("final")

        self.logger.info(f"Rollout complete. Total reward: {self.rewards}")
        self.logger.info("="*50 + " SESSION COMPLETE " + "="*50)

        self.ocatari_env.close()

    def save_summary(self):
        """Save final summary"""
        # CSV
        with open(os.path.join(self.new_dir, "actions_rewards.csv"), "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["action", "cumulative_reward"])
            for a, r in zip(self.action_list, self.cum_rewards):
                writer.writerow([a, r])

        # JSON summary
        summary = {
            "game": self.game_type,
            "provider": self.provider,
            "model": self.model_name,
            "num_frames": self.num_frames,
            "total_reward": self.rewards,
            "seed": self.seed,
            "avg_reward_per_step": self.rewards / self.num_frames if self.num_frames > 0 else 0,
            "mode": "ocatari_symbolic",
            "timestamp": datetime.now().isoformat()
        }

        with open(os.path.join(self.new_dir, "summary.json"), 'w') as f:
            json.dump(summary, f, indent=2)

        self.logger.info(f"Summary saved: Total reward={self.rewards}, Avg={summary['avg_reward_per_step']:.2f}")


def main():
    parser = argparse.ArgumentParser(description="Run Atari games using OCAtari ground truth coordinates")
    parser.add_argument("--game", type=str, required=True,
                       choices=["pong", "breakout", "space_invaders", "assault", "tennis", "pacman", "mspacman"])
    parser.add_argument("--provider", type=str, required=True, choices=["bedrock", "openrouter"])
    parser.add_argument("--model", type=str, required=True, help="Model name")
    parser.add_argument("--num_frames", type=int, default=600)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default="./ocatari_experiments")
    parser.add_argument("--aws_region", type=str, default="us-east-1")
    parser.add_argument("--openrouter_key_file", type=str, default="OPENROUTER_API_KEY.txt")

    args = parser.parse_args()

    # Load API key if needed
    api_key = None
    if args.provider == 'openrouter':
        if os.path.exists(args.openrouter_key_file):
            with open(args.openrouter_key_file, 'r') as f:
                api_key = f.read().strip()
            print(f"Loaded OpenRouter API key from {args.openrouter_key_file}")
        else:
            print(f"Error: {args.openrouter_key_file} not found")
            return

    # Run
    OCAtariSymbolicRunner(
        game_type=args.game,
        provider=args.provider,
        model_name=args.model,
        output_dir=args.output_dir,
        openrouter_api_key=api_key,
        num_frames=args.num_frames,
        aws_region=args.aws_region,
        seed=args.seed
    )


if __name__ == "__main__":
    main()
