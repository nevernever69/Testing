import os
import csv
import pickle
import json
import gymnasium as gym
import ale_py
import cv2
import numpy as np
from tqdm import tqdm
from gymnasium.wrappers import RecordVideo, OrderEnforcing
from advanced_zero_shot_pipeline import (
    BreakoutAdvancedDetector,
    FroggerAdvancedDetector,
    SpaceInvadersAdvancedDetector,
    PacmanAdvancedDetector,
    MsPacmanAdvancedDetector,
    PongAdvancedDetector,
    TennisAdvancedDetector,
    AssaultAdvancedDetector
)
import argparse
import tempfile
from datetime import datetime
import logging


class AdvanceGameRunner:
    def __init__(self, env_name, provider, game_type, output_dir="./experiments/", prompt=None, model_id=None,
                 openrouter_api_key=None, detection_model="anthropic/claude-sonnet-4",
                 num_frames=600, aws_region="us-east-1"):
        self.provider = provider.lower()  # 'openai', 'gemini', 'claude', 'bedrock' or 'rand'
        self.sys_prompt = prompt or ""
        self.env_name = env_name
        self.game_type = game_type.lower()
        self.action_list = []
        self.cum_rewards = []
        self.rewards = 0
        self.steps_taken = 0
        self.num_timesteps = num_frames
        self.checkpoint_interval = 50  # Checkpoint every 50 steps
        self.video_frames = []  # Store all frames for cumulative checkpoint videos
        self.skip_initial_frames = self._get_skip_frames()
        self.aws_region = aws_region  # AWS region for Bedrock

        # Results directory structure
        self.base_dir = output_dir
        self.temp_env_name = env_name.replace("ALE/", "")
        base = self.temp_env_name[:-3]
        self.display_name = model_id.replace('/', '_').replace(':', '_') if provider == 'openrouter' else provider
        self.new_dir = os.path.join(self.base_dir, f"{base}_{self.display_name}_symbolic_only")
        self.results_dir = os.path.join(self.new_dir, "Results")
        self.frames_dir = os.path.join(self.results_dir, "frames")
        self.detections_dir = os.path.join(self.results_dir, "detections")
        self.responses_dir = os.path.join(self.results_dir, "responses")
        self.prompts_dir = os.path.join(self.results_dir, "prompts")
        self.videos_dir = os.path.join(self.results_dir, "videos")
        self.logs_dir = os.path.join(self.results_dir, "logs")

        # Create directory structure
        for directory in [self.new_dir, self.results_dir, self.frames_dir,
                         self.detections_dir, self.responses_dir, self.prompts_dir,
                         self.videos_dir, self.logs_dir]:
            os.makedirs(directory, exist_ok=True)

        # Set up comprehensive logging
        self.setup_logging()

        # Initialize symbolic detector if API key provided
        self.symbolic_detector = None
        if openrouter_api_key:
            self.symbolic_detector = self._init_detector(openrouter_api_key, detection_model)

        if self.provider != 'rand' and model_id is None:
            MODELS = {
                "openai":  ["gpt-4-turbo", "gpt-4.1-mini"],
                "claude":  ["claude-3-opus-20240229","claude-3-opus-20240229"],
                "gemini":  ["gemini-2.5-flash-preview-04-17","gemini-2.5-flash-preview-04-17"],
                "openrouter": ["mistralai/mistral-7b-instruct", "mistralai/mistral-7b-instruct"],
                "bedrock": ["llama3.1-70b", "llama3.1-70b"],  # Default Bedrock model
            }
            if self.provider in MODELS:
                model_id = MODELS[self.provider][1]

        # Skip test if already done
        state_file = os.path.join(self.new_dir, f"env_{base}_state.pkl")
        if os.path.exists(state_file):
            print(f"\n\nEnvironment '{env_name}' already processed—skipping.\n\n")
            return

        # Build environment and video recorder
        gym.register_envs(ale_py)
        env = gym.make(env_name, render_mode="rgb_array")
        env = OrderEnforcing(env, disable_render_order_enforcing=True)
        env.reset()
        self.env = RecordVideo(env=env,
                                         video_folder=self.new_dir,
                                         name_prefix=base + "_rollout",
                                         episode_trigger=lambda x: x == 0)

        # Get appropriate prompt
        self.base_prompt = self._get_game_prompt()

        # Run based on provider
        if self.provider == 'rand':
            self.rand_rollout()
        else:
            self.symbolic_rollout()

        # Write summary CSV
        with open(os.path.join(self.new_dir, "actions_rewards.csv"), "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["action", "cumulative_reward"])
            for a, r in zip(self.action_list, self.cum_rewards):
                writer.writerow([a, r])

        # Save final checkpoint
        self.save_checkpoint("final")

        # Log final summary
        self.logger.info(f"Game completed. Total reward: {self.rewards}, Total steps: {self.steps_taken}")
        self.logger.info("="*50 + " GAME COMPLETE " + "="*50)

    def setup_logging(self):
        """Set up comprehensive logging for the game session"""
        # Create logger
        self.logger = logging.getLogger(f"SymbolicOnly_{self.display_name}_{self.game_type}")
        self.logger.setLevel(logging.DEBUG)

        # Clear any existing handlers
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)

        # Create file handler
        log_file = os.path.join(self.logs_dir, f"symbolic_only_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)

        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        # Add handlers to logger
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

        # Log initial session info
        self.logger.info("="*50 + " NEW SYMBOLIC ONLY SESSION " + "="*50)
        self.logger.info(f"Environment: {self.env_name}")
        self.logger.info(f"Game Type: {self.game_type}")
        self.logger.info(f"Provider: {self.provider}")
        self.logger.info(f"Output directory: {self.new_dir}")
        self.logger.info(f"Number of frames: {self.num_timesteps}")
        self.logger.info(f"Skip initial frames: {self.skip_initial_frames}")
        self.logger.info(f"AWS Region: {self.aws_region}")

        # Log the exact prompt that will be used
        prompt = self._get_game_prompt()
        self.logger.info("="*30 + " BASE PROMPT BEING USED " + "="*30)
        self.logger.info(f"Full prompt text:\n{prompt}")
        self.logger.info("="*80)

    def _get_skip_frames(self):
        """Get number of initial frames to skip based on game type"""
        skip_frames = {
            "breakout": 15,
            "frogger": 120,
            "space_invaders": 30,
            "pacman": 40,
            "mspacman": 96,
            "pong": 15,
            "tennis": 15,
            "assault": 30
        }
        return skip_frames.get(self.game_type, 30)

    def _init_detector(self, api_key, model_name):
        """Initialize the appropriate detector based on game type"""
        detectors = {
            "breakout": BreakoutAdvancedDetector,
            "frogger": FroggerAdvancedDetector,
            "space_invaders": SpaceInvadersAdvancedDetector,
            "pacman": PacmanAdvancedDetector,
            "mspacman": MsPacmanAdvancedDetector,
            "pong": PongAdvancedDetector,
            "tennis": TennisAdvancedDetector,
            "assault": AssaultAdvancedDetector
        }

        detector_class = detectors.get(self.game_type)
        if detector_class:
            # Use "specific" detection mode for symbolic only
            detection_mode = "specific"
            # If using Bedrock provider, pass the provider info to the detector
            if self.provider == 'bedrock':
                return detector_class(api_key, model_name, detection_mode, provider='bedrock', aws_region=self.aws_region)
            else:
                return detector_class(api_key, model_name, detection_mode)
        return None

    def _get_game_controls(self):
        """Get game-specific control mappings"""
        controls = {
            "breakout": {
                0: "NOOP (do nothing)",
                1: "FIRE (primary action - often shoot/serve/activate)",
                2: "RIGHT (move right or right action)",
                3: "LEFT (move left or left action)",
                4: "RIGHTFIRE (combination of right + fire)",
                5: "LEFTFIRE (combination of left + fire)"
            },
            "frogger": {
                0: "NOOP (do nothing)",
                1: "UP (move up)",
                2: "RIGHT (move right)",
                3: "LEFT (move left)",
                4: "DOWN (move down)"
            },
            "space_invaders": {
                0: "NOOP (do nothing)",
                1: "FIRE (primary action - often shoot/serve/activate)",
                2: "RIGHT (move right or right action)",
                3: "LEFT (move left or left action)",
                4: "RIGHTFIRE (combination of right + fire)",
                5: "LEFTFIRE (combination of left + fire)"
            },
            "pacman": {
                0: "NOOP (do nothing)",
                1: "UP (move up)",
                2: "RIGHT (move right)",
                3: "LEFT (move left)",
                4: "DOWN (move down)"
            },
            "mspacman": {
                0: "NOOP (do nothing)",
                1: "UP (move up)",
                2: "RIGHT (move right)",
                3: "LEFT (move left)",
                4: "DOWN (move down)",
                5: "UPRIGHT (diagonal up-right)",
                6: "UPLEFT (diagonal up-left)",
                7: "DOWNRIGHT (diagonal down-right)",
                8: "DOWNLEFT (diagonal down-left)"
            },
            "pong": {
                0: "NOOP (do nothing)",
                1: "FIRE (primary action - often shoot/serve/activate)",
                2: "RIGHT (move right or right action)",
                3: "LEFT (move left or left action)",
                4: "RIGHTFIRE (combination of right + fire)",
                5: "LEFTFIRE (combination of left + fire)"
            },
            "tennis": {
                0: "NOOP (do nothing)",
                1: "FIRE (hit/serve ball)",
                2: "UP (move toward net/forward)",
                3: "RIGHT (move right)",
                4: "LEFT (move left)",
                5: "DOWN (move toward baseline/backward)"
            },
            "assault": {
                0: "NOOP (do nothing)",
                1: "FIRE (shoot)",
                2: "UP (move up)",
                3: "RIGHT (move right)",
                4: "LEFT (move left)",
                5: "RIGHTFIRE (move right and shoot)",
                6: "LEFTFIRE (move left and shoot)"
            }
        }
        return controls.get(self.game_type, {
            0: "NOOP (do nothing)",
            1: "FIRE (primary action - often shoot/serve/activate)",
            2: "RIGHT (move right or right action)",
            3: "LEFT (move left or left action)",
            4: "RIGHTFIRE (combination of right + fire)",
            5: "LEFTFIRE (combination of left + fire)"
        })

    def _get_game_prompt(self):
        """Get game-specific prompt"""
        if self.game_type == "tennis":
            return "You are an expert Tennis player controlling the RED PLAYER."
        else:
            return "You are an expert game player analyzing a game frame."

    def _create_enhanced_prompt(self, symbolic_state):
        """Create enhanced prompt with symbolic information using game-specific format"""
        # Get game controls
        game_controls = self._get_game_controls()

        # Build controls text
        controls_text = ""
        for action_num, description in game_controls.items():
            controls_text += f"- Action {action_num}: {description}\n"

        # Game-specific introduction
        if self.game_type == "tennis":
            game_intro = """You are an expert Tennis player controlling the RED PLAYER.

IMPORTANT: You are controlling the RED PLAYER."""
        else:
            game_intro = "You are an expert game player analyzing a game frame."

        # Start building the prompt
        prompt = """{game_intro}

Game controls:
{controls_text}
Current frame analysis:
- Total objects detected: {total_objects}

Detected objects with coordinates and positions:
""".format(
            game_intro=game_intro,
            controls_text=controls_text,
            total_objects=symbolic_state.get("total_objects", 0) if symbolic_state else 0
        )

        # Add object positions if symbolic_state exists
        if symbolic_state:
            if 'objects' in symbolic_state and isinstance(symbolic_state['objects'], list):
                # New format: symbolic_state = {"objects": [{"label": "paddle", "x": 645, "y": 657}, ...]}
                for obj in symbolic_state.get("objects", []):
                    x = obj.get('x', 'unknown')
                    y = obj.get('y', 'unknown')
                    label = obj.get('label', 'unknown_object')
                    prompt += f"- Object '{label}': positioned at coordinates x={x}, y={y}\n"
            else:
                # Legacy format: process all detected objects dynamically
                for obj_label, obj_data in symbolic_state.items():
                    if isinstance(obj_data, dict):
                        x = obj_data.get('x', 'unknown')
                        y = obj_data.get('y', 'unknown')
                        prompt += f"- Object '{obj_label}': positioned at coordinates x={x}, y={y}\n"
                    elif isinstance(obj_data, list):
                        for i, instance in enumerate(obj_data):
                            if isinstance(instance, dict):
                                x = instance.get('x', 'unknown')
                                y = instance.get('y', 'unknown')
                                if len(obj_data) > 1:
                                    prompt += f"- Object '{obj_label} #{i+1}': positioned at coordinates x={x}, y={y}\n"
                                else:
                                    prompt += f"- Object '{obj_label}': positioned at coordinates x={x}, y={y}\n"

        # Add game-specific ending with visual focus instruction
        if self.game_type == "tennis":
            strategy_section = """

IMPORTANT: Prioritize understanding the visual frame for spatial and gameplay context, while drawing on symbolic information when it provides useful complementary insight

As an expert Tennis player controlling the RED PLAYER, analyze the scene and choose the optimal action.

Consider:
- Object positions and their relationships
- Movement patterns and game dynamics
- Strategic positioning for optimal gameplay

Return ONLY JSON:
{
    "reasoning": "your expert analysis focusing on the red player",
    "action": integer_action_code
}
"""
        else:
            strategy_section = """

IMPORTANT:Prioritize understanding the visual frame for spatial and gameplay context, while drawing on symbolic information when it provides useful complementary insight
As an expert player, analyze the scene and choose the optimal action.

Consider:
- Object positions and their relationships
- Movement patterns and game dynamics
- Strategic positioning for optimal gameplay

Return ONLY JSON:
{
    "reasoning": "your expert analysis with positional awareness",
    "action": integer_action_code
}
"""

        prompt += strategy_section
        return prompt

    def save_states(self, rewards, action):
        # Only save ALE states for ALE environments
        if hasattr(self.env.unwrapped, 'ale'):
            state = self.env.unwrapped.ale.cloneState()
            rand_state = getattr(self.env.unwrapped, 'np_random', self.env.unwrapped.np_random)
            self.states = getattr(self, 'states', [])
            self.states.append((state, rand_state, rewards, self.steps_taken, action))
            with open(os.path.join(self.new_dir, f"env_{self.temp_env_name[:-3]}_state.pkl"), "wb") as f:
                pickle.dump(self.states, f)
        else:
            # For non-ALE environments like FlappyBird, just save basic info
            self.states = getattr(self, 'states', [])
            state_info = {
                'rewards': rewards,
                'steps': self.steps_taken,
                'action': action,
                'game_type': self.game_type
            }
            self.states.append(state_info)
            with open(os.path.join(self.new_dir, f"env_{self.temp_env_name[:-3]}_state.pkl"), "wb") as f:
                pickle.dump(self.states, f)

    def save_video_segment(self, frames, checkpoint_name):
        """Save a video segment from a list of frames"""
        if not frames:
            return

        # Create video for this segment
        video_path = os.path.join(self.videos_dir, f"gameplay_segment_{checkpoint_name}.mp4")

        # Get frame dimensions
        if len(frames) > 0 and frames[0] is not None:
            height, width, layers = frames[0].shape
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            # Use 30 FPS for smooth playback
            video = cv2.VideoWriter(video_path, fourcc, 30.0, (width, height))

            # Duplicate frames to achieve 30 FPS
            for frame in frames:
                if frame is not None:
                    # Convert from RGB to BGR for OpenCV
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    # Duplicate frame 15 times to get 30 FPS from 2 FPS source
                    for _ in range(15):
                        video.write(frame_bgr)

            video.release()
            print(f"Video segment saved: {video_path}")

    def save_checkpoint(self, checkpoint_name):
        """Save a checkpoint with the current state"""
        checkpoint_dir = os.path.join(self.results_dir, f"checkpoint_{checkpoint_name}")
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Save actions and rewards up to this point
        with open(os.path.join(checkpoint_dir, "actions_rewards.csv"), "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["action", "cumulative_reward"])
            for a, r in zip(self.action_list, self.cum_rewards):
                writer.writerow([a, r])

        # Save current reward
        with open(os.path.join(checkpoint_dir, "reward.txt"), "w") as f:
            f.write(f"Checkpoint {checkpoint_name}\n")
            f.write(f"Steps taken: {self.steps_taken}\n")
            f.write(f"Current reward: {self.rewards}\n")

        # Save video segment (cumulative - all frames from start to current point)
        self.save_video_segment(self.video_frames, checkpoint_name)

        print(f"Checkpoint {checkpoint_name} saved at step {self.steps_taken} with reward {self.rewards}")

    def save_frame_and_response(self, frame, action, reasoning, step_number, symbolic_state=None, enhanced_prompt=None):
        """Save frame and response with comprehensive logging"""
        # Save frame
        frame_path = os.path.join(self.frames_dir, f"frame_{step_number:04d}.jpg")
        cv2.imwrite(frame_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        # Save response with additional metadata
        response_data = {
            "step": int(step_number),
            "action": int(action),
            "reasoning": reasoning,
            "analysis_type": "symbolic_detection",
            "timestamp": datetime.now().isoformat(),
            "cumulative_reward": float(self.rewards),
            "frame_path": frame_path
        }

        response_path = os.path.join(self.responses_dir, f"response_{step_number:04d}.json")
        with open(response_path, 'w') as f:
            json.dump(response_data, f, indent=2)

        # Save prompt information
        if symbolic_state is not None or enhanced_prompt is not None:
            self.save_prompts_and_api_info(step_number, self.base_prompt, symbolic_state, enhanced_prompt)

        # Log the decision
        self.logger.debug(f"STEP {step_number:04d}: Action={action}, Reward={self.rewards}")
        self.logger.debug(f"REASONING: {reasoning}")

        # Log action details
        self.logger.info(f"Step {step_number}: Action {action} - Reward: {self.rewards}")

        print(f"Saved frame and response for step {step_number} (symbolic detection)")

    def save_prompts_and_api_info(self, step_number, base_prompt, symbolic_state=None, final_prompt=None):
        """Save all prompt information for analysis"""
        prompt_info = {
            "step": step_number,
            "base_prompt": base_prompt,
            "symbolic_state": symbolic_state,
            "final_prompt_sent_to_api": final_prompt,
            "timestamp": datetime.now().isoformat()
        }

        prompt_file = os.path.join(self.prompts_dir, f"prompt_{step_number:04d}.json")
        with open(prompt_file, 'w') as f:
            json.dump(prompt_info, f, indent=2)

        self.logger.debug(f"Step {step_number}: Saved prompt information")

    def get_action_from_symbolic(self, frame_path, step_number):
        """Get action from symbolic detection pipeline only"""
        if not self.symbolic_detector:
            return 0, "No symbolic detector available", None, None

        try:
            # Create permanent analysis directory for this step
            analysis_dir = os.path.join(self.detections_dir, f"step_{step_number:04d}")
            os.makedirs(analysis_dir, exist_ok=True)

            # Process frame with symbolic detector
            results = self.symbolic_detector.process_single_frame(frame_path, analysis_dir)

            # Extract symbolic state
            symbolic_state = results.get('symbolic_state', {})

            # Get the actual prompt that was sent to the API
            enhanced_prompt = results.get('actual_prompt_used', self._create_enhanced_prompt(symbolic_state))

            # Extract action and reasoning from results
            action_decision = results.get('action_decision', {})
            action = action_decision.get('action', 0)
            reasoning = action_decision.get('reasoning', 'No reasoning provided')

            # Log symbolic detection results
            self.logger.debug(f"Step {step_number}: Symbolic detection completed")
            self.logger.debug(f"Detected objects: {list(symbolic_state.keys()) if symbolic_state else 'None'}")

            return action, reasoning, symbolic_state, enhanced_prompt

        except Exception as e:
            self.logger.error(f"Error getting action from symbolic detection: {e}")
            return 0, f"Error in symbolic detection: {e}", None, None

    def rand_rollout(self):
        obs, info = self.env.reset()
        self.save_states(self.rewards, 0)
        pbar = tqdm(total=self.num_timesteps, desc=f"Random Rollout ({self.temp_env_name})", unit="step")

        for step in range(self.num_timesteps - self.steps_taken):
            obs = cv2.resize(obs, (512, 512))
            action = self.env.action_space.sample()
            self.action_list.append(action)
            obs, rew, term, trunc, info = self.env.step(action)
            self.save_states(self.rewards, action)
            self.rewards += rew
            self.cum_rewards.append(self.rewards)

            # Store frame for video checkpointing (keep all frames for cumulative videos)
            frame = self.env.render()
            if frame is not None:
                self.video_frames.append(frame)

            # Save frame and response for every frame (no skipping)
            self.save_frame_and_response(frame, action, f"Random action: {action}", self.steps_taken)

            # Save checkpoint every 50 steps
            if self.steps_taken > 0 and self.steps_taken % self.checkpoint_interval == 0:
                self.save_checkpoint(f"step_{self.steps_taken}")

            if term or trunc:
                obs, info = self.env.reset()
            self.steps_taken += 1
            pbar.update(1)
            pbar.set_postfix({"reward": self.rewards})

        pbar.close()
        print(f"Total reward (random): {self.rewards}")

        self.env.close()

    def symbolic_rollout(self):
        """Symbolic detection rollout only - processes every frame after skip period"""

        self.logger.info("Starting symbolic detection rollout")
        self.logger.info(f"Skip first {self.skip_initial_frames} frames, then process every frame")

        obs, info = self.env.reset()
        self.save_states(self.rewards, 0)
        pbar = tqdm(total=self.num_timesteps, desc=f"{self.display_name} {self.game_type.title()} Symbolic ({self.temp_env_name})", unit="step")

        for step in range(self.num_timesteps - self.steps_taken):
            # Skip the initial frames for decision making, then process every frame
            if step < self.skip_initial_frames:
                # No-op for initial frames
                action = 0
                reasoning = f"NOOP for initial frame {step} (skipping first {self.skip_initial_frames})"
                symbolic_state = None
                enhanced_prompt = None
                self.logger.debug(f"Step {step}: NOOP (skip initial frame)")
            else:
                # Get current frame for processing
                frame = self.env.render()
                if frame is not None:
                    # Create temporary frame for symbolic processing (don't save to frames dir)
                    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
                        temp_frame_path = temp_file.name
                    cv2.imwrite(temp_frame_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

                    try:
                        # Get action from symbolic detection pipeline
                        self.logger.debug(f"Step {step}: Processing frame with symbolic detection...")
                        action, reasoning, symbolic_state, enhanced_prompt = self.get_action_from_symbolic(temp_frame_path, self.steps_taken)

                        # Log the results
                        self.logger.debug(f"Step {step}: Symbolic detection returned action={action}")
                        self.logger.debug(f"Step {step}: Enhanced prompt length: {len(enhanced_prompt) if enhanced_prompt else 0}")

                    finally:
                        # Clean up temp file
                        try:
                            os.unlink(temp_frame_path)
                        except:
                            pass
                else:
                    action = 0  # NOOP
                    reasoning = "No frame available for symbolic processing"
                    symbolic_state = None
                    enhanced_prompt = None

            # Record action & step
            self.action_list.append(action)
            obs, rew, term, trunc, info = self.env.step(action)
            self.env.render()
            self.save_states(self.rewards, action)

            # Log reward changes
            old_reward = self.rewards
            self.rewards += rew
            if rew != 0:
                self.logger.info(f"Step {step}: REWARD CHANGE! +{rew} (total: {old_reward} -> {self.rewards})")

            self.cum_rewards.append(self.rewards)

            # Store frame for video checkpointing (keep all frames for cumulative videos)
            frame = self.env.render()
            if frame is not None:
                self.video_frames.append(frame)

            # Save frame and response immediately for every frame
            self.save_frame_and_response(frame, action, reasoning, self.steps_taken, symbolic_state, enhanced_prompt)

            # Save checkpoint every 50 steps
            if self.steps_taken > 0 and self.steps_taken % self.checkpoint_interval == 0:
                self.save_checkpoint(f"step_{self.steps_taken}")
                self.logger.info(f"Checkpoint saved at step {self.steps_taken}")

            if term or trunc:
                self.logger.info(f"Step {step}: Episode ended (term={term}, trunc={trunc}), resetting environment")
                obs, info = self.env.reset()

            self.steps_taken += 1
            pbar.update(1)
            pbar.set_postfix({"reward": self.rewards})

        pbar.close()
        self.logger.info(f"Rollout completed. Final reward: {self.rewards}")

        self.env.close()


def main():
    parser = argparse.ArgumentParser(description="Run Atari Games with Symbolic Detection Only")
    parser.add_argument("--game", type=str, default="ALE/Breakout-v5", help="Game name")
    parser.add_argument("--game_type", type=str, choices=["breakout", "frogger", "space_invaders", "pacman", "mspacman", "pong", "tennis", "assault"],
                        required=True, help="Game type for pipeline")
    parser.add_argument("--provider", type=str, required=True, help="Model provider (openai, gemini, claude, openrouter, bedrock)")
    parser.add_argument("--model_name", type=str, required=True, help="Specific model name")
    parser.add_argument("--output_dir", default="./experiments", help="Output directory")
    parser.add_argument("--openrouter_key_file", type=str, default="OPENROUTER_API_KEY.txt",
                        help="Path to OpenRouter API key file")
    parser.add_argument("--detection_model", default="anthropic/claude-sonnet-4",
                        help="Model for symbolic detection")
    parser.add_argument("--num_frames", type=int, default=600, help="Number of frames to run (default: 400)")
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

    # Run game with symbolic detection only
    AdvanceGameRunner(
        env_name=args.game,
        provider=args.provider,
        game_type=args.game_type,
        model_id=args.model_name,
        output_dir=args.output_dir,
        openrouter_api_key=openrouter_api_key,
        detection_model=detection_model,
        num_frames=args.num_frames,
        aws_region=args.aws_region
    )


if __name__ == "__main__":
    main()
