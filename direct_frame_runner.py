import os
import csv
import pickle
import json
import base64
import requests
import re
import gymnasium as gym
import ale_py
import cv2
import numpy as np
from tqdm import tqdm
from gymnasium.wrappers import RecordVideo, OrderEnforcing
import argparse
import tempfile
from datetime import datetime
import logging


class DirectFrameRunner:
    def __init__(self, env_name, provider, game_type, output_dir="./experiments/", prompt=None, model_id=None,
                 api_key=None, num_frames=600, aws_region="us-east-1", disable_history=False, seed=None):
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
        self.disable_history = disable_history  # Flag to disable history mechanism
        self.api_key = api_key
        self.seed = seed

        # Results directory structure
        self.base_dir = output_dir
        self.temp_env_name = env_name.replace("ALE/", "")
        base = self.temp_env_name[:-3]
        self.display_name = model_id.replace('/', '_').replace(':', '_') if provider == 'openrouter' else provider
        self.new_dir = os.path.join(self.base_dir, f"{base}_{self.display_name}_direct_frame")
        self.results_dir = os.path.join(self.new_dir, "Results")
        self.frames_dir = os.path.join(self.results_dir, "frames")
        self.responses_dir = os.path.join(self.results_dir, "responses")
        self.prompts_dir = os.path.join(self.results_dir, "prompts")
        self.videos_dir = os.path.join(self.results_dir, "videos")
        self.logs_dir = os.path.join(self.results_dir, "logs")

        # Create directory structure
        for directory in [self.new_dir, self.results_dir, self.frames_dir,
                         self.responses_dir, self.prompts_dir,
                         self.videos_dir, self.logs_dir]:
            os.makedirs(directory, exist_ok=True)

        # Set up comprehensive logging
        self.setup_logging()

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

        self.model_id = model_id

        # API configuration
        self.api_url = "https://openrouter.ai/api/v1/chat/completions"

        # Initialize Bedrock client if using Bedrock
        if self.provider == 'bedrock':
            try:
                from aws_model import BedrockUnifiedClient
                self.bedrock_client = BedrockUnifiedClient(region_name=aws_region)
            except Exception as e:
                print(f"Warning: Failed to initialize Bedrock client: {e}")
                self.bedrock_client = None

        # Skip test if already done
        state_file = os.path.join(self.new_dir, f"env_{base}_state.pkl")
        if os.path.exists(state_file):
            print(f"\n\nEnvironment '{env_name}' already processed—skipping.\n\n")
            return

        # Build environment and video recorder
        gym.register_envs(ale_py)
        env = gym.make(env_name, render_mode="rgb_array")
        env = OrderEnforcing(env, disable_render_order_enforcing=True)

        # Set seed if provided
        if self.seed is not None:
            env.reset(seed=self.seed)
            np.random.seed(self.seed)
        else:
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
            self.direct_frame_rollout()

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
        self.logger = logging.getLogger(f"DirectFrame_{self.display_name}_{self.game_type}")
        self.logger.setLevel(logging.DEBUG)

        # Clear any existing handlers
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)

        # Create file handler
        log_file = os.path.join(self.logs_dir, f"direct_frame_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
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
        self.logger.info("="*50 + " NEW DIRECT FRAME SESSION " + "="*50)
        self.logger.info(f"Environment: {self.env_name}")
        self.logger.info(f"Game Type: {self.game_type}")
        self.logger.info(f"Provider: {self.provider}")
        self.logger.info(f"Output directory: {self.new_dir}")
        self.logger.info(f"Number of frames: {self.num_timesteps}")
        self.logger.info(f"Skip initial frames: {self.skip_initial_frames}")
        self.logger.info(f"AWS Region: {self.aws_region}")
        self.logger.info(f"Seed: {self.seed if self.seed is not None else 'None (random)'}")

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
                1: "FIRE (serve/start ball - rarely used)",
                2: "RIGHT/UP (move paddle up)",
                3: "LEFT/DOWN (move paddle down)",
                4: "RIGHTFIRE/UPFIRE (move up and fire)",
                5: "LEFTFIRE/DOWNFIRE (move down and fire)"
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
        """Get game-specific prompt with direct frame analysis"""
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
        elif self.game_type == "pong":
            game_intro = """You are an expert Pong player controlling the GREEN PADDLE.

IMPORTANT: You are controlling the GREEN PADDLE."""
        elif self.game_type == "breakout":
            game_intro = """You are an expert Breakout player controlling the ORANGE PADDLE at the bottom.

IMPORTANT: You are controlling the ORANGE PADDLE at the bottom."""
        else:
            game_intro = "You are an expert game player analyzing a game frame."

        # Create direct frame prompt without symbolic information
        prompt = """{game_intro}

Game controls:
{controls_text}

Analyze the current game frame image and choose the best action based on what you see.

""".format(
            game_intro=game_intro,
            controls_text=controls_text
        )

        # Add game-specific strategy instructions
        if self.game_type == "tennis":
            strategy_section = """
As an expert Tennis player controlling the RED PLAYER, analyze the visual scene and choose the optimal action.

Think step by step:
1. Observe the current state of the game
2. Predict the trajectory or movement patterns
3. Consider your strategic options
4. Choose the optimal action

Return ONLY JSON:
{
    "reasoning": "your expert analysis and decision rationale",
    "action": integer_action_code
}
"""
        elif self.game_type == "pong":
            strategy_section = """
As an expert Pong player controlling the GREEN PADDLE, analyze the visual scene and choose the optimal action.

Think step by step:
1. Observe the current state of the game
2. Predict the trajectory or movement patterns
3. Consider your strategic options
4. Choose the optimal action

Return ONLY JSON:
{
    "reasoning": "your expert analysis and decision rationale",
    "action": integer_action_code
}
"""
        elif self.game_type == "breakout":
            strategy_section = """
As an expert Breakout player controlling the ORANGE PADDLE at the bottom, analyze the visual scene and choose the optimal action.

Think step by step:
1. Observe the current state of the game
2. Predict the trajectory or movement patterns
3. Consider your strategic options
4. Choose the optimal action

Return ONLY JSON:
{
    "reasoning": "your expert analysis and decision rationale",
    "action": integer_action_code
}
"""
        else:
            strategy_section = """
As an expert player, analyze the visual scene and choose the optimal action.

Think step by step:
1. Observe the current state of the game
2. Predict the trajectory or movement patterns
3. Consider your strategic options
4. Choose the optimal action

Return ONLY JSON:
{
    "reasoning": "your expert analysis and decision rationale",
    "action": integer_action_code
}
"""

        prompt += strategy_section
        return prompt

    def encode_image_to_base64(self, image_path: str) -> str:
        """Convert image to base64 string"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def _make_api_call(self, messages: list, max_tokens: int = 1000) -> str:
        """Make API call to OpenRouter or Bedrock"""
        # Use Bedrock if provider is Bedrock
        if self.provider == 'bedrock' and hasattr(self, 'bedrock_client') and self.bedrock_client:
            try:
                # Convert messages to Bedrock format
                bedrock_messages = []
                for msg in messages:
                    content = msg["content"]
                    if isinstance(content, str):
                        bedrock_messages.append({
                            "role": msg["role"],
                            "content": [{"text": content}]
                        })
                    else:
                        # Handle complex content (images, etc.)
                        bedrock_content = []
                        for item in content:
                            if "text" in item:
                                bedrock_content.append({"text": item["text"]})
                            elif "image" in item and "source" in item["image"]:
                                # Claude format
                                image_data = item["image"]["source"]["data"]
                                bedrock_content.append({
                                    "image": {
                                        "format": "jpeg",
                                        "source": {"bytes": base64.b64decode(image_data)}
                                    }
                                })
                            elif "image_url" in item:
                                # OpenAI format - extract base64 data
                                image_url = item["image_url"]["url"]
                                if image_url.startswith("data:image"):
                                    base64_data = image_url.split(",", 1)[1] if "," in image_url else image_url
                                    bedrock_content.append({
                                        "image": {
                                            "format": "jpeg",
                                            "source": {"bytes": base64.b64decode(base64_data)}
                                        }
                                    })
                        bedrock_messages.append({
                            "role": msg["role"],
                            "content": bedrock_content
                        })

                # Make Bedrock API call
                response = self.bedrock_client.chat_completion(
                    model=self.model_id,
                    messages=bedrock_messages,
                    max_tokens=max_tokens,
                    temperature=0
                )

                if 'choices' not in response or not response['choices']:
                    return None

                return response['choices'][0]['message']['content']
            except Exception as e:
                self.logger.error(f"Bedrock API request failed: {e}")
                return None
        else:
            # Use OpenRouter (default)
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }

            payload = {
                "model": self.model_id,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": 0
            }

            try:
                response = requests.post(self.api_url, headers=headers, json=payload)
                response.raise_for_status()
                result = response.json()

                if 'choices' not in result or not result['choices']:
                    return None

                return result['choices'][0]['message']['content']
            except requests.exceptions.RequestException as e:
                self.logger.error(f"OpenRouter API request failed: {e}")
                return None

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

    def save_frame_and_response(self, frame, action, reasoning, step_number, final_prompt=None):
        """Save frame and response with comprehensive logging"""
        # Save frame
        frame_path = os.path.join(self.frames_dir, f"frame_{step_number:04d}.jpg")
        cv2.imwrite(frame_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        # Save response with additional metadata
        response_data = {
            "step": int(step_number),
            "action": int(action),
            "reasoning": reasoning,
            "analysis_type": "direct_frame",
            "timestamp": datetime.now().isoformat(),
            "cumulative_reward": float(self.rewards),
            "frame_path": frame_path
        }

        response_path = os.path.join(self.responses_dir, f"response_{step_number:04d}.json")
        with open(response_path, 'w') as f:
            json.dump(response_data, f, indent=2)

        # Save prompt information
        if final_prompt is not None:
            self.save_prompts_and_api_info(step_number, self.base_prompt, final_prompt)

        # Log the decision
        self.logger.debug(f"STEP {step_number:04d}: Action={action}, Reward={self.rewards}")
        self.logger.debug(f"REASONING: {reasoning}")

        # Log action details
        self.logger.info(f"Step {step_number}: Action {action} - Reward: {self.rewards}")

        print(f"Saved frame and response for step {step_number} (direct frame)")

    def save_prompts_and_api_info(self, step_number, base_prompt, final_prompt=None):
        """Save all prompt information for analysis"""
        prompt_info = {
            "step": step_number,
            "base_prompt": base_prompt,
            "final_prompt_sent_to_api": final_prompt,
            "timestamp": datetime.now().isoformat()
        }

        prompt_file = os.path.join(self.prompts_dir, f"prompt_{step_number:04d}.json")
        with open(prompt_file, 'w') as f:
            json.dump(prompt_info, f, indent=2)

        self.logger.debug(f"Step {step_number}: Saved prompt information")

    def get_action_from_direct_frame(self, frame_path, step_number):
        """Get action from direct frame analysis using internal API client"""
        try:
            # Encode the frame
            img_b64 = self.encode_image_to_base64(frame_path)

            # Prepare the prompt for the LLM
            prompt = self.base_prompt

            # Prepare messages based on provider
            if self.provider.lower() == 'claude' or self.provider.lower() == 'bedrock':
                messages = [{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": img_b64}}
                    ]
                }]
            else:
                # For OpenAI-style APIs (OpenRouter)
                messages = [{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}}
                    ]
                }]

            # Make API call
            raw_response = self._make_api_call(messages, max_tokens=1000)

            if not raw_response:
                return 0, "Failed to get response from API", prompt

            # Parse the JSON response
            match = re.search(r"\{.*\}", raw_response, flags=re.DOTALL)
            if match:
                payload = match.group(0)
                try:
                    obj = json.loads(payload)
                    action = int(obj.get('action', 0))

                    # Validate action is within valid range based on game
                    game_action_ranges = {
                        'breakout': 5,
                        'frogger': 4,
                        'space_invaders': 5,
                        'pacman': 4,
                        'mspacman': 4,
                        'pong': 5,
                        'tennis': 5,
                        'assault': 5
                    }
                    max_action = game_action_ranges.get(self.game_type, 5)
                    if action < 0 or action > max_action:
                        self.logger.warning(f"Invalid action {action} for game {self.game_type}, clipping to 0 (NOOP)")
                        action = 0  # Default to no-op

                    reasoning = obj.get('reasoning', 'No reasoning provided')

                    self.logger.debug(f"Step {step_number}: LLM returned action={action}, reasoning={reasoning}")
                    return action, reasoning, prompt

                except json.JSONDecodeError as e:
                    self.logger.warning(f"JSON decode error: {e}, raw response: {raw_response}")
                    return 0, f"JSON decode error in response: {raw_response[:200]}...", prompt
            else:
                self.logger.warning(f"No JSON found in response: {raw_response}")
                return 0, f"No valid JSON found in response: {raw_response[:200]}...", prompt

        except Exception as e:
            self.logger.error(f"Error getting action from direct frame analysis: {e}")
            return 0, f"Error in direct frame analysis: {e}", self.base_prompt

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

    def direct_frame_rollout(self):
        """Direct frame analysis rollout - processes every frame after skip period"""

        self.logger.info("Starting direct frame rollout")
        self.logger.info(f"Skip first {self.skip_initial_frames} frames, then process every frame")

        obs, info = self.env.reset()
        self.save_states(self.rewards, 0)
        pbar = tqdm(total=self.num_timesteps, desc=f"{self.display_name} {self.game_type.title()} Direct Frame ({self.temp_env_name})", unit="step")

        for step in range(self.num_timesteps - self.steps_taken):
            # Skip the initial frames for decision making, then process every frame
            if step < self.skip_initial_frames:
                # No-op for initial frames
                action = 0
                reasoning = f"NOOP for initial frame {step} (skipping first {self.skip_initial_frames})"
                final_prompt = None
                self.logger.debug(f"Step {step}: NOOP (skip initial frame)")
            else:
                # Get current frame for processing
                frame = self.env.render()
                if frame is not None:
                    # Create temporary frame for processing (don't save to frames dir)
                    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
                        temp_frame_path = temp_file.name
                    cv2.imwrite(temp_frame_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

                    try:
                        # Get action from direct frame analysis
                        self.logger.debug(f"Step {step}: Processing frame with direct analysis...")
                        action, reasoning, final_prompt = self.get_action_from_direct_frame(temp_frame_path, self.steps_taken)

                        # Log the results
                        self.logger.debug(f"Step {step}: Direct frame analysis returned action={action}")

                    finally:
                        # Clean up temp file
                        try:
                            os.unlink(temp_frame_path)
                        except:
                            pass
                else:
                    action = 0  # NOOP
                    reasoning = "No frame available for direct frame processing"
                    final_prompt = None

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
            self.save_frame_and_response(frame, action, reasoning, self.steps_taken, final_prompt)

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
    parser = argparse.ArgumentParser(description="Run Atari Games with Direct Frame Analysis Only")
    parser.add_argument("--game", type=str, default="ALE/Breakout-v5", help="Game name")
    parser.add_argument("--game_type", type=str, choices=["breakout", "frogger", "space_invaders", "pacman", "mspacman", "pong", "tennis", "assault"],
                        required=True, help="Game type for pipeline")
    parser.add_argument("--provider", type=str, required=True, help="Model provider (openai, gemini, claude, openrouter, bedrock)")
    parser.add_argument("--model_name", type=str, required=True, help="Specific model name")
    parser.add_argument("--output_dir", default="./experiments", help="Output directory")
    parser.add_argument("--api_key_file", type=str, default="API_KEY.txt",
                        help="Path to API key file")
    parser.add_argument("--num_frames", type=int, default=600, help="Number of frames to run (default: 600)")
    parser.add_argument("--aws_region", type=str, default="us-east-1", help="AWS region for Bedrock (default: us-east-1)")
    parser.add_argument("--disable_history", action="store_true", help="Disable history mechanism (conversation memory)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducible runs (default: None for random)")

    args = parser.parse_args()

    # Read API key from file if it exists
    api_key = None
    if os.path.exists(args.api_key_file):
        with open(args.api_key_file, 'r') as f:
            api_key = f.read().strip()
        print(f"Loaded API key from {args.api_key_file}")
    else:
        print(f"Warning: API key file {args.api_key_file} not found")

    # Run game with direct frame analysis only
    DirectFrameRunner(
        env_name=args.game,
        provider=args.provider,
        game_type=args.game_type,
        model_id=args.model_name,
        output_dir=args.output_dir,
        api_key=api_key,
        num_frames=args.num_frames,
        aws_region=args.aws_region,
        disable_history=args.disable_history,
        seed=args.seed
    )


if __name__ == "__main__":
    main()