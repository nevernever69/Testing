"""
Pipeline adapters for integrating existing game runners with the benchmark.
Wraps direct_frame_runner.py and advance_game_runner.py for evaluation.
"""

import os
import json
import tempfile
import cv2
import numpy as np
from typing import Dict, Any, Optional


class PipelineAdapter:
    """Base adapter class for pipelines."""

    def __init__(self, pipeline_name: str):
        self.pipeline_name = pipeline_name
        self.last_actual_prompt = None  # Store the actual prompt sent to VLM
        self.last_detection_results = None  # Store detection results (for Vision+Symbol)

    def process(self, frame: np.ndarray, prompt: str) -> str:
        """
        Process a frame with a prompt and return VLM response.

        Args:
            frame: RGB frame array (210, 160, 3)
            prompt: Task prompt

        Returns:
            VLM response text
        """
        raise NotImplementedError

    def get_actual_prompt(self) -> str:
        """Get the actual prompt that was sent to the VLM (may include symbolic info)."""
        return self.last_actual_prompt or "Prompt not available"

    def get_detection_results(self) -> Optional[Dict[str, Any]]:
        """Get the last detection results (for Vision+Symbol pipelines)."""
        return self.last_detection_results


class DirectFrameAdapter(PipelineAdapter):
    """
    Adapter for direct_frame_runner.py (Vision-Only pipeline).
    Sends frame directly to VLM without symbolic information.
    """

    def __init__(self, provider: str = 'bedrock', model_id: str = None,
                 api_key: str = None, aws_region: str = 'us-east-1'):
        """
        Initialize Vision-Only adapter.

        Args:
            provider: 'openai', 'anthropic', 'bedrock', etc.
            model_id: Model identifier
            api_key: API key (if needed)
            aws_region: AWS region for Bedrock
        """
        super().__init__("Vision-Only")
        self.provider = provider
        self.model_id = model_id
        self.api_key = api_key
        self.aws_region = aws_region

        # Initialize the appropriate client
        if provider == 'bedrock':
            try:
                from aws_model import BedrockUnifiedClient
                self.client = BedrockUnifiedClient(region_name=aws_region)
            except Exception as e:
                print(f"Warning: Failed to initialize Bedrock client: {e}")
                self.client = None
        elif provider == 'openai':
            try:
                import openai
                self.client = openai.OpenAI(api_key=api_key)
            except Exception as e:
                print(f"Warning: Failed to initialize OpenAI client: {e}")
                self.client = None
        elif provider == 'anthropic':
            try:
                import anthropic
                self.client = anthropic.Anthropic(api_key=api_key)
            except Exception as e:
                print(f"Warning: Failed to initialize Anthropic client: {e}")
                self.client = None
        else:
            raise ValueError(f"Unsupported provider: {provider}")

    def process(self, frame: np.ndarray, prompt: str) -> str:
        """
        Process frame with vision-only (no symbolic info).

        Args:
            frame: RGB frame array, should already be scaled to 1280x720
            prompt: Task prompt
        """
        try:
            # Store the actual prompt (same as input for vision-only)
            self.last_actual_prompt = prompt

            # Convert frame to base64 for API
            import base64
            from io import BytesIO
            from PIL import Image

            # Frame should already be scaled to 1280x720 by caller
            # Convert numpy array to PIL Image
            pil_image = Image.fromarray(frame)

            # Convert to base64
            buffered = BytesIO()
            pil_image.save(buffered, format="PNG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

            # Call appropriate API
            if self.provider == 'bedrock':
                response = self._call_bedrock(img_base64, prompt)
            elif self.provider == 'openai':
                response = self._call_openai(img_base64, prompt)
            elif self.provider == 'anthropic':
                response = self._call_anthropic(img_base64, prompt)
            else:
                response = "Unsupported provider"

            return response

        except Exception as e:
            return f"ERROR: {str(e)}"

    def _call_bedrock(self, img_base64: str, prompt: str) -> str:
        """Call AWS Bedrock."""
        if not self.client:
            return "ERROR: Bedrock client not initialized"

        try:
            import base64

            # Use Claude 4 Sonnet as default vision model
            model_id = self.model_id or 'claude-4-sonnet'

            # Decode base64 to bytes for Bedrock converse API
            image_bytes = base64.b64decode(img_base64)

            response = self.client.chat_completion(
                model=model_id,
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "image": {
                                "format": "png",
                                "source": {
                                    "bytes": image_bytes
                                }
                            }
                        },
                        {
                            "text": prompt
                        }
                    ]
                }],
                temperature=0.0,
                max_tokens=500
            )
            return response['choices'][0]['message']['content']
        except Exception as e:
            return f"ERROR: Bedrock call failed: {str(e)}"

    def _call_openai(self, img_base64: str, prompt: str) -> str:
        """Call OpenAI Vision API."""
        if not self.client:
            return "ERROR: OpenAI client not initialized"

        try:
            response = self.client.chat.completions.create(
                model=self.model_id or "gpt-4-vision-preview",
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{img_base64}"
                            }
                        },
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                }],
                temperature=0.0,
                max_tokens=500
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"ERROR: OpenAI call failed: {str(e)}"

    def _call_anthropic(self, img_base64: str, prompt: str) -> str:
        """Call Anthropic Vision API."""
        if not self.client:
            return "ERROR: Anthropic client not initialized"

        try:
            response = self.client.messages.create(
                model=self.model_id or "claude-3-5-sonnet-20241022",
                max_tokens=500,
                temperature=0.0,
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": img_base64
                            }
                        },
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                }]
            )
            return response.content[0].text
        except Exception as e:
            return f"ERROR: Anthropic call failed: {str(e)}"


class AdvancedGameAdapter(PipelineAdapter):
    """
    Adapter for advance_game_runner.py (Vision+Symbol pipeline).
    First extracts symbolic info via detection, then reasons with it.
    """

    def __init__(self, provider: str = 'bedrock', model_id: str = None,
                 openrouter_api_key: str = None, detection_model: str = None,
                 aws_region: str = 'us-east-1', game_type: str = 'pong'):
        """
        Initialize Vision+Symbol adapter.

        Args:
            provider: Provider for reasoning VLM
            model_id: Model for reasoning
            openrouter_api_key: API key for detection model
            detection_model: Model for object detection
            aws_region: AWS region
            game_type: Type of game for detector initialization
        """
        super().__init__("Vision+Symbol")
        self.provider = provider
        self.model_id = model_id
        self.openrouter_api_key = openrouter_api_key
        self.detection_model = detection_model or "anthropic/claude-sonnet-4"
        self.aws_region = aws_region
        self.game_type = game_type.lower()

        # Initialize detector
        self.detector = self._init_detector()

        # Initialize reasoning client (same as DirectFrameAdapter)
        if provider == 'bedrock':
            try:
                from aws_model import BedrockUnifiedClient
                self.client = BedrockUnifiedClient(region_name=aws_region)
            except Exception as e:
                print(f"Warning: Failed to initialize Bedrock client: {e}")
                self.client = None
        else:
            self.client = None

    def _init_detector(self):
        """Initialize the game-specific detector."""
        try:
            from advanced_zero_shot_pipeline import (
                PongAdvancedDetector,
                BreakoutAdvancedDetector,
                SpaceInvadersAdvancedDetector
            )

            detectors = {
                'pong': PongAdvancedDetector,
                'breakout': BreakoutAdvancedDetector,
                'space_invaders': SpaceInvadersAdvancedDetector,
                'spaceinvaders': SpaceInvadersAdvancedDetector
            }

            detector_class = detectors.get(self.game_type)
            if detector_class:
                # Use Bedrock for detection if provider is Bedrock
                if self.provider == 'bedrock':
                    # Use Claude 4 Sonnet for detection
                    bedrock_detection_model = self.detection_model if self.detection_model else 'claude-4-sonnet'
                    return detector_class(
                        openrouter_api_key=self.openrouter_api_key or 'dummy',  # Not used for Bedrock
                        model_name=bedrock_detection_model,
                        detection_mode='specific',
                        provider='bedrock',
                        aws_region=self.aws_region,
                        disable_history=True
                    )
                elif self.openrouter_api_key:
                    # Use OpenRouter for detection
                    return detector_class(
                        self.openrouter_api_key,
                        self.detection_model or 'anthropic/claude-sonnet-4',
                        'specific',
                        provider='openrouter',
                        disable_history=True
                    )
            return None
        except Exception as e:
            print(f"Warning: Failed to initialize detector: {e}")
            return None

    def process(self, frame: np.ndarray, prompt: str) -> str:
        """
        Process frame with symbolic detection first, then reasoning.

        Args:
            frame: RGB frame array, should already be scaled to 1280x720
            prompt: Task prompt
        """
        if not self.detector:
            return "ERROR: Detector not initialized"

        try:
            # Detect task type from prompt
            task_type = self._detect_task_type(prompt)

            # Frame should already be scaled to 1280x720 by caller
            # Save frame to temporary file for detector
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                temp_path = tmp.name

            # Save frame
            from PIL import Image
            pil_image = Image.fromarray(frame)
            pil_image.save(temp_path)

            # Run detection ONLY (not full game-playing pipeline)
            detection_results = self.detector.detect_objects(temp_path, self.game_type.title())

            # Store detection results for logging
            self.last_detection_results = detection_results

            # Convert detection results to symbolic state format
            symbolic_state = self._convert_detection_to_symbolic(detection_results)

            # Build response that includes symbolic information
            # Pass frame so VLM can verify visually
            response = self._format_symbolic_response(symbolic_state, prompt, task_type, frame)

            # Clean up temp file
            try:
                os.unlink(temp_path)
            except:
                pass

            return response

        except Exception as e:
            return f"ERROR: {str(e)}"

    def _convert_detection_to_symbolic(self, detection_results: Dict) -> Dict[str, Any]:
        """
        Convert raw detection results to symbolic state format.

        Args:
            detection_results: Output from detector.detect_objects()
                {
                    "objects": [
                        {
                            "label": "Ball",
                            "coordinates": [x1, y1, x2, y2],
                            "confidence": 0.95
                        }
                    ]
                }

        Returns:
            Symbolic state dict with object positions
        """
        if "error" in detection_results:
            return {}

        objects = detection_results.get("objects", [])
        symbolic_state = {'objects': []}

        for obj in objects:
            label = obj.get('label', 'Unknown')
            coords = obj.get('coordinates', [0, 0, 0, 0])

            # Calculate center position and size
            if len(coords) == 4:
                x1, y1, x2, y2 = coords
                x_center = int((x1 + x2) / 2)
                y_center = int((y1 + y2) / 2)
                width = int(x2 - x1)
                height = int(y2 - y1)

                symbolic_state['objects'].append({
                    'label': label,
                    'x': x_center,
                    'y': y_center,
                    'width': width,
                    'height': height,
                    'confidence': obj.get('confidence', 0.95)
                })

        return symbolic_state

    def _detect_task_type(self, prompt: str) -> str:
        """Detect task type from prompt."""
        prompt_lower = prompt.lower()
        if "identify all key elements" in prompt_lower or "key elements in this image" in prompt_lower:
            return "visual"
        elif "where are" in prompt_lower or "located relative" in prompt_lower:
            return "spatial"
        elif "ideal next move" in prompt_lower or "strategy" in prompt_lower:
            return "strategy"
        elif "identify the game name" in prompt_lower or "what game" in prompt_lower:
            return "identification"
        return "unknown"

    def _format_symbolic_response(self, symbolic_state: Dict[str, Any], prompt: str, task_type: str = "unknown", frame=None) -> str:
        """Use reasoning VLM to answer prompt based on symbolic state with task-specific guidance."""
        if not symbolic_state:
            return "No objects detected"

        # Build symbolic context
        context_parts = []

        if 'objects' in symbolic_state and isinstance(symbolic_state['objects'], list):
            for obj in symbolic_state['objects']:
                label = obj.get('label', 'unknown')
                x = obj.get('x', 'unknown')
                y = obj.get('y', 'unknown')
                w = obj.get('width', 'unknown')
                h = obj.get('height', 'unknown')
                context_parts.append(f"- {label}: position (x={x}, y={y}), size ({w}x{h})")
        else:
            for obj_label, obj_data in symbolic_state.items():
                if isinstance(obj_data, dict):
                    x = obj_data.get('x', 'unknown')
                    y = obj_data.get('y', 'unknown')
                    context_parts.append(f"- {obj_label}: position (x={x}, y={y})")

        if not context_parts:
            return "Symbolic detection completed but no clear objects identified"

        symbolic_context = "\n".join(context_parts)

        # Task-specific reasoning instructions
        task_instructions = {
            "visual": """Focus on IDENTIFYING and DESCRIBING each object:
- First verify the detections match what you see in the frame visually
- What type of object is it? (paddle, ball, score, etc.)
- What are its visual characteristics?
- Describe WHERE each object is located in the frame using detailed qualitative descriptions:
  * Use combinations like: "upper left", "lower right", "middle right", "slightly below center"
  * Be precise: "just below the middle", "near the top-left corner", "in the upper-right portion"
  * Avoid simple "top/bottom/left/right" - be more nuanced
- Do NOT just repeat coordinates - translate them into natural spatial language
- Be specific about what you see in the frame.""",

            "spatial": """Focus on SPATIAL RELATIONSHIPS between the DETECTED GAME OBJECTS:
- First verify the detected objects match what you see in the frame visually
- Use the provided coordinates AND visual observation to describe RELATIVE POSITIONS
- For EACH PAIR of game objects (paddles, ball, enemies), describe:
  * Horizontal relationship: "far to the left of", "slightly left of", "directly left of", "to the right of"
  * Vertical relationship: "well above", "just above", "at same height as", "below"
  * Distance/proximity: "very close to", "moderately separated from", "far apart from"
- Compare ALL game object pairs systematically (ball vs paddle, paddle vs paddle, etc.)
- Express relationships in natural, nuanced language
- Stay focused on RELATIONSHIPS between objects (not describing individual object appearance)
- Be precise with directional and distance descriptions.""",

            "strategy": """Focus on GAMEPLAY STRATEGY:
- Based on object positions, what should the player do next?
- Which direction should paddles/ships move?
- Should the player shoot, move up, move down, or wait?
- Explain WHY this is the optimal move based on coordinates.""",

            "identification": """Focus on GAME IDENTIFICATION:
- Based on the detected objects and layout, what game is this?
- What specific features identify this game?
- State the game name clearly and justify your answer."""
        }

        instruction = task_instructions.get(task_type, "Answer the question based on the symbolic information.")

        reasoning_prompt = f"""You have detected objects in a game frame with precise coordinates:

{symbolic_context}

TASK: {prompt}

SPECIFIC INSTRUCTIONS FOR THIS TASK:
{instruction}

Answer concisely (under 100 words) using the coordinate data to support your answer."""

        # Store the actual prompt sent to VLM
        self.last_actual_prompt = reasoning_prompt

        # Call reasoning VLM with BOTH frame image and symbolic coordinates
        try:
            if self.provider == 'bedrock' and self.client:
                model_id = self.model_id or 'claude-4-sonnet'

                # Prepare message content with image + text
                message_content = []

                # Add frame image if available
                if frame is not None:
                    import base64
                    from io import BytesIO
                    from PIL import Image

                    # Convert frame to base64
                    pil_img = Image.fromarray(frame)
                    buffered = BytesIO()
                    pil_img.save(buffered, format="PNG")
                    img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

                    message_content.append({
                        "image": {
                            "format": "png",
                            "source": {"bytes": base64.b64decode(img_base64)}
                        }
                    })

                # Add text prompt with coordinates
                message_content.append({"text": reasoning_prompt})

                response = self.client.chat_completion(
                    model=model_id,
                    messages=[{
                        "role": "user",
                        "content": message_content
                    }],
                    temperature=0.0,
                    max_tokens=500
                )
                return response['choices'][0]['message']['content']
            else:
                # Fallback: just return symbolic info
                return "Based on symbolic detection: " + ", ".join([p.replace("- ", "") for p in context_parts])
        except Exception as e:
            print(f"Warning: Reasoning VLM call failed: {e}")
            return "Based on symbolic detection: " + ", ".join([p.replace("- ", "") for p in context_parts])
