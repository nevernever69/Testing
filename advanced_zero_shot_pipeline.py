import cv2
import os
import json
import base64
import requests
from PIL import Image, ImageDraw, ImageFont
from typing import List, Dict, Optional
import argparse
from datetime import datetime


class AdvancedSymbolicDetector:
    def __init__(self, openrouter_api_key: str, model_name: str = "anthropic/claude-sonnet-4", detection_mode: str = "specific", provider: str = "openrouter", aws_region: str = "us-east-1", disable_history: bool = False):
        """
        Initialize the Advanced Symbolic Detector with CoT, Planning, and Memory modules
        """
        self.provider = provider.lower()
        self.aws_region = aws_region
        self.api_key = openrouter_api_key
        self.model_name = model_name
        self.api_url = "https://openrouter.ai/api/v1/chat/completions"
        self.scaled_width = 1280
        self.scaled_height = 720
        self.prompts_dir = None  # Will be set when processing frames
        self.conversation_history = []  # For last 4 reasoning and actions
        self.detection_mode = detection_mode  # "specific" or "generic"
        self.disable_history = disable_history  # Flag to disable history mechanism
        
        # Initialize Bedrock client if using Bedrock
        if self.provider == 'bedrock':
            try:
                from aws_model import BedrockUnifiedClient
                self.bedrock_client = BedrockUnifiedClient(region_name=aws_region)
            except Exception as e:
                print(f"Warning: Failed to initialize Bedrock client: {e}")
                self.bedrock_client = None
        
    def encode_image_to_base64(self, image_path: str) -> str:
        """Convert image to base64 string"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    
    def scale_image(self, image_path: str, output_path: str) -> str:
        """
        Scale image to 1280x720 and save to output_path
        Returns the path to the scaled image
        """
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image from {image_path}")
        
        # Scale to 1280x720
        scaled_image = cv2.resize(image, (self.scaled_width, self.scaled_height), interpolation=cv2.INTER_LINEAR)
        cv2.imwrite(output_path, scaled_image)
        return output_path
    
    def _make_api_call(self, messages: List[Dict], max_tokens: int = 1000, call_id: str = "detection") -> Optional[str]:
        """Make API call to OpenRouter or Bedrock and save the prompt"""
        # Save the prompt that will be used for this API call
        if self.prompts_dir:
            prompt_text = messages[0]["content"][0]["text"] if isinstance(messages[0]["content"], list) else messages[0]["content"]
            prompt_file = os.path.join(self.prompts_dir, f"api_prompt_{call_id}.txt")
            with open(prompt_file, 'w') as f:
                f.write(prompt_text)
            print(f"Saved API prompt to: {prompt_file}")
        
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
                            elif "image_url" in item:
                                # Extract base64 data
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
                    model=self.model_name,
                    messages=bedrock_messages,
                    max_tokens=max_tokens,
                    temperature=0
                )
                
                if 'choices' not in response or not response['choices']:
                    return None

                return response['choices'][0]['message']['content']
            except Exception as e:
                print(f"Bedrock API request failed: {e}")
                return None
        else:
            # Use OpenRouter (default)
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": self.model_name,
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
                print(f"OpenRouter API request failed: {e}")
                return None
    
    def detect_objects(self, image_path: str, game_name: str) -> Dict:
        """
        Detects objects in a frame for a specified game.
        """
        print(f"  → Step 2: Object detection with VLM (game: {game_name})...")
        base64_image = self.encode_image_to_base64(image_path)

        # Add game-specific instructions for better detection
        game_specific_info = """

Before detecting objects, follow these steps:
1. First, visually identify what game this is based on the gameplay elements and visual style
2. Based on your knowledge of this game, recall what objects typically exist (e.g., paddles, balls, bricks, enemies)
3. Determine how many of each object type should reasonably be present in a typical frame
4. Now, carefully verify visually which of those expected objects are actually present in THIS specific frame
5. Only report objects that you can clearly see AND that match the expected object types for this game

This ensures your detections are grounded in both visual evidence and game knowledge."""

        # Game-specific can be expanded here if needed
        if game_name.lower() == "breakout":
            pass  # Using default knowledge-grounded instructions
        elif game_name.lower() == "pong":
            pass  # Using default knowledge-grounded instructions
        elif game_name.lower() in ["space invaders", "spaceinvaders"]:
            pass  # Using default knowledge-grounded instructions

        prompt = f"""You are an expert game frame analyzer for the game {game_name}.{game_specific_info}

Your task is to detect ALL visible objects in the image with high precision. Detect all distinct, visible objects (like players, enemies, projectiles, items, scores). For each object, provide its label, a tight bounding box [x1, y1, x2, y2], and a confidence score.

Return ONLY valid JSON in the following format:
{{
    "objects": [
        {{
            "id": "unique_id",
            "label": "object_type_or_description",
            "coordinates": [x1, y1, x2, y2],
            "confidence": 0.95,
            "description": "brief description of the object"
        }}
    ],
    "image_info": {{
        "total_objects": 0,
        "frame_analysis": "brief description of what you see in the frame"
    }}
}}
"""
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ]
        
        response = self._make_api_call(messages, max_tokens=4096, call_id="detection")
        
        if not response:
            return {"error": "api_error"}

        try:
            json_str = self._clean_json_response(response)
            result = json.loads(json_str)

            # Basic validation
            if "objects" not in result:
                return {"error": "invalid_response_format"}

            # Filter detections by confidence
            filtered_objects = [
                obj for obj in result.get("objects", [])
                if obj.get("confidence", 0) >= 0.7 and self._validate_detection(obj)
            ]
            result["objects"] = filtered_objects
            result.get("image_info", {})["total_objects"] = len(filtered_objects)
            result["full_api_response"] = response

            return result

        except (json.JSONDecodeError, ValueError) as e:
            # Try parsing markdown format (Bedrock Llama models)
            print(f"JSON parsing failed, trying markdown parser...")
            try:
                result = self._parse_markdown_detection(response)
                if result and "objects" in result and len(result["objects"]) > 0:
                    print(f"✅ Markdown parsing successful! Found {len(result['objects'])} objects")
                    return result
            except Exception as markdown_error:
                print(f"Markdown parsing also failed: {markdown_error}")

            print(f"Error parsing detection response: {e}")
            print(f"Response was: {response[:500]}...")
            return {"error": "parsing_error"}
    
    def draw_bounding_boxes(self, image_path: str, detections: Dict, output_path: str) -> None:
        """
        Draw bounding boxes on image and save it
        """
        image = Image.open(image_path)
        image_width, image_height = image.size
        draw = ImageDraw.Draw(image)
        
        try:
            font = ImageFont.truetype("arial.ttf", 16)
        except:
            font = ImageFont.load_default()
        
        # Use different colors for different objects
        colors = ['red', 'blue', 'green', 'yellow', 'purple', 'orange', 'pink', 'brown', 'cyan', 'magenta']
        
        for i, obj in enumerate(detections.get('objects', [])):
            label = obj.get('label', 'unknown')
            coords = obj.get('coordinates', [])
            confidence = obj.get('confidence', 0.0)
            
            if len(coords) == 4:
                x1, y1, x2, y2 = coords
                
                # Convert coordinates to image space if they're normalized (0-1 range)
                if all(c <= 1.0 for c in coords):
                    # Normalized coordinates - convert to pixel coordinates
                    x1, x2 = x1 * image_width, x2 * image_width
                    y1, y2 = y1 * image_height, y2 * image_height
                
                color = colors[i % len(colors)]
                
                # Draw bounding box
                draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
                
                # Prepare label text
                text = f"{label} ({confidence:.2f})"
                
                # Draw label background and text
                bbox = draw.textbbox((x1, y1-20), text, font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
                
                # Background rectangle
                draw.rectangle([x1, y1-20, x1 + text_width + 4, y1], fill=color, outline=color)
                
                # Text
                draw.text((x1 + 2, y1-18), text, fill='black', font=font)
        
        image.save(output_path)
    
    def generate_symbolic_state(self, detections: Dict) -> Dict:
        """
        Generate symbolic representation from detections
        """
        # Extract objects
        objects = detections.get("objects", [])
        
        # Initialize state with object information
        state = {
            "objects": [],
            "total_objects": len(objects)
        }
        
        # Extract object positions and basic info
        for i, obj in enumerate(objects):
            coords = obj.get("coordinates", [])
            if coords:
                # Determine if coordinates are normalized (0-1) or pixel coordinates
                are_normalized = all(c <= 1.0 for c in coords)
                
                if are_normalized:
                    # Convert normalized coordinates to pixel coordinates (1280x720)
                    center_x = int((coords[0] + coords[2]) / 2 * self.scaled_width)
                    center_y = int((coords[1] + coords[3]) / 2 * self.scaled_height)
                    width = int((coords[2] - coords[0]) * self.scaled_width)
                    height = int((coords[3] - coords[1]) * self.scaled_height)
                else:
                    # Use pixel coordinates as-is
                    center_x = int((coords[0] + coords[2]) / 2)
                    center_y = int((coords[1] + coords[3]) / 2)
                    width = int(coords[2] - coords[0])
                    height = int(coords[3] - coords[1])

                # Calculate area
                area = width * height

                obj_info = {
                    "id": obj.get("id", f"obj_{i}"),
                    "label": obj.get("label", "unknown"),
                    "x": center_x,
                    "y": center_y,
                    "width": width,
                    "height": height,
                    "area": area,
                    "bbox": coords,
                    "confidence": obj.get("confidence", 0.0),
                    "description": obj.get("description", "")
                }
                state["objects"].append(obj_info)
        
        return state
    
    
    def generate_simple_action_prompt(self, symbolic_state: Dict, game_name: str, game_controls: Dict) -> str:
        """
        Generate a simple action prompt with last 4 reasoning and actions included
        """
        # Build controls text
        controls_text = ""
        for action_num, description in game_controls.items():
            controls_text += f"- Action {action_num}: {description}\n"

        prompt = f"""You are an expert game player analyzing a game frame.

Game controls:
{controls_text}
Current frame analysis:
- Total objects detected: {symbolic_state.get("total_objects", 0)}

Detected objects with coordinates and positions:
"""

        # Add object positions with size information
        for obj in symbolic_state.get("objects", []):
            x = obj.get('x', 'unknown')
            y = obj.get('y', 'unknown')
            width = obj.get('width', 'unknown')
            height = obj.get('height', 'unknown')
            label = obj.get('label', 'unknown_object')
            prompt += f"- Object '{label}': positioned at coordinates x={x}, y={y}, size {width}x{height}\n"

        prompt += """
IMPORTANT: Use the symbolic information when available and reliable, but prioritize visual reasoning if objects are missing or the symbolic data seems incomplete. When symbolic data is present and comprehensive, use it for precise positioning and coordinates. If key objects are not detected symbolically, rely more heavily on visual analysis of the frame to make decisions

"""

        # Add last 4 reasoning and actions for better decision making (only if history is not disabled)
        if not self.disable_history and self.conversation_history:
            prompt += f"\nLast 4 Reasoning and Actions (for context):\n"
            for i, interaction in enumerate(self.conversation_history[-4:]):
                prompt += f"- Previous reasoning {i+1}: {interaction['reasoning']}\n"
                prompt += f"  Action taken: {interaction['action']}\n"

        prompt += """

As an expert player, analyze the scene and choose the optimal action.

Think step by step:
1. Observe the current state of the game
2. Identify the positions of all key objects
3. Predict the trajectory or movement patterns
4. Consider your strategic options
5. Choose the optimal action

Return ONLY JSON:
{
    "reasoning": "your expert analysis with positional awareness and learning from history",
    "action": integer_action_code
}
"""
        return prompt

    def generate_action_prompt_with_memory(self, symbolic_state: Dict, game_name: str, game_controls: Dict) -> str:
        """
        Generate the text component of the action prompt with memory and planning modules.
        """
        # Get action controls text
        controls_text = "\n".join([f"- Action {k}: {v}" for k, v in game_controls.items()])

        prompt = f"""You are an expert {game_name} player analyzing a game frame to choose the next action.\n\nYour task is to synthesize all available information: the visual frame, the symbolic object data, your recent memory, and the current strategic plan to make the optimal move.\n\nGame Controls:\n{controls_text}\n\nSymbolic State (for reference):\n- Total objects detected: {symbolic_state.get("total_objects", 0)}\n"""

        # Add object positions with size information
        for obj in symbolic_state.get("objects", []):
            width = obj.get('width', 'unknown')
            height = obj.get('height', 'unknown')
            prompt += f"- Object '{obj['label']}': coordinates at x={obj['x']}, y={obj['y']}, size {width}x{height}\n"

        prompt += """
IMPORTANT: Use the symbolic information when available and reliable, but prioritize visual reasoning if objects are missing or the symbolic data seems incomplete. When symbolic data is present and comprehensive, use it for precise positioning and coordinates. If key objects are not detected symbolically, rely more heavily on visual analysis of the frame to make decisions

"""

        # Add conversation history if available (changed to last 4 interactions) (only if history is not disabled)
        if not self.disable_history and self.conversation_history:
            prompt += f"""\nPrevious 4 Reasoning and Actions (for decision context):\n"""
            for i, interaction in enumerate(self.conversation_history[-4:]):  # Last 4 interactions
                prompt += f"- Decision {i+1}: {interaction['reasoning']}\n"
                prompt += f"  Action taken: {interaction['action']}\n"
        
        prompt += f"""Based on the visual evidence in the image and all the contextual information provided, choose the best action. Your primary guide should be the image itself.

Return ONLY valid JSON in the following format:
{{
    "reasoning": "Your expert analysis and decision rationale, citing visual evidence.",
    "action": integer_action_code
}}
"""
        
        return prompt

    def decide_next_action(self, prompt_text: str, image_path: str) -> Dict:
        """
        Makes a multi-modal call to decide the next action.
        """
        print("Step 8: Deciding next action with multi-modal input...")
        base64_image = self.encode_image_to_base64(image_path)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ]

        response = self._make_api_call(messages, max_tokens=1000, call_id="action_decision")
        if not response:
            return {"reasoning": "Failed to get action from API", "action": 0} # Default to NOOP

        # Save the full response for debugging
        self._last_action_response = response

        try:
            json_str = self._clean_json_response(response)
            action_result = json.loads(json_str)
            
            # Validate the result
            if "reasoning" not in action_result or "action" not in action_result:
                return {"reasoning": "Invalid action response format", "action": 0}

            print(f"Action decided: {action_result['action']} ({action_result['reasoning']})")
            return action_result

        except (json.JSONDecodeError, ValueError) as e:
            print(f"Error parsing action decision: {e}")
            print(f"Response was: {response}")
            return {"reasoning": "Error parsing action JSON", "action": 0}
    
    def process_single_frame(self, image_path: str, output_folder: str, game_name: str, game_controls: Dict) -> Dict:
        """
        Process a single frame with advanced pipeline (CoT, Planning, Memory)
        """
        os.makedirs(output_folder, exist_ok=True)
        
        self.prompts_dir = os.path.join(output_folder, "prompts")
        os.makedirs(self.prompts_dir, exist_ok=True)
        
        print(f"Processing frame: {os.path.basename(image_path)}")
        
        # Step 1: Scale image
        scaled_image_path = os.path.join(output_folder, "scaled_frame.jpg")
        self.scale_image(image_path, scaled_image_path)
        print("Step 1: Image scaled to 1280x720")
        
        # Step 2: Detect objects for the known game
        detections = self.detect_objects(scaled_image_path, game_name)
        if "error" in detections:
            print(f"Object detection failed: {detections['error']}")
            return detections

        if not detections.get("objects"):
            print("No objects detected - creating empty symbolic state")
            detections = {"objects": []}
        else:
            print(f"Detected {len(detections['objects'])} objects")

        # Step 3: Draw bounding boxes
        annotated_image_path = os.path.join(output_folder, "annotated_frame.jpg")
        if detections.get("objects"):
            self.draw_bounding_boxes(scaled_image_path, detections, annotated_image_path)
            print("Step 3: Bounding boxes drawn")
        else:
            # Copy original if no objects to annotate
            import shutil
            shutil.copy2(scaled_image_path, annotated_image_path)
            print("Step 3: No objects to annotate, copied original frame")

        # Step 4: Generate symbolic state
        print("Step 4: Generating symbolic state...")
        symbolic_state = self.generate_symbolic_state(detections)

        # Step 5: Generate simple action prompt (no planning/memory)
        print("Step 5: Generating action prompt...")
        action_prompt_text = self.generate_simple_action_prompt(symbolic_state, game_name, game_controls)

        # Step 6: Decide next action (multi-modal)
        print("Step 6: Deciding action...")
        action_decision = self.decide_next_action(action_prompt_text, scaled_image_path)
        print(f"Action decided: {action_decision['action']} ({action_decision['reasoning'][:100]}...)")

        # Step 7: Update memory with this decision for future reference (only if history is not disabled)
        if not self.disable_history:
            self.update_memory(action_decision['reasoning'], action_decision['action'])

        # Save simplified results
        results = {
            "identified_game": game_name,
            "symbolic_state": symbolic_state,
            "action_decision": action_decision,
            "actual_prompt_used": action_prompt_text,
            "scaled_image_path": scaled_image_path,
            "annotated_image_path": annotated_image_path
        }

        results_file = os.path.join(output_folder, "analysis.json")
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"Results saved to: {results_file}")
        return results
    
    def update_memory(self, reasoning: str, action: int):
        """
        Update memory with the latest decision
        """
        self.conversation_history.append({
            "timestamp": datetime.now().isoformat(),
            "reasoning": reasoning,
            "action": action
        })
        
        # Keep only the last 8 interactions (to ensure we always have at least 4 for context)
        if len(self.conversation_history) > 8:
            self.conversation_history.pop(0)
    
    
    def _clean_json_response(self, response: str) -> str:
        """Clean JSON response from API by extracting valid JSON from natural language text."""
        try:
            # First try to find JSON between code blocks
            import re
            
            # Look for JSON in code blocks first
            code_block_pattern = r"```(?:json)?\\s*({.*?})\\s*```"
            match = re.search(code_block_pattern, response, re.DOTALL)
            if match:
                return match.group(1).strip()
            
            # If no code blocks, try to find the first valid JSON object
            # Find the first '{' and try to parse from there
            start_index = response.find('{')
            if start_index != -1:
                # Try to extract JSON by brace counting
                brace_count = 0
                in_string = False
                escape_next = False
                i = start_index
                
                while i < len(response):
                    char = response[i]
                    
                    if escape_next:
                        escape_next = False
                    elif char == '\\\\':
                        escape_next = True
                    elif char == '"' and not escape_next:
                        in_string = not in_string
                    elif not in_string:
                        if char == '{':
                            brace_count += 1
                        elif char == '}':
                            brace_count -= 1
                            if brace_count == 0:
                                # Found the matching closing brace
                                json_str = response[start_index:i+1]
                                return json_str.strip()
                    i += 1
            
            # Fallback for cases where the response might be just the JSON without extra text
            return response.strip()
        except Exception as e:
            print(f"Error in _clean_json_response: {e}")
            # If anything goes wrong, just return the original response
            return response
    
    def _parse_markdown_detection(self, response: str) -> Dict:
        """
        Parse markdown-formatted detection response from Bedrock Llama models.

        Expected format:
        **Objects:**
        *   **Object Name**
            *   Label: Object Name
            *   Coordinates: [x1, y1, x2, y2]
            *   Confidence: 0.95
        """
        import re

        objects = []

        # Split by object sections (lines starting with * followed by **)
        # Pattern: *   **Label Name**
        object_sections = re.split(r'\n\*\s+\*\*', response)

        for section in object_sections[1:]:  # Skip first section (before first object)
            try:
                # Extract label (first line before ** closing)
                label_match = re.search(r'^([^*\n]+)\*\*', section)
                if not label_match:
                    continue
                label = label_match.group(1).strip()

                # Extract coordinates
                coords_match = re.search(r'Coordinates:\s*\[([^\]]+)\]', section, re.IGNORECASE)
                if not coords_match:
                    continue
                coords_str = coords_match.group(1)
                coordinates = [float(x.strip()) for x in coords_str.split(',')]

                if len(coordinates) != 4:
                    continue

                # Extract confidence
                conf_match = re.search(r'Confidence:\s*([\d.]+)', section, re.IGNORECASE)
                confidence = float(conf_match.group(1)) if conf_match else 0.95

                # Extract description (optional)
                desc_match = re.search(r'Description:\s*(.+?)(?=\n\*|$)', section, re.IGNORECASE | re.DOTALL)
                description = desc_match.group(1).strip() if desc_match else ""

                obj = {
                    "label": label,
                    "coordinates": coordinates,
                    "confidence": confidence,
                    "description": description
                }

                # Validate detection
                if self._validate_detection(obj):
                    objects.append(obj)

            except Exception as e:
                print(f"Warning: Failed to parse object section: {e}")
                continue

        if not objects:
            return {"error": "no_objects_found"}

        return {
            "objects": objects,
            "image_info": {
                "total_objects": len(objects)
            }
        }

    def _validate_detection(self, detection: Dict) -> bool:
        """Validate detection object structure"""
        required_fields = ['label', 'coordinates', 'confidence']
        
        for field in required_fields:
            if field not in detection:
                return False
        
        coords = detection['coordinates']
        if not isinstance(coords, list) or len(coords) != 4:
            return False
        
        x1, y1, x2, y2 = coords
        if not all(isinstance(c, (int, float)) for c in coords):
            return False
        
        if x1 >= x2 or y1 >= y2 or any(c < 0 for c in coords):
            return False
        
        # Check if coordinates are within scaled image bounds
        if x1 >= self.scaled_width or x2 >= self.scaled_width or y1 >= self.scaled_height or y2 >= self.scaled_height:
            return False
        
        confidence = detection['confidence']
        if not isinstance(confidence, (int, float)) or not (0 <= confidence <= 1):
            return False
        
        return True


# Game-specific detector classes
class BreakoutAdvancedDetector(AdvancedSymbolicDetector):
    def __init__(self, openrouter_api_key: str, model_name: str = "anthropic/claude-sonnet-4", detection_mode: str = "specific", provider: str = "openrouter", aws_region: str = "us-east-1", disable_history: bool = False):
        super().__init__(openrouter_api_key, model_name, detection_mode, provider, aws_region, disable_history)
        self.game_name = "Breakout"
        self.game_controls = {
            0: "NOOP (do nothing)",
            1: "FIRE (primary action - often shoot/serve/activate)",
            2: "RIGHT (move right or right action)",
            3: "LEFT (move left or left action)",
            4: "RIGHTFIRE (combination of right + fire)",
            5: "LEFTFIRE (combination of left + fire)"
        }
    
    def process_single_frame(self, image_path: str, output_folder: str) -> Dict:
        return super().process_single_frame(image_path, output_folder, self.game_name, self.game_controls)


class FroggerAdvancedDetector(AdvancedSymbolicDetector):
    def __init__(self, openrouter_api_key: str, model_name: str = "anthropic/claude-sonnet-4", detection_mode: str = "specific", provider: str = "openrouter", aws_region: str = "us-east-1", disable_history: bool = False):
        super().__init__(openrouter_api_key, model_name, detection_mode, provider, aws_region, disable_history)
        self.game_name = "Frogger"
        self.game_controls = {
            0: "NOOP (do nothing)",
            1: "UP (move up)",
            2: "RIGHT (move right)",
            3: "LEFT (move left)",
            4: "DOWN (move down)"
        }
    
    def process_single_frame(self, image_path: str, output_folder: str) -> Dict:
        return super().process_single_frame(image_path, output_folder, self.game_name, self.game_controls)


class SpaceInvadersAdvancedDetector(AdvancedSymbolicDetector):
    def __init__(self, openrouter_api_key: str, model_name: str = "anthropic/claude-sonnet-4", detection_mode: str = "specific", provider: str = "openrouter", aws_region: str = "us-east-1", disable_history: bool = False):
        super().__init__(openrouter_api_key, model_name, detection_mode, provider, aws_region, disable_history)
        self.game_name = "Space Invaders"
        self.game_controls = {
            0: "NOOP (do nothing)",
            1: "FIRE (shoot)",
            2: "RIGHT (move right)",
            3: "LEFT (move left)",
            4: "RIGHTFIRE (move right and shoot)",
            5: "LEFTFIRE (move left and shoot)"
        }
    
    def process_single_frame(self, image_path: str, output_folder: str) -> Dict:
        return super().process_single_frame(image_path, output_folder, self.game_name, self.game_controls)


class PacmanAdvancedDetector(AdvancedSymbolicDetector):
    def __init__(self, openrouter_api_key: str, model_name: str = "anthropic/claude-sonnet-4", detection_mode: str = "specific", provider: str = "openrouter", aws_region: str = "us-east-1", disable_history: bool = False):
        super().__init__(openrouter_api_key, model_name, detection_mode, provider, aws_region, disable_history)
        self.game_name = "Pacman"
        self.game_controls = {
            0: "NOOP (do nothing)",
            1: "UP (move up)",
            2: "RIGHT (move right)",
            3: "LEFT (move left)",
            4: "DOWN (move down)"
        }
    
    def process_single_frame(self, image_path: str, output_folder: str) -> Dict:
        return super().process_single_frame(image_path, output_folder, self.game_name, self.game_controls)


class MsPacmanAdvancedDetector(AdvancedSymbolicDetector):
    def __init__(self, openrouter_api_key: str, model_name: str = "anthropic/claude-sonnet-4", detection_mode: str = "specific", provider: str = "openrouter", aws_region: str = "us-east-1", disable_history: bool = False):
        super().__init__(openrouter_api_key, model_name, detection_mode, provider, aws_region, disable_history)
        self.game_name = "Ms. Pacman"
        self.game_controls = {
            0: "NOOP (do nothing)",
            1: "UP (move up)",
            2: "RIGHT (move right)",
            3: "LEFT (move left)",
            4: "DOWN (move down)",
            5: "UPRIGHT (move up and right diagonally)",
            6: "UPLEFT (move up and left diagonally)",
            7: "DOWNRIGHT (move down and right diagonally)",
            8: "DOWNLEFT (move down and left diagonally)"
        }

    def process_single_frame(self, image_path: str, output_folder: str) -> Dict:
        return super().process_single_frame(image_path, output_folder, self.game_name, self.game_controls)




class TennisAdvancedDetector(AdvancedSymbolicDetector):
    def __init__(self, openrouter_api_key: str, model_name: str = "anthropic/claude-sonnet-4", detection_mode: str = "specific", provider: str = "openrouter", aws_region: str = "us-east-1", disable_history: bool = False):
        super().__init__(openrouter_api_key, model_name, detection_mode, provider, aws_region, disable_history)
        self.game_name = "Tennis"
        self.game_controls = {
            0: "NOOP (do nothing)",
            1: "FIRE (hit/serve ball)",
            2: "UP (move toward net/forward)",
            3: "RIGHT (move right)",
            4: "LEFT (move left)",
            5: "DOWN (move toward baseline/backward)"
        }

    def generate_tennis_action_prompt(self, symbolic_state: Dict, game_name: str, game_controls: Dict) -> str:
        """
        Generate Tennis-specific action prompt with RED PLAYER focus and last 4 reasoning/actions
        """
        # Build controls text
        controls_text = ""
        for action_num, description in game_controls.items():
            controls_text += f"- Action {action_num}: {description}\n"

        prompt = f"""You are an expert Tennis player controlling the RED PLAYER analyzing a game frame.

Game controls:
{controls_text}
Current frame analysis:
- Total objects detected: {symbolic_state.get("total_objects", 0)}

Detected objects with coordinates and positions:
"""

        # Add object positions with size information
        for obj in symbolic_state.get("objects", []):
            x = obj.get('x', 'unknown')
            y = obj.get('y', 'unknown')
            width = obj.get('width', 'unknown')
            height = obj.get('height', 'unknown')
            label = obj.get('label', 'unknown_object')
            prompt += f"- Object '{label}': positioned at coordinates x={x}, y={y}, size {width}x{height}\n"

        prompt += """
IMPORTANT: Use the symbolic information when available and reliable, but prioritize visual reasoning if objects are missing or the symbolic data seems incomplete. When symbolic data is present and comprehensive, use it for precise positioning and coordinates. If key objects are not detected symbolically, rely more heavily on visual analysis of the frame to make decisions

"""

        # Add last 4 reasoning and actions for better decision making (only if history is not disabled)
        if not self.disable_history and self.conversation_history:
            prompt += f"\nLast 4 Reasoning and Actions (for context):\n"
            for i, interaction in enumerate(self.conversation_history[-4:]):
                prompt += f"- Previous reasoning {i+1}: {interaction['reasoning']}\n"
                prompt += f"  Action taken: {interaction['action']}\n"

        prompt += """

As an expert Tennis player controlling the RED PLAYER, analyze the scene and choose the optimal action.

Think step by step:
1. Observe the current state of the game
2. Identify the positions of all key objects
3. Predict the trajectory or movement patterns
4. Consider your strategic options
5. Choose the optimal action

Return ONLY JSON:
{
    "reasoning": "your expert analysis focusing on the RED PLAYER with positional awareness and learning from history",
    "action": integer_action_code
}
"""
        return prompt

    def process_single_frame(self, image_path: str, output_folder: str) -> Dict:
        """
        Process a single frame with Tennis-specific RED PLAYER prompt
        """
        os.makedirs(output_folder, exist_ok=True)

        self.prompts_dir = os.path.join(output_folder, "prompts")
        os.makedirs(self.prompts_dir, exist_ok=True)

        print(f"Processing frame: {os.path.basename(image_path)}")

        # Step 1: Scale image
        scaled_image_path = os.path.join(output_folder, "scaled_frame.jpg")
        self.scale_image(image_path, scaled_image_path)
        print("Step 1: Image scaled to 1280x720")

        # Step 2: Detect objects for Tennis
        detections = self.detect_objects(scaled_image_path, self.game_name)
        if "error" in detections:
            print(f"Object detection failed: {detections['error']}")
            return detections
        if not detections.get("objects"):
            print("No objects detected - creating empty symbolic state")
            detections = {"objects": []}
        else:
            print(f"Detected {len(detections['objects'])} objects")

        # Step 3: Draw bounding boxes
        annotated_image_path = os.path.join(output_folder, "annotated_frame.jpg")
        if detections.get("objects"):
            self.draw_bounding_boxes(scaled_image_path, detections, annotated_image_path)
            print("Step 3: Bounding boxes drawn")
        else:
            # Copy original if no objects to annotate
            import shutil
            shutil.copy2(scaled_image_path, annotated_image_path)
            print("Step 3: No objects to annotate, copied original frame")

        # Step 4: Generate symbolic state
        print("Step 4: Generating symbolic state...")
        symbolic_state = self.generate_symbolic_state(detections)

        # Step 5: Generate Tennis-specific action prompt with RED PLAYER focus
        print("Step 5: Generating Tennis action prompt with RED PLAYER...")
        action_prompt_text = self.generate_tennis_action_prompt(symbolic_state, self.game_name, self.game_controls)

        # Step 6: Decide next action (multi-modal)
        print("Step 6: Deciding action...")
        action_decision = self.decide_next_action(action_prompt_text, scaled_image_path)
        print(f"Action decided: {action_decision['action']} ({action_decision['reasoning'][:100]}...)")

        # Step 7: Update memory with this decision for future reference (only if history is not disabled)
        if not self.disable_history:
            self.update_memory(action_decision['reasoning'], action_decision['action'])

        # Save simplified results
        results = {
            "identified_game": self.game_name,
            "symbolic_state": symbolic_state,
            "action_decision": action_decision,
            "actual_prompt_used": action_prompt_text,
            "scaled_image_path": scaled_image_path,
            "annotated_image_path": annotated_image_path
        }

        results_file = os.path.join(output_folder, "analysis.json")
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"Results saved to: {results_file}")
        return results


class PongAdvancedDetector(AdvancedSymbolicDetector):
    def __init__(self, openrouter_api_key: str, model_name: str = "anthropic/claude-sonnet-4", detection_mode: str = "specific", provider: str = "openrouter", aws_region: str = "us-east-1", disable_history: bool = False):
        super().__init__(openrouter_api_key, model_name, detection_mode, provider, aws_region, disable_history)
        self.game_name = "Pong"
        self.game_controls = {
            0: "NOOP (do nothing)",
            1: "FIRE (serve/start ball - rarely used)",
            2: "RIGHT/UP (move paddle up)",
            3: "LEFT/DOWN (move paddle down)",
            4: "RIGHTFIRE/UPFIRE (move up and fire)",
            5: "LEFTFIRE/DOWNFIRE (move down and fire)"
        }

    def generate_pong_action_prompt(self, symbolic_state: Dict, game_name: str, game_controls: Dict) -> str:
        """
        Generate Pong-specific action prompt with GREEN PADDLE focus and last 4 reasoning/actions
        """
        # Build controls text
        controls_text = ""
        for action_num, description in game_controls.items():
            controls_text += f"- Action {action_num}: {description}\n"

        prompt = f"""You are an expert Pong player controlling the GREEN PADDLE analyzing a game frame.

Game controls:
{controls_text}
Current frame analysis:
- Total objects detected: {symbolic_state.get("total_objects", 0)}

Detected objects with coordinates and positions:
"""

        # Add object positions with size information
        for obj in symbolic_state.get("objects", []):
            x = obj.get('x', 'unknown')
            y = obj.get('y', 'unknown')
            width = obj.get('width', 'unknown')
            height = obj.get('height', 'unknown')
            label = obj.get('label', 'unknown_object')
            prompt += f"- Object '{label}': positioned at coordinates x={x}, y={y}, size {width}x{height}\n"

        prompt += """
IMPORTANT: Use the symbolic information when available and reliable, but prioritize visual reasoning if objects are missing or the symbolic data seems incomplete. When symbolic data is present and comprehensive, use it for precise positioning and coordinates. If key objects are not detected symbolically, rely more heavily on visual analysis of the frame to make decisions

"""

        # Add last 4 reasoning and actions for better decision making (only if history is not disabled)
        if not self.disable_history and self.conversation_history:
            prompt += f"\nLast 4 Reasoning and Actions (for context):\n"
            for i, interaction in enumerate(self.conversation_history[-4:]):
                prompt += f"- Previous reasoning {i+1}: {interaction['reasoning']}\n"
                prompt += f"  Action taken: {interaction['action']}\n"

        prompt += """

As an expert Pong player controlling the GREEN PADDLE, analyze the scene and choose the optimal action.

Think step by step:
1. Observe the current state of the game
2. Identify the positions of all key objects
3. Predict the trajectory or movement patterns
4. Consider your strategic options
5. Choose the optimal action

Return ONLY JSON:
{
    "reasoning": "your expert analysis focusing on the GREEN PADDLE with positional awareness and learning from history",
    "action": integer_action_code
}
"""
        return prompt

    def process_single_frame(self, image_path: str, output_folder: str) -> Dict:
        """
        Process a single frame with Pong-specific GREEN PADDLE prompt
        """
        os.makedirs(output_folder, exist_ok=True)

        self.prompts_dir = os.path.join(output_folder, "prompts")
        os.makedirs(self.prompts_dir, exist_ok=True)

        print(f"Processing frame: {os.path.basename(image_path)}")

        # Step 1: Scale image
        scaled_image_path = os.path.join(output_folder, "scaled_frame.jpg")
        self.scale_image(image_path, scaled_image_path)
        print("Step 1: Image scaled to 1280x720")

        # Step 2: Detect objects for Pong
        detections = self.detect_objects(scaled_image_path, self.game_name)
        if "error" in detections:
            print(f"Object detection failed: {detections['error']}")
            return detections
        if not detections.get("objects"):
            print("No objects detected - creating empty symbolic state")
            detections = {"objects": []}
        else:
            print(f"Detected {len(detections['objects'])} objects")

        # Step 3: Draw bounding boxes
        annotated_image_path = os.path.join(output_folder, "annotated_frame.jpg")
        if detections.get("objects"):
            self.draw_bounding_boxes(scaled_image_path, detections, annotated_image_path)
            print("Step 3: Bounding boxes drawn")
        else:
            # Copy original if no objects to annotate
            import shutil
            shutil.copy2(scaled_image_path, annotated_image_path)
            print("Step 3: No objects to annotate, copied original frame")

        # Step 4: Generate symbolic state
        print("Step 4: Generating symbolic state...")
        symbolic_state = self.generate_symbolic_state(detections)

        # Step 5: Generate Pong-specific action prompt with GREEN PADDLE focus
        print("Step 5: Generating Pong action prompt with GREEN PADDLE...")
        action_prompt_text = self.generate_pong_action_prompt(symbolic_state, self.game_name, self.game_controls)

        # Step 6: Decide next action (multi-modal)
        print("Step 6: Deciding action...")
        action_decision = self.decide_next_action(action_prompt_text, scaled_image_path)
        print(f"Action decided: {action_decision['action']} ({action_decision['reasoning'][:100]}...)")

        # Step 7: Update memory with this decision for future reference (only if history is not disabled)
        if not self.disable_history:
            self.update_memory(action_decision['reasoning'], action_decision['action'])

        # Save simplified results
        results = {
            "identified_game": self.game_name,
            "symbolic_state": symbolic_state,
            "action_decision": action_decision,
            "actual_prompt_used": action_prompt_text,
            "scaled_image_path": scaled_image_path,
            "annotated_image_path": annotated_image_path
        }

        results_file = os.path.join(output_folder, "analysis.json")
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"Results saved to: {results_file}")
        return results


class AssaultAdvancedDetector(AdvancedSymbolicDetector):
    def __init__(self, openrouter_api_key: str, model_name: str = "anthropic/claude-sonnet-4", detection_mode: str = "specific", provider: str = "openrouter", aws_region: str = "us-east-1", disable_history: bool = False):
        super().__init__(openrouter_api_key, model_name, detection_mode, provider, aws_region, disable_history)
        self.game_name = "Assault"
        self.game_controls = {
            0: "NOOP (do nothing)",
            1: "FIRE (shoot)",
            2: "UP (move up)",
            3: "RIGHT (move right)",
            4: "LEFT (move left)",
            5: "RIGHTFIRE (move right and shoot)",
            6: "LEFTFIRE (move left and shoot)"
        }

    def process_single_frame(self, image_path: str, output_folder: str) -> Dict:
        return super().process_single_frame(image_path, output_folder, self.game_name, self.game_controls)


def main():
    parser = argparse.ArgumentParser(description="Advanced Zero-Shot Pipeline with CoT, Planning, and Memory")
    parser.add_argument("--input", required=True, help="Path to input image file")
    parser.add_argument("--output", required=True, help="Output folder for results")
    parser.add_argument("--api-key", required=True, help="OpenRouter API key")
    parser.add_argument("--model", default="anthropic/claude-sonnet-4", help="Model to use")
    parser.add_argument("--game", choices=["breakout", "frogger", "space_invaders", "pacman", "mspacman", "pong", "tennis", "assault"], required=True,
                        help="Game type")
    parser.add_argument("--detection-mode", choices=["specific", "generic"], default="specific",
                        help="Detection mode: specific (game-specific) or generic (universal)")
    parser.add_argument("--disable-history", action="store_true",
                        help="Disable history mechanism (conversation memory)")

    args = parser.parse_args()
    
    # Initialize detector based on game type and detection mode
    if args.game == "breakout":
        detector = BreakoutAdvancedDetector(args.api_key, args.model, args.detection_mode, disable_history=args.disable_history)
    elif args.game == "frogger":
        detector = FroggerAdvancedDetector(args.api_key, args.model, args.detection_mode, disable_history=args.disable_history)
    elif args.game == "space_invaders":
        detector = SpaceInvadersAdvancedDetector(args.api_key, args.model, args.detection_mode, disable_history=args.disable_history)
    elif args.game == "pacman":
        detector = PacmanAdvancedDetector(args.api_key, args.model, args.detection_mode, disable_history=args.disable_history)
    elif args.game == "mspacman":
        detector = MsPacmanAdvancedDetector(args.api_key, args.model, args.detection_mode, disable_history=args.disable_history)
    elif args.game == "pong":
        detector = PongAdvancedDetector(args.api_key, args.model, args.detection_mode, disable_history=args.disable_history)
    elif args.game == "tennis":
        detector = TennisAdvancedDetector(args.api_key, args.model, args.detection_mode, disable_history=args.disable_history)
    elif args.game == "assault":
        detector = AssaultAdvancedDetector(args.api_key, args.model, args.detection_mode, disable_history=args.disable_history)
    else:
        raise ValueError(f"Unsupported game: {args.game}")
    
    # Process frame
    results = detector.process_single_frame(args.input, args.output)
    
    print("\nProcessing complete!")
    print(f"Results saved to: {args.output}")


if __name__ == "__main__":
    main()
