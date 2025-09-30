"""
Test VLM Integration

Quick test to verify that the VLM pipeline is making real API calls
and returning actual responses, not mock data.
"""

import numpy as np
from advanced_zero_shot_pipeline import PongAdvancedDetector
import tempfile
import os
from PIL import Image


def test_vlm_api_call():
    """Test that VLM actually makes API calls and returns real responses."""

    print("🧪 Testing VLM Integration...")

    # Create a test detector
    detector = PongAdvancedDetector(
        openrouter_api_key=os.getenv('OPENROUTER_API_KEY', 'dummy_key'),
        model_name="anthropic/claude-sonnet-4",
        detection_mode="specific",
        provider="bedrock",  # Use your preferred provider
        aws_region="us-east-1",
        disable_history=True
    )

    # Create temp directory for test
    temp_dir = tempfile.mkdtemp()
    detector.prompts_dir = temp_dir

    # Create a simple test frame (random data for testing)
    test_frame = np.random.randint(0, 255, (210, 160, 3), dtype=np.uint8)

    # Save as image
    frame_path = os.path.join(temp_dir, "test_frame.png")
    frame_image = Image.fromarray(test_frame)
    frame_image.save(frame_path)

    # Test simple prompt
    test_prompt = "Identify all key elements in this image. Be specific. Use at most 100 words."

    try:
        # Convert to base64
        base64_image = detector.encode_image_to_base64(frame_path)

        # Create message for VLM
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": test_prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
                ]
            }
        ]

        print("📞 Making VLM API call...")
        response = detector._make_api_call(messages, max_tokens=200, call_id="test")

        if response:
            print("✅ VLM API call successful!")
            print(f"📝 Response: {response[:200]}...")
            print(f"🔗 Response type: {type(response)}")
            print(f"📏 Response length: {len(response)} characters")

            # Check if it's a real response (not mock)
            if "unable to" in response.lower() or len(response) < 10:
                print("⚠️  Response seems generic - check API credentials")
            else:
                print("🎯 Response seems genuine - VLM integration working!")

        else:
            print("❌ VLM API call failed - no response received")
            print("🔧 Check your API credentials and provider configuration")

    except Exception as e:
        print(f"💥 VLM API call error: {e}")
        print("🔧 This indicates a configuration or credentials issue")

    # Cleanup
    os.unlink(frame_path)
    os.rmdir(temp_dir)


if __name__ == "__main__":
    test_vlm_api_call()