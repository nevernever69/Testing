#!/usr/bin/env python3
"""Quick test to verify OCAtari coordinate extraction works correctly."""

from ocatari_ground_truth import OCAtariGroundTruth

# Test with Pong
print("Testing OCAtari coordinate extraction with Pong...")
env = OCAtariGroundTruth('Pong', render_mode='rgb_array')
env.reset(seed=42)

# Take a few steps to get game going
for _ in range(20):
    env.step(1)  # FIRE to start

# Get frame and objects
frame, objects = env.get_frame_and_objects()

print(f"\nFrame shape: {frame.shape}")
print(f"Number of objects: {len(objects)}")

if objects:
    print("\nObject details:")
    for i, obj in enumerate(objects):
        obj_dict = obj.to_dict()
        print(f"\nObject {i}:")
        print(f"  Category: {obj_dict.get('category', 'unknown')}")
        print(f"  Position (tuple): {obj_dict.get('position', (0, 0))}")
        print(f"  Size (tuple): {obj_dict.get('size', (0, 0))}")
        print(f"  Velocity: {obj_dict.get('velocity', (0, 0))}")

        # Test coordinate extraction
        position = obj_dict.get('position', (0, 0))
        size = obj_dict.get('size', (0, 0))
        x, y = position
        w, h = size
        print(f"  Extracted: x={x}, y={y}, w={w}, h={h}")

        # Test scaling
        x_scale = 1280 / 160
        y_scale = 720 / 210
        x_scaled = int(x * x_scale)
        y_scaled = int(y * y_scale)
        w_scaled = int(w * x_scale)
        h_scaled = int(h * y_scale)
        print(f"  Scaled to 1280×720: x={x_scaled}, y={y_scaled}, w={w_scaled}, h={h_scaled}")
else:
    print("\nNo objects detected!")

env.close()
print("\n✓ Test complete")
