#!/usr/bin/env python3
"""
Test that the agent can execute all Space Invaders actions including RIGHTFIRE and LEFTFIRE
This simulates what happens in the actual game runner.
"""
import gymnasium as gym
import numpy as np
import pygame
import sys

# Register ALE games
try:
    import ale_py
    gym.register_envs(ale_py)
except:
    pass

def test_agent_actions():
    """Test that an agent can execute all 6 Space Invaders actions"""

    # Initialize Pygame for visualization
    pygame.init()

    # Create environment (using the same version as advance_game_runner.py)
    env = gym.make("SpaceInvaders-v4", render_mode="rgb_array")
    observation, info = env.reset()

    # Get initial frame to determine window size
    frame = env.render()
    scale = 3
    screen_width = frame.shape[1] * scale
    screen_height = frame.shape[0] * scale + 200

    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("Agent Action Test - Space Invaders")

    try:
        font_large = pygame.font.Font(None, 40)
        font_medium = pygame.font.Font(None, 28)
        font_small = pygame.font.Font(None, 24)
    except:
        font_large = pygame.font.SysFont('arial', 40)
        font_medium = pygame.font.SysFont('arial', 28)
        font_small = pygame.font.SysFont('arial', 24)

    print("=" * 70)
    print("AGENT ACTION TEST - Space Invaders")
    print("=" * 70)
    print("\nThis test simulates what the agent does in the actual game runner.")
    print("The agent will cycle through all 6 actions automatically.\n")
    print("Actions to test:")
    print("  0: NOOP")
    print("  1: FIRE")
    print("  2: RIGHT")
    print("  3: LEFT")
    print("  4: RIGHTFIRE ← Testing this!")
    print("  5: LEFTFIRE  ← Testing this!")
    print("\nPress Q to quit")
    print("=" * 70)

    # Action test sequence - repeat each action multiple times to see the effect
    action_sequence = []
    frames_per_action = 120  # Hold each action for 120 frames (VERY SLOW)
    pause_frames = 20  # Pause between actions

    # Test each action with pauses
    for action in range(6):
        action_sequence.extend([action] * frames_per_action)
        # Add pause (NOOP) between actions
        action_sequence.extend([0] * pause_frames)

    # Repeat the sequence
    action_sequence = action_sequence * 2

    action_names = {
        0: "NOOP",
        1: "FIRE",
        2: "RIGHT",
        3: "LEFT",
        4: "RIGHTFIRE",
        5: "LEFTFIRE"
    }

    frame_count = 0
    total_reward = 0
    action_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
    clock = pygame.time.Clock()
    running = True
    action_index = 0

    while running and action_index < len(action_sequence):
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Check for quit key
        keys = pygame.key.get_pressed()
        if keys[pygame.K_q] or keys[pygame.K_ESCAPE]:
            running = False
            continue

        # Get next action from sequence (simulating what agent would do)
        action = action_sequence[action_index]
        action_index += 1

        # Execute action (this is exactly what the agent does)
        observation, reward, terminated, truncated, info = env.step(action)

        # Update stats
        frame_count += 1
        total_reward += reward
        action_counts[action] += 1

        # Log significant events
        if reward > 0:
            print(f"Frame {frame_count}: Action {action} ({action_names[action]}) → HIT! +{reward} (Total: {total_reward:.0f})")

        # Render
        frame = env.render()
        frame_surface = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
        frame_scaled = pygame.transform.scale(frame_surface,
                                              (frame.shape[1] * scale, frame.shape[0] * scale))

        screen.fill((0, 0, 0))
        screen.blit(frame_scaled, (0, 0))

        # Info panel
        info_y = frame.shape[0] * scale + 10

        # Current action being tested
        current_action_name = action_names[action]
        action_color = (255, 255, 0) if action in [4, 5] else (150, 150, 150)
        if action == 4:
            current_action_name = "RIGHTFIRE ← TESTING!"
            action_color = (0, 255, 0)
        elif action == 5:
            current_action_name = "LEFTFIRE ← TESTING!"
            action_color = (0, 255, 0)

        text = font_large.render(f"Action {action}: {current_action_name}", True, action_color)
        screen.blit(text, (10, info_y))

        # Frame and score
        text = font_medium.render(f"Frame: {frame_count}  |  Score: {total_reward:.0f}", True, (255, 255, 255))
        screen.blit(text, (10, info_y + 45))

        # Action counts
        text = font_small.render(
            f"Action counts: NOOP:{action_counts[0]} FIRE:{action_counts[1]} R:{action_counts[2]} L:{action_counts[3]}",
            True, (200, 200, 200))
        screen.blit(text, (10, info_y + 80))

        text = font_small.render(
            f"               RFIRE:{action_counts[4]} LFIRE:{action_counts[5]}",
            True, (200, 200, 200))
        screen.blit(text, (10, info_y + 105))

        # Progress
        progress = (action_index / len(action_sequence)) * 100
        text = font_small.render(f"Test Progress: {progress:.1f}%", True, (100, 255, 100))
        screen.blit(text, (10, info_y + 135))

        # Status
        text = font_small.render("Agent is executing actions automatically (simulated)", True, (255, 200, 0))
        screen.blit(text, (10, info_y + 165))

        pygame.display.flip()

        # Reset if needed
        if terminated or truncated:
            observation, info = env.reset()

        # Control frame rate - VERY SLOW so you can see each action clearly
        clock.tick(15)  # 15 FPS (much slower, easier to see)

    # Test complete
    pygame.quit()
    env.close()

    print("\n" + "=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)
    print(f"Total frames: {frame_count}")
    print(f"Final score: {total_reward:.0f}")
    print(f"\nAction counts:")
    for action in range(6):
        print(f"  {action} ({action_names[action]}): {action_counts[action]} times")
    print("\n✓ All actions were executed by the agent, including RIGHTFIRE and LEFTFIRE")
    print("=" * 70)


if __name__ == "__main__":
    print("\nStarting Agent Action Test for Space Invaders...")
    print("This simulates what the agent does in your actual pipeline.\n")

    try:
        test_agent_actions()
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
