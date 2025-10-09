#!/usr/bin/env python3
"""
Visual Space Invaders Control Tester using Pygame
Play Space Invaders with visual rendering to see what each action does.
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

def test_space_invaders_visual():
    """Test Space Invaders controls with visual feedback."""

    # Initialize Pygame
    pygame.init()

    # Create environment
    env = gym.make("ALE/SpaceInvaders-v5", render_mode="rgb_array")
    observation, info = env.reset()

    # Get initial frame to determine window size
    frame = env.render()
    scale = 3  # Scale up 3x for visibility
    screen_width = frame.shape[1] * scale
    screen_height = frame.shape[0] * scale + 180  # Extra space for info

    # Create display
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("Space Invaders Control Test")

    # Font for text
    try:
        font_large = pygame.font.Font(None, 36)
        font_medium = pygame.font.Font(None, 28)
        font_small = pygame.font.Font(None, 24)
    except:
        font_large = pygame.font.SysFont('arial', 36)
        font_medium = pygame.font.SysFont('arial', 28)
        font_small = pygame.font.SysFont('arial', 24)

    print("=" * 70)
    print("SPACE INVADERS VISUAL CONTROL TEST")
    print("=" * 70)
    print()
    print("Keyboard Controls:")
    print("  Q or ESC   : Quit")
    print("  SPACE      : FIRE (action 1)")
    print("  RIGHT ARROW: RIGHT (action 2)")
    print("  LEFT ARROW : LEFT (action 3)")
    print("  D          : RIGHTFIRE (action 4)")
    print("  A          : LEFTFIRE (action 5)")
    print("  N          : NOOP (action 0)")
    print()
    print("  R          : Reset game")
    print()
    print("=" * 70)
    print()

    frame_count = 0
    total_reward = 0
    last_action = 0
    action_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
    clock = pygame.time.Clock()
    running = True

    # Action names for display
    action_names = {
        0: "NOOP",
        1: "FIRE",
        2: "RIGHT",
        3: "LEFT",
        4: "RIGHTFIRE",
        5: "LEFTFIRE"
    }

    while running:
        # Default action
        action = 0

        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Check currently pressed keys
        keys = pygame.key.get_pressed()

        if keys[pygame.K_q] or keys[pygame.K_ESCAPE]:
            running = False
            continue

        if keys[pygame.K_r]:
            observation, info = env.reset()
            frame_count = 0
            total_reward = 0
            last_action = 0
            action_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
            print("\nGame reset!")
            continue

        # Map keys to actions
        if keys[pygame.K_d]:
            action = 4  # RIGHTFIRE
        elif keys[pygame.K_a]:
            action = 5  # LEFTFIRE
        elif keys[pygame.K_SPACE]:
            action = 1  # FIRE
        elif keys[pygame.K_RIGHT]:
            action = 2  # RIGHT
        elif keys[pygame.K_LEFT]:
            action = 3  # LEFT
        elif keys[pygame.K_n]:
            action = 0  # NOOP
        elif keys[pygame.K_1]:
            action = 1
        elif keys[pygame.K_2]:
            action = 2
        elif keys[pygame.K_3]:
            action = 3
        elif keys[pygame.K_4]:
            action = 4
        elif keys[pygame.K_5]:
            action = 5
        else:
            action = 0  # NOOP if no key pressed

        # Execute action
        observation, reward, terminated, truncated, info = env.step(action)

        # Update stats
        frame_count += 1
        total_reward += reward
        last_action = action
        action_counts[action] += 1

        # Print feedback for significant events
        if reward > 0:
            print(f"Frame {frame_count}: {action_names[action]} → HIT! +{reward} (Total: {total_reward:.0f})")

        # Render game
        frame = env.render()

        # Convert to pygame surface and scale up
        frame_surface = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
        frame_scaled = pygame.transform.scale(frame_surface,
                                              (frame.shape[1] * scale, frame.shape[0] * scale))

        # Draw to screen
        screen.fill((0, 0, 0))
        screen.blit(frame_scaled, (0, 0))

        # Draw info panel
        info_y = frame.shape[0] * scale + 10

        # Line 1: Frame and Score
        text = font_medium.render(f"Frame: {frame_count}  |  Score: {total_reward:.0f}",
                                 True, (255, 255, 255))
        screen.blit(text, (10, info_y))

        # Line 2: Last Action
        action_color = (0, 255, 255) if action != 0 else (150, 150, 150)
        text = font_medium.render(f"Last Action: {last_action} ({action_names[last_action]})",
                                 True, action_color)
        screen.blit(text, (10, info_y + 35))

        # Line 3: Action counts
        text = font_small.render(
            f"NOOP:{action_counts[0]} FIRE:{action_counts[1]} R:{action_counts[2]} L:{action_counts[3]} " +
            f"RFIRE:{action_counts[4]} LFIRE:{action_counts[5]}",
            True, (200, 200, 200))
        screen.blit(text, (10, info_y + 70))

        # Line 4: Controls hint
        text = font_small.render("A=LEFTFIRE  D=RIGHTFIRE  SPACE=FIRE  Arrows=Move  R=Reset  Q=Quit",
                                True, (100, 255, 100))
        screen.blit(text, (10, info_y + 100))

        # Line 5: Current key indicator
        current_key = "None"
        if keys[pygame.K_a]:
            current_key = "A (LEFTFIRE)"
        elif keys[pygame.K_d]:
            current_key = "D (RIGHTFIRE)"
        elif keys[pygame.K_SPACE]:
            current_key = "SPACE (FIRE)"
        elif keys[pygame.K_LEFT]:
            current_key = "LEFT"
        elif keys[pygame.K_RIGHT]:
            current_key = "RIGHT"

        text = font_small.render(f"Key Pressed: {current_key}", True, (255, 255, 0))
        screen.blit(text, (10, info_y + 130))

        pygame.display.flip()

        # Check if game over
        if terminated or truncated:
            print("\n" + "=" * 70)
            print("GAME OVER!")
            print(f"Final Score: {total_reward:.0f}")
            print(f"Total Frames: {frame_count}")
            print(f"Action Usage: {action_counts}")
            print("=" * 70)
            print("\nPress R to restart or Q to quit...")

            # Wait for R or Q
            waiting = True
            while waiting:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                        waiting = False
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_r:
                            observation, info = env.reset()
                            frame_count = 0
                            total_reward = 0
                            last_action = 0
                            action_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
                            print("\nGame reset!")
                            waiting = False
                        elif event.key == pygame.K_q or event.key == pygame.K_ESCAPE:
                            running = False
                            waiting = False

                # Draw game over screen
                screen.fill((0, 0, 0))
                screen.blit(frame_scaled, (0, 0))

                # Game over overlay
                overlay = pygame.Surface((screen_width, 200))
                overlay.set_alpha(200)
                overlay.fill((0, 0, 0))
                screen.blit(overlay, (0, screen_height // 2 - 100))

                text = font_large.render("GAME OVER", True, (255, 0, 0))
                text_rect = text.get_rect(center=(screen_width // 2, screen_height // 2 - 40))
                screen.blit(text, text_rect)

                text = font_medium.render(f"Score: {total_reward:.0f}", True, (255, 255, 0))
                text_rect = text.get_rect(center=(screen_width // 2, screen_height // 2))
                screen.blit(text, text_rect)

                text = font_small.render("Press R to restart or Q to quit", True, (255, 255, 255))
                text_rect = text.get_rect(center=(screen_width // 2, screen_height // 2 + 40))
                screen.blit(text, text_rect)

                pygame.display.flip()
                clock.tick(10)

        # Control frame rate (~50 FPS)
        clock.tick(50)

    # Cleanup
    pygame.quit()
    env.close()
    print("\nEnvironment closed.")
    print(f"\nFinal Stats:")
    print(f"  Frames played: {frame_count}")
    print(f"  Final score: {total_reward:.0f}")
    print(f"  Action counts: {action_counts}")


if __name__ == "__main__":
    print("\nStarting Space Invaders Visual Test with Pygame...")
    print("Use the keyboard controls in the game window!")
    print()

    try:
        test_space_invaders_visual()
    except Exception as e:
        print(f"\nError: {e}")
        print("\nIf you see a pygame error, install it with: pip install pygame")
        import traceback
        traceback.print_exc()
