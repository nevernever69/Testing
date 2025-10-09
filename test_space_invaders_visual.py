#!/usr/bin/env python3
"""
Visual Space Invaders Control Tester
Play Space Invaders with visual rendering to see what each action does.
"""
import gymnasium as gym
import numpy as np
import cv2
import sys
import time

# Register ALE games
try:
    import ale_py
    gym.register_envs(ale_py)
except:
    pass

def test_space_invaders_visual():
    """Test Space Invaders controls with visual feedback."""

    # Create environment
    env = gym.make("ALE/SpaceInvaders-v5", render_mode="rgb_array")
    observation, info = env.reset()

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
    print("  1-5        : Execute action 1-5 once")
    print("  R          : Reset game")
    print()
    print("=" * 70)
    print()
    print("Press any key in the game window to start...")

    frame_count = 0
    total_reward = 0
    last_action = 0
    action_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0}

    # Action names for display
    action_names = {
        0: "NOOP",
        1: "FIRE",
        2: "RIGHT",
        3: "LEFT",
        4: "RIGHTFIRE",
        5: "LEFTFIRE"
    }

    # Key mapping
    key_to_action = {
        ord('n'): 0,
        ord('N'): 0,
        ord(' '): 1,  # SPACE = FIRE
        ord('1'): 1,
        ord('2'): 2,
        ord('3'): 3,
        ord('4'): 4,
        ord('5'): 5,
        ord('a'): 5,  # A = LEFTFIRE
        ord('A'): 5,
        ord('d'): 4,  # D = RIGHTFIRE
        ord('D'): 4,
        83: 2,  # RIGHT ARROW
        81: 3,  # LEFT ARROW
    }

    try:
        while True:
            # Get current frame
            frame = env.render()

            # Resize for better visibility (scale up 3x)
            display_frame = cv2.resize(frame, (frame.shape[1] * 3, frame.shape[0] * 3),
                                      interpolation=cv2.INTER_NEAREST)

            # Add information overlay
            info_height = 150
            info_panel = np.zeros((info_height, display_frame.shape[1], 3), dtype=np.uint8)

            # Display stats
            cv2.putText(info_panel, f"Frame: {frame_count}", (10, 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(info_panel, f"Score: {total_reward:.0f}", (10, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(info_panel, f"Last Action: {last_action} ({action_names[last_action]})", (10, 75),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            # Action counts
            cv2.putText(info_panel,
                       f"Actions: NOOP:{action_counts[0]} FIRE:{action_counts[1]} R:{action_counts[2]} L:{action_counts[3]}",
                       (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            cv2.putText(info_panel,
                       f"         RFIRE:{action_counts[4]} LFIRE:{action_counts[5]}",
                       (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            # Controls hint
            cv2.putText(info_panel, "Controls: A=LFIRE  D=RFIRE  SPACE=FIRE  Arrows=Move  Q=Quit",
                       (10, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 255, 100), 1)

            # Combine game frame and info panel
            full_display = np.vstack([display_frame, info_panel])

            # Show the frame
            cv2.imshow('Space Invaders Control Test', full_display)

            # Wait for key press (1ms delay)
            key = cv2.waitKey(1) & 0xFF

            # Default action is repeat last action (for continuous movement)
            action = 0  # Default to NOOP

            # Handle key presses
            if key == ord('q') or key == ord('Q') or key == 27:  # Q or ESC
                print("\nExiting...")
                break
            elif key == ord('r') or key == ord('R'):  # Reset
                observation, info = env.reset()
                frame_count = 0
                total_reward = 0
                last_action = 0
                action_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
                print("\nGame reset!")
                continue
            elif key in key_to_action:
                action = key_to_action[key]
            elif key != 255:  # 255 = no key pressed
                # Unknown key, just do NOOP
                action = 0
            else:
                # No key pressed, do NOOP
                action = 0

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
                while True:
                    frame = env.render()
                    display_frame = cv2.resize(frame, (frame.shape[1] * 3, frame.shape[0] * 3),
                                              interpolation=cv2.INTER_NEAREST)

                    # Game over message
                    cv2.putText(display_frame, "GAME OVER",
                               (display_frame.shape[1]//2 - 100, display_frame.shape[0]//2),
                               cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
                    cv2.putText(display_frame, f"Score: {total_reward:.0f}",
                               (display_frame.shape[1]//2 - 80, display_frame.shape[0]//2 + 40),
                               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)
                    cv2.putText(display_frame, "Press R to restart or Q to quit",
                               (display_frame.shape[1]//2 - 200, display_frame.shape[0]//2 + 80),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                    cv2.imshow('Space Invaders Control Test', display_frame)
                    key = cv2.waitKey(100) & 0xFF

                    if key == ord('r') or key == ord('R'):
                        observation, info = env.reset()
                        frame_count = 0
                        total_reward = 0
                        last_action = 0
                        action_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
                        print("\nGame reset!")
                        break
                    elif key == ord('q') or key == ord('Q') or key == 27:
                        print("\nExiting...")
                        cv2.destroyAllWindows()
                        env.close()
                        return

            # Small delay to make it playable
            time.sleep(0.02)  # ~50 FPS

    except KeyboardInterrupt:
        print("\n\nInterrupted by user (Ctrl+C)")

    finally:
        cv2.destroyAllWindows()
        env.close()
        print("\nEnvironment closed.")
        print(f"\nFinal Stats:")
        print(f"  Frames played: {frame_count}")
        print(f"  Final score: {total_reward:.0f}")
        print(f"  Action counts: {action_counts}")


if __name__ == "__main__":
    print("\nStarting Space Invaders Visual Test...")
    print("Make sure the game window has focus to use keyboard controls!")
    print()

    test_space_invaders_visual()
