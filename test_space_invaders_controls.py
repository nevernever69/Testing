#!/usr/bin/env python3
"""
Manual Space Invaders Control Tester
Play Space Invaders manually to test which controls work.
"""
import gymnasium as gym
import numpy as np
from PIL import Image
import sys

# Register ALE games
try:
    import ale_py
    gym.register_envs(ale_py)
except:
    pass

def test_space_invaders():
    """Test Space Invaders controls manually."""

    # Create environment
    env = gym.make("SpaceInvaders-v4", render_mode="rgb_array")
    observation, info = env.reset()

    print("=" * 70)
    print("SPACE INVADERS MANUAL CONTROL TEST")
    print("=" * 70)
    print()
    print("Available Actions:")
    print("  0: NOOP (Do nothing)")
    print("  1: FIRE")
    print("  2: RIGHT")
    print("  3: LEFT")
    print("  4: RIGHTFIRE (Move right + fire)")
    print("  5: LEFTFIRE (Move left + fire)")
    print()
    print("Controls:")
    print("  Press 0-5 to select action")
    print("  Press ENTER to execute action and advance frame")
    print("  Type 'auto <action>' to repeat action (e.g., 'auto 4' for continuous rightfire)")
    print("  Type 'q' or 'quit' to exit")
    print("=" * 70)
    print()

    frame_count = 0
    total_reward = 0
    last_action = 0
    auto_mode = False
    auto_action = 0

    # Action names for display
    action_names = {
        0: "NOOP",
        1: "FIRE",
        2: "RIGHT",
        3: "LEFT",
        4: "RIGHTFIRE",
        5: "LEFTFIRE"
    }

    try:
        while True:
            # Display current state
            print(f"\n--- Frame {frame_count} | Score: {total_reward:.0f} ---")

            # Get user input
            if auto_mode:
                user_input = str(auto_action)
                print(f"[AUTO MODE] Action: {auto_action} ({action_names[auto_action]})")
            else:
                user_input = input(f"Enter action (0-5) [last: {last_action}]: ").strip().lower()

            # Handle special commands
            if user_input in ['q', 'quit', 'exit']:
                print("\nExiting test...")
                break

            if user_input.startswith('auto'):
                parts = user_input.split()
                if len(parts) == 2 and parts[1].isdigit():
                    auto_action = int(parts[1])
                    if 0 <= auto_action <= 5:
                        auto_mode = True
                        print(f"Auto mode enabled: Action {auto_action} ({action_names[auto_action]})")
                        user_input = str(auto_action)
                    else:
                        print("Invalid action! Must be 0-5")
                        continue
                else:
                    print("Usage: auto <action> (e.g., 'auto 4')")
                    continue

            if user_input == 'stop':
                auto_mode = False
                print("Auto mode disabled")
                continue

            # Parse action
            if user_input == '':
                action = last_action
            elif user_input.isdigit():
                action = int(user_input)
                if action < 0 or action > 5:
                    print("Invalid action! Must be 0-5")
                    continue
            else:
                print("Invalid input! Enter 0-5, 'auto <action>', 'stop', or 'q'")
                continue

            # Execute action
            observation, reward, terminated, truncated, info = env.step(action)

            # Update stats
            frame_count += 1
            total_reward += reward
            last_action = action

            # Display feedback
            action_name = action_names[action]
            print(f"  → Executed: {action} ({action_name})")
            print(f"  → Reward: {reward}")
            print(f"  → Total Score: {total_reward:.0f}")

            if reward > 0:
                print(f"  🎯 HIT! +{reward}")

            # Check if game over
            if terminated or truncated:
                print("\n" + "=" * 70)
                print("GAME OVER!")
                print(f"Final Score: {total_reward:.0f}")
                print(f"Total Frames: {frame_count}")
                print("=" * 70)

                # Ask to restart
                restart = input("\nPlay again? (y/n): ").strip().lower()
                if restart == 'y':
                    observation, info = env.reset()
                    frame_count = 0
                    total_reward = 0
                    last_action = 0
                    auto_mode = False
                    print("\nGame reset!")
                else:
                    break

            # Small delay in auto mode to see what's happening
            if auto_mode:
                import time
                time.sleep(0.05)  # 50ms delay between frames

    except KeyboardInterrupt:
        print("\n\nInterrupted by user (Ctrl+C)")

    finally:
        env.close()
        print("\nEnvironment closed.")
        print(f"\nFinal Stats:")
        print(f"  Frames played: {frame_count}")
        print(f"  Final score: {total_reward:.0f}")


def test_all_actions():
    """Test each action systematically to see what happens."""
    print("=" * 70)
    print("SYSTEMATIC ACTION TEST")
    print("=" * 70)
    print("Testing each action 30 times to see behavior...\n")

    env = gym.make("ALE/SpaceInvaders-v5", render_mode="rgb_array")

    action_names = {
        0: "NOOP",
        1: "FIRE",
        2: "RIGHT",
        3: "LEFT",
        4: "RIGHTFIRE",
        5: "LEFTFIRE"
    }

    for action in range(6):
        print(f"\nTesting Action {action}: {action_names[action]}")
        print("-" * 40)

        observation, info = env.reset(seed=42)
        total_reward = 0

        for i in range(30):
            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            if reward > 0:
                print(f"  Frame {i+1}: Reward +{reward} (Total: {total_reward})")

        print(f"  Total reward after 30 frames: {total_reward}")

    env.close()
    print("\n" + "=" * 70)
    print("Test complete!")


if __name__ == "__main__":
    print("\nSpace Invaders Control Tester")
    print("1. Manual test (play yourself)")
    print("2. Systematic test (test each action automatically)")

    choice = input("\nSelect mode (1 or 2): ").strip()

    if choice == "1":
        test_space_invaders()
    elif choice == "2":
        test_all_actions()
    else:
        print("Invalid choice. Running manual test...")
        test_space_invaders()
