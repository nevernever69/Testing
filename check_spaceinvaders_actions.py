#!/usr/bin/env python3
"""
Check the actual action space for Space Invaders v4 and v5
"""
import gymnasium as gym

# Register ALE games
try:
    import ale_py
    gym.register_envs(ale_py)
except:
    pass

def check_actions(env_name):
    """Check the action meanings for a given environment"""
    print(f"\n{'='*70}")
    print(f"Environment: {env_name}")
    print(f"{'='*70}")

    try:
        env = gym.make(env_name)

        # Get action space info
        print(f"Action space: {env.action_space}")
        print(f"Number of actions: {env.action_space.n}")

        # Get action meanings
        if hasattr(env.unwrapped, 'get_action_meanings'):
            action_meanings = env.unwrapped.get_action_meanings()
            print(f"\nAction Meanings:")
            for i, meaning in enumerate(action_meanings):
                print(f"  {i}: {meaning}")
        else:
            print("\nNo action meanings available")

        env.close()

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    # Check both v4 and v5
    check_actions("SpaceInvaders-v4")
    check_actions("ALE/SpaceInvaders-v5")

    print(f"\n{'='*70}")
    print("COMPARISON")
    print(f"{'='*70}")
