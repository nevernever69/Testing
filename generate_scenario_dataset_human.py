#!/usr/bin/env python3
"""
Human-played scenario-based dataset generation.
Play the games with keyboard controls and frames will be automatically captured.

This version uses OCAtari's built-in rendering and shows progress in terminal.

Controls vary by game:
- Pong: UP/DOWN arrow keys to move paddle
- Breakout: LEFT/RIGHT arrow keys to move paddle, FIRE to launch
- SpaceInvaders: LEFT/RIGHT to move, FIRE to shoot

Usage:
    python generate_scenario_dataset_human.py
"""

import sys
import time
from pathlib import Path
from automatic_benchmark.dataset.scenario_based_creator import ScenarioBasedDatasetCreator
from ocatari_ground_truth import OCAtariGroundTruth
from automatic_benchmark.config import GAMES


class HumanPlayedDatasetGenerator:
    """
    Dataset generator where human plays the game and scenarios are auto-captured.
    """

    def __init__(self, output_dir: str = "./benchmark_v2.0_scenarios"):
        self.creator = ScenarioBasedDatasetCreator(output_dir=output_dir)

        # Track collected scenarios per game
        self.collected_scenarios = {game: set() for game in GAMES}

    def print_progress(self, game: str, scenarios: list):
        """Print current progress for this game."""
        print(f"\n{'=' * 60}")
        print(f"  {game} - Progress: {len(self.collected_scenarios[game])}/{len(scenarios)}")
        print(f"{'=' * 60}")

        collected = self.collected_scenarios[game]

        for scenario in scenarios:
            status = "✓" if scenario.name in collected else "○"
            print(f"{status} {scenario.description}")

        print(f"{'=' * 60}\n")

    def play_game_human(self, game: str):
        """
        Let human play the game and auto-capture matching scenarios.

        Args:
            game: Game name ('Pong', 'Breakout', 'SpaceInvaders')
        """
        print(f"\n{'=' * 80}")
        print(f"  PLAYING: {game}")
        print(f"{'=' * 80}")

        # Get scenarios for this game
        scenarios = self.creator.scenarios[game]
        scenarios_needed = [s for s in scenarios if s.name not in self.collected_scenarios[game]]

        if not scenarios_needed:
            print(f"✓ All {len(scenarios)} scenarios already collected for {game}!")
            return True

        # Show what we need to collect
        self.print_progress(game, scenarios)

        # Game-specific controls
        controls = {
            'Pong': 'UP/DOWN arrows to move paddle',
            'Breakout': 'LEFT/RIGHT arrows to move paddle, SPACE to launch ball',
            'SpaceInvaders': 'LEFT/RIGHT arrows to move, SPACE to shoot'
        }

        print(f"Controls: {controls.get(game, 'Use arrow keys and spacebar')}")
        print("Speed Control: +/- keys to adjust game speed (starts at 15 FPS)")
        print("The game window will open - play until all scenarios are captured.")
        print("Close the game window when done with this game.\n")

        input("Press Enter to start playing...")

        # Initialize OCAtari with rgb_array mode (for frame capture) but we'll also render to screen
        ocatari = OCAtariGroundTruth(game, render_mode='rgb_array')
        obs, info = ocatari.reset()

        frame_count = 0
        check_interval = 5  # Check every N frames to reduce overhead
        game_speed = 10  # FPS - adjustable with +/- keys

        # Import pygame for keyboard handling and rendering
        import pygame
        import numpy as np
        pygame.init()

        # Create window for game display (scale up the 160x210 game frame)
        game_display_width = 640  # 160 * 4
        game_display_height = 840  # 210 * 4
        display_screen = pygame.display.set_mode((game_display_width, game_display_height))
        pygame.display.set_caption(f"{game} - Play to Capture Scenarios")
        font = pygame.font.Font(None, 24)
        clock = pygame.time.Clock()

        def get_action_from_keys(game_name: str) -> int:
            """Get action based on keyboard state."""
            keys = pygame.key.get_pressed()

            if game_name == 'Pong':
                # Pong: 0=NOOP, 2=UP, 3=DOWN
                if keys[pygame.K_UP]:
                    return 2
                elif keys[pygame.K_DOWN]:
                    return 3
            elif game_name == 'Breakout':
                # Breakout: 0=NOOP, 1=FIRE, 2=RIGHT, 3=LEFT
                if keys[pygame.K_SPACE]:
                    return 1
                elif keys[pygame.K_RIGHT]:
                    return 2
                elif keys[pygame.K_LEFT]:
                    return 3
            elif 'Space' in game_name or 'Invader' in game_name:
                # SpaceInvaders: 0=NOOP, 1=FIRE, 2=RIGHT, 3=LEFT
                if keys[pygame.K_SPACE]:
                    return 1
                elif keys[pygame.K_RIGHT]:
                    return 2
                elif keys[pygame.K_LEFT]:
                    return 3

            return 0  # NOOP

        try:
            running = True
            while running and scenarios_needed:
                # Handle pygame events
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                    elif event.type == pygame.KEYDOWN:
                        # Speed control: +/- or =/- keys
                        if event.key == pygame.K_EQUALS or event.key == pygame.K_PLUS:
                            game_speed = min(60, game_speed + 5)
                            print(f"Speed increased to {game_speed} FPS")
                        elif event.key == pygame.K_MINUS or event.key == pygame.K_UNDERSCORE:
                            game_speed = max(5, game_speed - 5)
                            print(f"Speed decreased to {game_speed} FPS")

                # Get action from keyboard
                action = get_action_from_keys(game)

                # Take game step
                obs, reward, terminated, truncated, info = ocatari.step(action)

                # Reset if game over
                if terminated or truncated:
                    obs, info = ocatari.reset()

                # Check frames periodically
                if frame_count % check_interval == 0:
                    # Get frame and objects
                    frame, objects_info = ocatari.get_frame_and_objects()

                    # Convert ObjectInfo to dict format
                    objects = []
                    for obj_info in objects_info:
                        objects.append({
                            'category': obj_info.category,
                            'position': obj_info.position,
                            'velocity': obj_info.velocity,
                            'size': obj_info.size
                        })

                    # Validate frame before checking scenarios
                    if frame is None or frame.size == 0 or frame.shape[0] == 0 or frame.shape[1] == 0:
                        continue

                    # Check if frame matches any needed scenario
                    for scenario in scenarios_needed:
                        if scenario.validator(objects):
                            # Capture this frame!
                            print(f"\n✓ CAPTURED: {scenario.name} - {scenario.description}")

                            # Save frame
                            frame_data = {
                                'frame': frame,
                                'objects': objects,
                                'scenario_name': scenario.name,
                                'scenario_description': scenario.description,
                                'attempts': frame_count
                            }

                            index = len(self.collected_scenarios[game])
                            self.creator._save_frame(game, index, frame_data)

                            # Mark as collected
                            self.collected_scenarios[game].add(scenario.name)
                            scenarios_needed = [s for s in scenarios
                                               if s.name not in self.collected_scenarios[game]]

                            # Show updated progress
                            self.print_progress(game, scenarios)

                            break  # Only capture one scenario per frame

                # Render game frame to display
                # Get current observation and scale it up
                current_frame = obs  # This is the RGB array from last step

                # Convert numpy array to pygame surface
                # OCAtari returns (210, 160, 3), need to transpose to (160, 210, 3) for pygame
                if current_frame is not None and current_frame.size > 0:
                    # Scale up the frame
                    surf = pygame.surfarray.make_surface(np.transpose(current_frame, (1, 0, 2)))
                    surf = pygame.transform.scale(surf, (game_display_width, game_display_height - 200))
                    display_screen.fill((0, 0, 0))
                    display_screen.blit(surf, (0, 0))

                    # Draw info overlay at bottom
                    info_bg = pygame.Rect(0, game_display_height - 200, game_display_width, 200)
                    pygame.draw.rect(display_screen, (30, 30, 30), info_bg)

                    # Draw title
                    title = font.render(f"{game}", True, (255, 255, 0))
                    display_screen.blit(title, (10, game_display_height - 190))

                    # Draw progress
                    progress_text = f"Collected: {len(self.collected_scenarios[game])}/{len(scenarios)}"
                    progress = font.render(progress_text, True, (0, 255, 0))
                    display_screen.blit(progress, (10, game_display_height - 160))

                    # Draw speed indicator
                    speed_text = f"Speed: {game_speed} FPS (+/- to adjust)"
                    speed = font.render(speed_text, True, (200, 200, 255))
                    display_screen.blit(speed, (10, game_display_height - 130))

                    # Draw controls hint
                    y = game_display_height - 100
                    hints = {
                        'Pong': ['UP/DOWN: Move paddle'],
                        'Breakout': ['LEFT/RIGHT: Move paddle, SPACE: Fire'],
                        'SpaceInvaders': ['LEFT/RIGHT: Move ship, SPACE: Fire']
                    }

                    for hint in hints.get(game, ['Use arrow keys']):
                        text = font.render(hint, True, (200, 200, 200))
                        display_screen.blit(text, (10, y))
                        y += 25

                    # Draw exit hint
                    exit_text = font.render("Close window to stop", True, (150, 150, 150))
                    display_screen.blit(exit_text, (10, game_display_height - 30))

                pygame.display.flip()
                clock.tick(game_speed)  # Adjustable FPS for human playability

                frame_count += 1

        except KeyboardInterrupt:
            print("\n\nInterrupted by user.")
        finally:
            pygame.quit()
            ocatari.close()

        if not scenarios_needed:
            print(f"\n{'=' * 80}")
            print(f"✓ All {len(scenarios)} scenarios collected for {game}!")
            print(f"{'=' * 80}\n")
        else:
            print(f"\n⚠ Still need {len(scenarios_needed)} scenarios for {game}:")
            for s in scenarios_needed:
                print(f"  - {s.description}")
            print()

        return True

    def generate_human_played_dataset(self):
        """
        Main loop for human-played dataset generation.
        """
        print("=" * 80)
        print("  HUMAN-PLAYED SCENARIO-BASED DATASET GENERATION")
        print("=" * 80)
        print()
        print("You will play each game and matching scenarios will be automatically captured!")
        print()
        print("Games to play:")
        print("  1. Pong (10 scenarios)")
        print("  2. Breakout (10 scenarios)")
        print("  3. Space Invaders (10 scenarios)")
        print()
        print("Total: 30 scenarios")
        print()
        print("The system will detect when you reach each scenario and save it automatically.")
        print("You can take breaks between games.")
        print()

        for i, game in enumerate(GAMES, 1):
            print(f"\n{'=' * 80}")
            print(f"  GAME {i}/{len(GAMES)}: {game}")
            print(f"{'=' * 80}")

            proceed = input(f"\nReady to play {game}? (y/n/q to quit): ").lower()

            if proceed == 'q':
                print("Quitting...")
                break
            elif proceed == 'n':
                print(f"Skipping {game}")
                continue

            self.play_game_human(game)

        # Create dataset index after all games
        self.finalize_dataset()

    def finalize_dataset(self):
        """Create final dataset metadata and index."""
        print("\n" + "=" * 80)
        print("  FINALIZING DATASET")
        print("=" * 80)

        dataset_metadata = {
            'total_frames': 0,
            'games': {},
            'generation_method': 'human_played',
            'scenarios_per_game': 'Pong:10, Breakout:10, SpaceInvaders:10'
        }

        for game in GAMES:
            collected = self.collected_scenarios[game]
            dataset_metadata['games'][game] = {
                'frames_collected': len(collected),
                'scenarios': list(collected)
            }
            dataset_metadata['total_frames'] += len(collected)

        # Save metadata
        import json
        metadata_path = self.creator.output_dir / "dataset_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(dataset_metadata, f, indent=2)

        # Create dataset index
        self.creator._create_dataset_index(dataset_metadata)

        print(f"\n✓ Dataset generation complete!")
        print(f"\nFinal Statistics:")
        print(f"  Total frames collected: {dataset_metadata['total_frames']}/30")
        for game in GAMES:
            count = len(self.collected_scenarios[game])
            status = "✓" if count == 10 else "⚠"
            print(f"  {status} {game}: {count}/10")

        if dataset_metadata['total_frames'] == 30:
            print(f"\n✓✓✓ Perfect! All 30 scenarios collected! ✓✓✓")
        else:
            missing = 30 - dataset_metadata['total_frames']
            print(f"\n⚠ {missing} scenarios still missing. You can run again to collect them.")

        print(f"\nDataset saved to: {self.creator.output_dir}")
        print("=" * 80)


def main():
    print("\n" + "=" * 80)
    print("  HUMAN-PLAYED SCENARIO DATASET GENERATOR")
    print("=" * 80)
    print()
    print("This tool lets you play Atari games while automatically capturing")
    print("frames that match specific scenarios for benchmarking.")
    print()
    print("Benefits of human-played dataset:")
    print("  ✓ More realistic gameplay")
    print("  ✓ Better strategic situations")
    print("  ✓ You control when scenarios occur")
    print("  ✓ Guaranteed to reach all scenarios")
    print()
    print("You'll play 3 games (Pong, Breakout, Space Invaders)")
    print("and collect 10 scenarios from each (30 total).")
    print()

    ready = input("Ready to start? (y/n): ").lower()

    if ready != 'y':
        print("Exiting...")
        return

    generator = HumanPlayedDatasetGenerator(output_dir="./benchmark_v2.0_scenarios")
    generator.generate_human_played_dataset()


if __name__ == '__main__':
    main()
