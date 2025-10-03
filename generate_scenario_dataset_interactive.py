#!/usr/bin/env python3
"""
Interactive scenario-based dataset generation.
Play the games yourself and frames will be automatically captured when they match scenarios.

Controls:
- Arrow keys / WASD: Move
- Space: Fire/Action
- R: Reset game
- Q: Quit current game
- ESC: Exit completely

Usage:
    python generate_scenario_dataset_interactive.py
"""

import sys
import time
import pygame
from pathlib import Path
from automatic_benchmark.dataset.scenario_based_creator import ScenarioBasedDatasetCreator
from ocatari_ground_truth import OCAtariGroundTruth
from automatic_benchmark.config import GAMES


class InteractiveDatasetGenerator:
    """
    Interactive dataset generator - play games and auto-capture matching scenarios.
    """

    def __init__(self, output_dir: str = "./benchmark_v2.0_scenarios"):
        self.creator = ScenarioBasedDatasetCreator(output_dir=output_dir)

        # Track collected scenarios per game
        self.collected_scenarios = {game: set() for game in GAMES}

        # Initialize pygame for rendering and input
        pygame.init()
        self.screen = pygame.display.set_mode((640, 480))
        pygame.display.set_caption("Interactive Dataset Generator")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 24)
        self.large_font = pygame.font.Font(None, 36)

    def get_human_action(self, action_space_size: int) -> int:
        """
        Get action from keyboard input.
        Maps keyboard to Atari actions.
        """
        keys = pygame.key.get_pressed()

        # Common Atari action mappings:
        # 0: NOOP
        # 1: FIRE
        # 2: UP (Pong) or RIGHT (Breakout/SpaceInvaders)
        # 3: DOWN (Pong) or LEFT (Breakout/SpaceInvaders)
        # 4: UP+FIRE (some games)
        # 5: DOWN+FIRE (some games)

        # Fire action (Space)
        if keys[pygame.K_SPACE]:
            return 1

        # Movement
        if keys[pygame.K_UP] or keys[pygame.K_w]:
            return 2
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]:
            return 3
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            return 2  # For Breakout/SpaceInvaders (RIGHT)
        elif keys[pygame.K_LEFT] or keys[pygame.K_a]:
            return 3  # For Breakout/SpaceInvaders (LEFT)

        # Default: NOOP
        return 0

    def draw_hud(self, game: str, scenarios_total: int, scenarios_collected: int,
                 last_capture: str = None):
        """Draw heads-up display showing progress."""
        # Background for HUD
        hud_rect = pygame.Rect(10, 10, 620, 120)
        pygame.draw.rect(self.screen, (0, 0, 0), hud_rect)
        pygame.draw.rect(self.screen, (255, 255, 255), hud_rect, 2)

        # Game name
        text = self.large_font.render(f"Playing: {game}", True, (0, 255, 0))
        self.screen.blit(text, (20, 20))

        # Progress
        progress_text = f"Progress: {scenarios_collected}/{scenarios_total} scenarios"
        text = self.font.render(progress_text, True, (255, 255, 255))
        self.screen.blit(text, (20, 55))

        # Progress bar
        bar_width = 600
        bar_height = 20
        progress_ratio = scenarios_collected / scenarios_total if scenarios_total > 0 else 0

        # Bar background
        bar_rect = pygame.Rect(10, 85, bar_width, bar_height)
        pygame.draw.rect(self.screen, (50, 50, 50), bar_rect)

        # Bar fill
        fill_rect = pygame.Rect(10, 85, int(bar_width * progress_ratio), bar_height)
        pygame.draw.rect(self.screen, (0, 255, 0), fill_rect)

        # Bar border
        pygame.draw.rect(self.screen, (255, 255, 255), bar_rect, 2)

        # Last capture notification
        if last_capture:
            capture_text = f"✓ Captured: {last_capture}"
            text = self.font.render(capture_text, True, (0, 255, 0))
            self.screen.blit(text, (20, 110))

    def draw_controls_help(self):
        """Draw control hints at bottom of screen."""
        controls = [
            "Controls: Arrow Keys/WASD=Move  Space=Fire  R=Reset  Q=Next Game  ESC=Exit"
        ]

        y = 450
        for control in controls:
            text = self.font.render(control, True, (200, 200, 200))
            self.screen.blit(text, (10, y))
            y += 25

    def draw_scenario_list(self, game: str, scenarios: list):
        """Draw list of scenarios with checkmarks for collected ones."""
        y = 140
        title = self.font.render("Scenarios:", True, (255, 255, 0))
        self.screen.blit(title, (10, y))
        y += 30

        collected = self.collected_scenarios[game]

        for i, scenario in enumerate(scenarios):
            if i >= 10:  # Limit to 10 visible
                break

            status = "✓" if scenario.name in collected else "○"
            color = (0, 255, 0) if scenario.name in collected else (150, 150, 150)

            text = self.font.render(f"{status} {scenario.description}", True, color)
            self.screen.blit(text, (15, y))
            y += 25

    def play_game_interactive(self, game: str):
        """
        Play a game interactively and auto-capture matching scenarios.

        Args:
            game: Game name ('Pong', 'Breakout', 'SpaceInvaders')
        """
        print(f"\n{'=' * 80}")
        print(f"INTERACTIVE MODE: {game}")
        print(f"{'=' * 80}")
        print("Play the game and scenarios will be automatically captured!")
        print("Controls: Arrow Keys/WASD=Move, Space=Fire, R=Reset, Q=Next Game")
        print()

        # Get scenarios for this game
        scenarios = self.creator.scenarios[game]
        scenarios_needed = [s for s in scenarios if s.name not in self.collected_scenarios[game]]

        if not scenarios_needed:
            print(f"✓ All {len(scenarios)} scenarios already collected for {game}!")
            return

        # Initialize OCAtari
        ocatari = OCAtariGroundTruth(game)
        ocatari.reset()

        running = True
        frame_count = 0
        last_capture = None
        last_capture_time = 0

        while running and scenarios_needed:
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return False  # Signal to exit completely
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        return False  # Exit completely
                    elif event.key == pygame.K_q:
                        return True  # Move to next game
                    elif event.key == pygame.K_r:
                        ocatari.reset()
                        print("Game reset!")

            # Get human action
            action = self.get_human_action(ocatari.env.action_space.n)

            # Take step
            obs, reward, terminated, truncated, info = ocatari.step(action)

            # Reset if game over
            if terminated or truncated:
                ocatari.reset()

            # Get frame and objects
            frame, objects_info = ocatari.get_frame_and_objects()

            # Convert ObjectInfo to dict format for validators
            objects = []
            for obj_info in objects_info:
                objects.append({
                    'category': obj_info.category,
                    'position': obj_info.position,
                    'velocity': obj_info.velocity,
                    'size': obj_info.size
                })

            # Check if frame matches any needed scenario
            for scenario in scenarios_needed:
                if scenario.validator(objects):
                    # Capture this frame!
                    print(f"✓ CAPTURED: {scenario.name} - {scenario.description}")

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
                    scenarios_needed = [s for s in scenarios if s.name not in self.collected_scenarios[game]]

                    last_capture = scenario.description
                    last_capture_time = time.time()

                    break  # Only capture one scenario per frame

            # Render game frame
            frame_surface = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
            frame_surface = pygame.transform.scale(frame_surface, (640, 420))
            self.screen.fill((0, 0, 0))
            self.screen.blit(frame_surface, (0, 0))

            # Draw HUD overlay
            self.draw_hud(
                game,
                len(scenarios),
                len(self.collected_scenarios[game]),
                last_capture if (time.time() - last_capture_time < 3) else None
            )

            # Draw controls at bottom
            self.draw_controls_help()

            pygame.display.flip()
            self.clock.tick(60)  # 60 FPS

            frame_count += 1

        if not scenarios_needed:
            print(f"\n✓ All {len(scenarios)} scenarios collected for {game}!")
            print("Press Q to continue to next game...")

            # Wait for Q or ESC
            waiting = True
            while waiting:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        return False
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            return False
                        elif event.key == pygame.K_q:
                            return True

                # Keep displaying success screen
                self.screen.fill((0, 0, 0))
                success_text = self.large_font.render(f"✓ {game} Complete!", True, (0, 255, 0))
                self.screen.blit(success_text, (200, 200))
                press_text = self.font.render("Press Q to continue...", True, (255, 255, 255))
                self.screen.blit(press_text, (230, 250))
                pygame.display.flip()
                self.clock.tick(30)

        return True

    def generate_interactive_dataset(self):
        """
        Main loop for interactive dataset generation.
        Play through all games collecting scenarios.
        """
        print("=" * 80)
        print("INTERACTIVE SCENARIO-BASED DATASET GENERATION")
        print("=" * 80)
        print("Play each game and matching scenarios will be automatically captured!")
        print(f"Total: 30 scenarios (10 per game)")
        print()

        for game in GAMES:
            continue_playing = self.play_game_interactive(game)
            if not continue_playing:
                print("\nExiting...")
                break

        # Create dataset index after all games
        self.finalize_dataset()

        pygame.quit()

    def finalize_dataset(self):
        """Create final dataset metadata and index."""
        print("\n" + "=" * 80)
        print("FINALIZING DATASET")
        print("=" * 80)

        dataset_metadata = {
            'total_frames': 0,
            'games': {},
            'generation_method': 'interactive_human_played',
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
        metadata_path = self.creator.output_dir / "dataset_metadata.json"
        import json
        with open(metadata_path, 'w') as f:
            json.dump(dataset_metadata, f, indent=2)

        # Create dataset index
        self.creator._create_dataset_index(dataset_metadata)

        print(f"\n✓ Dataset generation complete!")
        print(f"Total frames collected: {dataset_metadata['total_frames']}/30")
        for game in GAMES:
            count = len(self.collected_scenarios[game])
            print(f"  - {game}: {count}/10")
        print(f"\nDataset saved to: {self.creator.output_dir}")
        print("=" * 80)


def main():
    print("Starting INTERACTIVE scenario-based dataset generation...")
    print()
    print("You will play each game and the system will automatically capture frames")
    print("when they match the required scenarios.")
    print()
    print("Games to play:")
    print("  - Pong (10 scenarios)")
    print("  - Breakout (10 scenarios)")
    print("  - Space Invaders (10 scenarios)")
    print()
    input("Press Enter to start playing...")

    generator = InteractiveDatasetGenerator(output_dir="./benchmark_v2.0_scenarios")
    generator.generate_interactive_dataset()


if __name__ == '__main__':
    main()
