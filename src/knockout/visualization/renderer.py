"""Pygame renderer for square arena visualization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from knockout.core.config import GameConfig, DEFAULTS
from knockout.core.physics_engine import PhysicsEngine

# Only import pygame when needed (optional dependency)
try:
    import pygame

    HAS_PYGAME = True
except ImportError:
    HAS_PYGAME = False


@dataclass
class ScreenConfig:
    width: int = 900
    height: int = 700
    margin: int = 50
    fps: int = 30


class Renderer:
    """Pygame renderer for the knockout game with square arena."""

    def __init__(
        self,
        config: GameConfig = DEFAULTS,
        screen_cfg: Optional[ScreenConfig] = None,
    ):
        if not HAS_PYGAME:
            raise ImportError(
                "pygame is required for visualization. Install with: pip install pygame"
            )

        self.config = config
        self.screen_cfg = screen_cfg or ScreenConfig()

        pygame.init()
        self.screen = pygame.display.set_mode((self.screen_cfg.width, self.screen_cfg.height))
        pygame.display.set_caption("Knockout v2")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("arial", 16)
        self.big_font = pygame.font.SysFont("arial", 28, bold=True)

    def world_to_screen(self, world_pos: tuple[float, float]) -> tuple[int, int]:
        """Convert world coordinates to screen pixel coordinates."""
        # Map arena square to screen
        cx = self.screen_cfg.width // 2
        cy = self.screen_cfg.height // 2
        scale = self._get_scale()

        sx = int(cx + world_pos[0] * scale)
        sy = int(cy - world_pos[1] * scale)  # Y flipped
        return (sx, sy)

    def _get_scale(self) -> float:
        """Calculate world-to-screen scale factor."""
        half_width = self.config.ARENA_HALF_WIDTH
        return min(
            (self.screen_cfg.width - 2 * self.screen_cfg.margin),
            (self.screen_cfg.height - 2 * self.screen_cfg.margin),
        ) / (2 * half_width)

    def draw_frame(
        self,
        engine: PhysicsEngine,
        extra_text: str = "",
        actions: dict[str, tuple[float, float]] | None = None,
    ) -> None:
        """Draw one frame of the game state.

        Args:
            engine: The physics engine with current game state.
            extra_text: Optional HUD text to display.
            actions: Optional dict mapping agent_id -> (angle_degrees, power)
                     to draw action direction/power arrows on penguins.
        """
        self.screen.fill((20, 26, 36))  # Dark background

        self.scale = scale = self._get_scale()
        cx = self.screen_cfg.width // 2
        cy = self.screen_cfg.height // 2

        # Use current (possibly shrunk) half_width from physics engine
        half_width = engine.ice_sheet.half_width
        arena_hw_px = int(half_width * scale)

        # Draw arena (square)
        arena_rect = pygame.Rect(
            cx - arena_hw_px, cy - arena_hw_px, 2 * arena_hw_px, 2 * arena_hw_px
        )
        pygame.draw.rect(self.screen, (40, 60, 90), arena_rect)  # Fill
        pygame.draw.rect(self.screen, (100, 140, 180), arena_rect, 2)  # Border

        # Draw grid lines (horizontal and vertical at 25%, 50%, 75%)
        for frac in [0.25, 0.5, 0.75]:
            offset = int(arena_hw_px * frac)
            # Vertical lines
            pygame.draw.line(
                self.screen, (50, 70, 100),
                (cx - offset, cy - arena_hw_px), (cx - offset, cy + arena_hw_px), 1,
            )
            pygame.draw.line(
                self.screen, (50, 70, 100),
                (cx + offset, cy - arena_hw_px), (cx + offset, cy + arena_hw_px), 1,
            )
            # Horizontal lines
            pygame.draw.line(
                self.screen, (50, 70, 100),
                (cx - arena_hw_px, cy - offset), (cx + arena_hw_px, cy - offset), 1,
            )
            pygame.draw.line(
                self.screen, (50, 70, 100),
                (cx - arena_hw_px, cy + offset), (cx + arena_hw_px, cy + offset), 1,
            )

        # Draw center dot
        pygame.draw.circle(self.screen, (80, 100, 130), (cx, cy), 3)

        # Draw penguins — visual radius scaled from physics radius
        penguin_radius_px = max(3, int(self.config.PENGUIN_RADIUS * self.scale))

        for penguin in engine.penguins.values():
            pos = self.world_to_screen(penguin.position)

            if penguin.alive:
                # Team colors
                if penguin.team_id == 0:
                    color = (70, 200, 255)  # Cyan for Team A
                    border_color = (40, 160, 220)
                else:
                    color = (255, 120, 80)  # Orange-red for Team B
                    border_color = (220, 80, 40)

                pygame.draw.circle(self.screen, color, pos, penguin_radius_px)
                pygame.draw.circle(self.screen, border_color, pos, penguin_radius_px, 2)

                # Draw velocity indicator (small line showing direction)
                vx, vy = penguin.velocity
                speed = np.sqrt(vx**2 + vy**2)
                if speed > 1.0:
                    end_x = pos[0] + int(vx * scale * 0.3)
                    end_y = pos[1] - int(vy * scale * 0.3)
                    pygame.draw.line(self.screen, (255, 255, 255), pos, (end_x, end_y), 1)

                # Draw action arrow if actions provided for this penguin
                if actions and penguin.agent_id in actions:
                    angle_deg, power = actions[penguin.agent_id]
                    angle_rad = np.radians(angle_deg)
                    # Arrow length proportional to power (max ~50px at full power)
                    max_arrow_len = 50.0
                    arrow_len = max_arrow_len * (power / self.config.MAX_LAUNCH_FORCE)
                    if arrow_len > 3:  # Only draw if meaningful
                        dx = np.cos(angle_rad) * arrow_len
                        dy = np.sin(angle_rad) * arrow_len
                        end = (int(pos[0] + dx), int(pos[1] - dy))  # Y flipped
                        # Shaft
                        pygame.draw.line(self.screen, (255, 255, 80), pos, end, 2)
                        # Arrowhead — two small lines angled back from the tip
                        head_len = min(8.0, arrow_len * 0.35)
                        for offset in [2.5, -2.5]:  # ~145 deg spread
                            ha = angle_rad + np.pi + np.radians(25 * (1 if offset > 0 else -1))
                            hx = int(end[0] + np.cos(ha) * head_len)
                            hy = int(end[1] - np.sin(ha) * head_len)
                            pygame.draw.line(self.screen, (255, 255, 80), end, (hx, hy), 2)

                # Label
                label = self.font.render(penguin.agent_id[-1], True, (255, 255, 255))
                self.screen.blit(label, (pos[0] - 4, pos[1] - 8))
            else:
                # Dead penguin: gray with X
                pygame.draw.circle(self.screen, (80, 80, 80), pos, penguin_radius_px, 2)
                pygame.draw.line(
                    self.screen,
                    (120, 80, 80),
                    (pos[0] - 4, pos[1] - 4),
                    (pos[0] + 4, pos[1] + 4),
                    2,
                )
                pygame.draw.line(
                    self.screen,
                    (120, 80, 80),
                    (pos[0] - 4, pos[1] + 4),
                    (pos[0] + 4, pos[1] - 4),
                    2,
                )

        # HUD
        team_a_alive = engine.get_alive_count(0)
        team_b_alive = engine.get_alive_count(1)

        hud_lines = [
            f"Step: {engine.step_count}",
            f"Team A (cyan): {team_a_alive}/3",
            f"Team B (orange): {team_b_alive}/3",
        ]
        if extra_text:
            hud_lines.append(extra_text)

        for i, text in enumerate(hud_lines):
            surface = self.font.render(text, True, (220, 220, 220))
            self.screen.blit(surface, (10, 10 + 20 * i))

        # Game over overlay
        if engine.is_game_over():
            winner = engine.get_winner()
            if winner == 0:
                msg = "Team A Wins!"
                color = (70, 200, 255)
            elif winner == 1:
                msg = "Team B Wins!"
                color = (255, 120, 80)
            else:
                msg = "Draw!"
                color = (200, 200, 200)

            text_surface = self.big_font.render(msg, True, color)
            text_rect = text_surface.get_rect(center=(cx, cy))
            self.screen.blit(text_surface, text_rect)

        pygame.display.flip()

    def tick(self) -> None:
        """Wait for next frame."""
        self.clock.tick(self.screen_cfg.fps)

    def check_quit(self) -> bool:
        """Check for quit events. Returns True if should quit."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                return True
        return False

    def close(self) -> None:
        """Clean up pygame."""
        pygame.quit()
