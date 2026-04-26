#!/usr/bin/env python3
"""Interactive Pygame-based play mode for Penguin Knockout.

Play as Team A (cyan, penguins 0-2) against a CPU opponent (Team B, orange, penguins 3-5).
Click-and-drag aiming with animated physics -- see penguins slide and collide in real-time.

The opponent is chosen from the central lineup (``eval/lineup.py``) by name.
Use ``--list`` to print the available bots.

Usage:
    python scripts/play.py
    python scripts/play.py --seed 42
    python scripts/play.py --opponent ppo
    python scripts/play.py --opponent self_play --seed 7
    python scripts/play.py --opponent llm_anthropic
    python scripts/play.py --list

Controls:
    Click + drag on a Team A penguin  = aim (drag direction = launch direction)
    SPACE / ENTER                     = confirm round (launch all penguins)
    R                                 = reset current round's aims
    Right-click on a Team A penguin   = auto-aim (heuristic decides)
    ESC                               = quit
"""

from __future__ import annotations

import argparse
import math
import sys
from enum import Enum, auto

import numpy as np
import pygame

from eval.lineup import (
    LINEUP_BY_NAME,
    build_team,
    lineup_summary,
)
from knockout.agents.base import Agent
from knockout.agents.heuristic_agent import HeuristicAgent
from knockout.core.config import GameConfig, DEFAULTS
from knockout.core.physics_engine import PhysicsEngine
from knockout.env.observations import ObservationBuilder
from knockout.visualization.renderer import ScreenConfig


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Colors
COL_BG = (20, 26, 36)
COL_TEXT = (220, 220, 220)
COL_DIM_TEXT = (140, 140, 140)
COL_TEAM_A = (70, 200, 255)
COL_TEAM_B = (255, 120, 80)
COL_SELECTED = (255, 255, 100)
COL_ARROW_PREVIEW = (255, 255, 80)
COL_ARROW_CONFIRMED = (100, 255, 100)
COL_HUD_BG = (10, 14, 22, 180)
COL_INSTRUCTIONS = (180, 180, 200)

# Arrow drawing
MAX_ARROW_PX = 80  # max arrow length in pixels at full power
DRAG_SCALE = 1.0   # pixels-of-drag to power mapping factor (tuned below)

# Penguin click radius tolerance (in screen pixels)
CLICK_RADIUS_TOLERANCE = 20


class Phase(Enum):
    AIM = auto()
    SIMULATING = auto()
    SETTLE_PAUSE = auto()
    GAME_OVER = auto()


# ---------------------------------------------------------------------------
# Main play class
# ---------------------------------------------------------------------------

class PlayMode:
    """Interactive Pygame play mode with click-and-drag aiming."""

    def __init__(
        self,
        seed: int | None = None,
        config: GameConfig = DEFAULTS,
        opponent_name: str = "heuristic",
    ):
        self.config = config
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.opponent_name = opponent_name

        # Physics engine (used directly for frame-by-frame control)
        self.engine = PhysicsEngine(seed=seed, config=config)
        self.engine.initialize_game()

        # Observation builder (for heuristic agents and RL agents)
        self.obs_builder = ObservationBuilder(config=config)

        # CPU agents (Team B) -- loaded from the lineup spec
        self.cpu_agents: dict[str, Agent] = self._load_opponent(opponent_name, seed)

        # Auto-aim agents (Team A, used for right-click)
        self.auto_agents: dict[str, HeuristicAgent] = {
            f"penguin_{i}": HeuristicAgent(
                agent_id=f"penguin_{i}",
                seed=(seed + 100 + i) if seed is not None else None,
                config=config,
            )
            for i in range(3)
        }

        # Renderer
        screen_cfg = ScreenConfig(width=1000, height=750, fps=60)
        self.screen_cfg = screen_cfg

        pygame.init()
        self.screen = pygame.display.set_mode((screen_cfg.width, screen_cfg.height))
        pygame.display.set_caption("Knockout v2 -- Play Mode")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("arial", 16)
        self.font_bold = pygame.font.SysFont("arial", 16, bold=True)
        self.big_font = pygame.font.SysFont("arial", 32, bold=True)
        self.medium_font = pygame.font.SysFont("arial", 22, bold=True)

        # Game state
        self.round_num = 0
        self.phase = Phase.AIM
        self.running = True

        # Aim state: dict of agent_id -> (angle_deg, power)
        self.confirmed_aims: dict[str, tuple[float, float]] = {}
        self.selected_penguin: str | None = None
        self.dragging = False
        self.drag_start: tuple[int, int] | None = None
        self.drag_current: tuple[int, int] | None = None

        # Animation state
        self.all_actions: dict[str, tuple[float, float]] = {}
        self.settle_steps = 0
        self.max_settle_steps = 3000

        # Settle pause
        self.settle_pause_frames = 0
        self.settle_pause_duration = 45  # ~0.75 seconds at 60fps

        # Shrink animation
        self.shrink_pending = False
        self.shrink_anim_frames = 0
        self.shrink_old_hw = 0.0
        self.shrink_new_hw = 0.0

        # Compute drag scale: we want a full-screen drag (say 300px) to map
        # to MAX_LAUNCH_FORCE
        self.drag_power_scale = config.MAX_LAUNCH_FORCE / 300.0

        # Build opponent label for HUD
        self._build_opponent_label()

    @staticmethod
    def _load_opponent(opponent_name: str, seed: int | None) -> dict[str, Agent]:
        """Load CPU opponent agents for Team B from the central lineup.

        Args:
            opponent_name: Bot name from ``LINEUP_BY_NAME`` (e.g. "heuristic", "ppo",
                "self_play", "llm_anthropic").
            seed: Random seed forwarded to ``build_team``.

        Returns:
            Dict mapping ``penguin_3..5`` agent ids to Agent instances.
        """
        spec = LINEUP_BY_NAME[opponent_name]
        # Seed=0 is acceptable when seed is None; build_team adds per-slot offsets.
        team_b = build_team(spec, team=1, seed=seed if seed is not None else 0)
        return team_b

    def _build_opponent_label(self) -> None:
        """Build the HUD label describing the current opponent."""
        spec = LINEUP_BY_NAME.get(self.opponent_name)
        if spec is not None:
            self.opponent_label = f"vs {spec.display_name}"
        else:
            self.opponent_label = f"vs {self.opponent_name}"

    # -------------------------------------------------------------------
    # Coordinate conversion
    # -------------------------------------------------------------------

    def _get_scale(self) -> float:
        """World-to-screen scale factor based on initial arena size."""
        half_width = self.config.ARENA_HALF_WIDTH
        return min(
            self.screen_cfg.width - 100,
            self.screen_cfg.height - 160,
        ) / (2 * half_width)

    def world_to_screen(self, world_pos: tuple[float, float]) -> tuple[int, int]:
        """Convert world (physics) coordinates to screen pixel coordinates."""
        cx = self.screen_cfg.width // 2
        cy = (self.screen_cfg.height + 60) // 2  # offset down for HUD
        scale = self._get_scale()
        sx = int(cx + world_pos[0] * scale)
        sy = int(cy - world_pos[1] * scale)  # Y flipped
        return (sx, sy)

    def screen_to_world(self, screen_pos: tuple[int, int]) -> tuple[float, float]:
        """Convert screen pixel coordinates to world coordinates."""
        cx = self.screen_cfg.width // 2
        cy = (self.screen_cfg.height + 60) // 2
        scale = self._get_scale()
        wx = (screen_pos[0] - cx) / scale
        wy = -(screen_pos[1] - cy) / scale
        return (wx, wy)

    # -------------------------------------------------------------------
    # Drawing helpers
    # -------------------------------------------------------------------

    def _draw_arena(self) -> None:
        """Draw the square arena with grid."""
        scale = self._get_scale()
        cx = self.screen_cfg.width // 2
        cy = (self.screen_cfg.height + 60) // 2
        half_width = self.engine.ice_sheet.half_width
        arena_hw_px = int(half_width * scale)

        # Arena fill
        arena_rect = pygame.Rect(
            cx - arena_hw_px, cy - arena_hw_px, 2 * arena_hw_px, 2 * arena_hw_px
        )
        pygame.draw.rect(self.screen, (40, 60, 90), arena_rect)

        # Grid lines
        for frac in [0.25, 0.5, 0.75]:
            offset = int(arena_hw_px * frac)
            for sign in (-1, 1):
                pygame.draw.line(
                    self.screen, (50, 70, 100),
                    (cx + sign * offset, cy - arena_hw_px),
                    (cx + sign * offset, cy + arena_hw_px), 1,
                )
                pygame.draw.line(
                    self.screen, (50, 70, 100),
                    (cx - arena_hw_px, cy + sign * offset),
                    (cx + arena_hw_px, cy + sign * offset), 1,
                )

        # Border
        pygame.draw.rect(self.screen, (100, 140, 180), arena_rect, 2)

        # Center dot
        pygame.draw.circle(self.screen, (80, 100, 130), (cx, cy), 3)

    def _draw_penguins(self) -> None:
        """Draw all penguins with team colors and labels."""
        scale = self._get_scale()
        penguin_radius_px = max(6, int(self.config.PENGUIN_RADIUS * scale))

        for penguin in self.engine.penguins.values():
            pos = self.world_to_screen(penguin.position)

            if penguin.alive:
                # Team colors
                if penguin.team_id == 0:
                    color = COL_TEAM_A
                    border_color = (40, 160, 220)
                else:
                    color = COL_TEAM_B
                    border_color = (220, 80, 40)

                # Highlight selected penguin
                is_selected = (
                    self.phase == Phase.AIM
                    and self.selected_penguin == penguin.agent_id
                )

                if is_selected:
                    # Glow effect
                    pygame.draw.circle(
                        self.screen, COL_SELECTED, pos, penguin_radius_px + 5, 2
                    )

                pygame.draw.circle(self.screen, color, pos, penguin_radius_px)
                pygame.draw.circle(self.screen, border_color, pos, penguin_radius_px, 2)

                # Velocity indicator during simulation
                if self.phase == Phase.SIMULATING:
                    vx, vy = penguin.velocity
                    speed = math.sqrt(vx ** 2 + vy ** 2)
                    if speed > 1.0:
                        end_x = pos[0] + int(vx * scale * 0.3)
                        end_y = pos[1] - int(vy * scale * 0.3)
                        pygame.draw.line(
                            self.screen, (255, 255, 255), pos, (end_x, end_y), 1
                        )

                # Label (penguin number)
                idx_str = penguin.agent_id[-1]
                label = self.font_bold.render(idx_str, True, (255, 255, 255))
                label_rect = label.get_rect(center=pos)
                self.screen.blit(label, label_rect)

                # Show check mark if aim is confirmed (during AIM phase)
                if (
                    self.phase == Phase.AIM
                    and penguin.team_id == 0
                    and penguin.agent_id in self.confirmed_aims
                ):
                    check = self.font_bold.render("v", True, COL_ARROW_CONFIRMED)
                    self.screen.blit(
                        check, (pos[0] + penguin_radius_px + 2, pos[1] - 10)
                    )

            else:
                # Dead penguin: gray X
                pygame.draw.circle(self.screen, (80, 80, 80), pos, penguin_radius_px, 2)
                pygame.draw.line(
                    self.screen, (120, 80, 80),
                    (pos[0] - 5, pos[1] - 5), (pos[0] + 5, pos[1] + 5), 2,
                )
                pygame.draw.line(
                    self.screen, (120, 80, 80),
                    (pos[0] - 5, pos[1] + 5), (pos[0] + 5, pos[1] - 5), 2,
                )

    def _draw_arrow(
        self,
        start_screen: tuple[int, int],
        angle_deg: float,
        power: float,
        color: tuple[int, int, int],
        width: int = 2,
    ) -> None:
        """Draw an arrow from a screen position at given angle and power."""
        # Arrow length proportional to power
        arrow_len = MAX_ARROW_PX * (power / self.config.MAX_LAUNCH_FORCE)
        if arrow_len < 3:
            return

        angle_rad = math.radians(angle_deg)
        dx = math.cos(angle_rad) * arrow_len
        dy = -math.sin(angle_rad) * arrow_len  # Y flipped for screen
        end = (int(start_screen[0] + dx), int(start_screen[1] + dy))

        # Shaft
        pygame.draw.line(self.screen, color, start_screen, end, width)

        # Arrowhead
        head_len = min(10.0, arrow_len * 0.3)
        for sign in (1, -1):
            ha = angle_rad + math.pi + math.radians(25 * sign)
            hx = int(end[0] + math.cos(ha) * head_len)
            hy = int(end[1] - math.sin(ha) * head_len)
            pygame.draw.line(self.screen, color, end, (hx, hy), width)

    def _draw_confirmed_arrows(self) -> None:
        """Draw arrows for all confirmed aims."""
        for agent_id, (angle, power) in self.confirmed_aims.items():
            penguin = self.engine.penguins.get(agent_id)
            if penguin and penguin.alive:
                pos = self.world_to_screen(penguin.position)
                self._draw_arrow(pos, angle, power, COL_ARROW_CONFIRMED, width=2)

    def _draw_drag_preview(self) -> None:
        """Draw the aim preview arrow while dragging."""
        if not self.dragging or not self.drag_start or not self.drag_current:
            return
        if not self.selected_penguin:
            return

        penguin = self.engine.penguins.get(self.selected_penguin)
        if not penguin or not penguin.alive:
            return

        # Compute angle and power from drag
        angle_deg, power = self._compute_drag_action(
            self.drag_start, self.drag_current
        )

        pos = self.world_to_screen(penguin.position)
        self._draw_arrow(pos, angle_deg, power, COL_ARROW_PREVIEW, width=3)

        # Show power percentage near cursor
        pct = power / self.config.MAX_LAUNCH_FORCE * 100
        pct_text = self.font.render(f"{pct:.0f}%", True, COL_ARROW_PREVIEW)
        self.screen.blit(
            pct_text,
            (self.drag_current[0] + 12, self.drag_current[1] - 8),
        )

    def _draw_simulation_arrows(self) -> None:
        """Draw all action arrows during simulation (fading out)."""
        for agent_id, (angle, power) in self.all_actions.items():
            penguin = self.engine.penguins.get(agent_id)
            if penguin and penguin.alive:
                pos = self.world_to_screen(penguin.position)
                # Fade the arrow opacity based on settle progress
                alpha = max(0.2, 1.0 - self.settle_steps / 300.0)
                r, g, b = COL_ARROW_CONFIRMED if penguin.team_id == 0 else (255, 200, 100)
                color = (int(r * alpha), int(g * alpha), int(b * alpha))
                self._draw_arrow(pos, angle, power, color, width=1)

    def _draw_hud(self) -> None:
        """Draw the heads-up display at the top."""
        # Dark semi-transparent bar at top
        hud_surface = pygame.Surface((self.screen_cfg.width, 55), pygame.SRCALPHA)
        hud_surface.fill((10, 14, 22, 200))
        self.screen.blit(hud_surface, (0, 0))

        team_a_alive = self.engine.get_alive_count(0)
        team_b_alive = self.engine.get_alive_count(1)
        arena_hw = self.engine.ice_sheet.half_width

        # Phase text
        if self.phase == Phase.AIM:
            phase_text = "AIM"
            phase_color = COL_SELECTED
        elif self.phase == Phase.SIMULATING:
            phase_text = "SIMULATING"
            phase_color = (255, 180, 80)
        elif self.phase == Phase.SETTLE_PAUSE:
            phase_text = "SETTLED"
            phase_color = (100, 255, 100)
        else:
            phase_text = "GAME OVER"
            phase_color = (255, 80, 80)

        # Left section: round and phase
        round_surf = self.font_bold.render(
            f"Round {self.round_num}", True, COL_TEXT
        )
        self.screen.blit(round_surf, (12, 6))

        phase_surf = self.font_bold.render(phase_text, True, phase_color)
        self.screen.blit(phase_surf, (12, 28))

        # Center section: team scores
        mid_x = self.screen_cfg.width // 2
        a_label = self.font_bold.render(f"Team A: {team_a_alive}/3", True, COL_TEAM_A)
        b_label = self.font_bold.render(f"Team B: {team_b_alive}/3", True, COL_TEAM_B)
        vs_label = self.font_bold.render(" vs ", True, COL_DIM_TEXT)

        total_w = a_label.get_width() + vs_label.get_width() + b_label.get_width()
        start_x = mid_x - total_w // 2
        self.screen.blit(a_label, (start_x, 10))
        self.screen.blit(vs_label, (start_x + a_label.get_width(), 10))
        self.screen.blit(
            b_label, (start_x + a_label.get_width() + vs_label.get_width(), 10)
        )

        # Arena size and opponent type below scores
        info_text = f"Arena: {arena_hw:.0f}  |  {self.opponent_label}"
        arena_label = self.font.render(info_text, True, COL_DIM_TEXT)
        arena_rect = arena_label.get_rect(centerx=mid_x, top=32)
        self.screen.blit(arena_label, arena_rect)

        # Right section: aim status during AIM phase
        if self.phase == Phase.AIM:
            alive_team_a = [
                aid for aid in ("penguin_0", "penguin_1", "penguin_2")
                if self.engine.penguins[aid].alive
            ]
            aimed = sum(1 for aid in alive_team_a if aid in self.confirmed_aims)
            total = len(alive_team_a)
            status_text = f"Aimed: {aimed}/{total}"
            if aimed == total and total > 0:
                status_text += "  [SPACE to launch]"
            status_surf = self.font.render(status_text, True, COL_TEXT)
            self.screen.blit(
                status_surf,
                (self.screen_cfg.width - status_surf.get_width() - 12, 10),
            )

    def _draw_instructions(self) -> None:
        """Draw instruction text at the bottom."""
        if self.phase == Phase.AIM:
            lines = [
                "Click a cyan penguin to select, drag to aim (direction=launch).  "
                "Right-click = auto-aim.  R = reset aims.  SPACE = launch.",
            ]
        elif self.phase == Phase.SIMULATING:
            lines = ["Simulating physics..."]
        elif self.phase == Phase.SETTLE_PAUSE:
            lines = ["Round complete."]
        else:
            lines = ["Press any key to exit."]

        y = self.screen_cfg.height - 25
        for line in lines:
            surf = self.font.render(line, True, COL_INSTRUCTIONS)
            rect = surf.get_rect(centerx=self.screen_cfg.width // 2, top=y)
            self.screen.blit(surf, rect)
            y += 20

    def _draw_game_over_overlay(self) -> None:
        """Draw game-over overlay with winner text."""
        winner = self.engine.get_winner()
        if winner == 0:
            msg = "Team A Wins!"
            color = COL_TEAM_A
        elif winner == 1:
            msg = "Team B Wins!"
            color = COL_TEAM_B
        elif winner == -1:
            msg = "Draw!"
            color = COL_DIM_TEXT
        else:
            msg = "Game Over"
            color = COL_DIM_TEXT

        # Darken background
        overlay = pygame.Surface(
            (self.screen_cfg.width, self.screen_cfg.height), pygame.SRCALPHA
        )
        overlay.fill((0, 0, 0, 120))
        self.screen.blit(overlay, (0, 0))

        cx = self.screen_cfg.width // 2
        cy = self.screen_cfg.height // 2

        text_surface = self.big_font.render(msg, True, color)
        text_rect = text_surface.get_rect(center=(cx, cy - 20))
        self.screen.blit(text_surface, text_rect)

        # Sub-info
        team_a_alive = self.engine.get_alive_count(0)
        team_b_alive = self.engine.get_alive_count(1)
        sub_msg = f"Rounds played: {self.round_num}  |  Team A: {team_a_alive}/3  |  Team B: {team_b_alive}/3"
        sub_surface = self.font.render(sub_msg, True, COL_TEXT)
        sub_rect = sub_surface.get_rect(center=(cx, cy + 20))
        self.screen.blit(sub_surface, sub_rect)

        exit_msg = self.font.render("Press any key to exit", True, COL_DIM_TEXT)
        exit_rect = exit_msg.get_rect(center=(cx, cy + 55))
        self.screen.blit(exit_msg, exit_rect)

    def _draw_round_label(self) -> None:
        """Draw 'Round N' banner briefly at start of round if desired."""
        pass  # Already in HUD

    # -------------------------------------------------------------------
    # Full frame render
    # -------------------------------------------------------------------

    def render_frame(self) -> None:
        """Render one complete frame."""
        self.screen.fill(COL_BG)
        self._draw_arena()
        self._draw_penguins()

        if self.phase == Phase.AIM:
            self._draw_confirmed_arrows()
            self._draw_drag_preview()
        elif self.phase in (Phase.SIMULATING, Phase.SETTLE_PAUSE):
            self._draw_simulation_arrows()

        self._draw_hud()
        self._draw_instructions()
        self._draw_shrink_warning()

        if self.phase == Phase.GAME_OVER:
            self._draw_game_over_overlay()

        pygame.display.flip()

    # -------------------------------------------------------------------
    # Input helpers
    # -------------------------------------------------------------------

    def _find_penguin_at_screen(
        self, screen_pos: tuple[int, int], team_id: int | None = None
    ) -> str | None:
        """Find alive penguin near a screen position (within tolerance).

        Args:
            screen_pos: Screen pixel (x, y).
            team_id: If set, only match penguins of this team.

        Returns:
            agent_id or None.
        """
        best_id = None
        best_dist = float("inf")

        for penguin in self.engine.penguins.values():
            if not penguin.alive:
                continue
            if team_id is not None and penguin.team_id != team_id:
                continue

            ppos = self.world_to_screen(penguin.position)
            dx = screen_pos[0] - ppos[0]
            dy = screen_pos[1] - ppos[1]
            dist = math.sqrt(dx * dx + dy * dy)

            penguin_radius_px = max(6, int(self.config.PENGUIN_RADIUS * self._get_scale()))
            tolerance = max(CLICK_RADIUS_TOLERANCE, penguin_radius_px + 4)

            if dist <= tolerance and dist < best_dist:
                best_dist = dist
                best_id = penguin.agent_id

        return best_id

    def _compute_drag_action(
        self,
        start: tuple[int, int],
        end: tuple[int, int],
    ) -> tuple[float, float]:
        """Compute (angle_degrees, power) from a drag vector.

        Drag direction = launch direction (arrow points where the penguin will go).
        """
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        dist = math.sqrt(dx * dx + dy * dy)

        if dist < 2:
            return (0.0, 0.0)

        # Screen Y is inverted vs world Y
        angle_rad = math.atan2(-dy, dx)
        angle_deg = math.degrees(angle_rad)
        if angle_deg < 0:
            angle_deg += 360.0

        power = min(dist * self.drag_power_scale, self.config.MAX_LAUNCH_FORCE)

        return (angle_deg, power)

    def _auto_aim_penguin(self, agent_id: str) -> tuple[float, float]:
        """Use heuristic agent to auto-aim a Team A penguin."""
        obs = self.obs_builder.build_observation(
            agent_id, self.engine.penguins, self.engine.step_count
        )
        action = self.auto_agents[agent_id].get_action(obs)
        return (float(action[0]), float(action[1]))

    def _get_cpu_actions(self) -> dict[str, tuple[float, float]]:
        """Get actions from CPU agents for Team B."""
        actions: dict[str, tuple[float, float]] = {}
        for agent_id in ("penguin_3", "penguin_4", "penguin_5"):
            penguin = self.engine.penguins[agent_id]
            if not penguin.alive:
                continue
            obs = self.obs_builder.build_observation(
                agent_id, self.engine.penguins, self.engine.step_count
            )
            action = self.cpu_agents[agent_id].get_action(obs)
            actions[agent_id] = (float(action[0]), float(action[1]))
        return actions

    def _alive_team_a_ids(self) -> list[str]:
        """Return list of alive Team A agent IDs."""
        return [
            aid for aid in ("penguin_0", "penguin_1", "penguin_2")
            if self.engine.penguins[aid].alive
        ]

    def _all_team_a_aimed(self) -> bool:
        """Check if all alive Team A penguins have confirmed aims."""
        for aid in self._alive_team_a_ids():
            if aid not in self.confirmed_aims:
                return False
        return True

    # -------------------------------------------------------------------
    # Phase handlers
    # -------------------------------------------------------------------

    def _handle_aim_events(self) -> None:
        """Process events during the AIM phase."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                return
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                    return
                elif event.key in (pygame.K_SPACE, pygame.K_RETURN):
                    # Launch if we have any aims (or all aimed)
                    alive_a = self._alive_team_a_ids()
                    if not alive_a:
                        # No alive team A -- auto-transition
                        self._start_simulation()
                        return
                    if self._all_team_a_aimed():
                        self._start_simulation()
                    else:
                        # If some penguins aren't aimed, auto-aim them
                        for aid in alive_a:
                            if aid not in self.confirmed_aims:
                                self.confirmed_aims[aid] = self._auto_aim_penguin(aid)
                        self._start_simulation()
                    return
                elif event.key == pygame.K_r:
                    # Reset aims
                    self.confirmed_aims.clear()
                    self.selected_penguin = None
                    self.dragging = False

            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left click
                    clicked = self._find_penguin_at_screen(event.pos, team_id=0)
                    if clicked:
                        self.selected_penguin = clicked
                        self.dragging = True
                        # Start drag from the penguin's screen position (not click pos)
                        self.drag_start = self.world_to_screen(
                            self.engine.penguins[clicked].position
                        )
                        self.drag_current = event.pos
                    else:
                        # Clicked outside any penguin -- deselect
                        self.selected_penguin = None
                        self.dragging = False

                elif event.button == 3:  # Right click
                    clicked = self._find_penguin_at_screen(event.pos, team_id=0)
                    if clicked:
                        # Auto-aim
                        self.confirmed_aims[clicked] = self._auto_aim_penguin(clicked)
                        self.selected_penguin = clicked
                        self.dragging = False

            elif event.type == pygame.MOUSEMOTION:
                if self.dragging:
                    self.drag_current = event.pos

            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1 and self.dragging:
                    self.dragging = False
                    if self.selected_penguin and self.drag_start:
                        angle_deg, power = self._compute_drag_action(
                            self.drag_start, event.pos
                        )
                        if power > 3.0:
                            self.confirmed_aims[self.selected_penguin] = (
                                angle_deg, power
                            )
                        # If drag was too short, treat as just selection (no aim set)
                    self.drag_start = None
                    self.drag_current = None

    def _start_simulation(self) -> None:
        """Transition from AIM to SIMULATING phase."""
        # Get CPU actions
        cpu_actions = self._get_cpu_actions()

        # Merge human + CPU
        self.all_actions = {**self.confirmed_aims, **cpu_actions}

        # Any alive Team A penguin without an aim gets zero power
        for aid in self._alive_team_a_ids():
            if aid not in self.all_actions:
                self.all_actions[aid] = (0.0, 0.0)

        # Apply all actions simultaneously
        self.engine.apply_actions(self.all_actions)

        self.phase = Phase.SIMULATING
        self.settle_steps = 0
        self.selected_penguin = None
        self.dragging = False

    def _handle_simulation_step(self) -> None:
        """Run one physics step and render during SIMULATING phase.

        Physics must fully resolve (all penguins settled or eliminated)
        before we check game-over.  Using ``is_game_over()`` as an early
        exit would miss mutual knockouts where the last two penguins from
        opposite teams collide and both fly off the map.
        """
        # Process quit events during simulation
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                return
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                self.running = False
                return

        # Step physics
        self.engine.space.step(self.config.FIXED_DT)
        self.engine.step_count += 1
        self.settle_steps += 1
        self.engine._check_eliminations()

        # Only finish the round when ALL alive penguins have settled.
        # are_all_settled() returns True when no alive penguins remain
        # (vacuously true), which correctly covers the mutual-KO case.
        if (
            self.engine.are_all_settled()
            or self.settle_steps >= self.max_settle_steps
        ):
            self._finish_round()

    def _finish_round(self) -> None:
        """End the simulation phase, handle shrink, prepare for next round."""
        self.round_num += 1

        # Arena shrink check
        if self.round_num > 0 and self.round_num % self.config.SHRINK_INTERVAL == 0:
            self.shrink_old_hw = self.engine.ice_sheet.half_width
            scale = self.engine.ice_sheet.shrink(
                self.config.SHRINK_FACTOR, self.config.MIN_ARENA_HALF_WIDTH
            )
            # Rescale all alive penguin positions/velocities so their
            # normalized coordinates within the arena are preserved.
            # Shrinking must NEVER eliminate penguins.
            self.engine.rescale_penguins(scale)
            self.shrink_new_hw = self.engine.ice_sheet.half_width
            self.shrink_pending = True
            self.shrink_anim_frames = 0

        # Check game over
        if self.engine.is_game_over():
            self.phase = Phase.GAME_OVER
        else:
            # Brief pause so player can see the settled positions
            self.phase = Phase.SETTLE_PAUSE
            self.settle_pause_frames = 0

    def _handle_settle_pause(self) -> None:
        """Brief pause after simulation settles before next AIM phase."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                return
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                self.running = False
                return

        self.settle_pause_frames += 1
        if self.settle_pause_frames >= self.settle_pause_duration:
            self.phase = Phase.AIM
            self.confirmed_aims.clear()
            self.all_actions.clear()

    def _handle_game_over_events(self) -> None:
        """Process events in GAME_OVER phase."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                return
            if event.type == pygame.KEYDOWN:
                self.running = False
                return

    # -------------------------------------------------------------------
    # Shrink animation
    # -------------------------------------------------------------------

    def _draw_shrink_warning(self) -> None:
        """Draw arena shrink notification briefly."""
        if not self.shrink_pending:
            return

        self.shrink_anim_frames += 1

        # Show for 90 frames (~1.5 seconds at 60fps)
        if self.shrink_anim_frames > 90:
            self.shrink_pending = False
            return

        cx = self.screen_cfg.width // 2
        cy = (self.screen_cfg.height + 60) // 2

        # Flash border in red/orange
        alpha = abs(math.sin(self.shrink_anim_frames * 0.15))
        r = int(255 * alpha)
        g = int(100 * alpha)
        color = (r, g, 30)

        half_width = self.engine.ice_sheet.half_width
        scale = self._get_scale()
        arena_hw_px = int(half_width * scale)
        arena_rect = pygame.Rect(
            cx - arena_hw_px, cy - arena_hw_px, 2 * arena_hw_px, 2 * arena_hw_px
        )
        pygame.draw.rect(self.screen, color, arena_rect, 4)

        # Text notification
        msg = f"Arena shrinks! {self.shrink_old_hw:.0f} -> {self.shrink_new_hw:.0f}"
        text_surf = self.medium_font.render(msg, True, (255, 160, 40))
        text_rect = text_surf.get_rect(
            centerx=self.screen_cfg.width // 2,
            top=60,
        )
        self.screen.blit(text_surf, text_rect)

    # -------------------------------------------------------------------
    # Main loop
    # -------------------------------------------------------------------

    def run(self) -> None:
        """Main game loop."""
        try:
            while self.running:
                if self.phase == Phase.AIM:
                    self._handle_aim_events()
                    self.render_frame()
                    self.clock.tick(self.screen_cfg.fps)

                elif self.phase == Phase.SIMULATING:
                    self._handle_simulation_step()
                    if self.running:
                        self.render_frame()
                        self.clock.tick(self.screen_cfg.fps)

                elif self.phase == Phase.SETTLE_PAUSE:
                    self._handle_settle_pause()
                    if self.running:
                        self.render_frame()
                        self.clock.tick(self.screen_cfg.fps)

                elif self.phase == Phase.GAME_OVER:
                    self._handle_game_over_events()
                    if self.running:
                        self.render_frame()
                        self.clock.tick(self.screen_cfg.fps)

        except KeyboardInterrupt:
            pass
        finally:
            pygame.quit()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Play Penguin Knockout (Pygame interactive mode)."
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--opponent",
        type=str,
        default="heuristic",
        help=(
            "Opponent bot name from eval/lineup.py "
            "(e.g. random, heuristic, ppo, mappo, self_play, "
            "contrastive, attention_reward, llm_cli_v2, llm_anthropic). "
            "Default: heuristic."
        ),
    )
    parser.add_argument(
        "-l", "--list",
        action="store_true",
        help="Print the bot lineup and exit.",
    )
    args = parser.parse_args()

    if args.list:
        print(lineup_summary())
        return

    if args.opponent not in LINEUP_BY_NAME:
        valid = ", ".join(LINEUP_BY_NAME.keys())
        print(
            f"ERROR: unknown opponent {args.opponent!r}.\n"
            f"Valid bot names: {valid}",
            file=sys.stderr,
        )
        sys.exit(1)

    spec = LINEUP_BY_NAME[args.opponent]
    ok, reason = spec.is_available()
    if not ok:
        print(
            f"ERROR: opponent {args.opponent!r} is unavailable: {reason}",
            file=sys.stderr,
        )
        sys.exit(1)

    play = PlayMode(
        seed=args.seed,
        opponent_name=args.opponent,
    )
    play.run()


if __name__ == "__main__":
    main()
