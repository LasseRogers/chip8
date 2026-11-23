"""Shared pygame drawing helpers."""

import numpy as np
import pygame


def blit_frame(surface: pygame.Surface, frame: np.ndarray, pos: tuple[int, int] = (0, 0)) -> None:
    """Draws an RGB frame of shape (height, width, 3) onto `surface` at `pos`."""
    # pygame indexes surfaces as (x, y); numpy frames are (y, x).
    surface.blit(pygame.surfarray.make_surface(frame.swapaxes(0, 1)), pos)


Q_LABEL_WIDTH = 70
Q_VALUE_WIDTH = 70
TEXT_COLOR = (200, 200, 200)
MUTED_COLOR = (90, 90, 90)
HIGHLIGHT_COLOR = (55, 138, 221)


def draw_q_values(
    surface: pygame.Surface,
    rect: pygame.Rect,
    labels: tuple[str, ...],
    q_values: np.ndarray,
    chosen: int,
    font: pygame.font.Font,
) -> list[int]:
    """Draws one bar per action inside `rect`, highlighting `chosen`; returns the bar widths.

    Bars are scaled between the lowest and highest Q-value, so small differences
    stay visible; the lowest keeps a short stub so it never disappears.
    """
    lo, hi = float(q_values.min()), float(q_values.max())
    span = (hi - lo) or 1.0
    row_height = rect.height // len(labels)
    max_width = rect.width - Q_LABEL_WIDTH - Q_VALUE_WIDTH
    widths = []
    for i, (label, q) in enumerate(zip(labels, q_values, strict=True)):
        y = rect.top + i * row_height
        color = HIGHLIGHT_COLOR if i == chosen else MUTED_COLOR
        width = int(max_width * (0.1 + 0.9 * (float(q) - lo) / span))
        surface.blit(font.render(label, True, TEXT_COLOR), (rect.left, y + 4))
        pygame.draw.rect(surface, color, (rect.left + Q_LABEL_WIDTH, y + 4, width, row_height - 12))
        surface.blit(font.render(f"{q:+.3f}", True, color), (rect.right - Q_VALUE_WIDTH + 8, y + 4))
        widths.append(width)
    return widths
