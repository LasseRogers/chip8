"""Shared pygame drawing helpers."""

import numpy as np
import pygame


def blit_frame(surface: pygame.Surface, frame: np.ndarray, pos: tuple[int, int] = (0, 0)) -> None:
    """Draws an RGB frame of shape (height, width, 3) onto `surface` at `pos`."""
    # pygame indexes surfaces as (x, y); numpy frames are (y, x).
    surface.blit(pygame.surfarray.make_surface(frame.swapaxes(0, 1)), pos)


Q_LABEL_WIDTH = 90
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


CHART_TITLE_HEIGHT = 20
CHART_BORDER_COLOR = (50, 50, 50)


def draw_line_chart(
    surface: pygame.Surface,
    rect: pygame.Rect,
    title: str,
    series: list[tuple[list[float], tuple[int, int, int]]],
    font: pygame.font.Font,
) -> pygame.Rect:
    """Draws each (values, color) series in `rect` on a shared y-range; returns the plot area.

    The title shows the latest value of the last non-empty series.
    """
    pygame.draw.rect(surface, CHART_BORDER_COLOR, rect, width=1)
    latest = next((values[-1] for values, _ in reversed(series) if values), None)
    label = title if latest is None else f"{title}   {latest:.3g}"
    surface.blit(font.render(label, True, TEXT_COLOR), (rect.left + 6, rect.top + 4))
    plot = pygame.Rect(
        rect.left + 6,
        rect.top + CHART_TITLE_HEIGHT + 4,
        rect.width - 12,
        rect.height - CHART_TITLE_HEIGHT - 10,
    )

    points = [v for values, _ in series for v in values]
    if len(points) < 2:
        return plot
    lo, hi = min(points), max(points)
    span = (hi - lo) or 1.0

    def y(value: float) -> float:
        return plot.bottom - 1 - (value - lo) / span * (plot.height - 1)

    if lo < 0 < hi:
        pygame.draw.line(surface, CHART_BORDER_COLOR, (plot.left, y(0)), (plot.right - 1, y(0)))
    for values, color in series:
        if len(values) < 2:
            continue
        step = (plot.width - 1) / (len(values) - 1)
        pygame.draw.lines(
            surface, color, False, [(plot.left + i * step, y(v)) for i, v in enumerate(values)], 2
        )
    return plot
