"""Tests for pygame drawing helpers; draws off-screen."""

import numpy as np
import pygame
import pytest

from chip8_rl.display import HIGHLIGHT_COLOR, Q_LABEL_WIDTH, draw_line_chart, draw_q_values


@pytest.fixture(scope="module")
def font() -> pygame.font.Font:
    pygame.font.init()
    return pygame.font.Font(None, 20)


def test_q_value_bars_scale_with_values_and_highlight_choice(font):
    surface = pygame.Surface((300, 120))
    rect = pygame.Rect(0, 0, 300, 120)
    q_values = np.array([0.1, 0.5, -0.2])
    widths = draw_q_values(surface, rect, ("a", "b", "c"), q_values, chosen=1, font=font)

    assert widths[1] > widths[0] > widths[2] > 0
    row = rect.height // 3
    assert tuple(surface.get_at((Q_LABEL_WIDTH + 1, row + row // 2)))[:3] == HIGHLIGHT_COLOR


def test_line_chart_spans_plot_area(font):
    surface = pygame.Surface((200, 100))
    color = (255, 0, 0)
    plot = draw_line_chart(
        surface, pygame.Rect(0, 0, 200, 100), "t", [([0.0, 1.0, 2.0], color)], font
    )

    assert tuple(surface.get_at((plot.left, plot.bottom - 1)))[:3] == color
    assert tuple(surface.get_at((plot.right - 1, plot.top)))[:3] == color
