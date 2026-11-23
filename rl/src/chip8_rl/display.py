"""Shared pygame drawing helpers."""

import numpy as np
import pygame


def blit_frame(surface: pygame.Surface, frame: np.ndarray, pos: tuple[int, int] = (0, 0)) -> None:
    """Draws an RGB frame of shape (height, width, 3) onto `surface` at `pos`."""
    # pygame indexes surfaces as (x, y); numpy frames are (y, x).
    surface.blit(pygame.surfarray.make_surface(frame.swapaxes(0, 1)), pos)
