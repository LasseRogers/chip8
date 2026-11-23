"""Tests for the chip8 Python bindings."""

import chip8
import pytest


def test_unknown_profile_is_rejected():
    with pytest.raises(ValueError, match="unknown profile"):
        chip8.Chip8(profile="nope")


def test_registers_expose_v0_to_vf():
    c = chip8.Chip8(profile="superchip")
    c.load_rom(bytes([0x6A, 0x42]))
    c.step()
    registers = c.registers()
    assert len(registers) == 16
    assert registers[0xA] == 0x42
