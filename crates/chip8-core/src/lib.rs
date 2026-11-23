//! CHIP-8 interpreter core.
//!
//! Frontend-agnostic by design: no I/O, rendering, or timing here.
//! Callers drive execution and read state through the public API.

mod chip8;
mod rng;

pub use chip8::{
    Chip8, DEFAULT_INSTRUCTIONS_PER_FRAME, DISPLAY_HEIGHT, DISPLAY_WIDTH, ExecError, RomTooLarge,
};
