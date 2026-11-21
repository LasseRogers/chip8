//! CHIP-8 interpreter core.
//!
//! Frontend-agnostic by design: no I/O, rendering, or timing here.
//! Callers drive execution and read state through the public API.
