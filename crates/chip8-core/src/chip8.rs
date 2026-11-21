//! CHIP-8 machine state.

pub const MEMORY_SIZE: usize = 4096;
pub const PROGRAM_START: u16 = 0x200;

pub struct Chip8 {
    memory: [u8; MEMORY_SIZE],
    pc: u16,
}

impl Chip8 {
    /// Creates a machine in its power-on state.
    pub fn new() -> Self {
        Self {
            memory: [0; MEMORY_SIZE],
            pc: PROGRAM_START,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_starts_at_program_start() {
        assert_eq!(Chip8::new().pc, PROGRAM_START);
    }
}
