//! CHIP-8 machine state.

pub const MEMORY_SIZE: usize = 4096;
pub const PROGRAM_START: u16 = 0x200;
/// Largest ROM that fits between `PROGRAM_START` and the end of memory.
pub const MAX_ROM_SIZE: usize = MEMORY_SIZE - PROGRAM_START as usize;

/// ROM exceeds `MAX_ROM_SIZE`; holds the rejected size in bytes.
#[derive(Debug, PartialEq, Eq)]
pub struct RomTooLarge(pub usize);

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

    /// Copies `rom` into memory at `PROGRAM_START`.
    pub fn load_rom(&mut self, rom: &[u8]) -> Result<(), RomTooLarge> {
        if rom.len() > MAX_ROM_SIZE {
            return Err(RomTooLarge(rom.len()));
        }
        let start = PROGRAM_START as usize;
        self.memory[start..start + rom.len()].copy_from_slice(rom);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_starts_at_program_start() {
        assert_eq!(Chip8::new().pc, PROGRAM_START);
    }

    #[test]
    fn load_rom_copies_to_program_start() {
        let mut c = Chip8::new();
        c.load_rom(&[0xAB, 0xCD]).unwrap();
        assert_eq!(c.memory[0x200..0x202], [0xAB, 0xCD]);
    }

    #[test]
    fn load_rom_rejects_oversized_rom() {
        let rom = vec![0; MAX_ROM_SIZE + 1];
        assert_eq!(Chip8::new().load_rom(&rom), Err(RomTooLarge(rom.len())));
    }
}
