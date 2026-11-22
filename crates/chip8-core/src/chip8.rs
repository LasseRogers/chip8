//! CHIP-8 machine state.

pub const MEMORY_SIZE: usize = 4096;
pub const PROGRAM_START: u16 = 0x200;
/// Largest ROM that fits between `PROGRAM_START` and the end of memory.
pub const MAX_ROM_SIZE: usize = MEMORY_SIZE - PROGRAM_START as usize;

pub const DISPLAY_WIDTH: usize = 64;
pub const DISPLAY_HEIGHT: usize = 32;

const REGISTER_COUNT: usize = 16;

/// ROM exceeds `MAX_ROM_SIZE`; holds the rejected size in bytes.
#[derive(Debug, PartialEq, Eq)]
pub struct RomTooLarge(pub usize);

/// Opcode not recognised by the interpreter.
#[derive(Debug, PartialEq, Eq)]
pub struct UnknownOpcode(pub u16);

pub struct Chip8 {
    memory: [u8; MEMORY_SIZE],
    pc: u16,
    /// General-purpose registers V0..VF; VF doubles as the flag register.
    v: [u8; REGISTER_COUNT],
    /// Index register; holds memory addresses for sprite and load/store ops.
    i: u16,
    /// Row-major framebuffer; `true` is a lit pixel.
    display: [bool; DISPLAY_WIDTH * DISPLAY_HEIGHT],
}

impl Chip8 {
    /// Creates a machine in its power-on state.
    pub fn new() -> Self {
        Self {
            memory: [0; MEMORY_SIZE],
            pc: PROGRAM_START,
            v: [0; REGISTER_COUNT],
            i: 0,
            display: [false; DISPLAY_WIDTH * DISPLAY_HEIGHT],
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

    /// Reads the big-endian opcode at `pc` and advances past it.
    fn fetch(&mut self) -> u16 {
        let pc = self.pc as usize;
        let op = u16::from_be_bytes([self.memory[pc], self.memory[pc + 1]]);
        self.pc += 2;
        op
    }

    /// Executes a single instruction.
    pub fn step(&mut self) -> Result<(), UnknownOpcode> {
        let op = self.fetch();
        let x = usize::from((op >> 8) & 0xF);
        let nn = (op & 0xFF) as u8;
        let nnn = op & 0x0FFF;

        match op >> 12 {
            0x0 => match op {
                0x00E0 => self.display.fill(false),
                _ => return Err(UnknownOpcode(op)),
            },
            0x1 => self.pc = nnn,
            0x6 => self.v[x] = nn,
            // Wraps on overflow; unlike 8XY4, VF is not affected.
            0x7 => self.v[x] = self.v[x].wrapping_add(nn),
            0xA => self.i = nnn,
            _ => return Err(UnknownOpcode(op)),
        }
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

    #[test]
    fn fetch_reads_big_endian_and_advances_pc() {
        let mut c = Chip8::new();
        c.load_rom(&[0x12, 0x34]).unwrap();
        assert_eq!(c.fetch(), 0x1234);
        assert_eq!(c.pc, PROGRAM_START + 2);
    }

    #[test]
    fn step_1nnn_jumps() {
        let mut c = Chip8::new();
        c.load_rom(&[0x1A, 0xBC]).unwrap();
        c.step().unwrap();
        assert_eq!(c.pc, 0xABC);
    }

    #[test]
    fn step_rejects_unknown_opcode() {
        let mut c = Chip8::new();
        c.load_rom(&[0xFF, 0xFF]).unwrap();
        assert_eq!(c.step(), Err(UnknownOpcode(0xFFFF)));
    }

    #[test]
    fn step_6xnn_sets_register() {
        let mut c = Chip8::new();
        c.load_rom(&[0x6A, 0x42]).unwrap();
        c.step().unwrap();
        assert_eq!(c.v[0xA], 0x42);
    }

    #[test]
    fn step_7xnn_adds_to_register() {
        let mut c = Chip8::new();
        c.load_rom(&[0x63, 0x10, 0x73, 0x05]).unwrap();
        c.step().unwrap();
        c.step().unwrap();
        assert_eq!(c.v[3], 0x15);
    }

    #[test]
    fn step_7xnn_wraps_without_touching_vf() {
        let mut c = Chip8::new();
        c.load_rom(&[0x63, 0xFF, 0x73, 0x02]).unwrap();
        c.step().unwrap();
        c.step().unwrap();
        assert_eq!(c.v[3], 0x01);
        assert_eq!(c.v[0xF], 0);
    }

    #[test]
    fn step_annn_sets_index() {
        let mut c = Chip8::new();
        c.load_rom(&[0xA1, 0x23]).unwrap();
        c.step().unwrap();
        assert_eq!(c.i, 0x123);
    }

    #[test]
    fn step_00e0_clears_display() {
        let mut c = Chip8::new();
        c.display[0] = true;
        c.load_rom(&[0x00, 0xE0]).unwrap();
        c.step().unwrap();
        assert!(c.display.iter().all(|&p| !p));
    }
}
