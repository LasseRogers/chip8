//! CHIP-8 machine state.

use std::fmt;

pub const MEMORY_SIZE: usize = 4096;
pub const PROGRAM_START: u16 = 0x200;
/// Largest ROM that fits between `PROGRAM_START` and the end of memory.
pub const MAX_ROM_SIZE: usize = MEMORY_SIZE - PROGRAM_START as usize;

pub const DISPLAY_WIDTH: usize = 64;
pub const DISPLAY_HEIGHT: usize = 32;

const REGISTER_COUNT: usize = 16;
const STACK_DEPTH: usize = 16;

/// ROM exceeds `MAX_ROM_SIZE`; holds the rejected size in bytes.
#[derive(Debug, PartialEq, Eq)]
pub struct RomTooLarge(pub usize);

/// Failure while executing an instruction.
#[derive(Debug, PartialEq, Eq)]
pub enum ExecError {
    /// Opcode not recognised by the interpreter.
    UnknownOpcode(u16),
    /// `2NNN` with every stack level in use.
    StackOverflow,
}

impl fmt::Display for ExecError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnknownOpcode(op) => write!(f, "unknown opcode {op:#06X}"),
            Self::StackOverflow => f.write_str("stack overflow"),
        }
    }
}

impl std::error::Error for ExecError {}

pub struct Chip8 {
    memory: [u8; MEMORY_SIZE],
    pc: u16,
    /// General-purpose registers V0..VF; VF doubles as the flag register.
    v: [u8; REGISTER_COUNT],
    /// Index register; holds memory addresses for sprite and load/store ops.
    i: u16,
    /// Row-major framebuffer; `true` is a lit pixel.
    display: [bool; DISPLAY_WIDTH * DISPLAY_HEIGHT],
    /// Return addresses for `2NNN`/`00EE`.
    stack: [u16; STACK_DEPTH],
    /// Index of the next free stack slot.
    sp: usize,
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
            stack: [0; STACK_DEPTH],
            sp: 0,
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

    /// Row-major framebuffer of `DISPLAY_WIDTH * DISPLAY_HEIGHT` pixels.
    pub fn display(&self) -> &[bool] {
        &self.display
    }

    /// Reads the big-endian opcode at `pc` and advances past it.
    fn fetch(&mut self) -> u16 {
        let pc = self.pc as usize;
        let op = u16::from_be_bytes([self.memory[pc], self.memory[pc + 1]]);
        self.pc += 2;
        op
    }

    /// Executes a single instruction.
    pub fn step(&mut self) -> Result<(), ExecError> {
        let op = self.fetch();
        let x = usize::from((op >> 8) & 0xF);
        let y = usize::from((op >> 4) & 0xF);
        let n = usize::from(op & 0xF);
        let nn = (op & 0xFF) as u8;
        let nnn = op & 0x0FFF;

        match op >> 12 {
            0x0 => match op {
                0x00E0 => self.display.fill(false),
                _ => return Err(ExecError::UnknownOpcode(op)),
            },
            0x1 => self.pc = nnn,
            0x2 => self.call(nnn)?,
            0x6 => self.v[x] = nn,
            // Wraps on overflow; unlike 8XY4, VF is not affected.
            0x7 => self.v[x] = self.v[x].wrapping_add(nn),
            0xA => self.i = nnn,
            0xD => self.draw_sprite(x, y, n),
            _ => return Err(ExecError::UnknownOpcode(op)),
        }
        Ok(())
    }

    /// Pushes the return address and jumps to `addr`.
    fn call(&mut self, addr: u16) -> Result<(), ExecError> {
        let slot = self.stack.get_mut(self.sp).ok_or(ExecError::StackOverflow)?;
        *slot = self.pc;
        self.sp += 1;
        self.pc = addr;
        Ok(())
    }

    /// XOR-draws an `n`-row sprite from memory at `i` to (Vx, Vy).
    /// The origin wraps; pixels past the right/bottom edge are clipped.
    /// VF is set to 1 if any lit pixel is turned off, else 0.
    fn draw_sprite(&mut self, x: usize, y: usize, n: usize) {
        let x0 = usize::from(self.v[x]) % DISPLAY_WIDTH;
        let y0 = usize::from(self.v[y]) % DISPLAY_HEIGHT;
        self.v[0xF] = 0;

        for row in 0..n {
            let py = y0 + row;
            if py >= DISPLAY_HEIGHT {
                break;
            }
            let bits = self.memory[(usize::from(self.i) + row) % MEMORY_SIZE];
            for col in 0..8 {
                let px = x0 + col;
                if px >= DISPLAY_WIDTH {
                    break;
                }
                if bits & (0x80 >> col) == 0 {
                    continue;
                }
                let pixel = &mut self.display[py * DISPLAY_WIDTH + px];
                if *pixel {
                    self.v[0xF] = 1;
                }
                *pixel = !*pixel;
            }
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
        assert_eq!(c.step(), Err(ExecError::UnknownOpcode(0xFFFF)));
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

    #[test]
    fn step_dxyn_draws_sprite() {
        let mut c = Chip8::new();
        // I = 0x204; draw 2 rows at (V0, V1) = (0, 0); sprite data.
        c.load_rom(&[0xA2, 0x04, 0xD0, 0x12, 0xF0, 0x90]).unwrap();
        c.step().unwrap();
        c.step().unwrap();
        let (on, off) = (true, false);
        assert_eq!(c.display[0..8], [on, on, on, on, off, off, off, off]);
        assert_eq!(c.display[64..72], [on, off, off, on, off, off, off, off]);
        assert_eq!(c.v[0xF], 0);
    }

    #[test]
    fn step_dxyn_sets_vf_on_collision() {
        let mut c = Chip8::new();
        // I = 0x206; draw the same 1-row sprite twice; sprite data.
        c.load_rom(&[0xA2, 0x06, 0xD0, 0x11, 0xD0, 0x11, 0x80]).unwrap();
        for _ in 0..3 {
            c.step().unwrap();
        }
        assert!(!c.display[0]);
        assert_eq!(c.v[0xF], 1);
    }

    #[test]
    fn display_exposes_full_framebuffer() {
        let c = Chip8::new();
        assert_eq!(c.display().len(), DISPLAY_WIDTH * DISPLAY_HEIGHT);
        assert!(c.display().iter().all(|&p| !p));
    }

    #[test]
    fn step_2nnn_pushes_return_address_and_jumps() {
        let mut c = Chip8::new();
        c.load_rom(&[0x23, 0x00]).unwrap();
        c.step().unwrap();
        assert_eq!(c.pc, 0x300);
        assert_eq!(c.sp, 1);
        assert_eq!(c.stack[0], PROGRAM_START + 2);
    }

    #[test]
    fn step_2nnn_rejects_stack_overflow() {
        let mut c = Chip8::new();
        // Calls itself forever.
        c.load_rom(&[0x22, 0x00]).unwrap();
        for _ in 0..STACK_DEPTH {
            c.step().unwrap();
        }
        assert_eq!(c.step(), Err(ExecError::StackOverflow));
    }
}
