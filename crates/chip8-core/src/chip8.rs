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
    /// `00EE` with an empty stack.
    StackUnderflow,
}

impl fmt::Display for ExecError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnknownOpcode(op) => write!(f, "unknown opcode {op:#06X}"),
            Self::StackOverflow => f.write_str("stack overflow"),
            Self::StackUnderflow => f.write_str("stack underflow"),
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
                0x00EE => self.ret()?,
                _ => return Err(ExecError::UnknownOpcode(op)),
            },
            0x1 => self.pc = nnn,
            0x2 => self.call(nnn)?,
            0x3 => self.skip_if(self.v[x] == nn),
            0x4 => self.skip_if(self.v[x] != nn),
            0x5 if n == 0 => self.skip_if(self.v[x] == self.v[y]),
            0x6 => self.v[x] = nn,
            // Wraps on overflow; unlike 8XY4, VF is not affected.
            0x7 => self.v[x] = self.v[x].wrapping_add(nn),
            0x8 => self.alu(op, x, y, n)?,
            0x9 if n == 0 => self.skip_if(self.v[x] != self.v[y]),
            0xA => self.i = nnn,
            0xD => self.draw_sprite(x, y, n),
            _ => return Err(ExecError::UnknownOpcode(op)),
        }
        Ok(())
    }

    /// Executes the `8XYN` register-to-register family.
    fn alu(&mut self, op: u16, x: usize, y: usize, n: usize) -> Result<(), ExecError> {
        let vy = self.v[y];
        match n {
            0x0 => self.v[x] = vy,
            // COSMAC VIP quirk: the logic ops also reset VF.
            0x1 => {
                self.v[x] |= vy;
                self.v[0xF] = 0;
            }
            0x2 => {
                self.v[x] &= vy;
                self.v[0xF] = 0;
            }
            0x3 => {
                self.v[x] ^= vy;
                self.v[0xF] = 0;
            }
            0x4 => {
                let (sum, carry) = self.v[x].overflowing_add(vy);
                self.set_with_flag(x, sum, carry);
            }
            0x5 => {
                let (diff, borrow) = self.v[x].overflowing_sub(vy);
                self.set_with_flag(x, diff, !borrow);
            }
            0x7 => {
                let (diff, borrow) = vy.overflowing_sub(self.v[x]);
                self.set_with_flag(x, diff, !borrow);
            }
            _ => return Err(ExecError::UnknownOpcode(op)),
        }
        Ok(())
    }

    /// Writes `value` to Vx, then `flag` to VF, so the flag wins when x is F.
    fn set_with_flag(&mut self, x: usize, value: u8, flag: bool) {
        self.v[x] = value;
        self.v[0xF] = u8::from(flag);
    }

    /// Skips the next instruction when `cond` holds.
    fn skip_if(&mut self, cond: bool) {
        if cond {
            self.pc += 2;
        }
    }

    /// Pushes the return address and jumps to `addr`.
    fn call(&mut self, addr: u16) -> Result<(), ExecError> {
        let slot = self.stack.get_mut(self.sp).ok_or(ExecError::StackOverflow)?;
        *slot = self.pc;
        self.sp += 1;
        self.pc = addr;
        Ok(())
    }

    /// Pops the return address pushed by `call`.
    fn ret(&mut self) -> Result<(), ExecError> {
        self.sp = self.sp.checked_sub(1).ok_or(ExecError::StackUnderflow)?;
        self.pc = self.stack[self.sp];
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
mod tests;
