//! CHIP-8 machine state.

use std::fmt;

use crate::rng::Rng;

pub const MEMORY_SIZE: usize = 4096;
pub const PROGRAM_START: u16 = 0x200;
/// Largest ROM that fits between `PROGRAM_START` and the end of memory.
pub const MAX_ROM_SIZE: usize = MEMORY_SIZE - PROGRAM_START as usize;

pub const DISPLAY_WIDTH: usize = 64;
pub const DISPLAY_HEIGHT: usize = 32;

const REGISTER_COUNT: usize = 16;
const STACK_DEPTH: usize = 16;
/// Seed used by `new`; fixed so default runs are reproducible.
const DEFAULT_SEED: u64 = 0;

const FONT_START: usize = 0x050;
const FONT_GLYPH_SIZE: usize = 5;
/// Built-in 4x5 sprites for hex digits 0..F, one byte per row.
#[rustfmt::skip]
const FONT: [u8; 16 * FONT_GLYPH_SIZE] = [
    0xF0, 0x90, 0x90, 0x90, 0xF0, // 0
    0x20, 0x60, 0x20, 0x20, 0x70, // 1
    0xF0, 0x10, 0xF0, 0x80, 0xF0, // 2
    0xF0, 0x10, 0xF0, 0x10, 0xF0, // 3
    0x90, 0x90, 0xF0, 0x10, 0x10, // 4
    0xF0, 0x80, 0xF0, 0x10, 0xF0, // 5
    0xF0, 0x80, 0xF0, 0x90, 0xF0, // 6
    0xF0, 0x10, 0x20, 0x40, 0x40, // 7
    0xF0, 0x90, 0xF0, 0x90, 0xF0, // 8
    0xF0, 0x90, 0xF0, 0x10, 0xF0, // 9
    0xF0, 0x90, 0xF0, 0x90, 0x90, // A
    0xE0, 0x90, 0xE0, 0x90, 0xE0, // B
    0xF0, 0x80, 0x80, 0x80, 0xF0, // C
    0xE0, 0x90, 0x90, 0x90, 0xE0, // D
    0xF0, 0x80, 0xF0, 0x80, 0xF0, // E
    0xF0, 0x80, 0xF0, 0x80, 0x80, // F
];

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
    /// Source for `CXNN`; seeded so runs can be replayed exactly.
    rng: Rng,
    /// Counts down at 60 Hz; programs use it for timing.
    delay_timer: u8,
    /// Counts down at 60 Hz; the buzzer sounds while non-zero.
    sound_timer: u8,
}

impl Chip8 {
    /// Creates a machine in its power-on state with a fixed RNG seed.
    pub fn new() -> Self {
        Self::with_seed(DEFAULT_SEED)
    }

    /// Creates a machine in its power-on state; `seed` drives `CXNN`.
    pub fn with_seed(seed: u64) -> Self {
        let mut memory = [0; MEMORY_SIZE];
        memory[FONT_START..FONT_START + FONT.len()].copy_from_slice(&FONT);
        Self {
            memory,
            pc: PROGRAM_START,
            v: [0; REGISTER_COUNT],
            i: 0,
            display: [false; DISPLAY_WIDTH * DISPLAY_HEIGHT],
            stack: [0; STACK_DEPTH],
            sp: 0,
            rng: Rng::new(seed),
            delay_timer: 0,
            sound_timer: 0,
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

    /// Decrements both timers towards zero. Call at 60 Hz.
    pub fn tick_timers(&mut self) {
        self.delay_timer = self.delay_timer.saturating_sub(1);
        self.sound_timer = self.sound_timer.saturating_sub(1);
    }

    /// Whether the buzzer should be sounding.
    pub fn sound_active(&self) -> bool {
        self.sound_timer > 0
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
            // COSMAC VIP quirk: the offset is always V0, not VX.
            0xB => self.pc = nnn + u16::from(self.v[0]),
            0xC => self.v[x] = self.rng.next_u8() & nn,
            0xD => self.draw_sprite(x, y, n),
            0xF => self.misc(op, x, nn)?,
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
            // COSMAC VIP quirk: shifts read VY, not VX.
            0x6 => self.set_with_flag(x, vy >> 1, vy & 0x01 != 0),
            0x7 => {
                let (diff, borrow) = vy.overflowing_sub(self.v[x]);
                self.set_with_flag(x, diff, !borrow);
            }
            0xE => self.set_with_flag(x, vy << 1, vy & 0x80 != 0),
            _ => return Err(ExecError::UnknownOpcode(op)),
        }
        Ok(())
    }

    /// Executes the `FXNN` family: timers, keypad, memory and font ops.
    fn misc(&mut self, op: u16, x: usize, nn: u8) -> Result<(), ExecError> {
        match nn {
            // VF is not affected (COSMAC VIP behaviour).
            0x07 => self.v[x] = self.delay_timer,
            0x15 => self.delay_timer = self.v[x],
            0x18 => self.sound_timer = self.v[x],
            0x1E => self.i += u16::from(self.v[x]),
            // Only the low nibble of VX selects a glyph.
            0x29 => self.i = (FONT_START + usize::from(self.v[x] & 0xF) * FONT_GLYPH_SIZE) as u16,
            // Hundreds, tens and ones digits of VX at I, I+1, I+2.
            0x33 => {
                let vx = self.v[x];
                let i = usize::from(self.i);
                self.memory[i..i + 3].copy_from_slice(&[vx / 100, vx / 10 % 10, vx % 10]);
            }
            // COSMAC VIP quirk: I advances past the copied bytes.
            0x55 => {
                let i = usize::from(self.i);
                self.memory[i..=i + x].copy_from_slice(&self.v[..=x]);
                self.i += x as u16 + 1;
            }
            0x65 => {
                let i = usize::from(self.i);
                self.v[..=x].copy_from_slice(&self.memory[i..=i + x]);
                self.i += x as u16 + 1;
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
