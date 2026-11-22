//! Unit tests for `Chip8`.

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

#[test]
fn step_00ee_returns_after_call() {
    let mut c = Chip8::new();
    // 0x200: call 0x204; 0x202: (return lands here); 0x204: return.
    c.load_rom(&[0x22, 0x04, 0x00, 0x00, 0x00, 0xEE]).unwrap();
    c.step().unwrap();
    c.step().unwrap();
    assert_eq!(c.pc, PROGRAM_START + 2);
    assert_eq!(c.sp, 0);
}

#[test]
fn step_00ee_rejects_stack_underflow() {
    let mut c = Chip8::new();
    c.load_rom(&[0x00, 0xEE]).unwrap();
    assert_eq!(c.step(), Err(ExecError::StackUnderflow));
}

#[test]
fn step_3xnn_skips_if_equal() {
    // V0 = 0x05, then 3 0 NN.
    for (nn, expected_pc) in [(0x05, 0x206), (0x06, 0x204)] {
        let mut c = Chip8::new();
        c.load_rom(&[0x60, 0x05, 0x30, nn]).unwrap();
        c.step().unwrap();
        c.step().unwrap();
        assert_eq!(c.pc, expected_pc, "nn = {nn:#04X}");
    }
}

#[test]
fn step_4xnn_skips_if_not_equal() {
    // V0 = 0x05, then 4 0 NN.
    for (nn, expected_pc) in [(0x05, 0x204), (0x06, 0x206)] {
        let mut c = Chip8::new();
        c.load_rom(&[0x60, 0x05, 0x40, nn]).unwrap();
        c.step().unwrap();
        c.step().unwrap();
        assert_eq!(c.pc, expected_pc, "nn = {nn:#04X}");
    }
}

#[test]
fn step_5xy0_skips_if_registers_equal() {
    // V0 = 0x05, V1 = v1, then 5 0 1 0.
    for (v1, expected_pc) in [(0x05, 0x208), (0x06, 0x206)] {
        let mut c = Chip8::new();
        c.load_rom(&[0x60, 0x05, 0x61, v1, 0x50, 0x10]).unwrap();
        for _ in 0..3 {
            c.step().unwrap();
        }
        assert_eq!(c.pc, expected_pc, "v1 = {v1:#04X}");
    }
}

#[test]
fn step_9xy0_skips_if_registers_differ() {
    // V0 = 0x05, V1 = v1, then 9 0 1 0.
    for (v1, expected_pc) in [(0x05, 0x206), (0x06, 0x208)] {
        let mut c = Chip8::new();
        c.load_rom(&[0x60, 0x05, 0x61, v1, 0x90, 0x10]).unwrap();
        for _ in 0..3 {
            c.step().unwrap();
        }
        assert_eq!(c.pc, expected_pc, "v1 = {v1:#04X}");
    }
}

#[test]
fn step_rejects_register_skip_with_nonzero_low_nibble() {
    let mut c = Chip8::new();
    c.load_rom(&[0x50, 0x11]).unwrap();
    assert_eq!(c.step(), Err(ExecError::UnknownOpcode(0x5011)));
}

#[test]
fn step_8xy0_to_8xy3_copy_and_logic() {
    // V0 = 0b1100, V1 = 0b1010, VF = 1, then 8 0 1 N.
    for (n, expected_v0, expected_vf) in [
        (0x0, 0b1010, 1), // copy; VF untouched
        (0x1, 0b1110, 0), // OR
        (0x2, 0b1000, 0), // AND
        (0x3, 0b0110, 0), // XOR
    ] {
        let mut c = Chip8::new();
        c.load_rom(&[0x60, 0x0C, 0x61, 0x0A, 0x6F, 0x01, 0x80, 0x10 | n])
            .unwrap();
        for _ in 0..4 {
            c.step().unwrap();
        }
        assert_eq!(c.v[0], expected_v0, "8XY{n:X}");
        assert_eq!(c.v[0xF], expected_vf, "8XY{n:X} VF");
    }
}

#[test]
fn step_8xy4_8xy5_8xy7_arithmetic_with_flags() {
    // V0 = v0, V1 = v1, then 8 0 1 N.
    for (n, v0, v1, expected_v0, expected_vf) in [
        (0x4, 0x10, 0x20, 0x30, 0), // add
        (0x4, 0xFF, 0x02, 0x01, 1), // add with carry
        (0x5, 0x30, 0x10, 0x20, 1), // VX - VY, no borrow
        (0x5, 0x10, 0x30, 0xE0, 0), // VX - VY, borrow
        (0x7, 0x10, 0x30, 0x20, 1), // VY - VX, no borrow
        (0x7, 0x30, 0x10, 0xE0, 0), // VY - VX, borrow
    ] {
        let mut c = Chip8::new();
        c.load_rom(&[0x60, v0, 0x61, v1, 0x80, 0x10 | n]).unwrap();
        for _ in 0..3 {
            c.step().unwrap();
        }
        assert_eq!(c.v[0], expected_v0, "8XY{n:X} {v0:#04X} {v1:#04X}");
        assert_eq!(c.v[0xF], expected_vf, "8XY{n:X} {v0:#04X} {v1:#04X} VF");
    }
}

#[test]
fn step_8xy4_flag_wins_when_target_is_vf() {
    // VF = 0x10, V1 = 0x20, then VF += V1: sum 0x30, but VF ends as carry 0.
    let mut c = Chip8::new();
    c.load_rom(&[0x6F, 0x10, 0x61, 0x20, 0x8F, 0x14]).unwrap();
    for _ in 0..3 {
        c.step().unwrap();
    }
    assert_eq!(c.v[0xF], 0);
}

#[test]
fn step_8xy6_8xye_shift_vy_into_vx() {
    // V0 = 0xFF (must be ignored), V1 = v1, then 8 0 1 N.
    for (n, v1, expected_v0, expected_vf) in [
        (0x6, 0x81, 0x40, 1), // right, low bit out
        (0x6, 0x42, 0x21, 0),
        (0xE, 0x81, 0x02, 1), // left, high bit out
        (0xE, 0x42, 0x84, 0),
    ] {
        let mut c = Chip8::new();
        c.load_rom(&[0x60, 0xFF, 0x61, v1, 0x80, 0x10 | n]).unwrap();
        for _ in 0..3 {
            c.step().unwrap();
        }
        assert_eq!(c.v[0], expected_v0, "8XY{n:X} {v1:#04X}");
        assert_eq!(c.v[0xF], expected_vf, "8XY{n:X} {v1:#04X} VF");
    }
}
