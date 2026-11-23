//! Unit tests for `Chip8`.

use super::*;

/// Loads `rom` into a fresh machine and executes `steps` instructions.
fn run(rom: &[u8], steps: usize) -> Chip8 {
    let mut c = Chip8::new();
    c.load_rom(rom).unwrap();
    for _ in 0..steps {
        c.step().unwrap();
    }
    c
}

// --- Machine setup ---

#[test]
fn new_starts_at_program_start() {
    assert_eq!(Chip8::new().pc, PROGRAM_START);
}

#[test]
fn load_rom_copies_to_program_start() {
    assert_eq!(run(&[0xAB, 0xCD], 0).memory[0x200..0x202], [0xAB, 0xCD]);
}

#[test]
fn load_rom_rejects_oversized_rom() {
    let rom = vec![0; MAX_ROM_SIZE + 1];
    assert_eq!(Chip8::new().load_rom(&rom), Err(RomTooLarge(rom.len())));
}

#[test]
fn fetch_reads_big_endian_and_advances_pc() {
    let mut c = run(&[0x12, 0x34], 0);
    assert_eq!(c.fetch(), 0x1234);
    assert_eq!(c.pc, PROGRAM_START + 2);
}

#[test]
fn display_exposes_full_framebuffer() {
    let c = Chip8::new();
    assert_eq!(c.display().len(), DISPLAY_WIDTH * DISPLAY_HEIGHT);
    assert!(c.display().iter().all(|&p| !p));
}

#[test]
fn step_rejects_unknown_opcode() {
    assert_eq!(run(&[0xFF, 0xFF], 0).step(), Err(ExecError::UnknownOpcode(0xFFFF)));
}

#[test]
fn memory_exposes_rom_and_font() {
    let c = run(&[0xAB, 0xCD], 0);
    assert_eq!(c.memory().len(), MEMORY_SIZE);
    assert_eq!(c.memory()[0x200..0x202], [0xAB, 0xCD]);
    assert_eq!(c.memory()[FONT_START], FONT[0]);
}

// --- Flow control ---

#[test]
fn step_1nnn_jumps() {
    assert_eq!(run(&[0x1A, 0xBC], 1).pc, 0xABC);
}

#[test]
fn step_2nnn_pushes_return_address_and_jumps() {
    let c = run(&[0x23, 0x00], 1);
    assert_eq!(c.pc, 0x300);
    assert_eq!(c.sp, 1);
    assert_eq!(c.stack[0], PROGRAM_START + 2);
}

#[test]
fn step_2nnn_rejects_stack_overflow() {
    // Calls itself; the call after STACK_DEPTH has no free slot.
    let mut c = run(&[0x22, 0x00], STACK_DEPTH);
    assert_eq!(c.step(), Err(ExecError::StackOverflow));
}

#[test]
fn step_00ee_returns_after_call() {
    // 0x200: call 0x204; 0x202: (return lands here); 0x204: return.
    let c = run(&[0x22, 0x04, 0x00, 0x00, 0x00, 0xEE], 2);
    assert_eq!(c.pc, PROGRAM_START + 2);
    assert_eq!(c.sp, 0);
}

#[test]
fn step_00ee_rejects_stack_underflow() {
    assert_eq!(run(&[0x00, 0xEE], 0).step(), Err(ExecError::StackUnderflow));
}

#[test]
fn step_bnnn_jumps_with_v0_offset() {
    // V0 = 0x10, V2 = 0x99, then B200: target is 0x200 + V0, not + V2.
    assert_eq!(run(&[0x60, 0x10, 0x62, 0x99, 0xB2, 0x00], 3).pc, 0x210);
}

// --- Conditional skips ---

#[test]
fn step_3xnn_skips_if_equal() {
    // V0 = 0x05, then 3 0 NN.
    for (nn, expected_pc) in [(0x05, 0x206), (0x06, 0x204)] {
        assert_eq!(run(&[0x60, 0x05, 0x30, nn], 2).pc, expected_pc, "nn = {nn:#04X}");
    }
}

#[test]
fn step_4xnn_skips_if_not_equal() {
    // V0 = 0x05, then 4 0 NN.
    for (nn, expected_pc) in [(0x05, 0x204), (0x06, 0x206)] {
        assert_eq!(run(&[0x60, 0x05, 0x40, nn], 2).pc, expected_pc, "nn = {nn:#04X}");
    }
}

#[test]
fn step_5xy0_skips_if_registers_equal() {
    // V0 = 0x05, V1 = v1, then 5 0 1 0.
    for (v1, expected_pc) in [(0x05, 0x208), (0x06, 0x206)] {
        let c = run(&[0x60, 0x05, 0x61, v1, 0x50, 0x10], 3);
        assert_eq!(c.pc, expected_pc, "v1 = {v1:#04X}");
    }
}

#[test]
fn step_9xy0_skips_if_registers_differ() {
    // V0 = 0x05, V1 = v1, then 9 0 1 0.
    for (v1, expected_pc) in [(0x05, 0x206), (0x06, 0x208)] {
        let c = run(&[0x60, 0x05, 0x61, v1, 0x90, 0x10], 3);
        assert_eq!(c.pc, expected_pc, "v1 = {v1:#04X}");
    }
}

#[test]
fn step_rejects_register_skip_with_nonzero_low_nibble() {
    assert_eq!(run(&[0x50, 0x11], 0).step(), Err(ExecError::UnknownOpcode(0x5011)));
}

// --- Registers and arithmetic ---

#[test]
fn step_6xnn_sets_register() {
    assert_eq!(run(&[0x6A, 0x42], 1).v[0xA], 0x42);
}

#[test]
fn step_7xnn_adds_to_register() {
    assert_eq!(run(&[0x63, 0x10, 0x73, 0x05], 2).v[3], 0x15);
}

#[test]
fn step_7xnn_wraps_without_touching_vf() {
    let c = run(&[0x63, 0xFF, 0x73, 0x02], 2);
    assert_eq!(c.v[3], 0x01);
    assert_eq!(c.v[0xF], 0);
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
        let c = run(&[0x60, 0x0C, 0x61, 0x0A, 0x6F, 0x01, 0x80, 0x10 | n], 4);
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
        let c = run(&[0x60, v0, 0x61, v1, 0x80, 0x10 | n], 3);
        assert_eq!(c.v[0], expected_v0, "8XY{n:X} {v0:#04X} {v1:#04X}");
        assert_eq!(c.v[0xF], expected_vf, "8XY{n:X} {v0:#04X} {v1:#04X} VF");
    }
}

#[test]
fn step_8xy4_flag_wins_when_target_is_vf() {
    // VF = 0x10, V1 = 0x20, then VF += V1: sum 0x30, but VF ends as carry 0.
    assert_eq!(run(&[0x6F, 0x10, 0x61, 0x20, 0x8F, 0x14], 3).v[0xF], 0);
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
        let c = run(&[0x60, 0xFF, 0x61, v1, 0x80, 0x10 | n], 3);
        assert_eq!(c.v[0], expected_v0, "8XY{n:X} {v1:#04X}");
        assert_eq!(c.v[0xF], expected_vf, "8XY{n:X} {v1:#04X} VF");
    }
}

#[test]
fn step_cxnn_is_deterministic_per_seed() {
    let rolls = |seed| {
        let mut c = Chip8::with_seed(seed);
        c.load_rom(&[0xC0, 0xFF, 0xC1, 0xFF, 0xC2, 0xFF]).unwrap();
        for _ in 0..3 {
            c.step().unwrap();
        }
        [c.v[0], c.v[1], c.v[2]]
    };
    assert_eq!(rolls(7), rolls(7));
    assert_ne!(rolls(7), rolls(8));
}

#[test]
fn step_cxnn_masks_with_nn() {
    // 0x200: V0 = rand & 0x0F; 0x202: jump back to 0x200.
    let mut c = run(&[0xC0, 0x0F, 0x12, 0x00], 0);
    for _ in 0..100 {
        c.step().unwrap();
        assert!(c.v[0] <= 0x0F, "got {:#04X}", c.v[0]);
        c.step().unwrap();
    }
}

// --- Index register and memory ---

#[test]
fn step_annn_sets_index() {
    assert_eq!(run(&[0xA1, 0x23], 1).i, 0x123);
}

#[test]
fn step_fx1e_adds_vx_to_index() {
    // I = 0x100, V0 = 0x20, then F01E.
    assert_eq!(run(&[0xA1, 0x00, 0x60, 0x20, 0xF0, 0x1E], 3).i, 0x120);
}

#[test]
fn step_fx33_stores_decimal_digits() {
    // I = 0x300, V0 = 157, then F033.
    let c = run(&[0xA3, 0x00, 0x60, 157, 0xF0, 0x33], 3);
    assert_eq!(c.memory[0x300..0x303], [1, 5, 7]);
}

#[test]
fn step_fx55_stores_registers_and_advances_index() {
    // V0..V2 = 0x11, 0x22, 0x33; I = 0x300; then F255.
    let c = run(&[0x60, 0x11, 0x61, 0x22, 0x62, 0x33, 0xA3, 0x00, 0xF2, 0x55], 5);
    assert_eq!(c.memory[0x300..0x303], [0x11, 0x22, 0x33]);
    assert_eq!(c.i, 0x303);
}

#[test]
fn step_fx65_loads_registers_and_advances_index() {
    // I = 0x204; F165 loads V0, V1 from the two data bytes that follow.
    let c = run(&[0xA2, 0x04, 0xF1, 0x65, 0xAA, 0xBB], 2);
    assert_eq!(c.v[..3], [0xAA, 0xBB, 0x00]);
    assert_eq!(c.i, 0x206);
}

// --- Display ---

#[test]
fn step_00e0_clears_display() {
    let mut c = run(&[0x00, 0xE0], 0);
    c.display[0] = true;
    c.step().unwrap();
    assert!(c.display.iter().all(|&p| !p));
}

#[test]
fn step_dxyn_draws_sprite() {
    // I = 0x204; draw 2 rows at (V0, V1) = (0, 0); sprite data.
    let c = run(&[0xA2, 0x04, 0xD0, 0x12, 0xF0, 0x90], 2);
    let (on, off) = (true, false);
    assert_eq!(c.display[0..8], [on, on, on, on, off, off, off, off]);
    assert_eq!(c.display[64..72], [on, off, off, on, off, off, off, off]);
    assert_eq!(c.v[0xF], 0);
}

#[test]
fn step_dxyn_sets_vf_on_collision() {
    // I = 0x206; draw the same 1-row sprite twice; sprite data.
    let mut c = run(&[0xA2, 0x06, 0xD0, 0x11, 0xD0, 0x11, 0x80], 2);
    c.tick_timers(); // next frame, so the second draw can run
    c.step().unwrap();
    assert!(!c.display[0]);
    assert_eq!(c.v[0xF], 1);
}

#[test]
fn step_fx29_points_index_at_font_glyph() {
    // V0 = 0xA, then F029; draw 5 rows at (V1, V2) = (0, 0).
    let c = run(&[0x60, 0x0A, 0xF0, 0x29, 0xD1, 0x25], 3);
    assert_eq!(usize::from(c.i), FONT_START + 0xA * FONT_GLYPH_SIZE);
    // First 4 pixels of each row, read as bits.
    let rows: Vec<u8> = (0..5)
        .map(|r| (0..4).fold(0, |acc, col| acc << 1 | u8::from(c.display[r * DISPLAY_WIDTH + col])))
        .collect();
    assert_eq!(rows, [0b1111, 0b1001, 0b1111, 0b1001, 0b1001]);
}

#[test]
fn step_dxyn_waits_for_next_frame() {
    // Two draws in a row: the second blocks until the next frame.
    let mut c = run(&[0xA2, 0x06, 0xD0, 0x11, 0xD0, 0x11, 0x80], 3);
    assert_eq!(c.pc, PROGRAM_START + 4, "second draw is waiting");
    c.tick_timers();
    c.step().unwrap();
    assert_eq!(c.pc, PROGRAM_START + 6);
}

// --- Timers ---

#[test]
fn step_fx15_fx18_fx07_set_and_read_timers() {
    // V0 = 0x30; delay = V0; sound = V0; V1 = delay.
    let c = run(&[0x60, 0x30, 0xF0, 0x15, 0xF0, 0x18, 0xF1, 0x07], 4);
    assert_eq!(c.delay_timer, 0x30);
    assert_eq!(c.sound_timer, 0x30);
    assert_eq!(c.v[1], 0x30);
}

#[test]
fn tick_timers_counts_down_and_stops_at_zero() {
    // V0 = 2; delay = V0; sound = V0.
    let mut c = run(&[0x60, 0x02, 0xF0, 0x15, 0xF0, 0x18], 3);
    c.tick_timers();
    assert_eq!((c.delay_timer, c.sound_timer), (1, 1));
    assert!(c.sound_active());
    c.tick_timers();
    c.tick_timers();
    assert_eq!((c.delay_timer, c.sound_timer), (0, 0));
    assert!(!c.sound_active());
}

// --- Keypad ---

#[test]
fn step_ex9e_exa1_skip_on_key_state() {
    // V0 = 0x5, then E09E or E0A1 on key V0.
    for (nn, pressed, expected_pc) in [
        (0x9E, true, 0x206),
        (0x9E, false, 0x204),
        (0xA1, true, 0x204),
        (0xA1, false, 0x206),
    ] {
        let mut c = run(&[0x60, 0x05, 0xE0, nn], 1);
        c.set_key(0x5, pressed);
        c.step().unwrap();
        assert_eq!(c.pc, expected_pc, "E0{nn:02X} pressed = {pressed}");
    }
}

#[test]
fn step_fx0a_blocks_without_key() {
    assert_eq!(run(&[0xF0, 0x0A], 3).pc, PROGRAM_START);
}

#[test]
fn step_fx0a_stores_key_on_release() {
    let mut c = run(&[0xF0, 0x0A], 1);
    c.set_key(0x7, true);
    c.step().unwrap();
    assert_eq!(c.pc, PROGRAM_START, "still blocked while held");
    c.set_key(0x7, false);
    c.step().unwrap();
    assert_eq!(c.v[0], 0x7);
    assert_eq!(c.pc, PROGRAM_START + 2);
}

#[test]
fn run_frame_executes_instructions_then_ticks() {
    // V0 = 5; delay = V0; 0x204: jump to self.
    let mut c = run(&[0x60, 0x05, 0xF0, 0x15, 0x12, 0x04], 0);
    c.run_frame(3).unwrap();
    assert_eq!(c.pc, 0x204);
    assert_eq!(c.delay_timer, 4);
}
