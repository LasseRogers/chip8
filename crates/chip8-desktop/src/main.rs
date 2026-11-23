//! Desktop frontend for the CHIP-8 interpreter.

use std::{
    env, fs,
    process::ExitCode,
    time::{SystemTime, UNIX_EPOCH},
};

use chip8_core::{Chip8, DISPLAY_HEIGHT, DISPLAY_WIDTH};
use minifb::{Key, Scale, Window, WindowOptions};

const USAGE: &str = "usage: chip8-desktop <rom> [--seed N]";
const PIXEL_ON: u32 = 0x00FF_FFFF;
const PIXEL_OFF: u32 = 0x0000_0000;
/// ~700 instructions/s at 60 fps, the conventional CHIP-8 speed.
const INSTRUCTIONS_PER_FRAME: usize = 12;

fn main() -> ExitCode {
    let mut args = env::args().skip(1);
    let Some(path) = args.next() else {
        eprintln!("{USAGE}");
        return ExitCode::FAILURE;
    };
    let seed = match (args.next().as_deref(), args.next()) {
        (None, _) => clock_seed(),
        (Some("--seed"), Some(n)) => match n.parse() {
            Ok(seed) => seed,
            Err(_) => {
                eprintln!("invalid seed: {n}");
                return ExitCode::FAILURE;
            }
        },
        _ => {
            eprintln!("{USAGE}");
            return ExitCode::FAILURE;
        }
    };
    // Printed so any run can be replayed with --seed.
    eprintln!("seed: {seed}");

    let rom = match fs::read(&path) {
        Ok(rom) => rom,
        Err(e) => {
            eprintln!("failed to read {path}: {e}");
            return ExitCode::FAILURE;
        }
    };

    let mut chip8 = Chip8::with_seed(seed);
    if let Err(e) = chip8.load_rom(&rom) {
        eprintln!("failed to load {path}: {e:?}");
        return ExitCode::FAILURE;
    }

    let opts = WindowOptions {
        scale: Scale::X16,
        ..WindowOptions::default()
    };
    let mut window = match Window::new("CHIP-8", DISPLAY_WIDTH, DISPLAY_HEIGHT, opts) {
        Ok(w) => w,
        Err(e) => {
            eprintln!("failed to open window: {e}");
            return ExitCode::FAILURE;
        }
    };
    window.set_target_fps(60);

    let mut buffer = vec![PIXEL_OFF; DISPLAY_WIDTH * DISPLAY_HEIGHT];
    while window.is_open() && !window.is_key_down(Key::Escape) {
        for _ in 0..INSTRUCTIONS_PER_FRAME {
            if let Err(e) = chip8.step() {
                eprintln!("{e}");
                return ExitCode::FAILURE;
            }
        }
        // One tick per frame: 60 Hz at the window's 60 fps.
        chip8.tick_timers();
        render(chip8.display(), &mut buffer);
        if let Err(e) = window.update_with_buffer(&buffer, DISPLAY_WIDTH, DISPLAY_HEIGHT) {
            eprintln!("failed to update window: {e}");
            return ExitCode::FAILURE;
        }
    }
    ExitCode::SUCCESS
}

/// Nanoseconds since the Unix epoch; differs on every run.
fn clock_seed() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_nanos() as u64)
}

/// Converts the CHIP-8 framebuffer to minifb's 0RGB pixel format.
fn render(display: &[bool], buffer: &mut [u32]) {
    for (dst, &lit) in buffer.iter_mut().zip(display) {
        *dst = if lit { PIXEL_ON } else { PIXEL_OFF };
    }
}
