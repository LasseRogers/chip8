//! Desktop frontend for the CHIP-8 interpreter.

use std::{env, fs, process::ExitCode};

use chip8_core::{Chip8, DISPLAY_HEIGHT, DISPLAY_WIDTH};
use minifb::{Key, Scale, Window, WindowOptions};

const PIXEL_ON: u32 = 0x00FF_FFFF;
const PIXEL_OFF: u32 = 0x0000_0000;

fn main() -> ExitCode {
    let Some(path) = env::args().nth(1) else {
        eprintln!("usage: chip8-desktop <rom>");
        return ExitCode::FAILURE;
    };

    let rom = match fs::read(&path) {
        Ok(rom) => rom,
        Err(e) => {
            eprintln!("failed to read {path}: {e}");
            return ExitCode::FAILURE;
        }
    };

    let mut chip8 = Chip8::new();
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
        render(chip8.display(), &mut buffer);
        if let Err(e) = window.update_with_buffer(&buffer, DISPLAY_WIDTH, DISPLAY_HEIGHT) {
            eprintln!("failed to update window: {e}");
            return ExitCode::FAILURE;
        }
    }
    ExitCode::SUCCESS
}

/// Converts the CHIP-8 framebuffer to minifb's 0RGB pixel format.
fn render(display: &[bool], buffer: &mut [u32]) {
    for (dst, &lit) in buffer.iter_mut().zip(display) {
        *dst = if lit { PIXEL_ON } else { PIXEL_OFF };
    }
}
