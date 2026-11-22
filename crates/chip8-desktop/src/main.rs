//! Desktop frontend for the CHIP-8 interpreter.

use std::{env, fs, process::ExitCode};

use chip8_core::Chip8;

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

    println!("loaded {path} ({} bytes)", rom.len());
    ExitCode::SUCCESS
}
