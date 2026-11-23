//! Desktop frontend for the CHIP-8 interpreter.

use std::{
    env, fs,
    process::ExitCode,
    time::{SystemTime, UNIX_EPOCH},
};

use chip8_core::{Chip8, DEFAULT_INSTRUCTIONS_PER_FRAME, DISPLAY_HEIGHT, DISPLAY_WIDTH, Quirks};
use minifb::{Key, Scale, Window, WindowOptions};

const USAGE: &str = "usage: chip8-desktop <rom> [--seed N] [--profile cosmac|superchip]";
const PIXEL_ON: u32 = 0x00FF_FFFF;
const PIXEL_OFF: u32 = 0x0000_0000;

/// Conventional layout: the left 4x4 block of a QWERTY keyboard
/// mirrors the COSMAC VIP hex keypad.
#[rustfmt::skip]
const KEYMAP: [(Key, u8); 16] = [
    (Key::Key1, 0x1), (Key::Key2, 0x2), (Key::Key3, 0x3), (Key::Key4, 0xC),
    (Key::Q,    0x4), (Key::W,    0x5), (Key::E,    0x6), (Key::R,    0xD),
    (Key::A,    0x7), (Key::S,    0x8), (Key::D,    0x9), (Key::F,    0xE),
    (Key::Z,    0xA), (Key::X,    0x0), (Key::C,    0xB), (Key::V,    0xF),
];

fn main() -> ExitCode {
    let Args { path, seed, quirks } = match parse_args(env::args().skip(1)) {
        Ok(args) => args,
        Err(message) => {
            eprintln!("{message}\n{USAGE}");
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

    let mut chip8 = Chip8::with_seed(seed).with_quirks(quirks);
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
        for &(key, chip8_key) in &KEYMAP {
            chip8.set_key(chip8_key, window.is_key_down(key));
        }
        // One emulated frame per window refresh: timers tick at 60 Hz.
        if let Err(e) = chip8.run_frame(DEFAULT_INSTRUCTIONS_PER_FRAME) {
            eprintln!("{e}");
            return ExitCode::FAILURE;
        }
        render(chip8.display(), &mut buffer);
        if let Err(e) = window.update_with_buffer(&buffer, DISPLAY_WIDTH, DISPLAY_HEIGHT) {
            eprintln!("failed to update window: {e}");
            return ExitCode::FAILURE;
        }
    }
    ExitCode::SUCCESS
}

struct Args {
    path: String,
    seed: u64,
    quirks: Quirks,
}

/// Parses `<rom> [--seed N] [--profile cosmac|superchip]`, in any order.
fn parse_args(mut args: impl Iterator<Item = String>) -> Result<Args, String> {
    let mut path = None;
    let mut seed = None;
    let mut quirks = Quirks::COSMAC;
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--seed" => {
                let value = args.next().ok_or("--seed needs a value")?;
                seed = Some(
                    value
                        .parse()
                        .map_err(|_| format!("invalid seed: {value}"))?,
                );
            }
            "--profile" => {
                quirks = match args.next().as_deref() {
                    Some("cosmac") => Quirks::COSMAC,
                    Some("superchip") => Quirks::SUPER_CHIP,
                    other => return Err(format!("unknown profile: {}", other.unwrap_or(""))),
                };
            }
            _ if arg.starts_with("--") => return Err(format!("unknown option: {arg}")),
            _ if path.is_none() => path = Some(arg),
            _ => return Err(format!("unexpected argument: {arg}")),
        }
    }
    Ok(Args {
        path: path.ok_or("missing ROM path")?,
        seed: seed.unwrap_or_else(clock_seed),
        quirks,
    })
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

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(args: &[&str]) -> Result<Args, String> {
        parse_args(args.iter().map(|s| s.to_string()))
    }

    #[test]
    fn parses_flags_in_any_order() {
        let args = parse(&["--profile", "superchip", "game.ch8", "--seed", "7"]).unwrap();
        assert_eq!(
            (args.path.as_str(), args.seed, args.quirks),
            ("game.ch8", 7, Quirks::SUPER_CHIP)
        );
    }

    #[test]
    fn defaults_to_cosmac() {
        assert_eq!(parse(&["game.ch8"]).unwrap().quirks, Quirks::COSMAC);
    }

    #[test]
    fn rejects_bad_input() {
        let cases: [&[&str]; 5] = [
            &[],
            &["game.ch8", "--profile", "x"],
            &["game.ch8", "--seed"],
            &["a.ch8", "b.ch8"],
            &["game.ch8", "--fast"],
        ];
        for bad in cases {
            assert!(parse(bad).is_err(), "{bad:?}");
        }
    }
}
