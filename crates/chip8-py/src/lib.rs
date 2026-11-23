//! Python bindings for `chip8-core`.

use chip8_core::{Chip8, ExecError, RomTooLarge};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyBytes;

/// A CHIP-8 machine. Thin wrapper over `chip8_core::Chip8`.
#[pyclass(name = "Chip8", module = "chip8")]
struct PyChip8(Chip8);

#[pymethods]
impl PyChip8 {
    #[new]
    #[pyo3(signature = (seed = 0))]
    fn new(seed: u64) -> Self {
        Self(Chip8::with_seed(seed))
    }

    /// Copies `rom` into memory at 0x200. Raises `ValueError` if it doesn't fit.
    fn load_rom(&mut self, rom: &[u8]) -> PyResult<()> {
        self.0
            .load_rom(rom)
            .map_err(|RomTooLarge(size)| PyValueError::new_err(format!("ROM too large: {size} bytes")))
    }

    /// Executes one instruction. Raises `RuntimeError` on an execution error.
    fn step(&mut self) -> PyResult<()> {
        self.0.step().map_err(exec_error)
    }

    /// Runs one 60 Hz frame: `instructions` steps, then a timer tick.
    /// The default matches the desktop frontend's speed.
    #[pyo3(signature = (instructions = 12))]
    fn run_frame(&mut self, instructions: usize) -> PyResult<()> {
        self.0.run_frame(instructions).map_err(exec_error)
    }

    /// `length` bytes of memory from `addr`. Raises `ValueError` if out of range.
    fn read_memory<'py>(&self, py: Python<'py>, addr: usize, length: usize) -> PyResult<Bound<'py, PyBytes>> {
        let bytes = addr
            .checked_add(length)
            .and_then(|end| self.0.memory().get(addr..end))
            .ok_or_else(|| PyValueError::new_err(format!("out of range: {addr:#x} + {length}")))?;
        Ok(PyBytes::new(py, bytes))
    }

    /// Presses or releases hex key `key` (0-15). Raises `ValueError` otherwise.
    fn set_key(&mut self, key: u8, pressed: bool) -> PyResult<()> {
        if key > 0xF {
            return Err(PyValueError::new_err(format!("key out of range: {key}")));
        }
        self.0.set_key(key, pressed);
        Ok(())
    }

    /// Advances one 60 Hz frame: decrements timers, releases the display wait.
    fn tick_timers(&mut self) {
        self.0.tick_timers();
    }

    /// The 64x32 display as 2048 bytes, row-major; 1 is a lit pixel.
    fn screen<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        let pixels: Vec<u8> = self.0.display().iter().map(|&p| u8::from(p)).collect();
        PyBytes::new(py, &pixels)
    }
}

fn exec_error(e: ExecError) -> PyErr {
    PyRuntimeError::new_err(e.to_string())
}

#[pymodule]
mod chip8 {
    use pyo3::prelude::*;

    #[pymodule_export]
    use super::PyChip8;

    #[pymodule_init]
    fn init(m: &Bound<'_, PyModule>) -> PyResult<()> {
        m.add("__version__", env!("CARGO_PKG_VERSION"))
    }
}
