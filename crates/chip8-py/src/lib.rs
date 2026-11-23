//! Python bindings for `chip8-core`.

use pyo3::prelude::*;

#[pymodule]
mod chip8 {
    use pyo3::prelude::*;

    #[pymodule_init]
    fn init(m: &Bound<'_, PyModule>) -> PyResult<()> {
        m.add("__version__", env!("CARGO_PKG_VERSION"))
    }
}
