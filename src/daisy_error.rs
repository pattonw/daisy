use pyo3::{exceptions, PyErr, PyResult};

use std::error::Error;
use std::fmt;

#[derive(Debug)]
pub struct DaisyError {
    details: String,
}

impl DaisyError {
    pub fn new(msg: &str) -> DaisyError {
        DaisyError {
            details: msg.to_string(),
        }
    }
}

impl fmt::Display for DaisyError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.details)
    }
}

impl Error for DaisyError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        // Generic error, underlying cause isn't tracked.
        None
    }
    fn description(&self) -> &str {
        &self.details
    }
}

impl std::convert::From<std::num::TryFromIntError> for DaisyError {
    fn from(err: std::num::TryFromIntError) -> DaisyError {
        DaisyError::new(&err.to_string())
    }
}

impl std::convert::From<DaisyError> for PyErr {
    fn from(err: DaisyError) -> PyErr {
        exceptions::OSError::py_err(err.to_string())
    }
}
