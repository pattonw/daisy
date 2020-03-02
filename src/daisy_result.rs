use crate::daisy_error::DaisyError;

pub type DaisyResult<T> = std::result::Result<T, DaisyError>;
