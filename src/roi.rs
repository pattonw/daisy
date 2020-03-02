use crate::coordinate::Coordinate;
use crate::daisy_result::DaisyResult;
use crate::daisy_error::DaisyError;

use pyo3::prelude::*;
use std::ops::Add;

#[pyclass]
#[derive(Debug, Clone)]
pub struct Roi {
    // #[pyo3(get)]
    pub offset: Coordinate,
    // #[pyo3(get)]
    pub shape: Coordinate,
}

impl Roi {
    pub fn new(offset: Coordinate, shape: Coordinate) -> DaisyResult<Self> {
        match offset.dims() == shape.dims() {
            true => Ok(Roi {
                offset: offset,
                shape: shape,
            }),
            false => Err(DaisyError::new(
                "Roi offset and shape must have same dimensions",
            )),
        }
    }
    pub fn contains(&self, rhs: &Roi) -> bool {
        self.get_begin() <= rhs.get_begin() && self.end() >= rhs.end()
    }
    pub fn contains_coord(&self, rhs: &Coordinate) -> bool {
        self.get_begin() <= *rhs && self.end() > *rhs
    }
    pub fn intersect(&self, rhs: &Roi) -> Roi {
        let start = self.get_begin().max(&rhs.get_begin());
        let end = self.end().min(&rhs.end());
        let shape = &end - &start;
        Roi::new(start, shape).unwrap()
    }
}

#[pymethods]
impl Roi {
    pub fn get_begin(&self) -> Coordinate {
        self.offset.clone()
    }
    pub fn end(&self) -> Coordinate {
        &self.offset + &self.shape
    }
    pub fn dims(&self) -> usize {
        usize::min(self.offset.dims(), self.shape.dims())
    }
}

impl<'a, 'b> Add<&'b Coordinate> for &'a Roi {
    type Output = Roi;

    fn add(self, rhs: &'b Coordinate) -> Roi {
        let new_offset = &self.offset + rhs;
        Roi::new(new_offset, self.shape.clone()).unwrap()
    }
}
