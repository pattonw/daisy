use std::cmp::Ordering;
use std::ops::{Add, Div, Mul, Neg, Sub};

use pyo3::class::basic::CompareOp;
use pyo3::class::{PyIterProtocol, PyNumberProtocol, PyObjectProtocol, PySequenceProtocol};
use pyo3::exceptions;
use pyo3::prelude::*;

use crate::daisy_error::DaisyError;

#[pyclass]
#[derive(Debug, Eq, Clone)]
pub struct Coordinate {
    // #[pyo3(get)]
    pub value: Vec<Option<i64>>,
    iterindex: usize,
}

impl<'a> Neg for &'a Coordinate {
    type Output = Coordinate;

    fn neg(self) -> Coordinate {
        let new_value: Vec<Option<i64>> = self
            .value
            .iter()
            .map(|a| match a {
                Some(c) => Some(-c),
                None => None,
            })
            .collect();
        Coordinate::new(new_value)
    }
}

impl<'a, 'b> Add<&'b Coordinate> for &'a Coordinate {
    type Output = Coordinate;

    fn add(self, rhs: &'b Coordinate) -> Coordinate {
        let new_value: Vec<Option<i64>> = self
            .value
            .iter()
            .zip(rhs.value.iter())
            .map(|(a, b)| match (a, b) {
                (Some(a), Some(b)) => Some(a + b),
                _ => None,
            })
            .collect();
        Coordinate::new(new_value)
    }
}

impl<T: Into<i64> + Copy> Add<T> for &Coordinate {
    type Output = Coordinate;

    fn add(self, rhs: T) -> Coordinate {
        let new_value: Vec<Option<i64>> = self
            .value
            .iter()
            .map(|a| match a {
                Some(a) => Some(a + rhs.into()),
                _ => None,
            })
            .collect();
        Coordinate::new(new_value)
    }
}

impl<'a, 'b> Sub<&'b Coordinate> for &'a Coordinate {
    type Output = Coordinate;

    fn sub(self, rhs: &'b Coordinate) -> Coordinate {
        let new_value: Vec<Option<i64>> = self
            .value
            .iter()
            .zip(rhs.value.iter())
            .map(|(a, b)| match (a, b) {
                (Some(a), Some(b)) => Some(a - b),
                _ => None,
            })
            .collect();
        Coordinate::new(new_value)
    }
}

impl<T: Into<i64> + Copy> Sub<T> for &Coordinate {
    type Output = Coordinate;

    fn sub(self, rhs: T) -> Coordinate {
        let new_value: Vec<Option<i64>> = self
            .value
            .iter()
            .map(|a| match a {
                Some(a) => Some(*a - rhs.into()),
                _ => None,
            })
            .collect();
        Coordinate::new(new_value)
    }
}

impl<'a, 'b> Div<&'b Coordinate> for &'a Coordinate {
    type Output = Coordinate;

    fn div(self, rhs: &'b Coordinate) -> Coordinate {
        let new_value: Vec<Option<i64>> = self
            .value
            .iter()
            .zip(rhs.value.iter())
            .map(|(a, b)| match (a, b) {
                (Some(a), Some(b)) => Some(a / b),
                _ => None,
            })
            .collect();
        Coordinate::new(new_value)
    }
}

impl<'a, 'b> Mul<&'b Coordinate> for &'a Coordinate {
    type Output = Coordinate;

    fn mul(self, rhs: &'b Coordinate) -> Coordinate {
        let new_value: Vec<Option<i64>> = self
            .value
            .iter()
            .zip(rhs.value.iter())
            .map(|(a, b)| match (a, b) {
                (Some(a), Some(b)) => Some(a * b),
                _ => None,
            })
            .collect();
        Coordinate::new(new_value)
    }
}

impl PartialOrd for Coordinate {
    fn partial_cmp(&self, rhs: &Coordinate) -> Option<Ordering> {
        let comparison: Vec<Ordering> = self
            .value
            .iter()
            .zip(rhs.value.iter())
            .map(|(a, b)| a.cmp(b))
            .collect();
        comparison
            .iter()
            .fold(Some(Ordering::Equal), |acc, x| match (acc, x) {
                (Some(Ordering::Less), Ordering::Less) => Some(Ordering::Less),
                (Some(Ordering::Greater), Ordering::Greater) => Some(Ordering::Greater),
                (acc, Ordering::Equal) => acc,
                (Some(Ordering::Equal), x) => Some(*x),
                _ => None,
            })
    }
}

impl PartialEq for Coordinate {
    fn eq(&self, rhs: &Coordinate) -> bool {
        return self.value.len() == rhs.value.len()
            && self
                .value
                .iter()
                .zip(rhs.value.iter())
                .map(|(a, b)| a == b)
                .all(|a| a);
    }
}

impl Coordinate {
    pub fn max(&self, rhs: &Coordinate) -> Coordinate {
        let new_value: Vec<Option<i64>> = self
            .value
            .iter()
            .zip(rhs.value.iter())
            .map(|(a, b)| match a.cmp(b) {
                Ordering::Less => *b,
                _ => *a,
            })
            .collect();
        Self::new(new_value)
    }
    pub fn min(&self, rhs: &Coordinate) -> Coordinate {
        let new_value: Vec<Option<i64>> = self
            .value
            .iter()
            .zip(rhs.value.iter())
            .map(|(a, b)| match a.cmp(b) {
                Ordering::Greater => *b,
                _ => *a,
            })
            .collect();
        Self::new(new_value)
    }
}

#[pymethods]
impl Coordinate {
    #[new]
    #[args(value = "vec![Some(0),Some(1),Some(2)]")]
    pub fn new(value: Vec<Option<i64>>) -> Self {
        Coordinate {
            value: value,
            iterindex: 0,
        }
    }
    fn __getitem__(&self, x: usize) -> PyResult<Option<i64>> {
        match self.value.get(x) {
            Some(coord) => Ok(*coord),
            None => Err(PyErr::from(DaisyError::new("Index out of bounds"))),
        }
    }
    fn __setstate__(mut slf: PyRefMut<Self>, state: Vec<Option<i64>>) -> () {
        println!["set state with state: {:?}", state];
        slf.value = state;
    }
    fn __getstate__(&self) -> PyResult<Vec<Option<i64>>> {
        println!["get state: {:?}", self.value];
        Ok(self.value.clone())
    }
    pub fn dims(&self) -> usize {
        self.value.len()
    }
}

#[pyproto]
impl PySequenceProtocol for Coordinate {
    fn __len__(&self) -> PyResult<usize> {
        Ok(self.value.len())
    }
    fn __getitem__(&self, idx: isize) -> PyResult<Option<i64>> {
        match self.value.get(idx as usize) {
            Some(x) => Ok(*x),
            None => Err(PyErr::new::<exceptions::IndexError, _>(format![
                "index {} out of bounds for Coordinate of length {}",
                idx,
                self.value.len(),
            ])),
        }
    }
    fn __setitem__(&mut self, idx: isize, value: Option<i64>) -> PyResult<()> {
        match self.value.get_mut(idx as usize) {
            Some(slice) => {
                *slice = value;
                Ok(())
            }
            None => Err(PyErr::new::<exceptions::IndexError, _>(format![
                "index {} out of bounds for Coordinate of length {}",
                idx,
                self.value.len(),
            ])),
        }
    }
    fn __delitem__(&mut self, _idx: isize) -> PyResult<()> {
        Err(PyErr::new::<exceptions::NotImplementedError, _>(
            "`del` is not supported on Coordinates",
        ))
    }
    fn __contains__(&self, item: Option<i64>) -> PyResult<bool> {
        Ok(self.value.contains(&item))
    }
    fn __repeat__(&self, count: isize) -> PyResult<()> {
        Err(PyErr::new::<exceptions::NotImplementedError, _>(
            "`repeat` is not supported on Coordinates",
        ))
    }
}

#[pyproto]
impl PyIterProtocol for Coordinate {
    fn __iter__(mut slf: PyRefMut<Self>) -> PyResult<Py<Coordinate>> {
        slf.iterindex = 0;
        Ok(slf.into())
    }
    fn __next__(mut slf: PyRefMut<Self>) -> PyResult<Option<Option<i64>>> {
        let current = slf.value.get(slf.iterindex).copied();
        slf.iterindex += 1;
        Ok(current)
    }
}

#[pyproto]
impl PyObjectProtocol for Coordinate {
    fn __str__(&self) -> PyResult<String> {
        Ok(format!["{:?}", self.value])
    }
    fn __repr__(&self) -> PyResult<String> {
        Ok(format!["{:?}", self.value])
    }
    fn __richcmp__(self, other: Self, op: CompareOp) -> PyResult<bool> {
        match op {
            CompareOp::Eq => Ok(self == &other),
            x => Err(PyErr::new::<exceptions::NotImplementedError, _>(format![
                "Comparison not implemented for operation {:?}",
                x
            ])),
        }
    }
}

#[pyproto]
impl PyNumberProtocol for Coordinate {
    fn __add__(lhs: Self, rhs: Self) -> PyResult<Self> {
        Ok(&lhs + &rhs)
    }
    fn __sub__(lhs: Self, rhs: Self) -> PyResult<Self> {
        println!["lhs: {:?}, rhs: {:?}", lhs, rhs];
        Ok(&lhs - &rhs)
    }
    fn __neg__(self) -> PyResult<Self> {
        Ok(-self)
    }
}
