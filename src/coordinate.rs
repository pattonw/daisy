use std::ops::{Add, Div, Mul, Sub};
use std::cmp::Ordering;

use pyo3::prelude::*;

#[pyclass]
#[derive(Debug, Eq, Clone)]
pub struct Coordinate {
    // #[pyo3(get)]
    pub value: Vec<i64>,
}

impl<'a, 'b> Add<&'b Coordinate> for &'a Coordinate {
    type Output = Coordinate;

    fn add(self, rhs: &'b Coordinate) -> Coordinate {
        let new_value: Vec<i64> = self
            .value
            .iter()
            .zip(rhs.value.iter())
            .map(|(a, b)| *a + *b)
            .collect();
        Coordinate::new(new_value)
    }
}

impl<T: Into<i64> + Copy> Add<T> for &Coordinate {
    type Output = Coordinate;

    fn add(self, rhs: T) -> Coordinate {
        let new_value: Vec<i64> = self.value.iter().map(|a| *a + rhs.into()).collect();
        Coordinate::new(new_value)
    }
}

impl<'a, 'b> Sub<&'b Coordinate> for &'a Coordinate {
    type Output = Coordinate;

    fn sub(self, rhs: &'b Coordinate) -> Coordinate {
        let new_value: Vec<i64> = self
            .value
            .iter()
            .zip(rhs.value.iter())
            .map(|(a, b)| *a - *b)
            .collect();
        Coordinate::new(new_value)
    }
}

impl<T: Into<i64> + Copy> Sub<T> for &Coordinate {
    type Output = Coordinate;

    fn sub(self, rhs: T) -> Coordinate {
        let new_value: Vec<i64> = self.value.iter().map(|a| *a - rhs.into()).collect();
        Coordinate::new(new_value)
    }
}

impl<'a, 'b> Div<&'b Coordinate> for &'a Coordinate {
    type Output = Coordinate;

    fn div(self, rhs: &'b Coordinate) -> Coordinate {
        let new_value: Vec<i64> = self
            .value
            .iter()
            .zip(rhs.value.iter())
            .map(|(a, b)| *a / *b)
            .collect();
        Coordinate::new(new_value)
    }
}

impl<'a, 'b> Mul<&'b Coordinate> for &'a Coordinate {
    type Output = Coordinate;

    fn mul(self, rhs: &'b Coordinate) -> Coordinate {
        let new_value: Vec<i64> = self
            .value
            .iter()
            .zip(rhs.value.iter())
            .map(|(a, b)| *a * *b)
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
    pub fn new(coordinate: Vec<i64>) -> Self {
        Coordinate { value: coordinate }
    }
    pub fn max(&self, rhs: &Coordinate) -> Coordinate {
        let new_value: Vec<i64> = self
            .value
            .iter()
            .zip(rhs.value.iter())
            .map(|(a, b)| match a.cmp(b) {
                Ordering::Less => *b,
                _ => *a,
            })
            .collect();
        Self { value: new_value }
    }
    pub fn min(&self, rhs: &Coordinate) -> Coordinate {
        let new_value: Vec<i64> = self
            .value
            .iter()
            .zip(rhs.value.iter())
            .map(|(a, b)| match a.cmp(b) {
                Ordering::Greater => *b,
                _ => *a,
            })
            .collect();
        Self { value: new_value }
    }
    pub fn dims(&self) -> usize {
        self.value.len()
    }
}
