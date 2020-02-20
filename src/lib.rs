#[macro_use]
extern crate itertools;

use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use pyo3::{exceptions, PyErr, PyResult};

use std::cmp::Ordering;
use std::convert::TryFrom;
use std::iter::StepBy;
use std::ops::{Add, Div, Mul, Range, Sub};

mod cantor;

#[pyfunction]
fn get_42() -> PyResult<usize> {
    Ok(42)
}

#[derive(Clone, Copy)]
enum Fit {
    Valid,
    Overhang,
    Shrink,
}

fn get_z_order_id(total_roi: &Roi, write_roi: &Roi) -> u64 {
    // 3D specific
    let large_block = Coordinate::new(vec![2048, 2048, 2048]);
    let block_index_z = &write_roi.offset / &large_block;
    let mut indices = block_index_z.value;
    let bit32_constant = 1 << 31;
    let mut n = 0;
    let mut z_order_id = 0;
    while n < 32 {
        for i in 0..total_roi.dims() {
            z_order_id = z_order_id >> 1;
            if (indices[i] & 1) != 0 {
                z_order_id += bit32_constant;
            };
            indices[i] = indices[i] >> 1;
            n += 1;
        }
    }
    return z_order_id;
}

#[pyclass]
#[derive(Debug)]
struct Block {
    #[pyo3(get)]
    read_roi: Roi,
    #[pyo3(get)]
    write_roi: Roi,
    #[pyo3(get)]
    block_id: u64,
    #[pyo3(get)]
    z_order_id: u64,
}

impl Block {
    fn new(total_roi: &Roi, read_roi: &Roi, write_roi: &Roi) -> Self {
        let block_index = &write_roi.offset / &write_roi.shape;
        let block_id =
            cantor::cantor_number(&block_index.value.iter().map(|a| *a as u64).collect());
        let z_order_id = get_z_order_id(total_roi, write_roi);
        return Block {
            read_roi: read_roi.clone(),
            write_roi: write_roi.clone(),
            block_id: block_id,
            z_order_id: z_order_id,
        };
    }
    fn shrink_to(self, container: &Roi) -> Self {
        unimplemented![];
    }
}

#[pyclass]
#[derive(Debug, Clone)]
struct Roi {
    #[pyo3(get)]
    offset: Coordinate,
    #[pyo3(get)]
    shape: Coordinate,
}

impl Roi {
    fn new(offset: Coordinate, shape: Coordinate) -> Self {
        Roi {
            offset: offset,
            shape: shape,
        }
    }
    fn begin(&self) -> Coordinate {
        self.offset.clone()
    }
    fn end(&self) -> Coordinate {
        &self.offset + &self.shape
    }
    fn contains(&self, rhs: &Roi) -> bool {
        self.begin() <= rhs.begin() && self.end() >= rhs.end()
    }
    fn contains_coord(&self, rhs: &Coordinate) -> bool {
        self.begin() <= *rhs && self.end() > *rhs
    }
    fn dims(&self) -> usize {
        usize::min(self.offset.dims(), self.shape.dims())
    }
}

impl<'a, 'b> Add<&'b Coordinate> for &'a Roi {
    type Output = Roi;

    fn add(self, rhs: &'b Coordinate) -> Roi {
        let new_offset = &self.offset + rhs;
        Roi::new(new_offset, self.shape.clone())
    }
}

#[pyclass]
#[derive(Debug, Eq, Clone)]
struct Coordinate {
    #[pyo3(get)]
    value: Vec<i64>,
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
    fn new(coordinate: Vec<i64>) -> Self {
        Coordinate { value: coordinate }
    }
    fn max(&self, rhs: &Coordinate) -> Coordinate {
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
    fn min(&self, rhs: &Coordinate) -> Coordinate {
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
    fn dims(&self) -> usize {
        self.value.len()
    }
}

fn compute_level_stride(read_roi: &Roi, write_roi: &Roi) -> Coordinate {
    if !read_roi.contains(&write_roi) {
        panic![format!["{:?} does not contain {:?}", read_roi, write_roi]]
    };

    let context_lower: Coordinate = &write_roi.offset - &read_roi.offset;
    let context_upper: Coordinate = &read_roi.end() - &write_roi.end();

    let max_context: Coordinate = Coordinate::max(&context_lower, &context_upper);
    let min_level_stride = &max_context + &write_roi.shape;

    &(&(&(&min_level_stride - 1) / &write_roi.shape) + 1) * &write_roi.shape
}

fn compute_level_offsets(write_roi: &Roi, stride: &Coordinate) -> Vec<Coordinate> {
    let write_step = &write_roi.shape;
    let mut dim_offsets: Vec<StepBy<std::ops::Range<i64>>> = write_step
        .value
        .iter()
        .zip(stride.value.iter())
        .map(|(step, stride)| (0..*stride).step_by(usize::try_from(*step).unwrap()))
        .collect();
    match dim_offsets.len() {
        3 => 3,
        _ => panic!["Only supporting 3 dimensions at the moment"],
    };
    let mut offsets: Vec<Coordinate> = vec![];
    for (c, b, a) in iproduct![
        dim_offsets.pop().unwrap(),
        dim_offsets.pop().unwrap(),
        dim_offsets.pop().unwrap()
    ] {
        offsets.push(Coordinate {
            value: vec![a, b, c],
        });
    }
    offsets.into_iter().rev().collect()
}

fn get_conflict_offsets(
    level_offset: &Coordinate,
    previous_level_offset: &Coordinate,
    level_stride: &Coordinate,
) -> Vec<Coordinate> {
    let offset_to_prev = previous_level_offset - level_offset;

    let mut conflict_dim_offsets: Vec<Vec<i64>> = offset_to_prev
        .value
        .iter()
        .zip(level_stride.value.iter())
        .map(|(offset, stride)| match *offset < 0i64 {
            true => vec![*offset, offset + stride],
            false => vec![offset - stride, *offset],
        })
        .collect();

    let mut conflict_offsets = vec![];
    let ks = conflict_dim_offsets.pop().unwrap();
    let js = conflict_dim_offsets.pop().unwrap();
    let is = conflict_dim_offsets.pop().unwrap();
    for (a, b, c) in iproduct![is.iter(), js.iter(), ks.iter()] {
        conflict_offsets.push(Coordinate {
            value: vec![*a, *b, *c],
        })
    }
    return conflict_offsets;
}

fn enumerate_blocks(
    total_roi: &Roi,
    block_read_roi: &Roi,
    block_write_roi: &Roi,
    conflict_offsets: &Vec<Coordinate>,
    block_offsets: Vec<Coordinate>,
    fit: Fit,
) -> Vec<(Block, Vec<Block>)> {
    let mut blocks = vec![];
    let mut num_skipped_blocks = 0;
    let mut num_skipped_conflicts = 0;
    let num_conflicts = block_offsets.len() * conflict_offsets.len();
    for block_offset in block_offsets {
        let block = Block::new(
            total_roi,
            &(block_read_roi + &block_offset),
            &(block_write_roi + &block_offset),
        );
        let contained = match fit {
            Fit::Valid => total_roi.contains(&block.read_roi),
            Fit::Overhang => total_roi.contains_coord(&block.write_roi.begin()),
            Fit::Shrink => {
                let upper_context = &block.read_roi.end() - &block.write_roi.end();
                total_roi.contains_coord(&block.write_roi.begin())
                    && total_roi.contains_coord(&(&(&block.write_roi.begin() + &upper_context) + 1))
            }
        };
        if !contained {
            num_skipped_blocks += 1;
            continue;
        }
        let mut conflicts = vec![];
        for conflict_offset in conflict_offsets {
            let conflict = Block::new(
                total_roi,
                &(&block.read_roi + conflict_offset),
                &(&block.write_roi + conflict_offset),
            );
            let contained = match fit {
                Fit::Valid => total_roi.contains(&conflict.read_roi),
                Fit::Overhang => total_roi.contains_coord(&conflict.write_roi.begin()),
                Fit::Shrink => {
                    let upper_context = &conflict.read_roi.end() - &conflict.write_roi.end();
                    total_roi.contains_coord(&conflict.write_roi.begin())
                        && total_roi
                            .contains_coord(&(&(&conflict.write_roi.begin() + &upper_context) + 1))
                }
            };
            if !contained {
                num_skipped_conflicts += 1;
                continue;
            }
            let fit_conflict = match fit {
                Fit::Shrink => conflict.shrink_to(total_roi),
                _ => conflict,
            };
            conflicts.push(fit_conflict);
        }

        let fit_block = match fit {
            Fit::Shrink => block.shrink_to(total_roi),
            _ => block,
        };

        blocks.push((fit_block, conflicts));
    }
    return blocks;
}

#[pyfunction]
fn create_dependency_graph(
    total_roi_offset: Vec<i64>,
    total_roi_shape: Vec<i64>,
    block_read_offset: Vec<i64>,
    block_read_shape: Vec<i64>,
    block_write_offset: Vec<i64>,
    block_write_shape: Vec<i64>,
    read_write_conflict: bool,
    fit: &'static str,
) -> PyResult<Vec<(Block, Vec<Block>)>> {
    let total_roi = Roi::new(
        Coordinate {
            value: total_roi_offset,
        },
        Coordinate {
            value: total_roi_shape,
        },
    );
    let block_read_roi = Roi::new(
        Coordinate {
            value: block_read_offset,
        },
        Coordinate {
            value: block_read_shape,
        },
    );
    let block_write_roi = Roi::new(
        Coordinate {
            value: block_write_offset,
        },
        Coordinate {
            value: block_write_shape,
        },
    );
    let fit = match fit {
        "valid" => Fit::Valid,
        "overhang" => Fit::Overhang,
        "shrink" => Fit::Shrink,
        _ => panic!["fit must be one of 'valid', 'overhang', or 'shrink'"],
    };
    return Ok(create_dep_graph(
        total_roi,
        block_read_roi,
        block_write_roi,
        read_write_conflict,
        fit,
    ));
}

fn create_dep_graph(
    total_roi: Roi,
    block_read_roi: Roi,
    block_write_roi: Roi,
    read_write_conflict: bool,
    fit: Fit,
) -> Vec<(Block, Vec<Block>)> {
    let level_stride = compute_level_stride(&block_read_roi, &block_write_roi);
    let level_offsets = compute_level_offsets(&block_write_roi, &level_stride);

    let total_shape = &total_roi.shape;

    let mut level_conflict_offsets: Vec<Vec<Coordinate>> = vec![];

    let mut previous_level_offset = None;

    for (level, level_offset) in level_offsets.iter().enumerate() {
        if previous_level_offset.is_some() && read_write_conflict {
            let conflict_offsets =
                get_conflict_offsets(level_offset, previous_level_offset.unwrap(), &level_stride);
            level_conflict_offsets.push(conflict_offsets);
        } else {
            let conflict_offsets = vec![];
            level_conflict_offsets.push(conflict_offsets);
        };
        previous_level_offset = Some(level_offset);
    }

    let mut blocks = vec![];

    for (level, level_offset) in level_offsets.iter().enumerate() {
        let mut block_dim_offsets: Vec<StepBy<std::ops::Range<i64>>> = vec![];
        for (lo, e, s) in izip![
            level_offset.value.iter(),
            total_shape.value.iter(),
            level_stride.value.iter()
        ] {
            block_dim_offsets.push((*lo..*e).step_by(usize::try_from(*s).unwrap()));
        }
        let mut block_offsets = vec![];
        for (c, b, a) in iproduct![
            block_dim_offsets.pop().unwrap(),
            block_dim_offsets.pop().unwrap(),
            block_dim_offsets.pop().unwrap()
        ] {
            block_offsets.push(Coordinate {
                value: vec![a, b, c],
            });
        }

        let block_offsets: Vec<Coordinate> = block_offsets
            .iter()
            .map(|offset| &(offset + &total_roi.begin()) - &block_read_roi.begin())
            .collect();

        let mut new_blocks = enumerate_blocks(
            &total_roi,
            &block_read_roi,
            &block_write_roi,
            &level_conflict_offsets[level],
            block_offsets,
            fit,
        );
        blocks.append(&mut new_blocks);
    }
    return blocks;
}

/// This module is a python module implemented in Rust.
#[pymodule]
fn blocks(py: Python, m: &PyModule) -> PyResult<()> {
    //m.add_wrapped(wrap_pyfunction!(create_dependency_graph))?;

    //m.add_class::<Block>()?;


    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_graph() {
        let total_roi = Roi::new(
            Coordinate::new(vec![0, 0, 0]),
            Coordinate::new(vec![42, 42, 42]),
        );
        let read_roi = Roi::new(
            Coordinate::new(vec![0, 0, 0]),
            Coordinate::new(vec![4, 4, 4]),
        );
        let write_roi = Roi::new(
            Coordinate::new(vec![1, 1, 1]),
            Coordinate::new(vec![2, 2, 2]),
        );
        let read_write_conflict = true;
        let fit = Fit::Valid;
        let graph = create_dep_graph(total_roi, read_roi, write_roi, read_write_conflict, fit);
        let blocks: Vec<(Vec<i64>, Vec<i64>)> = graph
            .iter()
            .map(|(block, conflicts)| (block.read_roi.begin().value, block.read_roi.end().value))
            .collect();
        assert_eq![graph.len(), 8000];
    }

    #[test]
    fn test_compute_level_stride() {
        let read_roi = Roi::new(
            Coordinate::new(vec![0, 0, 0]),
            Coordinate::new(vec![4, 4, 4]),
        );
        let write_roi = Roi::new(
            Coordinate::new(vec![1, 1, 1]),
            Coordinate::new(vec![2, 2, 2]),
        );
        let expected_stride = Coordinate::new(vec![4, 4, 4]);
        assert_eq![compute_level_stride(&read_roi, &write_roi), expected_stride];
    }

    #[test]
    fn test_compute_level_offset() {
        let write_roi = Roi::new(
            Coordinate::new(vec![1, 1, 1]),
            Coordinate::new(vec![2, 2, 2]),
        );
        let stride = Coordinate::new(vec![2, 2, 2]);
        assert_eq![compute_level_offsets(&write_roi, &stride).len(), 1];
    }

    #[test]
    fn test_get_conflict_offset() {
        let level_offset = Coordinate::new(vec![0, 2, 4]);
        let prev_level_offset = Coordinate::new(vec![0, 2, 2]);
        let level_stride = Coordinate::new(vec![2, 2, 2]);
        let expected_conflict_offsets = vec![
            Coordinate::new(vec![-2, -2, -2]),
            Coordinate::new(vec![-2, -2, 0]),
            Coordinate::new(vec![-2, 0, -2]),
            Coordinate::new(vec![-2, 0, 0]),
            Coordinate::new(vec![0, -2, -2]),
            Coordinate::new(vec![0, -2, 0]),
            Coordinate::new(vec![0, 0, -2]),
            Coordinate::new(vec![0, 0, 0]),
        ];
        assert_eq![
            get_conflict_offsets(&level_offset, &prev_level_offset, &level_stride),
            expected_conflict_offsets
        ];
    }

    #[test]
    fn test_enumerate_blocks() {
        let total_roi = Roi::new(
            Coordinate::new(vec![0, 0, 0]),
            Coordinate::new(vec![10, 10, 10]),
        );
        let read_roi = Roi::new(
            Coordinate::new(vec![0, 0, 0]),
            Coordinate::new(vec![4, 4, 4]),
        );
        let write_roi = Roi::new(
            Coordinate::new(vec![1, 1, 1]),
            Coordinate::new(vec![2, 2, 2]),
        );
        let conflict_offsets = vec![Coordinate::new(vec![2, 0, 0])];
        let block_offsets = vec![
            Coordinate::new(vec![2, 2, 2]),
            Coordinate::new(vec![2, 2, 4]),
        ];
        let fit = Fit::Valid;
        let blocks = enumerate_blocks(
            &total_roi,
            &read_roi,
            &write_roi,
            &conflict_offsets,
            block_offsets,
            fit,
        );
        assert_eq![blocks.len(), 2];
    }
}
