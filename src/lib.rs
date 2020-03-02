#[macro_use]
extern crate itertools;

extern crate serde;

use pyo3::prelude::*;
use pyo3::{exceptions, PyErr, PyResult};
use pyo3::{wrap_pyfunction, wrap_pymodule};

use std::convert::TryFrom;
use std::iter::StepBy;

use std::error::Error;
use std::fmt;

mod block;
use block::Block;
mod cantor;
mod coordinate;
use coordinate::Coordinate;
mod roi;
use roi::Roi;

mod daisy_result;
mod daisy_error;

use daisy_error::DaisyError;
use daisy_result::DaisyResult;


#[derive(Clone, Copy)]
enum Fit {
    Valid,
    Overhang,
    Shrink,
}

fn cartesian_product<I, T>(remaining: &Vec<I>, complete: Vec<Vec<T>>) -> Vec<Vec<T>>
where
    I: Iterator<Item = T> + Clone,
    T: Copy,
{
    match remaining.len() {
        0 => complete,
        1 => iproduct![remaining[0].clone().into_iter(), complete.into_iter()]
            .map(|(r, mut acc)| {
                acc.push(r);
                return acc;
            })
            .collect(),
        _ => cartesian_product(
            &remaining[1..].to_vec(),
            iproduct![remaining[0].clone().into_iter(), complete.into_iter()]
                .map(|(r, mut acc)| {
                    acc.push(r);
                    return acc;
                })
                .collect(),
        ),
    }
}

fn compute_level_stride(read_roi: &Roi, write_roi: &Roi) -> DaisyResult<Coordinate> {
    if !read_roi.contains(&write_roi) {
        return Err(DaisyError::new(&format![
            "{:?} does not contain {:?}",
            read_roi, write_roi
        ]));
    };

    let context_lower: Coordinate = &write_roi.offset - &read_roi.offset;
    let context_upper: Coordinate = &read_roi.end() - &write_roi.end();

    let max_context: Coordinate = Coordinate::max(&context_lower, &context_upper);
    let min_level_stride = &max_context + &write_roi.shape;

    Ok(&(&(&(&min_level_stride - 1) / &write_roi.shape) + 1) * &write_roi.shape)
}

fn compute_level_offsets(write_roi: &Roi, stride: &Coordinate) -> DaisyResult<Vec<Coordinate>> {
    let write_step = &write_roi.shape;
    let write_step: Vec<usize> = write_step
        .value
        .iter()
        .map(|s| usize::try_from(*s).unwrap())
        .collect();
    let dim_offsets: Vec<StepBy<std::ops::Range<i64>>> = write_step
        .iter()
        .zip(stride.value.iter())
        .map(|(step, stride)| (0..*stride).step_by(*step))
        .collect();
    let mut offsets: Vec<Coordinate> = vec![];
    for coord in cartesian_product(&dim_offsets, vec![vec![]]) {
        offsets.push(Coordinate::new(coord));
    }
    Ok(offsets.into_iter().rev().collect())
}

fn get_conflict_offsets(
    level_offset: &Coordinate,
    previous_level_offset: &Coordinate,
    level_stride: &Coordinate,
) -> DaisyResult<Vec<Coordinate>> {
    let offset_to_prev = previous_level_offset - level_offset;

    let conflict_dim_offsets: Vec<std::vec::IntoIter<i64>> = offset_to_prev
        .value
        .iter()
        .zip(level_stride.value.iter())
        .map(|(offset, stride)| match *offset < 0i64 {
            true => vec![*offset, *offset + *stride].into_iter(),
            false => vec![*offset - *stride, *offset].into_iter(),
        })
        .collect();

    let mut conflict_offsets = vec![];
    for coord in cartesian_product(&conflict_dim_offsets, vec![vec![]]) {
        conflict_offsets.push(Coordinate::new(coord))
    }
    return Ok(conflict_offsets);
}

fn enumerate_blocks(
    total_roi: &Roi,
    block_read_roi: &Roi,
    block_write_roi: &Roi,
    conflict_offsets: &Vec<Coordinate>,
    block_offsets: Vec<Coordinate>,
    fit: Fit,
) -> DaisyResult<Vec<(Block, Vec<Block>)>> {
    let mut blocks = vec![];
    let num_conflicts = block_offsets.len() * conflict_offsets.len();
    for block_offset in block_offsets {
        let read_roi = block_read_roi + &block_offset;
        let write_roi = block_write_roi + &block_offset;
        let contained = match fit {
            Fit::Valid => total_roi.contains(&read_roi),
            Fit::Overhang => total_roi.contains_coord(&write_roi.get_begin()),
            Fit::Shrink => {
                let upper_context = &read_roi.end() - &write_roi.end();
                total_roi.contains_coord(&write_roi.get_begin())
                    && total_roi.contains_coord(&(&(&write_roi.get_begin() + &upper_context) + 1))
            }
        };
        let block = match contained {
            true => Some(Block::new_block(&total_roi, &read_roi, &write_roi)),
            false => None,
        };
        if block.is_none() {
            continue;
        };
        let block = block.unwrap();
        let mut conflicts = vec![];
        for conflict_offset in conflict_offsets {
            let read_roi = block_read_roi + &conflict_offset;
            let write_roi = block_write_roi + &conflict_offset;
            let contained = match fit {
                Fit::Valid => total_roi.contains(&read_roi),
                Fit::Overhang => total_roi.contains_coord(&write_roi.get_begin()),
                Fit::Shrink => {
                    let upper_context = &read_roi.end() - &write_roi.end();
                    total_roi.contains_coord(&write_roi.get_begin())
                        && total_roi
                            .contains_coord(&(&(&write_roi.get_begin() + &upper_context) + 1))
                }
            };
            let conflict = match contained {
                true => Some(Block::new_block(&total_roi, &read_roi, &write_roi)),
                false => None,
            };
            if conflict.is_none() {
                continue;
            };
            let conflict = conflict.unwrap();
            let fit_conflict = match fit {
                Fit::Shrink => conflict.shrink_to(total_roi),
                _ => Some(conflict),
            };
            if fit_conflict.is_some() {
                conflicts.push(fit_conflict.unwrap());
            }
        }

        let fit_block = match fit {
            Fit::Shrink => block.shrink_to(total_roi),
            _ => Some(block),
        };

        if fit_block.is_some() {
            blocks.push((fit_block.unwrap(), conflicts));
        }
    }
    return Ok(blocks);
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
    )?;
    let block_read_roi = Roi::new(
        Coordinate {
            value: block_read_offset,
        },
        Coordinate {
            value: block_read_shape,
        },
    )?;
    let block_write_roi = Roi::new(
        Coordinate {
            value: block_write_offset,
        },
        Coordinate {
            value: block_write_shape,
        },
    )?;
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
    )?);
}

fn create_dep_graph(
    total_roi: Roi,
    block_read_roi: Roi,
    block_write_roi: Roi,
    read_write_conflict: bool,
    fit: Fit,
) -> DaisyResult<Vec<(Block, Vec<Block>)>> {
    let level_stride = compute_level_stride(&block_read_roi, &block_write_roi)?;
    let level_offsets = compute_level_offsets(&block_write_roi, &level_stride)?;

    let total_shape = &total_roi.shape;

    let mut level_conflict_offsets: Vec<Vec<Coordinate>> = vec![];

    let mut previous_level_offset = None;

    for (level, level_offset) in level_offsets.iter().enumerate() {
        if previous_level_offset.is_some() && read_write_conflict {
            let conflict_offsets =
                get_conflict_offsets(level_offset, previous_level_offset.unwrap(), &level_stride)?;
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
        for coord in cartesian_product(&block_dim_offsets, vec![vec![]]) {
            block_offsets.push(Coordinate::new(coord));
        }

        let block_offsets: Vec<Coordinate> = block_offsets
            .iter()
            .map(|offset| &(offset + &total_roi.get_begin()) - &block_read_roi.get_begin())
            .collect();

        let mut new_blocks = enumerate_blocks(
            &total_roi,
            &block_read_roi,
            &block_write_roi,
            &level_conflict_offsets[level],
            block_offsets,
            fit,
        )?;
        blocks.append(&mut new_blocks);
    }
    Ok(blocks)
}

#[pymodule]
fn daisy(py: Python, m: &PyModule) -> PyResult<()> {
    m.add_wrapped(wrap_pymodule!(blocks))?;
    m.add_wrapped(wrap_pymodule!(block))?;
    Ok(())
}

#[pymodule]
fn blocks(py: Python, m: &PyModule) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(create_dependency_graph))?;
    Ok(())
}

#[pymodule]
fn block(py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<Block>()?;
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
        )
        .unwrap();
        let read_roi = Roi::new(
            Coordinate::new(vec![0, 0, 0]),
            Coordinate::new(vec![4, 4, 4]),
        )
        .unwrap();
        let write_roi = Roi::new(
            Coordinate::new(vec![1, 1, 1]),
            Coordinate::new(vec![2, 2, 2]),
        )
        .unwrap();
        let read_write_conflict = true;
        let fit = Fit::Valid;
        let graph =
            create_dep_graph(total_roi, read_roi, write_roi, read_write_conflict, fit).unwrap();
        assert_eq![graph.len(), 8000];
    }

    #[test]
    fn test_compute_level_stride() {
        let read_roi = Roi::new(
            Coordinate::new(vec![0, 0, 0]),
            Coordinate::new(vec![4, 4, 4]),
        )
        .unwrap();
        let write_roi = Roi::new(
            Coordinate::new(vec![1, 1, 1]),
            Coordinate::new(vec![2, 2, 2]),
        )
        .unwrap();
        let expected_stride = Coordinate::new(vec![4, 4, 4]);
        assert_eq![
            compute_level_stride(&read_roi, &write_roi).unwrap(),
            expected_stride
        ];
    }

    #[test]
    fn test_compute_level_offset() {
        let write_roi = Roi::new(
            Coordinate::new(vec![1, 1, 1]),
            Coordinate::new(vec![2, 2, 2]),
        )
        .unwrap();
        let stride = Coordinate::new(vec![2, 2, 2]);
        assert_eq![compute_level_offsets(&write_roi, &stride).unwrap().len(), 1];
    }

    #[test]
    fn test_cartesian_product() {
        let a = vec![0, 1];
        let b = vec![0, 1];
        let c = vec![0, 1];

        let expected_product = vec![
            vec![0, 0, 0],
            vec![1, 0, 0],
            vec![0, 1, 0],
            vec![1, 1, 0],
            vec![0, 0, 1],
            vec![1, 0, 1],
            vec![0, 1, 1],
            vec![1, 1, 1],
        ];

        assert_eq!(
            cartesian_product(
                &vec![a.into_iter(), b.into_iter(), c.into_iter()],
                vec![vec![]]
            ),
            expected_product
        );
    }

    #[test]
    fn test_get_conflict_offset() {
        let level_offset = Coordinate::new(vec![0, 2, 4]);
        let prev_level_offset = Coordinate::new(vec![0, 2, 2]);
        let level_stride = Coordinate::new(vec![2, 2, 2]);
        let expected_conflict_offsets = vec![
            Coordinate::new(vec![-2, -2, -2]),
            Coordinate::new(vec![0, -2, -2]),
            Coordinate::new(vec![-2, 0, -2]),
            Coordinate::new(vec![0, 0, -2]),
            Coordinate::new(vec![-2, -2, 0]),
            Coordinate::new(vec![0, -2, 0]),
            Coordinate::new(vec![-2, 0, 0]),
            Coordinate::new(vec![0, 0, 0]),
        ];
        assert_eq![
            get_conflict_offsets(&level_offset, &prev_level_offset, &level_stride).unwrap(),
            expected_conflict_offsets
        ];
    }

    #[test]
    fn test_enumerate_blocks() {
        let total_roi = Roi::new(
            Coordinate::new(vec![0, 0, 0]),
            Coordinate::new(vec![10, 10, 10]),
        )
        .unwrap();
        let read_roi = Roi::new(
            Coordinate::new(vec![0, 0, 0]),
            Coordinate::new(vec![4, 4, 4]),
        )
        .unwrap();
        let write_roi = Roi::new(
            Coordinate::new(vec![1, 1, 1]),
            Coordinate::new(vec![2, 2, 2]),
        )
        .unwrap();
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
        )
        .unwrap();
        assert_eq![blocks.len(), 2];
    }
}
