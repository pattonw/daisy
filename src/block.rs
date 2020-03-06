use crate::coordinate::Coordinate;
use crate::roi::Roi;

use pyo3::prelude::*;
use pyo3::{exceptions, PyErr, PyResult};

use crate::cantor::cantor_number;

#[pyclass(module = "daisy")]
#[derive(Debug, Clone)]
pub struct Block {
    #[pyo3(get)]
    pub read_roi: Roi,
    #[pyo3(get)]
    pub write_roi: Roi,
    #[pyo3(get)]
    pub block_id: u64,
    #[pyo3(get)]
    pub z_order_id: u64,
}

#[pymethods]
impl Block {
    fn __getstate__(&self) -> PyResult<(Vec<i64>, Vec<i64>, Vec<i64>, Vec<i64>, u64, u64)> {
        Ok((
            self.read_roi.offset.value.clone(),
            self.read_roi.shape.value.clone(),
            self.write_roi.offset.value.clone(),
            self.write_roi.shape.value.clone(),
            self.block_id,
            self.z_order_id,
        ))
    }
    fn __setstate__(&mut self, data: (Vec<i64>, Vec<i64>, Vec<i64>, Vec<i64>, u64, u64)) -> () {
        self.read_roi = Roi::new(Coordinate::new(data.0), Coordinate::new(data.1)).unwrap();
        self.write_roi = Roi::new(Coordinate::new(data.2), Coordinate::new(data.3)).unwrap();
        self.block_id = data.4;
        self.z_order_id = data.5;
    }
    #[new]
    fn new() -> Self {
        let read_roi = Roi::new(Coordinate::new(vec![0]), Coordinate::new(vec![1])).unwrap();
        let write_roi = Roi::new(Coordinate::new(vec![0]), Coordinate::new(vec![1])).unwrap();
        let block_id = u64::max_value();
        let z_order_id = u64::max_value();
        return Block {
            read_roi: read_roi.clone(),
            write_roi: write_roi.clone(),
            block_id: block_id,
            z_order_id: z_order_id,
        };
    }
}

impl Block {
    pub fn new_block(total_roi: &Roi, read_roi: &Roi, write_roi: &Roi) -> Self {
        let block_index = &write_roi.offset / &write_roi.shape;
        let block_id = cantor_number(&block_index.value.iter().map(|a| *a as u64).collect());
        let z_order_id = get_z_order_id(total_roi, write_roi);
        return Block {
            read_roi: read_roi.clone(),
            write_roi: write_roi.clone(),
            block_id: block_id,
            z_order_id: z_order_id,
        };
    }
    pub fn shrink_to(self, container: &Roi) -> Option<Self> {
        let contained_read_roi = container.intersect(&self.read_roi);
        let diff_top = &self.read_roi.end() - &contained_read_roi.end();
        let diff_bot = &contained_read_roi.get_begin() - &self.read_roi.get_begin();
        let contained_write_roi = Roi::new(
            &self.write_roi.offset + &diff_bot,
            &(&self.write_roi.shape - &diff_bot) - &diff_top,
        )
        .unwrap();

        let d = container.dims();

        let mut shrunken_block = self.clone();
        shrunken_block.read_roi = contained_read_roi;
        shrunken_block.write_roi = contained_write_roi;

        match shrunken_block.write_roi.shape > Coordinate::new((0..d).map(|_| 0).collect()) {
            true => Some(shrunken_block),
            false => None,
        }
    }
}

fn get_z_order_id(total_roi: &Roi, write_roi: &Roi) -> u64 {
    let large_vec: Vec<i64> = (0..total_roi.dims()).map(|_| 2048).collect();
    let large_block = Coordinate::new(large_vec);
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
