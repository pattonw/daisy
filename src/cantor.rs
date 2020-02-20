use std::cmp::Ordering;

pub fn cantor_number(coordinate: &Vec<u64>) -> u64 {
    match coordinate.len() {
        1 => coordinate[0],
        _ => {
            pyramide_volume(coordinate.len(), coordinate.iter().sum())
                + cantor_number(&coordinate[0..(coordinate.len() - 1)].to_vec())
        }
    }
}

fn inv_cantor_number(c: u64, dims: usize) -> Vec<u64> {
    if dims == 1 {
        return vec![c];
    }
    let el = inv_pyramide_volume(c, dims);
    let pv = pyramide_volume(dims, el);
    let mut coord = inv_cantor_number(c - pv, dims - 1);
    let prev_el: u64 = coord.iter().sum();
    let last: u64 = (el - prev_el) as u64;
    coord.push(last);
    return coord;
}

fn pyramide_volume(dims: usize, edge_length: u64) -> u64 {
    if edge_length == 0 {
        return 0;
    }
    let mut v = 1f64;
    for d in 0..dims {
        v *= (edge_length + d as u64) as f64;
        v /= (d + 1) as f64;
    }
    return v as u64;
}

fn inv_pyramide_volume(c: u64, dims: usize) -> u64 {
    let mut el = floating_inv_pyramide_volume(c, dims);
    let pv = pyramide_volume(dims, el);
    if c == pv || dims == 1 {
        return el;
    }
    let increment = match pv < c {
        true => 1,
        false => -1i64,
    };
    loop {
        let new_el = el + increment as u64;
        let pv = pyramide_volume(dims, new_el);
        match (increment.cmp(&0i64), pv.cmp(&c)) {
            (Ordering::Greater, Ordering::Greater) => break,
            (Ordering::Less, Ordering::Less) => {
                el = new_el;
                break;
            }
            (Ordering::Less, Ordering::Equal) => {
                el = new_el;
                break;
            }
            _ => el = new_el,
        };
    }
    return el;
}

fn floating_inv_pyramide_volume(c: u64, dims: usize) -> u64 {
    if c == 0 {
        return 0;
    };
    match dims {
        1 => 0,
        2 => ((((8 * c + 1) as f64).sqrt() - 1_f64) / 2_f64).floor() as u64,
        3 => {
            let cf = c as f64;
            let inv_c = (729_f64.sqrt() * cf + 27_f64 * cf).powf(1_f64 / 3_f64)
                / 3_f64.powf(2f64 / 3f64)
                + 1_f64 / (3_f64 * 729_f64.sqrt() * cf + 81_f64 * cf).powf(1f64 / 3f64)
                - 1_f64;
            return inv_c as u64;
        }
        _ => unimplemented!["Higher dimensions not yet handled"],
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_1d() {
        for i in 0..1000 {
            assert_eq![cantor_number(&vec![i]), i];
        }
    }

    #[test]
    fn test_2d() {
        let coords = vec![
            vec![0, 0],
            vec![0, 1],
            vec![1, 0],
            vec![0, 2],
            vec![1, 1],
            vec![2, 0],
            vec![0, 3],
            vec![1, 2],
            vec![2, 1],
            vec![3, 0],
        ];
        for (i, x) in coords.iter().enumerate() {
            assert_eq![cantor_number(x), i as u64];
        }
    }

    #[test]
    fn test_4d() {
        let coords = vec![
            vec![0, 0, 0, 0],
            vec![0, 0, 0, 1],
            vec![0, 0, 1, 0],
            vec![0, 1, 0, 0],
            vec![1, 0, 0, 0],
            vec![0, 0, 0, 2],
            vec![0, 0, 1, 1],
            vec![0, 1, 0, 1],
            vec![1, 0, 0, 1],
            vec![0, 0, 2, 0],
            vec![0, 1, 1, 0],
            vec![1, 0, 1, 0],
            vec![0, 2, 0, 0],
            vec![1, 1, 0, 0],
            vec![2, 0, 0, 0],
            vec![0, 0, 0, 3],
            vec![0, 0, 1, 2],
            vec![0, 1, 0, 2],
            vec![1, 0, 0, 2],
            vec![0, 0, 2, 1],
        ];
        for (i, x) in coords.iter().enumerate() {
            assert_eq![cantor_number(x), i as u64];
        }
    }
}
