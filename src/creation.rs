use crate::error::OpError;
use crate::ndarray::NdArray;

pub fn full<T: Clone>(shape: &[usize], value: T) -> NdArray<T> {
    let n: usize = shape.iter().product();
    NdArray::from_vec(vec![value; n], shape).expect("element count matches the shape")
}

pub fn arange_len(start: f64, stop: f64, step: f64) -> Result<usize, OpError> {
    if step == 0.0 || !step.is_finite() || !start.is_finite() || !stop.is_finite() {
        return Err(OpError::InvalidArange);
    }
    let n = ((stop - start) / step).ceil();
    Ok(if n > 0.0 { n as usize } else { 0 })
}

pub fn arange_f64(start: f64, stop: f64, step: f64) -> Result<NdArray<f64>, OpError> {
    let n = arange_len(start, stop, step)?;
    let delta = (start + step) - start;
    let data: Vec<f64> = (0..n).map(|i| start + i as f64 * delta).collect();
    NdArray::from_vec(data, &[n]).map_err(OpError::from)
}

pub fn arange_i64(start: i64, stop: i64, step: i64) -> Result<NdArray<i64>, OpError> {
    if step == 0 {
        return Err(OpError::InvalidArange);
    }
    let mut data = Vec::new();
    let mut v = start;
    while (step > 0 && v < stop) || (step < 0 && v > stop) {
        data.push(v);
        match v.checked_add(step) {
            Some(next) => v = next,
            None => break,
        }
    }
    let n = data.len();
    NdArray::from_vec(data, &[n]).map_err(OpError::from)
}

pub fn linspace(start: f64, stop: f64, num: usize, endpoint: bool) -> NdArray<f64> {
    let div = if endpoint { num.saturating_sub(1) } else { num };
    let delta = stop - start;
    let mut data: Vec<f64> = if div > 0 {
        let step = delta / div as f64;
        if step == 0.0 {
            (0..num).map(|i| (i as f64 / div as f64) * delta + start).collect()
        } else {
            (0..num).map(|i| i as f64 * step + start).collect()
        }
    } else {
        vec![start; num]
    };
    if endpoint && num > 1 {
        data[num - 1] = stop;
    }
    NdArray::from_vec(data, &[num]).expect("length is num")
}

pub fn eye<T: Clone + Default>(rows: usize, cols: usize, k: isize, one: T) -> NdArray<T> {
    let mut a: NdArray<T> = NdArray::zeros(&[rows, cols]);
    for i in 0..rows {
        let j = i as isize + k;
        if j >= 0 && (j as usize) < cols {
            a.set(&[i, j as usize], one.clone()).expect("index is in bounds");
        }
    }
    a
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arange_matches_numpy_lengths_and_values() {
        let a = arange_f64(0.0, 1.0, 0.1).unwrap();
        assert_eq!(a.len(), 10);
        assert_eq!(a.as_slice()[3], 0.30000000000000004);
        assert_eq!(arange_f64(1.0, 2.0, 0.3).unwrap().as_slice(), &[1.0, 1.3, 1.6, 1.9000000000000001]);
        assert_eq!(arange_f64(0.0, 1.0, 0.3).unwrap().len(), 4);
        assert_eq!(arange_i64(10, 0, -3).unwrap().as_slice(), &[10, 7, 4, 1]);
        assert!(arange_i64(0, 5, 0).is_err());
        assert!(arange_f64(0.0, 5.0, 0.0).is_err());
        assert_eq!(arange_f64(5.0, 0.0, 1.0).unwrap().len(), 0);
    }

    #[test]
    fn linspace_matches_numpy() {
        assert_eq!(linspace(0.0, 1.0, 5, true).as_slice(), &[0.0, 0.25, 0.5, 0.75, 1.0]);
        assert_eq!(linspace(0.0, 1.0, 5, false).as_slice(), &[0.0, 0.2, 0.4, 0.6000000000000001, 0.8]);
        assert_eq!(linspace(2.0, 3.0, 1, true).as_slice(), &[2.0]);
        assert_eq!(linspace(1.0, 1.0, 3, true).as_slice(), &[1.0, 1.0, 1.0]);
        assert_eq!(linspace(0.0, 10.0, 7, true).as_slice()[3], 5.0);
        assert_eq!(linspace(0.0, 1.0, 0, true).len(), 0);
    }

    #[test]
    fn eye_and_full() {
        let e = eye(3, 4, 1, 1.0);
        assert_eq!(e.as_slice(), &[0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]);
        assert_eq!(eye(2, 2, -1, 1i32).as_slice(), &[0, 0, 1, 0]);
        assert_eq!(full(&[2, 2], 7u8).as_slice(), &[7, 7, 7, 7]);
    }
}
