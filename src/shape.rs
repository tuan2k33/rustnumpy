pub fn c_contiguous_strides(shape: &[usize]) -> Vec<isize> {
    let mut strides = vec![0isize; shape.len()];
    let mut acc: isize = 1;
    for i in (0..shape.len()).rev() {
        strides[i] = acc;
        acc *= shape[i] as isize;
    }
    strides
}

pub fn offset_of(index: &[usize], strides: &[isize]) -> isize {
    index
        .iter()
        .zip(strides.iter())
        .map(|(&i, &s)| i as isize * s)
        .sum()
}

pub fn index_in_bounds(index: &[usize], shape: &[usize]) -> bool {
    index.len() == shape.len() && index.iter().zip(shape.iter()).all(|(&i, &s)| i < s)
}

pub fn broadcast_shapes(a: &[usize], b: &[usize]) -> Option<Vec<usize>> {
    let ndim = a.len().max(b.len());
    let mut result = vec![0usize; ndim];
    for i in 0..ndim {

        let da = a.len().checked_sub(1 + i).map(|idx| a[idx]).unwrap_or(1);
        let db = b.len().checked_sub(1 + i).map(|idx| b[idx]).unwrap_or(1);
        let out = match (da, db) {
            (x, y) if x == y => x,
            (1, y) => y,
            (x, 1) => x,
            _ => return None,
        };
        result[ndim - 1 - i] = out;
    }
    Some(result)
}

pub fn broadcast_strides(
    shape: &[usize],
    strides: &[isize],
    target_shape: &[usize],
) -> Option<Vec<isize>> {
    let ndim = target_shape.len();
    if shape.len() > ndim {
        return None;
    }
    let offset = ndim - shape.len();
    let mut result = vec![0isize; ndim];
    for i in 0..ndim {
        if i < offset {

            result[i] = 0;
        } else {
            let orig_dim = shape[i - offset];
            let orig_stride = strides[i - offset];
            let target_dim = target_shape[i];
            if orig_dim == target_dim {
                result[i] = orig_stride;
            } else if orig_dim == 1 {
                result[i] = 0;
            } else {
                return None;
            }
        }
    }
    Some(result)
}

pub fn unravel_index(mut flat: usize, shape: &[usize]) -> Vec<usize> {
    let mut index = vec![0usize; shape.len()];
    for axis in (0..shape.len()).rev() {
        let dim = shape[axis];
        index[axis] = if dim == 0 { 0 } else { flat % dim };
        flat /= dim.max(1);
    }
    index
}

pub struct IndexIter<'a> {
    shape: &'a [usize],
    current: Option<Vec<usize>>,
}

impl<'a> IndexIter<'a> {
    pub fn new(shape: &'a [usize]) -> Self {
        let start = if shape.contains(&0) {
            None
        } else {
            Some(vec![0usize; shape.len()])
        };
        Self { shape, current: start }
    }
}

impl<'a> Iterator for IndexIter<'a> {
    type Item = Vec<usize>;

    fn next(&mut self) -> Option<Vec<usize>> {
        let current = self.current.take()?;
        if self.shape.is_empty() {

            self.current = None;
            return Some(current);
        }
        let mut next = current.clone();
        for axis in (0..self.shape.len()).rev() {
            next[axis] += 1;
            if next[axis] < self.shape[axis] {
                self.current = Some(next);
                return Some(current);
            }
            next[axis] = 0;
        }

        self.current = None;
        Some(current)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn c_contiguous_strides_row_major() {
        assert_eq!(c_contiguous_strides(&[2, 3, 4]), vec![12, 4, 1]);
        assert_eq!(c_contiguous_strides(&[5]), vec![1]);
        assert_eq!(c_contiguous_strides(&[]), Vec::<isize>::new());
    }

    #[test]
    fn broadcast_shapes_matches_numpy_rules() {
        assert_eq!(broadcast_shapes(&[3, 4], &[4]), Some(vec![3, 4]));
        assert_eq!(broadcast_shapes(&[3, 1], &[1, 4]), Some(vec![3, 4]));
        assert_eq!(broadcast_shapes(&[8, 1, 6], &[7, 1, 5]), None);
        assert_eq!(broadcast_shapes(&[5, 4], &[1]), Some(vec![5, 4]));
        assert_eq!(broadcast_shapes(&[], &[3]), Some(vec![3]));
    }

    #[test]
    fn broadcast_strides_zeroes_stretched_axes() {

        let strides = broadcast_strides(&[3, 1], &[1, 1], &[3, 4]).unwrap();
        assert_eq!(strides, vec![1, 0]);

        let strides = broadcast_strides(&[4], &[1], &[3, 4]).unwrap();
        assert_eq!(strides, vec![0, 1]);
    }

    #[test]
    fn index_iter_is_row_major() {
        let shape = vec![2, 3];
        let all: Vec<_> = IndexIter::new(&shape).collect();
        assert_eq!(
            all,
            vec![
                vec![0, 0], vec![0, 1], vec![0, 2],
                vec![1, 0], vec![1, 1], vec![1, 2],
            ]
        );
    }

    #[test]
    fn index_iter_zero_size_axis_is_empty() {
        let shape = vec![0, 3];
        assert_eq!(IndexIter::new(&shape).count(), 0);
    }

    #[test]
    fn index_iter_zero_dim_yields_single_empty_index() {
        let shape: Vec<usize> = vec![];
        let all: Vec<_> = IndexIter::new(&shape).collect();
        assert_eq!(all, vec![Vec::<usize>::new()]);
    }

    #[test]
    fn unravel_index_matches_index_iter_order() {

        let shape = vec![2, 3];
        let expected: Vec<Vec<usize>> = IndexIter::new(&shape).collect();
        for (flat, idx) in expected.iter().enumerate() {
            assert_eq!(&unravel_index(flat, &shape), idx);
        }
    }

    #[test]
    fn unravel_index_roundtrips_through_offset_of() {
        let shape = vec![4, 5, 3];
        let strides = c_contiguous_strides(&shape);
        for flat in 0..shape.iter().product::<usize>() {
            let idx = unravel_index(flat, &shape);
            assert_eq!(offset_of(&idx, &strides), flat as isize);
        }
    }
}
