use crate::error::ShapeError;
use crate::ndarray::NdArray;
use crate::shape::{broadcast_shapes, IndexIter};
use crate::view::ArrayView;

fn common_shape(shapes: &[&[usize]]) -> Result<Vec<usize>, ShapeError> {
    let mut acc: Vec<usize> = Vec::new();
    for s in shapes {
        acc = broadcast_shapes(&acc, s).ok_or_else(|| ShapeError::ChoiceShapeMismatch {
            shapes: shapes.iter().map(|s| s.to_vec()).collect(),
        })?;
    }
    Ok(acc)
}

pub fn where_cond<T: Copy>(
    cond: &ArrayView<bool>,
    x: &ArrayView<T>,
    y: &ArrayView<T>,
) -> Result<NdArray<T>, ShapeError> {
    let shape = common_shape(&[cond.shape(), x.shape(), y.shape()])?;
    let (c, x, y) = (cond.broadcast_to(&shape)?, x.broadcast_to(&shape)?, y.broadcast_to(&shape)?);
    let data: Vec<T> = IndexIter::new(&shape)
        .map(|i| if c.get(&i).unwrap() { x.get(&i).unwrap() } else { y.get(&i).unwrap() })
        .collect();
    NdArray::from_vec(data, &shape)
}

pub fn select<T: Copy>(
    conditions: &[&ArrayView<bool>],
    choices: &[&ArrayView<T>],
    default: T,
) -> Result<NdArray<T>, ShapeError> {
    if conditions.is_empty() || conditions.len() != choices.len() {
        return Err(ShapeError::EmptyArrayList);
    }
    let shapes: Vec<&[usize]> = conditions
        .iter()
        .map(|c| c.shape())
        .chain(choices.iter().map(|c| c.shape()))
        .collect();
    let shape = common_shape(&shapes)?;
    let conds: Vec<ArrayView<bool>> = conditions.iter().map(|c| c.broadcast_to(&shape)).collect::<Result<_, _>>()?;
    let chs: Vec<ArrayView<T>> = choices.iter().map(|c| c.broadcast_to(&shape)).collect::<Result<_, _>>()?;
    let data: Vec<T> = IndexIter::new(&shape)
        .map(|i| {
            conds
                .iter()
                .position(|c| c.get(&i).unwrap())
                .map_or(default, |k| chs[k].get(&i).unwrap())
        })
        .collect();
    NdArray::from_vec(data, &shape)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChooseMode {
    Raise,
    Wrap,
    Clip,
}

pub fn choose<T: Copy>(
    indices: &ArrayView<i64>,
    choices: &[&ArrayView<T>],
    mode: ChooseMode,
) -> Result<NdArray<T>, ShapeError> {
    if choices.is_empty() {
        return Err(ShapeError::EmptyArrayList);
    }
    let shapes: Vec<&[usize]> = std::iter::once(indices.shape()).chain(choices.iter().map(|c| c.shape())).collect();
    let shape = common_shape(&shapes)?;
    let idx = indices.broadcast_to(&shape)?;
    let chs: Vec<ArrayView<T>> = choices.iter().map(|c| c.broadcast_to(&shape)).collect::<Result<_, _>>()?;
    let n = chs.len() as i64;
    let mut data = Vec::with_capacity(shape.iter().product());
    for i in IndexIter::new(&shape) {
        let raw = idx.get(&i).unwrap();
        let k = match mode {
            ChooseMode::Raise if (0..n).contains(&raw) => raw,
            ChooseMode::Raise => {
                return Err(ShapeError::ChoiceIndexOutOfBounds { index: raw, choices: chs.len() });
            }
            ChooseMode::Wrap => raw.rem_euclid(n),
            ChooseMode::Clip => raw.clamp(0, n - 1),
        };
        data.push(chs[k as usize].get(&i).unwrap());
    }
    NdArray::from_vec(data, &shape)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn arr<T>(data: Vec<T>, shape: &[usize]) -> NdArray<T> {
        NdArray::from_vec(data, shape).unwrap()
    }

    #[test]
    fn where_broadcasts_all_three_operands() {
        let c = arr(vec![true, false, true], &[1, 3]);
        let x = arr((0..6).collect::<Vec<i32>>(), &[2, 3]);
        let y = arr(vec![10, 20, 30], &[3]);
        let out = where_cond(&c.view(), &x.view(), &y.view()).unwrap();
        assert_eq!(out.shape(), &[2, 3]);
        assert_eq!(out.as_slice(), &[0, 20, 2, 3, 20, 5]);
    }

    #[test]
    fn where_rejects_unbroadcastable_shapes() {
        let c = arr(vec![true, false], &[2]);
        let x = arr(vec![1, 2, 3], &[3]);
        assert!(matches!(
            where_cond(&c.view(), &x.view(), &x.view()),
            Err(ShapeError::ChoiceShapeMismatch { .. })
        ));
    }

    #[test]
    fn select_first_true_condition_wins_and_default_fills_the_rest() {
        let x = arr((0..6).collect::<Vec<i32>>(), &[6]);
        let x10 = arr((0..6).map(|v| v * 10).collect::<Vec<i32>>(), &[6]);
        let lt2 = arr(x.as_slice().iter().map(|&v| v < 2).collect(), &[6]);
        let lt4 = arr(x.as_slice().iter().map(|&v| v < 4).collect(), &[6]);
        let out = select(&[&lt2.view(), &lt4.view()], &[&x.view(), &x10.view()], -1).unwrap();
        assert_eq!(out.as_slice(), &[0, 1, 20, 30, -1, -1]);
        let out0 = select(&[&lt2.view(), &lt4.view()], &[&x.view(), &x10.view()], 0).unwrap();
        assert_eq!(out0.as_slice(), &[0, 1, 20, 30, 0, 0]);
        let swapped = select(&[&lt4.view(), &lt2.view()], &[&x10.view(), &x.view()], -1).unwrap();
        assert_eq!(swapped.as_slice(), &[0, 10, 20, 30, -1, -1]);
        assert!(select::<i32>(&[], &[], 0).is_err());
    }

    #[test]
    fn choose_matches_numpy_in_every_mode() {
        let choices: Vec<NdArray<i32>> =
            vec![arr(vec![10, 11, 12, 13], &[4]), arr(vec![20, 21, 22, 23], &[4]), arr(vec![30, 31, 32, 33], &[4])];
        let views: Vec<ArrayView<i32>> = choices.iter().map(|c| c.view()).collect();
        let refs: Vec<&ArrayView<i32>> = views.iter().collect();

        let ok = arr(vec![0, 1, 2, 1], &[4]);
        assert_eq!(choose(&ok.view(), &refs, ChooseMode::Raise).unwrap().as_slice(), &[10, 21, 32, 23]);

        let wild = arr(vec![-1, 3, 5, 1], &[4]);
        assert_eq!(choose(&wild.view(), &refs, ChooseMode::Wrap).unwrap().as_slice(), &[30, 11, 32, 23]);
        assert_eq!(choose(&wild.view(), &refs, ChooseMode::Clip).unwrap().as_slice(), &[10, 31, 32, 23]);
        assert_eq!(
            choose(&wild.view(), &refs, ChooseMode::Raise).unwrap_err(),
            ShapeError::ChoiceIndexOutOfBounds { index: -1, choices: 3 }
        );
    }

    #[test]
    fn choose_broadcasts_indices_against_choices() {
        let a = arr(vec![1, 2, 3], &[3]);
        let b = arr(vec![10, 20, 30], &[3]);
        let idx = arr(vec![0i64, 1], &[2, 1]);
        let out = choose(&idx.view(), &[&a.view(), &b.view()], ChooseMode::Raise).unwrap();
        assert_eq!(out.shape(), &[2, 3]);
        assert_eq!(out.as_slice(), &[1, 2, 3, 10, 20, 30]);
    }
}
