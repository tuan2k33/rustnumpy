use crate::error::{OpError, ShapeError};
use crate::ndarray::NdArray;
use crate::ufunc::zip_with;
use crate::view::ArrayView;
use crate::dispatch::{WrapAdd, WrapMul};

fn advance(idx: &mut [usize], labels: &[usize], sizes: &[usize], strides: &[Vec<isize>], off: &mut [isize]) -> bool {
    for d in (0..labels.len()).rev() {
        let l = labels[d];
        idx[d] += 1;
        for (o, s) in off.iter_mut().zip(strides) {
            *o += s[l];
        }
        if idx[d] < sizes[l] {
            return true;
        }
        for (o, s) in off.iter_mut().zip(strides) {
            *o -= s[l] * sizes[l] as isize;
        }
        idx[d] = 0;
    }
    false
}

fn contract<T>(
    ops: &[&ArrayView<T>],
    op_labels: &[Vec<usize>],
    out_labels: &[usize],
    sizes: &[usize],
) -> Result<NdArray<T>, ShapeError>
where
    T: Copy + Default + WrapAdd + WrapMul,
{
    let nl = sizes.len();
    let strides: Vec<Vec<isize>> = ops
        .iter()
        .zip(op_labels)
        .map(|(v, labels)| {
            let mut s = vec![0isize; nl];
            for (d, &l) in labels.iter().enumerate() {
                if v.shape()[d] == sizes[l] {
                    s[l] += v.strides()[d];
                }
            }
            s
        })
        .collect();
    let mut used = vec![false; nl];
    op_labels.iter().flatten().for_each(|&l| used[l] = true);
    let sum_labels: Vec<usize> = (0..nl).filter(|l| used[*l] && !out_labels.contains(l)).collect();
    let out_shape: Vec<usize> = out_labels.iter().map(|&l| sizes[l]).collect();
    let out_len: usize = out_shape.iter().product();
    let sum_len: usize = sum_labels.iter().map(|&l| sizes[l]).product();

    let raws: Vec<(&[T], isize)> = ops
        .iter()
        .map(|v| {
            let (d, o) = v.raw();
            (d, o as isize)
        })
        .collect();
    let mut data = Vec::with_capacity(out_len);
    let mut out_idx = vec![0usize; out_labels.len()];
    let mut base = vec![0isize; ops.len()];
    let mut sum_idx = vec![0usize; sum_labels.len()];
    if out_len > 0 {
        loop {
            let mut acc = T::default();
            if sum_len > 0 {
                sum_idx.iter_mut().for_each(|i| *i = 0);
                let mut off = base.clone();
                loop {
                    let mut prod = raws[0].0[(raws[0].1 + off[0]) as usize];
                    for k in 1..raws.len() {
                        prod = prod.wrap_mul(raws[k].0[(raws[k].1 + off[k]) as usize]);
                    }
                    acc = acc.wrap_add(prod);
                    if !advance(&mut sum_idx, &sum_labels, sizes, &strides, &mut off) {
                        break;
                    }
                }
            }
            data.push(acc);
            if !advance(&mut out_idx, out_labels, sizes, &strides, &mut base) {
                break;
            }
        }
    }
    NdArray::from_vec(data, &out_shape)
}

fn merge_size(current: usize, dim: usize) -> Option<usize> {
    match (current, dim) {
        (c, d) if c == d => Some(c),
        (1, d) => Some(d),
        (c, 1) => Some(c),
        _ => None,
    }
}

pub trait MatmulKernel: Copy + Default + WrapAdd + WrapMul {
    fn matmul_2d(a: &[Self], b: &[Self], m: usize, k: usize, n: usize) -> Vec<Self> {
        let mut c = vec![Self::default(); m * n];
        if n == 0 {
            return c;
        }
        for (a_row, c_row) in a.chunks_exact(k.max(1)).zip(c.chunks_exact_mut(n)).take(m) {
            for (&aip, b_row) in a_row.iter().zip(b.chunks_exact(n)) {
                for (cj, &bj) in c_row.iter_mut().zip(b_row) {
                    *cj = cj.wrap_add(aip.wrap_mul(bj));
                }
            }
        }
        c
    }
}

macro_rules! matmul_by_loops {
    ($($t:ty),*) => {$( impl MatmulKernel for $t {} )*};
}
matmul_by_loops!(bool, i8, i16, i32, i64, u8, u16, u32, u64, half::f16);

fn matmul_par(work: usize) -> faer::Par {
    let cores = std::thread::available_parallelism().map_or(1, |n| n.get());
    let want = match work {
        w if w < 1 << 23 => 1,
        w if w < 1 << 26 => 4,
        _ => 8,
    };
    match want.min(cores) {
        0 | 1 => faer::Par::Seq,
        n => faer::Par::rayon(n),
    }
}

macro_rules! matmul_by_faer {
    ($($t:ty),*) => {$(
        impl MatmulKernel for $t {
            fn matmul_2d(a: &[Self], b: &[Self], m: usize, k: usize, n: usize) -> Vec<Self> {
                if m == 0 || n == 0 || k == 0 {
                    return vec![Self::default(); m * n];
                }
                let a_t = faer::MatRef::from_column_major_slice(a, k, m);
                let b_t = faer::MatRef::from_column_major_slice(b, n, k);
                let mut out = vec![Self::default(); m * n];
                let c_t = faer::MatMut::from_column_major_slice_mut(&mut out, n, m);
                faer::linalg::matmul::matmul(c_t, faer::Accum::Replace, b_t, a_t, <$t as num_traits::One>::one(), matmul_par(m * n * k));
                out
            }
        }
    )*};
}
matmul_by_faer!(f32, f64, num_complex::Complex<f32>, num_complex::Complex<f64>);

pub fn matmul<T>(a: &ArrayView<T>, b: &ArrayView<T>) -> Result<NdArray<T>, ShapeError>
where
    T: MatmulKernel,
{
    if a.ndim() == 0 || b.ndim() == 0 {
        return Err(ShapeError::ZeroDimOperand);
    }
    if a.ndim() == 2 && b.ndim() == 2 {
        let (m, k, n) = (a.shape()[0], a.shape()[1], b.shape()[1]);
        if b.shape()[0] != k {
            return Err(ShapeError::ContractionMismatch { lhs: a.shape().to_vec(), rhs: b.shape().to_vec() });
        }
        let (oa, ob);
        let sa = match a.as_slice_c() {
            Some(s) => s,
            None => {
                oa = a.to_owned();
                oa.as_slice()
            }
        };
        let sb = match b.as_slice_c() {
            Some(s) => s,
            None => {
                ob = b.to_owned();
                ob.as_slice()
            }
        };
        return NdArray::from_vec(T::matmul_2d(sa, sb, m, k, n), &[m, n]);
    }
    let mismatch = || ShapeError::ContractionMismatch { lhs: a.shape().to_vec(), rhs: b.shape().to_vec() };
    let (a_batch, b_batch) = (a.ndim().saturating_sub(2), b.ndim().saturating_sub(2));
    let batch = a_batch.max(b_batch);
    let mut sizes: Vec<usize> = vec![1; batch];
    let mut la: Vec<usize> = Vec::new();
    let mut lb: Vec<usize> = Vec::new();
    for (i, size) in sizes.iter_mut().enumerate() {
        let from_end = batch - i;
        if a_batch >= from_end {
            let d = a.shape()[a_batch - from_end];
            *size = merge_size(*size, d).ok_or_else(|| ShapeError::NotBroadcastable { lhs: a.shape().to_vec(), rhs: b.shape().to_vec() })?;
            la.push(i);
        }
        if b_batch >= from_end {
            let d = b.shape()[b_batch - from_end];
            *size = merge_size(*size, d).ok_or_else(|| ShapeError::NotBroadcastable { lhs: a.shape().to_vec(), rhs: b.shape().to_vec() })?;
            lb.push(i);
        }
    }
    let k_id = sizes.len();
    let k_a = *a.shape().last().unwrap();
    let k_b = if b.ndim() == 1 { b.shape()[0] } else { b.shape()[b.ndim() - 2] };
    if k_a != k_b {
        return Err(mismatch());
    }
    sizes.push(k_a);
    let mut out: Vec<usize> = (0..batch).collect();
    if a.ndim() >= 2 {
        let m_id = sizes.len();
        sizes.push(a.shape()[a.ndim() - 2]);
        la.push(m_id);
        out.push(m_id);
    }
    la.push(k_id);
    lb.push(k_id);
    if b.ndim() >= 2 {
        let n_id = sizes.len();
        sizes.push(*b.shape().last().unwrap());
        lb.push(n_id);
        out.push(n_id);
    }
    contract(&[a, b], &[la, lb], &out, &sizes)
}

pub fn tensordot<T>(
    a: &ArrayView<T>,
    b: &ArrayView<T>,
    axes_a: &[usize],
    axes_b: &[usize],
) -> Result<NdArray<T>, ShapeError>
where
    T: Copy + Default + WrapAdd + WrapMul,
{
    let mismatch = || ShapeError::ContractionMismatch { lhs: a.shape().to_vec(), rhs: b.shape().to_vec() };
    if axes_a.len() != axes_b.len() {
        return Err(mismatch());
    }
    for (&x, &y) in axes_a.iter().zip(axes_b) {
        if x >= a.ndim() {
            return Err(ShapeError::AxisOutOfBounds { axis: x, ndim: a.ndim() });
        }
        if y >= b.ndim() {
            return Err(ShapeError::AxisOutOfBounds { axis: y, ndim: b.ndim() });
        }
        if a.shape()[x] != b.shape()[y] {
            return Err(mismatch());
        }
    }
    let mut sizes: Vec<usize> = Vec::new();
    let mut la = vec![usize::MAX; a.ndim()];
    let mut lb = vec![usize::MAX; b.ndim()];
    for (&x, &y) in axes_a.iter().zip(axes_b) {
        la[x] = sizes.len();
        lb[y] = sizes.len();
        sizes.push(a.shape()[x]);
    }
    let mut out = Vec::new();
    for (d, l) in la.iter_mut().enumerate() {
        if *l == usize::MAX {
            *l = sizes.len();
            out.push(sizes.len());
            sizes.push(a.shape()[d]);
        }
    }
    for (d, l) in lb.iter_mut().enumerate() {
        if *l == usize::MAX {
            *l = sizes.len();
            out.push(sizes.len());
            sizes.push(b.shape()[d]);
        }
    }
    contract(&[a, b], &[la, lb], &out, &sizes)
}

pub fn tensordot_n<T>(a: &ArrayView<T>, b: &ArrayView<T>, n: usize) -> Result<NdArray<T>, ShapeError>
where
    T: Copy + Default + WrapAdd + WrapMul,
{
    if n > a.ndim() || n > b.ndim() {
        return Err(ShapeError::ContractionMismatch { lhs: a.shape().to_vec(), rhs: b.shape().to_vec() });
    }
    let axes_a: Vec<usize> = (a.ndim() - n..a.ndim()).collect();
    let axes_b: Vec<usize> = (0..n).collect();
    tensordot(a, b, &axes_a, &axes_b)
}

pub fn dot<T>(a: &ArrayView<T>, b: &ArrayView<T>) -> Result<NdArray<T>, ShapeError>
where
    T: Copy + Default + WrapAdd + WrapMul,
{
    if a.ndim() == 0 || b.ndim() == 0 {
        return zip_with(a, b, |x, y| x.wrap_mul(y));
    }
    let axis_b = if b.ndim() == 1 { 0 } else { b.ndim() - 2 };
    tensordot(a, b, &[a.ndim() - 1], &[axis_b])
}

pub fn outer<T>(a: &ArrayView<T>, b: &ArrayView<T>) -> Result<NdArray<T>, ShapeError>
where
    T: Copy + Default + WrapAdd + WrapMul,
{
    let (fa, fb) = (a.to_owned(), b.to_owned());
    let (xs, ys) = (fa.as_slice(), fb.as_slice());
    let data: Vec<T> = xs.iter().flat_map(|&x| ys.iter().map(move |&y| x.wrap_mul(y))).collect();
    NdArray::from_vec(data, &[xs.len(), ys.len()])
}

pub fn trace<T>(view: &ArrayView<T>, offset: isize) -> Result<T, ShapeError>
where
    T: Copy + Default + WrapAdd,
{
    Ok(view.diagonal(offset)?.iter().fold(T::default(), |acc, x| acc.wrap_add(x)))
}

pub fn kron<T>(a: &ArrayView<T>, b: &ArrayView<T>) -> Result<NdArray<T>, ShapeError>
where
    T: Copy + WrapMul,
{
    let nd = a.ndim().max(b.ndim());
    let pad = |shape: &[usize]| -> Vec<usize> {
        std::iter::repeat_n(1, nd - shape.len()).chain(shape.iter().copied()).collect()
    };
    let (sa, sb) = (pad(a.shape()), pad(b.shape()));
    let out_shape: Vec<usize> = sa.iter().zip(&sb).map(|(x, y)| x * y).collect();
    let (av, bv) = (a.to_owned(), b.to_owned());
    let (a_strides, b_strides) = (crate::shape::c_contiguous_strides(&sa), crate::shape::c_contiguous_strides(&sb));
    let data: Vec<T> = crate::shape::IndexIter::new(&out_shape)
        .map(|idx| {
            let (mut ia, mut ib) = (0isize, 0isize);
            for d in 0..nd {
                ia += (idx[d] / sb[d]) as isize * a_strides[d];
                ib += (idx[d] % sb[d]) as isize * b_strides[d];
            }
            av.as_slice()[ia as usize].wrap_mul(bv.as_slice()[ib as usize])
        })
        .collect();
    NdArray::from_vec(data, &out_shape)
}

pub fn cross<T>(a: &ArrayView<T>, b: &ArrayView<T>) -> Result<NdArray<T>, OpError>
where
    T: Copy + Default + WrapMul + crate::dispatch::WrapSub,
{
    for v in [a, b] {
        if v.shape().last() != Some(&3) {
            return Err(OpError::InvalidGufunc {
                reason: format!("cross needs 3-dimensional vectors along the last axis, got shape {:?}", v.shape()),
            });
        }
    }
    let mut out = crate::gufunc::gufunc("(n),(n)->(n)", &[a, b], |ins, outs, _| {
        let (x, y) = (ins[0], ins[1]);
        outs[0][0] = x[1].wrap_mul(y[2]).wrap_sub(x[2].wrap_mul(y[1]));
        outs[0][1] = x[2].wrap_mul(y[0]).wrap_sub(x[0].wrap_mul(y[2]));
        outs[0][2] = x[0].wrap_mul(y[1]).wrap_sub(x[1].wrap_mul(y[0]));
    })?;
    Ok(out.remove(0))
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Token {
    Label(char),
    Ellipsis,
}

fn tokenize(term: &str) -> Result<Vec<Token>, OpError> {
    let bad = |reason: String| OpError::InvalidEinsum { reason };
    let chars: Vec<char> = term.chars().collect();
    let mut out = Vec::new();
    let mut i = 0;
    while i < chars.len() {
        match chars[i] {
            c if c.is_ascii_alphabetic() => {
                out.push(Token::Label(c));
                i += 1;
            }
            '.' if chars.get(i..i + 3) == Some(&['.', '.', '.']) => {
                if out.contains(&Token::Ellipsis) {
                    return Err(bad("an ellipsis may appear only once per operand".into()));
                }
                out.push(Token::Ellipsis);
                i += 3;
            }
            c => return Err(bad(format!("invalid character {c:?} in subscripts"))),
        }
    }
    Ok(out)
}

pub fn einsum<T>(subscripts: &str, operands: &[&ArrayView<T>]) -> Result<NdArray<T>, OpError>
where
    T: Copy + Default + WrapAdd + WrapMul,
{
    let bad = |reason: &str| OpError::InvalidEinsum { reason: reason.to_string() };
    if operands.is_empty() {
        return Err(ShapeError::EmptyArrayList.into());
    }
    let text: String = subscripts.chars().filter(|c| !c.is_whitespace()).collect();
    let (lhs, rhs) = match text.split_once("->") {
        Some((l, r)) => (l.to_string(), Some(r.to_string())),
        None => (text.clone(), None),
    };
    let terms: Vec<&str> = lhs.split(',').collect();
    if terms.len() != operands.len() {
        return Err(bad("number of subscript terms does not match the number of operands"));
    }
    let parsed: Vec<Vec<Token>> = terms.iter().map(|t| tokenize(t)).collect::<Result<_, _>>()?;

    let mut ellipsis_dims = vec![0usize; operands.len()];
    for (k, (toks, op)) in parsed.iter().zip(operands).enumerate() {
        let labels = toks.iter().filter(|t| matches!(t, Token::Label(_))).count();
        if toks.contains(&Token::Ellipsis) {
            ellipsis_dims[k] = op.ndim().checked_sub(labels).ok_or_else(|| bad("too many subscripts for operand"))?;
        } else if labels != op.ndim() {
            return Err(bad(if labels > op.ndim() {
                "subscripts string contains too many subscripts for operand"
            } else {
                "operand has more dimensions than subscripts given, but no '...' ellipsis provided to broadcast the extra dimensions"
            }));
        }
    }
    let e = ellipsis_dims.iter().copied().max().unwrap_or(0);

    let mut letters: Vec<char> = Vec::new();
    let id_of = |c: char, letters: &mut Vec<char>| -> usize {
        e + letters.iter().position(|&x| x == c).unwrap_or_else(|| {
            letters.push(c);
            letters.len() - 1
        })
    };
    let mut op_labels: Vec<Vec<usize>> = Vec::new();
    let mut counts: std::collections::HashMap<char, usize> = std::collections::HashMap::new();
    for (k, toks) in parsed.iter().enumerate() {
        let mut ids = Vec::new();
        for t in toks {
            match *t {
                Token::Label(c) => {
                    *counts.entry(c).or_insert(0) += 1;
                    ids.push(id_of(c, &mut letters));
                }
                Token::Ellipsis => ids.extend(e - ellipsis_dims[k]..e),
            }
        }
        op_labels.push(ids);
    }
    let mut sizes = vec![1usize; e + letters.len()];
    let mut seen_dim = vec![false; sizes.len()];
    for (op, labels) in operands.iter().zip(&op_labels) {
        let mut in_this_op = vec![false; sizes.len()];
        for (d, &l) in labels.iter().enumerate() {
            let dim = op.shape()[d];
            if in_this_op[l] && dim != sizes[l] {
                return Err(bad("dimensions for a repeated index in one operand do not match"));
            }
            in_this_op[l] = true;
            sizes[l] = if seen_dim[l] {
                merge_size(sizes[l], dim).ok_or_else(|| bad("operands could not be broadcast together with remapped shapes"))?
            } else {
                dim
            };
            seen_dim[l] = true;
        }
    }

    let out_labels: Vec<usize> = match rhs {
        Some(r) => {
            let toks = tokenize(&r)?;
            let mut ids = Vec::new();
            for t in toks {
                match t {
                    Token::Label(c) => {
                        let pos = letters.iter().position(|&x| x == c).ok_or_else(|| bad("output subscript never appeared in an input"))?;
                        if ids.contains(&(e + pos)) {
                            return Err(bad("output subscript appears multiple times"));
                        }
                        ids.push(e + pos);
                    }
                    Token::Ellipsis => ids.extend(0..e),
                }
            }
            if e > 0 && !(0..e).all(|l| ids.contains(&l)) {
                return Err(bad("output has more dimensions than subscripts given, but no '...' ellipsis provided to broadcast the extra dimensions"));
            }
            ids
        }
        None => {
            let mut singles: Vec<char> = counts.iter().filter(|(_, n)| **n == 1).map(|(&c, _)| c).collect();
            singles.sort_unstable();
            (0..e).chain(singles.into_iter().map(|c| e + letters.iter().position(|&x| x == c).unwrap())).collect()
        }
    };
    contract(operands, &op_labels, &out_labels, &sizes).map_err(OpError::from)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matmul_2d_fast_path_matches_the_general_contraction() {
        let a = NdArray::from_vec((0..12).map(|i| i as f64 * 0.5 - 2.0).collect(), &[3, 4]).unwrap();
        let b = NdArray::from_vec((0..8).map(|i| 1.0 - i as f64 * 0.25).collect(), &[4, 2]).unwrap();
        let general = contract(&[&a.view(), &b.view()], &[vec![0, 1], vec![1, 2]], &[0, 2], &[3, 4, 2]).unwrap();
        let fast = matmul(&a.view(), &b.view()).unwrap();
        assert_eq!(fast.shape(), &[3, 2]);
        for (x, y) in fast.as_slice().iter().zip(general.as_slice()) {
            assert!((x - y).abs() < 1e-12);
        }
        let at = a.view().matrix_transpose().unwrap();
        let t = matmul(&at, &a.view()).unwrap();
        assert_eq!(t.shape(), &[4, 4]);
        assert_eq!(t.get(&[0, 0]), Some(4.0 + 0.0 + 4.0));
        let i8s = NdArray::from_vec(vec![100i8, 100, 1, 1], &[2, 2]).unwrap();
        assert_eq!(matmul(&i8s.view(), &i8s.view()).unwrap().as_slice(), &[116, 116, 101, 101]);
        let c = NdArray::from_vec(vec![num_complex::Complex::new(0.0f32, 1.0); 4], &[2, 2]).unwrap();
        assert_eq!(matmul(&c.view(), &c.view()).unwrap().as_slice()[0], num_complex::Complex::new(-2.0, 0.0));
        let empty = NdArray::<f64>::zeros(&[2, 0]);
        assert_eq!(matmul(&empty.view(), &NdArray::<f64>::zeros(&[0, 3]).view()).unwrap().as_slice(), &[0.0; 6]);
    }

    fn ar(n: usize, shape: &[usize]) -> NdArray<i64> {
        NdArray::from_vec((0..n as i64).collect(), shape).unwrap()
    }

    trait MapAdd {
        fn map_add(self, k: i64) -> Self;
    }

    impl MapAdd for NdArray<i64> {
        fn map_add(self, k: i64) -> Self {
            let shape = self.shape().to_vec();
            NdArray::from_vec(self.into_vec().into_iter().map(|v| v + k).collect(), &shape).unwrap()
        }
    }

    #[test]
    fn matmul_covers_2d_1d_and_batched_shapes() {
        let (a, b) = (ar(6, &[2, 3]), ar(12, &[3, 4]));
        assert_eq!(matmul(&a.view(), &b.view()).unwrap().as_slice(), &[20, 23, 26, 29, 56, 68, 80, 92]);
        let v = ar(3, &[3]);
        assert_eq!(matmul(&v.view(), &v.view()).unwrap().shape(), &[] as &[usize]);
        assert_eq!(matmul(&v.view(), &v.view()).unwrap().as_slice(), &[5]);
        assert_eq!(matmul(&v.view(), &b.view()).unwrap().as_slice(), &[20, 23, 26, 29]);
        assert_eq!(matmul(&a.view(), &v.view()).unwrap().as_slice(), &[5, 14]);

        let (s, t) = (ar(24, &[2, 3, 4]), ar(8, &[4, 2]));
        let out = matmul(&s.view(), &t.view()).unwrap();
        assert_eq!(out.shape(), &[2, 3, 2]);
        assert_eq!(&out.as_slice()[6..], &[172, 226, 220, 290, 268, 354]);

        let x = NdArray::from_vec((0..12).collect::<Vec<i64>>(), &[3, 1, 4]).unwrap();
        let xv = x.view().slice(&[0..3, 0..1, 0..3]).unwrap();
        let y = ar(6, &[1, 3, 2]);
        assert_eq!(matmul(&xv, &y.view()).unwrap().shape(), &[3, 1, 2]);
    }

    #[test]
    fn matmul_errors_match_numpy() {
        let a = ar(6, &[2, 3]);
        let scalar = NdArray::from_vec(vec![3i64], &[]).unwrap();
        assert_eq!(matmul(&scalar.view(), &a.view()).unwrap_err(), ShapeError::ZeroDimOperand);
        assert!(matches!(matmul(&a.view(), &a.view()), Err(ShapeError::ContractionMismatch { .. })));
        let p = ar(12, &[3, 2, 2]);
        let q = ar(8, &[2, 2, 2]);
        assert!(matches!(matmul(&p.view(), &q.view()), Err(ShapeError::NotBroadcastable { .. })));
    }

    #[test]
    fn dot_follows_numpys_sum_over_last_and_second_to_last() {
        let (a, b) = (ar(6, &[2, 3]), ar(12, &[3, 4]));
        assert_eq!(dot(&a.view(), &b.view()).unwrap().as_slice(), &[20, 23, 26, 29, 56, 68, 80, 92]);
        let three = NdArray::from_vec(vec![3i64], &[]).unwrap();
        assert_eq!(dot(&three.view(), &a.view()).unwrap().as_slice(), &[0, 3, 6, 9, 12, 15]);
        let (s, t) = (ar(24, &[2, 3, 4]), ar(40, &[5, 4, 2]));
        let r = dot(&s.view(), &t.view()).unwrap();
        assert_eq!(r.shape(), &[2, 3, 5, 2]);
        assert_eq!(r.get(&[1, 2, 3, 1]), Some(2418));
        let v = ar(4, &[4]);
        assert_eq!(dot(&s.view(), &v.view()).unwrap().shape(), &[2, 3]);
    }

    #[test]
    fn tensordot_supports_integer_and_explicit_axes() {
        let (a, b) = (ar(6, &[2, 3]), ar(12, &[3, 4]));
        assert_eq!(tensordot_n(&a.view(), &b.view(), 1).unwrap().as_slice(), &[20, 23, 26, 29, 56, 68, 80, 92]);
        assert_eq!(tensordot_n(&a.view(), &a.view(), 2).unwrap().as_slice(), &[55]);
        assert_eq!(tensordot_n(&a.view(), &a.view(), 0).unwrap().shape(), &[2, 3, 2, 3]);
        let (s, t) = (ar(24, &[2, 3, 4]), ar(24, &[4, 3, 2]));
        let r = tensordot(&s.view(), &t.view(), &[1, 2], &[1, 0]).unwrap();
        assert_eq!(r.as_slice(), &[880, 946, 2464, 2674]);
        assert!(matches!(tensordot(&s.view(), &t.view(), &[1], &[0]), Err(ShapeError::ContractionMismatch { .. })));
    }

    #[test]
    fn outer_flattens_both_operands() {
        let (a, b) = (ar(3, &[3]), ar(4, &[4]));
        assert_eq!(outer(&a.view(), &b.view()).unwrap().as_slice(), &[0, 0, 0, 0, 0, 1, 2, 3, 0, 2, 4, 6]);
        let m = ar(6, &[2, 3]);
        let v = NdArray::from_vec(vec![1i64, 2], &[2]).unwrap();
        assert_eq!(outer(&m.view(), &v.view()).unwrap().shape(), &[6, 2]);
    }

    #[test]
    fn einsum_explicit_and_implicit_forms_match_numpy() {
        let (a, b) = (ar(6, &[2, 3]), ar(12, &[3, 4]));
        let (av, bv) = (a.view(), b.view());
        assert_eq!(einsum("ij,jk->ik", &[&av, &bv]).unwrap().as_slice(), &[20, 23, 26, 29, 56, 68, 80, 92]);
        assert_eq!(einsum("ij,jk", &[&av, &bv]).unwrap().shape(), &[2, 4]);
        assert_eq!(einsum("ij->ji", &[&av]).unwrap().as_slice(), &[0, 3, 1, 4, 2, 5]);
        assert_eq!(einsum("ji", &[&av]).unwrap().shape(), &[3, 2]);
        assert_eq!(einsum("ij->", &[&av]).unwrap().as_slice(), &[15]);
        assert_eq!(einsum("ij->j", &[&av]).unwrap().as_slice(), &[3, 5, 7]);
        assert_eq!(einsum("ij,ij->", &[&av, &av]).unwrap().as_slice(), &[55]);
        assert_eq!(einsum("ij,jk->ki", &[&av, &bv]).unwrap().as_slice(), &[20, 56, 23, 68, 26, 80, 29, 92]);
        let (u, v) = (ar(2, &[2]), ar(3, &[3]));
        assert_eq!(einsum("i,j->ij", &[&u.view(), &v.view()]).unwrap().as_slice(), &[0, 0, 0, 0, 1, 2]);
        assert_eq!(einsum("i,i", &[&v.view(), &v.view()]).unwrap().as_slice(), &[5]);
        let sq = ar(4, &[2, 2]);
        assert_eq!(einsum("ba,ab", &[&sq.view(), &sq.view()]).unwrap().as_slice(), &[13]);
    }

    #[test]
    fn einsum_traces_diagonals_and_batches() {
        let m = ar(9, &[3, 3]);
        assert_eq!(einsum("ii", &[&m.view()]).unwrap().as_slice(), &[12]);
        assert_eq!(einsum("ii->i", &[&m.view()]).unwrap().as_slice(), &[0, 4, 8]);
        let (p, q) = (ar(12, &[2, 2, 3]), ar(12, &[2, 3, 2]));
        let want = [10, 13, 28, 40, 172, 193, 244, 274];
        assert_eq!(einsum("bij,bjk->bik", &[&p.view(), &q.view()]).unwrap().as_slice(), &want);
        let ell = einsum("...ij,...jk->...ik", &[&p.view(), &q.view()]).unwrap();
        assert_eq!(ell.shape(), &[2, 2, 2]);
        assert_eq!(ell.as_slice(), &want);
        let a = ar(6, &[2, 3]);
        assert_eq!(einsum("i...->...", &[&a.view()]).unwrap().as_slice(), &[3, 5, 7]);
        let s = ar(12, &[2, 2, 3]);
        assert_eq!(einsum("ij...->...", &[&s.view()]).unwrap().as_slice(), &[18, 22, 26]);
        assert_eq!(einsum("ii...->i...", &[&s.view()]).unwrap().as_slice(), &[0, 1, 2, 9, 10, 11]);
    }

    #[test]
    fn einsum_broadcasts_size_one_dims_and_rejects_bad_subscripts() {
        let ones_col = NdArray::from_vec(vec![1i64, 2, 3], &[3, 1]).unwrap();
        let full = NdArray::from_vec(vec![1i64; 12], &[3, 4]).unwrap();
        let out = einsum("ij,ij->ij", &[&ones_col.view(), &full.view()]).unwrap();
        assert_eq!(out.shape(), &[3, 4]);
        assert_eq!(&out.as_slice()[..4], &[1, 1, 1, 1]);
        assert_eq!(&out.as_slice()[8..], &[3, 3, 3, 3]);

        let a = ar(6, &[2, 3]);
        let v = ar(3, &[3]);
        let av = a.view();
        for (s, ops) in [
            ("ij,j->ikk", vec![&av, &v.view()]),
            ("ij,j->k", vec![&av, &v.view()]),
            ("ijk", vec![&av]),
            ("i", vec![&av]),
            ("ij,jk", vec![&av]),
            ("i->ii", vec![&v.view()]),
            ("ij...->...i...", vec![&av]),
        ] {
            let borrowed: Vec<&ArrayView<i64>> = ops.to_vec();
            assert!(matches!(einsum(s, &borrowed), Err(OpError::InvalidEinsum { .. })), "{s}");
        }
        assert!(einsum("...j->j", &[&ar(12, &[2, 2, 3]).view()]).is_err());
        assert!(einsum("ii", &[&NdArray::from_vec(vec![1i64, 2, 3], &[3, 1]).unwrap().view()]).is_err());
        assert!(einsum("ab,b", &[&ar(6, &[2, 3]).view(), &ar(4, &[4]).view()]).is_err());
    }

    #[test]
    fn trace_kron_and_cross_match_numpy() {
        let a = NdArray::from_vec((0..12).map(f64::from).collect(), &[3, 4]).unwrap();
        assert_eq!(trace(&a.view(), 0).unwrap(), 15.0);
        assert_eq!(trace(&a.view(), -1).unwrap(), 13.0);
        assert_eq!(trace(&a.view(), 5).unwrap(), 0.0);

        let (p, q) = (ar(4, &[2, 2]).map_add(1), NdArray::from_vec(vec![0i64, 1, 1, 0], &[2, 2]).unwrap());
        assert_eq!(kron(&p.view(), &q.view()).unwrap().as_slice(), &[0, 1, 0, 2, 1, 0, 2, 0, 0, 3, 0, 4, 3, 0, 4, 0]);
        let one_d = ar(2, &[2]).map_add(1);
        let row = NdArray::from_vec(vec![1i64, 2, 3], &[1, 3]).unwrap();
        assert_eq!(kron(&one_d.view(), &row.view()).unwrap().shape(), &[1, 6]);
        let v3 = NdArray::from_vec(vec![1i64, 2, 3], &[3]).unwrap();
        let v2 = NdArray::from_vec(vec![1i64, 1], &[2]).unwrap();
        assert_eq!(kron(&v3.view(), &v2.view()).unwrap().as_slice(), &[1, 1, 2, 2, 3, 3]);

        let (x, y) = (NdArray::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap(), NdArray::from_vec(vec![4.0, 5.0, 6.0], &[3]).unwrap());
        assert_eq!(cross(&x.view(), &y.view()).unwrap().as_slice(), &[-3.0, 6.0, -3.0]);
        let batch = NdArray::from_vec((0..6).map(f64::from).collect(), &[2, 3]).unwrap();
        let e0 = NdArray::from_vec(vec![1.0, 0.0, 0.0], &[3]).unwrap();
        assert_eq!(cross(&batch.view(), &e0.view()).unwrap().as_slice(), &[0.0, 2.0, -1.0, 0.0, 5.0, -4.0]);
        let two = NdArray::from_vec(vec![1.0, 2.0], &[2]).unwrap();
        assert!(cross(&two.view(), &two.view()).is_err());
    }

    #[test]
    fn contraction_works_on_floats_and_complex_and_non_contiguous_views() {
        let a = NdArray::from_vec(vec![1.5f64, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        let t = a.view().slice(&[0..2, 0..1]).unwrap();
        let out = matmul(&a.view(), &t).unwrap();
        assert_eq!(out.as_slice(), &[1.5 * 1.5 + 2.0 * 3.0, 3.0 * 1.5 + 4.0 * 3.0]);
        let c = crate::fft::Complex64::new(0.0, 1.0);
        let z = NdArray::from_vec(vec![c, c], &[2]).unwrap();
        assert_eq!(dot(&z.view(), &z.view()).unwrap().as_slice(), &[crate::fft::Complex64::new(-2.0, 0.0)]);
    }
}
