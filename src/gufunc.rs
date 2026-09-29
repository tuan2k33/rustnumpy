use crate::error::ShapeError;
use crate::ndarray::NdArray;
use crate::shape::broadcast_shapes;
use crate::view::ArrayView;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Signature {
    pub inputs: Vec<Vec<String>>,
    pub outputs: Vec<Vec<String>>,
}

fn bad(reason: impl Into<String>) -> ShapeError {
    ShapeError::InvalidGufunc { reason: reason.into() }
}

fn parse_side(side: &str) -> Result<Vec<Vec<String>>, ShapeError> {
    let mut groups = Vec::new();
    let mut rest = side;
    while !rest.is_empty() {
        let body = rest.strip_prefix('(').ok_or_else(|| bad(format!("expected '(' in {side:?}")))?;
        let close = body.find(')').ok_or_else(|| bad(format!("unclosed '(' in {side:?}")))?;
        let names: Vec<String> = if body[..close].is_empty() {
            Vec::new()
        } else {
            body[..close].split(',').map(str::to_string).collect()
        };
        if names.iter().any(|n| n.is_empty() || !n.chars().all(|c| c.is_alphanumeric() || c == '_')) {
            return Err(bad(format!("invalid dimension name in {side:?}")));
        }
        groups.push(names);
        rest = &body[close + 1..];
        if let Some(next) = rest.strip_prefix(',') {
            if next.is_empty() {
                return Err(bad(format!("trailing ',' in {side:?}")));
            }
            rest = next;
        } else if !rest.is_empty() {
            return Err(bad(format!("expected ',' between operands in {side:?}")));
        }
    }
    Ok(groups)
}

impl Signature {
    pub fn parse(text: &str) -> Result<Self, ShapeError> {
        let clean: String = text.chars().filter(|c| !c.is_whitespace()).collect();
        let (lhs, rhs) = clean.split_once("->").ok_or_else(|| bad("signature needs '->'"))?;
        let (inputs, outputs) = (parse_side(lhs)?, parse_side(rhs)?);
        if inputs.is_empty() || outputs.is_empty() {
            return Err(bad("signature needs at least one input and one output"));
        }
        Ok(Self { inputs, outputs })
    }

    pub fn core_dim_names(&self) -> Vec<String> {
        let mut names: Vec<String> = Vec::new();
        for n in self.inputs.iter().chain(&self.outputs).flatten() {
            if !names.contains(n) {
                names.push(n.clone());
            }
        }
        names
    }
}

pub fn gufunc<T: Copy + Default>(
    signature: &str,
    inputs: &[&ArrayView<T>],
    kernel: impl Fn(&[&[T]], &mut [&mut [T]], &[usize]),
) -> Result<Vec<NdArray<T>>, ShapeError> {
    let sig = Signature::parse(signature)?;
    if inputs.len() != sig.inputs.len() {
        return Err(bad(format!("signature expects {} inputs, got {}", sig.inputs.len(), inputs.len())));
    }
    let names = sig.core_dim_names();
    let mut dims: Vec<Option<usize>> = vec![None; names.len()];
    let mut loop_shape: Vec<usize> = Vec::new();
    for (k, (op, core)) in inputs.iter().zip(&sig.inputs).enumerate() {
        if op.ndim() < core.len() {
            return Err(bad(format!(
                "input operand {k} does not have enough dimensions (has {}, gufunc core requires {})",
                op.ndim(),
                core.len()
            )));
        }
        let split = op.ndim() - core.len();
        for (name, &size) in core.iter().zip(&op.shape()[split..]) {
            let slot = &mut dims[names.iter().position(|n| n == name).unwrap()];
            match *slot {
                Some(existing) if existing != size => {
                    return Err(bad(format!(
                        "input operand {k} has a mismatch in core dimension {name:?} (size {size} is different from {existing})"
                    )));
                }
                _ => *slot = Some(size),
            }
        }
        loop_shape = broadcast_shapes(&loop_shape, &op.shape()[..split]).ok_or_else(|| ShapeError::NotBroadcastable {
            lhs: loop_shape.clone(),
            rhs: op.shape()[..split].to_vec(),
        })?;
    }
    let sizes: Vec<usize> = names
        .iter()
        .zip(&dims)
        .map(|(n, d)| d.ok_or_else(|| bad(format!("core dimension {n:?} appears only in outputs and cannot be inferred"))))
        .collect::<Result<_, _>>()?;
    let size_of = |name: &String| sizes[names.iter().position(|n| n == name).unwrap()];

    let blocks: Vec<(ArrayView<T>, usize)> = inputs
        .iter()
        .zip(&sig.inputs)
        .map(|(op, core)| {
            let core_shape: Vec<usize> = core.iter().map(size_of).collect();
            let full: Vec<usize> = loop_shape.iter().copied().chain(core_shape.iter().copied()).collect();
            Ok((op.broadcast_to(&full)?, core_shape.iter().product()))
        })
        .collect::<Result<_, ShapeError>>()?;
    let out_shapes: Vec<Vec<usize>> = sig
        .outputs
        .iter()
        .map(|core| loop_shape.iter().copied().chain(core.iter().map(size_of)).collect())
        .collect();
    let out_core_len: Vec<usize> = sig.outputs.iter().map(|core| core.iter().map(size_of).product()).collect();
    let loop_len: usize = loop_shape.iter().product();
    let mut outs: Vec<Vec<T>> = out_core_len.iter().map(|&n| vec![T::default(); n * loop_len]).collect();

    for l in 0..loop_len {
        let in_blocks: Vec<Vec<T>> = blocks.iter().map(|(v, n)| v.iter_range(l * n, *n).collect()).collect();
        let in_refs: Vec<&[T]> = in_blocks.iter().map(Vec::as_slice).collect();
        let mut out_refs: Vec<&mut [T]> = outs
            .iter_mut()
            .zip(&out_core_len)
            .map(|(o, &n)| &mut o[l * n..(l + 1) * n])
            .collect();
        kernel(&in_refs, &mut out_refs, &sizes);
    }
    outs.into_iter().zip(&out_shapes).map(|(data, shape)| NdArray::from_vec(data, shape)).collect()
}

pub fn vecdot<T>(a: &ArrayView<T>, b: &ArrayView<T>) -> Result<NdArray<T>, ShapeError>
where
    T: Copy + Default + crate::dispatch::WrapAdd + crate::dispatch::WrapMul,
{
    let mut out = gufunc("(n),(n)->()", &[a, b], |ins, outs, _| {
        outs[0][0] = ins[0].iter().zip(ins[1]).fold(T::default(), |acc, (&x, &y)| acc.wrap_add(x.wrap_mul(y)));
    })?;
    Ok(out.remove(0))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::contraction::matmul;

    fn ar(n: usize, shape: &[usize]) -> NdArray<f64> {
        NdArray::from_vec((0..n).map(|v| v as f64).collect(), shape).unwrap()
    }

    #[test]
    fn signature_parses_operands_and_names_core_dims_in_order() {
        let s = Signature::parse("(m,n),(n,p)->(m,p)").unwrap();
        assert_eq!(s.inputs, vec![vec!["m", "n"], vec!["n", "p"]]);
        assert_eq!(s.outputs, vec![vec!["m", "p"]]);
        assert_eq!(s.core_dim_names(), vec!["m", "n", "p"]);
        let scalar = Signature::parse("(i),(i)->()").unwrap();
        assert_eq!(scalar.outputs, vec![Vec::<String>::new()]);
        for bad in ["(n)", "(n),->(n)", "n->n", "(n,)->()", "(n)(n)->()", "->(n)", "(n->()", "(a-b)->()"] {
            assert!(Signature::parse(bad).is_err(), "{bad}");
        }
    }

    #[test]
    fn vecdot_matches_numpy_including_loop_broadcasting() {
        let (a, b) = (ar(24, &[2, 3, 4]), ar(4, &[4]));
        let out = vecdot(&a.view(), &b.view()).unwrap();
        assert_eq!(out.shape(), &[2, 3]);
        assert_eq!(out.as_slice(), &[14.0, 38.0, 62.0, 86.0, 110.0, 134.0]);
        let c = ar(12, &[3, 1, 4]);
        assert_eq!(vecdot(&a.view(), &c.view().slice(&[0..1, 0..1, 0..4]).unwrap()).unwrap().shape(), &[2, 3]);
    }

    #[test]
    fn vecdot_errors_match_numpy() {
        let ones = |shape: &[usize]| NdArray::from_vec(vec![1.0; shape.iter().product()], shape).unwrap();
        assert!(matches!(
            vecdot(&ones(&[2, 3, 4]).view(), &ones(&[5, 4]).view()),
            Err(ShapeError::NotBroadcastable { .. })
        ));
        assert!(matches!(vecdot(&ones(&[2, 4]).view(), &ones(&[2, 5]).view()), Err(ShapeError::InvalidGufunc { .. })));
        let scalar = NdArray::from_vec(vec![3.0], &[]).unwrap();
        assert!(matches!(vecdot(&scalar.view(), &ones(&[4]).view()), Err(ShapeError::InvalidGufunc { .. })));
    }

    #[test]
    fn a_matmul_kernel_over_loop_dims_agrees_with_the_contraction_engine() {
        let (a, b) = (ar(12, &[2, 2, 3]), ar(6, &[3, 2]));
        let got = gufunc("(m,n),(n,p)->(m,p)", &[&a.view(), &b.view()], |ins, outs, dims| {
            let (m, n, p) = (dims[0], dims[1], dims[2]);
            for i in 0..m {
                for j in 0..p {
                    outs[0][i * p + j] = (0..n).map(|k| ins[0][i * n + k] * ins[1][k * p + j]).sum();
                }
            }
        })
        .unwrap();
        assert_eq!(got[0], matmul(&a.view(), &b.view()).unwrap());
    }

    #[test]
    fn several_outputs_and_loop_only_broadcasting() {
        let a = ar(6, &[2, 3]);
        let out = gufunc("(n)->(),()", &[&a.view()], |ins, outs, _| {
            outs[0][0] = ins[0].iter().copied().fold(f64::INFINITY, f64::min);
            outs[1][0] = ins[0].iter().copied().fold(f64::NEG_INFINITY, f64::max);
        })
        .unwrap();
        assert_eq!(out[0].as_slice(), &[0.0, 3.0]);
        assert_eq!(out[1].as_slice(), &[2.0, 5.0]);
        assert_eq!(out[0].shape(), &[2]);

        let doubled = gufunc("(n)->(n)", &[&a.view()], |ins, outs, _| {
            for (o, x) in outs[0].iter_mut().zip(ins[0]) {
                *o = 2.0 * x;
            }
        })
        .unwrap();
        assert_eq!(doubled[0].shape(), &[2, 3]);
        assert_eq!(doubled[0].as_slice(), &[0.0, 2.0, 4.0, 6.0, 8.0, 10.0]);
    }

    #[test]
    fn output_only_core_dims_and_wrong_operand_counts_are_rejected() {
        let a = ar(3, &[3]);
        let err = gufunc::<f64>("(n)->(m)", &[&a.view()], |_, _, _| {}).unwrap_err();
        assert!(matches!(err, ShapeError::InvalidGufunc { .. }));
        let err = gufunc::<f64>("(n),(n)->()", &[&a.view()], |_, _, _| {}).unwrap_err();
        assert!(matches!(err, ShapeError::InvalidGufunc { .. }));
    }
}
