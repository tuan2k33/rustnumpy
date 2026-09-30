use crate::dynarray::{Arr, C32, C64};
use crate::{unsupported, with_arr};
use pyo3::prelude::*;
use half::f16;
use rustnumpy::{Cast, NdArray};

fn map_cast<S: Cast<T>, T: Copy>(a: &NdArray<S>) -> NdArray<T> {
    a.astype()
}

pub fn astype(src: &Arr, target: &str) -> PyResult<Arr> {
    Ok(match target {
        "bool" => with_arr!(src, a => Arr::from(map_cast::<_, bool>(a))),
        "int8" => with_arr!(src, a => Arr::from(map_cast::<_, i8>(a))),
        "int16" => with_arr!(src, a => Arr::from(map_cast::<_, i16>(a))),
        "int32" => with_arr!(src, a => Arr::from(map_cast::<_, i32>(a))),
        "int64" => with_arr!(src, a => Arr::from(map_cast::<_, i64>(a))),
        "uint8" => with_arr!(src, a => Arr::from(map_cast::<_, u8>(a))),
        "uint16" => with_arr!(src, a => Arr::from(map_cast::<_, u16>(a))),
        "uint32" => with_arr!(src, a => Arr::from(map_cast::<_, u32>(a))),
        "uint64" => with_arr!(src, a => Arr::from(map_cast::<_, u64>(a))),
        "float16" => with_arr!(src, a => Arr::from(map_cast::<_, f16>(a))),
        "float32" => with_arr!(src, a => Arr::from(map_cast::<_, f32>(a))),
        "float64" => with_arr!(src, a => Arr::from(map_cast::<_, f64>(a))),
        "complex64" => with_arr!(src, a => Arr::from(map_cast::<_, C32>(a))),
        "complex128" => with_arr!(src, a => Arr::from(map_cast::<_, C64>(a))),
        other => return Err(unsupported(format!("cannot cast to {other}"))),
    })
}

pub fn cast_ref<'a>(src: &'a Arr, target: &str, slot: &'a mut Option<Arr>) -> PyResult<&'a Arr> {
    if src.dtype_name() == target {
        return Ok(src);
    }
    Ok(slot.insert(astype(src, target)?))
}

pub fn kind_name(k: rustnumpy::Kind) -> &'static str {
    use rustnumpy::Kind::*;
    match k {
        Bool => "bool",
        Int(8) => "int8",
        Int(16) => "int16",
        Int(32) => "int32",
        Int(_) => "int64",
        Uint(8) => "uint8",
        Uint(16) => "uint16",
        Uint(32) => "uint32",
        Uint(_) => "uint64",
        Float(16) => "float16",
        Float(32) => "float32",
        Float(_) => "float64",
        Complex(32) => "complex64",
        Complex(_) => "complex128",
    }
}

pub fn common_arrs(a: &Arr, b: &Arr) -> PyResult<(Arr, Arr)> {
    let name = kind_name(rustnumpy::common_dtype(a.kind(), b.kind()));
    Ok((astype(a, name)?, astype(b, name)?))
}
