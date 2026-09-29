use crate::dynarray::{Arr, C32, C64};
use crate::{unsupported, with_arr};
use num_complex::Complex;
use pyo3::prelude::*;
use rustnumpy::NdArray;

pub trait Cast<U>: Copy {
    fn cast(self) -> U;
}

macro_rules! cast_row {
    ($s:ty; $($t:ty),*) => {$(
        impl Cast<$t> for $s {
            fn cast(self) -> $t { self as $t }
        }
    )*};
}

macro_rules! cast_table {
    ($($s:ty),*) => {$( cast_row!($s; i8, i16, i32, i64, u8, u16, u32, u64, f32, f64); )*};
}
cast_table!(i8, i16, i32, i64, u8, u16, u32, u64, f32, f64);

macro_rules! cast_bool_and_complex {
    ($($t:ty),*) => {$(
        impl Cast<$t> for bool {
            fn cast(self) -> $t { self as u8 as $t }
        }
        impl Cast<bool> for $t {
            fn cast(self) -> bool { self != (0 as $t) }
        }
        impl Cast<C32> for $t {
            fn cast(self) -> C32 { Complex::new(self as f32, 0.0) }
        }
        impl Cast<C64> for $t {
            fn cast(self) -> C64 { Complex::new(self as f64, 0.0) }
        }
        impl Cast<$t> for C32 {
            fn cast(self) -> $t { self.re as $t }
        }
        impl Cast<$t> for C64 {
            fn cast(self) -> $t { self.re as $t }
        }
    )*};
}
cast_bool_and_complex!(i8, i16, i32, i64, u8, u16, u32, u64, f32, f64);

impl Cast<bool> for bool {
    fn cast(self) -> bool {
        self
    }
}
impl Cast<C32> for bool {
    fn cast(self) -> C32 {
        Complex::new(f32::from(u8::from(self)), 0.0)
    }
}
impl Cast<C64> for bool {
    fn cast(self) -> C64 {
        Complex::new(f64::from(u8::from(self)), 0.0)
    }
}
impl Cast<bool> for C32 {
    fn cast(self) -> bool {
        self.re != 0.0 || self.im != 0.0
    }
}
impl Cast<bool> for C64 {
    fn cast(self) -> bool {
        self.re != 0.0 || self.im != 0.0
    }
}
impl Cast<C32> for C32 {
    fn cast(self) -> C32 {
        self
    }
}
impl Cast<C64> for C64 {
    fn cast(self) -> C64 {
        self
    }
}
impl Cast<C64> for C32 {
    fn cast(self) -> C64 {
        Complex::new(f64::from(self.re), f64::from(self.im))
    }
}
impl Cast<C32> for C64 {
    fn cast(self) -> C32 {
        Complex::new(self.re as f32, self.im as f32)
    }
}

fn map_cast<S: Cast<T>, T: Copy>(a: &NdArray<S>) -> NdArray<T> {
    let data: Vec<T> = a.as_slice().iter().map(|&x| x.cast()).collect();
    NdArray::from_vec(data, a.shape()).expect("same element count")
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
        "float32" => with_arr!(src, a => Arr::from(map_cast::<_, f32>(a))),
        "float64" => with_arr!(src, a => Arr::from(map_cast::<_, f64>(a))),
        "complex64" => with_arr!(src, a => Arr::from(map_cast::<_, C32>(a))),
        "complex128" => with_arr!(src, a => Arr::from(map_cast::<_, C64>(a))),
        other => return Err(unsupported(format!("cannot cast to {other}"))),
    })
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
