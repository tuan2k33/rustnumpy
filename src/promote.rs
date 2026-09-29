use crate::dispatch::{WeakFloat, WeakInt};
use crate::dtype::DType;
use num_complex::Complex;

pub trait Promote<U>: DType {
    type Output: DType;
}

pub trait Widen<O>: Copy {
    fn widen(self) -> O;
}

pub trait PromoteWeak<S>: DType {
    type Output: DType;
}

macro_rules! promote {
    ($a:ty, $b:ty => $o:ty) => {
        impl Promote<$b> for $a {
            type Output = $o;
        }
    };
}

macro_rules! promote_weak {
    ($a:ty, $s:ty => $o:ty) => {
        impl PromoteWeak<$s> for $a {
            type Output = $o;
        }
    };
}

macro_rules! widen_as {
    ($a:ty => $o:ty) => {
        impl Widen<$o> for $a {
            fn widen(self) -> $o {
                self as $o
            }
        }
    };
}

macro_rules! widen_bool {
    ($o:ty) => {
        impl Widen<$o> for bool {
            fn widen(self) -> $o {
                self as u8 as $o
            }
        }
    };
}

macro_rules! widen_prim {
    ($a:ty => $o:ty) => {
        impl Widen<$o> for $a {
            fn widen(self) -> $o {
                num_traits::AsPrimitive::<$o>::as_(self)
            }
        }
    };
}

macro_rules! widen_prim_complex {
    ($a:ty => $f:ty) => {
        impl Widen<Complex<$f>> for $a {
            fn widen(self) -> Complex<$f> {
                Complex::new(num_traits::AsPrimitive::<$f>::as_(self), 0.0)
            }
        }
    };
}

macro_rules! widen_bool_half {
    () => {
        impl Widen<half::f16> for bool {
            fn widen(self) -> half::f16 {
                half::f16::from_f32(f32::from(self as u8))
            }
        }
    };
}

macro_rules! widen_bool_complex {
    ($f:ty) => {
        impl Widen<Complex<$f>> for bool {
            fn widen(self) -> Complex<$f> {
                Complex::new(self as u8 as $f, 0.0)
            }
        }
    };
}

macro_rules! widen_real_complex {
    ($a:ty => $f:ty) => {
        impl Widen<Complex<$f>> for $a {
            fn widen(self) -> Complex<$f> {
                Complex::new(self as $f, 0.0)
            }
        }
    };
}

macro_rules! widen_complex {
    ($a:ty => $f:ty) => {
        impl Widen<Complex<$f>> for Complex<$a> {
            fn widen(self) -> Complex<$f> {
                Complex::new(self.re as $f, self.im as $f)
            }
        }
    };
}

promote!(bool, bool => bool);
promote!(bool, i8 => i8);
promote!(bool, i16 => i16);
promote!(bool, i32 => i32);
promote!(bool, i64 => i64);
promote!(bool, u8 => u8);
promote!(bool, u16 => u16);
promote!(bool, u32 => u32);
promote!(bool, u64 => u64);
promote!(bool, half::f16 => half::f16);
promote!(bool, f32 => f32);
promote!(bool, f64 => f64);
promote!(bool, Complex<f32> => Complex<f32>);
promote!(bool, Complex<f64> => Complex<f64>);
promote!(i8, bool => i8);
promote!(i8, i8 => i8);
promote!(i8, i16 => i16);
promote!(i8, i32 => i32);
promote!(i8, i64 => i64);
promote!(i8, u8 => i16);
promote!(i8, u16 => i32);
promote!(i8, u32 => i64);
promote!(i8, u64 => f64);
promote!(i8, half::f16 => half::f16);
promote!(i8, f32 => f32);
promote!(i8, f64 => f64);
promote!(i8, Complex<f32> => Complex<f32>);
promote!(i8, Complex<f64> => Complex<f64>);
promote!(i16, bool => i16);
promote!(i16, i8 => i16);
promote!(i16, i16 => i16);
promote!(i16, i32 => i32);
promote!(i16, i64 => i64);
promote!(i16, u8 => i16);
promote!(i16, u16 => i32);
promote!(i16, u32 => i64);
promote!(i16, u64 => f64);
promote!(i16, half::f16 => f32);
promote!(i16, f32 => f32);
promote!(i16, f64 => f64);
promote!(i16, Complex<f32> => Complex<f32>);
promote!(i16, Complex<f64> => Complex<f64>);
promote!(i32, bool => i32);
promote!(i32, i8 => i32);
promote!(i32, i16 => i32);
promote!(i32, i32 => i32);
promote!(i32, i64 => i64);
promote!(i32, u8 => i32);
promote!(i32, u16 => i32);
promote!(i32, u32 => i64);
promote!(i32, u64 => f64);
promote!(i32, half::f16 => f64);
promote!(i32, f32 => f64);
promote!(i32, f64 => f64);
promote!(i32, Complex<f32> => Complex<f64>);
promote!(i32, Complex<f64> => Complex<f64>);
promote!(i64, bool => i64);
promote!(i64, i8 => i64);
promote!(i64, i16 => i64);
promote!(i64, i32 => i64);
promote!(i64, i64 => i64);
promote!(i64, u8 => i64);
promote!(i64, u16 => i64);
promote!(i64, u32 => i64);
promote!(i64, u64 => f64);
promote!(i64, half::f16 => f64);
promote!(i64, f32 => f64);
promote!(i64, f64 => f64);
promote!(i64, Complex<f32> => Complex<f64>);
promote!(i64, Complex<f64> => Complex<f64>);
promote!(u8, bool => u8);
promote!(u8, i8 => i16);
promote!(u8, i16 => i16);
promote!(u8, i32 => i32);
promote!(u8, i64 => i64);
promote!(u8, u8 => u8);
promote!(u8, u16 => u16);
promote!(u8, u32 => u32);
promote!(u8, u64 => u64);
promote!(u8, half::f16 => half::f16);
promote!(u8, f32 => f32);
promote!(u8, f64 => f64);
promote!(u8, Complex<f32> => Complex<f32>);
promote!(u8, Complex<f64> => Complex<f64>);
promote!(u16, bool => u16);
promote!(u16, i8 => i32);
promote!(u16, i16 => i32);
promote!(u16, i32 => i32);
promote!(u16, i64 => i64);
promote!(u16, u8 => u16);
promote!(u16, u16 => u16);
promote!(u16, u32 => u32);
promote!(u16, u64 => u64);
promote!(u16, half::f16 => f32);
promote!(u16, f32 => f32);
promote!(u16, f64 => f64);
promote!(u16, Complex<f32> => Complex<f32>);
promote!(u16, Complex<f64> => Complex<f64>);
promote!(u32, bool => u32);
promote!(u32, i8 => i64);
promote!(u32, i16 => i64);
promote!(u32, i32 => i64);
promote!(u32, i64 => i64);
promote!(u32, u8 => u32);
promote!(u32, u16 => u32);
promote!(u32, u32 => u32);
promote!(u32, u64 => u64);
promote!(u32, half::f16 => f64);
promote!(u32, f32 => f64);
promote!(u32, f64 => f64);
promote!(u32, Complex<f32> => Complex<f64>);
promote!(u32, Complex<f64> => Complex<f64>);
promote!(u64, bool => u64);
promote!(u64, i8 => f64);
promote!(u64, i16 => f64);
promote!(u64, i32 => f64);
promote!(u64, i64 => f64);
promote!(u64, u8 => u64);
promote!(u64, u16 => u64);
promote!(u64, u32 => u64);
promote!(u64, u64 => u64);
promote!(u64, half::f16 => f64);
promote!(u64, f32 => f64);
promote!(u64, f64 => f64);
promote!(u64, Complex<f32> => Complex<f64>);
promote!(u64, Complex<f64> => Complex<f64>);
promote!(half::f16, bool => half::f16);
promote!(half::f16, i8 => half::f16);
promote!(half::f16, i16 => f32);
promote!(half::f16, i32 => f64);
promote!(half::f16, i64 => f64);
promote!(half::f16, u8 => half::f16);
promote!(half::f16, u16 => f32);
promote!(half::f16, u32 => f64);
promote!(half::f16, u64 => f64);
promote!(half::f16, half::f16 => half::f16);
promote!(half::f16, f32 => f32);
promote!(half::f16, f64 => f64);
promote!(half::f16, Complex<f32> => Complex<f32>);
promote!(half::f16, Complex<f64> => Complex<f64>);
promote!(f32, bool => f32);
promote!(f32, i8 => f32);
promote!(f32, i16 => f32);
promote!(f32, i32 => f64);
promote!(f32, i64 => f64);
promote!(f32, u8 => f32);
promote!(f32, u16 => f32);
promote!(f32, u32 => f64);
promote!(f32, u64 => f64);
promote!(f32, half::f16 => f32);
promote!(f32, f32 => f32);
promote!(f32, f64 => f64);
promote!(f32, Complex<f32> => Complex<f32>);
promote!(f32, Complex<f64> => Complex<f64>);
promote!(f64, bool => f64);
promote!(f64, i8 => f64);
promote!(f64, i16 => f64);
promote!(f64, i32 => f64);
promote!(f64, i64 => f64);
promote!(f64, u8 => f64);
promote!(f64, u16 => f64);
promote!(f64, u32 => f64);
promote!(f64, u64 => f64);
promote!(f64, half::f16 => f64);
promote!(f64, f32 => f64);
promote!(f64, f64 => f64);
promote!(f64, Complex<f32> => Complex<f64>);
promote!(f64, Complex<f64> => Complex<f64>);
promote!(Complex<f32>, bool => Complex<f32>);
promote!(Complex<f32>, i8 => Complex<f32>);
promote!(Complex<f32>, i16 => Complex<f32>);
promote!(Complex<f32>, i32 => Complex<f64>);
promote!(Complex<f32>, i64 => Complex<f64>);
promote!(Complex<f32>, u8 => Complex<f32>);
promote!(Complex<f32>, u16 => Complex<f32>);
promote!(Complex<f32>, u32 => Complex<f64>);
promote!(Complex<f32>, u64 => Complex<f64>);
promote!(Complex<f32>, half::f16 => Complex<f32>);
promote!(Complex<f32>, f32 => Complex<f32>);
promote!(Complex<f32>, f64 => Complex<f64>);
promote!(Complex<f32>, Complex<f32> => Complex<f32>);
promote!(Complex<f32>, Complex<f64> => Complex<f64>);
promote!(Complex<f64>, bool => Complex<f64>);
promote!(Complex<f64>, i8 => Complex<f64>);
promote!(Complex<f64>, i16 => Complex<f64>);
promote!(Complex<f64>, i32 => Complex<f64>);
promote!(Complex<f64>, i64 => Complex<f64>);
promote!(Complex<f64>, u8 => Complex<f64>);
promote!(Complex<f64>, u16 => Complex<f64>);
promote!(Complex<f64>, u32 => Complex<f64>);
promote!(Complex<f64>, u64 => Complex<f64>);
promote!(Complex<f64>, half::f16 => Complex<f64>);
promote!(Complex<f64>, f32 => Complex<f64>);
promote!(Complex<f64>, f64 => Complex<f64>);
promote!(Complex<f64>, Complex<f32> => Complex<f64>);
promote!(Complex<f64>, Complex<f64> => Complex<f64>);

promote_weak!(bool, WeakInt => i64);
promote_weak!(bool, WeakFloat => f64);
promote_weak!(i8, WeakInt => i8);
promote_weak!(i8, WeakFloat => f64);
promote_weak!(i16, WeakInt => i16);
promote_weak!(i16, WeakFloat => f64);
promote_weak!(i32, WeakInt => i32);
promote_weak!(i32, WeakFloat => f64);
promote_weak!(i64, WeakInt => i64);
promote_weak!(i64, WeakFloat => f64);
promote_weak!(u8, WeakInt => u8);
promote_weak!(u8, WeakFloat => f64);
promote_weak!(u16, WeakInt => u16);
promote_weak!(u16, WeakFloat => f64);
promote_weak!(u32, WeakInt => u32);
promote_weak!(u32, WeakFloat => f64);
promote_weak!(u64, WeakInt => u64);
promote_weak!(u64, WeakFloat => f64);
promote_weak!(half::f16, WeakInt => half::f16);
promote_weak!(half::f16, WeakFloat => half::f16);
promote_weak!(f32, WeakInt => f32);
promote_weak!(f32, WeakFloat => f32);
promote_weak!(f64, WeakInt => f64);
promote_weak!(f64, WeakFloat => f64);
promote_weak!(Complex<f32>, WeakInt => Complex<f32>);
promote_weak!(Complex<f32>, WeakFloat => Complex<f32>);
promote_weak!(Complex<f64>, WeakInt => Complex<f64>);
promote_weak!(Complex<f64>, WeakFloat => Complex<f64>);

impl Widen<bool> for bool {
    fn widen(self) -> bool {
        self
    }
}
widen_bool!(i8);
widen_bool!(i16);
widen_bool!(i32);
widen_bool!(i64);
widen_bool!(u8);
widen_bool!(u16);
widen_bool!(u32);
widen_bool!(u64);
widen_bool_half!();
widen_bool!(f32);
widen_bool!(f64);
widen_bool_complex!(f32);
widen_bool_complex!(f64);
impl Widen<i8> for i8 {
    fn widen(self) -> i8 {
        self
    }
}
widen_as!(i8 => i16);
widen_as!(i8 => i32);
widen_as!(i8 => i64);
widen_prim!(i8 => half::f16);
widen_as!(i8 => f32);
widen_as!(i8 => f64);
widen_real_complex!(i8 => f32);
widen_real_complex!(i8 => f64);
widen_as!(i16 => i8);
impl Widen<i16> for i16 {
    fn widen(self) -> i16 {
        self
    }
}
widen_as!(i16 => i32);
widen_as!(i16 => i64);
widen_prim!(i16 => half::f16);
widen_as!(i16 => f32);
widen_as!(i16 => f64);
widen_real_complex!(i16 => f32);
widen_real_complex!(i16 => f64);
widen_as!(i32 => i8);
widen_as!(i32 => i16);
impl Widen<i32> for i32 {
    fn widen(self) -> i32 {
        self
    }
}
widen_as!(i32 => i64);
widen_prim!(i32 => half::f16);
widen_as!(i32 => f32);
widen_as!(i32 => f64);
widen_real_complex!(i32 => f32);
widen_real_complex!(i32 => f64);
widen_as!(i64 => i8);
widen_as!(i64 => i16);
widen_as!(i64 => i32);
impl Widen<i64> for i64 {
    fn widen(self) -> i64 {
        self
    }
}
widen_prim!(i64 => half::f16);
widen_as!(i64 => f32);
widen_as!(i64 => f64);
widen_real_complex!(i64 => f32);
widen_real_complex!(i64 => f64);
widen_as!(u8 => i8);
widen_as!(u8 => i16);
widen_as!(u8 => i32);
widen_as!(u8 => i64);
impl Widen<u8> for u8 {
    fn widen(self) -> u8 {
        self
    }
}
widen_as!(u8 => u16);
widen_as!(u8 => u32);
widen_as!(u8 => u64);
widen_prim!(u8 => half::f16);
widen_as!(u8 => f32);
widen_as!(u8 => f64);
widen_real_complex!(u8 => f32);
widen_real_complex!(u8 => f64);
widen_as!(u16 => i8);
widen_as!(u16 => i16);
widen_as!(u16 => i32);
widen_as!(u16 => i64);
widen_as!(u16 => u8);
impl Widen<u16> for u16 {
    fn widen(self) -> u16 {
        self
    }
}
widen_as!(u16 => u32);
widen_as!(u16 => u64);
widen_prim!(u16 => half::f16);
widen_as!(u16 => f32);
widen_as!(u16 => f64);
widen_real_complex!(u16 => f32);
widen_real_complex!(u16 => f64);
widen_as!(u32 => i8);
widen_as!(u32 => i16);
widen_as!(u32 => i32);
widen_as!(u32 => i64);
widen_as!(u32 => u8);
widen_as!(u32 => u16);
impl Widen<u32> for u32 {
    fn widen(self) -> u32 {
        self
    }
}
widen_as!(u32 => u64);
widen_prim!(u32 => half::f16);
widen_as!(u32 => f32);
widen_as!(u32 => f64);
widen_real_complex!(u32 => f32);
widen_real_complex!(u32 => f64);
widen_as!(u64 => i8);
widen_as!(u64 => i16);
widen_as!(u64 => i32);
widen_as!(u64 => i64);
widen_as!(u64 => u8);
widen_as!(u64 => u16);
widen_as!(u64 => u32);
impl Widen<u64> for u64 {
    fn widen(self) -> u64 {
        self
    }
}
widen_prim!(u64 => half::f16);
widen_as!(u64 => f32);
widen_as!(u64 => f64);
widen_real_complex!(u64 => f32);
widen_real_complex!(u64 => f64);
impl Widen<half::f16> for half::f16 {
    fn widen(self) -> half::f16 {
        self
    }
}
widen_prim!(half::f16 => f32);
widen_prim!(half::f16 => f64);
widen_prim_complex!(half::f16 => f32);
widen_prim_complex!(half::f16 => f64);
widen_prim!(f32 => half::f16);
impl Widen<f32> for f32 {
    fn widen(self) -> f32 {
        self
    }
}
widen_as!(f32 => f64);
widen_real_complex!(f32 => f32);
widen_real_complex!(f32 => f64);
widen_prim!(f64 => half::f16);
widen_as!(f64 => f32);
impl Widen<f64> for f64 {
    fn widen(self) -> f64 {
        self
    }
}
widen_real_complex!(f64 => f32);
widen_real_complex!(f64 => f64);
impl Widen<Complex<f32>> for Complex<f32> {
    fn widen(self) -> Complex<f32> {
        self
    }
}
widen_complex!(f32 => f64);
widen_complex!(f64 => f32);
impl Widen<Complex<f64>> for Complex<f64> {
    fn widen(self) -> Complex<f64> {
        self
    }
}
