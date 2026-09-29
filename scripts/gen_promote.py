import numpy as np, sys

NAMES = ['bool', 'int8', 'int16', 'int32', 'int64', 'uint8', 'uint16', 'uint32', 'uint64',
         'float32', 'float64', 'complex64', 'complex128']
RUST = {'bool': 'bool', 'int8': 'i8', 'int16': 'i16', 'int32': 'i32', 'int64': 'i64',
        'uint8': 'u8', 'uint16': 'u16', 'uint32': 'u32', 'uint64': 'u64',
        'float32': 'f32', 'float64': 'f64', 'complex64': 'Complex<f32>', 'complex128': 'Complex<f64>'}


def kind(n):
    return 'bool' if n == 'bool' else 'complex' if n.startswith('complex') else 'float' if n.startswith('float') else 'int'


def comp_float(n):
    return 'f32' if n == 'complex64' else 'f64'


out = ["use crate::dispatch::{WeakFloat, WeakInt};", "use crate::dtype::DType;", "use num_complex::Complex;", ""]
out.append("pub trait Promote<U>: DType {\n    type Output: DType;\n}\n")
out.append("pub trait Widen<O>: Copy {\n    fn widen(self) -> O;\n}\n")
out.append("pub trait PromoteWeak<S>: DType {\n    type Output: DType;\n}\n")
out.append("""macro_rules! promote {
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
""")

for a in NAMES:
    for b in NAMES:
        r = np.result_type(a, b).name
        out.append(f"promote!({RUST[a]}, {RUST[b]} => {RUST[r]});")
out.append("")
for a in NAMES:
    for s, pyv in (('WeakInt', 1), ('WeakFloat', 1.0)):
        r = (np.array([1], dtype=a) + pyv).dtype.name
        out.append(f"promote_weak!({RUST[a]}, {s} => {RUST[r]});")
out.append("")
for src in NAMES:
    for dst in NAMES:
        if not np.can_cast(src, dst, 'same_kind'):
            continue
        ks, kd = kind(src), kind(dst)
        if src == dst:
            out.append(f"impl Widen<{RUST[dst]}> for {RUST[src]} {{\n    fn widen(self) -> {RUST[dst]} {{\n        self\n    }}\n}}")
        elif ks == 'bool' and kd == 'complex':
            out.append(f"widen_bool_complex!({comp_float(dst)});")
        elif ks == 'bool':
            out.append(f"widen_bool!({RUST[dst]});")
        elif kd == 'complex' and ks == 'complex':
            out.append(f"widen_complex!({comp_float(src)} => {comp_float(dst)});")
        elif kd == 'complex':
            out.append(f"widen_real_complex!({RUST[src]} => {comp_float(dst)});")
        else:
            out.append(f"widen_as!({RUST[src]} => {RUST[dst]});")
open(sys.argv[1], 'w').write("\n".join(out) + "\n")
