use crate::dynarray::Arr;
use crate::pyarray::PyArray;
use crate::with_arr;

pub trait Show: Copy {
    fn show(self) -> String;
}

macro_rules! show_int {
    ($($t:ty),*) => {$(impl Show for $t { fn show(self) -> String { self.to_string() } })*};
}
show_int!(i8, i16, i32, i64, u8, u16, u32, u64);

impl Show for bool {
    fn show(self) -> String {
        if self { "True".into() } else { "False".into() }
    }
}

fn float_text(v: f64) -> String {
    if v.is_nan() {
        return "nan".into();
    }
    if v.is_infinite() {
        return if v > 0.0 { "inf".into() } else { "-inf".into() };
    }
    let a = v.abs();
    if a != 0.0 && !(1e-4..1e16).contains(&a) {
        let text = format!("{v:.8e}");
        let (mantissa, exp) = text.split_once('e').unwrap_or((&text, "0"));
        let mantissa = mantissa.trim_end_matches('0');
        let e: i32 = exp.parse().unwrap_or(0);
        return format!("{}{}e{}{:02}", mantissa, if mantissa.contains('.') { "" } else { "." }, if e < 0 { "-" } else { "+" }, e.abs());
    }
    let text = format!("{v:.8}");
    let trimmed = text.trim_end_matches('0');
    trimmed.to_string()
}

impl Show for f32 {
    fn show(self) -> String {
        float_text(f64::from(self))
    }
}

impl Show for f64 {
    fn show(self) -> String {
        float_text(self)
    }
}

impl Show for num_complex::Complex<f32> {
    fn show(self) -> String {
        complex_text(f64::from(self.re), f64::from(self.im))
    }
}

impl Show for num_complex::Complex<f64> {
    fn show(self) -> String {
        complex_text(self.re, self.im)
    }
}

fn complex_text(re: f64, im: f64) -> String {
    let sign = if im.is_sign_negative() && !im.is_nan() { "-" } else { "+" };
    format!("{}{}{}j", float_text(re), sign, float_text(im.abs()))
}

fn body(a: &PyArray, sep: &str, indent: usize) -> String {
    let positions = a.flat_positions();
    let texts: Vec<String> = with_arr!(a.storage.arr(), s => positions.iter().map(|&p| s.as_slice()[p].show()).collect());
    if a.shape.is_empty() {
        return texts[0].clone();
    }
    let width = texts.iter().map(String::len).max().unwrap_or(0);
    let summarize = positions.len() > 1000;
    let mut out = String::new();
    render(&a.shape, &texts, width, 0, 0, summarize, &mut out, sep, indent);
    out
}

#[allow(clippy::too_many_arguments)]
fn render(shape: &[usize], texts: &[String], width: usize, axis: usize, base: usize, summarize: bool, out: &mut String, sep: &str, indent: usize) {
    if axis == shape.len() {
        out.push_str(&format!("{:>width$}", texts[base]));
        return;
    }
    let stride: usize = shape[axis + 1..].iter().product();
    let n = shape[axis];
    let keep_all = !summarize || n <= 6;
    out.push('[');
    let indices: Vec<Option<usize>> = if keep_all {
        (0..n).map(Some).collect()
    } else {
        vec![Some(0), Some(1), Some(2), None, Some(n - 3), Some(n - 2), Some(n - 1)]
    };
    for (k, item) in indices.iter().enumerate() {
        if k > 0 {
            if axis + 1 == shape.len() {
                out.push_str(sep);
            } else {
                out.push_str(sep.trim_end());
                out.push_str(&"\n".repeat(shape.len() - axis - 1));
                out.push_str(&" ".repeat(indent + axis + 1));
            }
        }
        match item {
            Some(i) => render(shape, texts, width, axis + 1, base + i * stride, summarize, out, sep, indent),
            None => out.push_str("..."),
        }
    }
    out.push(']');
}

pub fn array_str(a: &PyArray) -> String {
    body(a, " ", 0)
}

pub fn array_repr(a: &PyArray) -> String {
    let name = a.dtype_name();
    let default = matches!(name, "float64" | "int64" | "bool" | "complex128");
    if a.shape.contains(&0) {
        let dims: Vec<String> = a.shape.iter().map(|d| d.to_string()).collect();
        let shape = if dims.len() == 1 { format!("({},)", dims[0]) } else { format!("({})", dims.join(", ")) };
        return if default { format!("array([], shape={shape})") } else { format!("array([], shape={shape}, dtype={name})") };
    }
    let inner = body(a, ", ", "array(".len());
    let dtype = if default { String::new() } else { format!(", dtype={name}") };
    format!("array({inner}{dtype})")
}
