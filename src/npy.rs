use std::fmt;
use std::fs::File;
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::path::Path;

use crate::dtype::DType;
use crate::error::ShapeError;
use crate::ndarray::NdArray;

const MAGIC: &[u8; 6] = b"\x93NUMPY";

const ALIGN: usize = 64;

#[derive(Debug)]
pub enum NpyError {
    Io(io::Error),
    BadMagic([u8; 6]),
    UnsupportedVersion { major: u8, minor: u8 },
    HeaderParse(String),
    UnsupportedDtype(String),
    Shape(ShapeError),
}

impl fmt::Display for NpyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            NpyError::Io(e) => write!(f, "I/O error: {e}"),
            NpyError::BadMagic(got) => write!(f, "not a .npy file: bad magic bytes {got:?}"),
            NpyError::UnsupportedVersion { major, minor } => {
                write!(f, "unsupported .npy version {major}.{minor}")
            }
            NpyError::HeaderParse(msg) => write!(f, "could not parse .npy header: {msg}"),
            NpyError::UnsupportedDtype(descr) => {
                write!(f, "dtype {descr:?} does not match the requested element type (or is not a plain numeric dtype)")
            }
            NpyError::Shape(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for NpyError {}

impl From<io::Error> for NpyError {
    fn from(e: io::Error) -> Self {
        NpyError::Io(e)
    }
}

impl From<ShapeError> for NpyError {
    fn from(e: ShapeError) -> Self {
        NpyError::Shape(e)
    }
}

pub trait NpyElement: DType {
    const DESCR: &'static str;
    const SIZE: usize;
    const PART: usize;
    fn put_le(self, out: &mut Vec<u8>);
    fn take_le(bytes: &[u8]) -> Self;
}

macro_rules! npy_numeric {
    ($($t:ty => $descr:expr),* $(,)?) => {$(
        impl NpyElement for $t {
            const DESCR: &'static str = $descr;
            const SIZE: usize = std::mem::size_of::<$t>();
            const PART: usize = std::mem::size_of::<$t>();
            fn put_le(self, out: &mut Vec<u8>) {
                out.extend_from_slice(&self.to_le_bytes());
            }
            fn take_le(bytes: &[u8]) -> Self {
                <$t>::from_le_bytes(bytes.try_into().expect("caller passes exactly SIZE bytes"))
            }
        }
    )*};
}
npy_numeric!(i8 => "|i1", u8 => "|u1", i16 => "<i2", u16 => "<u2", i32 => "<i4", u32 => "<u4", i64 => "<i8", u64 => "<u8",
    half::f16 => "<f2", f32 => "<f4", f64 => "<f8");

impl NpyElement for bool {
    const DESCR: &'static str = "|b1";
    const SIZE: usize = 1;
    const PART: usize = 1;
    fn put_le(self, out: &mut Vec<u8>) {
        out.push(u8::from(self));
    }
    fn take_le(bytes: &[u8]) -> Self {
        bytes[0] != 0
    }
}

macro_rules! npy_complex {
    ($($t:ty => $descr:expr),* $(,)?) => {$(
        impl NpyElement for num_complex::Complex<$t> {
            const DESCR: &'static str = $descr;
            const SIZE: usize = 2 * std::mem::size_of::<$t>();
            const PART: usize = std::mem::size_of::<$t>();
            fn put_le(self, out: &mut Vec<u8>) {
                self.re.put_le(out);
                self.im.put_le(out);
            }
            fn take_le(bytes: &[u8]) -> Self {
                let half = std::mem::size_of::<$t>();
                num_complex::Complex::new(<$t>::take_le(&bytes[..half]), <$t>::take_le(&bytes[half..]))
            }
        }
    )*};
}
npy_complex!(f32 => "<c8", f64 => "<c16");

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NpyHeader {
    pub descr: String,
    pub fortran_order: bool,
    pub shape: Vec<usize>,
}

pub fn save_npy<T: NpyElement, P: AsRef<Path>>(path: P, arr: &NdArray<T>) -> Result<(), NpyError> {
    let file = File::create(path)?;
    write_npy(BufWriter::new(file), arr)
}

pub fn load_npy<T: NpyElement, P: AsRef<Path>>(path: P) -> Result<NdArray<T>, NpyError> {
    let file = File::open(path)?;
    read_npy(BufReader::new(file))
}

pub fn write_npy<T: NpyElement, W: Write>(mut w: W, arr: &NdArray<T>) -> Result<(), NpyError> {
    let header_dict = format!(
        "{{'descr': '{}', 'fortran_order': False, 'shape': {}, }}",
        T::DESCR,
        format_shape_tuple(arr.shape())
    );

    const PREFIX_LEN: usize = 10;
    let unpadded_len = header_dict.len() + 1;
    let total_before_pad = PREFIX_LEN + unpadded_len;
    let pad = (ALIGN - total_before_pad % ALIGN) % ALIGN;

    let mut header = header_dict.into_bytes();
    header.resize(header.len() + pad, b' ');
    header.push(b'\n');

    let header_len: u16 = header
        .len()
        .try_into()
        .expect("header too long for the u16 field (shape with too many dimensions?)");

    w.write_all(MAGIC)?;
    w.write_all(&[1, 0])?;
    w.write_all(&header_len.to_le_bytes())?;
    w.write_all(&header)?;

    let mut bytes = Vec::with_capacity(arr.len() * T::SIZE);
    for &value in arr.as_slice() {
        value.put_le(&mut bytes);
    }
    w.write_all(&bytes)?;
    Ok(())
}

pub fn read_header<R: Read>(r: &mut R) -> Result<NpyHeader, NpyError> {
    let mut magic = [0u8; 6];
    r.read_exact(&mut magic)?;
    if &magic != MAGIC {
        return Err(NpyError::BadMagic(magic));
    }

    let mut version = [0u8; 2];
    r.read_exact(&mut version)?;
    let (major, minor) = (version[0], version[1]);

    let header_len: usize = match major {
        1 => {
            let mut buf = [0u8; 2];
            r.read_exact(&mut buf)?;
            u16::from_le_bytes(buf) as usize
        }
        2 | 3 => {
            let mut buf = [0u8; 4];
            r.read_exact(&mut buf)?;
            u32::from_le_bytes(buf) as usize
        }
        _ => return Err(NpyError::UnsupportedVersion { major, minor }),
    };

    let mut header_bytes = vec![0u8; header_len];
    r.read_exact(&mut header_bytes)?;
    let header = String::from_utf8_lossy(&header_bytes);

    Ok(NpyHeader {
        descr: extract_str_field(&header, "descr")?,
        fortran_order: extract_bool_field(&header, "fortran_order")?,
        shape: extract_shape_field(&header)?,
    })
}

pub fn read_npy<T: NpyElement, R: Read>(mut r: R) -> Result<NdArray<T>, NpyError> {
    let header = read_header(&mut r)?;

    let mut chars = header.descr.chars();
    let order = chars.next().unwrap_or(' ');
    let big_endian = match order {
        '>' => T::SIZE > 1,
        '<' | '=' | '|' => false,
        _ => return Err(NpyError::UnsupportedDtype(header.descr)),
    };
    if !T::DESCR.ends_with(chars.as_str()) {
        return Err(NpyError::UnsupportedDtype(header.descr));
    }

    let len: usize = header.shape.iter().product();
    let mut raw = vec![0u8; len * T::SIZE];
    r.read_exact(&mut raw)?;

    let data: Vec<T> = raw
        .chunks_exact(T::SIZE.max(1))
        .map(|chunk| {
            if big_endian {
                let mut swapped = chunk.to_vec();
                swapped.chunks_exact_mut(T::PART).for_each(<[u8]>::reverse);
                T::take_le(&swapped)
            } else {
                T::take_le(chunk)
            }
        })
        .collect();

    if header.fortran_order && header.shape.len() > 1 {
        let reversed: Vec<usize> = header.shape.iter().rev().copied().collect();
        let stored = NdArray::from_vec(data, &reversed)?;
        return Ok(stored.view().transpose().to_owned());
    }
    Ok(NdArray::from_vec(data, &header.shape)?)
}

fn format_shape_tuple(shape: &[usize]) -> String {
    match shape.len() {
        0 => "()".to_string(),
        1 => format!("({},)", shape[0]),
        _ => {
            let inner = shape.iter().map(|d| d.to_string()).collect::<Vec<_>>().join(", ");
            format!("({inner})")
        }
    }
}

fn value_after_key<'a>(header: &'a str, key: &str) -> Result<&'a str, NpyError> {
    let needle = format!("'{key}'");
    let key_pos = header
        .find(&needle)
        .ok_or_else(|| NpyError::HeaderParse(format!("missing key {key:?} in header: {header:?}")))?;
    let after_key = &header[key_pos + needle.len()..];
    let colon = after_key
        .find(':')
        .ok_or_else(|| NpyError::HeaderParse(format!("missing ':' after key {key:?}")))?;
    Ok(after_key[colon + 1..].trim_start())
}

fn extract_str_field(header: &str, key: &str) -> Result<String, NpyError> {
    let value = value_after_key(header, key)?;
    let quote = value.chars().next().ok_or_else(|| {
        NpyError::HeaderParse(format!("expected quoted string value for {key:?}"))
    })?;
    if quote != '\'' && quote != '"' {
        return Err(NpyError::HeaderParse(format!(
            "expected quoted string value for {key:?}, got {value:?}"
        )));
    }
    let rest = &value[quote.len_utf8()..];
    let end = rest
        .find(quote)
        .ok_or_else(|| NpyError::HeaderParse(format!("unterminated string value for {key:?}")))?;
    Ok(rest[..end].to_string())
}

fn extract_bool_field(header: &str, key: &str) -> Result<bool, NpyError> {
    let value = value_after_key(header, key)?;
    if value.starts_with("True") {
        Ok(true)
    } else if value.starts_with("False") {
        Ok(false)
    } else {
        Err(NpyError::HeaderParse(format!(
            "expected True/False for {key:?}, got {value:?}"
        )))
    }
}

fn extract_shape_field(header: &str) -> Result<Vec<usize>, NpyError> {
    let value = value_after_key(header, "shape")?;
    let open = value
        .find('(')
        .ok_or_else(|| NpyError::HeaderParse("shape value missing '('".to_string()))?;
    let close = value
        .find(')')
        .ok_or_else(|| NpyError::HeaderParse("shape value missing ')'".to_string()))?;
    let inner = &value[open + 1..close];
    inner
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(|s| {
            s.parse::<usize>()
                .map_err(|e| NpyError::HeaderParse(format!("bad shape dimension {s:?}: {e}")))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn roundtrip(arr: &NdArray) -> NdArray {
        let mut buf = Vec::new();
        write_npy(&mut buf, arr).unwrap();
        read_npy(&buf[..]).unwrap()
    }

    #[test]
    fn roundtrip_preserves_shape_and_values() {
        let a = NdArray::from_vec((0..12).map(|i| i as f64).collect(), &[3, 4]).unwrap();
        let back = roundtrip(&a);
        assert_eq!(back.shape(), a.shape());
        assert_eq!(back.as_slice(), a.as_slice());
    }

    #[test]
    fn roundtrip_handles_0d_and_1d() {
        let scalar = NdArray::from_vec(vec![42.0], &[]).unwrap();
        assert_eq!(roundtrip(&scalar).shape(), &[] as &[usize]);

        let vector = NdArray::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap();
        assert_eq!(roundtrip(&vector).shape(), &[3]);
    }

    #[test]
    fn header_is_64_byte_aligned_like_real_numpy() {
        let a = NdArray::<f64>::zeros(&[3, 4]);
        let mut buf = Vec::new();
        write_npy(&mut buf, &a).unwrap();
        let header_len = u16::from_le_bytes([buf[8], buf[9]]) as usize;
        assert_eq!((10 + header_len) % 64, 0);
    }

    #[test]
    fn rejects_bad_magic() {
        let err = read_npy::<f64, _>(&b"not-an-npy-file-at-all!"[..]).unwrap_err();
        assert!(matches!(err, NpyError::BadMagic(_)));
    }

    fn fixture(name: &str) -> std::path::PathBuf {
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures").join(name)
    }

    #[test]
    fn reads_real_numpy_vector() {
        let arr = load_npy::<f64, _>(fixture("vector_f64.npy")).unwrap();
        assert_eq!(arr.shape(), &[5]);
        assert_eq!(arr.as_slice(), &[1.0, 2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn reads_real_numpy_matrix() {

        let arr = load_npy::<f64, _>(fixture("matrix_f64.npy")).unwrap();
        assert_eq!(arr.shape(), &[3, 4]);
        assert_eq!(arr.get(&[0, 0]), Some(0.0));
        assert_eq!(arr.get(&[2, 3]), Some(11.0));
    }

    #[test]
    fn reads_real_numpy_cube_and_scalar() {
        let cube = load_npy::<f64, _>(fixture("cube_f64.npy")).unwrap();
        assert_eq!(cube.shape(), &[2, 3, 4]);
        assert_eq!(cube.get(&[1, 2, 3]), Some(23.0));

        let scalar = load_npy::<f64, _>(fixture("scalar_f64.npy")).unwrap();
        assert_eq!(scalar.shape(), &[] as &[usize]);
        assert_eq!(scalar.get(&[]), Some(42.0));
    }

    #[test]
    fn writer_output_is_byte_identical_to_real_numpy() {

        let arr = NdArray::from_vec((0..12).map(|i| i as f64).collect(), &[3, 4]).unwrap();
        let mut ours = Vec::new();
        write_npy(&mut ours, &arr).unwrap();

        let reference = std::fs::read(fixture("matrix_f64.npy")).unwrap();
        assert_eq!(ours, reference);
    }

    fn load<T: NpyElement>(name: &str) -> NdArray<T> {
        load_npy(fixture(name)).unwrap()
    }

    #[test]
    fn reads_every_dtype_written_by_real_numpy() {
        assert_eq!(load::<i32>("vector_i32.npy").as_slice(), &[-2, 0, 7, i32::MAX]);
        let u = load::<u8>("matrix_u8.npy");
        assert_eq!((u.shape(), u.get(&[2, 3])), (&[3, 4][..], Some(11)));
        assert_eq!(load::<f32>("vector_f32.npy").as_slice(), &[0.5, -1.25, 3.0e10]);
        let h = load::<half::f16>("vector_f16.npy");
        assert_eq!(h.as_slice().iter().map(|x| x.to_f32()).collect::<Vec<_>>(), vec![0.5, -1.25, 65504.0]);
        assert_eq!(load::<bool>("vector_bool.npy").as_slice(), &[true, false, true]);
        let c = load::<num_complex::Complex<f64>>("vector_c16.npy");
        assert_eq!(c.as_slice(), &[num_complex::Complex::new(1.0, 2.0), num_complex::Complex::new(0.0, -0.5), num_complex::Complex::new(3.0, 0.0)]);
        let c8 = load::<num_complex::Complex<f32>>("vector_c8.npy");
        assert_eq!(c8.as_slice(), &[num_complex::Complex::new(1.0, 2.0), num_complex::Complex::new(0.0, -0.5)]);
    }

    #[test]
    fn reads_fortran_order_and_big_endian_and_empty() {
        let f = load::<f64>("matrix_f64_fortran.npy");
        assert_eq!(f.shape(), &[3, 4]);
        assert_eq!(f.get(&[1, 2]), Some(6.0));
        assert_eq!(f.as_slice(), load::<f64>("matrix_f64.npy").as_slice());
        assert_eq!(load::<i32>("vector_i4_bigendian.npy").as_slice(), &[1, -2, 300000]);
        let e = load::<f64>("empty_f64.npy");
        assert_eq!((e.shape(), e.len()), (&[0, 3][..], 0));
    }

    #[test]
    fn wrong_element_type_is_reported_not_reinterpreted() {
        let err = load_npy::<f32, _>(fixture("vector_f64.npy")).unwrap_err();
        assert!(matches!(err, NpyError::UnsupportedDtype(ref d) if d == "<f8"));
        assert!(load_npy::<i64, _>(fixture("vector_i32.npy")).is_err());
    }

    #[test]
    fn every_dtype_round_trips_and_matches_real_numpy_bytes() {
        fn same<T: NpyElement + PartialEq + std::fmt::Debug>(values: Vec<T>, fixture_name: &str) {
            let a = NdArray::from_vec(values, &[3]).unwrap();
            let mut ours = Vec::new();
            write_npy(&mut ours, &a).unwrap();
            assert_eq!(read_npy::<T, _>(&ours[..]).unwrap().as_slice(), a.as_slice());
            if !fixture_name.is_empty() {
                assert_eq!(ours, std::fs::read(fixture(fixture_name)).unwrap(), "{fixture_name}");
            }
        }
        same(vec![true, false, true], "vector_bool.npy");
        same(vec![0.5f32, -1.25, 3.0e10], "vector_f32.npy");
        same(vec![0.5f64, -1.25, 1e300], "");
        same(vec![i8::MIN, 0, i8::MAX], "");
        same(vec![u16::MAX, 0, 7], "");
        same(vec![i64::MIN, 0, i64::MAX], "");
        same(vec![u64::MAX, 0, 7], "");
        let c = vec![num_complex::Complex::new(1.0f64, 2.0), num_complex::Complex::new(-0.0, -0.5), num_complex::Complex::new(3.0, 0.0)];
        same(c, "vector_c16.npy");
    }
}
