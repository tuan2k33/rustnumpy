use std::fmt;
use std::fs::File;
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::path::Path;

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
    FortranOrderUnsupported,
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
            NpyError::UnsupportedDtype(descr) => write!(
                f,
                "unsupported dtype {descr:?}: this reader only understands little-endian float64 ('<f8')"
            ),
            NpyError::FortranOrderUnsupported => {
                write!(f, "fortran_order=True is not supported yet (only C order)")
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

pub fn save_npy<P: AsRef<Path>>(path: P, arr: &NdArray) -> Result<(), NpyError> {
    let file = File::create(path)?;
    write_npy(BufWriter::new(file), arr)
}

pub fn load_npy<P: AsRef<Path>>(path: P) -> Result<NdArray, NpyError> {
    let file = File::open(path)?;
    read_npy(BufReader::new(file))
}

pub fn write_npy<W: Write>(mut w: W, arr: &NdArray) -> Result<(), NpyError> {
    let header_dict = format!(
        "{{'descr': '<f8', 'fortran_order': False, 'shape': {}, }}",
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

    for &value in arr.as_slice() {
        w.write_all(&value.to_le_bytes())?;
    }
    Ok(())
}

pub fn read_npy<R: Read>(mut r: R) -> Result<NdArray, NpyError> {
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

    let descr = extract_str_field(&header, "descr")?;
    if descr != "<f8" && descr != "=f8" {

        return Err(NpyError::UnsupportedDtype(descr));
    }

    if extract_bool_field(&header, "fortran_order")? {
        return Err(NpyError::FortranOrderUnsupported);
    }

    let shape = extract_shape_field(&header)?;
    let len: usize = shape.iter().product();

    let mut data = Vec::with_capacity(len);
    let mut buf8 = [0u8; 8];
    for _ in 0..len {
        r.read_exact(&mut buf8)?;
        data.push(f64::from_le_bytes(buf8));
    }

    Ok(NdArray::from_vec(data, &shape)?)
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
        let a = NdArray::zeros(&[3, 4]);
        let mut buf = Vec::new();
        write_npy(&mut buf, &a).unwrap();
        let header_len = u16::from_le_bytes([buf[8], buf[9]]) as usize;
        assert_eq!((10 + header_len) % 64, 0);
    }

    #[test]
    fn rejects_bad_magic() {
        let err = read_npy(&b"not-an-npy-file-at-all!"[..]).unwrap_err();
        assert!(matches!(err, NpyError::BadMagic(_)));
    }

    fn fixture(name: &str) -> std::path::PathBuf {
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures").join(name)
    }

    #[test]
    fn reads_real_numpy_vector() {
        let arr = load_npy(fixture("vector_f64.npy")).unwrap();
        assert_eq!(arr.shape(), &[5]);
        assert_eq!(arr.as_slice(), &[1.0, 2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn reads_real_numpy_matrix() {

        let arr = load_npy(fixture("matrix_f64.npy")).unwrap();
        assert_eq!(arr.shape(), &[3, 4]);
        assert_eq!(arr.get(&[0, 0]), Some(0.0));
        assert_eq!(arr.get(&[2, 3]), Some(11.0));
    }

    #[test]
    fn reads_real_numpy_cube_and_scalar() {
        let cube = load_npy(fixture("cube_f64.npy")).unwrap();
        assert_eq!(cube.shape(), &[2, 3, 4]);
        assert_eq!(cube.get(&[1, 2, 3]), Some(23.0));

        let scalar = load_npy(fixture("scalar_f64.npy")).unwrap();
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
}
