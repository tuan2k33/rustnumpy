//! Read/write the `.npy` format (NEP 1), restricted to the `float64`
//! little-endian dtype (`<f8`), C order (row-major) — exactly the fixed
//! dtype `NdArray` supports since step 1.
//!
//! File layout (see NumPy.md, "Định dạng file .npy/.npz" section):
//!
//! ```text
//! offset  size            content
//! 0       6 bytes         magic string \x93NUMPY
//! 6       2 bytes         version (major, minor) — always written as 1.0 here
//! 8       2 or 4 bytes    header_len (u16 for v1.x, u32 for v2.x/v3.x)
//! 10/12   header_len byte header dict as a Python literal, padded with spaces + '\n'
//!                         so that (header offset + header_len) is a multiple of 64
//! ...     data_len bytes  raw data, row-major, in the declared dtype
//! ```
//!
//! The header reader **doesn't** use a full Python parser — it just picks
//! out the 3 fixed keys (`descr`, `fortran_order`, `shape`) via manual
//! string scanning, since this format is produced by NumPy itself with a
//! known dict shape, not an arbitrary Python literal.

use std::fmt;
use std::fs::File;
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::path::Path;

use crate::error::ShapeError;
use crate::ndarray::NdArray;

const MAGIC: &[u8; 6] = b"\x93NUMPY";
/// NumPy aligns the header to a multiple of 64 bytes from the start of the
/// file (good for SIMD/mmap reads of the data that follows) — this
/// constant is called `ARRAY_ALIGN` in the C core.
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

/// Write `arr` to `path` in `.npy` v1.0 format.
pub fn save_npy<P: AsRef<Path>>(path: P, arr: &NdArray) -> Result<(), NpyError> {
    let file = File::create(path)?;
    write_npy(BufWriter::new(file), arr)
}

/// Read an array from a `.npy` file at `path`.
pub fn load_npy<P: AsRef<Path>>(path: P) -> Result<NdArray, NpyError> {
    let file = File::open(path)?;
    read_npy(BufReader::new(file))
}

/// Write `arr` in `.npy` v1.0 format to any `Write` (file, in-memory
/// buffer, ...) — kept separate from `save_npy` so tests can write to a
/// `Vec<u8>` and compare byte-for-byte without touching the filesystem.
pub fn write_npy<W: Write>(mut w: W, arr: &NdArray) -> Result<(), NpyError> {
    let header_dict = format!(
        "{{'descr': '<f8', 'fortran_order': False, 'shape': {}, }}",
        format_shape_tuple(arr.shape())
    );

    // Prefix for version 1.0: magic(6) + version(2) + header_len field(2) = 10.
    const PREFIX_LEN: usize = 10;
    let unpadded_len = header_dict.len() + 1; // +1 for the mandatory trailing '\n'
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
    w.write_all(&[1, 0])?; // version 1.0
    w.write_all(&header_len.to_le_bytes())?;
    w.write_all(&header)?;

    for &value in arr.as_slice() {
        w.write_all(&value.to_le_bytes())?;
    }
    Ok(())
}

/// Read an array from any `Read`.
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
        // '=f8' means "native byte order" — on most machines (modern
        // x86/ARM) native is little-endian, so we treat it as equivalent
        // to '<f8'. '>f8' (big-endian) is deliberately unsupported: it
        // would need byte-swapping, which this step doesn't do.
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

/// Print a shape as a Python tuple literal, matching NumPy's own 3
/// conventions exactly: `()` for 0-D, `(N,)` for 1-D (the trailing comma
/// is mandatory so Python doesn't mistake it for a number in parens),
/// `(a, b, ...)` for 2-D and up.
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

/// Find `'key': <rest after the ':'>` in the header, returning the
/// remaining string after the colon (leading whitespace trimmed) so each
/// `extract_*_field` function can parse the expected value type from there.
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

    // --- Cross-check against .npy files produced by real NumPy (scripts/gen_fixtures.py) ---

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
        // np.arange(12, dtype='<f8').reshape(3, 4)
        let arr = load_npy(fixture("matrix_f64.npy")).unwrap();
        assert_eq!(arr.shape(), &[3, 4]);
        assert_eq!(arr.get(&[0, 0]), Some(0.0));
        assert_eq!(arr.get(&[2, 3]), Some(11.0));
    }

    #[test]
    fn reads_real_numpy_cube_and_scalar() {
        let cube = load_npy(fixture("cube_f64.npy")).unwrap();
        assert_eq!(cube.shape(), &[2, 3, 4]);
        assert_eq!(cube.get(&[1, 2, 3]), Some(23.0)); // last element of np.arange(24)

        let scalar = load_npy(fixture("scalar_f64.npy")).unwrap();
        assert_eq!(scalar.shape(), &[] as &[usize]);
        assert_eq!(scalar.get(&[]), Some(42.0));
    }

    #[test]
    fn writer_output_is_byte_identical_to_real_numpy() {
        // Same data as np.arange(12, dtype='<f8').reshape(3, 4) used to
        // generate the fixture — if the bytes match 100%, the writer is
        // genuinely compatible with NumPy, not just "reads back its own output".
        let arr = NdArray::from_vec((0..12).map(|i| i as f64).collect(), &[3, 4]).unwrap();
        let mut ours = Vec::new();
        write_npy(&mut ours, &arr).unwrap();

        let reference = std::fs::read(fixture("matrix_f64.npy")).unwrap();
        assert_eq!(ours, reference);
    }
}
