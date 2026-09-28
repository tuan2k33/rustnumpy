//! Step 18 (part 1): structured / record dtypes — an ordered list of named
//! fields packed into one fixed-size "row", the way `np.dtype([('x','i4'),
//! ('y','f8')])` describes a C struct rather than a single scalar type.
//!
//! Scope, per `NumPy.md`'s NumPy >= 2.5 target: this only implements the
//! **packed** layout (`align=False`), which is NumPy's own default for
//! `np.dtype([...])` — no padding between fields, `itemsize` is exactly
//! the sum of the field sizes. `align=True` (natural C-struct alignment,
//! with padding) is a real NumPy feature but is an opt-in convenience on
//! top of the packed layout, not something deprecated or load-bearing for
//! parity — left out to keep the offset arithmetic (and the unsafe code
//! reading/writing through it) as simple as it can be. `np.recarray`
//! (attribute-style field access) is already listed in `NumPy.md` as
//! dropped entirely; this module only offers index/name-based access.
//!
//! This is also a standalone module, like `dtype.rs`'s `DType` trait: it
//! is not yet wired into `NdArray` (which is still hardcoded to a flat
//! `f64` buffer) — that integration is later work once the ufunc engine
//! itself becomes dtype-generic.

use crate::dtype::{DType, Kind};

/// One named field within a [`RecordDType`]: a name, its scalar `Kind`,
/// and its packed byte offset within one record.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Field {
    pub name: String,
    pub kind: Kind,
    pub offset: usize,
}

/// Errors from building or accessing a [`RecordDType`]/[`RecordArray`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RecordError {
    /// No field with this name exists in the dtype.
    FieldNotFound { name: String },
    /// The field exists, but its `Kind` doesn't match the Rust type `T`
    /// the caller tried to read/write it as (e.g. reading an `f64` field
    /// as `i32`).
    FieldTypeMismatch { name: String, expected: Kind, requested: Kind },
    /// The record index is `>= len()`.
    IndexOutOfBounds { index: usize, len: usize },
}

impl std::fmt::Display for RecordError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RecordError::FieldNotFound { name } => write!(f, "no field named {name:?}"),
            RecordError::FieldTypeMismatch { name, expected, requested } => write!(
                f,
                "field {name:?} has kind {expected:?}, but was accessed as {requested:?}"
            ),
            RecordError::IndexOutOfBounds { index, len } => {
                write!(f, "record index {index} is out of bounds for length {len}")
            }
        }
    }
}

impl std::error::Error for RecordError {}

/// A structured dtype: an ordered, packed list of named fields, built once
/// and shared by every record in a [`RecordArray`] — mirrors
/// `np.dtype([(name, kind), ...]).fields`/`.itemsize`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RecordDType {
    fields: Vec<Field>,
    itemsize: usize,
}

/// A `Kind`'s size in bytes — the same rule NumPy uses to compute a packed
/// structured dtype's field offsets (`bits / 8`).
fn kind_size_bytes(kind: Kind) -> usize {
    match kind {
        Kind::Bool => 1,
        Kind::Int(bits) | Kind::Uint(bits) | Kind::Float(bits) => bits as usize / 8,
        // A complex value is stored as two components of its own width
        // (e.g. `complex128` = two `f64`s), matching NumPy's own
        // `itemsize` for a complex dtype.
        Kind::Complex(component_bits) => 2 * (component_bits as usize / 8),
    }
}

impl RecordDType {
    /// Build a packed structured dtype from `(name, kind)` pairs, in
    /// order — offsets are assigned as a running sum of each preceding
    /// field's size, with no padding, exactly like
    /// `np.dtype([...])` (without `align=True`).
    pub fn new(fields: &[(&str, Kind)]) -> Self {
        let mut offset = 0;
        let built: Vec<Field> = fields
            .iter()
            .map(|&(name, kind)| {
                let field = Field { name: name.to_string(), kind, offset };
                offset += kind_size_bytes(kind);
                field
            })
            .collect();
        Self { fields: built, itemsize: offset }
    }

    pub fn fields(&self) -> &[Field] {
        &self.fields
    }

    /// Total bytes per record — the sum of every field's size, since this
    /// is always the packed (no-padding) layout.
    pub fn itemsize(&self) -> usize {
        self.itemsize
    }

    pub fn field(&self, name: &str) -> Option<&Field> {
        self.fields.iter().find(|f| f.name == name)
    }
}

/// A 1-D array of fixed-size records sharing one [`RecordDType`] — the
/// structured-dtype counterpart to `NdArray`. Kept 1-D and standalone
/// (rather than reusing `NdArray`'s own shape/stride machinery) since the
/// element type here is a whole packed byte row, not a single `f64`.
pub struct RecordArray {
    dtype: RecordDType,
    /// `len` records, each exactly `dtype.itemsize()` bytes, back-to-back.
    buf: Vec<u8>,
    len: usize,
}

impl RecordArray {
    /// A zero-filled array of `len` records under `dtype`.
    pub fn zeros(dtype: RecordDType, len: usize) -> Self {
        let buf = vec![0u8; dtype.itemsize() * len];
        Self { dtype, buf, len }
    }

    pub fn dtype(&self) -> &RecordDType {
        &self.dtype
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// The raw packed bytes of one record — `dtype().itemsize()` long.
    pub fn record_bytes(&self, index: usize) -> Result<&[u8], RecordError> {
        if index >= self.len {
            return Err(RecordError::IndexOutOfBounds { index, len: self.len });
        }
        let start = index * self.dtype.itemsize();
        Ok(&self.buf[start..start + self.dtype.itemsize()])
    }

    /// Read field `name` of record `index` as `T`. `T::KIND` must match
    /// the field's declared `Kind` exactly — this is a structural-typing
    /// check, not a numeric cast (reading an `i32` field as `f32` is a
    /// `FieldTypeMismatch`, not an implicit conversion), matching how
    /// real NumPy's structured-field access hands back the field's own
    /// declared dtype rather than silently casting.
    pub fn get<T: DType>(&self, index: usize, name: &str) -> Result<T, RecordError> {
        let field = self.resolve_field::<T>(name)?;
        if index >= self.len {
            return Err(RecordError::IndexOutOfBounds { index, len: self.len });
        }
        let start = index * self.dtype.itemsize() + field.offset;
        // SAFETY: `start` was computed from a validated field offset (built
        // by `RecordDType::new`, always `<= itemsize - size_of::<T>()`
        // because `Kind::size_bytes()` and `size_of::<T>()` agree whenever
        // `T::KIND == field.kind`, which `resolve_field` just checked) and
        // a validated record index, so `start + size_of::<T>()` is within
        // `self.buf`. `read_unaligned` is required (not plain `read`)
        // because a packed structured dtype can put a field at any byte
        // offset, e.g. an `f64` after an `i32` lands 4 bytes off its
        // natural 8-byte alignment.
        Ok(unsafe { std::ptr::read_unaligned(self.buf.as_ptr().add(start) as *const T) })
    }

    /// Write field `name` of record `index`. Same `T::KIND == field.kind`
    /// requirement as [`RecordArray::get`].
    pub fn set<T: DType>(&mut self, index: usize, name: &str, value: T) -> Result<(), RecordError> {
        let field = self.resolve_field::<T>(name)?;
        if index >= self.len {
            return Err(RecordError::IndexOutOfBounds { index, len: self.len });
        }
        let start = index * self.dtype.itemsize() + field.offset;
        // SAFETY: see `get` — same offset/bounds/alignment reasoning,
        // `write_unaligned` for the same packed-layout reason.
        unsafe { std::ptr::write_unaligned(self.buf.as_mut_ptr().add(start) as *mut T, value) };
        Ok(())
    }

    fn resolve_field<T: DType>(&self, name: &str) -> Result<&Field, RecordError> {
        let field = self.dtype.field(name).ok_or_else(|| RecordError::FieldNotFound { name: name.to_string() })?;
        if field.kind != T::KIND {
            return Err(RecordError::FieldTypeMismatch {
                name: name.to_string(),
                expected: field.kind,
                requested: T::KIND,
            });
        }
        Ok(field)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn point_dtype() -> RecordDType {
        // np.dtype([('x','i4'),('y','f8'),('z','i2')])
        RecordDType::new(&[("x", Kind::Int(32)), ("y", Kind::Float(64)), ("z", Kind::Int(32))])
    }

    #[test]
    fn packed_offsets_match_real_numpy_align_false() {
        // np.dtype([('x','i4'),('y','f8'),('z','i2')]).itemsize == 14 and
        // .fields == {'x': (int32, 0), 'y': (float64, 4), 'z': (int16, 12)}
        // -- our `z` uses Kind::Int(32) (this exercise's DType trait only
        // covers i32/f32/f64/bool, no i16), so offsets/itemsize differ from
        // that exact example but the *packing rule* (no padding, running
        // sum of sizes) is the same thing under test.
        let dtype = point_dtype();
        assert_eq!(dtype.field("x").unwrap().offset, 0);
        assert_eq!(dtype.field("y").unwrap().offset, 4);
        assert_eq!(dtype.field("z").unwrap().offset, 12);
        assert_eq!(dtype.itemsize(), 16);
    }

    #[test]
    fn get_set_roundtrip_across_fields() {
        let mut records = RecordArray::zeros(point_dtype(), 2);
        records.set(0, "x", 1i32).unwrap();
        records.set(0, "y", 2.5f64).unwrap();
        records.set(0, "z", 3i32).unwrap();
        records.set(1, "x", -7i32).unwrap();
        records.set(1, "y", -1.25f64).unwrap();
        records.set(1, "z", -2i32).unwrap();

        assert_eq!(records.get::<i32>(0, "x").unwrap(), 1);
        assert_eq!(records.get::<f64>(0, "y").unwrap(), 2.5);
        assert_eq!(records.get::<i32>(0, "z").unwrap(), 3);
        assert_eq!(records.get::<i32>(1, "x").unwrap(), -7);
        assert_eq!(records.get::<f64>(1, "y").unwrap(), -1.25);
        assert_eq!(records.get::<i32>(1, "z").unwrap(), -2);
    }

    #[test]
    fn packed_bytes_match_real_numpy_tobytes_layout() {
        // Same dtype/values as the real-NumPy check this was verified
        // against: np.dtype([('x','i4'),('y','f8'),('z','i4')]) (note:
        // 'z' as i4 here, not i2, since only i32 is available) with
        // records (1, 2.5, 3) and (-7, -1.25, -2).
        let mut records = RecordArray::zeros(point_dtype(), 2);
        records.set(0, "x", 1i32).unwrap();
        records.set(0, "y", 2.5f64).unwrap();
        records.set(0, "z", 3i32).unwrap();
        records.set(1, "x", -7i32).unwrap();
        records.set(1, "y", -1.25f64).unwrap();
        records.set(1, "z", -2i32).unwrap();

        // Little-endian bytes: x=1 (i32), y=2.5 (f64), z=3 (i32).
        let mut expected = Vec::new();
        expected.extend_from_slice(&1i32.to_le_bytes());
        expected.extend_from_slice(&2.5f64.to_le_bytes());
        expected.extend_from_slice(&3i32.to_le_bytes());
        expected.extend_from_slice(&(-7i32).to_le_bytes());
        expected.extend_from_slice(&(-1.25f64).to_le_bytes());
        expected.extend_from_slice(&(-2i32).to_le_bytes());

        assert_eq!(records.record_bytes(0).unwrap(), &expected[0..16]);
        assert_eq!(records.record_bytes(1).unwrap(), &expected[16..32]);
    }

    #[test]
    fn field_type_mismatch_is_rejected() {
        let records = RecordArray::zeros(point_dtype(), 1);
        let err = records.get::<f32>(0, "x").unwrap_err();
        assert_eq!(
            err,
            RecordError::FieldTypeMismatch {
                name: "x".to_string(),
                expected: Kind::Int(32),
                requested: Kind::Float(32),
            }
        );
    }

    #[test]
    fn unknown_field_is_rejected() {
        let records = RecordArray::zeros(point_dtype(), 1);
        let err = records.get::<i32>(0, "w").unwrap_err();
        assert_eq!(err, RecordError::FieldNotFound { name: "w".to_string() });
    }

    #[test]
    fn out_of_bounds_index_is_rejected() {
        let records = RecordArray::zeros(point_dtype(), 1);
        let err = records.get::<i32>(5, "x").unwrap_err();
        assert_eq!(err, RecordError::IndexOutOfBounds { index: 5, len: 1 });
    }
}
