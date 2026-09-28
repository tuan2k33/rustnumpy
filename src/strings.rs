//! Step 10: `StringDType` (NEP 55) + a subset of `numpy.strings`'s
//! string-processing ufuncs.
//!
//! Scope, per `NumPy.md`'s NumPy >= 2.5 target and NEP 55's own design:
//!
//! - Fixed-width `U<n>`/`S<n>` string dtypes and the whole `numpy.char`
//!   module (which operates on them) are already listed in `NumPy.md` as
//!   dropped, superseded by `StringDType`. This module implements the
//!   `numpy.strings` namespace instead — real NumPy 2.5's actual current
//!   API for `StringDType` arrays (verified by listing
//!   `dir(numpy.strings)` against the stock 2.5.3 venv).
//! - `numpy.strings` has **no `split`** in this version at all (it only
//!   exists on the legacy `numpy.char` side, for fixed-width arrays) —
//!   confirmed by checking, not assumed from an older memory of the API.
//!   So `split` is intentionally not implemented here, even though an
//!   earlier draft of this project's plan mentioned it as an example.
//! - Rust's `String`/`&str` are already always UTF-8, so the "variable-
//!   length UTF-8 representation" and "small-string optimization is a
//!   memory-layout nicety" parts of NEP 55 fall out for free — no custom
//!   arena allocator or small-string crate is needed to match behavior
//!   (`NumPy.md`'s own "Rust mapping" note for NEP 55 says the same).
//! - The **missing-data sentinel** maps directly onto `Option<String>` —
//!   verified against real NumPy that this is *not* the same behavior as
//!   `datetime64`'s NaT: `None == None` is `true` for `StringDType`, not
//!   `false` (NaT is never equal to itself; a missing string is). That's
//!   exactly `Option<T>`'s own default `PartialEq`, so no custom `Eq` impl
//!   is needed here (unlike `datetime.rs`'s `Timedelta64`/`Datetime64`).
//! - Also verified: unlike missing strings under `==`, every
//!   *transformation* function (`upper`, `str_len`, `add`, ...) does
//!   **not** silently propagate a missing value — real NumPy raises
//!   (`ValueError: The length of a null string is undefined`,
//!   `Cannot add null that is not a nan-like value`). This module matches
//!   that: a `Null` element hitting any transform is `Err`, not a silent
//!   `None` result.

/// A 1-D array of optional strings — `None` is `StringDType`'s missing-
/// data sentinel. Standalone (like `structured.rs`/`datetime.rs`), not yet
/// wired into `NdArray`, which is still a fixed-`f64` buffer.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct StringArray {
    values: Vec<Option<String>>,
}

/// Errors from a string ufunc — currently just "one of the inputs was the
/// missing-data sentinel, and this operation doesn't support that" (see
/// the module doc comment for which operations do/don't).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StringError {
    /// Element `index` was `None` (the `StringDType` missing-data
    /// sentinel), and `function` doesn't accept it.
    NullValue { index: usize, function: &'static str },
    /// Two arrays passed to an elementwise binary op had different
    /// lengths (this module has no broadcasting machinery of its own —
    /// see the module doc comment).
    LengthMismatch { lhs: usize, rhs: usize },
}

impl std::fmt::Display for StringError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StringError::NullValue { index, function } => {
                write!(f, "{function}: element {index} is null (missing-data sentinel)")
            }
            StringError::LengthMismatch { lhs, rhs } => {
                write!(f, "arrays have different lengths: {lhs} vs {rhs}")
            }
        }
    }
}

impl std::error::Error for StringError {}

impl StringArray {
    pub fn from_values(values: Vec<Option<String>>) -> Self {
        Self { values }
    }

    /// Build from plain strings — a convenience for the common case with
    /// no missing values at all.
    pub fn from_strings<I, S>(values: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        Self { values: values.into_iter().map(|s| Some(s.into())).collect() }
    }

    pub fn len(&self) -> usize {
        self.values.len()
    }

    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    pub fn get(&self, index: usize) -> Option<&Option<String>> {
        self.values.get(index)
    }

    pub fn values(&self) -> &[Option<String>] {
        &self.values
    }

    /// Apply `f` to every non-null element, producing a new `StringArray`
    /// — the shared engine behind `upper`/`lower`/`strip`/etc. `Err`s on
    /// the first null element encountered, naming which `function` and
    /// `index` (matching how real NumPy's error identifies the failing
    /// element only implicitly via the exception; naming the index here
    /// is a small improvement this port can afford since it isn't bound
    /// by NumPy's existing C error-reporting path).
    fn map_str(&self, function: &'static str, f: impl Fn(&str) -> String) -> Result<StringArray, StringError> {
        let mut out = Vec::with_capacity(self.values.len());
        for (index, value) in self.values.iter().enumerate() {
            match value {
                Some(s) => out.push(Some(f(s))),
                None => return Err(StringError::NullValue { index, function }),
            }
        }
        Ok(StringArray { values: out })
    }

    /// `np.strings.upper(a)`.
    pub fn upper(&self) -> Result<StringArray, StringError> {
        self.map_str("upper", |s| s.to_uppercase())
    }

    /// `np.strings.lower(a)`.
    pub fn lower(&self) -> Result<StringArray, StringError> {
        self.map_str("lower", |s| s.to_lowercase())
    }

    /// `np.strings.strip(a)` — strips ASCII/Unicode whitespace from both
    /// ends, matching Python's (and NumPy's) default `str.strip()`.
    pub fn strip(&self) -> Result<StringArray, StringError> {
        self.map_str("strip", |s| s.trim().to_string())
    }

    /// `np.strings.lstrip(a)`.
    pub fn lstrip(&self) -> Result<StringArray, StringError> {
        self.map_str("lstrip", |s| s.trim_start().to_string())
    }

    /// `np.strings.rstrip(a)`.
    pub fn rstrip(&self) -> Result<StringArray, StringError> {
        self.map_str("rstrip", |s| s.trim_end().to_string())
    }

    /// `np.strings.capitalize(a)` — first character uppercased, the rest
    /// lowercased (matches Python's `str.capitalize`, which is what real
    /// NumPy's version wraps).
    pub fn capitalize(&self) -> Result<StringArray, StringError> {
        self.map_str("capitalize", |s| {
            let mut chars = s.chars();
            match chars.next() {
                Some(first) => first.to_uppercase().collect::<String>() + &chars.as_str().to_lowercase(),
                None => String::new(),
            }
        })
    }

    /// `np.strings.str_len(a)` — returns plain lengths (in Unicode scalar
    /// values, i.e. `chars().count()`, matching Python's `len(str)`
    /// semantics that NumPy itself follows) rather than an `NdArray`,
    /// since the ufunc engine (`ufunc.rs`) is still `f64`-only and a
    /// length is conceptually an integer count, not a float.
    pub fn str_len(&self) -> Result<Vec<usize>, StringError> {
        let mut out = Vec::with_capacity(self.values.len());
        for (index, value) in self.values.iter().enumerate() {
            match value {
                Some(s) => out.push(s.chars().count()),
                None => return Err(StringError::NullValue { index, function: "str_len" }),
            }
        }
        Ok(out)
    }

    /// `np.strings.add(a, suffix)` — concatenate a fixed string onto every
    /// element (the scalar-broadcast form).
    pub fn add_scalar(&self, suffix: &str) -> Result<StringArray, StringError> {
        self.map_str("add", |s| format!("{s}{suffix}"))
    }

    /// `np.strings.add(a, b)` — elementwise concatenation of two arrays of
    /// equal length (the array-array form; this module has no
    /// broadcasting of its own, unlike `NdArray`'s ufuncs, so lengths must
    /// match exactly).
    pub fn add(&self, other: &StringArray) -> Result<StringArray, StringError> {
        if self.len() != other.len() {
            return Err(StringError::LengthMismatch { lhs: self.len(), rhs: other.len() });
        }
        let mut out = Vec::with_capacity(self.values.len());
        for (index, (a, b)) in self.values.iter().zip(other.values.iter()).enumerate() {
            match (a, b) {
                (Some(a), Some(b)) => out.push(Some(format!("{a}{b}"))),
                _ => return Err(StringError::NullValue { index, function: "add" }),
            }
        }
        Ok(StringArray { values: out })
    }

    /// `np.strings.replace(a, from, to)`.
    pub fn replace(&self, from: &str, to: &str) -> Result<StringArray, StringError> {
        self.map_str("replace", |s| s.replace(from, to))
    }

    /// `np.strings.startswith(a, prefix)`.
    pub fn startswith(&self, prefix: &str) -> Result<Vec<bool>, StringError> {
        let mut out = Vec::with_capacity(self.values.len());
        for (index, value) in self.values.iter().enumerate() {
            match value {
                Some(s) => out.push(s.starts_with(prefix)),
                None => return Err(StringError::NullValue { index, function: "startswith" }),
            }
        }
        Ok(out)
    }

    /// `np.strings.endswith(a, suffix)`.
    pub fn endswith(&self, suffix: &str) -> Result<Vec<bool>, StringError> {
        let mut out = Vec::with_capacity(self.values.len());
        for (index, value) in self.values.iter().enumerate() {
            match value {
                Some(s) => out.push(s.ends_with(suffix)),
                None => return Err(StringError::NullValue { index, function: "endswith" }),
            }
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> StringArray {
        // np.array(["  Hello ", "WORLD", "  foo bar "], dtype=np.dtypes.StringDType())
        StringArray::from_strings(["  Hello ", "WORLD", "  foo bar "])
    }

    fn as_strs(arr: &StringArray) -> Vec<&str> {
        arr.values().iter().map(|v| v.as_deref().unwrap()).collect()
    }

    #[test]
    fn upper_matches_real_numpy() {
        // np.strings.upper(a) -> ['  HELLO ' 'WORLD' '  FOO BAR ']
        assert_eq!(as_strs(&sample().upper().unwrap()), vec!["  HELLO ", "WORLD", "  FOO BAR "]);
    }

    #[test]
    fn lower_matches_real_numpy() {
        // np.strings.lower(a) -> ['  hello ' 'world' '  foo bar ']
        assert_eq!(as_strs(&sample().lower().unwrap()), vec!["  hello ", "world", "  foo bar "]);
    }

    #[test]
    fn strip_lstrip_rstrip_match_real_numpy() {
        // np.strings.strip(a) -> ['Hello' 'WORLD' 'foo bar']
        assert_eq!(as_strs(&sample().strip().unwrap()), vec!["Hello", "WORLD", "foo bar"]);
        // np.strings.lstrip(a) -> ['Hello ' 'WORLD' 'foo bar ']
        assert_eq!(as_strs(&sample().lstrip().unwrap()), vec!["Hello ", "WORLD", "foo bar "]);
        // np.strings.rstrip(a) -> ['  Hello' 'WORLD' '  foo bar']
        assert_eq!(as_strs(&sample().rstrip().unwrap()), vec!["  Hello", "WORLD", "  foo bar"]);
    }

    #[test]
    fn capitalize_matches_real_numpy() {
        // np.strings.capitalize(a) -> ['  hello ' 'World' '  foo bar ']
        assert_eq!(as_strs(&sample().capitalize().unwrap()), vec!["  hello ", "World", "  foo bar "]);
    }

    #[test]
    fn str_len_matches_real_numpy() {
        // np.strings.str_len(a) -> [8, 5, 10]
        assert_eq!(sample().str_len().unwrap(), vec![8, 5, 10]);
    }

    #[test]
    fn add_scalar_matches_real_numpy() {
        // np.strings.add(a, "!") -> ['  Hello !' 'WORLD!' '  foo bar !']
        assert_eq!(
            as_strs(&sample().add_scalar("!").unwrap()),
            vec!["  Hello !", "WORLD!", "  foo bar !"]
        );
    }

    #[test]
    fn add_array_matches_real_numpy() {
        // np.strings.add(["foo","bar"], ["1","2"]) -> ['foo1' 'bar2']
        let a = StringArray::from_strings(["foo", "bar"]);
        let b = StringArray::from_strings(["1", "2"]);
        assert_eq!(as_strs(&a.add(&b).unwrap()), vec!["foo1", "bar2"]);
    }

    #[test]
    fn add_array_length_mismatch_errs() {
        let a = StringArray::from_strings(["foo", "bar"]);
        let b = StringArray::from_strings(["1"]);
        assert_eq!(a.add(&b).unwrap_err(), StringError::LengthMismatch { lhs: 2, rhs: 1 });
    }

    #[test]
    fn replace_matches_real_numpy() {
        // np.strings.replace(a, "o", "0") -> ['  Hell0 ' 'WORLD' '  f00 bar ']
        assert_eq!(
            as_strs(&sample().replace("o", "0").unwrap()),
            vec!["  Hell0 ", "WORLD", "  f00 bar "]
        );
    }

    #[test]
    fn startswith_endswith_match_real_numpy() {
        // np.strings.startswith(a, "  ") -> [True, False, True]
        assert_eq!(sample().startswith("  ").unwrap(), vec![true, false, true]);
        // np.strings.endswith(a, "r ") -> [False, False, True]
        assert_eq!(sample().endswith("r ").unwrap(), vec![false, false, true]);
    }

    #[test]
    fn null_element_is_rejected_by_transforms_not_silently_propagated() {
        // np.strings.str_len(np.array(["x", None, "z"], dtype=...)) raises
        // "ValueError: The length of a null string is undefined" -- our
        // Err plays the same role.
        let arr = StringArray::from_values(vec![Some("x".to_string()), None, Some("z".to_string())]);
        assert_eq!(arr.str_len().unwrap_err(), StringError::NullValue { index: 1, function: "str_len" });
        assert_eq!(arr.upper().unwrap_err(), StringError::NullValue { index: 1, function: "upper" });
        // np.strings.add(b, "!") raises "Cannot add null that is not a
        // nan-like value".
        assert_eq!(
            arr.add_scalar("!").unwrap_err(),
            StringError::NullValue { index: 1, function: "add" }
        );
    }

    #[test]
    fn null_equals_null_unlike_datetime64s_nat() {
        // np.array(["x",None,"z"],dtype=na) == np.array(["x",None,"q"],dtype=na)
        // -> [True, True, False] -- None == None is True for StringDType,
        // verified against real NumPy specifically because it's the
        // opposite of datetime64's NaT (never equal to itself).
        let a = StringArray::from_values(vec![Some("x".to_string()), None, Some("z".to_string())]);
        let b = StringArray::from_values(vec![Some("x".to_string()), None, Some("q".to_string())]);
        let equal: Vec<bool> = a.values().iter().zip(b.values()).map(|(x, y)| x == y).collect();
        assert_eq!(equal, vec![true, true, false]);
    }
}
