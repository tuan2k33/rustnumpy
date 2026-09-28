//! Run: `cargo run --example step19_strings`
//!
//! Step 19: `StringDType` (NEP 55) + a subset of `numpy.strings`. Every
//! value below was checked against real NumPy 2.5.3 first (see the
//! comment above each block) -- including the two counter-intuitive
//! findings from that check: `numpy.strings` has no `split` at all in
//! this version, and its missing-data sentinel compares equal to itself
//! (unlike `datetime64`'s NaT), but every transform function *rejects* a
//! null input instead of silently propagating it.

use rustnumpy::StringArray;

fn main() {
    // Python: np.array(["  Hello ", "WORLD", "  foo bar "], dtype=np.dtypes.StringDType())
    let a = StringArray::from_strings(["  Hello ", "WORLD", "  foo bar "]);

    println!("a.upper()      -> {:?}", a.upper().unwrap().values());
    println!("a.lower()      -> {:?}", a.lower().unwrap().values());
    println!("a.strip()      -> {:?}", a.strip().unwrap().values());
    println!("a.capitalize() -> {:?}", a.capitalize().unwrap().values());
    println!("a.str_len()    -> {:?}", a.str_len().unwrap());
    println!("a.add(\"!\")     -> {:?}", a.add_scalar("!").unwrap().values());
    println!("a.replace(\"o\",\"0\") -> {:?}", a.replace("o", "0").unwrap().values());
    println!("a.startswith(\"  \") -> {:?}\n", a.startswith("  ").unwrap());

    // Python: np.array(["x", None, "z"], dtype=np.dtypes.StringDType(na_object=None))
    let b = StringArray::from_values(vec![Some("x".to_string()), None, Some("z".to_string())]);
    let c = StringArray::from_values(vec![Some("x".to_string()), None, Some("q".to_string())]);

    // np.strings.equal(b, c) -> [True, True, False] -- None == None is
    // True here, the opposite of datetime64's NaT.
    let equal: Vec<bool> = b.values().iter().zip(c.values()).map(|(x, y)| x == y).collect();
    println!("null == null (StringDType) -> {equal:?}  (True at the null position, unlike datetime64's NaT)");

    // np.strings.str_len(b) raises "ValueError: The length of a null
    // string is undefined" -- transforms reject null, they don't
    // propagate it silently.
    let err = b.str_len().unwrap_err();
    println!("b.str_len() with a null element -> Err: {err}");
}
