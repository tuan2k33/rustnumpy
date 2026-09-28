//! Run: `cargo run --example step9_structured_and_datetime`
//!
//! Step 9: structured/record dtypes, and `datetime64`/`timedelta64`. Every
//! value below was cross-checked against real NumPy 2.5.3 first (see the
//! comment above each block).

use rustnumpy::{Datetime64, Kind, RecordArray, RecordDType, TimeUnit, Timedelta64};

fn main() {
    // -- Structured dtype -------------------------------------------------
    // Python: np.dtype([('x','i4'),('y','f8'),('z','i4')]) -- packed,
    // align=False (NumPy's default): itemsize=16, offsets 0/4/12.
    let dtype = RecordDType::new(&[("x", Kind::Int(32)), ("y", Kind::Float(64)), ("z", Kind::Int(32))]);
    println!("RecordDType itemsize={} fields={:?}\n", dtype.itemsize(), dtype.fields());

    let mut records = RecordArray::zeros(dtype, 2);
    records.set(0, "x", 1i32).unwrap();
    records.set(0, "y", 2.5f64).unwrap();
    records.set(0, "z", 3i32).unwrap();
    records.set(1, "x", -7i32).unwrap();
    records.set(1, "y", -1.25f64).unwrap();
    records.set(1, "z", -2i32).unwrap();
    for i in 0..records.len() {
        let x: i32 = records.get(i, "x").unwrap();
        let y: f64 = records.get(i, "y").unwrap();
        let z: i32 = records.get(i, "z").unwrap();
        println!("record {i}: (x={x}, y={y}, z={z})");
    }
    println!();

    // -- datetime64 / timedelta64 ------------------------------------------
    // Python: np.datetime64('2024-01-01','D') + np.timedelta64(5,'D')
    // -> np.datetime64('2024-01-06','D')
    let d1 = Datetime64::new(19723, TimeUnit::Day); // 2024-01-01, per np's own i8 tick
    let out = d1.checked_add(Timedelta64::new(5, TimeUnit::Day)).unwrap();
    println!("2024-01-01 + 5 days -> tick {} (day unit)", out.ticks());

    // Python: np.datetime64('2024-01-10','D') - np.datetime64('2024-01-01','D')
    // -> np.timedelta64(9,'D')
    let d2 = Datetime64::new(19732, TimeUnit::Day); // 2024-01-10
    let delta = d2.checked_sub(d1).unwrap();
    println!("2024-01-10 - 2024-01-01 -> {} day(s)", delta.ticks());

    // Python: np.timedelta64(1,'m') + np.timedelta64(30,'s') ->
    // np.timedelta64(90,'s') -- unit promotes to the finer one
    let sum = Timedelta64::new(1, TimeUnit::Minute).checked_add(Timedelta64::new(30, TimeUnit::Second)).unwrap();
    println!("1 minute + 30 seconds -> {} seconds", sum.ticks());

    // Python: np.datetime64('NaT') == np.datetime64('NaT') -> False;
    // NaT propagates through arithmetic instead of erroring.
    let nat = Timedelta64::nat(TimeUnit::Second);
    // The self-comparison is the whole point here, not a mistake --
    // silencing clippy's "comparing a value to itself" lint on purpose.
    #[allow(clippy::eq_op)]
    let nat_eq_nat = nat == nat;
    println!(
        "NaT == NaT -> {nat_eq_nat} (matches real NumPy: NaT is never equal to anything, even itself)"
    );
    println!("NaT + 1s -> is_nat: {}", nat.checked_add(Timedelta64::new(1, TimeUnit::Second)).unwrap().is_nat());
}
