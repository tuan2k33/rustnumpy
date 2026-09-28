//! Step 18 (part 2): `datetime64`/`timedelta64` — a signed 64-bit tick count
//! against a fixed-ratio unit, the way real NumPy represents both types
//! internally.
//!
//! Scope, per `NumPy.md`'s NumPy >= 2.5 target: real NumPy also has `Y`
//! (year) and `M` (month) units, and a "generic" (unitless) datetime64/
//! timedelta64 you get from e.g. bare `np.datetime64('NaT')`. Both are
//! deliberately **not** implemented here:
//!
//! - NumPy >= 2.5 itself already emits a `DeprecationWarning` for the
//!   generic/unitless form ("this includes implicit conversion of bare
//!   integers... please use a specific unit instead") — exactly the kind
//!   of legacy-only path `NumPy.md` says to skip.
//! - `Y`/`M` have no fixed ratio to any other unit (a year is not a fixed
//!   number of days) — supporting them would mean picking an arbitrary
//!   calendar approximation NumPy itself avoids doing implicitly. Every
//!   [`TimeUnit`] here converts to every other one by exact integer
//!   arithmetic, with no calendar assumptions at all.
//!
//! Also out of scope: parsing ISO-8601 date/time strings (`'2024-01-01'`)
//! — construction here is by raw epoch tick count plus unit, the same
//! representation NumPy stores internally, just without the string
//! front-end.

/// The sentinel "Not a Time" tick value, identical to NumPy's own
/// internal representation (`np.datetime64('NaT').astype('i8') ==
/// i64::MIN`).
const NAT: i64 = i64::MIN;

/// A fixed-ratio time unit. Ordered finest-to-coarsest is *not* what
/// `derive(Ord)` gives here (declaration order is coarsest-to-finest,
/// matching how they're usually listed) — use [`TimeUnit::nanos_per_unit`]
/// to compare resolution.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TimeUnit {
    Week,
    Day,
    Hour,
    Minute,
    Second,
    Millisecond,
    Microsecond,
    Nanosecond,
}

impl TimeUnit {
    /// How many nanoseconds one tick of this unit represents — the fixed
    /// ratio that makes conversion between any two `TimeUnit`s exact
    /// integer arithmetic, unlike `Y`/`M` (see the module doc comment).
    pub const fn nanos_per_unit(self) -> i64 {
        match self {
            TimeUnit::Week => 7 * 24 * 60 * 60 * 1_000_000_000,
            TimeUnit::Day => 24 * 60 * 60 * 1_000_000_000,
            TimeUnit::Hour => 60 * 60 * 1_000_000_000,
            TimeUnit::Minute => 60 * 1_000_000_000,
            TimeUnit::Second => 1_000_000_000,
            TimeUnit::Millisecond => 1_000_000,
            TimeUnit::Microsecond => 1_000,
            TimeUnit::Nanosecond => 1,
        }
    }

    /// The finer (higher-resolution) of two units — the unit real NumPy
    /// promotes to when combining two different units, e.g.
    /// `timedelta64(1,'m') + timedelta64(30,'s') -> timedelta64(90,'s')`.
    fn finer(self, other: TimeUnit) -> TimeUnit {
        if self.nanos_per_unit() <= other.nanos_per_unit() {
            self
        } else {
            other
        }
    }
}

/// Errors converting or combining [`Timedelta64`]/[`Datetime64`] values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TimeError {
    /// Converting between units overflowed `i64` — matches real NumPy's
    /// own `Overflow when converting between datetime64 units`.
    Overflow,
}

impl std::fmt::Display for TimeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TimeError::Overflow => write!(f, "overflow when converting between time units"),
        }
    }
}

impl std::error::Error for TimeError {}

/// Convert `ticks` of `from` into an exact tick count of `to`, or
/// `Err(TimeError::Overflow)` if the result doesn't fit in `i64`. Uses
/// `i128` internally purely as scratch space for the intermediate
/// multiply — the final result is always checked back down to `i64`,
/// exactly the width real NumPy stores ticks in.
fn convert_ticks(ticks: i64, from: TimeUnit, to: TimeUnit) -> Result<i64, TimeError> {
    let nanos = ticks as i128 * from.nanos_per_unit() as i128;
    let converted = nanos / to.nanos_per_unit() as i128;
    i64::try_from(converted).map_err(|_| TimeError::Overflow)
}

/// A duration: a signed tick count against a [`TimeUnit`], e.g.
/// `Timedelta64::new(5, TimeUnit::Day)` for `np.timedelta64(5, 'D')`.
#[derive(Debug, Clone, Copy)]
pub struct Timedelta64 {
    ticks: i64,
    unit: TimeUnit,
}

impl Timedelta64 {
    pub fn new(ticks: i64, unit: TimeUnit) -> Self {
        Self { ticks, unit }
    }

    /// The "Not a Time" sentinel in the given unit — real NumPy's NaT
    /// carries a unit too (only the deprecated generic/unitless form
    /// doesn't, which this port doesn't implement; see the module doc
    /// comment).
    pub fn nat(unit: TimeUnit) -> Self {
        Self { ticks: NAT, unit }
    }

    pub fn is_nat(self) -> bool {
        self.ticks == NAT
    }

    pub fn ticks(self) -> i64 {
        self.ticks
    }

    pub fn unit(self) -> TimeUnit {
        self.unit
    }

    /// Re-express this duration in `target` units. NaT stays NaT
    /// regardless of the target unit (matches `nat.astype(...)` in NumPy:
    /// the sentinel value doesn't get rescaled).
    pub fn to_unit(self, target: TimeUnit) -> Result<Timedelta64, TimeError> {
        if self.is_nat() {
            return Ok(Timedelta64::nat(target));
        }
        Ok(Timedelta64 { ticks: convert_ticks(self.ticks, self.unit, target)?, unit: target })
    }

    /// `self + other`, promoted to the finer of the two units — matches
    /// `np.timedelta64(1,'m') + np.timedelta64(30,'s') ->
    /// np.timedelta64(90,'s')`. NaT propagates through, like NaN.
    pub fn checked_add(self, other: Timedelta64) -> Result<Timedelta64, TimeError> {
        if self.is_nat() || other.is_nat() {
            return Ok(Timedelta64::nat(self.unit.finer(other.unit)));
        }
        let unit = self.unit.finer(other.unit);
        let a = self.to_unit(unit)?;
        let b = other.to_unit(unit)?;
        a.ticks
            .checked_add(b.ticks)
            .map(|ticks| Timedelta64 { ticks, unit })
            .ok_or(TimeError::Overflow)
    }

    /// `self - other`, same unit-promotion rule as [`Timedelta64::checked_add`].
    pub fn checked_sub(self, other: Timedelta64) -> Result<Timedelta64, TimeError> {
        if self.is_nat() || other.is_nat() {
            return Ok(Timedelta64::nat(self.unit.finer(other.unit)));
        }
        let unit = self.unit.finer(other.unit);
        let a = self.to_unit(unit)?;
        let b = other.to_unit(unit)?;
        a.ticks
            .checked_sub(b.ticks)
            .map(|ticks| Timedelta64 { ticks, unit })
            .ok_or(TimeError::Overflow)
    }
}

/// `Timedelta64`s compare equal only when neither is NaT and they denote
/// the same duration once converted to a common unit — NaT is never equal
/// to anything, **including another NaT**, matching real NumPy
/// (`np.timedelta64('NaT','s') == np.timedelta64('NaT','s')` is `False`)
/// and the same deliberate exception `f64::NAN` makes to `Eq`'s usual
/// reflexivity rule.
impl PartialEq for Timedelta64 {
    fn eq(&self, other: &Self) -> bool {
        if self.is_nat() || other.is_nat() {
            return false;
        }
        match self.to_unit(other.unit) {
            Ok(converted) => converted.ticks == other.ticks,
            Err(_) => false,
        }
    }
}

/// Ordering following the same NaT rule as [`PartialEq`]: any comparison
/// touching NaT returns `None` (`Ord`'s `<`/`>`/etc. all become `false`
/// for it automatically, and `!=` — from `PartialEq`, not `PartialOrd` —
/// stays `true`), matching `np.timedelta64('NaT','s') < x` being `False`.
impl PartialOrd for Timedelta64 {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        if self.is_nat() || other.is_nat() {
            return None;
        }
        let converted = self.to_unit(other.unit).ok()?;
        Some(converted.ticks.cmp(&other.ticks))
    }
}

/// A point in time: a signed tick count since the Unix epoch
/// (1970-01-01T00:00:00) against a [`TimeUnit`], e.g.
/// `Datetime64::new(19723, TimeUnit::Day)` for `np.datetime64('2024-01-01',
/// 'D')` (19723 days after the epoch).
#[derive(Debug, Clone, Copy)]
pub struct Datetime64 {
    ticks: i64,
    unit: TimeUnit,
}

impl Datetime64 {
    pub fn new(ticks: i64, unit: TimeUnit) -> Self {
        Self { ticks, unit }
    }

    pub fn nat(unit: TimeUnit) -> Self {
        Self { ticks: NAT, unit }
    }

    pub fn is_nat(self) -> bool {
        self.ticks == NAT
    }

    pub fn ticks(self) -> i64 {
        self.ticks
    }

    pub fn unit(self) -> TimeUnit {
        self.unit
    }

    pub fn to_unit(self, target: TimeUnit) -> Result<Datetime64, TimeError> {
        if self.is_nat() {
            return Ok(Datetime64::nat(target));
        }
        Ok(Datetime64 { ticks: convert_ticks(self.ticks, self.unit, target)?, unit: target })
    }

    /// `date + duration`, promoted to the finer unit — matches
    /// `np.datetime64('2024-01-01','D') + np.timedelta64(5,'D')`.
    pub fn checked_add(self, delta: Timedelta64) -> Result<Datetime64, TimeError> {
        if self.is_nat() || delta.is_nat() {
            return Ok(Datetime64::nat(self.unit.finer(delta.unit)));
        }
        let unit = self.unit.finer(delta.unit);
        let a = self.to_unit(unit)?;
        let b = delta.to_unit(unit)?;
        a.ticks
            .checked_add(b.ticks)
            .map(|ticks| Datetime64 { ticks, unit })
            .ok_or(TimeError::Overflow)
    }

    /// `date - duration -> date`.
    pub fn checked_sub_delta(self, delta: Timedelta64) -> Result<Datetime64, TimeError> {
        if self.is_nat() || delta.is_nat() {
            return Ok(Datetime64::nat(self.unit.finer(delta.unit)));
        }
        let unit = self.unit.finer(delta.unit);
        let a = self.to_unit(unit)?;
        let b = delta.to_unit(unit)?;
        a.ticks
            .checked_sub(b.ticks)
            .map(|ticks| Datetime64 { ticks, unit })
            .ok_or(TimeError::Overflow)
    }

    /// `date - date -> duration`, promoted to the finer unit — matches
    /// `np.datetime64('2024-01-10','D') - np.datetime64('2024-01-01','D')
    /// -> np.timedelta64(9,'D')`.
    pub fn checked_sub(self, other: Datetime64) -> Result<Timedelta64, TimeError> {
        if self.is_nat() || other.is_nat() {
            return Ok(Timedelta64::nat(self.unit.finer(other.unit)));
        }
        let unit = self.unit.finer(other.unit);
        let a = self.to_unit(unit)?;
        let b = other.to_unit(unit)?;
        a.ticks
            .checked_sub(b.ticks)
            .map(|ticks| Timedelta64::new(ticks, unit))
            .ok_or(TimeError::Overflow)
    }
}

impl PartialEq for Datetime64 {
    fn eq(&self, other: &Self) -> bool {
        if self.is_nat() || other.is_nat() {
            return false;
        }
        match self.to_unit(other.unit) {
            Ok(converted) => converted.ticks == other.ticks,
            Err(_) => false,
        }
    }
}

impl PartialOrd for Datetime64 {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        if self.is_nat() || other.is_nat() {
            return None;
        }
        let converted = self.to_unit(other.unit).ok()?;
        Some(converted.ticks.cmp(&other.ticks))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use TimeUnit::*;

    #[test]
    fn unit_conversion_matches_real_numpy() {
        // np.timedelta64(1,'m').astype('timedelta64[s]') == 60 seconds
        assert_eq!(Timedelta64::new(1, Minute).to_unit(Second).unwrap().ticks(), 60);
        // np.timedelta64(5,'s').astype('timedelta64[ns]') == 5000000000 ns
        assert_eq!(Timedelta64::new(5, Second).to_unit(Nanosecond).unwrap().ticks(), 5_000_000_000);
    }

    #[test]
    fn overflow_on_conversion_is_an_error_not_a_panic() {
        // np.timedelta64(2**60,'s').astype('timedelta64[ns]') raises
        // "Overflow when converting between datetime64 units".
        let huge = Timedelta64::new(1i64 << 60, Second);
        assert_eq!(huge.to_unit(Nanosecond), Err(TimeError::Overflow));
    }

    #[test]
    fn add_promotes_to_the_finer_unit() {
        // np.timedelta64(1,'m') + np.timedelta64(30,'s') ->
        // np.timedelta64(90,'s')
        let sum = Timedelta64::new(1, Minute).checked_add(Timedelta64::new(30, Second)).unwrap();
        assert_eq!(sum.ticks(), 90);
        assert_eq!(sum.unit(), Second);

        // np.timedelta64(1,'D') + np.timedelta64(1,'W') ->
        // np.timedelta64(8,'D')
        let sum = Timedelta64::new(1, Day).checked_add(Timedelta64::new(1, Week)).unwrap();
        assert_eq!(sum.ticks(), 8);
        assert_eq!(sum.unit(), Day);
    }

    #[test]
    fn datetime_arithmetic_matches_real_numpy() {
        // np.datetime64('2024-01-01','D') + np.timedelta64(5,'D')
        // -> np.datetime64('2024-01-06','D'); using days-since-epoch
        // 19723 for 2024-01-01 (matches real NumPy's own internal tick).
        let d1 = Datetime64::new(19723, Day);
        let out = d1.checked_add(Timedelta64::new(5, Day)).unwrap();
        assert_eq!(out.ticks(), 19728);
        assert_eq!(out.unit(), Day);

        // np.datetime64('2024-01-10','D') - np.datetime64('2024-01-01','D')
        // -> np.timedelta64(9,'D')
        let d2 = Datetime64::new(19732, Day);
        let delta = d2.checked_sub(d1).unwrap();
        assert_eq!(delta.ticks(), 9);
        assert_eq!(delta.unit(), Day);
    }

    #[test]
    fn mixed_unit_datetime_comparison_converts_to_common_unit() {
        // np.datetime64('2024-01-01','D') == np.datetime64('2024-01-01T12:00','h')
        // -> False; D < h(that timestamp) -> True
        let d = Datetime64::new(19723, Day);
        let h = Datetime64::new(19723 * 24 + 12, Hour);
        assert_ne!(d, h);
        assert!(d < h);

        // (h - d) -> 12 hours
        let delta = h.checked_sub(d).unwrap();
        assert_eq!(delta.ticks(), 12);
        assert_eq!(delta.unit(), Hour);
    }

    #[test]
    fn nat_equals_nothing_including_itself() {
        // np.datetime64('NaT') == np.datetime64('NaT') -> False
        // np.datetime64('NaT') != np.datetime64('NaT') -> True
        let nat = Timedelta64::nat(Second);
        assert_ne!(nat, nat); // eq() returns false even when comparing NaT to itself
    }

    #[test]
    fn nat_compares_false_to_everything_but_propagates_through_arithmetic() {
        // np.datetime64('NaT') < np.datetime64('2024-01-01') -> False
        let nat = Datetime64::nat(Day);
        let d = Datetime64::new(19723, Day);
        // NaT is incomparable to anything, not just "less than everything"
        // or "greater than everything" -- partial_cmp is None, so every
        // one of <, >, <=, >= built on it is false (never true) in the
        // same way NaN makes float comparisons false, matching
        // `np.datetime64('NaT') < np.datetime64('2024-01-01')` being False.
        assert_eq!(nat.partial_cmp(&d), None);
        assert_ne!(nat, d);

        // np.timedelta64('NaT','s') + np.timedelta64(1,'s') -> NaT
        let nat_td = Timedelta64::nat(Second);
        let sum = nat_td.checked_add(Timedelta64::new(1, Second)).unwrap();
        assert!(sum.is_nat());

        // np.datetime64('2024-01-01') + np.timedelta64('NaT','s') -> NaT
        let out = d.checked_add(nat_td).unwrap();
        assert!(out.is_nat());
    }
}
