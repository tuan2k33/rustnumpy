# rustnumpy

A step-by-step Rust port of NumPy's core ideas — not a drop-in replacement,
a from-scratch reimplementation built one NEP (NumPy Enhancement Proposal)
at a time. The full plan, and the reasoning behind every design choice
below, lives in [`NumPy.md`](NumPy.md).

Targets NumPy >= 2.5 semantics only; deprecated or backward-compat-only
NumPy behavior is out of scope by design.

## What's here

- A generic `NdArray<T = f64>` (shape/strides/buffer), monomorphized per
  concrete `T` at compile time — Rust's compiler-level equivalent of
  NumPy's own per-dtype `.c.src` code-generation templates.
- Manual immutable/mutable views (`ArrayView`/`ArrayViewMut`): slicing and
  broadcasting without copying data.
- `.npy` file read/write (NEP 1), cross-checked byte-for-byte against real
  NumPy's own output.
- A `DType` trait plus the NEP 50 promotion algorithm (`Kind`,
  `common_dtype`, `can_cast`), covering bool/signed/unsigned/float/complex.
- A small generic ufunc engine (broadcasting, closures, `out=`), with both
  sequential and Rayon-parallel element loops.
- A custom `Allocator` trait (NEP 49): a system allocator and a bump-arena
  implementation, audited for thread-safety (see `NumPy.md`'s step 19).
- PyO3 bindings (`python/`) so the array is callable from real Python.
- Advanced indexing (`oindex`/`vindex`, boolean masks) per NEP 21's
  never-shipped proposal.
- A packed structured/record dtype, fixed-ratio-unit `datetime64`/
  `timedelta64` with NaT semantics, `StringDType` (NEP 55) plus a
  `numpy.strings`-shaped subset of string ufuncs.
- A `numpy.testing` equivalent (`assert_array_equal`, `assert_allclose`,
  `assert_array_almost_equal`), so this project's own tests never depend
  on a running NumPy.
- Whole-array reductions/statistics, `lib`-layer utilities
  (`unique`/`concatenate`/`stack`/`split`/`interp`/`gradient`/...).
- Full `linalg` (`solve`/`inv`/`det`/`qr`/`cholesky`/`eigh`/`eigvals`/`svd`/
  norms/`matrix_power`) and `fft` (`fft`/`fftn`/`rfft`/...), delegating the
  actual numerics to the pure-Rust `faer` and `rustfft` crates.
- A NEP 19 `Generator` (`random`/`uniform`/`normal`/`gamma`/`beta`/...) on
  top of `rand_pcg`/`rand_distr`.
- `polynomial` (`Chebyshev`/`Hermite`/`Laguerre`/`Legendre` bases).
- A step 18 audit of this crate's public names against the Python Array
  API standard (NEP 56).

See the crate-level doc comment in [`src/lib.rs`](src/lib.rs) for the
authoritative, up-to-date list of what's covered and what's deliberately
still missing.

## Using it

```sh
cargo test    # 199+ tests, all self-contained (no Python/NumPy dependency)
cargo run --example step18_array_api
```

From Python, via the separate PyO3 binding crate — see
[`python/README.md`](python/README.md).

## Status

Under active, incremental development against the plan in `NumPy.md`.
Not yet published to crates.io/PyPI. No NumPy performance benchmark yet,
on purpose — see `NumPy.md`'s step 19 notes.

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or
[MIT license](LICENSE-MIT) at your option (the same dual-license
convention most of the Rust ecosystem uses).
