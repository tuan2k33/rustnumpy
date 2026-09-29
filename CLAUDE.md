# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A from-scratch, step-by-step Rust port of NumPy's core ideas — not a
binding, not a drop-in replacement. Every design decision is driven by
NumPy's own NEPs (Enhancement Proposals), and the *entire* rationale for
why the code looks the way it does lives in **[`NumPy.md`](NumPy.md)**,
not in code comments. Read `NumPy.md` before making architectural
changes; it is the authoritative design/plan document and is kept up to
date after every change.

Two facts that shape everything else in this repo:

- **Target is NumPy >= 2.5 semantics only.** Deprecated/backward-compat-only
  NumPy behavior is out of scope by design — see `NumPy.md`'s "NumPy Parts
  Worth Dropping" section before reproducing any old quirk.
- **This is the numeric-core-only public snapshot** (`NumPy.md`'s steps
  1–17: container, views, dtype/casting, ufuncs, allocator, PyO3 binding,
  indexing, reductions, `lib`-utilities, `linalg`, `fft`,
  `random`, `polynomial`, the Array API audit, and the free-threading +
  packaging audit). Steps 27–29 (structured/datetime dtypes, StringDType,
  masked arrays) exist in `NumPy.md`'s plan and in git history, but their
  source files aren't part of this snapshot. Step 18 (core array
  mechanics: reshape, sorting, selection, math ufuncs, matmul/einsum,
  gufuncs, the strided iterator) has since landed on top of steps 1–17.

**Source comments were deliberately stripped** from every `.rs` file in
this snapshot (see the "Public snapshot" commit). Do not expect `//`,
`///`, or `//!` explanations in the code — the design rationale, NEP
citations, and "verified against real NumPy X.Y.Z" notes all live in
`NumPy.md` instead. When adding new code, put substantial design
rationale in `NumPy.md`, not in a doc comment, to stay consistent with
this snapshot's style.

## Commands

```sh
cargo build                                   # core library
cargo build --examples                        # every examples/*.rs
cargo test --lib                               # full test suite (embedded #[cfg(test)] per module)
cargo test --lib dtype::                       # one module's tests, e.g. dtype.rs
cargo test --lib can_cast_int_to_uint_is_always_unsafe   # one test by name
cargo clippy --all-targets -- -D warnings      # must be clean before any commit
cargo run --example step12_linalg              # run a specific example (see examples/ for the full list)
```

The `python/` PyO3 binding is a **separate crate** (its own `Cargo.toml`,
path-dependency on the root crate) specifically so the core crate's
`cargo build`/`cargo test`/`cargo clippy` never need PyO3 present:

```sh
cd python
cargo build                                    # builds as a plain rlib too, no Python needed
python -m maturin develop --release            # build + install into the active venv
python -m pytest tests -q                       # ~2100 tests; NumPy is only the oracle here, the package never imports it
```

There is no top-level test runner beyond `cargo test --lib` — every
module's tests live in its own `#[cfg(test)] mod tests` block at the
bottom of that file, and `.npy`-format tests read fixtures from
`tests/fixtures/*.npy` (regenerate via `scripts/gen_fixtures.py`, which
needs a real NumPy install to write them).

## Architecture

### The generic container and its dependents

`NdArray<T = f64>` (`src/ndarray.rs`) is the one core type everything
else is built on: `{ data: Vec<T>, shape: Vec<usize>, strides: Vec<isize> }`.
The default type parameter means every other module can keep writing the
bare, unqualified name `NdArray`/`ArrayView`/`ArrayViewMut` in a type
position and it resolves to `<f64>` automatically — monomorphization
does the rest, the same way NumPy's own `.c.src` templates expand per
dtype at build time, except the Rust compiler does it instead of a
codegen tool. `ArrayView`/`ArrayViewMut` (`src/view.rs`) borrow `&[T]`/
`&mut [T]` plus their own `shape`/`strides`/`offset`, so slicing and
broadcasting never copy data — Rust's borrow checker (not manual
discipline, unlike NumPy's C core) is what prevents an `ArrayView` and
an `ArrayViewMut` from aliasing the same buffer.

**Not every module is generic over `T` yet.** `ufunc.rs`/`reductions.rs`
are (bounded per-function by the relevant `std::ops`/`PartialOrd` trait,
e.g. `add<T: Copy + Add<Output = T>>`), so they work on any of the
integer/float/complex types `dtype.rs`'s `DType` trait covers. `linalg.rs`/
`fft.rs`/`random.rs`/`polynomial.rs` are still hardcoded to plain `NdArray`
(`f64`)/`NdArray<Complex64>`, matching the fact that `faer`/`rustfft` themselves
only support `f32`/`f64`/`Complex<f32/f64>`, and real NumPy's own
LAPACK/FFT bindings upcast every input to `float64` internally too.

### Module dependency shape (leaves → core → everything else)

`shape.rs` (stride/broadcast math, `IndexIter`) and `error.rs`
(`ShapeError`) are pure leaves with no `use crate::` deps. `dtype.rs`
(NEP 50: `Kind`, `DType` trait, `common_dtype`, `can_cast`) and
`allocator.rs` (NEP 49: `Allocator` trait, `System`, `BumpArena`,
`PooledVec`) are also standalone — `ufunc::add`/`sub`/`mul` do promote mixed dtypes at compile
time (step 19): `promote.rs` is *generated* from real NumPy by
`scripts/gen_promote.py` — regenerate it, never hand-edit it — and
`dispatch.rs` holds `Common<B>`, weak Python scalars, `*_assign`, `where=`
masks and `reduce`/`accumulate`/`outer_with`. Integer arithmetic uses
`WrapAdd`/`WrapSub`/`WrapMul` so it wraps like NumPy in debug builds too;
don't reintroduce plain `+`/`*` on generic integer `T`. `ndarray.rs` depends only on
`error`/`shape`/`view`. Everything above that (`ufunc`, `reductions`,
`index`, `utils`, `linalg`, `fft`, `random`, `polynomial`,
`npy`) depends on `ndarray` (and usually `view` too, for the ones that
operate on borrowed slices rather than whole owned arrays).

`index.rs` splits fancy indexing into explicit `.oindex()` (outer/
orthogonal, NumPy's actual current default) vs `.vindex()` (vectorized,
NEP 21's never-shipped proposal) rather than one method with implicit
mode-switching — this split is a deliberate divergence from real NumPy's
single, context-dependent `__getitem__`.

`allocator.rs`'s `BumpArena` is `Send` but deliberately **not** `Sync`
(an explicit `unsafe impl Send`, see the type's usage in tests) — its
`Cell<usize>` bump offset makes concurrent `&BumpArena` allocation an
actual data race, and that's load-bearing, not an oversight to "fix".

### Error handling

Every fallible public function returns a `Result` with a module-scoped
error enum (`ShapeError`, `LinalgError`, `FftError`, `RandomError`,
`ReductionError`, `NpyError`) rather than
panicking — the PyO3 layer (`python/src/lib.rs`) maps each of these to a
specific Python exception type (`ValueError`, `OSError`, ...) rather than
letting a Rust panic cross the FFI boundary.

### The Python package (`python/`, importable as `rustnumpy`)

A standalone array library, not a NumPy accessory: it has its own `ndarray`
(`Arc<Storage>` + signed element strides + offset, so slicing/transposing/
reshaping are real views) and `dtype` types, and **never imports NumPy**
(`python/tests/test_ndarray.py` proves it by blocking the import). Other
libraries reach it only through the buffer protocol / `__array_interface__`.
Function modules turn any input into the core's `Arr` (a 13-variant enum of
`NdArray<T>`), call the core, and wrap the result; `pyarray.rs` holds the
few `unsafe` blocks (each has a SAFETY comment, and the `Sync` claim assumes
the GIL). NumPy is used only by the pytest suites and `numpy_suite/` as the
oracle. Don't add `import numpy` to `python/src/`.

### Testing convention

Tests are **tolerance-based, not exact-equality**, throughout — every
formula/edge case was checked against a real NumPy install at
implementation time (see `NumPy.md` for exactly which NumPy version and
which venv), and expected values are baked into the tests as literals.
There is no live NumPy dependency in `cargo test --lib`; the only place
that talks to a real Python/NumPy process is the pytest suite in `python/tests/`
and `scripts/gen_fixtures.py`. There is no `numpy.testing` port (it was
dropped, see `NumPy.md`): each module's tests use small local helpers
(`close`, `close_all`, ...) with a per-element tolerance.

## Layout (module ↔ NumPy namespace)

```
rustnumpy/
├── NumPy.md                  ← the actual design doc / plan; read this, not code comments
├── src/
│   ├── ndarray.rs             NdArray<T>            (core container, all subpackages build on this)
│   ├── view.rs                 ArrayView/ArrayViewMut (borrowed views: numpy's non-copying slices)
│   ├── shape.rs                 strides/broadcast/IndexIter (internal, no numpy.* equivalent)
│   ├── error.rs                  ShapeError (internal)
│   ├── dtype.rs                numpy.dtype            (Kind/DType, NEP 50 promotion, can_cast)
│   ├── ufunc.rs                numpy's ufunc machinery (add/sub/mul, broadcasting, out=, rayon)
│   ├── reductions.rs           ndarray reduction methods (sum/mean/var/std/median/percentile/nan*)
│   ├── index.rs                fancy indexing         (oindex/vindex, NEP 21)
│   ├── utils.rs                numpy's lib/ layer     (unique/concatenate/stack/split/array_split/interp/gradient)
│   ├── sorting.rs              sort/argsort/searchsorted (NaN last, stable argsort)
│   ├── selection.rs            where/select/choose
│   ├── mathfunc.rs             named elementwise math (sqrt/exp/log/trig/rounding/power/...)
│   ├── contraction.rs          matmul/dot/tensordot/outer/einsum (one strided odometer engine)
│   ├── gufunc.rs               NEP 20 generalized ufuncs + vecdot
│   ├── manipulation.rs         view ops (permute_dims/moveaxis/flip/squeeze/expand_dims/unstack/broadcast_arrays), repeat/roll, unique_*
│   ├── dispatch.rs             NEP 50 promotion in ufuncs, weak scalars, *_assign, where=, reduce/accumulate/outer
│   ├── promote.rs              GENERATED promotion/cast tables (scripts/gen_promote.py)
│   ├── linalg.rs               numpy.linalg           (solve/inv/det/qr/cholesky/eigh/svd/norms), via faer
│   ├── fft.rs                  numpy.fft              (fft/ifft/rfft/irfft/fftn/...), via rustfft
│   ├── random.rs               numpy.random           (NEP 19 Generator), via rand_pcg/rand_distr
│   ├── polynomial.rs           numpy.polynomial       (Chebyshev/Hermite/Laguerre/Legendre)
│   ├── npy.rs                  .npy format            (NEP 1 read/write)
│   └── allocator.rs             NEP 49 Allocator trait (System, BumpArena, PooledVec)
├── examples/                  one runnable demo per implementation step (step1_ndarray.rs ... step19_promotion.rs; the old step9_testing was removed with numpy.testing)
├── tests/fixtures/*.npy       .npy files written by real NumPy, read back by npy.rs's tests
├── scripts/gen_fixtures.py    regenerates tests/fixtures/ (needs a real NumPy install)
└── python/                    separate PyO3 binding crate (own Cargo.toml, path-deps on root)
    ├── src/pyarray.rs           `rustnumpy.ndarray`: Arc<Storage> + shape/strides/offset, views, buffer protocol
    ├── src/{pyops,pyindex}.rs   operators/methods and __getitem__/__setitem__ on that type
    ├── src/{ops,arrayfns,shapefns,viewfns,logicfns,createfns,linalgfns,rngfns}.rs   the function surface
    ├── numpy_suite/             rnp_shim: runs NumPy's own test files against the package (shadow/serve modes)
    └── tests/                   pytest suite (~2100 cases) comparing against a real NumPy install
```
