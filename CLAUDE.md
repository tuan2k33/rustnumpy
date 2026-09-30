# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A from-scratch, step-by-step Rust port of NumPy's core ideas, plus a standalone Python package (`import rustnumpy`) built on it. Every design decision is driven by NumPy's own NEPs, and the rationale for why the code looks the way it does lives in **[`NumPy.md`](NumPy.md)** (design, per-step logs, the NEP status table), not in code comments. Deliberate differences from NumPy are listed in **[`docs/CONVENTIONS.md`](docs/CONVENTIONS.md)**; read both before making architectural changes, and update them after.

Facts that shape everything else:

- **Target is NumPy >= 2.5 semantics only.** Deprecated or backward-compat-only NumPy behaviour is out of scope (see "NumPy Parts Worth Dropping" in `NumPy.md`). The oracle is stock NumPy 2.5.3.
- **Working rule for failing comparisons:** print and analyse the failure first. It is a bug only if NumPy's behaviour is the intended one; otherwise record it in `docs/CONVENTIONS.md`. Do not "fix" a difference just because a test fails.
- **State of the tree:** steps 1-26d of `NumPy.md` are here. Steps 27-28 (structured/datetime dtypes, StringDType) were written earlier but are not in this tree (git history: `5a3933f`, `083b9af`); step 29 (masked arrays) is not started.
- **Source comments were deliberately stripped** from every `.rs` file. Put design rationale in `NumPy.md`, not in doc comments, and match the surrounding uncommented style.
- **`rust-version = 1.85`**: clippy rejects newer std APIs (e.g. `is_multiple_of`); use the older spelling.

## Commands

Core crate (needs no Python):

```sh
cargo build
cargo test --lib                                  # all core tests (each module has #[cfg(test)] mod tests)
cargo test --lib npy::                            # one module; or one test by name
cargo clippy --all-targets -- -D warnings         # must be clean before any commit
cargo build --examples && cargo run --example step12_linalg
```

Python package (`python/` is a **separate crate**, path-dependency on the root, so core builds never need PyO3):

```sh
cd python
cargo build --release
python -m maturin develop --release               # build + install into the active venv
# quick dev loop without installing (the .so is gitignored):
cp target/release/librustnumpy_python.so python_src/rustnumpy/_core.cpython-314-x86_64-linux-gnu.so
PYTHONPATH=python_src python -m pytest tests -q -n 3     # ~11,300 tests, compares against a real NumPy in the same process
PYTHONPATH=python_src python -m pytest tests/test_lib.py -k bigint    # one case
```

The test environment needs NumPy >= 2.5, pytest, pytest-xdist (`python/requirements-dev.txt`). Run big test sessions under `ulimit -v 8000000`: an earlier allocation test exhausted WSL memory, and `createfns::alloc_guard` (MemAvailable / 2) exists for that reason.

Conformance against the official Array API suite (spec 2025.12; clone `data-apis/array-api-tests` with its submodule, plus `hypothesis pytest-json-report pytest-xdist pytest-timeout`):

```sh
ARRAY_API_TESTS_MODULE=rustnumpy ARRAY_API_TESTS_VERSION=2025.12 \
  python -m pytest array_api_tests -n 3 --max-examples=20 --hypothesis-disable-deadline -W ignore --timeout=120
```

Reference numbers (step 26d): rustnumpy 1347 passed / 29 failed, stock NumPy 2.5.3 1331 / 46; counts vary by a few between runs (Hypothesis draws). Also: `python/numpy_suite/run_suite.py` runs NumPy's own test files through a shim (see `NumPy.md`, "Step 25"); shadow mode after step 26d: 455,254 comparisons match, 4 known mismatches. `.npy` fixtures in `tests/fixtures/` come from `scripts/gen_fixtures.py` (needs NumPy).

## Architecture

### Core (`src/`)

`NdArray<T = f64>` (`ndarray.rs`) is the one container: `{ data: Vec<T>, shape, strides }`. `ArrayView`/`ArrayViewMut` (`view.rs`) borrow slices plus their own shape/strides/offset, so slicing and broadcasting never copy. Generic over the 14 dtypes through the `DType` trait (`dtype.rs`: `Kind`, `common_dtype`, `can_cast`, NEP 50).

Dependency shape: `error.rs` is the leaf (`shape.rs` only uses its `ShapeError`); `ndarray.rs` and `view.rs` are one unit (they import each other); `allocator.rs` is standalone; everything else builds on `ndarray`. `allocator.rs` is deliberately not wired into `NdArray` (decided in step 5).

- **Promotion** (`dtype.rs`) is runtime only: `common_dtype` = first dtype in a fixed order both sides `can_cast` safely (64-bit ints fall back to float64), `with_weak` is the NEP 50 weak-scalar rule, `result_type` folds both; tested against literal NumPy tables. Core kernels are same-type (`add<T>`, `*_assign<T>`, `zip_into_where<T>`); mixed dtypes are cast explicitly with `NdArray::astype::<U>()` / `Cast<U>`. `dispatch.rs` holds `*_assign`, `where=`, `reduce`/`accumulate`/`outer_with`. Integer arithmetic goes through `WrapAdd`/`WrapSub`/`WrapMul` so it wraps like NumPy in debug builds; don't use plain `+`/`*` on generic integer `T`.
- **Errors** (`error.rs`): `ShapeError` is for shape/axis/index problems only. Everything else an op can reject is `OpError` (which wraps `ShapeError` and `ReductionError`, so `?` converts). `Error` is an opt-in umbrella over all module errors. The PyO3 layer maps them to Python exceptions; a Rust panic must never cross the FFI boundary.
- **Genericity:** `ufunc`/`reductions`/`manipulation`/`npy` (via `NpyElement`)/`fft` (f32/f64, via `FftFloat`) are generic; `linalg`/`random`/`polynomial` stay `f64` (faer and LAPACK upcast anyway). `fft.rs` is two-tier on purpose: 1-D line kernels on slices, n-D functions on `NdArray`.
- `index.rs` exposes explicit `.oindex()` / `.vindex()` instead of NumPy's context-dependent `__getitem__` (NEP 21 stance). `BumpArena` is `Send` but deliberately not `Sync`.

### Python package (`python/`, `import rustnumpy`)

Standalone array library that **never imports NumPy** (`tests/test_ndarray.py` blocks the import; NumPy is only the test oracle). `rustnumpy._core` is the compiled module (`python/src/`); `python/python_src/rustnumpy/` is a Python layer over it.

- Native side: `Arr` (`dynarray.rs`) is a 14-variant enum of `NdArray<T>`; function modules convert any input to `Arr`, call the core, wrap the result. `pyarray.rs` holds the few `unsafe` blocks (SAFETY comments; the `Sync` claim assumes the GIL). `ops.rs` has `Operand` (array or weak Python int/float/complex) and NEP 50 loop resolution; `umath.rs` has every ufunc kernel and complex special values.
- Python side: `_ufunc.py` (ufunc objects with `out=`/`where=`/`reduce`...), `_reductions.py`, `_manip.py`, `_numeric.py`, `_creation.py`, `_indexing.py`, `_io.py` (`.npy`/`.npz` for every dtype), `_print.py` (port of NumPy's printing), `linalg.py`, `fft.py` (n-D composed from 1-D line transforms; norm scaling is in `linalgfns.rs`), `random.py`, `_arrayapi.py` (`__array_namespace_info__`), `_scalars.py`.
- **0-d results** for `bool`/`int64`/`float64`/`complex128` are scalar objects (`float64`/`complex128` subclass `float`/`complex`; `int64`/`bool` are plain classes, as in NumPy) with `.dtype`/`.shape`/`.astype`; every operator delegates to the 0-d array, so they are strong under NEP 50 like `np.float64`; other dtypes give 0-d arrays. Internal Python code that feeds such a value back into an op must be aware it is strong, not weak (use `float(x)` to get a weak Python number). Details in `docs/CONVENTIONS.md`.
- Matrices with a zero dimension are answered from their shapes in `linalg.py` (`_guard`); the numeric kernels never see them. SVD rescales extreme magnitudes before calling faer.

### Tests

Rust: small tolerance-based tests per module with expected values baked in as literals (checked against real NumPy when written); no live NumPy in `cargo test`. Python: differential tests against real NumPy in the same process. `tests/test_lib.py` `CASES` is the easiest place to add a regression (a lambda taking `(m, A)` run against both NumPy and rustnumpy; results and raised exception types are compared), `test_ufunc_matrix.py` covers dtype x ufunc promotion, `difftest.py` has the comparison helpers.

## Layout

```
├── NumPy.md               design doc, per-step logs, NEP status table
├── docs/CONVENTIONS.md    deliberate differences from NumPy
├── src/                   core crate (one file per NumPy area)
│   ├── ndarray, view, shape, error, dtype, dispatch, allocator
│   ├── ufunc, mathfunc, logic, reductions, sorting, selection, index, creation
│   ├── manipulation       view ops, unique*/set ops, concatenate/stack/split/tile, interp/gradient
│   ├── contraction, gufunc    matmul/dot/tensordot/einsum, NEP 20 gufuncs
│   ├── linalg, linalg_complex, fft, random, polynomial, npy
├── examples/              one runnable demo per early step
├── tests/fixtures/*.npy   written by real NumPy (scripts/gen_fixtures.py)
├── scripts/               gen_fixtures.py
└── python/                PyO3 binding crate + Python package
    ├── src/               native modules (dynarray, pyarray, pyindex, ops, umath, arrayfns, shapefns, logicfns, createfns, linalgfns, clinalgfns, ...)
    ├── python_src/rustnumpy/   Python layer (see above)
    ├── tests/             pytest suite (differential vs NumPy)
    └── numpy_suite/       shim that runs NumPy's own test files against the package
```
