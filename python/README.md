# rustnumpy-python

The Python package `rustnumpy` (steps 6, 25 and 26 of `NumPy.md`): PyO3 bindings
that give a real Python interpreter a standalone NumPy-style array library
built on the Rust core.

Kept as its own crate so the core `rustnumpy` library's `cargo
test`/`cargo clippy` never has to know PyO3 exists.

## Build and install into a venv

```sh
export VIRTUAL_ENV=/path/to/your/venv   # or just `source venv/bin/activate`
python -m pip install maturin
python -m maturin develop --release
```

This project was built and verified against the stock NumPy venv already
used elsewhere in this repo's history:

```sh
export VIRTUAL_ENV=/home/tuannq/venvs/numpy-upstream
/home/tuannq/venvs/numpy-upstream/bin/python -m maturin develop --release
```

## Verify

```sh
cd python
/home/tuannq/venvs/numpy-upstream/bin/python -m pytest tests -q
```

Every bound function is compared against real NumPy on the same input,
in the same process (`tests/`): all 169 dtype pairs for the promoted
ufuncs, weak Python scalars, edge values (`nan`, `inf`, `-0.0`, integer
extremes), every axis, error cases, and *bit-exact* equality for the
random module (`SeedSequence`, the PCG64 stream, `random`, `integers`,
`shuffle`, `permutation`, `choice`). See "Testing the Python Binding" in
`../NumPy.md` for the strategy and why NumPy's own suite needs the shim.

Performance against NumPy: `python ../scripts/bench_vs_numpy.py` (results
and the remaining gaps in `../NumPy.md`, "Step 26e").

## What's exposed

`import rustnumpy` gives a standalone NumPy-style array library. It has its
own `ndarray` and `dtype` types and **does not import NumPy** (NumPy is only
used by the test suites, as the oracle).

```python
import rustnumpy as rn

a = rn.array([[1, 2, 3], [4, 5, 6]])     # int64
b = a.T                                  # a view: shares memory with `a`
a[0, 0] = 99                             # ...so b[0, 0] is 99 too
(a @ b) / 2 + 1                          # operators, NEP 50 promotion
a[a > 4], a[::-1, ::2], a[:, [0, 2]]     # basic, boolean and fancy indexing
rn.linspace(0, 1, 5), rn.arange(10).reshape(2, 5).sum(axis=0)
rn.det(rn.eye(3) * 2)                    # linalg, fft, random: rn.fft(...), rn.default_rng(42)
```

Interop is by duck typing only: `rustnumpy.ndarray` exports the buffer
protocol and `__array_interface__`, so `numpy.asarray(a)`, `memoryview(a)`
and anything else that speaks those protocols read it **without copying**,
and `rn.array(x)` accepts numbers, nested lists/tuples, `array.array`,
`memoryview`s, NumPy arrays (through the buffer protocol) and
`__array_interface__` objects.

Unsupported dtypes (`float16`, `object`, strings...) and options raise
`rustnumpy.Unsupported` (a `NotImplementedError`); linalg errors raise
`rustnumpy.LinAlgError` (a `ValueError`). `rn.save(path, a)` / `rn.load(path)`
read and write `.npy` (float64 only). 0-d results are Python scalars for
`bool`/`int64`/`float64`/`complex128` and 0-d arrays for other dtypes. See
"Step 26" in `../NumPy.md` for the design and its limits.


## Layout and limits (step 26b)

`rustnumpy._core` is the compiled module; `rustnumpy` is a small Python package over it, with `rustnumpy.linalg`, `rustnumpy.fft` and `rustnumpy.random`. It supports 14 dtypes (`bool`, 8 integer types, `float16/32/64`, `complex64/128`), ufunc objects (`out=`, `where=`, `dtype=`, `reduce/accumulate/outer/reduceat/at`), NumPy's printing, `.npy`/`.npz` I/O, pickling and DLPack. Not supported: structured/datetime/string/object dtypes, masked arrays, `longdouble`, complex `linalg`, the legacy `np.random.*` functions and `np.matrix`. Details and the list of deliberate differences from NumPy are in `NumPy.md` ("Step 26b").


## Install

Needs a Rust toolchain (`rustup`, Rust >= 1.85) and Python >= 3.9 (developed and tested on 3.14, Linux x86_64).

```sh
python -m venv .venv && source .venv/bin/activate
pip install "maturin>=1.5,<2.0"
cd python
maturin develop --release        # editable install into the active venv
# or: maturin build --release -o dist && pip install dist/rustnumpy-*.whl
python -c "import rustnumpy as rn; print(rn.arange(6.).reshape(2, 3).sum(axis=0))"
```

The library has no Python dependencies (it never imports NumPy). It is not on PyPI.

## Tests

- `cargo test --lib` (repository root): the core's unit tests. No NumPy needed; expected values are literals checked against NumPy when written.
- `pip install -r requirements-dev.txt && python -m pytest tests -q -n 2`: differential tests that call NumPy at run time as the oracle, so NumPy is required here (and only here).
- `numpy_suite/run_suite.py`: runs NumPy's own test files through a shim; needs the NumPy install that ships its tests.
