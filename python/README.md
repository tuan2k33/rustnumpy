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

**No performance benchmark yet, on purpose** (see `../NumPy.md`: no
NumPy comparison until functionality is complete).

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
