# rustnumpy-python

Step 6 of `NumPy.md`'s plan: PyO3 bindings so a real Python interpreter
can create and operate on the Rust `NdArray` from earlier steps.

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

The extension is *NumPy in, NumPy out*: every function takes anything
`numpy.asarray` accepts (or a plain Python `int`/`float`, which is a
*weak* scalar as in NEP 50) and returns a real `numpy.ndarray` (a NumPy
scalar for 0-d results). Arrays cross the boundary by value (bytes), so
this layer exists to *test* the Rust core against NumPy, not to be fast.

```python
import numpy as np, rustnumpy_python as rnp

rnp.add(np.int8([1, 2]), 3)             # int8, like NumPy
rnp.matmul(a, b); rnp.einsum("ij,jk", a, b)
rnp.inv(a); rnp.eig(a); rnp.svd(a, full_matrices=False)
rnp.fft(x); rnp.rfftn(x)
g = rnp.default_rng(42); g.random(3)    # bit-identical to np.random.default_rng(42)
```

Unsupported dtypes (`float16`, `longdouble`, `object`, strings...) and
unsupported options raise `rustnumpy_python.Unsupported`, a subclass of
`NotImplementedError`, so callers (and the NumPy-suite shim) can fall back
to NumPy. Linalg errors raise `rustnumpy_python.LinAlgError` (a
`ValueError`).

Also kept from step 6: a small `NdArray` class (`f64`, `from_list`,
`to_list`, ...) and `save_npy`/`load_npy`.
