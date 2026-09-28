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
/home/tuannq/venvs/numpy-upstream/bin/python verify_against_numpy.py
```

Runs a set of correctness checks (construction, broadcasting add/sub/mul,
error handling, `.npy` round-trips) in the *same process* as real NumPy,
so results are compared directly rather than by eyeballing two separate
outputs.

**No performance benchmark yet, on purpose.** The core (`../src/`) is
still `f64`-only with two binary ufuncs and no reductions — benchmarking
that against NumPy now would only measure "how fast is one
closure-driven loop", not "is this a viable NumPy replacement". That
comparison belongs later in `NumPy.md`'s plan, once there's enough
surface area for it to mean something.

## What's exposed

```python
import rustnumpy_python as rnp

a = rnp.NdArray.from_list([1.0, 2.0, 3.0, 4.0], [2, 2])
b = rnp.NdArray.zeros([2, 2])
a.shape       # [2, 2]
a.to_list()   # [1.0, 2.0, 3.0, 4.0]  (flat, row-major)

rnp.add(a, b)  # -> NdArray, broadcasting like NumPy
rnp.sub(a, b)
rnp.mul(a, b)

rnp.save_npy("/tmp/out.npy", a)  # readable by np.load()
rnp.load_npy("/tmp/out.npy")     # reads files written by np.save() too
```

Shape errors surface as Python `ValueError`, I/O errors as `OSError` —
not a Rust panic across the FFI boundary.
