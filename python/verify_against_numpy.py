"""Cross-checks for the rustnumpy_python extension, run in the same
process as real NumPy so results can be compared directly.

This is correctness verification, not a benchmark — see README.md in
this directory for why a performance comparison is deferred until the
core has more than f64 + broadcasting + two binary ufuncs.

Run after `maturin develop` (see README.md):

    /home/tuannq/venvs/numpy-upstream/bin/python python/verify_against_numpy.py
"""

import numpy as np
import rustnumpy_python as rnp


def check(label, actual, expected):
    assert actual == expected, f"{label}: {actual!r} != {expected!r}"
    print(f"  ok: {label}")


def main():
    print("-- construction & basic access --")
    a = rnp.NdArray.from_list([1.0, 2.0, 3.0, 4.0], [2, 2])
    check("a.shape", a.shape, [2, 2])
    check("a.ndim", a.ndim, 2)
    check("a.get([1, 0])", a.get([1, 0]), 3.0)

    print("\n-- add/mul with broadcasting, cross-checked against real numpy --")
    b = rnp.NdArray.from_list([10.0, 20.0, 30.0, 40.0], [2, 2])
    np_a = np.array(a.to_list()).reshape(a.shape)
    np_b = np.array(b.to_list()).reshape(b.shape)
    check("a + b", rnp.add(a, b).to_list(), (np_a + np_b).flatten().tolist())
    check("a * b", rnp.mul(a, b).to_list(), (np_a * np_b).flatten().tolist())
    check("a - b", rnp.sub(a, b).to_list(), (np_a - np_b).flatten().tolist())

    print("\n-- broadcasting a (3,) vector over a (2,3) matrix --")
    row = rnp.NdArray.from_list([0.0] * 6, [2, 3])
    vec = rnp.NdArray.from_list([1.0, 2.0, 3.0], [3])
    expected = (np.zeros((2, 3)) + np.array([1.0, 2.0, 3.0])).flatten().tolist()
    check("row + vec", rnp.add(row, vec).to_list(), expected)

    print("\n-- incompatible shapes raise a Python ValueError, not a panic --")
    try:
        rnp.add(a, vec)
        raise AssertionError("expected a ValueError")
    except ValueError as e:
        print(f"  ok: caught ValueError({e!r})")

    print("\n-- .npy round-trip across the language boundary --")
    rnp.save_npy("/tmp/rustnumpy_from_rust.npy", a)
    check(
        "numpy reads a file written by rust",
        np.load("/tmp/rustnumpy_from_rust.npy").tolist(),
        [[1.0, 2.0], [3.0, 4.0]],
    )
    np.save("/tmp/rustnumpy_from_numpy.npy", np.array([[5.0, 6.0], [7.0, 8.0]]))
    loaded = rnp.load_npy("/tmp/rustnumpy_from_numpy.npy")
    check("rust reads a file written by numpy (data)", loaded.to_list(), [5.0, 6.0, 7.0, 8.0])
    check("rust reads a file written by numpy (shape)", loaded.shape, [2, 2])

    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    main()
