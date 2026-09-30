"""Time common operations in NumPy and rustnumpy side by side.

    PYTHONPATH=python/python_src python scripts/bench_vs_numpy.py

Best of 5 repeats per case (timeit autorange); the last column is rustnumpy / NumPy.
"""

import timeit

import numpy as np
import rustnumpy as rnp


def cases(m):
    rng = np.random.default_rng(0)
    n = 1_000_000
    f64, f64b = m.asarray(rng.random(n)), m.asarray(rng.random(n))
    f32 = m.asarray(rng.random(n).astype(np.float32))
    i32, i64 = m.asarray(rng.integers(0, 100, n).astype(np.int32)), m.asarray(rng.integers(0, 100, n))
    mat, big = m.asarray(rng.random((200, 200))), m.asarray(rng.random((1000, 1000)))
    small, idx = m.asarray(rng.random(10)), m.asarray(rng.integers(0, n, 10_000))
    mask = f64 > 0.5
    s, si = m.sum(m.asarray([1.5, 2.5])), m.sum(m.asarray([1, 2]))
    return {
        "add f64+f64 (1e6)": lambda: f64 + f64b,
        "add i32+f64 mixed (1e6)": lambda: i32 + f64,
        "add f32+f64 mixed (1e6)": lambda: f32 + f64,
        "add i64+i64 (1e6)": lambda: i64 + i64,
        "mul f64*2.5 weak (1e6)": lambda: f64 * 2.5,
        "sqrt f64 (1e6)": lambda: m.sqrt(f64),
        "sum f64 (1e6)": lambda: m.sum(f64),
        "mean axis0 (1000x1000)": lambda: m.mean(big, axis=0),
        "sort f64 (1e6)": lambda: m.sort(f64),
        "matmul 200x200": lambda: mat @ mat,
        "fancy index 1e4": lambda: f64[idx],
        "bool mask": lambda: f64[mask],
        "add small (10)": lambda: small + small,
        "scalar f64 * 2.0": lambda: s * 2.0,
        "scalar int64 + 1": lambda: si + 1,
    }


def best(fn):
    timer = timeit.Timer(fn)
    number, _ = timer.autorange()
    return min(timer.repeat(5, number)) / number


def fmt(t):
    return f"{t * 1e6:9.1f} us" if t >= 1e-6 else f"{t * 1e9:9.0f} ns"


if __name__ == "__main__":
    ours, theirs = cases(rnp), cases(np)
    print(f"{'case':28}{'numpy':>13}{'rustnumpy':>13}{'ratio':>8}")
    for name in theirs:
        t_np, t_rnp = best(theirs[name]), best(ours[name])
        print(f"{name:28}{fmt(t_np):>13}{fmt(t_rnp):>13}{t_rnp / t_np:>7.1f}x")
