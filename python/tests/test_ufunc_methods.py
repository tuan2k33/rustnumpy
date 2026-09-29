import numpy as np
import pytest

from difftest import differential


def d(m, shape=(3, 4), dtype="float64", seed=0, kind="normal"):
    rng = np.random.default_rng(seed)
    base = rng.integers(-5, 20, size=shape) if kind == "int" else rng.standard_normal(shape) * 5
    return m.array(base.astype(dtype))


BINARY = ["add", "subtract", "multiply", "maximum", "minimum", "fmax", "fmin", "logical_and", "logical_or", "logical_xor",
          "bitwise_and", "bitwise_or", "bitwise_xor", "floor_divide", "hypot", "arctan2", "logaddexp", "gcd", "lcm", "power",
          "divide", "remainder", "copysign", "equal", "less"]
INT_ONLY = {"bitwise_and", "bitwise_or", "bitwise_xor", "gcd", "lcm"}


def data_for(m, name):
    if name in INT_ONLY or name in ("power", "floor_divide", "remainder"):
        base = np.random.default_rng(1).integers(1, 9, size=(3, 4))
        return m.array(base)
    return d(m)


@pytest.mark.parametrize("name", BINARY)
@pytest.mark.parametrize("axis", [0, 1, None, (0, 1), -1])
def test_reduce(name, axis):
    ok_multi = name in ("add", "multiply", "maximum", "minimum", "fmax", "fmin", "logical_and", "logical_or", "logical_xor",
                        "bitwise_and", "bitwise_or", "bitwise_xor", "gcd", "hypot", "logaddexp")
    if isinstance(axis, tuple) and not ok_multi:
        pytest.skip("not reorderable")
    if axis is None and not ok_multi:
        pytest.skip("not reorderable")
    differential(lambda m, A: getattr(m, name).reduce(data_for(m, name), axis=axis))


@pytest.mark.parametrize("name", BINARY)
@pytest.mark.parametrize("axis", [0, 1])
def test_accumulate(name, axis):
    differential(lambda m, A: getattr(m, name).accumulate(data_for(m, name), axis=axis))


@pytest.mark.parametrize("name", BINARY)
def test_outer(name):
    differential(lambda m, A: getattr(m, name).outer(data_for(m, name)[0], data_for(m, name)[1][:3]))


@pytest.mark.parametrize("name", ["add", "multiply", "maximum"])
def test_reduce_options(name):
    differential(lambda m, A: getattr(m, name).reduce(d(m), axis=1, keepdims=True))
    differential(lambda m, A: getattr(m, name).reduce(d(m), axis=0, initial=3.0))
    differential(lambda m, A: getattr(m, name).reduce(d(m), axis=1, where=A([True, False, True, True]), initial=0.0 if name == "add" else (1.0 if name == "multiply" else -1e9)))
    differential(lambda m, A: getattr(m, name).reduce(d(m, kind="int", dtype="int8"), axis=0, dtype="float32"))


def test_reduce_dtypes_and_empty():
    differential(lambda m, A: m.add.reduce(d(m, kind="int", dtype="int8"), axis=0))
    differential(lambda m, A: m.add.reduce(A([True, True, False])))
    differential(lambda m, A: m.multiply.reduce(A([], dtype="float64")))
    differential(lambda m, A: m.maximum.reduce(A([], dtype="float64")))
    differential(lambda m, A: m.add.reduce(A([[], []], dtype="float64"), axis=1))
    differential(lambda m, A: m.logical_and.reduce(A([2, 3])))


def test_reduceat_and_at():
    differential(lambda m, A: m.add.reduceat(A([1, 2, 3, 4, 5, 6, 7, 8]), [0, 4, 1, 5]))
    differential(lambda m, A: m.maximum.reduceat(d(m), [0, 2], axis=1))

    def at(m, A):
        x = A([1.0, 2.0, 3.0, 4.0])
        m.add.at(x, [0, 0, 2], 10.0)
        return x

    differential(at)

    def at2(m, A):
        x = A([[1.0, 2.0], [3.0, 4.0]])
        m.negative.at(x, ([0, 1], [1, 0]))
        return x

    differential(at2)


def test_out_where_dtype_casting():
    def f(m, A):
        out = m.zeros((3, 4))
        r = m.add(d(m), 1.0, out=out)
        return out, r is out

    differential(f)
    differential(lambda m, A: m.add(d(m), d(m), out=m.zeros((3, 4), dtype="float32"), casting="same_kind"))
    differential(lambda m, A: m.add(d(m, kind="int", dtype="int8"), 1.5, out=m.zeros((3, 4), dtype="int8")))
    differential(lambda m, A: m.add(d(m, kind="int", dtype="int8"), 1, out=m.zeros((3, 4), dtype="int64")))
    differential(lambda m, A: m.add(d(m, kind="int", dtype="int8"), 1.5, dtype="float32"))
    differential(lambda m, A: m.multiply(d(m), 2, where=A([True, False, True, False]), out=m.zeros((3, 4))))
    differential(lambda m, A: m.sqrt(d(m, kind="int", dtype="int32"), dtype="float32"))
    differential(lambda m, A: m.add(d(m, kind="int", dtype="int32"), 1, dtype="int8", casting="unsafe"))
    differential(lambda m, A: m.add(d(m, kind="int", dtype="int32"), 1, dtype="int8"))
    differential(lambda m, A: m.sin(d(m), out=m.zeros((3, 4))))
    differential(lambda m, A: m.add(d(m), d(m), m.zeros((3, 4))))

    def overlap(m, A):
        x = A(np.random.default_rng(0).standard_normal(20))
        y = m.sin(x[::-1])
        m.sin(x[::-1], out=x)
        return y, x

    differential(overlap)

    def two_out(m, A):
        q, r = m.zeros((3, 4)), m.zeros((3, 4))
        res = m.modf(d(m), out=(q, r))
        return q, r

    differential(two_out)


def test_ufunc_attributes():
    import rustnumpy as rnp

    assert rnp.add.nin == 2 and rnp.add.nout == 1 and rnp.add.identity == 0 and rnp.sin.nin == 1
    assert rnp.multiply.identity == 1 and rnp.maximum.identity is None
    assert rnp.modf.nout == 2 and rnp.divmod.nout == 2
    assert rnp.add.__name__ == "add" and "add" in repr(rnp.add)
    assert rnp.absolute is rnp.abs and rnp.remainder is rnp.mod
    with pytest.raises(TypeError):
        rnp.add(1)
    with pytest.raises(ValueError):
        rnp.sin.reduce(rnp.arange(3.0))
