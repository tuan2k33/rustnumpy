import numpy as np
import pytest

import rustnumpy as rnp


def test_npy_round_trip_in_both_directions(tmp_path):
    a = rnp.array([[1.0, 2.0], [3.0, 4.0]])
    written = tmp_path / "from_rust.npy"
    rnp.save(str(written), a)
    np.testing.assert_array_equal(np.load(written), [[1.0, 2.0], [3.0, 4.0]])
    source = tmp_path / "from_numpy.npy"
    np.save(source, np.array([[5.0, 6.0], [7.0, 8.0]]))
    loaded = rnp.load(str(source))
    assert loaded.shape == (2, 2) and loaded.tolist() == [[5.0, 6.0], [7.0, 8.0]]
    with pytest.raises(OSError):
        rnp.load(str(tmp_path / "missing.npy"))


@pytest.mark.parametrize("dtype", ["bool", "int8", "int32", "uint64", "float16", "float32", "float64", "complex64", "complex128"])
def test_save_keeps_the_dtype_and_numpy_reads_it(tmp_path, dtype):
    path = tmp_path / "x.npy"
    x = np.arange(12).reshape(3, 4).astype(dtype)
    rnp.save(str(path), rnp.array(x))
    back = np.load(path)
    assert back.dtype == x.dtype
    np.testing.assert_array_equal(back, x)
    np.testing.assert_array_equal(np.asarray(rnp.load(str(path))), x)


def test_fortran_ordered_and_npz_files(tmp_path):
    x = np.asfortranarray(np.arange(6.0).reshape(2, 3))
    np.save(tmp_path / "f.npy", x)
    np.testing.assert_array_equal(np.asarray(rnp.load(str(tmp_path / "f.npy"))), x)
    rnp.savez(str(tmp_path / "z.npz"), a=rnp.arange(3), b=rnp.ones((2, 2)))
    with np.load(tmp_path / "z.npz") as z:
        np.testing.assert_array_equal(z["a"], np.arange(3))
    with rnp.load(str(tmp_path / "z.npz")) as z:
        assert sorted(z.files) == ["a", "b"]
        np.testing.assert_array_equal(np.asarray(z["b"]), np.ones((2, 2)))
