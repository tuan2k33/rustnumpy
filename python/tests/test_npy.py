import numpy as np
import pytest

import rustnumpy_python as rnp


def test_ndarray_class_basics():
    a = rnp.NdArray.from_list([1.0, 2.0, 3.0, 4.0], [2, 2])
    assert a.shape == [2, 2] and a.ndim == 2 and a.get([1, 0]) == 3.0
    assert a.to_list() == [1.0, 2.0, 3.0, 4.0]
    with pytest.raises(ValueError):
        rnp.NdArray.from_list([1.0], [2, 2])


def test_npy_round_trip_in_both_directions(tmp_path):
    a = rnp.NdArray.from_list([1.0, 2.0, 3.0, 4.0], [2, 2])
    written = tmp_path / "from_rust.npy"
    rnp.save_npy(str(written), a)
    np.testing.assert_array_equal(np.load(written), [[1.0, 2.0], [3.0, 4.0]])
    source = tmp_path / "from_numpy.npy"
    np.save(source, np.array([[5.0, 6.0], [7.0, 8.0]]))
    loaded = rnp.load_npy(str(source))
    assert loaded.shape == [2, 2] and loaded.to_list() == [5.0, 6.0, 7.0, 8.0]
    with pytest.raises(OSError):
        rnp.load_npy(str(tmp_path / "missing.npy"))
