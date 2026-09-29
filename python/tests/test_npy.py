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


def test_save_casts_other_dtypes_to_float64(tmp_path):
    path = tmp_path / "ints.npy"
    rnp.save(str(path), rnp.arange(4))
    assert np.load(path).dtype == np.float64
