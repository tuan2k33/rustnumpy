"""Generate .npy fixtures with real NumPy, used to test the compatibility
of rustnumpy's .npy reader/writer. Re-run this if you need more fixtures:

    /home/tuannq/venvs/numpy-upstream/bin/python scripts/gen_fixtures.py
"""
import numpy as np
import pathlib

out = pathlib.Path(__file__).parent.parent / "tests" / "fixtures"
out.mkdir(parents=True, exist_ok=True)

np.save(out / "vector_f64.npy", np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype="<f8"))
np.save(out / "matrix_f64.npy", np.arange(12, dtype="<f8").reshape(3, 4))
np.save(out / "cube_f64.npy", np.arange(24, dtype="<f8").reshape(2, 3, 4))
np.save(out / "scalar_f64.npy", np.array(42.0, dtype="<f8"))  # 0-D array

np.save(out / "vector_i32.npy", np.array([-2, 0, 7, 2147483647], dtype="<i4"))
np.save(out / "matrix_u8.npy", np.arange(12, dtype="u1").reshape(3, 4))
np.save(out / "vector_f32.npy", np.array([0.5, -1.25, 3.0e10], dtype="<f4"))
np.save(out / "vector_f16.npy", np.array([0.5, -1.25, 65504.0], dtype="<f2"))
np.save(out / "vector_bool.npy", np.array([True, False, True], dtype="?"))
np.save(out / "vector_c16.npy", np.array([1 + 2j, -0.5j, 3.0], dtype="<c16"))
np.save(out / "vector_c8.npy", np.array([1 + 2j, -0.5j], dtype="<c8"))
np.save(out / "matrix_f64_fortran.npy", np.asfortranarray(np.arange(12, dtype="<f8").reshape(3, 4)))
np.save(out / "vector_i4_bigendian.npy", np.array([1, -2, 300000], dtype=">i4"))
np.save(out / "empty_f64.npy", np.zeros((0, 3), dtype="<f8"))

for f in sorted(out.glob("*.npy")):
    print(f.name, f.stat().st_size, "bytes")
