"""Sinh fixture .npy bằng NumPy thật, dùng để test tính tương thích của
bộ đọc/ghi .npy trong rustnumpy. Chạy lại nếu cần tạo thêm fixture:

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

for f in sorted(out.glob("*.npy")):
    print(f.name, f.stat().st_size, "bytes")
