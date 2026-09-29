"""Run NumPy's own test files with rnp_shim installed, then print the report.

    python run_suite.py [shadow|serve] [file ...]

shadow: every eligible call is checked against NumPy, tests see NumPy's answers.
serve : tests see rustnumpy's answers; compare pass/fail with the baseline.
"""

import os
import re
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
import numpy

SITE = os.path.dirname(os.path.dirname(numpy.__file__))

DEFAULT_FILES = [
    "numpy/_core/tests/test_umath.py",
    "numpy/_core/tests/test_ufunc.py",
    "numpy/_core/tests/test_numeric.py",
    "numpy/_core/tests/test_einsum.py",
    "numpy/_core/tests/test_function_base.py",
    "numpy/_core/tests/test_multiarray.py",
    "numpy/_core/tests/test_shape_base.py",
    "numpy/_core/tests/test_nep50_promotions.py",
    "numpy/_core/tests/test_item_selection.py",
    "numpy/_core/tests/test_regression.py",
    "numpy/_core/tests/test_indexing.py",
    "numpy/lib/tests/test_function_base.py",
    "numpy/lib/tests/test_nanfunctions.py",
    "numpy/lib/tests/test_arraysetops.py",
    "numpy/lib/tests/test_shape_base.py",
    "numpy/lib/tests/test_twodim_base.py",
    "numpy/lib/tests/test_histograms.py",
    "numpy/lib/tests/test_index_tricks.py",
    "numpy/lib/tests/test_stride_tricks.py",
    "numpy/linalg/tests/test_linalg.py",
    "numpy/linalg/tests/test_regression.py",
    "numpy/fft/tests/test_pocketfft.py",
    "numpy/fft/tests/test_helper.py",
]


# The ufunc proxy is not a real ufunc, so files that hand np.add & co. to C helpers cannot
# import with it; they run with only the function wrappers. The deselected tests probe NumPy
# internals (ufunc overrides/docstrings/loop replacement, FP status flags) that the shim
# perturbs by construction.
NO_UFUNC_PROXY = ("test_ufunc.py", "test_numeric.py", "test_multiarray.py", "test_stride_tricks.py")
ARTIFACT_TESTS = ("not test_unary_spurious_fpexception and not TestSpecialMethods and not TestReplaceLoopBySignature "
                  "and not test_get_signature and not test_params_common_gufunc and not test_load_ufunc_pickle")


def run(mode, files, log_dir, use_shim=True):
    results = {}
    for f in files:
        env = dict(os.environ, RNP_SHIM_MODE=mode, RNP_SHIM_LOG=log_dir, PYTHONPATH=HERE + os.pathsep + os.environ.get("PYTHONPATH", ""),
                   RNP_SHIM_UFUNCS="0" if f.endswith(NO_UFUNC_PROXY) else "1")
        cmd = [sys.executable, "-m", "pytest", f, "-q", "-p", "no:cacheprovider", "-n", "4", "--timeout=600", "-rfE", "-k", ARTIFACT_TESTS]
        if use_shim:
            cmd += ["-p", "rnp_shim"]
        proc = subprocess.run(cmd, cwd=SITE, env=env, capture_output=True, text=True)
        summary = [l for l in proc.stdout.splitlines() if re.search(r"\d+ (passed|failed|error)", l)]
        failed = {l.split()[1] for l in proc.stdout.splitlines() if l.startswith(("FAILED", "ERROR"))}
        results[f] = (summary[-1] if summary else proc.stdout.strip().splitlines()[-1:] or "no output", failed)
        print(f"{f:55} {results[f][0]}", flush=True)
    return results


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "shadow"
    files = sys.argv[2:] or DEFAULT_FILES
    log_dir = f"/tmp/rnp_suite_{mode}"
    shutil.rmtree(log_dir, ignore_errors=True)
    if mode == "serve":
        print("== baseline (plain NumPy) ==")
        base = run("shadow", files, "/tmp/rnp_suite_base", use_shim=False)
        print("\n== served by rustnumpy ==")
        served = run("serve", files, log_dir)
        print("\n== tests that pass on NumPy but fail when served by rustnumpy ==")
        for f in files:
            new = sorted(served[f][1] - base[f][1])
            if new:
                print(f"\n{f}: {len(new)}")
                for t in new[:40]:
                    print("   ", t.split("::", 1)[-1][:150])
    else:
        run("shadow", files, log_dir)
    subprocess.run([sys.executable, os.path.join(HERE, "report.py"), log_dir])
