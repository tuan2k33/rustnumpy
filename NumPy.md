# NumPy: foundations, needs, conventions per the NEPs

Sep 28, 2026 · @Kelvin Nguyen

This document summarizes NumPy Enhancement Proposals (NEPs, at numpy.org/neps) as the foundation for a project rewriting the core in Rust. Each section notes the source NEP's status (**Final** = implemented, treat as settled convention; **Accepted** = approved, being/will be implemented; **Draft**/**Open** = still under discussion, design not settled; **Deferred** = proposed but NOT adopted) — only Final/Accepted content should be treated as "agreed-upon convention" to port to Rust.

## What NumPy is (official scope)

Per the scope NEP, NumPy defines itself as: **a strided, homogeneously-typed (homogeneous dtype), in-memory, CPU-resident N-dimensional array library**. Three important boundaries:

- **Not targeting GPU/distributed** — that's CuPy, Dask, JAX's job; NumPy provides standard APIs/semantics (Array API, `__array_function__`, `__array_ufunc__`) so those libraries can be compatible, rather than building a multi-device backend itself.
- **Strided arrays, not sparse/ragged** — every element shares the same dtype, laid out regularly via strides; ragged (uneven) or sparse data is out of scope for the core.
- **Not a deep-learning framework** — no autograd, no lazy computation graph — that's PyTorch/JAX's job, built in the spirit of API compatibility with NumPy.

For the Rust project: this is a boundary worth keeping in the early stages — a CPU-only, strided, homogeneous-dtype core `ndarray` — before considering GPU or lazy evaluation.

## What NumPy needs (current roadmap)

Official roadmap priorities (numpy.org/neps/roadmap):

- **Array API interoperability** (NEP 56, Final) — standardize the namespace/semantics so code is portable across NumPy/CuPy/PyTorch/JAX.
- **SIMD & the move to C++** — gradually replacing C macros with universal intrinsics (NEP 38) and C++ templates to make extending dtypes/kernels easier.
- **Dtype extensibility** — a new DType system (NEP 41/42/43) letting third parties define dtypes (e.g. units, categorical) without modifying NumPy core.
- **Free-threaded CPython** — preparing for a GIL-free Python (PEP 703), requiring a review of thread-safety for every piece of global state in the C core.
- **Reducing binary size / build time** — since the C core has grown large over 15 accumulated years.

These are exactly the "technical debts" NumPy itself acknowledges — and also the points where a Rust rewrite has a real chance to improve things (in particular: dtype extensibility via traits instead of macros, and thread safety "for free" via the borrow checker instead of manual auditing).

## The ndarray memory model

Every `ndarray` consists of: a pointer to the raw data buffer, `shape` (a tuple of per-dimension sizes), `strides` (the number of bytes to jump to reach the next element along each axis), and `dtype`. No copy on slicing — a view only changes `shape`/`strides`/offset on the same buffer.

- **C-contiguous** (row-major, default): the trailing axis has the smallest stride. **F-contiguous** (Fortran/column-major): the leading axis has the smallest stride. Both flags exist for compatibility with BLAS/LAPACK and legacy Fortran code.
- **Broadcasting**: two arrays are compatible if, compared from the trailing axis, each pair of sizes is equal or one of them is 1 (a missing axis is treated as 1). A size-1 axis is "stretched" via stride 0, with no data copy.

**Rust mapping**: the equivalent struct needs `Vec<isize>` for shape/strides, a raw pointer or `Arc<[u8]>` for the buffer (to share views safely), and it must resolve the conflict between Rust's `&mut` uniqueness and NumPy allowing multiple aliasing views on the same buffer (e.g. the `out=` parameter) — this is the hardest design point when porting; Rust's `ndarray` crate has already solved it with lifetimes + separate `ArrayView`/`ArrayViewMut` types.

## The new DType system (NEP 41/42/43)

*Status: NEP 41/42 Accepted (mostly implemented since NumPy 1.21+), NEP 43 still Draft/Open — the ufunc-extensibility details aren't fully settled.*

Previously dtype was a hard-coded list in C. The new design turns each dtype into **a real Python/C class**:

- **DType** is a subclass of `np.dtype`; **DTypeMeta** is the metaclass defining how a DType is created. An instance of a DType (e.g. a concrete `float64` with metadata like byte order) is a "descriptor".
- **Abstract DType** (e.g. `Integer`) can't be instantiated directly, used to group concrete DTypes into a hierarchy.
- **CastingImpl / ArrayMethod**: each pair (source DType, destination DType) declares a cast object with 3 functions: `resolve_descriptors` (computes the output descriptor from the inputs), `get_loop` (selects the appropriate strided-loop function), `strided_loop` (executes on the actual data buffer). This same pattern is used for both casting and ufunc dispatch — unifying two previously separate mechanisms.

**Rust mapping**: this pattern naturally fits a **trait**: `trait DType { fn resolve_descriptors(...); }`, a concrete dtype is a struct implementing the trait, an abstract DType is a parent/marker trait. The central design question: dynamic dispatch via `dyn DType` (flexible like Python, but slower) vs. generic `<D: DType>` (fast, monomorphized, but harder to represent a runtime array with mixed dtypes) — NumPy's C core must use dynamic dispatch because Python relies on runtime types; Rust can choose a hybrid: a closed enum for built-in dtypes (fast) + `dyn DType` for third-party extensions.

## Promotion rules (NEP 50)

NEP 50 (applied since NumPy 2.0) fundamentally changes how NumPy mixes types:

- Python `int`/`float`/`complex` are **"weak" abstract DTypes** — they have no fixed size/precision, only taking on a concrete type when they collide with another NumPy array/scalar.
- **Removes value-based casting** from the old behavior: previously `np.float32(3) + 3.0` gave a different result depending on the *value* of the Python number; NEP 50 removes this unpredictability — NumPy arrays and NumPy scalars (including 0-D) always behave like real N-dimensional arrays, never "downgrading" to the other operand's precision.
- **Kind ordering**: `boolean < integral < inexact` (float/complex). Within the same kind, compare bit width.

**Rust mapping**: this is a concrete, well-defined algorithm, a great fit for a direct port — a good exercise for learning enum + pattern-matching design in Rust (`enum Kind { Bool, Int(u8), Float(u8), Complex(u8) }` + a `common_dtype` function).

## Casting rules

Every pair of dtypes has a **casting safety level**, ranked from safest to loosest:

- **equivalent** — a bit-preserving cast (e.g. changing byte order within the same dtype).
- **safe** — no information loss (int8 → int32).
- **same_kind** — same group but may lose precision (float64 → float32).
- **unsafe** — may lose information or change kind (float → int).

Each level also carries an extra **"+view"** flag — signaling the cast can be done via a view (no buffer copy) instead of an actual conversion.

Promotion (finding the common type when mixing two operands) relies on two hooks: `__common_dtype__` (finds the common DType between two DType *classes*) and `__common_instance__` (finds the common concrete instance/descriptor, e.g. a common byte order between two instances of the same DType).

**Rust mapping**: `enum CastSafety { Equivalent, Safe, SameKind, Unsafe }` + a `can_cast(from, to) -> CastSafety` method on the `DType` trait. Clearly separating the 4 safety levels is worth keeping — it lets a `.cast::<T>()` API accept a minimum required safety level, instead of every call site deciding on its own.

## Ufunc & generalized ufunc (NEP 5/20)

**Ufunc** (universal function, e.g. `np.add`) is an element-wise function that automatically broadcasts, supporting `out=`, `where=`, reduce/accumulate/outer.

**Generalized ufunc (gufunc)** extends this to functions operating on *sub-arrays* rather than scalars — e.g. `matmul`, `inv`. The signature declares **core dimensions** (axes belonging to each call's input/output) separately from **loop dimensions** (axes broadcast normally):

```
(i,j),(j)->(i)   # matrix-vector: matrix (i,j), vector (j) → vector (i)
```

The letters `i`, `j` are symbolic core-axis names; every axis before them is broadcast as a normal loop dimension.

Extending ufuncs to new dtypes (NEP 43, **still Draft**) reuses the exact same `resolve_descriptors`/`get_loop`/`strided_loop` pattern from the DType system — meaning this detail **isn't fully stable yet**, so treat it as a direction rather than a hard spec when designing the Rust side.

**Rust mapping**: a signature like `(i,j),(j)->(i)` could be represented with const generics or with a runtime shape-checked type (the way the `ndarray` crate does with `Ix1`, `Ix2`, ...). This is a good exercise for learning trait objects + generic dispatch in Rust.

## Iterator design (NEP 10)

`NpyIter` is the central iteration engine used by most ufuncs/reductions. Three main ideas:

- **Cache-coherent output layout selection**: the iterator picks the axis traversal order that makes memory access as sequential as possible (not necessarily strict C or F order), reducing cache misses.
- **Dimension coalescing**: memory-adjacent axes (with compatible strides) are merged into a single loop to reduce loop overhead.
- **Buffering + casting while iterating**: if the input dtype doesn't match the loop's expected dtype, the iterator automatically casts small buffers on the fly during iteration, avoiding casting the whole array upfront.

**Rust mapping**: this part genuinely requires algorithmic work (not just wrapping types) — rewriting it would be a good exercise in designing a generic iterator over multiple arrays at once (`itertools::multizip` is the closest idea, but NumPy also auto-optimizes axis order — no Rust crate fully does this yet).

## Indexing semantics (and NEP 21 - Deferred)

NumPy has two fundamentally different kinds of indexing:

- **Basic indexing** (slice, int, `...`, `None`) — always returns a **view** (no copy).
- **Advanced indexing** (integer arrays, boolean arrays) — always returns a **copy**. When mixing multiple non-adjacent advanced index arrays, the result broadcasts under its own rule that makes the result's axes "jump" to the front — one of the most confusing parts of NumPy.

**NEP 21 (Deferred — NOT adopted)** once proposed adding `arr.oindex[...]` (outer/orthogonal indexing — each index array applies independently to its own axis, like MATLAB/Fortran) and `arr.vindex[...]` (explicit vectorized indexing, equivalent to the current advanced-indexing behavior but more transparent) to untangle the ambiguity above. This proposal was never officially adopted — treat it only as historical reference, not current convention.

**Rust mapping**: since Rust can cleanly separate its API from the start (unconstrained by Python's backward compatibility), this is a chance to design **better** than original NumPy: provide explicit `.oindex()`/`.vindex()` as NEP 21 proposed but never delivered, instead of one overloaded `[]` operator.

## 0-D arrays vs. array scalars (NEP 27)

NumPy has **two** ways to represent "a single number": a 0-dimensional array (`np.array(5)`, with `.shape == ()`) and an array scalar (`np.int64(5)`, not an array). The historical/design reasons per NEP 27:

- An array scalar **inherits from the corresponding native Python type** (`np.float64` inherits from `float`), so it's implicitly compatible with Python code that expects a plain number (e.g. used as a dict key, compared with `isinstance(x, float)`).
- A 0-D array does **not** inherit that way, but keeps the full array API (can be `.reshape()`d, is the natural result of a reduction like `arr.sum()` with no `axis` specified).
- An array scalar is **immutable**, a 0-D array can be mutable.

**Rust mapping**: Rust doesn't need this distinction — it's a consequence of Python lacking real overloading/generics, forcing NumPy to "simulate" two worlds. In Rust, `T` (scalar) and `Array0<T>` can be cleanly separated by the type system without fake inheritance — a point worth **simplifying** rather than copying as-is.

## RNG architecture (NEP 19)

Since NumPy 1.17, the RNG architecture is split into two layers:

- **BitGenerator** — generates a raw random bit stream (a specific algorithm: the new default `PCG64`, or `MT19937`, `Philox`, `SFC64`, ...). The algorithm can be swapped without changing the distribution logic.
- **Generator** — wraps a BitGenerator, providing distribution methods (`.normal()`, `.uniform()`, `.choice()`, ...), converting raw bits into values following the desired distribution.

**RandomState** (the old API, `np.random.seed()`/`np.random.rand()`) is **permanently frozen** to guarantee reproducibility for old code — it receives no new algorithm improvements, only critical bug fixes.

**Rust mapping**: the BitGenerator/Generator split maps very naturally onto the Rust `rand` ecosystem (`trait RngCore` = BitGenerator, `Distribution<T>` = the Generator part) — it can be reused directly instead of rewritten from scratch, just make sure `PCG64` produces identical output to NumPy's if stream compatibility is needed.

## Custom memory allocator (NEP 49)

The `PyDataMem_Handler` C API lets you replace the array data's default allocator with a custom set of functions: `malloc`, `calloc`, `realloc`, and notably a **`free` that receives a size argument** (unlike standard C `free()`, which doesn't need a size) — helping an allocator track memory or use a pool more efficiently. Used to integrate GPU pinned memory, a tracing allocator, or a custom pool allocator.

**Rust mapping**: this is almost a **direct match** for Rust's `Allocator` trait (`allocator_api`, currently unstable but with a stable layout): `alloc`, `dealloc` (which also receives a `Layout`, i.e. both size and alignment, stricter than NEP 49 even). This is one of the clearest pieces of evidence that Rust's allocator design "arrived later but arrived right" in the direction NumPy had already set for itself — so the `Allocator` trait should be leveraged instead of building a separate mechanism.

## The .npy/.npz file format (NEP 1)

Structure of a `.npy` file:

1. **Magic string**, 6 bytes: `\x93NUMPY`.
2. **Version**, 2 bytes (major.minor, currently 1.0/2.0/3.0 depending on header size).
3. **Header length** (2 or 4 bytes depending on version) + **header dict** as a Python literal string (`{'descr': '<f8', 'fortran_order': False, 'shape': (3, 4), }`) declaring the dtype, memory order, and shape.
4. **Padding** so the whole header (magic+version+len+dict) aligns to a **multiple of 64 bytes**, helping the data that follows align well for SIMD/mmap.
5. **Raw data**, laid out exactly as declared in the header.

`.npz` is just a ZIP file containing multiple `.npy` files (like a vector of them).

**Rust mapping**: this is an **ideal starting exercise** for the project — a small format with a clear spec, immediately verifiable results (read a real NumPy file), teaching binary parsing, `serde`-style (de)serialization, and handling alignment/endianness in Rust without having to worry about the whole complex dtype system first.

## Backward-compatibility policy (NEP 23)

NumPy has an official process for breaking changes:

- **DeprecationWarning** when a feature starts being phased out — behavior stays the same, only a warning is added.
- **VisibleDeprecationWarning** for cases where even ordinary users (not just developers who configure warning filters) should see the warning.
- At least **one release version** between deprecation and when the change actually takes effect (often longer in practice).
- Breaking-change decisions go through community discussion + a dedicated NEP if the impact is large.

**Rust mapping**: not a technical convention to port code, but a **process lesson** worth applying to your own Rust project — especially since Rust has a `#[deprecated]` attribute equivalent to `DeprecationWarning`, and semver + Cargo make breaking changes much clearer than Python's versioning system.

## API cleanup for NumPy 2.0 (NEP 52)

Principles for reorganizing the public namespace:

- **Clear public/private split**: every public API must be in `__all__`; anything not meant for external use is renamed `_private`.
- **One-location-per-function**: each function has only **one** official import path (previously the same function could be reached via multiple aliases, confusing the documentation).
- **Namespace tiering**: separating `numpy` (core, highly stable) from specialized namespaces (`numpy.strings`, `numpy.exceptions`, ...).
- **Lazy submodule loading**: a submodule is only imported when actually used, reducing `import numpy` time and avoiding circular imports.

**Rust mapping**: Rust already has `pub`/`pub(crate)` and a module system much clearer than Python's — the "one location per function" principle should be the **default from day one**, not something fixed later the way NumPy had to after 15 years.

## Banning dtype=object inference for ragged sequences (NEP 34)

Previously, `np.array([[1, 2], [3, 4, 5]])` (unevenly-sized sublists, "ragged") silently created a `dtype=object` array along with a `VisibleDeprecationWarning`. NEP 34 requires the user to **explicitly declare** `dtype=object` if they want this behavior — implicit inference is no longer allowed, avoiding silent logic errors when the user actually wanted a regular array but mistyped a shape.

**Rust mapping**: this reinforces the boundary already noted in the "What NumPy is" section — ragged data is NOT part of the core's scope. In Rust, `Vec<Vec<T>>` already naturally represents ragged data, completely separate from the regular `Array<T, D>` type — the type system automatically prevents this error without needing a runtime warning the way Python has to.

## SIMD universal intrinsics (NEP 38)

Instead of writing separate kernels for each CPU architecture (SSE, AVX2, AVX-512, NEON, ...), NumPy defines a **common SIMD abstraction layer** (`npyv_*` intrinsics) implemented underneath by each concrete backend. Two important build flags:

- `--cpu-baseline`: the set of SIMD features **always assumed present** on any machine running the binary (fixed at build time).
- `--cpu-dispatch`: the set of features compiled into multiple versions, **selected at runtime** based on the actual CPU (runtime CPU detection/dispatch), trading a larger binary for optimal performance across many machine types.

**Rust mapping**: the `std::arch` crate provides similarly-shaped intrinsics, and `#[target_feature]` + runtime detection (`is_x86_feature_detected!`) achieve the equivalent of `--cpu-dispatch`. This is the part requiring the most effort to match NumPy's performance — the `wide` or `pulp` crates could help abstract this away instead of writing `std::arch` entirely by hand.

## The new StringDType (NEP 55)

The `np.dtypes.StringDType` dtype (NumPy 2.0+) gradually replaces `dtype='U'`/`dtype=object` for strings:

- **Variable-length UTF-8 representation** instead of the old `'U'`'s fixed-length UCS4 (which wastes a huge amount of memory for ASCII strings).
- **Small-string optimization**: short strings are stored inline directly in the descriptor, no heap allocation needed.
- **Arena allocator** for long strings — pooling allocations into a shared memory region instead of `malloc`ing each string separately.
- **Missing-data sentinel**: supports a value representing "no data" (similar to NA) directly in the dtype.
- **Thread safety via a per-descriptor mutex**: since the arena is shared, each descriptor holds its own mutex to synchronize concurrent access.

**Rust mapping**: this is one of the areas where Rust has the most natural advantage — `String`/`&str` have always been UTF-8, small-string optimization is available via crates like `smartstring`/`compact_str`, and thread safety can be guaranteed statically (at compile time via `Send`/`Sync`) instead of only relying on a runtime mutex the way C does.

## The Python Array API standard (NEP 56)

NumPy 2.0 adopts the **Python Array API** standard (specification v2022.12, defined by a cross-library consortium) into the main `numpy` namespace: function names, signatures, and broadcasting/dtype-promotion semantics are standardized so code written for NumPy also runs (or is easy to port) on CuPy, PyTorch, JAX, Dask through the same common API bindings.

**Rust mapping**: if the project's long-term goal is to interoperate with the Python ecosystem (via PyO3), following the Array API standard's function names/signatures (rather than inventing Rust-idiomatic names from scratch) will make it easier for people used to NumPy to switch over, and give the API design a reference point from day one instead of having to think it through from nothing.

## Index of NEPs read and their status

| NEP | Topic | Status |
| --- | --- | --- |
| 1 | .npy file format | Final |
| 5 | Generalized universal function API | Final |
| 10 | Iterator (NpyIter) | Final |
| 19 | New RNG architecture | Final |
| 20 | Extending the gufunc signature | Final |
| 21 | oindex/vindex indexing | **Deferred (not adopted)** |
| 23 | Backward-compatibility policy | Final |
| 27 | 0-D array vs. array scalar | Final |
| 34 | Banning dtype=object inference for ragged data | Final |
| 38 | Universal SIMD intrinsics | Final |
| 41 | New DType system (foundation) | Accepted |
| 42 | New DType system (API details) | Accepted |
| 43 | Extending ufunc for new DTypes | **Draft/Open** |
| 49 | Custom memory allocator C API | Final |
| 50 | Weak scalar promotion | Final |
| 52 | API cleanup for NumPy 2.0 | Final |
| 55 | New StringDType | Final |
| 56 | Python Array API standard alignment | Final |

Note: only NEP 41/42/43 are still being shaped (43 is still Draft); NEP 21 is a historical proposal that was rejected — both groups should be treated as direction/reference, not settled convention like the other NEPs.

## Mapping onto the Rust design

| NumPy convention | Rust idea | Note |
| --- | --- | --- |
| DType-as-class (NEP 41/42) | `trait DType` + struct/enum implementing it | Weigh `dyn DType` (flexible) vs. generic (fast) |
| Weak scalar promotion (NEP 50) | `enum Kind` + a `common_dtype` function | A clear, directly portable algorithm |
| Casting safety levels | `enum CastSafety` on the `DType` trait | The 4 safety levels are worth keeping as-is |
| Custom allocator (NEP 49) | Rust's `Allocator` trait | Nearly a direct match |
| .npy format (NEP 1) | A simple binary parser | An ideal starting exercise |
| RNG BitGenerator/Generator (NEP 19) | `rand` crate: `RngCore` + `Distribution<T>` | Reusable instead of rewritten |
| StringDType (NEP 55) | `String`/`&str` + `smartstring`/`compact_str` | Rust has a built-in advantage |
| SIMD universal intrinsics (NEP 38) | `std::arch` + `#[target_feature]` + `wide`/`pulp` crate | The most effort-intensive part to match NumPy's performance |
| oindex/vindex (NEP 21, Deferred) | Explicit `.oindex()`/`.vindex()` API | A chance to design better than the original |
| Parallelization (outside the NEPs, a roadmap direction) | The `rayon` crate | NumPy is mostly single-threaded; a real opportunity to improve |
| Python binding (outside NEP scope) | `PyO3` + `maturin` | So Python can call into the Rust core |

**Prior art worth studying** when designing (not necessarily reusing): `ndarray` (general strided arrays, closest to NumPy), `nalgebra` (linear algebra with compile-time sizes), `faer` (high-performance pure-Rust linear algebra), `candle` (tensors for ML, with a GPU backend).

## Missing data / numpy.ma (note, low priority)

`numpy.ma` is its own subclass of `ndarray`, not integrated into dtype or ufunc dispatch — leading to poor performance and inconsistent semantics. There was once a proposal to move missing-data support into the dtype layer, but it stalled due to design disagreements (the specific NEP number wasn't re-verified during this research session).

**Design decision for the Rust version (whenever it gets built)**: prefer extending the **container** rather than creating a separate dtype for "value may be missing":

- Add a `validity: Option<Bitmap>` field to `NdArray<T>` — separating "what a `T` value looks like" (dtype) from "does the value exist" (container) — similar to the design Arrow/Polars/DataFusion have already proven at production scale.
- Avoid turning `Nullable<T>` into a separate DType per type (`Nullable<i32>`, `Nullable<f64>`, ...) since that would duplicate the entire casting-rule/ufunc-dispatch machinery for every combination.
- `validity: None` = zero runtime cost; the kernel skips the bitmap check entirely.
- A NaN-style sentinel is still reasonable as a float-specific optimization, but shouldn't be the general architecture since it doesn't generalize to int/string/bool.

Not needed in the early stage (basic `NdArray` struct) — but the `validity` field should be planned into the struct design from the start to avoid a major refactor later.

## Real source → Rust module mapping

Based on the actual structure map of NumPy's source (871 files: 190 Python, 653 C/C++, 16 Cython, 12 vendored) — grouped by capability layer, not by the original directory tree:

| Rust module | Real NumPy source | Note |
| --- | --- | --- |
| `core/` | `_core/include/numpy/ndarraytypes.h` (`PyArrayObject` struct), `multiarray/arrayobject.c`, `alloc.c`, `ctors.c`, `iterators.c` | Convert — the core of the design |
| `dtype/` | `multiarray/descriptor.c`, `dtypemeta.c`, `arraytypes.c.src`, `scalartypes.c.src`, `_core/include/numpy/dtype_api.h` | Convert — the trait-learning centerpiece |
| `cast/` | `multiarray/convert_datatype.c`, `dtype_transfer.c` | Convert — the NEP 50 algorithm |
| `ufunc/` | `umath/ufunc_object.c`, `ufunc_type_resolution.c`, `loops.c.src` | Convert |
| `simd/` | `_core/src/_simd/`, `common/simd/{sse,avx2,avx512,neon,vec,lsx}/`, `_core/src/highway/` | Convert partially — use `std::arch`/the `pulp` crate instead of hand-writing 6 separate intrinsic sets like C does |
| `sort/` | `_core/src/npysort/` (excluding `x86-simd-sort/`) | Convert the algorithmic part, use standard `sort_unstable` for the basic case |
| `math/` | `_core/src/npymath/` | Mostly already available in `std`/the `half` crate, only convert what's missing |
| `iter/` | `multiarray/nditer_*.c` (the real NpyIter) | Convert |
| `io/` | `lib/_format_impl.py` | Convert — matches NEP 1 |
| `random/` | `random/*.pyx` (bit_generator, _pcg64, _generator, mtrand), `random/src/pcg64/`, `philox/` | Convert the wrapper, use the `rand`/`rand_pcg` crates for the core |
| `ma/` | `ma/core.py` (18 classes, 94 functions) | Not a 1:1 port — condensed into a `validity` field (see the separate note above) |
| **BIND, don't convert** |  |  |
| `linalg/lapack_lite/` | F2C-translated Fortran LAPACK/BLAS — machine-generated code, not hand-written | FFI-bind to OpenBLAS/MKL, or use the `faer` crate |
| `fft/pocketfft` | A separate FFT library by another author (Martin Reinecke), vendored as-is | Use the `rustfft` crate, or FFI-bind |
| `npysort/x86-simd-sort/` | Intel's vendored x86-simd-sort library, ultra-specialized SIMD kernels per microarchitecture | Use standard `sort_unstable`, only bind if the exact performance is truly needed |

**Convert vs. bind principle:**

| Convert when... | Bind when... |
| --- | --- |
| The code is NumPy's *design* (dtype system, iterator, ufunc dispatch) — the ideas are worth learning | The code is a numerical library proven over decades (LAPACK, FFT) |
| It's readable and has learning value for algorithms/OOP | The code is machine-generated (F2C) or an ultra-specialized SIMD kernel with nothing to learn from reading it |
| Rewriting it is a reasonable scope (hundreds to thousands of lines) | Rewriting it properly means building an entirely separate project taking years |

In short: convert the parts that are "NumPy's design ideas", bind the parts that are "numerical tools NumPy just borrows".

**Current decision**: FFI-bind directly to LAPACK/BLAS and pocketfft/x86-simd-sort (not using pure-Rust `faer`/`rustfft`/`sort_unstable` for these) — prioritizing implementation speed first, can switch to pure Rust later if needed.

**If later switching to pure Rust (`faer`/`rustfft`/`sort_unstable`), note the following behavioral differences (not mathematically wrong, but results may not be identical):**

- **Sort**: `f64`/`f32` don't implement `Ord` because of NaN — must use `sort_unstable_by(|a,b| a.total_cmp(b))` to match NumPy's behavior of pushing NaN to the end. Neither is stable (NumPy's default `quicksort`/introsort vs. Rust's pattern-defeating quicksort), so the order of equal elements may differ — use `sort()` (stable) if matching NumPy's `kind='stable'` is needed.
- **FFT**: `rustfft` and `pocketfft` are both mathematically correct but not bit-for-bit identical (floating-point addition order differs → ULP-level error).
- **Linear algebra (`faer` vs. LAPACK)**: the clearest difference — eigenvector/singular-vector signs can flip (both are mathematically correct), degenerate (repeated) eigenvalues/singular values may come out in a different order/corresponding subspace, degenerate-matrix error handling differs in API style (LAPACK's `info` code vs. faer's Rust-style `Result`/panic), and multi-threaded BLAS can produce non-deterministic results across runs at the ULP level.
- **Consequence for testing**: when comparing results against NumPy, use a tolerance (`atol`/`rtol`, like `numpy.allclose`) instead of an exact comparison, and normalize signs before comparing eigenvectors/SVD.

## Next steps

Proposed order of learning/implementation, easy to hard, each step producing immediately verifiable code:

1. **Basic `NdArray` struct** — shape/strides/buffer for one fixed dtype (e.g. `f64`), try manual slicing/view/broadcasting. Learn ownership, lifetimes, `&`/`&mut`.
2. **Read/write the .npy format** (NEP 1) — read a real NumPy-exported file, compare the result directly with Python. Learn binary parsing, error handling (`Result`/`?`).
3. **Basic DType via a trait** — implement 3-4 dtypes (`i32`, `f32`, `f64`, `bool`) via an `enum` or `trait`, try the NEP 50 promotion algorithm. Learn traits, generics, pattern matching.
4. **A simple ufunc** — element-wise `add`/`mul` with broadcasting, compare results with real NumPy. Learn iterator design, closures.
5. **Custom allocator** (NEP 49 → the `Allocator` trait) — try a simple pool allocator. Learn controlled unsafe Rust.
6. **PyO3 binding** — call the Rust array from Python, compare performance against NumPy on a real workload.
7. **Parallelization with `rayon`** — this is the point with a genuine chance to beat real NumPy (original NumPy is mostly single-threaded outside of BLAS).

Each step should pause to write a benchmark comparing against NumPy — both to track performance progress and to reinforce understanding of the real cost of each design decision.
