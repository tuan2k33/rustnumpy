# NumPy: Foundations, Needs, and NEP Conventions

Sep 28, 2026 · @Kelvin Nguyen

This document synthesizes NumPy Enhancement Proposals (NEPs, at numpy.org/neps) as the foundation for a project to rewrite the core in Rust. Each entry notes the status of the source NEP (**Final** = implemented, treat as settled convention; **Accepted** = approved, being/to be implemented; **Draft**/**Open** = still under discussion, design not settled; **Deferred** = once proposed but NOT adopted) — only Final/Accepted content should be treated as "agreed-upon convention" to port to Rust.

## What NumPy Is (Official Scope)

According to the scope NEP, NumPy defines itself as: an N-dimensional strided array library, with a homogeneous dtype, in-memory, CPU-based. Three important boundaries:

- **Not GPU/distributed-targeted** — that's the job of CuPy, Dask, JAX; NumPy provides standard APIs/semantics (Array API, `__array_function__`, `__array_ufunc__`) so those libraries can interoperate, rather than building a multi-device backend itself.
- **Strided arrays, not sparse/ragged** — every element shares the same dtype, with a regular layout via strides; irregular (ragged) or sparse data falls outside the core's scope.
- **Not a deep-learning framework** — no automatic differentiation (autograd), no lazy graph — that's the job of PyTorch/JAX, built in the spirit of API compatibility with NumPy.

For the Rust project: this boundary should be kept as-is in the early stage — a CPU-core `ndarray`, strided, homogeneous dtype — before considering GPU or lazy evaluation. Confirmed: the project currently prioritizes CPU only; GPU support is out of scope for now.

## What NumPy Needs (Current Roadmap)

The official roadmap priorities (numpy.org/neps/roadmap):

- **Array API interoperability** (NEP 56, Final) — standardizing namespace/semantics so code is portable across NumPy/CuPy/PyTorch/JAX.
- **SIMD & migration to C++** — gradually replacing C macros with universal intrinsics (NEP 38) and C++ templates to make dtype/kernel extension easier.
- **Dtype extensibility** — the new DType system (NEP 41/42/43) lets third parties define new dtypes (e.g. units, categorical) without modifying NumPy core.
- **Free-threaded CPython** — preparing for GIL-less Python (PEP 703), which requires auditing thread-safety across every global state in the C core.
- **Reducing binary size / build time** — because the C core has grown bloated over 15 accumulated years.

These are exactly the "technical debts" NumPy itself acknowledges — and also the points where a Rust rewrite has a real chance to improve things (especially: dtype extensibility via traits instead of macros, and thread safety "for free" thanks to the borrow checker instead of manual auditing).

## The ndarray Memory Model

Every `ndarray` consists of: a pointer to the raw data buffer, `shape` (a tuple of sizes per dimension), `strides` (the number of bytes to jump to the next element along each axis), and `dtype`. No copy happens on slicing — a view just changes `shape`/`strides`/offset over the same buffer.

- **C-contiguous** (row-major, the default): the last axis has the smallest stride. **F-contiguous** (Fortran/column-major): the first axis has the smallest stride. Both exist to stay compatible with BLAS/LAPACK and legacy Fortran code.
- **Broadcasting**: two arrays are compatible if, compared from the last axis, each pair of sizes is either equal or one of them is 1 (a missing axis counts as 1). A size-1 axis is "stretched" using a stride of 0, without copying data.

**Rust mapping**: an equivalent struct needs `Vec<isize>` for shape/strides, a raw pointer or `Arc<[u8]>` for the buffer (to share views safely), and must resolve the conflict between Rust's `&mut` uniqueness and NumPy allowing multiple aliased views on the same buffer (e.g. the `out=` param) — this is the hardest design point when porting; Rust's `ndarray` crate resolves it via lifetimes + separate `ArrayView`/`ArrayViewMut` types.

## The New DType System (NEP 41/42/43)

*Status: NEP 41/42 Accepted (largely implemented since NumPy 1.21+), NEP 43 is still Draft/Open — the ufunc-extensibility details aren't fully settled yet.*

Previously, dtype was a hardcoded list in C. The new design turns each dtype into a real Python/C class:

- **DType** is a subclass of `np.dtype`; **DTypeMeta** is the metaclass that defines how DTypes are created. An instance of a DType (e.g. a specific `float64` with metadata like byte order) is a "descriptor".
- **Abstract DType** (e.g. `Integer`) cannot be instantiated directly; it's used to group concrete DTypes into the type hierarchy.
- **CastingImpl / ArrayMethod**: every pair (source DType, target DType) declares a cast object with 3 functions: `resolve_descriptors` (compute the output descriptor from the input), `get_loop` (pick the right strided-loop function), `strided_loop` (execute on the actual data buffer). The same pattern is used for both casting and ufunc dispatch — unifying two mechanisms that used to be separate.

**Rust mapping**: this pattern maps naturally to a trait: `trait DType { fn resolve_descriptors(...); }`, with concrete dtypes as structs implementing the trait, and abstract DTypes as parent/marker traits. The central design question: dynamic dispatch via `dyn DType` (flexible like Python, but slower) or generic `<D: DType>` (fast, monomorphized, but harder to represent runtime-mixed-dtype arrays) — NumPy's C has to use dynamic dispatch because Python relies on runtime types; Rust can choose a hybrid: a closed enum for built-in dtypes (fast) + `dyn DType` for third-party extension.

## Promotion Rules (NEP 50, Final)

NEP 50 (in effect since NumPy 2.0) fundamentally changes how NumPy mixes types:

- Python `int`/`float`/`complex` are **"weak" abstract DTypes** — with no fixed size/precision, only taking on a concrete type when they meet another NumPy array/scalar.
- **Value-based casting is dropped** from the old behavior: previously `np.float32(3) + 3.0` gave different results depending on the value of the Python number; NEP 50 removes this unpredictability — NumPy arrays and NumPy scalars (including 0-D) always behave like real N-dimensional arrays, never "downgrading" to the other operand's precision.
- **Kind ordering**: `boolean < integral < inexact` (float/complex). Within the same kind, bit-width is compared.

**Rust mapping**: this is a concrete, well-defined algorithm, very suitable to port directly — a good exercise for learning enum + pattern-matching design in Rust (`enum Kind { Bool, Int(u8), Float(u8), Complex(u8) }` + a `common_dtype` function).

## Casting Rules

Every dtype pair has a casting safety level, ranked from safest to loosest:

- **equivalent** — a bit-preserving cast (e.g. changing byte-order within the same dtype).
- **safe** — no information loss (int8 → int32).
- **same_kind** — same group but may lose precision (float64 → float32).
- **unsafe** — may lose information or change kind (float → int).

Each level can also carry a **"+view"** flag — signaling that the cast can be done via a view (no buffer copy) instead of an actual conversion.

Promotion (finding the common type when mixing two operands) relies on two hooks: `__common_dtype__` (finds the common DType between two DType classes) and `__common_instance__` (finds the concrete instance/descriptor, e.g. a common byte-order between two instances of the same DType).

**Rust mapping**: `enum CastSafety { Equivalent, Safe, SameKind, Unsafe }` + a `can_cast(from, to) -> CastSafety` method on the `DType` trait. Clearly separating the 4 safety levels is worth keeping as-is — it lets a `.cast::<T>()` API accept a minimum required safety level as a parameter, instead of every call site deciding on its own.

## Ufuncs & Generalized Ufuncs (NEP 5/20, Final)

A **ufunc** (universal function, e.g. `np.add`) is an element-wise function that broadcasts automatically, supporting `out=`, `where=`, reduce/accumulate/outer.

A **generalized ufunc (gufunc)** extends this to functions that operate on sub-arrays instead of scalars — e.g. `matmul`, `inv`. Its signature declares **core dimensions** (axes belonging to each call's input/output) separately from **loop dimensions** (axes broadcast normally):

```
(i,j),(j)->(i)   # matrix-vector: matrix (i,j), vector (j) → vector (i)
```

The letters `i`, `j` are symbolic core-axis names; every axis before them is broadcast as a normal loop dimension.

Extending ufuncs to new dtypes (NEP 43, still Draft) reuses the exact same `resolve_descriptors`/`get_loop`/`strided_loop` pattern from the DType system — meaning this detail isn't fully settled yet, so treat it as a direction rather than a hard spec when designing Rust.

**Rust mapping**: a signature like `(i,j),(j)->(i)` can be represented with const generics or with runtime shape-checked types (as the `ndarray` crate does with `Ix1`, `Ix2`...). This is a good exercise for learning trait object + generic dispatch in Rust.

## Iterator Design (NEP 10, Final)

`NpyIter` is the central iteration engine used by most ufuncs/reductions. Three main ideas:

- **Cache-coherent output layout selection**: the iterator picks the axis traversal order that maximizes sequential memory access (not necessarily strict C or F order), reducing cache misses.
- **Dimension coalescing**: adjacent axes that are memory-compatible (matching strides) are merged into a single loop, reducing loop overhead.
- **Buffering + casting while iterating**: if the input dtype doesn't match the loop's expected dtype, the iterator automatically casts small buffers on the fly during traversal, avoiding an upfront cast of the whole array.

**Rust mapping**: this part genuinely requires algorithmic work (not just type wrapping) — rewriting it will be a good exercise in designing a generic iterator over multiple arrays at once (`itertools::multizip` is the closest idea, but NumPy also auto-optimizes axis order — no Rust crate fully does this yet).

## Indexing Semantics (and NEP 21 — Deferred)

NumPy has two fundamentally different kinds of indexing:

- **Basic indexing** (slice, int, `...`, `None`) — always returns a view (no copy).
- **Advanced indexing** (integer arrays, boolean arrays) — always returns a copy. When multiple non-adjacent advanced-index arrays are mixed, the result broadcasts under its own rule that makes the resulting axis "jump" to the front of the array — this is one of the most confusing parts of NumPy.

**NEP 21 (Deferred — NOT adopted)** once proposed adding `arr.oindex[...]` (outer/orthogonal indexing — each index array applies independently to its own axis, like MATLAB/Fortran) and `arr.vindex[...]` (explicit vectorized indexing, equivalent to today's advanced-indexing behavior but more transparent) to untangle the ambiguity above. This proposal was never formally adopted — treat it only as historical reference, not current convention.

**Rust mapping**: since Rust can separate its API clearly from the start (not constrained by backward compatibility like Python), this is a chance to design better than the original NumPy: offer explicit `.oindex()`/`.vindex()` as NEP 21 proposed but never delivered, instead of one overloaded `[]` operator.

## 0-D Arrays vs. Array Scalars (NEP 27, Final)

NumPy has two ways of representing "a single number": a 0-dimensional array (`np.array(5)`, with `.shape == ()`) and an array scalar (`np.int64(5)`, which is not an array). The historical/design reasoning per NEP 27:

- Array scalars **inherit from the corresponding native Python type** (`np.float64` inherits `float`), so they're implicitly compatible with Python code that expects a plain number (e.g. used as a dict key, compared with `isinstance(x, float)`).
- 0-D arrays **don't** inherit that way, but keep the full array API (can `.reshape()`, and naturally result from a reduction like `arr.sum()` when no `axis` is given).
- Array scalars are **immutable**, 0-D arrays can be **mutable**.

**Rust mapping**: Rust doesn't need this distinction — it's a consequence of Python lacking real overloading/generics, forcing NumPy to "simulate" two worlds. In Rust, `T` (scalar) and `Array0<T>` can be cleanly separated by the type system without fake inheritance — a point that should be simplified rather than copied as-is.

## RNG Architecture (NEP 19, Final)

Since NumPy 1.17, the RNG architecture is split into two layers:

- **BitGenerator** — generates the raw random bit stream (a specific algorithm: `PCG64` is the new default, or `MT19937`, `Philox`, `SFC64`...). The algorithm can be swapped without changing distribution logic.
- **Generator** — wraps a BitGenerator, providing distribution methods (`.normal()`, `.uniform()`, `.choice()`...), turning raw bits into values from the desired distribution.

**RandomState** (the old API, `np.random.seed()`/`np.random.rand()`) is **permanently frozen** to guarantee reproducibility for legacy code — it receives no new algorithm improvements, only critical bug fixes.

**Rust mapping**: the BitGenerator/Generator split maps very naturally onto Rust's `rand` ecosystem (`trait RngCore` = BitGenerator, `Distribution<T>` = the Generator part) — it can be reused directly instead of rewritten from scratch, you just need to ensure `PCG64` produces results identical to NumPy's if stream compatibility is required.

## Custom Memory Allocator (NEP 49, Final)

The `PyDataMem_Handler` C-API lets you replace an array's default allocator with a custom set of functions: `malloc`, `calloc`, `realloc`, and notably a `free` that receives a size parameter (unlike standard C `free()`, which doesn't need one) — enabling allocators that track memory or use pools more efficiently. Used to integrate GPU pinned memory, tracing allocators, or custom pool allocators.

**Rust mapping**: this is nearly a direct match with Rust's `Allocator` trait (`allocator_api`, currently unstable but with a stable layout): `alloc`, `dealloc` (which also takes a `Layout`, i.e. both size and alignment — even stricter than NEP 49). This is one of the clearest pieces of evidence that Rust's allocator design "arrived later but arrived right" in the direction NumPy itself set out — you should use the `Allocator` trait rather than writing your own mechanism.

## The .npy/.npz File Format (NEP 1, Final)

Structure of a `.npy` file:

1. **Magic string**, 6 bytes: `\x93NUMPY`.
2. **Version**, 2 bytes (major.minor; currently 1.0/2.0/3.0 depending on header size).
3. **Header length** (2 or 4 bytes depending on version) + a header dict as a Python literal string (`{'descr': '<f8', 'fortran_order': False, 'shape': (3, 4), }`) declaring dtype, memory order, shape.
4. **Padding** so the whole header (magic+version+len+dict) is aligned to a multiple of 64 bytes, so the data following the header aligns well for SIMD/mmap.
5. **Raw data**, laid out in the order declared in the header.

`.npz` is simply a ZIP file containing multiple `.npy` files (like a vector of them).

**Rust mapping**: this is an ideal starting exercise for the project — a small format, clearly specified, with results you can verify immediately (by reading a real NumPy file), teaching binary parsing, `serde`-style (de)serialization, and alignment/endianness handling in Rust without first having to worry about the whole complex dtype system.

## Backward-Compatibility Policy (NEP 23, Final)

NumPy has a formal process for breaking changes:

- **DeprecationWarning** when a feature starts being phased out — behavior stays the same, just a warning.
- **VisibleDeprecationWarning** for cases where even regular users (not just developers who configure warning filters) should see the warning.
- At least **one release version** between deprecation and when the change actually takes effect (usually longer in practice).
- Breaking changes are decided through community discussion + a dedicated NEP if the impact is large.

**Rust mapping**: not a technical convention to port code, but a process lesson worth applying to your own Rust project — especially since Rust has a `#[deprecated]` attribute equivalent to `DeprecationWarning`, and semver + Cargo make breaking changes far clearer than Python's versioning system.

## API Cleanup for NumPy 2.0 (NEP 52, Final)

Principles for reorganizing the public namespace:

- **Clear public/private split**: every public API must be in `__all__`; anything not meant for external use is renamed `_private`.
- **One-location-per-function**: every function has exactly one official import path (previously the same function could be reached via multiple aliases, causing documentation confusion).
- **Namespace tiering**: separating `numpy` (highly stable core) from specialized namespaces (`numpy.strings`, `numpy.exceptions`...).
- **Lazy submodule loading**: submodules are only imported when actually used, reducing `import numpy` time and avoiding circular imports.

**Rust mapping**: Rust already has `pub`/`pub(crate)` and a module system far clearer than Python's — the "one location per function" principle should be the default from day one, not something fixed later the way NumPy had to after 15 years.

## Banning dtype=object Inference for Ragged Sequences (NEP 34, Final)

Previously, `np.array([[1, 2], [3, 4, 5]])` (unevenly-lengthed sub-lists, "ragged") would silently create a `dtype=object` array with a `VisibleDeprecationWarning`. NEP 34 requires users to explicitly declare `dtype=object` if they want this behavior — implicit inference is no longer allowed, avoiding silent logic errors when a user actually meant a regular array but mistyped the shape.

**Rust mapping**: this reinforces the boundary already set out in the "What NumPy Is" section — ragged data is NOT within the core's scope. In Rust, `Vec<Vec<T>>` naturally represents ragged data, completely separate from the regular `Array<T, D>` type — the type system automatically prevents this error without needing a runtime warning the way Python does.

## SIMD Universal Intrinsics (NEP 38, Final)

Instead of writing separate kernels for every CPU architecture (SSE, AVX2, AVX-512, NEON...), NumPy defines a common SIMD abstraction layer (`npyv_*` intrinsics) that concrete backends implement underneath. Two important build flags:

- `--cpu-baseline`: the set of SIMD features always assumed available on every machine running the binary (fixed at build time).
- `--cpu-dispatch`: the set of features compiled into multiple versions, chosen at runtime based on the actual CPU (runtime CPU detection/dispatch), trading a larger binary for optimal performance across many machine types.

**Rust mapping**: `std::arch` provides similar-style intrinsics, and `#[target_feature]` + runtime detection (`is_x86_feature_detected!`) achieve the equivalent of `--cpu-dispatch`. This is the part requiring the most effort to reach NumPy-level performance — the `wide` or `pulp` crates can help abstract this instead of writing `std::arch` entirely by hand.

## The New StringDType (NEP 55, Final)

The `np.dtypes.StringDType` (NumPy 2.0+) is gradually replacing `dtype='U'`/`dtype=object` for strings:

- **Variable-length UTF-8 representation** instead of fixed-length UCS4 `'U'` (which wastes huge amounts of memory for ASCII strings).
- **Small-string optimization**: short strings are stored inline directly in the descriptor, no heap allocation needed.
- **Arena allocator** for long strings — pooling allocations into a shared memory region instead of `malloc`-ing each string individually.
- **Missing-data sentinel**: supports a value representing "no data" (similar to NA) directly within the dtype.
- **Thread-safety via a per-descriptor mutex**: since the arena is shared, each descriptor holds its own mutex to synchronize multi-threaded access.

**Rust mapping**: this is one of the areas where Rust has the most natural advantage — `String`/`&str` have always been UTF-8, small-string optimization is available via crates like `smartstring`/`compact_str`, and thread-safety can be guaranteed statically (at compile time via `Send`/`Sync`) instead of relying solely on a runtime mutex like C does.

## The Python Array API Standard (NEP 56, Final)

NumPy 2.0 adopts the Python Array API standard (specification v2022.12, defined by a cross-library consortium) into the main `numpy` namespace: function names, signatures, and broadcasting/dtype-promotion semantics are standardized so code written for NumPy also runs on (or is easily portable to) CuPy, PyTorch, JAX, Dask through the same shared API bindings.

**Rust mapping**: if the project's long-term goal is to interoperate with the Python ecosystem (via PyO3), sticking to the Array API standard's function names/signatures (rather than inventing purely Rust-style names) will help NumPy-familiar users migrate more easily, and give the initial API design a clear direction instead of starting from a blank slate.

## Index of NEPs Read, With Status

| NEP | Topic | Status |
| --- | --- | --- |
| 1 | .npy file format | Final |
| 5 | Generalized universal function API | Final |
| 10 | Iterator (NpyIter) | Final |
| 19 | New RNG architecture | Final |
| 20 | Extending gufunc signatures | Final |
| 21 | oindex/vindex indexing | **Deferred (not adopted)** |
| 23 | Backward-compatibility policy | Final |
| 27 | 0-D arrays vs. array scalars | Final |
| 34 | Banning dtype=object inference for ragged data | Final |
| 38 | Universal SIMD intrinsics | Final |
| 41 | New DType system (foundation) | Accepted |
| 42 | New DType system (API detail) | Accepted |
| 43 | Extending ufuncs for new DTypes | **Draft/Open** |
| 49 | Custom memory allocator C-API | Final |
| 50 | Weak scalar promotion | Final |
| 52 | API cleanup for NumPy 2.0 | Final |
| 55 | New StringDType | Final |
| 56 | Aligning to the Python Array API standard | Final |

Note: only NEP 41/42/43 are still being shaped (43 remains Draft); NEP 21 is a rejected historical proposal — both of these groups should be treated as direction/reference, not settled convention like the remaining NEPs.

## Mapping to Rust Design

| NumPy Convention | Rust Idea | Notes |
| --- | --- | --- |
| DType-as-class (NEP 41/42) | `trait DType` + struct/enum implementations | Consider `dyn DType` (flexible) vs generic (fast) |
| Weak scalar promotion (NEP 50) | `enum Kind` + a `common_dtype` function | Clear algorithm, can be ported directly |
| Casting safety levels | `enum CastSafety` on the `DType` trait | The 4 safety levels are worth keeping as-is |
| Custom allocator (NEP 49) | Rust's `Allocator` trait | Nearly a direct match |
| .npy format (NEP 1) | Simple binary parser | Ideal starting exercise |
| RNG BitGenerator/Generator (NEP 19) | `rand` crate: `RngCore` + `Distribution<T>` | Reusable instead of rewritten |
| StringDType (NEP 55) | `String`/`&str` + `smartstring`/`compact_str` | Rust has a built-in advantage |
| SIMD universal intrinsics (NEP 38) | `std::arch` + `#[target_feature]` + `wide`/`pulp` crate | Requires the most effort to match NumPy performance |
| oindex/vindex (NEP 21, Deferred) | Explicit `.oindex()`/`.vindex()` API | Chance to design better than the original |
| Parallelization (outside NEP, roadmap direction) | `rayon` crate | NumPy is mostly single-threaded; a real opportunity for improvement |
| Python binding (outside NEP scope) | `PyO3` + `maturin` | So the Rust core can be called from Python |

**Prior art worth studying** when designing (not necessarily reusing): `ndarray` (a general strided array, closest to NumPy), `nalgebra` (linear algebra with compile-time dimensions), `faer` (high-performance pure-Rust linear algebra), `candle` (tensors for ML, with a GPU backend).

## Missing Data / numpy.ma (Note, Low Priority)

`numpy.ma` is a separate subclass of `ndarray`, not integrated into the dtype or ufunc dispatch system — leading to poor performance and inconsistent semantics. There was once a proposal to bring missing-data support into the dtype layer, but it stalled due to design disagreement (the specific NEP number hasn't been re-verified in this research session).

**Design decision for the Rust version** (when this is tackled): prefer extending the container rather than creating a separate dtype for "possibly missing value":

- Add a `validity: Option<Bitmap>` field to `NdArray<T>` — separating "what the `T` value looks like" (dtype) from "whether the value exists" (container) — the same design Arrow/Polars/DataFusion have already proven at production scale.
- Avoid making `Nullable<T>` a separate DType per type (`Nullable<i32>`, `Nullable<f64>`...), since that would duplicate the entire casting-rule/ufunc-dispatch machinery for every combination.
- `validity: None` = zero runtime cost; kernels skip the bitmap check entirely.
- A NaN-style sentinel still makes sense as a float-specific optimization, but shouldn't be used as the general architecture since it doesn't generalize to int/string/bool.

Not needed in the early stage (the basic `NdArray` struct) — but the `validity` field should be reserved from the start when designing the struct, to avoid a large refactor later.

## Mapping Real Source Code → Rust Modules

Based on the actual structure map of NumPy's source (871 files: 190 Python, 653 C/C++, 16 Cython, 12 vendored) — grouped by capability layer, not by the original directory tree:

| Rust Module | Real NumPy Source | Notes |
| --- | --- | --- |
| `core/` | `_core/include/numpy/ndarraytypes.h` (struct `PyArrayObject`), `multiarray/arrayobject.c`, `alloc.c`, `ctors.c`, `iterators.c` | Convert — the design core |
| `dtype/` | `multiarray/descriptor.c`, `dtypemeta.c`, `arraytypes.c.src`, `scalartypes.c.src`, `_core/include/numpy/dtype_api.h` | Convert — main trait-learning focus |
| `cast/` | `multiarray/convert_datatype.c`, `dtype_transfer.c` | Convert — NEP 50 algorithm |
| `ufunc/` | `umath/ufunc_object.c`, `ufunc_type_resolution.c`, `loops.c.src` | Convert |
| `simd/` | `_core/src/_simd/`, `common/simd/{sse,avx2,avx512,neon,vec,lsx}/`, `_core/src/highway/` | Convert partially — use `std::arch`/`pulp` crate instead of writing 6 separate intrinsic sets like C |
| `sort/` | `_core/src/npysort/` (except `x86-simd-sort/`) | Convert the algorithmic part, use standard `sort_unstable` for the basics |
| `math/` | `_core/src/npymath/` | Mostly available in `std`/`half` crate, only convert what's missing |
| `iter/` | `multiarray/nditer_*.c` (the real NpyIter) | Convert |
| `io/` | `lib/_format_impl.py` | Convert — matches NEP 1 |
| `random/` | `random/*.pyx` (bit_generator, _pcg64, _generator, mtrand), `random/src/pcg64/`, `philox/` | Convert the wrapper, use `rand`/`rand_pcg` crate for the core |
| `ma/` | `ma/core.py` (18 classes, 94 functions) | Not converted 1-1 — condensed into the `validity` field (see separate note below) |
| **DEPEND ON A PURE-RUST CRATE, not hand-converted** | | |
| `linalg/lapack_lite/` | F2C-translated from Fortran LAPACK/BLAS — machine-generated code, not hand-written | Use the `faer` crate (pure Rust) — readable, debuggable, no FFI/build-system boundary to reason about |
| `fft/pocketfft` | A separate author's (Martin Reinecke) FFT library, vendored as-is | Use the `rustfft` crate (pure Rust) |
| `npysort/x86-simd-sort/` | Intel's own vendored x86-simd-sort library, highly specialized SIMD kernels per CPU microarchitecture | Use standard `sort_unstable` (`sort_unstable_by(|a,b| a.total_cmp(b))` for floats) — no plan to bind `x86-simd-sort` |

**Convert vs. depend-on-a-crate principles:**

| Convert by hand when... | Depend on a crate when... |
| --- | --- |
| The code is NumPy's own design (dtype system, iterator, ufunc dispatch) — there's an idea to learn | The code is a numerical library that's been battle-tested for decades (LAPACK, FFT) — reimplementing the *algorithm* teaches little that reimplementing the *design* doesn't |
| It's readable, has real value for learning algorithms/OOP | The original is machine-generated (F2C) or a highly specialized SIMD kernel — reading *that* specific code teaches nothing, even though the math it implements is worth understanding |
| Rewriting it is a reasonable scope (hundreds to a few thousand lines) | Writing a from-scratch LAPACK/FFT *implementation* (not just using one) would be a separate project unto itself taking years |

In short: convert the parts that are "NumPy's design ideas" by hand; for "numerical tools NumPy merely borrows", reach for an existing implementation rather than writing one from scratch or wrapping NumPy's own copy via FFI.

**Current decision: native Rust from the start** — `faer` for `linalg`, `rustfft` for FFT, `sort_unstable` for sort. None of LAPACK/BLAS/pocketfft/x86-simd-sort is FFI-bound at all.

The earlier plan bound to the C/Fortran libraries first, on the theory that it's faster to ship. That traded away the thing this project is actually for: every line of the core stays pure Rust, buildable with plain `cargo build`, debuggable with normal Rust tooling, with no FFI boundary, no C build toolchain, and no linker path to reason about when something goes wrong. Worth it even though it means accepting small, well-understood behavioral deviations from NumPy (below) instead of bit-for-bit compatibility — those deviations are handled by testing with a tolerance, not by chasing exact equality.

**Known deviations from NumPy this decision accepts** (not mathematically wrong, just not identical to NumPy's specific implementation):

- **Sort**: `f64`/`f32` don't implement `Ord` because of NaN — use `sort_unstable_by(|a,b| a.total_cmp(b))` to match NumPy's behavior of pushing NaN to the end. Neither `sort_unstable` nor NumPy's default `quicksort`/introsort is stable, so the order of equal elements may differ from NumPy's — use `sort()` (Rust's stable sort) if matching NumPy's `kind='stable'` specifically matters.
- **FFT**: `rustfft` and `pocketfft` are both mathematically correct but not bit-for-bit identical (different floating-point summation order → ULP-level error).
- **Linear algebra** (`faer` vs LAPACK): the clearest difference — eigenvector/singular-vector signs can flip (both are mathematically correct), degenerate (repeated) eigenvalues/singular values can come out in a different order/corresponding subspace, error handling for singular matrices follows a different API style (LAPACK's `info` code vs. faer's Rust-style `Result`/panic), and multi-threaded BLAS-backed code can produce non-deterministic results between runs at the ULP level (`faer` is less prone to this than a threaded LAPACK, but not immune).
- **Testing implication**: when comparing results with NumPy, always use tolerance-based comparison (`atol`/`rtol`, like `numpy.allclose`) instead of exact equality, and normalize signs before comparing eigenvectors/SVD. Treat exact equality against NumPy as the wrong test to write for anything touching sort, FFT, or linalg — a tolerance check that passes is the actual spec being met; an exact-equality check that fails on a sign flip or an ULP is a broken test, not a real bug.

## Next Steps

Suggested learning/implementation order, easy to hard, where every step produces immediately verifiable code:

1. **Basic `NdArray` struct** — shape/strides/buffer for a fixed dtype (e.g. `f64`), try manual slicing/view/broadcasting. Learn ownership, lifetimes, `&`/`&mut`.
2. **Reading/writing the .npy format** (NEP 1) — read a real NumPy-exported file, compare results directly with Python. Learn binary parsing, error handling (`Result`/`?`).
3. **Basic DType via trait** — implement 3-4 dtypes (`i32`, `f32`, `f64`, `bool`) via `enum` or `trait`, try the NEP 50 promotion algorithm. Learn traits, generics, pattern matching.
4. **Simple ufunc** — element-wise `add`/`mul` with broadcasting, compare results with real NumPy. Learn iterator design, closures.
5. **Custom allocator** (NEP 49 → `Allocator` trait) — try a simple pool allocator. Learn controlled unsafe Rust.
6. **PyO3 binding** — call the Rust array from Python, compare performance with NumPy on a real problem.
7. **Parallelization with `rayon`** — this is a genuine chance to beat NumPy (the original NumPy is mostly single-threaded outside of BLAS).

Each step should pause to write a benchmark comparing against NumPy — both to track performance progress and to reinforce understanding of the real cost of each design decision.

## Steps 8–20 — Reaching Functional Parity with NumPy

8. **Full advanced indexing** — fancy indexing (integer arrays), boolean mask indexing; decide whether to add explicit `.oindex()`/`.vindex()` (NEP 21) or not.
9. **Structured/record dtype + datetime64/timedelta64** — multi-field dtypes, and the specialized time-related type pair.
10. **StringDType + string ufuncs** — per NEP 55, plus a set of string-processing ufuncs (`upper`, `strip`, `split`...).
11. **`numpy.testing` equivalent** — `assert_array_equal`, `assert_allclose`... so you can write your own tests without depending on real NumPy.
12. **Full reductions/statistics** — `mean`/`std`/`var`/`median`/`percentile`, `nan*` variants, `histogram`, `cov`/`corrcoef`.
13. **`lib/`-layer utility functions** — set operations (`unique`, `intersect1d`, `union1d`), shape ops (`concatenate`, `stack`, `split`, `tile`), `interp`, `gradient` — the largest volume of functions, but built on top of the core that's already there.
14. **Full `linalg`** — solve, eig, SVD, QR, Cholesky, det, norm, matrix_power — via the `faer` crate (native Rust; see the convert-vs-depend-on-a-crate decision above).
15. **FFT module** — via the `rustfft` crate (native Rust).
16. **Full `random` distributions** — binomial, poisson, gamma, beta, dirichlet... (the earlier step 6 only covers basic uniform/normal).
17. **Masked array** — use the `validity: Option<Bitmap>` design already noted.
18. **`polynomial`** — Chebyshev/Hermite/Laguerre/Legendre, built on the existing `linalg`.
19. **Array API standard audit** (NEP 56) — reconcile the final namespace/function names to match the standard.
20. **Free-threading audit + packaging** — review thread safety, publish to crates.io/PyPI, a benchmark suite against real NumPy.

Steps 13–18 account for most of the raw workload (the rarely-used "long tail"), while steps 8–12 decide whether it's "actually usable" for ordinary use cases. If the goal is "usable" rather than 100% coverage, stopping after steps 12–14 can still be considered a success.

## NumPy Parts Worth Dropping When Rewriting in Rust

**Target version: NumPy >= 2.5 semantics only.** This project matches the behavior of current NumPy (2.5.x), not the full 15-20 year history behind it. Anything NumPy itself has deprecated, or keeps only as a backward-compatibility shim for code written against an older version, is out of scope here — don't implement it, don't test against it, don't budget time for it. When in doubt about whether something is "current" or "legacy", check what real NumPy >= 2.5 actually recommends/warns about (`DeprecationWarning`, docs saying "prefer X instead") and only port the recommended side.

Some designs in NumPy exist only for 15–20 years of backward-compatibility reasons — no need to carry them into the Rust version:

- **Fixed-width string dtype** (`U<n>`/`S<n>`) — now that StringDType exists (NEP 55, variable-length UTF-8 + small-string optimization), it does better for most cases. Only keep fixed-width if you need to read existing old `.npy` files — no need to make it a primary dtype.
- **`RandomState`/old random API** (`np.random.seed()`, `np.random.rand()`) — even NumPy itself has permanently frozen it, keeping it only for backward compatibility. The Rust version only needs to implement BitGenerator/Generator directly (NEP 19), skipping RandomState entirely.
- **`np.matrix`** — already officially discouraged by NumPy for a long time (replaced by the `@`/`matmul` operator) — drop entirely, no need for a separate "2D matrix" class.
- **Old value-based casting** (pre-NEP 50) — just implement NEP 50 logic (weak scalar promotion) correctly from the start, no need to install the old logic path only to deprecate it the way NumPy had to.
- **`numpy.char`** (old-style fixed-width string ops) — mostly superseded by string ufuncs on StringDType (step 10) — no need for a separate parallel module.
- **Already-deprecated dtype/function aliases** (`np.float_`, `np.int0`, `np.bool8`, `np.alltrue`/`np.sometrue`...) — NumPy 2.0 itself already cleaned these up under NEP 52 — no need to recreate the old aliases, just use the standard names (Rust's `i32`/`f64` are already clear, no need for aliases like `np.int_`).
- **`np.recarray`** (attribute-based field access) — a Python/NumPy convenience that adds an unnecessary layer of magic — keep normal index/method-based field access, no need to imitate attribute-access magic.
- **`f2py`** (the Fortran-wrapping tool) — entirely outside the scope of a Rust array library — drop it completely, no equivalent needed.
- **Old-style `ctypeslib`** — designed specifically for the CPython C-API — Rust has its own, better FFI/buffer-protocol mechanisms, no need to port this module.
- **Old `distutils` build system** — NumPy itself has already moved to Meson — use Cargo directly, nothing here to carry over.

### Specific deprecations tracked as of NumPy 2.4/2.5

The general principle above is easy to state but easy to under-apply without a concrete list — these are the specific deprecations/roadmap items confirmed as of NumPy 2.5.0 (June 2026). Whenever implementation work touches one of these, drop it rather than porting it:

- **In-place mutation via attribute** (`arr.shape = ...`, `arr.strides = ...`, `arr.dtype = ...` set directly, and in-place `resize`) — deprecated in 2.4/2.5 because it's unsafe when the array is shared or accessed across threads. NumPy's own guidance is to use `np.reshape`/`.view()`/`np.resize` (which return a new array) instead of mutating in place. This project already can't express this the old way regardless — `NdArray`'s `shape`/`strides` are private fields behind read-only getters (see `ndarray.rs`), so there's no attribute-mutation API to even consider adding. Keep it that way: any future "reshape" support should return a new view/array, never mutate `self.shape` behind a caller's back.
- **`numpy.char.chararray`, `numpy.char.array`/`asarray`** — superseded by plain `ndarray` with a string/bytes dtype. Already covered by this doc's existing `numpy.char` entry above (superseded by StringDType, step 10) — no separate chararray-like type needed.
- **`numpy.fix`** — deprecated in favor of `numpy.trunc`. When step 12 (reductions/math functions) implements truncation, implement `trunc` only; don't add `fix` as a second name for the same thing.
- **`numpy.typename`** — deprecated in favor of `dtype.name`. Not applicable to this port's design directly (dtypes are Rust types via the `DType` trait, not name strings looked up at runtime), but if a `numpy.testing`-equivalent (step 11) or introspection helper ever needs a dtype's display name, use the `DType::type_name()`/`Kind` naming already in `dtype.rs`, not a `typename`-style lookup table.
- **Generic (unitless) `timedelta64`/`datetime64`** — deprecated because comparisons on it are non-transitive. Already excluded from this project's design: [`datetime.rs`](src/datetime.rs)'s `TimeUnit` has no "generic" variant, and every `Datetime64`/`Timedelta64` always carries an explicit unit — this was decided independently during step 9's implementation and this deprecation confirms it was the right call, not something to revisit.
- **`np.testing.assert_warns`/`suppress_warnings`** — deprecated in favor of the standard library `warnings` module / `pytest.warns`. When step 11 builds a `numpy.testing` equivalent, skip a Rust analogue of these two entirely — Rust doesn't have Python's warning-system concept to mirror in the first place, so there's nothing to port here even before considering deprecation.
- **Non-integer input to `triu_indices`/`tril_indices`** — deprecated (these functions take indices, which must be integers; float input used to be silently coerced). When step 13's shape/index utilities implement these, only accept integer input types — don't add the old float-coercion behavior as a compatibility path.

Two related items tracked but **not** yet actual deprecations (don't drop, just don't over-invest until NumPy itself settles them):
- **`np.matrix`** — on the long-term roadmap for deprecation, but explicitly gated on SciPy finishing its own migration off sparse *matrix* onto sparse *array* first. Still fine to drop from this port now per the entry above (this project has no SciPy-style backward-compat obligation to wait for), just noting *why* real NumPy hasn't pulled the trigger yet.
- **`numpy.ma` (masked arrays)** — NumPy considers the current design "poorly designed and undermaintained" and is weighing a rewrite (not inheriting from `ndarray`, becoming a duck-array, or moving missing-value support into the dtype system itself) but hasn't committed to a direction. Step 17 of this project already plans a `validity: Option<Bitmap>` design rather than copying `numpy.ma`'s structure — keep that plan; if NumPy lands a concrete redesign before step 17 is reached, re-check this section against it then.

General principle: if something in NumPy exists only to avoid breaking code from decades past (a backward-compat sentinel), you don't carry that burden — just implement the most modern "correct" version (the latest NEP) directly from the start, without needing to implement-then-deprecate the way NumPy had to.
