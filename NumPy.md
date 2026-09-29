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

**Decided, NOT generic: `NdArray<T>` will not gain an `A: Allocator` type parameter.** Raised explicitly as a "decide before it touches every call site" question (like `Vec<T, A: Allocator = Global>` on nightly). Rejected for three reasons: (1) real NumPy's own NEP 49 mechanism is a **global, runtime-swappable handler** (`PyDataMem_SetHandler`), not a per-array compile-time type parameter — confirmed directly against a real NumPy install (no per-array allocator argument exists anywhere in the public API), so `NdArray<T, A>` would actually be *less* faithful to NumPy's own design than keeping `Allocator` a separate, standalone mechanism the way it already is; (2) `NdArray`'s backing store is a plain `Vec<T>` with zero `unsafe` in `ndarray.rs` today — making it allocator-generic on stable Rust (the nightly `allocator_api` is deliberately not a dependency, see above) means hand-rolling `Vec`'s own raw-pointer/layout/growth logic a second time, i.e. introducing `unsafe` into a module that currently has none, for a feature with no realized use case; (3) the change is viral — `ArrayView`/`ArrayViewMut` and every generic function across `ufunc.rs`/`reductions.rs` that constructs a *new* `NdArray` would need the same parameter threaded through, and `utils.rs`'s `concatenate`/`stack`/`tile`/etc. would need to accept-and-forward an allocator to produce any actual benefit — a large, crate-wide signature change for a benefit `allocator.rs`'s existing standalone `Allocator`/`BumpArena`/`PooledVec` (see the previous section) already delivers to anyone who explicitly opts in. If a real, benchmark-driven need for pool-backed `NdArray` storage ever shows up, revisit this then, with that need as the design driver — not before.

## Error Handling: Per-Domain Errors, Plus an Additive `Error` Umbrella

Raised as the other "decide before it touches every signature" question: should every fallible function return one crate-wide `Error` enum (with `impl From<X> for Error` per sub-error), instead of each module owning its own error type (`ShapeError`, `LinalgError`, `FftError`, `RandomError`, `ReductionError`, `ArrayAssertionError`, `NpyError`) and composing only where one module's failure genuinely depends on another's (`npy.rs`'s `NpyError::Shape(ShapeError)`, `polynomial::roots()` reusing `LinalgError` directly rather than inventing its own)?

**Decided: keep the per-domain errors as every function's actual return type; add `Error` as a purely additive, opt-in umbrella, not a replacement.** A single crate-wide `Result<_, Error>` on every public function is an application-level pattern (`anyhow`-style, useful in a `main()` that just wants to `?` everything up to one place), not a library one — real Rust libraries (this crate's own dependencies: `faer`, `rustfft`, `rand_distr`) expose specific, matchable error types per operation for exactly this reason: a caller handling a `LinalgError::Singular` shouldn't have to match through a variant of a giant enum first. This crate's own existing convention (reuse the most specific sub-error directly, wrap only when a function's failure modes genuinely span more than one domain, as `npy.rs` already does) was already the right shape and needed confirming, not replacing. Retrofitting one enum onto every function would also have been a much bigger change than it first looks: every public function in `linalg.rs`/`fft.rs`/`random.rs`/`reductions.rs`/`testing.rs`/`npy.rs`/`ufunc.rs`'s `ShapeError`-returning functions would need its signature edited, for a net *worse* API (coarser error matching) — the opposite of what the proposal was trying to buy.

What *was* missing, though: `error.rs` is the natural home for the "if you want one error type to `?`-propagate through, here it is" convenience real callers sometimes do want (an example `main()`, a future CLI) — added as `pub enum Error { Shape(ShapeError), Linalg(LinalgError), Fft(FftError), Random(RandomError), Reduction(ReductionError), Assertion(ArrayAssertionError), Npy(NpyError) }` with `From<X>` for each variant, `Display` delegating to the wrapped error, and `std::error::Error::source()` returning it — purely additive, zero existing function signatures changed. `error.rs` itself gained `use crate::{linalg, fft, random, reductions, testing, npy}` to name these types; those modules already (or now) import back from `error.rs` for `ShapeError`/`Display`/`Error` impls, which is an ordinary intra-crate cycle (fine within one crate, unlike a circular *Cargo* dependency between separate crates). One knock-on fix this surfaced: `LinalgError` and `FftError` had never actually implemented `Display`/`std::error::Error` at all (an existing, unrelated gap `Error`'s `impl Display for Error` immediately hit as a compile error) — added now, following the same per-variant message style `ShapeError`/`RandomError`/`ReductionError`/`NpyError` already use.

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

### Step 16 audit: this crate's names vs. the Array API standard (v2022.12)

Every public function's name checked against the standard's own function list (elementwise, statistical, manipulation, and linalg-extension modules), after steps 1–15 were already implemented — a genuine after-the-fact audit, not a design done with the standard open from the start.

**Already matched the standard's name exactly, no change needed**: `add`, `sum`, `mean`, `min`, `max`, `std`, `var`, `stack`, `tile`, `vector_norm` (this one's a pleasant surprise — the exact name was picked in step 12 for its own descriptive reasons, before this audit existed, and it happens to be letter-for-letter the standard's own name).

**Renamed/added as a standard-aligned alias, old name kept working** (the same "old name stays callable, new name is what the standard calls it" precedent `add_broadcast` already set in step 4):

| This crate's original name | Standard's name | Added as |
| --- | --- | --- |
| `sub` (`ufunc.rs`, step 4) | `subtract` | `ufunc::subtract`, thin wrapper |
| `mul` (`ufunc.rs`, step 4) | `multiply` | `ufunc::multiply`, thin wrapper |
| `concatenate` (`utils.rs`, step 11) | `concat` | `utils::concat`, thin wrapper (NumPy itself keeps both names too, even post-NEP 56) |
| `frobenius_norm` (`linalg.rs`, step 12) | `matrix_norm` (with `ord='fro'`, the standard's own default) | `linalg::matrix_norm`, thin wrapper — narrower than the standard: no `ord=` parameter, Frobenius only, documented in `matrix_norm`'s own doc comment |

**Standard functions/namespaces not implemented at all** (gaps, not naming mismatches — noted here rather than silently absent):
- `linalg`: `matmul`, `trace`, `pinv`, `slogdet`, `outer`, `cross`, `diagonal`, `matrix_rank`, `vecdot`, `svdvals`, `matrix_transpose` — none of these exist yet in `linalg.rs`.
- `fft`: the standard's optional Fourier extension also specifies `rfftn`/`irfftn` (N-D real transforms) and `hfft`/`ihfft` (Hermitian-symmetric transforms); `fft.rs` has neither. (`fft.rs`'s own `fft2`/`ifft2` are a NumPy-style convenience the standard itself doesn't define — kept as a documented extra, not a standard name.)
- Manipulation: the standard's `unique_all`/`unique_counts`/`unique_inverse`/`unique_values` (four distinct, more specific functions) vs. this crate's single `unique` (closer to plain NumPy's own `np.unique`) — a coarser API, not a wrong one.
- `reshape`, `broadcast_arrays`, `expand_dims`, `flip`, `moveaxis`, `permute_dims`, `repeat`, `roll`, `squeeze`, `unstack` — `reshape` itself now exists (step 18); the rest are still missing, and `flip`/`moveaxis`/`permute_dims`/`squeeze`/`expand_dims` are now cheap because `ArrayView` already carries arbitrary strides (see step 18).

**Out of the standard's scope entirely, so nothing to reconcile**: `random` (the Array API standard doesn't specify a random module), `polynomial`, `structured`/`datetime`/`strings` (NumPy-specific extensions with no Array-API equivalent at all) — these keep their existing, NumPy-flavored names since there's no standard name to align to.

See `examples/step16_array_api.rs` for the added aliases in use.

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

**Design decision for the Rust version** (when this is tackled), revised: a **separate `MaskedArray<T>` struct that wraps `NdArray<T>`** (composition, a newtype-with-extra-field), not a `validity` field bolted onto `NdArray<T>` itself:

```rust
struct NdArray<T> { ... }               // no validity field at all — always "clean"
struct MaskedArray<T> {
    data: NdArray<T>,
    validity: Bitmap,                    // not Option -- MaskedArray always has a mask
}
```

- `NdArray<T>` stays exactly as pure as it is today — it never grows a `validity` field, so there is nothing to "reserve from the start" and no risk of a later refactor touching it. Every `linalg` function (`solve`/`eig`/`det`/...) is implemented only for `NdArray<T>`, never for `MaskedArray<T>` — so passing a `MaskedArray<T>` where `linalg` expects an array is a plain **compile error** ("method not found" / "expected `NdArray<T>`, found `MaskedArray<T>`"), for free, with no runtime check anywhere. This is a stronger, simpler version of the "unsupported ops shouldn't exist as a method" rule below — it isn't even a rule `MaskedArray<T>` has to deliberately enforce by omission, it falls out of the two types just being different types.
- `MaskedArray<T>` implements its own reduction/elementwise ops, reading its own `validity` bitmap to decide what to skip — see the op-level breakdown below. Its `validity: Bitmap` is **not** `Option<Bitmap>`: a `MaskedArray` unconditionally has a mask (an all-ones bitmap is the "nothing is actually masked" case), since "maybe no mask at all" is exactly what having a separate, plain `NdArray<T>` type is already for. `Option` on the field would just re-introduce the two-states-in-one-type problem this split was meant to avoid.
- The fill-first ops (`dot`/`trace`/matmul-style) get an explicit, honestly-named method on `MaskedArray<T>` — e.g. `.filled_zero() -> NdArray<T>` to materialize the fill, or a method like `.dot_fill_zero(&other)` built on top of it — never a `Dot`/`Mul` trait impl that fills silently (same naming-over-runtime-notification principle as below).
- Trade-off worth naming: this does duplicate a little logic between `NdArray<T>` and `MaskedArray<T>` for ops both support in their own way (reductions, elementwise arithmetic) — mitigated cheaply with a shared generic trait (e.g. `trait Reducible { fn sum(&self) -> T; ... }`) implemented for both, where only the loop body differs (skip-if-masked vs. no check at all). Small, worth it for the compile-time guarantee above.
- Avoid making `Nullable<T>` a separate DType per type (`Nullable<i32>`, `Nullable<f64>`...), since that would duplicate the entire casting-rule/ufunc-dispatch machinery for every combination — the `NdArray<T>`/`MaskedArray<T>` split above already gets the same separation of "value" from "is it missing" without touching the dtype system at all.
- A NaN-style sentinel still makes sense as a float-specific optimization, but shouldn't be used as the general architecture since it doesn't generalize to int/string/bool.

**Op-level scope for masked array** (when this is eventually built — lowest priority, see step 28) — split by whether the op can just skip masked entries, or needs a filled (full) array first:

- **Skip-mask directly (no fill needed)**: reductions (`sum`, `mean`, `std`, `var`, `min`/`max`, `count`, `median`, `average`) — the kernel checks validity and excludes masked entries from the accumulation; mask-creation from a condition (`masked_where`, `masked_invalid`, `masked_equal`); elementwise arithmetic (`+`, `-`, `*`, `/`) — result is masked wherever either operand is masked, no filling involved; `filled()`/`compressed()` — conversion utilities, not computation.
- **Needs a full (filled) array first, fill = 0**: `dot`/`trace`/`outer`/matmul-style ops — mathematically these are just "treat masked entries as excluded terms in a sum of products", which is equivalent to filling with 0 (the additive identity) and running the normal op. Not a general linear-algebra solution — just a convenience shortcut for sum-of-products-shaped ops.
- **Needs a full array, but there's no principled fill — not supported**: `solve`, `eig`, `det`, `inv`, SVD — there is no generic mathematically valid way to fill missing entries for these; they require actual imputation (a modeling decision, not a default), so skip masked-array support for these entirely and require the caller to impute first, then hand a regular (unmasked) array to `linalg`.
- **Special case — pairwise deletion, not filling**: `cov`/`corrcoef` — each pairwise covariance between two columns uses only the rows where both columns are visible together, which is not equivalent to any single upfront fill (a global fill would bias the statistics). This is the one case where masked-array support has a real, non-fakeable advantage over "just fill and use a regular array".

**How to surface each of these to the caller — compile time and docs, not a runtime notification.** NumPy is Python: the only place it *can* flag "this op behaves specially on masked data" is at runtime (a warning, or just prose in the docstring the caller may not read). Rust has better tools than that available, and should use them instead of imitating a runtime warning:

- **Skip-mask ops** need no notification at all beyond the type signature/doc comment. `MaskedArray<T>::sum()` returning "the sum of the visible elements" *is* the contract — that's documented behavior, not an edge case, so printing/logging on every call would be pure noise (every reduction and every `+` "announcing itself") and a real perf cost (a check-and-log inside the hottest loop in the whole module, for a fact the caller already agreed to by calling this method on this type).
- **Fill = 0 ops** do deserve a warning, but the idiomatic Rust place for it is the function *name*, not a runtime side effect: `masked.dot_fill_zero(&other)` rather than a `Dot` trait impl that silently fills — the caller sees the fill behavior by reading the call site or during code review, at zero runtime cost, instead of only finding out by running it (or not finding out at all, if nobody's watching stdout/logs that day). If an optional debug-time sanity check is ever wanted, that's what `debug_assert!` or a feature flag is for — never on by default in a release build.
- **Unsupported ops** (`solve`/`eig`/`det`/`inv`/SVD) shouldn't exist as a callable method on `MaskedArray<T>` at all — and with `MaskedArray<T>` as its own separate struct (not `NdArray<T>` plus a field, see above), this isn't even something to remember to omit: `linalg` is only ever implemented for `NdArray<T>`, so `masked.solve(...)` is a compile error ("no method named `solve` found for `MaskedArray<T>`") automatically, not a runtime panic or an `Err` the caller has to remember to check. A mistake caught at `cargo build` beats one caught in production.

General rule for step 28: reach for the type system, method naming, and doc comments first; reserve any actual runtime mechanism (return values, `Result`, `debug_assert!`) for cases doc comments genuinely can't cover, and never use a runtime print/log as the *primary* way of communicating well-documented, contractual behavior.

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

**This tolerance principle is project-wide, not just for linalg/FFT/sort.** Any module computing something numerically (reductions, statistics, future `polynomial`/interpolation work, anything summing or averaging floats) can legitimately differ from NumPy's C implementation at the floating-point level — different summation order, a different but equally valid formula, a different intermediate rounding step — without either side being wrong. The standard from step 12 onward: match NumPy's *documented behavior and formulas* exactly (defaults, edge cases like empty-input or all-`NaN` handling, which case is `Err` vs. a `NaN` return), verify those against a real NumPy run before writing them into a test, and compare numeric results with a tolerance rather than bit-for-bit equality. When an implementation genuinely can't match a behavior exactly (not just a floating-point rounding difference, but a real scope gap or a deliberate simplification), say so explicitly in the code/commit rather than silently shipping a near-miss — small, well-understood, and disclosed beats undisclosed.

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

## Steps 8–28 — Reaching Functional Parity with NumPy

**Reordered twice.** First (after steps 8–17 below, plus the unnumbered `NdArray<T>` generification and full-numeric-support work, were already done): structured/record dtype + datetime64 and StringDType — originally steps 9/10, implemented early, right after advanced indexing — were moved down so that steps 1–17 form one uninterrupted, all-done "numeric core" track, instead of splitting the numeric steps in half with two dtype-extension steps in the middle. Second (after a systematic diff against real NumPy across ufunc/reductions/linalg/fft found no *incorrect* numeric behavior, but did surface a long list of missing surface area): the remaining gaps were triaged into concrete next steps (18–25, all still-numeric/core-mechanics work) and a genuinely-lowest-priority bucket (26–28: the two non-numeric dtype extensions, already implemented but excluded from this repo's numeric-only public snapshot, plus masked arrays, not yet started at all). Steps 1–17's content and order haven't changed since the first reorder; only what comes after has.

8. **Full advanced indexing** *(done)* — fancy indexing (integer arrays), boolean mask indexing; explicit `.oindex()`/`.vindex()` (NEP 21) split, `index.rs`.
9. **`numpy.testing` equivalent** *(done)* — `assert_array_equal`, `assert_allclose`... so you can write your own tests without depending on real NumPy, `testing.rs`.
10. **Full reductions/statistics** *(done)* — `mean`/`std`/`var`/`median`/`percentile`, `nan*` variants, `histogram`, `cov`/`corrcoef`, `reductions.rs`. Later (unnumbered full-numeric-support work, after step 17) made generic over `T`.
11. **`lib/`-layer utility functions** *(done)* — set operations (`unique`, `intersect1d`, `union1d`), shape ops (`concatenate`, `stack`, `split`, `tile`), `interp`, `gradient` — the largest volume of functions, but built on top of the core that's already there, `utils.rs`.
12. **Full `linalg`** *(done)* — solve, eig, SVD, QR, Cholesky, det, norm, matrix_power — via the `faer` crate (native Rust; see the convert-vs-depend-on-a-crate decision above), `linalg.rs`.
13. **FFT module** *(done)* — via the `rustfft` crate (native Rust), `fft.rs`.
14. **Full `random` distributions** *(done)* — binomial, poisson, gamma, beta, dirichlet, exponential, uniform, normal, integers... via a NEP 19 `Generator` (`rand_pcg::Pcg64` + `rand_distr`), statistically equivalent for the `rand_distr`-backed distributions (normal/gamma/beta/binomial/poisson/exponential/dirichlet; step 22 later made the raw stream, `random`, `integers`, `uniform`, `shuffle`, `permutation` and `choice` bit-identical) to NumPy's own `Generator` — see `random.rs`'s doc comment. (Note: this note previously said "the earlier step 6 only covers basic uniform/normal" — stale; step 6 is the PyO3 binding, not a random module. There was no random code before this step.)
15. **`polynomial`** *(done)* — Chebyshev/Hermite/Laguerre/Legendre, built on the existing `linalg`, `polynomial.rs`.
16. **Array API standard audit** (NEP 56) *(done)* — reconcile the final namespace/function names to match the standard.
17. **Free-threading audit + packaging** *(done)* — review thread safety, packaging metadata added; publishing to crates.io/PyPI and a NumPy benchmark suite both stay deliberately deferred (see the "Step 17" section further down).
18. **Core array mechanics** *(done)* — `reshape`/`into_shape`/`ravel`, `sort`/`argsort`/`searchsorted` (`sorting.rs`), `where_cond`/`select`/`choose` (`selection.rs`), `array_split`/`array_split_at` (`utils.rs`), named elementwise math (`mathfunc.rs`), `matmul`/`dot`/`tensordot`/`outer`/`einsum` (`contraction.rs`), NEP 20 `gufunc` + `vecdot` (`gufunc.rs`), NEP 10's real strided iterator (`ViewIter`, `view.rs`). Design notes and divergences in the "Step 18" section below.
19. **NEP 50 promotion wired into `ufunc`** *(done)* — `add`/`sub`/`mul` (and the `_parallel`/`subtract`/`multiply` variants) now promote mixed dtypes at compile time (`int32 + float64 -> float64`) through a `Promote` table generated from real NumPy; Python-scalar weak promotion (`WeakInt`/`WeakFloat`), in-place `add_assign` family with same-kind casting, `where=` masks, and `reduce`/`accumulate`/`outer_with` (`dispatch.rs`, `promote.rs`). See the "Step 19" section below.
20. **Remaining `linalg` gaps** *(done)* — `trace`/`diagonal`/`matrix_transpose`/`kron`/`cross` (generic, `contraction.rs` + zero-copy `ArrayView::diagonal`/`swap_axes`), `slogdet`, `svdvals`, `matrix_rank`, `pinv`, `lstsq`, `cond` + `matrix_norm_ord`, and a full complex `eig`; `matmul`/`outer`/`vecdot` re-exported from `linalg::`. See the "Step 20" section below.
21. **Remaining `fft` gaps** *(done)* — `rfftn`/`irfftn`/`rfft2`/`irfft2`, `hfft`/`ihfft`, and `ComplexArray` is gone: every N-D transform now takes and returns `NdArray<Complex64>`. `irfft` also gained NumPy's crop/zero-pad behaviour. See the "Step 21" section below.
22. **Remaining `random` gaps** *(done)* — `SeedSequence` + `.spawn()` (bit-identical to NumPy), `Generator::spawn`, `shuffle`/`permutation`/`choice` (bit-identical to NumPy for a given seed), `multivariate_normal`; `Generator::seed` was renamed `default_rng` and the raw PCG64 stream, `random()`, `integers()` and `uniform()` are now bit-identical to NumPy too. See the "Step 22" section below.
23. **Remaining indexing gaps** — `vindex` only accepts *adjacent* fancy-index axes today (non-adjacent errors, matching NEP 21's own restriction, but real NumPy handles the non-adjacent case by moving the fancy-indexed result to the front); boolean masks only apply to the whole array, not a prefix of axes.
24. **Remaining Array API standard gaps** (from the step 16 audit, manipulation module) — `unique_all`/`unique_counts`/`unique_inverse`/`unique_values` (four distinct functions vs. this crate's single, coarser `unique`), `broadcast_arrays`/`expand_dims`/`flip`/`moveaxis`/`permute_dims`/`repeat`/`roll`/`squeeze`/`unstack` (none implemented; several of these want step 18's `reshape` machinery first).
25. **Remaining `numpy.testing` gaps** — `assert_raises`-equivalent (Rust already has `#[should_panic]`/`Result`-returning tests as its own idiom, so this is lower-value than it sounds), a generic `assert_array_compare`, `assert_warns` (this crate has no warning/logging side-channel at all, by design — see `reductions.rs`'s module doc comment — so this one may stay permanently out of scope rather than becoming a real step).
26. **Structured/record dtype + datetime64/timedelta64** *(done, lowest priority)* — multi-field dtypes, and the specialized time-related type pair, `structured.rs`/`datetime.rs`. Implemented and tested (git history has the commits), but deliberately excluded from this repo's current public push, which only exposes the numeric core (steps 1–17) — see the "Public snapshot" commit.
27. **StringDType + string ufuncs** *(done, lowest priority)* — per NEP 55, plus a subset of `numpy.strings` (`upper`, `strip`, `str_len`, `add`, `replace`, `startswith`/`endswith`...). Note: `numpy.strings` has no `split` in NumPy 2.5 (`split` only exists on the legacy, fixed-width `numpy.char` side, which this project already drops) — checked directly against the stock venv rather than assumed, so not implemented here. `strings.rs`. Same public-snapshot caveat as step 26.
28. **Masked array** (lowest priority, still not started at all) — a separate `MaskedArray<T>` struct wrapping `NdArray<T>` (`validity: Bitmap`, not `Option`), per the design already noted; scoped down to reduction-style ops only — see the op-level breakdown in the "Missing Data / numpy.ma" section. The only step in this whole list with zero code written yet.

Steps 11–15 account for most of the raw workload (the rarely-used "long tail"), while steps 8–10 decide whether it's "actually usable" for ordinary use cases. If the goal is "usable" rather than 100% coverage, stopping after steps 10–12 can still be considered a success. Within the new 18–25 range, 18 (`reshape` specifically) and 19 (NEP 50 dispatch) are the two with the widest practical impact — most of what real NumPy code actually does day to day touches at least one of them.

**Generic `NdArray<T>` (done — the container, not yet the algorithms on top)**: `NdArray` was `f64`-only through step 15 — step 3's `DType` trait existed standalone but was never wired into the real `NdArray`/`ArrayView` core. Asked explicitly whether to generify `NdArray<T>` right after step 15 (a refactor that sounded like it would touch nearly every existing file) or defer it to a dedicated step; the user pushed back with the actual mechanism (Rust monomorphizes a generic struct per concrete `T` the same way NumPy's `.c.src` templates expand per dtype at build time — a language feature doing the same job a codegen tool does) and asked for it directly.

Turned out to be a much smaller change than expected, because Rust structs support a **default type parameter**: `pub struct NdArray<T = f64> { ... }`. Every place elsewhere in the crate that writes the bare name `NdArray`/`ArrayView`/`ArrayViewMut` in a *type position* (a function signature, a field type, a return type — `ufunc.rs`, `reductions.rs`, `linalg.rs`, `npy.rs`, `fft.rs`, `random.rs`, `testing.rs`, the whole `python/` PyO3 crate) keeps meaning exactly `<f64>`, unedited — the default resolves there automatically. Only `ndarray.rs`, `view.rs`, and `index.rs` (the container + its fancy-indexing methods) needed real edits to add the `<T>` parameter and per-method bounds (`T: Copy` for `get`, `T: Default + Clone` for `zeros`, no bound at all for `from_vec`/`set`). The one real surprise: the default does *not* help type inference inside a function body — `let a = NdArray::zeros(&[2, 3])` with nothing afterward that pins `T` is genuinely ambiguous and needs an explicit `let a: NdArray = ...` annotation; it only resolves automatically where the type is written out literally (a signature), not as an inference fallback. Exactly one test binding in the whole crate needed that annotation; everything else was already pinned by a later f64-literal argument or an f64-only function call.

`NdArray<i32>`, `NdArray<Complex64>`, `NdArray<String>` all now work as real, indexable, `oindex`/`vindex`-capable N-D containers (see `examples/generic_ndarray.rs`) — 183 tests passing, `cargo clippy --all-targets` clean, `cargo build` clean for both the library and the `python/` crate, with zero changes needed in either. **Update**: the "not done" numeric algorithms below are now done for `ufunc`/`reductions` — see the "Full numeric type support" section further down. `fft.rs`'s `ComplexArray` and `linalg::eigvals`'s `(re, im)` pairs still haven't been migrated to `NdArray<Complex64>` yet — they predate this change and still work exactly as before.

**Step 17 (free-threading audit + packaging)**: three parts — thread-safety audit, packaging metadata, and a benchmark suite. Only the first two are done; the benchmark suite stays deferred (see below).

*Thread-safety audit.* `grep`ing the whole `src/` tree for `static`/`thread_local`/`Mutex`/`RwLock` turns up nothing: this crate has **zero global mutable state**, which is exactly the "thread safety enforced by the type system, not a manual audit" case NumPy.md's roadmap section (What NumPy Needs) points to for free-threaded CPython/PEP 703 — there is no C-style module-global to audit in the first place, because nothing here was ever written that way. Checked concretely with a `Send`/`Sync` bound probe (`assert_send::<T>()`/`assert_sync::<T>()` compiled against every public type) rather than just asserted:

- `NdArray<T>`, `ArrayView`/`ArrayViewMut`, `Generator` are all `Send + Sync` automatically (they only ever hold a `Vec<T>`/`&[T]`/`Pcg64`, none of which have interior mutability) — an `NdArray<f64>` can be built on one thread and shared read-only or moved to another with no extra work. `Generator` being `Sync` only means `&Generator` is shareable, not that concurrent draws are safe — every RNG method still takes `&mut self`, so the compiler itself forces the same "one `Generator` per thread, or wrap it in a `Mutex`" discipline NumPy's own docs recommend for its `Generator`/`SeedSequence.spawn()` pattern (not yet implemented here — spawning independent, non-overlapping streams per thread is a real gap, tracked below).
- `ufunc::*_parallel` (Rayon-backed) already required `F: Sync` in their signatures since step 7 — that bound is the compiler surfacing Rayon's actual data-race requirement directly in the type, not something to newly discover here.
- `BumpArena` (the NEP 49 arena allocator) was found to be `!Send` — its `NonNull<u8>` field blocks the auto-trait by default, the same way it would for a hand-rolled `Box`. That ruled out even the ordinary "build the arena on one thread, hand it to a worker" pattern, so a manual `unsafe impl Send for BumpArena` was added (sound: the arena exclusively owns its allocation, nothing else aliases it, same reasoning `Box`'s own `Send` impl relies on) plus the matching `unsafe impl<T: Send, A: Allocator + Sync> Send for PooledVec<'_, T, A>`. Both are covered by new cross-thread tests (`bump_arena_can_be_built_on_one_thread_and_used_on_another`, `pooled_vec_can_be_moved_into_another_thread` in `allocator.rs`) that actually move values across a `thread::spawn` boundary rather than just type-checking a bound. `BumpArena` deliberately stays `!Sync`: `used: Cell<usize>` makes concurrent `&BumpArena` allocation a real data race, and Rust's auto-trait rules already forbid `Sync` for anything containing a `Cell` — no code needed to enforce that, only documentation of *why* it's correct as-is (turning it into a genuinely shareable arena would mean `Cell<usize>` → `AtomicUsize` with `fetch_add`, not done since nothing in this crate needs a cross-thread arena yet).
- The `python/` PyO3 crate: `PyNdArray { inner: NdArray }` is `Send` (its only field is), so it satisfies PyO3's own requirement that a `#[pyclass]` be `Send` (or explicitly `unsendable`). **Not verified**: whether the compiled extension module actually behaves correctly under a free-threaded (`Py_GIL_DISABLED`, Python 3.13t) interpreter — that needs a free-threaded CPython build to test against, which isn't available in this environment, and per this project's verify-before-writing policy, that's left as an explicit open gap rather than claimed. Concretely undone: declaring the module's free-threading support via PyO3's `#[pymodule(gil_used = false)]` attribute (added in PyO3 ≥0.23) — added only once actually tested against a `python3.13t` interpreter, not before.

*Packaging.* `cargo publish --dry-run` on the core crate initially warned about missing `license`/`readme`/`keywords`/`categories`. Added: `LICENSE-MIT` + `LICENSE-APACHE` (the standard Rust-ecosystem dual license), a root `README.md`, and the corresponding Cargo.toml/`pyproject.toml` fields for both the core crate and `python/`. `repository`/`homepage` are still left out since this repo has no git remote configured yet. **Not done, deliberately: no actual `cargo publish`/`maturin publish` was run** — publishing to crates.io/PyPI is a one-way action (crates.io in particular never lets a version be deleted, only yanked), so that step needs the user's explicit go-ahead, not just a clean dry run.

*Benchmark suite.* Still deferred, per this project's own standing rule (see `python/README.md`'s "No performance benchmark yet, on purpose" note, which predates this step): a Rust-vs-NumPy comparison is only meaningful once the comparison is apples-to-apples, and several surface gaps still make it not quite that yet (`ufunc`/`reductions`/`linalg`/`fft`/`random` are still `f64`-only despite `NdArray<T>` itself being generic; `axis=`-parameterized reductions don't exist; no `matmul`). Left for a later, explicitly-requested step.

**Full numeric type support for `ufunc`/`reductions` (unnumbered — extends steps 4/10)**: asked directly, right after step 17, to prioritize this over the plan's own masked-array step (now step 28) — "full numeric support first". Deliberately not given its own number in the list above, for the same reason `NdArray<T>` generification isn't either: it's a cross-cutting hardening of existing steps' machinery, not new NumPy surface area. Scoped down via two explicit choices: (1) `linalg`/`fft`/`random` stay `f64`/`Complex64`-only, since `faer`/`rustfft` themselves only support `f32`/`f64`/`Complex<f32>`/`Complex<f64>` and real NumPy's own LAPACK/FFT bindings upcast every input to `float64` internally anyway — genericizing those signatures wouldn't change their actual arithmetic, only `ufunc` + `reductions` were in scope; (2) `dtype.rs`'s `DType` trait was widened first, to every integer width (`i8`..`i64`, `u8`..`u64` — previously only `i32`/`u32`) and `complex64` (`Complex<f32>` — previously only `complex128`), so the generic functions below would actually have a full family of concrete types to be useful over.

`ufunc::{add,sub,mul,subtract,multiply,add_broadcast}` and their Rayon-parallel counterparts are now `fn add<T: Copy + Add<Output = T>>(a: &ArrayView<T>, b: &ArrayView<T>) -> Result<NdArray<T>, ShapeError>` (parallel versions add `Send + Sync`) — `zip_with`/`zip_with_into`/`map`/`map_parallel` (the shared engine underneath) just needed `T: Copy` with no arithmetic bound at all, since the actual `+`/`-`/`*` only appears in the closure each public function passes in. Verified working end-to-end on `i32`, `u64`, and `Complex64` (not just type-checking — a real test adds/subtracts/multiplies arrays of each and checks the numeric result, see `step_20_generic_ufunc_engine_works_on_non_f64_numeric_types` in `ufunc.rs`).

`reductions` needed a more deliberate design than "just add a `T` parameter", because real NumPy itself treats two different reduction families differently:
- `sum`/`min`/`max` (and their `nan*` counterparts) **preserve** the input's own type — `np.array([1,2,3], dtype=np.int32).min()` is still `int32`.
- `mean`/`var`/`std`/`median`/`percentile` (and their `nan*` counterparts) **always promote to `float64`**, even for integer input — `np.array([1,2,3], dtype=np.int32).mean()` is a `float64`, never an integer-truncated result.

Modeled with two small local traits in `reductions.rs`: `FloatIsh` (an `is_nan_ish()` method, `false` by default for every integer width, the real `is_nan()` for `f32`/`f64` — lets `min`/`max` share one generic implementation across ordered numeric types instead of a float-only NaN-aware copy plus a separate integer-only plain-`PartialOrd` copy) and `AsF64` (lossless-where-possible numeric widening to `f64`, for the always-promotes family). `cov`/`corrcoef`/`histogram` were deliberately left `f64`-only — real NumPy promotes their input to float64 internally too, so genericizing their signatures wouldn't change their actual arithmetic, and nothing yet calls them with a non-`f64` array.

One real, deliberate divergence from real NumPy found and documented (not silently papered over, same policy as the `int64->float64` `can_cast` quirk from the dtype work): verified real NumPy 2.5.3's `sum()` does **not** actually preserve a narrow integer width — `np.array([1,2,3], dtype=np.int32).sum().dtype` is `int64`, not `int32` (NumPy upcasts any integer narrower than the platform default to dodge silent overflow on the reduction, the same "always-promote" idea that makes `.mean()` land on `float64`). Reproducing that exactly isn't just a formula difference — it would mean `sum::<T>`'s return type depending on its input type in a way a plain Rust generic function can't express (`sum::<i32>` returning `i64`, a different concrete type, isn't possible without a second, different-output associated type or an enum-wrapped return). This project keeps the simpler, Rust-idiomatic rule instead — the output type is always exactly `T` — documented directly in `sum`'s own doc comment rather than left as a silent surprise.

Also **not** modeled here, on purpose: NEP 50 mixed-type promotion isn't wired into any of this — `add::<i32>` needs both operands to already be `i32`; it won't accept an `i32` array and an `f64` array and promote the result the way `dtype.rs`'s `common_dtype`/`can_cast` describe how it *should*. Actually dispatching to the right monomorphized instance from two different runtime dtypes (real NumPy's `resolve_descriptors`/`get_loop` machinery from NEP 41/42) is a separate, harder problem than genericizing a same-type operation, and isn't solved here.

201 tests passing (2 new), `cargo clippy --all-targets` clean, full workspace + `python/` crate build clean with zero call-site changes needed anywhere outside `ufunc.rs`/`reductions.rs`/`dtype.rs` themselves — the same "default type parameter, monomorphized per caller" mechanism from the `NdArray<T>` generification above held again: every existing caller (including the whole `python/` PyO3 crate) still just writes the bare `ArrayView`/`NdArray` name and gets `<f64>` automatically.

## Step 18 — Core Array Mechanics

Every value below was checked against NumPy 2.5.3 (`/home/tuannq/venvs/numpy-upstream/bin/python3`) before it went into a test.

**`reshape`.** `ArrayView::reshape(&[isize])` (and `NdArray::reshape`, which delegates) returns a *view* with fresh C-contiguous strides over the same buffer — no copy — and `-1` infers one dimension. It errors exactly where NumPy does: `(5,-1)` on 12 elements, two `-1`s, negative dims other than `-1`, and `(0,-1)` even on an empty array (NumPy refuses to divide by a zero product). The deliberate divergence: a **non-contiguous** view returns `ShapeError::NotContiguous` instead of silently copying the way NumPy does. Rust has no `Cow`-style "maybe a view" return type worth the API weight here, and a surprise `O(n)` allocation hidden inside `reshape` is precisely what this crate's borrow-checked views exist to avoid. Call `.to_owned().into_shape(..)` when a copy is wanted. `NdArray::into_shape(self, ..)` reuses the `Vec` with no copy at all; `ravel()` is `reshape(&[-1])`.

**`sort`/`argsort`/`searchsorted`.** `sorting.rs`, generic over `FloatIsh` (the same trait `reductions.rs` uses to make `min`/`max` NaN-aware). NaN sorts *last*, as in NumPy; `argsort` returns `NdArray<usize>` (NumPy's `intp`). Divergence: `argsort` is always **stable**. NumPy's default `kind='quicksort'` is not, so on ties NumPy *may* order differently for large arrays; matching an unspecified tie order would mean replicating introsort bug-for-bug, and stable is the behaviour callers actually want. `searchsorted` treats NaN as the largest value (`side=Left` of NaN lands on the first NaN, `Right` after the last).

**`where_cond`/`select`/`choose`.** `selection.rs`. All operands broadcast together. `select` takes the **first** true condition (verified: swapping the condition order changes the result) and fills `default` elsewhere (NumPy's default default is 0; here it is a required argument because Rust has no "T's zero" for every `T`). `choose` supports `Raise`/`Wrap`/`Clip`; `Raise` rejects negative indices too (NumPy: `ValueError: invalid entry in choice array`). Named `where_cond` because `where` is a Rust keyword. NumPy's one-argument `where(cond)` (= `nonzero`) is not implemented; the 3-argument form is what step 18 needed.

**`array_split`.** `utils.rs`, generic over `T: Copy`. The first `len % sections` pieces get one extra element; more sections than elements yields empty trailing pieces (`(1,),(1,),(1,),(0,),(0,)` for 3 elements into 5); `sections == 0` is an error. `array_split_at` takes split indices with Python-slice clamping (`[2,5,5,20]` on 10 elements gives `[0,1] [2,3,4] [] [5..9] []`).

**Named math ufuncs.** `mathfunc.rs`: floats through `num_traits::Float` (`sqrt`, `cbrt`, `exp`/`exp2`/`expm1`, `log`/`log2`/`log10`/`log1p`, trig/hyperbolic and inverses, `floor`/`ceil`/`trunc`/`rint`, `reciprocal`, `degrees`/`radians`, `arctan2`/`hypot`/`copysign`/`fmod`), and integers-or-floats through a small `Arith` trait (`abs`/`negative`/`square`/`sign`/`floor_divide`/`remainder`/`power`). Behaviours that differ from a naive Rust port and were verified: `rint` is round-half-*even* (`rint(-0.5) == -0.0`, and `f64::round` would give `-1`); `sign(0.0) == 0` and `sign(NaN)` is NaN (Rust's `signum` gives `1`); integer `abs`/`negative`/`square` **wrap** (`abs(int8(-128)) == -128`); `remainder`/`floor_divide` take the divisor's sign and integer division by zero yields 0; `maximum`/`minimum` propagate NaN whereas `fmax`/`fmin` ignore it; integer `power` with a negative exponent is an error (`NegativeIntegerPower`), matching NumPy's `ValueError`. Not covered: complex-valued math (`sqrt(-1+0j)` etc.) and integer `reciprocal`. The functions live under `rustnumpy::mathfunc::` rather than the crate root, because `min`/`max`/`sum`/`std` already occupy the root from `reductions.rs`.

**`contraction.rs`: one engine, five functions.** `matmul`, `dot`, `tensordot`/`tensordot_n`, `outer` and `einsum` all lower to one internal `contract()`: each operand gets a per-label stride vector (repeated labels add their strides, which is exactly a diagonal; size-1 dims broadcast with stride 0), then two nested odometers walk the output labels and the summed labels reading `data[offset + Σ idx·stride]` directly. No operand is copied, so a batch-broadcast `matmul` never materialises the broadcast. `matmul` follows NumPy's 1-D promotion and batch broadcasting, rejecting 0-D input; `dot` sums the last axis of `a` with the second-to-last of `b` (or the only axis of a 1-D `b`) and falls back to elementwise multiply for a 0-D operand; `einsum` supports explicit and implicit output (implicit = labels appearing once, alphabetically, ellipsis dims first), `...`, traces/diagonals and size-1 broadcasting. Error cases verified against NumPy: a repeated label whose dims disagree even by `1 vs 3` is an error, an explicit output that drops an input ellipsis is an error, and an output label absent from the inputs is an error. Divergence: einsum's `optimize=` path search is not implemented (NumPy's default is also `False`, so this only matters when it is asked for), and every operand must share one `T` until step 19.

**NEP 20 `gufunc`.** `gufunc.rs`: `gufunc("(m,n),(n,p)->(m,p)", &[&a, &b], kernel)`. The signature is parsed once (`Signature`); trailing dims of each operand are its *core* dims, leading dims are *loop* dims broadcast across operands; a core dim name must resolve to one size across all operands, and an output-only name (uninferable) is rejected. The kernel receives contiguous core blocks `(&[&[T]], &mut [&mut [T]], &[usize] /* sizes in first-appearance order */)`, so it is plain slice code with no strides in sight. Errors mirror NumPy's own `vecdot`/`matmul`: too few dims, a core-dim mismatch and a non-broadcastable loop shape all fail. `vecdot` is implemented on top of it as the first real client (and step 20 will re-export it). Not implemented: `?` (flexible/optional) core dims, `1` fixed-size dims, and a signature with an output-only dim supplied through `out=`; `matmul` here goes through `contract()` directly rather than through `gufunc` because the odometer avoids the per-block copy the kernel interface needs.

**NEP 10 iterator.** `ArrayView::iter()`/`iter_range(start, count)` (`ViewIter`) walk a view in C order by adding strides incrementally — one `unravel_index` per *chunk*, never per element — and back `to_owned`, `zip_with`, `map` and both parallel paths. Before this, `zip_with_parallel`/`map_parallel` allocated a fresh `Vec<usize>` per element and every sequential path built one per element through `IndexIter` plus a bounds-checked `get`. The parallel paths now split the output into 4096-element chunks and each rayon task builds its own two `ViewIter`s, so the result stays in order and cache-friendly. Still on the old per-element path (correct, just not yet fast): `zip_with_into`, `index.rs`, `reductions.rs`, `testing.rs` and the `utils.rs` helpers. Per the standing rule, no NumPy performance comparison was made.

## Step 19 — NEP 50 Promotion Wired Into `ufunc`

**Decision: compile-time promotion, not a dynamic array type.** NumPy picks a ufunc loop at *runtime* from the operands' dtypes. Rust cannot return a dtype-dependent `NdArray<T>` from one function, so there were two honest options: a type-erased `DynArray` enum with runtime dispatch, or trait-level promotion. This crate chose the second because it is the same monomorphization story as the rest of the crate (`NdArray<T>` is already generic) and costs nothing at runtime. The cost is real and is documented: an *unannotated* `NdArray::zeros(..)` passed to `add` no longer infers its `T` from the other operand (three tests needed `let b: NdArray = ...`), and the result type is whatever the table says, so callers who want one specific type annotate the result.

**The table is generated, not hand-written.** `scripts/gen_promote.py` asks real NumPy 2.5.3 for `np.result_type` over all 13 x 13 dtype pairs, `np.can_cast(.., 'same_kind')` for the conversion table, and `array + python_scalar` for the weak-scalar table, and writes `src/promote.rs` (`Promote<U>`, `Widen<O>`, `PromoteWeak<S>`). A test walks every pair and checks the generated `Output`'s `Kind` against `dtype.rs`'s hand-written `common_dtype`, so the two independent implementations must agree on all 169 pairs (they do). Regenerate with `python scripts/gen_promote.py src/promote.rs` (needs a NumPy install).

**`Common<B>`** is a blanket trait over `Promote` + `Widen` that gives each ufunc one bound (`A: Common<B>`) instead of four; `Out<A, B>` is the result type. Same-type calls still resolve to `Out = A`, so existing same-dtype code is unchanged.

**Integer overflow wraps everywhere.** Promotion exposed a bug that predates it: `add`/`sub`/`mul`, `matmul`/`dot`/`einsum`, `vecdot`, `sum` and `nansum` used `std::ops::Add/Mul`, which *panic on overflow in debug builds and wrap in release* -- while NumPy always wraps (`int8(127) + int8(1) == -128`). They now use `WrapAdd`/`WrapSub`/`WrapMul` (integers use `wrapping_*`, floats/complex are the plain operators, `bool` add is logical-or and mul is logical-and, and `bool` deliberately has **no** `WrapSub` because NumPy raises `TypeError` for boolean subtract, so it is a compile error here). NEP 50 says the wrapped result happens in the *promoted* type, which is what the tests check (`int8 + int8` wraps, `int8 + int16` does not).

**Weak Python scalars (NEP 50's other half).** `add_weak_int(&a, 3)` / `add_weak_float(&a, 3.0)` (+ `sub_`, `mul_`, and the generic `map_weak_int/float`) implement "a Python `int`/`float` never widens an array of the same kind": `int8 + 3 -> int8`, `float32 + 3.0 -> float32`, `int8 + 3.0 -> float64`, `bool + 3 -> int64`, `complex64 + 3.0 -> complex64`. A Python int that does not fit the result dtype is an error (`WeakScalarOverflow`, NumPy's `OverflowError`): `int8 + 300` and `uint8 + (-1)` fail, `uint8 + 255` wraps to 0 for the array element only. A Python *float* too large for `float32` silently becomes `inf`, as in NumPy. Not implemented: a weak *complex* scalar, and a Python int beyond `i64` (`int64 + 2**63`).

**`out=` and aliasing.** `np.add(a, b, out=a)` cannot be written in safe Rust (`&mut a` and `&a` at once), so the in-place case is its own API: `add_assign`/`sub_assign`/`mul_assign(&mut ArrayViewMut<A>, &ArrayView<B>)`, or the general `zip_assign(out, b, mask, f)`. NumPy computes in the promoted type and then casts to `out` under `same_kind`; so does this, and the bound `Out<A, B>: Widen<A>` *is* the same_kind rule -- `int32 += int64` compiles (and works, as in NumPy), `int32 += float64` does not compile (NumPy: `UFuncTypeError` at runtime). Partially overlapping views of one buffer (`a[1:] += a[:-1]`) are unrepresentable by construction. For a distinct output, `zip_promoted_into(out, a, b, mask, f)` accepts any `out` dtype reachable by same_kind.

**`where=`.** A `bool` mask (broadcast to the output shape) on `zip_assign`/`zip_promoted_into`: masked-out slots keep their previous value, exactly like `np.add(x, y, out=out, where=m)`. For reductions, `ReduceOptions { where_, .. }` skips masked elements; NumPy requires `initial` when a ufunc has no identity, and so does this (`ReduceWhereNeedsInitial`), though here it is checked for *every* function because `reduce` takes an arbitrary closure and cannot know its identity.

**`reduce`/`accumulate`/`outer_with`.** `reduce(view, axis, ReduceOptions { initial, where_, keepdims }, f)` = `np.add.reduce(..)`; an empty axis with no `initial` is `ReduceNoIdentity` (NumPy: "zero-size array to reduction operation maximum which has no identity" -- and, unlike NumPy, this also applies to `add`/`multiply`, whose identity NumPy knows; pass `initial: Some(0)` / `Some(1)`). `accumulate(view, axis, f)` keeps the input shape. `outer_with(a, b, f)` has shape `a.shape ++ b.shape` and promotes mixed dtypes. Values verified against NumPy 2.5.3 for `add`/`multiply`/`maximum` reduce and accumulate on 1-D and 2-D input. Divergences: one axis at a time (NumPy also takes a tuple or `None`), and integer `reduce` keeps `T` instead of NumPy's `int64` upcast, for the same reason `sum` does (see the full-numeric-support section).

**Not yet promoted:** the named functions in `mathfunc.rs` (`arctan2`, `hypot`, `maximum`, `power`, `floor_divide`, ...) still require both operands to be the same `T`; only `add`/`sub`/`mul`, `outer_with`, `reduce`-family and the assign/weak entry points dispatch. Extending them is mechanical (`zip_with_promoted` + a bound on the promoted type) and left for when a caller needs it.

## Step 20 — Remaining `linalg`

Verified against NumPy 2.5.3. The 2-D `f64` functions stay on `faer` like the rest of `linalg.rs`; the shape-only ones are generic.

- **Generic, in `contraction.rs`**: `trace(view, offset)`, `kron` (N-D, pads the shorter shape with leading 1s: `kron([1,2],[[1,2,3]])` has shape `(1,6)`) and `cross`. `cross` is built on `gufunc("(n),(n)->(n)")`, needs a last axis of exactly 3 and broadcasts leading axes; the 2-D-vector form is **not** implemented because NumPy 2.5 itself rejects it. All three use `WrapAdd`/`WrapMul` so integer results wrap like NumPy.
- **Views, not copies**: `ArrayView::diagonal(offset)` is a 1-D view with stride `s0 + s1` (NumPy's `diagonal` is also a view), `swap_axes` and `matrix_transpose` only swap strides. An out-of-range offset gives an empty view.
- **`slogdet`** is an own partial-pivot LU (returns `(sign, ln|det|)`, singular gives `(0, -inf)`) instead of `det().ln()`, so it does not overflow for large matrices.
- **`matrix_rank`** uses NumPy's default `tol = smax * max(M, N) * eps`, and returns 0/1 for inputs with fewer than 2 dims. **`pinv`** uses NumPy 2.x's fixed default `rcond = 1e-15` (not `max(M,N)*eps`, which is what `rcond=None` meant before 2.0), **`lstsq`** defaults to `eps * max(M,N)`; both compute from `svd`. `lstsq` returns `Lstsq { x, residuals, rank, singular_values }` and, as in NumPy, `residuals` is empty unless the system is over-determined *and* full rank.
- **`cond`/`matrix_norm_ord`**: `MatNormOrd` covers `Fro`, `Nuc`, `Inf`, `NegInf`, `One`, `NegOne`, `Two`, `NegTwo` (all eight values verified). `cond` with `Two`/`NegTwo` is a ratio of singular values; the other orders use `norm(a) * norm(inv(a))`, and a singular matrix gives `inf` like NumPy. A singular matrix under `Two` gives a huge finite number (~1e16, from round-off) in both libraries, so tests only assert `> 1e15`.
- **`eig`** returns `(Vec<Complex64>, NdArray<Complex64>)`. Eigenvectors are normalised to unit 2-norm with their largest component made real, as LAPACK does. **Not comparable to NumPy element-wise**: eigenvalue *order* and the sign/phase of eigenvectors differ between `faer` and LAPACK (the same divergence already documented for `eigh`), so the tests check the defining property `A v = lambda v`, unit norm, and sorted eigenvalues (real, complex-conjugate pairs, repeated and defective cases).
- **Not done**: batched (`>2`-D) inputs for all of the above, and `eigh`-style `UPLO`. `linalg::matmul`/`outer`/`vecdot`/`cross`/`kron`/`trace` are re-exports of the step 18 code, not copies.

## Step 21 — Remaining `fft`

- **`ComplexArray` removed.** It predated the generic `NdArray<T>`; `fftn`/`ifftn`/`fft2`/`ifft2` and the new functions now use `NdArray<Complex64>` directly, and `to_complex(&NdArray)` replaces `ComplexArray::from_real`. `FftError::ShapeMismatch` disappeared with it (shape errors are `ShapeError` from `NdArray::from_vec`). This is a breaking API change, accepted because the crate is pre-1.0.
- **`rfftn`/`rfft2`**: real FFT on the last axis (length `n/2 + 1`), then complex FFT on the remaining axes. **`irfftn`/`irfft2(input, s)`**: inverse FFT on the leading axes, then `irfft` on the last; default last length is `2*(m-1)` as in NumPy. Values verified on 2-D, 3-D, odd last axes and complex input.
- **`hfft`/`ihfft`** are `irfft(conj(a), n) * n` and `conj(rfft(a, n)) / n` with NumPy's default `n`, cropping and zero-padding; the imaginary part of the DC (and Nyquist) term is ignored, as in NumPy.
- **`irfft` behaviour change**: it used to reject an input whose length was not `n/2 + 1`; NumPy instead crops or zero-pads, so it now does too (the old "rejects wrong length" test became a crop/pad test).
- **Divergences**: only whole-array transforms (no `axes=` argument); in `irfftn`, `s` must equal the input shape on the leading axes (NumPy crops/pads those too), and passing `s` without `axes` is itself deprecated in NumPy 2.x, so only the "all axes" form is supported.

## Step 22 — Remaining `random`

**This is not the legacy API.** Nothing here touches `np.random.seed`/`RandomState`/`MT19937`. The design is NEP 19 as before: `Generator` on top of a PCG64 bit generator seeded through `SeedSequence`. The constructor used to be called `Generator::seed(u64)`, which read like the legacy `np.random.seed`; it is now `default_rng(seed)` (NumPy's own name), plus `Generator::from_seed_sequence` for explicit `SeedSequence`s. There is still no separate `BitGenerator` trait: PCG64 is the only one implemented, and an abstraction with a single implementor would be dead weight; it becomes worthwhile when a second bit generator (`Philox`, `SFC64`, `PCG64DXSM`) is wanted.

- **`SeedSequence` is NumPy's algorithm, not an approximation.** The entropy-pool hash mixing (`hashmix`/`mix`), `generate_state` and `spawn` (spawn keys, child counter) are re-implemented and checked against `np.random.SeedSequence` word for word for seeds `0`, `42`, `2**32 + 5`, `2**64 - 1`, three spawned children, a grandchild and a second `spawn` call. Seeding PCG64 from four 64-bit words (128-bit state and stream, `pcg_setseq_128_srandom_r`) is what `rand_pcg::Pcg64::new` already does.
- **Consequence: bit-identical streams.** `random_raw`, `random`, `uniform` and `integers` now equal NumPy's for the same seed (`default_rng(42).random(3)` is `[0.7739560485559633, 0.4388784397520523, 0.8585979199113825]` in both), because `random` uses NumPy's `(u64 >> 11) * 2^-53`, `integers` uses NumPy's Lemire bounded-integer algorithm (32-bit and 64-bit paths, with NumPy's buffered `next_uint32` halves), and `uniform` is `low + (high-low) * u`. **`integers` and `uniform` changed** from `rand`'s `Uniform` (values differ from before for the same seed; statistics unchanged).
- **`Generator::spawn(n)`** gives independent child generators (the multi-threading recommendation in the step 17 audit); the first values of the children and of a later `spawn` call match NumPy.
- **`shuffle`/`permutation`/`shuffle_rows`/`permutation_rows`**: NumPy's masked-rejection `random_interval` Fisher-Yates, so results equal NumPy's for the same seed (1-D and row-wise). `shuffle_rows` swaps whole rows (axis 0 only).
- **`choice`/`choice_indices`**: all four NumPy paths are implemented and verified value-for-value: with replacement (uniform via `integers`, weighted via cumulative-sum `searchsorted`), without replacement and no `p` (Floyd's algorithm + a shuffle, or NumPy's tail-shuffle branch when the population is over 10000 and `size > pop/50`), and without replacement with `p` (NumPy's iterative re-draw loop). The validation errors mirror NumPy's messages (too large a sample, `p` not summing to 1 within `sqrt(eps)`, negative or NaN `p`, fewer non-zero `p` than `size`, size mismatch). `choice` over a slice returns cloned elements; there is no `axis=` and no N-D `size` (flat `size` only).
- **`multivariate_normal(mean, cov, size, method, check_valid)`** supports `Svd` (NumPy's default), `Eigh` and `Cholesky`, with NumPy's shape errors and its positive-semi-definite check (`tol = 1e-8`). It is **statistically** equivalent only: the normals come from `rand_distr`'s ziggurat, not NumPy's, and for `Svd`/`Eigh` the factor's signs differ between `faer` and LAPACK, so even given identical normals the samples would differ. Tests verify `T^T T == cov` for all three methods, exact agreement with NumPy for the (unique) Cholesky factor, and sample mean/covariance over 100k draws. Divergences: `check_valid` is a bool (raise or ignore; Rust has no `warnings` channel to mirror `"warn"`), and a negative eigenvalue with `check_valid = false` is clamped to 0 where NumPy produces NaN.
- **Still statistical, not bit-for-bit**: `standard_normal`, `normal`, `gamma`, `beta`, `binomial`, `poisson`, `exponential`, `dirichlet` (they use `rand_distr`).

## NumPy Parts Worth Dropping When Rewriting in Rust

**Target version: NumPy >= 2.5 semantics only.** This project matches the behavior of current NumPy (2.5.x), not the full 15-20 year history behind it. Anything NumPy itself has deprecated, or keeps only as a backward-compatibility shim for code written against an older version, is out of scope here — don't implement it, don't test against it, don't budget time for it. When in doubt about whether something is "current" or "legacy", check what real NumPy >= 2.5 actually recommends/warns about (`DeprecationWarning`, docs saying "prefer X instead") and only port the recommended side.

Some designs in NumPy exist only for 15–20 years of backward-compatibility reasons — no need to carry them into the Rust version:

- **Fixed-width string dtype** (`U<n>`/`S<n>`) — now that StringDType exists (NEP 55, variable-length UTF-8 + small-string optimization), it does better for most cases. Only keep fixed-width if you need to read existing old `.npy` files — no need to make it a primary dtype.
- **`RandomState`/old random API** (`np.random.seed()`, `np.random.rand()`) — even NumPy itself has permanently frozen it, keeping it only for backward compatibility. The Rust version only needs to implement BitGenerator/Generator directly (NEP 19), skipping RandomState entirely.
- **`np.matrix`** — already officially discouraged by NumPy for a long time (replaced by the `@`/`matmul` operator) — drop entirely, no need for a separate "2D matrix" class.
- **Old value-based casting** (pre-NEP 50) — just implement NEP 50 logic (weak scalar promotion) correctly from the start, no need to install the old logic path only to deprecate it the way NumPy had to.
- **`numpy.char`** (old-style fixed-width string ops) — mostly superseded by string ufuncs on StringDType (step 19) — no need for a separate parallel module.
- **Already-deprecated dtype/function aliases** (`np.float_`, `np.int0`, `np.bool8`, `np.alltrue`/`np.sometrue`...) — NumPy 2.0 itself already cleaned these up under NEP 52 — no need to recreate the old aliases, just use the standard names (Rust's `i32`/`f64` are already clear, no need for aliases like `np.int_`).
- **`np.recarray`** (attribute-based field access) — a Python/NumPy convenience that adds an unnecessary layer of magic — keep normal index/method-based field access, no need to imitate attribute-access magic.
- **`f2py`** (the Fortran-wrapping tool) — entirely outside the scope of a Rust array library — drop it completely, no equivalent needed.
- **Old-style `ctypeslib`** — designed specifically for the CPython C-API — Rust has its own, better FFI/buffer-protocol mechanisms, no need to port this module.
- **Old `distutils` build system** — NumPy itself has already moved to Meson — use Cargo directly, nothing here to carry over.

### Specific deprecations tracked as of NumPy 2.4/2.5

The general principle above is easy to state but easy to under-apply without a concrete list — these are the specific deprecations/roadmap items confirmed as of NumPy 2.5.0 (June 2026). Whenever implementation work touches one of these, drop it rather than porting it:

- **In-place mutation via attribute** (`arr.shape = ...`, `arr.strides = ...`, `arr.dtype = ...` set directly, and in-place `resize`) — deprecated in 2.4/2.5 because it's unsafe when the array is shared or accessed across threads. NumPy's own guidance is to use `np.reshape`/`.view()`/`np.resize` (which return a new array) instead of mutating in place. This project already can't express this the old way regardless — `NdArray`'s `shape`/`strides` are private fields behind read-only getters (see `ndarray.rs`), so there's no attribute-mutation API to even consider adding. Keep it that way: any future "reshape" support should return a new view/array, never mutate `self.shape` behind a caller's back.
- **`numpy.char.chararray`, `numpy.char.array`/`asarray`** — superseded by plain `ndarray` with a string/bytes dtype. Already covered by this doc's existing `numpy.char` entry above (superseded by StringDType, step 19) — no separate chararray-like type needed.
- **`numpy.fix`** — deprecated in favor of `numpy.trunc`. When step 12 (reductions/math functions) implements truncation, implement `trunc` only; don't add `fix` as a second name for the same thing.
- **`numpy.typename`** — deprecated in favor of `dtype.name`. Not applicable to this port's design directly (dtypes are Rust types via the `DType` trait, not name strings looked up at runtime), but if a `numpy.testing`-equivalent (step 11) or introspection helper ever needs a dtype's display name, use the `DType::type_name()`/`Kind` naming already in `dtype.rs`, not a `typename`-style lookup table.
- **Generic (unitless) `timedelta64`/`datetime64`** — deprecated because comparisons on it are non-transitive. Already excluded from this project's design: [`datetime.rs`](src/datetime.rs)'s `TimeUnit` has no "generic" variant, and every `Datetime64`/`Timedelta64` always carries an explicit unit — this was decided independently during step 18's implementation and this deprecation confirms it was the right call, not something to revisit.
- **`np.testing.assert_warns`/`suppress_warnings`** — deprecated in favor of the standard library `warnings` module / `pytest.warns`. When step 11 builds a `numpy.testing` equivalent, skip a Rust analogue of these two entirely — Rust doesn't have Python's warning-system concept to mirror in the first place, so there's nothing to port here even before considering deprecation.
- **Non-integer input to `triu_indices`/`tril_indices`** — deprecated (these functions take indices, which must be integers; float input used to be silently coerced). When step 11's shape/index utilities implement these, only accept integer input types — don't add the old float-coercion behavior as a compatibility path.

Two related items tracked but **not** yet actual deprecations (don't drop, just don't over-invest until NumPy itself settles them):
- **`np.matrix`** — on the long-term roadmap for deprecation, but explicitly gated on SciPy finishing its own migration off sparse *matrix* onto sparse *array* first. Still fine to drop from this port now per the entry above (this project has no SciPy-style backward-compat obligation to wait for), just noting *why* real NumPy hasn't pulled the trigger yet.
- **`numpy.ma` (masked arrays)** — NumPy considers the current design "poorly designed and undermaintained" and is weighing a rewrite (not inheriting from `ndarray`, becoming a duck-array, or moving missing-value support into the dtype system itself) but hasn't committed to a direction. Step 28 of this project already plans a separate `MaskedArray<T>` struct (wrapping `NdArray<T>`, `validity: Bitmap`) rather than copying `numpy.ma`'s structure — keep that plan; if NumPy lands a concrete redesign before step 28 is reached, re-check this section against it then.

General principle: if something in NumPy exists only to avoid breaking code from decades past (a backward-compat sentinel), you don't carry that burden — just implement the most modern "correct" version (the latest NEP) directly from the start, without needing to implement-then-deprecate the way NumPy had to.
