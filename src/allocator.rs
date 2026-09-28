//! Step 5: NEP 49 (custom memory allocator) → Rust's `Allocator` trait
//! (see NumPy.md, "Custom memory allocator").
//!
//! Real Rust already has an `Allocator` trait shaped almost exactly like
//! NumPy's `PyDataMem_Handler` — but it's still behind the unstable
//! `allocator_api` feature (nightly-only), and pulling it in would force
//! this whole crate onto nightly just for one module. So this file
//! defines its **own** trait with the same shape (`allocate`/`deallocate`
//! taking a `Layout`), simplified in one way: it returns a plain
//! `NonNull<u8>` instead of `NonNull<[u8]>`, since the real trait's fat
//! pointer exists to let an allocator report back *more* memory than
//! requested — a nuance real NumPy's `PyDataMem_Handler` doesn't have
//! either, and not needed for the lesson here.
//!
//! What this module actually teaches: `std::alloc::{alloc, dealloc}` and
//! `Layout` (the stable primitives underneath both Rust's and NumPy's
//! allocator story), and controlled `unsafe` — every unsafe block below
//! carries a `# Safety` comment explaining exactly which invariant makes
//! it sound, the way real unsafe Rust code is expected to.

use std::alloc::{alloc, dealloc, Layout};
use std::cell::Cell;
use std::fmt;
use std::marker::PhantomData;
use std::ptr::NonNull;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AllocError;

impl fmt::Display for AllocError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "memory allocation failed")
    }
}

impl std::error::Error for AllocError {}

/// Mirrors the shape of the real (unstable) `std::alloc::Allocator`
/// trait, and of NEP 49's `PyDataMem_Handler`: swap this out to change
/// *how* memory is obtained without touching anything that uses it.
pub trait Allocator {
    /// Obtain `layout.size()` bytes aligned to `layout.align()`.
    fn allocate(&self, layout: Layout) -> Result<NonNull<u8>, AllocError>;

    /// Give back memory previously obtained from `self`.
    ///
    /// # Safety
    /// `ptr` must have been returned by a prior `self.allocate(layout)`
    /// call on this same allocator, with the exact same `layout`, and
    /// must not have already been deallocated.
    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout);
}

/// The default allocator: delegates straight to the global allocator via
/// the stable `std::alloc::{alloc, dealloc}` functions — equivalent to
/// NumPy's default `PyDataMem_Handler` before anyone installs a custom one.
pub struct System;

impl Allocator for System {
    fn allocate(&self, layout: Layout) -> Result<NonNull<u8>, AllocError> {
        if layout.size() == 0 {
            // `alloc()`'s contract forbids a zero-size layout; every
            // zero-size allocation in Rust is conventionally represented
            // by a dangling-but-non-null, correctly-aligned pointer that
            // is never actually read through.
            return Ok(NonNull::dangling());
        }
        // SAFETY: `layout.size() != 0` here, which is exactly what
        // `alloc`'s contract requires.
        let ptr = unsafe { alloc(layout) };
        NonNull::new(ptr).ok_or(AllocError)
    }

    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
        if layout.size() != 0 {
            // SAFETY: forwarded from this function's own contract — the
            // caller guarantees `ptr`/`layout` match a prior `allocate`
            // call on `self`, which is exactly what `dealloc` requires.
            unsafe { dealloc(ptr.as_ptr(), layout) };
        }
    }
}

/// Round `addr` up to the next multiple of `align` (`align` must be a
/// power of two, which `Layout` already guarantees for every layout we're
/// handed).
fn align_up(addr: usize, align: usize) -> usize {
    (addr + align - 1) & !(align - 1)
}

/// A bump (arena) allocator: one big upfront allocation, handed out in
/// increasing order by simply moving an offset forward. `deallocate` is a
/// deliberate no-op — nothing is freed individually, the whole arena goes
/// back to the system in one shot when the `BumpArena` itself drops.
///
/// This is exactly the use case NEP 49 calls out for a custom allocator:
/// a pool that batches frees together instead of paying per-object
/// bookkeeping — good for, say, building up many short-lived array
/// buffers during one computation and throwing them all away at once.
pub struct BumpArena {
    buffer: NonNull<u8>,
    capacity: usize,
    used: Cell<usize>,
    /// The layout of the *whole* arena block — needed to give it back
    /// correctly in `Drop`, since `dealloc` must be called with the same
    /// layout `alloc` was called with.
    block_layout: Layout,
}

impl BumpArena {
    pub fn with_capacity(capacity: usize) -> Self {
        let block_layout = Layout::array::<u8>(capacity).expect("capacity too large");
        let buffer = if capacity == 0 {
            NonNull::dangling()
        } else {
            // SAFETY: `capacity != 0`, so `block_layout.size() != 0`.
            let ptr = unsafe { alloc(block_layout) };
            NonNull::new(ptr).expect("system allocator returned null")
        };
        Self { buffer, capacity, used: Cell::new(0), block_layout }
    }

    /// Bytes handed out so far — grows monotonically, never shrinks
    /// (there's no way to "return" bytes to a bump allocator except
    /// dropping the whole arena).
    pub fn used(&self) -> usize {
        self.used.get()
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }
}

impl Allocator for BumpArena {
    fn allocate(&self, layout: Layout) -> Result<NonNull<u8>, AllocError> {
        if layout.size() == 0 {
            return Ok(NonNull::dangling());
        }
        let used = self.used.get();
        let base_ptr = self.buffer.as_ptr();
        let current = base_ptr.addr() + used;
        let aligned = align_up(current, layout.align());
        let padding = aligned - current;
        let new_used = used
            .checked_add(padding)
            .and_then(|u| u.checked_add(layout.size()))
            .ok_or(AllocError)?;
        if new_used > self.capacity {
            return Err(AllocError); // arena exhausted — a real allocator would fall back to the system
        }
        self.used.set(new_used);
        // Rebuild the pointer via `with_addr` instead of an integer->pointer
        // `as` cast: this keeps `base_ptr`'s original provenance (Miri's
        // strict-provenance model tracks *which* allocation a pointer is
        // allowed to touch separately from its numeric address — an `as`
        // cast from a bare `usize` would carry no such provenance).
        let result_ptr = base_ptr.with_addr(aligned);
        // SAFETY: `aligned` is within `[base, base + capacity)` by the
        // check above, correctly aligned to `layout.align()` by
        // construction, and this offset range hasn't been handed out by
        // any earlier `allocate` call on this arena (the bump pointer
        // only ever moves forward), so it can't alias a still-live
        // allocation.
        Ok(unsafe { NonNull::new_unchecked(result_ptr) })
    }

    unsafe fn deallocate(&self, _ptr: NonNull<u8>, _layout: Layout) {
        // Arena semantics: individually "freeing" would need real
        // bookkeeping (a free list) to reuse the space, which defeats the
        // point of a bump allocator's O(1) allocation. Everything is
        // reclaimed at once in `Drop` instead.
    }
}

impl Drop for BumpArena {
    fn drop(&mut self) {
        if self.capacity != 0 {
            // SAFETY: `self.buffer` was obtained from `alloc(self.block_layout)`
            // in `with_capacity` and this is the only place that frees it,
            // running at most once (Rust never calls `drop` twice).
            unsafe { dealloc(self.buffer.as_ptr(), self.block_layout) };
        }
    }
}

// SAFETY: `BumpArena` exclusively owns its `buffer` allocation the same way
// `Box<[u8]>` owns its heap allocation -- `buffer` is never aliased by any
// other `BumpArena`, so moving one to another thread and using/dropping it
// there is sound. Without this impl, `NonNull<u8>` (the raw pointer field)
// makes `BumpArena` `!Send` by default, which would rule out even the
// ordinary "build the arena on one thread, hand it to a worker" pattern.
//
// Deliberately **not** `Sync`: `used: Cell<usize>` makes concurrent
// `&BumpArena` access unsound (two threads racing on the bump offset), and
// Rust's auto-trait rules already forbid `Sync` for any type containing a
// `Cell` -- no explicit `impl` is needed (or possible) to block it, which is
// exactly the "thread safety enforced by the type system, not by a code
// review" case NumPy.md's roadmap section points to for free-threaded
// CPython. Making `BumpArena` genuinely shareable across threads would mean
// swapping `Cell<usize>` for `AtomicUsize` and `fetch_add`/CAS in
// `allocate` -- not done here since nothing in this crate needs a
// cross-thread arena yet (see step 19's audit in `NumPy.md`).
unsafe impl Send for BumpArena {}

/// A small owned buffer of `T`, allocated through any [`Allocator`]
/// instead of always going through Rust's global allocator — the whole
/// point of the exercise: swap `System` for `BumpArena` (or any other
/// `Allocator`) and every `PooledVec` built from it changes where its
/// bytes actually live, with no change to the code using `PooledVec`.
///
/// Restricted to `T: Copy` on construction so `Drop` never has to run
/// per-element destructors via `drop_in_place` — it only has to give the
/// raw bytes back. That restriction happens to match this whole project
/// so far, which has only ever stored `f64`.
pub struct PooledVec<'a, T, A: Allocator> {
    ptr: NonNull<T>,
    len: usize,
    layout: Layout,
    allocator: &'a A,
    _marker: PhantomData<T>,
}

impl<'a, T: Copy, A: Allocator> PooledVec<'a, T, A> {
    pub fn from_slice(allocator: &'a A, data: &[T]) -> Result<Self, AllocError> {
        let layout = Layout::array::<T>(data.len()).map_err(|_| AllocError)?;
        let ptr = allocator.allocate(layout)?.cast::<T>();
        // SAFETY: `ptr` points to `layout.size()` freshly allocated bytes
        // (from `allocator.allocate`), correctly aligned for `T` (that's
        // exactly what `Layout::array::<T>` computed), and large enough
        // for `data.len()` elements. Writing each element via `ptr::write`
        // is therefore in-bounds, and `write` (rather than plain
        // assignment) correctly skips dropping whatever
        // (uninitialized/garbage) value was already at that address.
        unsafe {
            for (i, &value) in data.iter().enumerate() {
                ptr.as_ptr().add(i).write(value);
            }
        }
        Ok(Self { ptr, len: data.len(), layout, allocator, _marker: PhantomData })
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn as_slice(&self) -> &[T] {
        // SAFETY: `from_slice` wrote exactly `self.len` valid `T` values
        // starting at `self.ptr`, and nothing else can mutate or free
        // them while this `&self` borrow is alive (the allocation is
        // privately owned by this `PooledVec`).
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }
}

// SAFETY: mirrors `BumpArena`'s reasoning above -- a `PooledVec` exclusively
// owns the `ptr` allocation it got back from `allocator.allocate` (nothing
// else holds a pointer into it), so moving one to another thread is sound
// as long as its element type can cross threads too (`T: Send`) and the
// borrowed allocator can be used from that other thread when `Drop` calls
// back into `self.allocator.deallocate` (`A: Sync`, since `&'a A` is a
// shared reference that may now be read from a different thread than the
// one that created it).
unsafe impl<'a, T: Send, A: Allocator + Sync> Send for PooledVec<'a, T, A> {}

impl<'a, T, A: Allocator> Drop for PooledVec<'a, T, A> {
    fn drop(&mut self) {
        // No `drop_in_place` needed for the individual elements: `T: Copy`
        // was required in `from_slice`, and `Copy` types can't implement
        // `Drop`, so there's no per-element destructor to run — handing
        // the raw bytes back is the whole job.
        //
        // SAFETY: `self.ptr`/`self.layout` are exactly what `from_slice`
        // got back from `self.allocator.allocate`, `drop` runs at most
        // once, and nothing else holds a reference to this buffer past
        // this point (this `PooledVec` was its sole owner).
        unsafe { self.allocator.deallocate(self.ptr.cast(), self.layout) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn system_allocator_roundtrip() {
        let layout = Layout::array::<f64>(4).unwrap();
        let ptr = System.allocate(layout).unwrap();
        // SAFETY: freshly allocated, correctly sized/aligned for 4 f64s.
        unsafe {
            let typed = ptr.as_ptr() as *mut f64;
            for i in 0..4 {
                typed.add(i).write(i as f64 * 1.5);
            }
            for i in 0..4 {
                assert_eq!(*typed.add(i), i as f64 * 1.5);
            }
            System.deallocate(ptr, layout);
        }
    }

    #[test]
    fn bump_arena_hands_out_growing_non_overlapping_offsets() {
        let arena = BumpArena::with_capacity(64);
        let layout = Layout::array::<u8>(8).unwrap();
        let first = arena.allocate(layout).unwrap();
        let second = arena.allocate(layout).unwrap();
        assert_eq!(arena.used(), 16);
        assert_eq!(second.as_ptr() as usize - first.as_ptr() as usize, 8);
    }

    #[test]
    fn bump_arena_respects_alignment() {
        let arena = BumpArena::with_capacity(64);
        // First allocate a single byte (align 1) to push the bump pointer
        // to an address that's very likely misaligned for f64 (align 8).
        arena.allocate(Layout::array::<u8>(1).unwrap()).unwrap();
        let f64_layout = Layout::array::<f64>(1).unwrap();
        let ptr = arena.allocate(f64_layout).unwrap();
        assert_eq!(ptr.as_ptr() as usize % 8, 0, "f64 allocation must be 8-byte aligned");
    }

    #[test]
    fn bump_arena_errors_when_exhausted() {
        let arena = BumpArena::with_capacity(8);
        let layout = Layout::array::<u8>(16).unwrap();
        assert_eq!(arena.allocate(layout), Err(AllocError));
    }

    #[test]
    fn pooled_vec_stores_and_reads_values() {
        let arena = BumpArena::with_capacity(1024);
        let data = [1.0, 2.0, 3.0, 4.0];
        let pooled = PooledVec::from_slice(&arena, &data).unwrap();
        assert_eq!(pooled.len(), 4);
        assert_eq!(pooled.as_slice(), &data);
    }

    #[test]
    fn multiple_pooled_vecs_from_the_same_arena_dont_corrupt_each_other() {
        let arena = BumpArena::with_capacity(1024);
        let a = PooledVec::from_slice(&arena, &[1.0, 2.0, 3.0]).unwrap();
        let b = PooledVec::from_slice(&arena, &[10.0, 20.0]).unwrap();
        let c = PooledVec::from_slice(&arena, &[100.0]).unwrap();
        // If any allocation overlapped another, writing `b` or `c` would
        // have corrupted `a`'s bytes (or vice versa).
        assert_eq!(a.as_slice(), &[1.0, 2.0, 3.0]);
        assert_eq!(b.as_slice(), &[10.0, 20.0]);
        assert_eq!(c.as_slice(), &[100.0]);
    }

    #[test]
    fn pooled_vec_works_with_the_system_allocator_too() {
        // Same PooledVec code, different Allocator — the whole point of
        // the trait.
        let pooled = PooledVec::from_slice(&System, &[7.0, 8.0, 9.0]).unwrap();
        assert_eq!(pooled.as_slice(), &[7.0, 8.0, 9.0]);
    }

    #[test]
    fn empty_pooled_vec_does_not_allocate_or_crash() {
        let arena = BumpArena::with_capacity(64);
        let pooled: PooledVec<f64, _> = PooledVec::from_slice(&arena, &[]).unwrap();
        assert!(pooled.is_empty());
        assert_eq!(arena.used(), 0);
    }

    /// Step 19 (free-threading audit): this only compiles at all *because*
    /// of the `unsafe impl Send for BumpArena` above -- before it existed,
    /// `BumpArena` was `!Send` (its raw `NonNull<u8>` field blocks the
    /// auto-trait by default), so `thread::spawn` wouldn't even accept a
    /// closure that moves one in. Building and draining the arena on
    /// separate threads exercises that the ownership transfer is actually
    /// sound, not just that it type-checks.
    #[test]
    fn bump_arena_can_be_built_on_one_thread_and_used_on_another() {
        let arena = std::thread::spawn(|| {
            let arena = BumpArena::with_capacity(64);
            arena.allocate(Layout::array::<u8>(8).unwrap()).unwrap();
            arena
        })
        .join()
        .unwrap();

        let used = std::thread::spawn(move || {
            arena.allocate(Layout::array::<u8>(8).unwrap()).unwrap();
            arena.used()
        })
        .join()
        .unwrap();
        assert_eq!(used, 16);
    }

    /// Same point as above, one level up: a `PooledVec::from_slice(&alloc, ..)`
    /// built on the main thread is *moved* (ownership transferred, not
    /// shared) into a worker thread and read there. This needs `A: Sync`
    /// on top of `T: Send` (see the `unsafe impl` above) -- `System` is a
    /// zero-field marker struct, so it's trivially `Sync`. Note this only
    /// exercises `Send` (moving), not `Sync` (sharing `&PooledVec` across
    /// threads): `PooledVec` has no manual `Sync` impl, so its `NonNull<T>`
    /// field keeps it `!Sync` by the auto-trait default, same as any other
    /// raw-pointer-backed owning type (`Box`, `Vec`) would need one to
    /// opt back in -- not needed here since nothing shares a `PooledVec`
    /// by reference across threads.
    #[test]
    fn pooled_vec_can_be_moved_into_another_thread() {
        let pooled = PooledVec::from_slice(&System, &[1.0, 2.0, 3.0]).unwrap();
        let sum = std::thread::spawn(move || pooled.as_slice().iter().sum::<f64>()).join().unwrap();
        assert_eq!(sum, 6.0);
    }
}
