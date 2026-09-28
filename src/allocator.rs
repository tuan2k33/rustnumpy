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

pub trait Allocator {

    fn allocate(&self, layout: Layout) -> Result<NonNull<u8>, AllocError>;

    #[allow(clippy::missing_safety_doc)]
    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout);
}

pub struct System;

impl Allocator for System {
    fn allocate(&self, layout: Layout) -> Result<NonNull<u8>, AllocError> {
        if layout.size() == 0 {

            return Ok(NonNull::dangling());
        }

        let ptr = unsafe { alloc(layout) };
        NonNull::new(ptr).ok_or(AllocError)
    }

    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
        if layout.size() != 0 {

            unsafe { dealloc(ptr.as_ptr(), layout) };
        }
    }
}

fn align_up(addr: usize, align: usize) -> usize {
    (addr + align - 1) & !(align - 1)
}

pub struct BumpArena {
    buffer: NonNull<u8>,
    capacity: usize,
    used: Cell<usize>,

    block_layout: Layout,
}

impl BumpArena {
    pub fn with_capacity(capacity: usize) -> Self {
        let block_layout = Layout::array::<u8>(capacity).expect("capacity too large");
        let buffer = if capacity == 0 {
            NonNull::dangling()
        } else {

            let ptr = unsafe { alloc(block_layout) };
            NonNull::new(ptr).expect("system allocator returned null")
        };
        Self { buffer, capacity, used: Cell::new(0), block_layout }
    }

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
            return Err(AllocError);
        }
        self.used.set(new_used);

        let result_ptr = base_ptr.with_addr(aligned);

        Ok(unsafe { NonNull::new_unchecked(result_ptr) })
    }

    unsafe fn deallocate(&self, _ptr: NonNull<u8>, _layout: Layout) {

    }
}

impl Drop for BumpArena {
    fn drop(&mut self) {
        if self.capacity != 0 {

            unsafe { dealloc(self.buffer.as_ptr(), self.block_layout) };
        }
    }
}

unsafe impl Send for BumpArena {}

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

        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }
}

unsafe impl<'a, T: Send, A: Allocator + Sync> Send for PooledVec<'a, T, A> {}

impl<'a, T, A: Allocator> Drop for PooledVec<'a, T, A> {
    fn drop(&mut self) {

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

        assert_eq!(a.as_slice(), &[1.0, 2.0, 3.0]);
        assert_eq!(b.as_slice(), &[10.0, 20.0]);
        assert_eq!(c.as_slice(), &[100.0]);
    }

    #[test]
    fn pooled_vec_works_with_the_system_allocator_too() {

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

    #[test]
    fn pooled_vec_can_be_moved_into_another_thread() {
        let pooled = PooledVec::from_slice(&System, &[1.0, 2.0, 3.0]).unwrap();
        let sum = std::thread::spawn(move || pooled.as_slice().iter().sum::<f64>()).join().unwrap();
        assert_eq!(sum, 6.0);
    }
}
