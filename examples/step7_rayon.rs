//! Run: `cargo run --release --example step7_rayon`
//! (use `--release`: the sequential/parallel gap is dominated by
//! optimizer output on the inner loop, not by anything Rayon-specific —
//! a debug build makes both look artificially slow and the comparison
//! meaningless)
//!
//! Sequential vs. Rayon-parallel `add`, timed at a few sizes. This is
//! deliberately a Rust-vs-Rust comparison, not Rust-vs-NumPy: the core is
//! still f64-only with two binary ufuncs and no reductions, so measuring
//! it against NumPy now would say more about "how fast is one loop" than
//! about whether this is a viable NumPy replacement (see
//! `python/README.md`). That comparison belongs later in `NumPy.md`'s
//! plan, once there's enough surface area for it to mean something.
//!
//! What this *does* show, honestly: parallelism has a fixed cost (Rayon
//! has to split work and join results across threads), and paying that
//! cost is a poor trade for a small array — you should expect small
//! inputs to show the parallel path *losing*, not winning. Real NumPy
//! ufuncs don't parallelize at all, in part for exactly this reason: a
//! naive always-parallel ufunc would be worse than NumPy for by far the
//! most common case (small-to-medium arrays), and only better for large
//! ones — a real port would need a size-based threshold, not just an
//! `_parallel` suffix, before this could be turned on by default.
//!
//! It also shows something less flattering: on this machine (16 cores)
//! the large-array speedup measured well under 16x. That's not
//! measurement noise — `zip_with`/`zip_with_parallel` still allocate a
//! fresh `Vec<usize>` per element for the index (see the doc comment on
//! `zip_with_parallel`), and the global allocator is one shared resource
//! every thread contends on. More threads means more contention on that
//! one lock, which caps the achievable speedup regardless of how many
//! idle cores there are. This is a concrete, measured argument for NEP
//! 10's real `NpyIter` design (a cache-coherent iterator with no
//! per-element allocation) — not a hypothetical one.

use rustnumpy::{add, add_parallel, NdArray};
use std::time::Instant;

fn bench_add(len: usize) {
    let shape = &[len];
    let a = NdArray::from_vec((0..len).map(|i| i as f64).collect(), shape).unwrap();
    let b = NdArray::from_vec((0..len).map(|i| i as f64 * 2.0).collect(), shape).unwrap();

    // Run a few times and keep the minimum: the first call pays for
    // Rayon's global thread pool spinning up, and OS scheduling noise
    // otherwise dominates the signal at these timescales.
    let runs = 5;

    let sequential_min = (0..runs)
        .map(|_| {
            let start = Instant::now();
            std::hint::black_box(add(&a.view(), &b.view()).unwrap());
            start.elapsed()
        })
        .min()
        .unwrap();

    let parallel_min = (0..runs)
        .map(|_| {
            let start = Instant::now();
            std::hint::black_box(add_parallel(&a.view(), &b.view()).unwrap());
            start.elapsed()
        })
        .min()
        .unwrap();

    let speedup = sequential_min.as_secs_f64() / parallel_min.as_secs_f64();
    println!(
        "  len={len:>10}  sequential={sequential_min:>10.2?}  parallel={parallel_min:>10.2?}  speedup={speedup:.2}x"
    );
}

fn main() {
    println!("rayon thread pool: {} threads\n", rayon::current_num_threads());
    println!("add() vs add_parallel(), best of 5 runs each:");
    for len in [1_000, 100_000, 10_000_000, 100_000_000] {
        bench_add(len);
    }
    println!(
        "\nExpect small `len` to favor sequential (thread split/join overhead dominates)\n\
         and large `len` to favor parallel (the actual work dominates) — that crossover\n\
         point is exactly what a real size-based dispatch threshold would need to find.\n\
         \n\
         Also expect the large-`len` speedup to land well under {}x (this machine's core\n\
         count): each element still heap-allocates its index (see zip_with_parallel's doc\n\
         comment), and threads contend on the one global allocator lock — a real cost, not\n\
         measurement noise, and the concrete reason a cache-coherent NEP 10 iterator (no\n\
         per-element allocation) would matter here.",
        rayon::current_num_threads()
    );
}
