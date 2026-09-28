use rustnumpy::{add, add_parallel, NdArray};
use std::time::Instant;

fn bench_add(len: usize) {
    let shape = &[len];
    let a = NdArray::from_vec((0..len).map(|i| i as f64).collect(), shape).unwrap();
    let b = NdArray::from_vec((0..len).map(|i| i as f64 * 2.0).collect(), shape).unwrap();

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
