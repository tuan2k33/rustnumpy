use rustnumpy::{BumpArena, PooledVec, System};

fn main() {
    println!("-- PooledVec backed by the System allocator --");
    let sys_vec = PooledVec::from_slice(&System, &[1.0, 2.0, 3.0]).unwrap();
    println!("  values: {:?}", sys_vec.as_slice());

    println!("\n-- PooledVec backed by a 256-byte BumpArena --");
    let arena = BumpArena::with_capacity(256);
    println!("  arena: used={} capacity={}", arena.used(), arena.capacity());

    let a = PooledVec::from_slice(&arena, &[1.0, 2.0, 3.0]).unwrap();
    println!("  allocated a={:?}, arena.used() = {}", a.as_slice(), arena.used());

    let b = PooledVec::from_slice(&arena, &[10.0, 20.0]).unwrap();
    println!("  allocated b={:?}, arena.used() = {}", b.as_slice(), arena.used());

    let c = PooledVec::from_slice(&arena, &[100.0]).unwrap();
    println!("  allocated c={:?}, arena.used() = {}", c.as_slice(), arena.used());

    println!("\n  after all three allocations:");
    println!("    a = {:?}", a.as_slice());
    println!("    b = {:?}", b.as_slice());
    println!("    c = {:?}", c.as_slice());

    drop(a);
    drop(b);
    drop(c);
    println!(
        "\n  dropped a/b/c individually — arena.used() is still {} (bump arenas don't reclaim per-object)",
        arena.used()
    );
    println!("  the whole 256-byte block is freed in one shot when `arena` drops at the end of main()");
}
