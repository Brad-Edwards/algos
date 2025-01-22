use algos::cs::sort::MergeSortBuilder;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use rand::prelude::*;
use rand::rngs::StdRng;

fn generate_random_array(size: usize) -> Vec<i32> {
    let mut rng = StdRng::seed_from_u64(42); // Fixed seed for reproducibility
    (0..size).map(|_| rng.gen()).collect()
}

fn generate_nearly_sorted_array(size: usize) -> Vec<i32> {
    let mut arr: Vec<i32> = (0..size as i32).collect();
    // Swap about 5% of elements randomly
    let swaps = size / 20;
    let mut rng = StdRng::seed_from_u64(42);
    for _ in 0..swaps {
        let i = rng.gen_range(0..size);
        let j = rng.gen_range(0..size);
        arr.swap(i, j);
    }
    arr
}

fn bench_mergesort(c: &mut Criterion) {
    let mut group = c.benchmark_group("mergesort");

    // Test different array sizes
    for size in [100, 1000, 10000, 100000].iter() {
        // Random arrays
        group.bench_with_input(BenchmarkId::new("random", size), size, |b, &size| {
            b.iter_batched(
                || generate_random_array(size),
                |mut arr| MergeSortBuilder::new().sort(&mut arr),
                criterion::BatchSize::LargeInput,
            )
        });

        // Nearly sorted arrays
        group.bench_with_input(BenchmarkId::new("nearly_sorted", size), size, |b, &size| {
            b.iter_batched(
                || generate_nearly_sorted_array(size),
                |mut arr| MergeSortBuilder::new().sort(&mut arr),
                criterion::BatchSize::LargeInput,
            )
        });

        // Parallel sorting for large arrays
        if *size >= 10000 {
            group.bench_with_input(
                BenchmarkId::new("parallel_random", size),
                size,
                |b, &size| {
                    b.iter_batched(
                        || generate_random_array(size),
                        |mut arr| MergeSortBuilder::new().parallel(true).sort(&mut arr),
                        criterion::BatchSize::LargeInput,
                    )
                },
            );
        }
    }

    group.finish();
}

fn bench_insertion_threshold(c: &mut Criterion) {
    let mut group = c.benchmark_group("insertion_threshold");
    let size = 10000;

    for threshold in [8, 16, 32, 64].iter() {
        group.bench_with_input(
            BenchmarkId::new("threshold", threshold),
            threshold,
            |b, &threshold| {
                b.iter_batched(
                    || generate_random_array(size),
                    |mut arr| {
                        MergeSortBuilder::new()
                            .insertion_threshold(threshold)
                            .sort(&mut arr)
                    },
                    criterion::BatchSize::LargeInput,
                )
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_mergesort, bench_insertion_threshold);
criterion_main!(benches);
