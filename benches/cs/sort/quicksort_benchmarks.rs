use algos::cs::sort::quick_sort;
use criterion::{criterion_group, criterion_main, BatchSize, Criterion};
mod common;
use common::{generate_data, DISTRIBUTIONS, SIZES};

fn benchmark_quicksort(c: &mut Criterion) {
    let mut group = c.benchmark_group("quicksort");
    group.sample_size(10);

    for &size in &SIZES {
        for dist in &DISTRIBUTIONS {
            let bench_name = format!("quicksort_{}_{}", dist, size);
            group.bench_function(&bench_name, |b| {
                b.iter_batched(
                    || generate_data(size, dist),
                    |mut data| quick_sort(&mut data),
                    BatchSize::LargeInput,
                )
            });
        }
    }

    group.finish();
}

#[cfg(feature = "parallel")]
fn benchmark_parallel_quicksort(c: &mut Criterion) {
    let sizes = [100_000, 1_000_000, 10_000_000]; // Keep larger sizes for parallel

    let mut group = c.benchmark_group("parallel_quicksort");
    group.sample_size(10);

    for &size in &sizes {
        for dist in &DISTRIBUTIONS {
            let bench_name = format!("parallel_quicksort_{}_{}", dist, size);
            group.bench_function(&bench_name, |b| {
                b.iter_batched(
                    || generate_data(size, dist),
                    |mut data| quick_sort(&mut data),
                    BatchSize::LargeInput,
                )
            });
        }
    }

    group.finish();
}

#[cfg(feature = "simd")]
fn benchmark_simd_sort(c: &mut Criterion) {
    let mut group = c.benchmark_group("simd_sort");
    group.sample_size(10);

    for &size in &SIZES {
        for dist in &DISTRIBUTIONS {
            let bench_name = format!("simd_sort_{}_{}", dist, size);
            group.bench_function(&bench_name, |b| {
                b.iter_batched(
                    || generate_data(size, dist),
                    |mut data| quick_sort(&mut data),
                    BatchSize::LargeInput,
                )
            });
        }
    }

    group.finish();
}

criterion_group!(benches, benchmark_quicksort);

#[cfg(feature = "parallel")]
criterion_group!(parallel_benches, benchmark_parallel_quicksort);

#[cfg(feature = "simd")]
criterion_group!(simd_benches, benchmark_simd_sort);

// Configuration remains the same
#[cfg(not(any(feature = "parallel", feature = "simd")))]
criterion_main!(benches);

#[cfg(all(feature = "parallel", not(feature = "simd")))]
criterion_main!(benches, parallel_benches);

#[cfg(all(feature = "simd", not(feature = "parallel")))]
criterion_main!(benches, simd_benches);

#[cfg(all(feature = "parallel", feature = "simd"))]
criterion_main!(benches, parallel_benches, simd_benches);
