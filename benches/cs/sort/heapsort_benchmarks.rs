use algos::cs::sort::heap_sort;
use criterion::{criterion_group, criterion_main, BatchSize, Criterion};
mod common;
use common::{generate_data, DISTRIBUTIONS, SIZES};

fn benchmark_heapsort(c: &mut Criterion) {
    let mut group = c.benchmark_group("heapsort");
    group.sample_size(10);

    for &size in &SIZES {
        for dist in &DISTRIBUTIONS {
            let bench_name = format!("heapsort_{}_{}", dist, size);
            group.bench_function(&bench_name, |b| {
                b.iter_batched(
                    || generate_data(size, dist),
                    |mut data| heap_sort(&mut data).expect("Sort should succeed"),
                    BatchSize::LargeInput,
                )
            });
        }
    }

    group.finish();
}

#[cfg(feature = "parallel")]
fn benchmark_parallel_heapsort(c: &mut Criterion) {
    let sizes = [100_000, 1_000_000, 10_000_000]; // Keep larger sizes for parallel

    let mut group = c.benchmark_group("parallel_heapsort");
    group.sample_size(10);

    for &size in &sizes {
        for dist in &DISTRIBUTIONS {
            let bench_name = format!("parallel_heapsort_{}_{}", dist, size);
            group.bench_function(&bench_name, |b| {
                b.iter_batched(
                    || generate_data(size, dist),
                    |mut data| heap_sort(&mut data).expect("Sort should succeed"),
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
                    |mut data| heap_sort(&mut data).expect("Sort should succeed"),
                    BatchSize::LargeInput,
                )
            });
        }
    }

    group.finish();
}

criterion_group!(benches, benchmark_heapsort);

#[cfg(feature = "parallel")]
criterion_group!(parallel_benches, benchmark_parallel_heapsort);

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
