use criterion::{criterion_group, criterion_main, BatchSize, Criterion};
mod common;
use common::{generate_data, DISTRIBUTIONS, SIZES};

fn benchmark_std_sort(c: &mut Criterion) {
    let mut group = c.benchmark_group("std_sort");
    group.sample_size(10);

    for &size in &SIZES {
        for dist in &DISTRIBUTIONS {
            let bench_name = format!("std_sort_{}_{}", dist, size);
            group.bench_function(&bench_name, |b| {
                b.iter_batched(
                    || generate_data(size, dist),
                    |mut data| data.sort(),
                    BatchSize::LargeInput,
                )
            });
        }
    }

    group.finish();
}

criterion_group!(benches, benchmark_std_sort);
criterion_main!(benches);
