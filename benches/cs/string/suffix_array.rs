use algos::cs::string::suffix_array::SuffixArray;
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

const BENCH_TEXT: &str = "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.";
const PATTERNS: [&str; 4] = ["dolor", "ipsum", "exercitation", "nonexistent"];

fn bench_suffix_array(c: &mut Criterion) {
    let mut group = c.benchmark_group("suffix_array");

    // Construction benchmarks
    group.bench_function("construction/short", |b| {
        b.iter(|| SuffixArray::new(black_box(BENCH_TEXT)))
    });

    let long_text = "a".repeat(10000) + "b";
    group.bench_function("construction/long", |b| {
        b.iter(|| SuffixArray::new(black_box(&long_text)))
    });

    // Search benchmarks with different text sizes
    let text_sizes = [100, 1000, 10000];
    for size in text_sizes.iter() {
        let text = "a".repeat(*size) + "b";
        let sa = SuffixArray::new(&text);
        group.bench_with_input(BenchmarkId::new("search/text_size", size), size, |b, _| {
            b.iter(|| black_box(sa.find_all("aaa").unwrap()))
        });
    }

    // Pattern length benchmarks
    let patterns = ["a", "aa", "aaa", "aaaa", "aaaaa"];
    let text = "a".repeat(1000);
    let sa = SuffixArray::new(&text);
    for pattern in patterns.iter() {
        group.bench_with_input(
            BenchmarkId::new("search/pattern_length", pattern.len()),
            &pattern.len(),
            |b, _| b.iter(|| black_box(sa.find_all(pattern).unwrap())),
        );
    }

    // Multiple pattern search
    let sa = SuffixArray::new(BENCH_TEXT);
    group.bench_function("search/multiple_patterns", |b| {
        b.iter(|| {
            for pattern in PATTERNS.iter() {
                black_box(sa.find_all(pattern).unwrap());
            }
        })
    });

    // Unicode text
    let unicode_text = "こんにちは世界".repeat(100);
    let sa = SuffixArray::new(&unicode_text);
    group.bench_function("search/unicode", |b| {
        b.iter(|| black_box(sa.find_all("にち").unwrap()))
    });

    // Compare with naive search
    let pattern = "dolor";
    let sa = SuffixArray::new(BENCH_TEXT);

    group.bench_function("comparison/suffix_array", |b| {
        b.iter(|| black_box(sa.find_all(pattern).unwrap()))
    });

    group.bench_function("comparison/naive", |b| {
        b.iter(|| {
            let mut results = Vec::new();
            for i in 0..BENCH_TEXT.len() {
                if BENCH_TEXT[i..].starts_with(pattern) {
                    results.push(i);
                }
            }
            black_box(results)
        })
    });

    // Overlapping patterns
    let overlap_text = "a".repeat(1000);
    let sa = SuffixArray::new(&overlap_text);
    group.bench_function("search/overlapping", |b| {
        b.iter(|| black_box(sa.find_all("aa").unwrap()))
    });

    group.finish();
}

criterion_group!(benches, bench_suffix_array);
criterion_main!(benches);
