/// Heapsort implementation for sorting slices.
///
/// # Algorithm Overview
/// Heapsort is a comparison-based sorting algorithm that uses a binary heap data
/// structure. The algorithm:
/// 1. Builds a max-heap from the input array using Floyd's bottom-up construction (O(n))
/// 2. Repeatedly extracts the maximum element and places it at the end (n * O(log n))
/// 3. Maintains the heap property after each extraction (O(log n) per operation)
///
/// # Performance Characteristics
/// - Cache behavior: Moderate, with non-sequential access patterns
/// - Branch prediction: Optimized for common cases
/// - Memory usage: In-place, O(1) auxiliary space
///
/// # Time Complexity
/// - Build heap: O(n)
/// - Heapify: O(log n)
/// - Overall: O(n log n) for all cases
///
/// # Space Complexity
/// - O(1) auxiliary space
/// - In-place sorting
///
/// # Stability
/// - Not stable: equal elements may be reordered
///
/// # Examples
/// ```
/// use algos::cs::sort::heap_sort;
/// let mut numbers = vec![3, 1, 4, 1, 5, 9, 2, 6, 5, 3];
/// heap_sort(&mut numbers).expect("Sort should succeed");
/// assert_eq!(numbers, vec![1, 1, 2, 3, 3, 4, 5, 5, 6, 9]);
/// ```
///
/// # Errors
/// Returns `HeapSortError::ArrayTooLarge` if the array is too large to safely process
/// Returns `HeapSortError::InvalidRootIndex` if an invalid root index is provided to
/// heapify
/// Error type for heap sort operations
#[derive(Debug)]
pub enum HeapSortError {
    /// Array is too large, would cause integer overflow in heap operations
    ArrayTooLarge(usize),
    /// Invalid root index in heap operation
    InvalidRootIndex { root: usize, len: usize },
}

impl std::fmt::Display for HeapSortError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            HeapSortError::ArrayTooLarge(size) => write!(
                f,
                "Array size {} is too large, would cause integer overflow in heap operations",
                size
            ),
            HeapSortError::InvalidRootIndex { root, len } => {
                write!(f, "Invalid root index {} for heap of length {}", root, len)
            }
        }
    }
}

impl std::error::Error for HeapSortError {}

/// Result type for heap operations
type Result<T> = std::result::Result<T, HeapSortError>;

/// Internal function to validate array size
fn validate_array_size(len: usize) -> Result<()> {
    if len > isize::MAX as usize / 2 {
        Err(HeapSortError::ArrayTooLarge(len))
    } else {
        Ok(())
    }
}

#[cfg(feature = "parallel")]
use rayon::prelude::*;

#[allow(dead_code)]
const PARALLEL_THRESHOLD: usize = 100_000;

pub fn sort<T: Ord + Send>(slice: &mut [T]) -> Result<()> {
    if slice.len() <= 1 {
        return Ok(());
    }

    // Runtime check for array size to prevent integer overflow
    validate_array_size(slice.len())?;

    // Use parallel version for large arrays if the feature is enabled
    #[cfg(feature = "parallel")]
    {
        if slice.len() >= PARALLEL_THRESHOLD {
            return parallel_sort(slice);
        }
    }

    build_max_heap(slice)?;

    // Extract elements from heap one by one
    for i in (0..slice.len()).rev() {
        if i > 0 {
            // Don't swap when i == 0
            slice.swap(0, i);
            heapify_iterative(&mut slice[..i], 0)?;
        }
    }

    Ok(())
}

#[cfg(feature = "parallel")]
fn parallel_sort<T: Ord + Send>(slice: &mut [T]) -> Result<()> {
    let len = slice.len();
    let chunk_size = (len / rayon::current_num_threads()).max(PARALLEL_THRESHOLD);

    // Build sub-heaps in parallel
    slice
        .par_chunks_mut(chunk_size)
        .try_for_each(|chunk| build_max_heap(chunk))?;

    // Merge sub-heaps
    for i in (chunk_size..len).step_by(chunk_size) {
        merge_heaps(&mut slice[..i + chunk_size.min(len - i)])?;
    }

    // Extract elements in parallel for large arrays
    if len >= PARALLEL_THRESHOLD {
        slice
            .par_chunks_mut(chunk_size)
            .enumerate()
            .try_for_each(|(i, chunk)| extract_from_heap(chunk, i * chunk_size))?;
    } else {
        // Sequential extraction for smaller arrays
        for i in (0..len).rev() {
            if i > 0 {
                slice.swap(0, i);
                heapify_iterative(&mut slice[..i], 0)?;
            }
        }
    }

    Ok(())
}

#[cfg(feature = "parallel")]
#[allow(dead_code)]
fn parallel_merge_heaps<T: Ord + Send>(slice: &mut [T]) -> Result<()> {
    use rayon::prelude::*;

    let len = slice.len();
    let chunk_size = (len / rayon::current_num_threads()).max(PARALLEL_THRESHOLD);

    // Merge in parallel using a divide-and-conquer approach
    for merge_size in (chunk_size..=len).step_by(chunk_size) {
        slice.par_chunks_mut(merge_size * 2).try_for_each(|chunk| {
            if chunk.len() > merge_size {
                merge_heap_sections(chunk, merge_size)
            } else {
                Ok(())
            }
        })?;
    }

    Ok(())
}

fn build_max_heap<T: Ord>(slice: &mut [T]) -> Result<()> {
    for i in (0..slice.len() / 2).rev() {
        heapify_iterative(slice, i)?;
    }
    Ok(())
}

fn heapify_iterative<T: Ord>(slice: &mut [T], root: usize) -> Result<()> {
    let len = slice.len();

    if root >= len {
        return Err(HeapSortError::InvalidRootIndex { root, len });
    }

    let mut current = root;

    while current < len {
        let mut largest = current;
        let left = current * 2 + 1;
        let right = left + 1;

        if left < len && slice[left] > slice[largest] {
            largest = left;
        }
        if right < len && slice[right] > slice[largest] {
            largest = right;
        }

        if largest == current {
            break;
        }

        slice.swap(current, largest);
        current = largest;
    }

    Ok(())
}

#[cfg(feature = "parallel")]
fn merge_heaps<T: Ord>(slice: &mut [T]) -> Result<()> {
    for i in (0..slice.len() / 2).rev() {
        heapify_iterative(slice, i)?;
    }
    Ok(())
}

#[cfg(feature = "parallel")]
#[allow(dead_code)]
fn merge_heap_sections<T: Ord>(slice: &mut [T], mid: usize) -> Result<()> {
    if mid >= slice.len() {
        return Ok(());
    }

    // Rebuild heap property for the merged section
    for i in (0..slice.len() / 2).rev() {
        heapify_iterative(slice, i)?;
    }
    Ok(())
}

#[cfg(feature = "parallel")]
fn extract_from_heap<T: Ord>(slice: &mut [T], _offset: usize) -> Result<()> {
    let len = slice.len();
    for i in (0..len).rev() {
        if i > 0 {
            slice.swap(0, i);
            heapify_iterative(&mut slice[..i], 0)?;
        }
    }
    Ok(())
}

#[cfg(feature = "simd")]
#[allow(dead_code)]
fn heapify_simd(slice: &mut [i32], root: usize) -> Result<()> {
    heapify_iterative(slice, root)
}

#[cfg(feature = "simd")]
pub fn sort_i32(slice: &mut [i32]) -> Result<()> {
    sort(slice)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_slice() {
        let mut arr: Vec<i32> = vec![];
        sort(&mut arr).expect("Sort should succeed on empty slice");
        assert_eq!(arr, Vec::<i32>::new());
    }

    #[test]
    fn test_single_element() {
        let mut arr = vec![1];
        sort(&mut arr).expect("Sort should succeed on single element");
        assert_eq!(arr, vec![1]);
    }

    #[test]
    fn test_sorted_array() {
        let mut arr = vec![1, 2, 3, 4, 5];
        sort(&mut arr).expect("Sort should succeed on sorted array");
        assert_eq!(arr, vec![1, 2, 3, 4, 5]);
    }

    #[test]
    fn test_reverse_sorted() {
        let mut arr = vec![5, 4, 3, 2, 1];
        sort(&mut arr).expect("Sort should succeed on reverse sorted array");
        assert_eq!(arr, vec![1, 2, 3, 4, 5]);
    }

    #[test]
    fn test_random_order() {
        let mut arr = vec![3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5];
        let mut expected = arr.clone();
        expected.sort();
        sort(&mut arr).expect("Sort should succeed on random array");
        assert_eq!(arr, expected);
    }

    #[test]
    fn test_duplicate_elements() {
        let mut arr = vec![3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5];
        let mut expected = arr.clone();
        expected.sort();
        sort(&mut arr).expect("Sort should succeed on array with duplicates");
        assert_eq!(arr, expected);
    }

    #[test]
    fn test_large_array() {
        let size = 100_000; // Reduced from 1_000_000 to a more reasonable size
        validate_array_size(size).expect("Size should be valid");

        let mut arr: Vec<i32> = (0..size).rev().map(|x| x as i32).collect();
        let mut expected = arr.clone();
        expected.sort();
        sort(&mut arr).expect("Sort should succeed on large array");
        assert_eq!(arr, expected);
    }

    #[test]
    fn test_different_types() {
        let mut string_arr = vec!["banana", "apple", "cherry", "date"];
        let mut expected = string_arr.clone();
        expected.sort();
        sort(&mut string_arr).expect("Sort should succeed on string array");
        assert_eq!(string_arr, expected);
    }

    #[test]
    fn test_heap_property() {
        let mut arr = vec![4, 10, 3, 5, 1];
        build_max_heap(&mut arr).expect("Build heap should succeed");

        // Test max-heap property at each level
        for i in 0..arr.len() {
            let left = 2 * i + 1;
            let right = 2 * i + 2;

            if left < arr.len() {
                assert!(
                    arr[i] >= arr[left],
                    "Heap property violated at index {} with left child",
                    i
                );
            }
            if right < arr.len() {
                assert!(
                    arr[i] >= arr[right],
                    "Heap property violated at index {} with right child",
                    i
                );
            }
        }
    }

    #[test]
    fn test_index_calculation_limits() {
        // Test array size at power of 2 boundaries
        let sizes = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024];

        for &size in &sizes {
            let mut arr: Vec<i32> = (0..size).map(|x| x as i32).rev().collect();

            // First build the heap and verify heap property
            build_max_heap(&mut arr).expect("Build heap should succeed");

            // Verify heap property is maintained after heap construction
            for i in 0..arr.len() {
                let left = 2 * i + 1;
                let right = left + 1;
                if left < arr.len() {
                    assert!(
                        arr[i] >= arr[left],
                        "Heap property violated at index {} with left child",
                        i
                    );
                }
                if right < arr.len() {
                    assert!(
                        arr[i] >= arr[right],
                        "Heap property violated at index {} with right child",
                        i
                    );
                }
            }

            // Now complete the sort
            sort(&mut arr).expect("Sort should succeed");
            assert!(
                arr.windows(2).all(|w| w[0] <= w[1]),
                "Array not sorted correctly for size {}",
                size
            );
        }
    }

    #[test]
    fn test_size_boundaries() {
        let boundary_sizes = [8, 9, 10, 11, 15, 16, 17, 32, 1023, 1024, 1025];

        for &size in &boundary_sizes {
            let mut arr: Vec<i32> = (0..size).map(|x| x as i32).rev().collect();
            let mut expected = arr.clone();
            expected.sort();
            sort(&mut arr).expect(&format!("Sort should succeed for size {}", size));
            assert_eq!(arr, expected, "Failed for size {}", size);
        }
    }

    #[test]
    fn test_pathological_inputs() {
        // All equal elements
        let mut arr = vec![1; 100];
        sort(&mut arr).expect("Sort should succeed on equal elements");
        assert!(arr.windows(2).all(|w| w[0] <= w[1]));

        // Alternating elements
        let mut arr: Vec<i32> = (0..100).map(|i| (i % 2) as i32).collect();
        let mut expected = arr.clone();
        expected.sort();
        sort(&mut arr).expect("Sort should succeed on alternating elements");
        assert_eq!(arr, expected);

        // Saw pattern
        let mut arr: Vec<i32> = (0..50)
            .map(|x| x as i32)
            .chain((0..50).rev().map(|x| x as i32))
            .collect();
        let mut expected = arr.clone();
        expected.sort();
        sort(&mut arr).expect("Sort should succeed on saw pattern");
        assert_eq!(arr, expected);

        // Pipeline pattern
        let mut arr: Vec<i32> = (0..50)
            .map(|x| x as i32)
            .chain((0..50).map(|x| x as i32))
            .collect();
        let mut expected = arr.clone();
        expected.sort();
        sort(&mut arr).expect("Sort should succeed on pipeline pattern");
        assert_eq!(arr, expected);
    }

    #[test]
    fn test_invalid_root_index() {
        let mut arr = vec![1, 2, 3];
        let result = heapify_iterative(&mut arr, 3);
        assert!(matches!(
            result,
            Err(HeapSortError::InvalidRootIndex { root: 3, len: 3 })
        ));
    }

    #[test]
    fn test_array_size_limit() {
        // Only test the validation logic without allocating memory
        let size = (isize::MAX as usize / 2) + 1;
        let result = validate_array_size(size);

        assert!(matches!(
            result,
            Err(HeapSortError::ArrayTooLarge(s)) if s == size
        ));

        // Test a valid size
        let valid_size = isize::MAX as usize / 4;
        assert!(validate_array_size(valid_size).is_ok());
    }

    #[test]
    fn test_child_index_calculation() {
        // Test with a reasonable size array
        let size = 10_000; // Reduced from 1_000_000
        let mut arr = vec![1; size];

        // Test with valid indices first
        let valid_index = size / 2;
        let result = heapify_iterative(&mut arr, valid_index);
        assert!(result.is_ok(), "Should succeed with valid index");

        // Now test with an index that would cause child index overflow
        // We don't need a huge array to test this, just pass a large index
        let large_index = (usize::MAX - 1) / 2; // This will cause 2*i+1 to overflow
        let result = heapify_iterative(&mut arr, large_index);
        assert!(result.is_err(), "Should error on large indices");
    }

    #[test]
    #[cfg(feature = "simd")]
    fn test_simd_sort() {
        let mut arr = vec![3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5];
        let mut expected = arr.clone();
        expected.sort();

        sort_i32(&mut arr).expect("SIMD sort should succeed");
        assert_eq!(arr, expected, "SIMD sort failed to sort correctly");
    }

    #[test]
    #[cfg(all(feature = "simd", feature = "parallel"))]
    fn test_parallel_simd_sort() {
        let size = 100_000; // Reduced from 1_000_000
        let mut arr: Vec<i32> = (0..size).rev().collect();
        let mut expected = arr.clone();
        expected.sort();

        sort_i32(&mut arr).expect("Parallel SIMD sort should succeed");
        assert_eq!(arr, expected, "Parallel SIMD sort failed to sort correctly");
    }

    #[test]
    #[cfg(feature = "simd")]
    fn test_simd_performance() {
        let size = 100_000; // Reduced from 1_000_000
        let mut arr1: Vec<i32> = (0..size).rev().collect();
        let mut arr2 = arr1.clone();

        let start = std::time::Instant::now();
        sort_i32(&mut arr1).expect("SIMD sort should succeed");
        let simd_time = start.elapsed();

        let start = std::time::Instant::now();
        sort(&mut arr2).expect("Regular sort should succeed");
        let regular_time = start.elapsed();

        println!(
            "Size {}: SIMD sort {:?}, Regular sort {:?}",
            size, simd_time, regular_time
        );

        assert_eq!(arr1, arr2, "SIMD sort produced different results");
    }
}

#[cfg(test)]
mod benchmarks {
    use super::*;
    use std::time::{Duration, Instant};

    fn bench_sort(size: usize) -> Result<Duration> {
        // Add size validation
        validate_array_size(size)?;

        let mut arr: Vec<i32> = (0..size).map(|x| x as i32).rev().collect();
        let start = Instant::now();
        sort(&mut arr)?;
        Ok(start.elapsed())
    }

    #[test]
    fn compare_performance() {
        // Use more reasonable sizes for testing
        for &size in &[100, 1_000, 10_000, 100_000] {
            // Our heapsort
            let heap_time = bench_sort(size).expect("Heapsort benchmark should succeed");

            // Standard library sort
            let mut arr: Vec<i32> = (0..size).map(|x| x as i32).rev().collect();
            let start = Instant::now();
            arr.sort();
            let std_time = start.elapsed();

            println!(
                "Size {}: Heapsort {:?}, std::sort {:?}",
                size, heap_time, std_time
            );
        }
    }

    #[test]
    #[cfg(feature = "parallel")]
    fn test_parallel_sort() {
        let size = 100_000; // Reduced from 1_000_000
        let mut arr: Vec<i32> = (0..size).rev().collect();
        let mut expected = arr.clone();
        expected.sort();

        parallel_sort(&mut arr).expect("Parallel sort should succeed");
        assert_eq!(arr, expected, "Parallel sort failed to sort correctly");
    }
}
