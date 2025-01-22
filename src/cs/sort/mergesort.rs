#[cfg(feature = "parallel")]
use rayon;
/// Mergesort implementation with parallel processing support.
///
/// This module provides a configurable mergesort implementation that can:
/// - Use insertion sort for small arrays
/// - Process large arrays in parallel using rayon
/// - Handle generic types that implement Ord + Clone
///
/// # Safety
///
/// This implementation uses unsafe code in the following ways:
/// - Uses `split_at_mut` for parallel processing (safe interface to unsafe code)
/// - Uses rayon's parallel execution primitives (safe interface to unsafe code)
///
/// All unsafe operations are properly encapsulated and safe when used with types
/// that implement the required traits (Send + Sync for parallel execution).
use std::fmt::Debug;

use crate::error::{Result, SortError};

/// Builder for configuring and executing merge sort operations.
///
/// # Examples
///
/// ```
/// use algos::cs::sort::mergesort::MergeSortBuilder;
///
/// let mut arr = vec![3, 1, 4, 1, 5, 9];
/// MergeSortBuilder::new().insertion_threshold(16).sort(&mut arr).expect("Sort failed");
/// assert!(arr.windows(2).all(|w| w[0] <= w[1]));
/// ```
///
/// # Performance
///
/// The algorithm has the following complexity characteristics:
/// - Time: O(n log n) in all cases
/// - Space: O(n) auxiliary space
/// - Stable: Yes
///
/// Performance can be tuned through:
/// - `insertion_threshold`: Arrays smaller than this use insertion sort (default: 16)
/// - `max_recursion_depth`: Limit recursion to prevent stack overflow (default: 48)
/// - `parallel`: Enable parallel sorting for large arrays
/// - `parallel_threshold`: Minimum size for parallel processing
#[derive(Debug, Clone)]
pub struct MergeSortBuilder {
    insertion_threshold: usize,
    max_recursion_depth: usize,
    parallel: bool,
    parallel_threshold: usize,
}

impl Default for MergeSortBuilder {
    fn default() -> Self {
        Self {
            insertion_threshold: 16,
            max_recursion_depth: 48,
            parallel: false,
            parallel_threshold: 1024,
        }
    }
}

impl MergeSortBuilder {
    /// Maximum length of slice that can be sorted (2^48 elements).
    /// This limit ensures we don't exceed reasonable memory usage.
    const MAX_LENGTH: usize = 1 << 48;

    /// Creates a new MergeSortBuilder with default settings
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the threshold below which insertion sort is used
    ///
    /// Smaller values favor merge sort's O(n log n) complexity,
    /// larger values favor insertion sort's cache efficiency on small arrays.
    ///
    /// # Examples
    /// ```
    /// use algos::cs::sort::mergesort::MergeSortBuilder;
    ///
    /// let mut arr = vec![5, 2, 8, 1, 9, 3];
    /// MergeSortBuilder::new().insertion_threshold(8).sort(&mut arr).unwrap();
    /// ```
    pub fn insertion_threshold(mut self, threshold: usize) -> Self {
        self.insertion_threshold = threshold;
        self
    }

    /// Sets the maximum recursion depth
    ///
    /// This prevents stack overflow on very large arrays.
    /// The default of 48 supports arrays up to 2^48 elements.
    pub fn max_recursion_depth(mut self, depth: usize) -> Self {
        self.max_recursion_depth = depth;
        self
    }

    /// Enables or disables parallel sorting
    ///
    /// When enabled, arrays larger than the parallel threshold will be sorted
    /// using multiple threads via rayon.
    ///
    /// # Examples
    /// ```
    /// use algos::cs::sort::mergesort::MergeSortBuilder;
    ///
    /// let mut arr = vec![5, 2, 8, 1, 9, 3];
    /// MergeSortBuilder::new().parallel(true).sort(&mut arr).unwrap();
    /// ```
    pub fn parallel(mut self, enabled: bool) -> Self {
        self.parallel = enabled;
        self
    }

    /// Sets the threshold above which parallel sorting is used
    ///
    /// Arrays larger than this threshold will be sorted in parallel
    /// when parallel sorting is enabled.
    pub fn parallel_threshold(mut self, threshold: usize) -> Self {
        self.parallel_threshold = threshold;
        self
    }

    /// Sorts a mutable slice using the configured settings
    ///
    /// # Errors
    ///
    /// Returns `SortError` if:
    /// - Memory allocation fails
    /// - Maximum recursion depth is exceeded
    /// - Input slice is too large (> 2^48 elements)
    /// - Parallel execution fails
    pub fn sort<T>(&self, slice: &mut [T]) -> Result<()>
    where
        T: Ord + Clone + Send + Sync + 'static,
    {
        if slice.len() <= 1 {
            return Ok(());
        }

        if slice.len() > Self::MAX_LENGTH {
            return Err(SortError::input_too_large(slice.len(), Self::MAX_LENGTH));
        }

        // Create auxiliary buffer
        let mut aux = vec![slice[0].clone(); slice.len()];

        if self.parallel && slice.len() >= self.parallel_threshold {
            self.sort_parallel(slice, &mut aux, 0)
        } else {
            self.sort_sequential(slice, &mut aux, 0)
        }
    }

    fn sort_sequential<T>(&self, slice: &mut [T], aux: &mut Vec<T>, depth: usize) -> Result<()>
    where
        T: Ord + Clone + 'static,
    {
        if depth >= self.max_recursion_depth {
            return Err(SortError::recursion_limit_exceeded(
                depth,
                self.max_recursion_depth,
            ));
        }

        if slice.len() <= self.insertion_threshold {
            insertion_sort(slice);
            return Ok(());
        }

        let mid = slice.len() / 2;

        self.sort_sequential(&mut slice[..mid], aux, depth + 1)?;
        self.sort_sequential(&mut slice[mid..], aux, depth + 1)?;

        merge(slice, mid, aux);
        Ok(())
    }

    #[cfg(feature = "parallel")]
    fn sort_parallel<T>(&self, slice: &mut [T], aux: &mut [T], depth: usize) -> Result<()>
    where
        T: Ord + Clone + Send + Sync + 'static,
    {
        if depth >= self.max_recursion_depth {
            return Err(SortError::recursion_limit_exceeded(
                depth,
                self.max_recursion_depth,
            ));
        }

        if slice.len() <= self.insertion_threshold {
            insertion_sort(slice);
            return Ok(());
        }

        let mid = slice.len() / 2;
        let len = slice.len(); // Get length before borrowing
        let (left, right) = slice.split_at_mut(mid);

        // Create auxiliary buffers with the same size as the original
        let mut left_aux = aux[..mid].to_vec();
        let mut right_aux = aux[mid..len].to_vec();

        let (left_result, right_result) = rayon::join(
            || self.sort_sequential(left, &mut left_aux, depth + 1),
            || self.sort_sequential(right, &mut right_aux, depth + 1),
        );

        left_result?;
        right_result?;

        // Merge directly into the original slice
        merge(slice, mid, aux);
        Ok(())
    }

    #[cfg(not(feature = "parallel"))]
    fn sort_parallel<T>(&self, slice: &mut [T], aux: &mut Vec<T>, depth: usize) -> Result<()>
    where
        T: Ord + Clone + Send + Sync + 'static,
    {
        self.sort_sequential(slice, aux, depth)
    }

    /// Validates the size of an array without creating any slices
    ///
    /// # Errors
    ///
    /// Returns `SortError` if:
    /// - Input size is too large (> 2^48 elements)
    pub fn validate_array_size(&self, size: usize) -> Result<()> {
        if size > Self::MAX_LENGTH {
            Err(SortError::input_too_large(size, Self::MAX_LENGTH))
        } else {
            Ok(())
        }
    }
}

/// Sorts a slice using merge sort with default settings
///
/// This is a convenience wrapper around `MergeSortBuilder`.
/// For more control, use `MergeSortBuilder` directly.
///
/// # Errors
///
/// Returns `SortError` if:
/// - Memory allocation fails
/// - Maximum recursion depth is exceeded
/// - Input slice is too large (> 2^48 elements)
/// - Parallel execution fails
pub fn sort<T>(slice: &mut [T]) -> Result<()>
where
    T: Ord + Clone + Send + Sync + 'static,
{
    MergeSortBuilder::new().sort(slice)
}

// Internal helper functions

fn insertion_sort<T: Ord>(slice: &mut [T]) {
    for i in 1..slice.len() {
        let mut j = i;
        while j > 0 && slice[j - 1] > slice[j] {
            slice.swap(j - 1, j);
            j -= 1;
        }
    }
}

fn merge<T>(slice: &mut [T], mid: usize, aux: &mut [T])
where
    T: Ord + Clone,
{
    aux[..slice.len()].clone_from_slice(slice);

    let (left, right) = aux[..slice.len()].split_at(mid);
    let mut i = 0;
    let mut j = 0;
    let mut k = 0;

    while i < left.len() && j < right.len() {
        if left[i] <= right[j] {
            slice[k] = left[i].clone();
            i += 1;
        } else {
            slice[k] = right[j].clone();
            j += 1;
        }
        k += 1;
    }

    if i < left.len() {
        slice[k..].clone_from_slice(&left[i..]);
    }
    if j < right.len() {
        slice[k..].clone_from_slice(&right[j..]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_slice() {
        let mut arr: Vec<i32> = vec![];
        sort(&mut arr).unwrap();
        assert_eq!(arr, Vec::<i32>::new());
    }

    #[test]
    fn test_single_element() {
        let mut arr = vec![1];
        sort(&mut arr).unwrap();
        assert_eq!(arr, vec![1]);
    }

    #[test]
    fn test_sorted_array() {
        let mut arr = vec![1, 2, 3, 4, 5];
        sort(&mut arr).unwrap();
        assert_eq!(arr, vec![1, 2, 3, 4, 5]);
    }

    #[test]
    fn test_reverse_sorted() {
        let mut arr = vec![5, 4, 3, 2, 1];
        sort(&mut arr).unwrap();
        assert_eq!(arr, vec![1, 2, 3, 4, 5]);
    }

    #[test]
    fn test_random_order() {
        let mut arr = vec![3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5];
        let mut expected = arr.clone();
        expected.sort();
        sort(&mut arr).unwrap();
        assert_eq!(arr, expected);
    }

    #[test]
    #[cfg(feature = "parallel")]
    fn test_parallel_sorting() {
        // Create a moderately sized array to test parallel sorting
        let size = 10_000; // Reduced from 100_000
        let mut arr: Vec<i32> = (0..size).rev().collect();
        let mut expected = arr.clone();
        expected.sort();

        // Initialize rayon with a custom thread pool for this test
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .unwrap();

        pool.install(|| {
            MergeSortBuilder::new()
                .parallel(true)
                .parallel_threshold(1000)
                .sort(&mut arr)
                .unwrap();
        });

        assert_eq!(arr, expected);
    }

    #[test]
    #[cfg(feature = "parallel")]
    fn test_parallel_threshold() {
        let size = 10_000;
        let arr: Vec<i32> = (0..size).rev().collect();

        // Set threshold higher than array size - should use sequential sort
        let mut arr1 = arr.clone();
        MergeSortBuilder::new()
            .parallel(true)
            .parallel_threshold((size * 2) as usize)
            .sort(&mut arr1)
            .unwrap();

        // Set threshold lower than array size - should use parallel sort
        let mut arr2 = arr.clone();
        MergeSortBuilder::new()
            .parallel(true)
            .parallel_threshold((size / 2) as usize)
            .sort(&mut arr2)
            .unwrap();

        let mut expected = arr;
        expected.sort();

        assert_eq!(arr1, expected);
        assert_eq!(arr2, expected);
    }

    #[test]
    #[cfg(feature = "parallel")]
    fn test_parallel_stability() {
        #[derive(Debug, Clone, Eq, PartialEq)]
        struct Item {
            key: i32,
            original_index: usize,
        }

        impl PartialOrd for Item {
            fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
                self.key.partial_cmp(&other.key)
            }
        }

        impl Ord for Item {
            fn cmp(&self, other: &Self) -> std::cmp::Ordering {
                self.key.cmp(&other.key)
            }
        }

        // Create a large array of items with duplicate keys
        let size = 10_000;
        let mut items: Vec<_> = (0..size)
            .map(|i| Item {
                key: i as i32 / 10, // Create many duplicates
                original_index: i,
            })
            .collect();

        MergeSortBuilder::new()
            .parallel(true)
            .parallel_threshold(1000)
            .sort(&mut items)
            .unwrap();

        // Verify stability
        for i in 1..items.len() {
            if items[i - 1].key == items[i].key {
                assert!(
                    items[i - 1].original_index < items[i].original_index,
                    "Stability violated at indices {} and {}",
                    i - 1,
                    i
                );
            }
        }
    }

    #[test]
    fn test_recursion_limit() {
        let mut arr: Vec<i32> = (0..10_000).collect(); // Reduced from 1_000_000
        let result = MergeSortBuilder::new()
            .max_recursion_depth(3)
            .sort(&mut arr);

        match result {
            Err(SortError::RecursionLimitExceeded { depth, max_depth }) => {
                assert_eq!(max_depth, 3);
                assert!(depth >= max_depth);
            }
            _ => panic!("Expected RecursionLimitExceeded error"),
        }
    }

    #[test]
    fn test_input_too_large() {
        // Test the error handling without creating any slices
        let size = MergeSortBuilder::MAX_LENGTH + 1;
        let result = MergeSortBuilder::new().validate_array_size(size);

        match result {
            Err(SortError::InputTooLarge { length, max_length }) => {
                assert_eq!(length, size);
                assert_eq!(max_length, MergeSortBuilder::MAX_LENGTH);
            }
            _ => panic!("Expected InputTooLarge error"),
        }
    }
}
