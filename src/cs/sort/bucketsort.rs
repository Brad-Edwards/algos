/// Bucket Sort implementation for sorting slices of floating-point numbers in the range
/// [0, 1).
///
/// # Algorithm Overview
/// Bucket sort works by:
/// 1. Creating n empty buckets (where n is the length of the input)
/// 2. Putting each element into its corresponding bucket based on its value
/// 3. Sorting each non-empty bucket (using insertion sort)
/// 4. Concatenating all buckets in order
///
/// # Time Complexity
/// - Best Case: Ω(n + k) when elements are uniformly distributed
/// - Average Case: Θ(n + k) when elements are uniformly distributed
/// - Worst Case: O(n²) when all elements go into the same bucket
///
/// # Space Complexity
/// - O(n + k) auxiliary space where k is the number of buckets
///
/// # Stability
/// - Stable if the underlying sort is stable (insertion sort in this case)
///
/// # Advantages
/// - Linear time complexity for uniformly distributed data
/// - Works well with floating-point numbers
/// - Can be parallelized easily
/// - Good cache performance due to locality of reference
///
/// # Limitations
/// - Requires uniformly distributed input for best performance
/// - Not in-place sorting algorithm
/// - Requires additional space
/// - Input must be in a known range (typically [0, 1))
pub fn sort(slice: &mut [f64]) {
    if slice.len() <= 1 {
        return;
    }

    let n = slice.len();

    // Create n empty buckets
    let mut buckets: Vec<Vec<f64>> = vec![Vec::new(); n];

    // Put array elements in different buckets
    for &num in slice.iter() {
        let idx = get_bucket_index(num, n);
        buckets[idx].push(num);
    }

    // Sort individual buckets
    for bucket in buckets.iter_mut() {
        insertion_sort(bucket);
    }

    // Concatenate all buckets into slice
    let mut index = 0;
    for bucket in buckets.iter() {
        for &value in bucket {
            slice[index] = value;
            index += 1;
        }
    }
}

/// Sorts a bucket using insertion sort
fn insertion_sort(bucket: &mut [f64]) {
    for i in 1..bucket.len() {
        let key = bucket[i];
        let mut j = i;

        while j > 0 && bucket[j - 1] > key {
            bucket[j] = bucket[j - 1];
            j -= 1;
        }

        bucket[j] = key;
    }
}

/// Determines the appropriate bucket index for a value
fn get_bucket_index(value: f64, num_buckets: usize) -> usize {
    let bucket_idx = (value * num_buckets as f64) as usize;
    // Handle edge case where value = 1.0
    bucket_idx.min(num_buckets - 1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::EPSILON;

    #[test]
    fn test_empty_slice() {
        let mut arr: Vec<f64> = vec![];
        sort(&mut arr);
        assert_eq!(arr, Vec::<f64>::new());
    }

    #[test]
    fn test_single_element() {
        let mut arr = vec![0.5];
        sort(&mut arr);
        assert_eq!(arr, vec![0.5]);
    }

    #[test]
    fn test_sorted_array() {
        let mut arr = vec![0.1, 0.2, 0.3, 0.4, 0.5];
        let expected = arr.clone();
        sort(&mut arr);
        assert_eq!(arr, expected);
    }

    #[test]
    fn test_reverse_sorted() {
        let mut arr = vec![0.9, 0.7, 0.5, 0.3, 0.1];
        sort(&mut arr);
        assert_eq!(arr, vec![0.1, 0.3, 0.5, 0.7, 0.9]);
    }

    #[test]
    fn test_random_order() {
        let mut arr = vec![0.3, 0.1, 0.4, 0.1, 0.5, 0.9, 0.2, 0.6, 0.5, 0.3, 0.5];
        let mut expected = arr.clone();
        expected.sort_by(|a, b| a.partial_cmp(b).unwrap());
        sort(&mut arr);
        assert_eq!(arr, expected);
    }

    #[test]
    fn test_duplicate_elements() {
        let mut arr = vec![0.5, 0.5, 0.5, 0.5, 0.5];
        sort(&mut arr);
        assert_eq!(arr, vec![0.5, 0.5, 0.5, 0.5, 0.5]);
    }

    #[test]
    fn test_large_array() {
        let mut arr: Vec<f64> = (0..1000).map(|x| (x as f64) / 1000.0).rev().collect();
        let mut expected = arr.clone();
        expected.sort_by(|a, b| a.partial_cmp(b).unwrap());
        sort(&mut arr);
        assert_eq!(arr, expected);
    }

    #[test]
    fn test_stability() {
        // Test stability by sorting pairs and checking if relative order is preserved
        #[derive(Debug, Clone, PartialEq)]
        struct Pair {
            key: f64,
            original_index: usize,
        }

        let pairs = vec![
            Pair {
                key: 0.5,
                original_index: 0,
            },
            Pair {
                key: 0.5,
                original_index: 1,
            },
            Pair {
                key: 0.7,
                original_index: 2,
            },
            Pair {
                key: 0.7,
                original_index: 3,
            },
        ];

        let mut values: Vec<f64> = pairs.iter().map(|p| p.key).collect();
        sort(&mut values);

        // Verify that relative order is preserved for equal keys
        for i in 0..pairs.len() - 1 {
            for j in i + 1..pairs.len() {
                if (pairs[i].key - pairs[j].key).abs() < EPSILON {
                    let pos_i = values
                        .iter()
                        .position(|&x| (x - pairs[i].key).abs() < EPSILON)
                        .unwrap();
                    let pos_j = values
                        .iter()
                        .rposition(|&x| (x - pairs[j].key).abs() < EPSILON)
                        .unwrap();
                    assert!(
                        pos_i < pos_j,
                        "Stability violated for equal elements at original positions {} and {}",
                        pairs[i].original_index,
                        pairs[j].original_index
                    );
                }
            }
        }
    }

    #[test]
    fn test_uniform_distribution() {
        // Test with uniformly distributed values
        let mut arr = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9];
        let expected = arr.clone();
        sort(&mut arr);
        assert_eq!(arr, expected);
    }

    #[test]
    fn test_clustered_values() {
        // Test with values clustered in a small range
        let mut arr = vec![0.51, 0.52, 0.53, 0.54, 0.55];
        let expected = arr.clone();
        sort(&mut arr);
        assert_eq!(arr, expected);
    }

    #[test]
    fn test_edge_values() {
        // Test with values very close to 0 and 1
        let mut arr = vec![0.001, 0.999, 0.002, 0.998];
        sort(&mut arr);
        assert_eq!(arr, vec![0.001, 0.002, 0.998, 0.999]);
    }

    #[test]
    fn test_bucket_index() {
        let num_buckets = 10;
        // Test bucket index calculation
        assert_eq!(get_bucket_index(0.1, num_buckets), 1);
        assert_eq!(get_bucket_index(0.5, num_buckets), 5);
        assert_eq!(get_bucket_index(0.99, num_buckets), 9);
        assert_eq!(get_bucket_index(0.0, num_buckets), 0);
    }

    #[test]
    fn test_insertion_sort() {
        let mut bucket = vec![0.5, 0.3, 0.4, 0.2, 0.1];
        insertion_sort(&mut bucket);
        assert_eq!(bucket, vec![0.1, 0.2, 0.3, 0.4, 0.5]);
    }

    #[test]
    fn test_sparse_distribution() {
        // Test with sparsely distributed values
        let mut arr = vec![0.1, 0.9, 0.2, 0.8, 0.3, 0.7];
        sort(&mut arr);
        assert_eq!(arr, vec![0.1, 0.2, 0.3, 0.7, 0.8, 0.9]);
    }

    #[test]
    fn test_almost_sorted() {
        // Test with almost sorted array
        let mut arr = vec![0.1, 0.2, 0.4, 0.3, 0.5];
        sort(&mut arr);
        assert_eq!(arr, vec![0.1, 0.2, 0.3, 0.4, 0.5]);
    }
}
