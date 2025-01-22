/// Counting Sort implementation for sorting slices of unsigned integers.
///
/// # Algorithm Overview
/// Counting sort works by:
/// 1. Finding the range of input data (min to max)
/// 2. Counting the frequency of each value in the input range
/// 3. Building the cumulative frequency array
/// 4. Placing each element in its sorted position
///
/// # Time Complexity
/// - Best Case: O(n + k) where k is the range of input
/// - Average Case: O(n + k)
/// - Worst Case: O(n + k)
///
/// # Space Complexity
/// - O(k) auxiliary space where k is the range of input
///
/// # Stability
/// - Stable sort algorithm
///
/// # Advantages
/// - Linear time complexity when k = O(n)
/// - Excellent for integers with known, limited range
/// - Stable sorting algorithm
/// - Can be used as a subroutine in radix sort
///
/// # Limitations
/// - Only works with non-negative integers
/// - Not efficient when the range of input values is much larger than n
/// - Requires extra space proportional to the range of input
/// - Falls back to standard library sort for ranges > 1_000_000
pub fn sort(slice: &mut [u32]) {
    if slice.len() <= 1 {
        return;
    }

    // Find the range of input array
    let max = find_max(slice);

    // If range is too large, fall back to standard library sort
    if max > 1_000_000 {
        slice.sort_unstable();
        return;
    }

    // Create a count array to store count of each unique value
    let mut count = vec![0; (max + 1) as usize];

    // Store count of each value
    for &value in slice.iter() {
        count[value as usize] += 1;
    }

    // Modify count array to store actual position of each value
    for i in 1..count.len() {
        count[i] += count[i - 1];
    }

    // Build the output array
    let mut output = vec![0; slice.len()];

    // Place elements in sorted order
    // Moving from end to start maintains stability
    for &value in slice.iter().rev() {
        count[value as usize] -= 1;
        output[count[value as usize]] = value;
    }

    // Copy back to original array
    slice.copy_from_slice(&output);
}

/// Finds the maximum value in the slice
fn find_max(slice: &[u32]) -> u32 {
    slice.iter().max().copied().unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_slice() {
        let mut arr: Vec<u32> = vec![];
        sort(&mut arr);
        assert_eq!(arr, Vec::<u32>::new());
    }

    #[test]
    fn test_single_element() {
        let mut arr = vec![1];
        sort(&mut arr);
        assert_eq!(arr, vec![1]);
    }

    #[test]
    fn test_sorted_array() {
        let mut arr = vec![1, 2, 3, 4, 5];
        sort(&mut arr);
        assert_eq!(arr, vec![1, 2, 3, 4, 5]);
    }

    #[test]
    fn test_reverse_sorted() {
        let mut arr = vec![5, 4, 3, 2, 1];
        sort(&mut arr);
        assert_eq!(arr, vec![1, 2, 3, 4, 5]);
    }

    #[test]
    fn test_random_order() {
        let mut arr = vec![3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5];
        let mut expected = arr.clone();
        expected.sort();
        sort(&mut arr);
        assert_eq!(arr, expected);
    }

    #[test]
    fn test_duplicate_elements() {
        let mut arr = vec![3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5];
        let mut expected = arr.clone();
        expected.sort();
        sort(&mut arr);
        assert_eq!(arr, expected);
    }

    #[test]
    fn test_large_array() {
        let mut arr: Vec<u32> = (0..1000).rev().collect();
        let mut expected = arr.clone();
        expected.sort();
        sort(&mut arr);
        assert_eq!(arr, expected);
    }

    #[test]
    fn test_all_equal() {
        // Test with an array where all elements are equal
        let mut arr = vec![1, 1, 1, 1, 1];
        sort(&mut arr);
        assert_eq!(arr, vec![1, 1, 1, 1, 1]);
    }

    #[test]
    fn test_sparse_array() {
        // Test with an array that has large gaps between values
        let mut arr = vec![2, 1000, 5, 20000, 3];
        sort(&mut arr);
        assert_eq!(arr, vec![2, 3, 5, 1000, 20000]);
    }

    #[test]
    fn test_stability() {
        // Test stability by sorting pairs and checking if relative order is preserved
        #[derive(Debug, Clone, PartialEq)]
        struct Pair {
            key: u32,
            original_index: usize,
        }

        let pairs = vec![
            Pair {
                key: 1,
                original_index: 0,
            },
            Pair {
                key: 1,
                original_index: 1,
            },
            Pair {
                key: 2,
                original_index: 2,
            },
            Pair {
                key: 2,
                original_index: 3,
            },
        ];

        let mut values: Vec<u32> = pairs.iter().map(|p| p.key).collect();
        sort(&mut values);

        // Create a mapping of sorted positions
        let mut position_map = vec![0; pairs.len()];
        let mut count = vec![0; 3]; // Count array for values 0-2

        // Count frequencies
        for &value in values.iter() {
            count[value as usize] += 1;
        }

        // Calculate cumulative frequencies
        for i in 1..count.len() {
            count[i] += count[i - 1];
        }

        // Build position map
        for pair in pairs.iter().rev() {
            count[pair.key as usize] -= 1;
            position_map[pair.original_index] = count[pair.key as usize];
        }

        // Verify that relative order is preserved for equal keys
        for i in 0..pairs.len() - 1 {
            for j in i + 1..pairs.len() {
                if pairs[i].key == pairs[j].key {
                    assert!(
                        position_map[pairs[i].original_index]
                            < position_map[pairs[j].original_index],
                        "Stability violated for equal elements at original positions {} and {}",
                        pairs[i].original_index,
                        pairs[j].original_index
                    );
                }
            }
        }
    }

    #[test]
    fn test_zero_and_max() {
        // Test with array containing zero and maximum u32 values
        let mut arr = vec![0, u32::MAX, 5, u32::MAX - 1, 0];
        sort(&mut arr);
        assert_eq!(arr, vec![0, 0, 5, u32::MAX - 1, u32::MAX]);
    }

    #[test]
    fn test_find_max() {
        assert_eq!(find_max(&[1, 5, 3, 9, 2]), 9);
        assert_eq!(find_max(&[1]), 1);
        assert_eq!(find_max(&[u32::MAX, 0, 5]), u32::MAX);
    }

    #[test]
    fn test_small_range() {
        // Test with small range of values (good case for counting sort)
        let mut arr = vec![2, 1, 0, 2, 1, 0, 1, 2];
        sort(&mut arr);
        assert_eq!(arr, vec![0, 0, 1, 1, 1, 2, 2, 2]);
    }

    #[test]
    fn test_large_range() {
        // Test with large range but few unique values
        let mut arr = vec![0, 1000000, 0, 1000000, 0];
        sort(&mut arr);
        assert_eq!(arr, vec![0, 0, 0, 1000000, 1000000]);
    }
}
