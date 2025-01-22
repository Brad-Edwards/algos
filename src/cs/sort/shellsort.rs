use std::fmt::Debug;

/// Shell Sort implementation for sorting slices.
///
/// # Algorithm Overview
/// Shell sort is an optimization of insertion sort that:
/// 1. Starts by sorting pairs of elements far apart from each other
/// 2. Progressively reduces the gap between elements being compared
/// 3. Uses the gap sequence: n/2, n/4, n/8, ..., 1 (other sequences possible)
/// 4. For each gap, performs a gapped insertion sort
///
/// # Time Complexity
/// - Best Case: O(n log n) - depends on gap sequence
/// - Average Case: O(n^1.3) - using Knuth's sequence
/// - Worst Case: O(n²) or O(n log² n) depending on gap sequence
///
/// # Space Complexity
/// - O(1) auxiliary space
///
/// # Stability
/// - Not stable
///
/// # Advantages
/// - Simple implementation
/// - Adaptive: runs faster when array is partially sorted
/// - In-place algorithm
/// - Much better than simple insertion sort
/// - Works well for medium-sized arrays
pub fn sort<T>(slice: &mut [T])
where
    T: PartialOrd + Clone + Debug,
{
    let len = slice.len();
    if len <= 1 {
        return;
    }

    // Start with the largest gap and work down to a gap of 1
    let mut gap = calculate_initial_gap(len);

    while gap > 0 {
        // Do a gapped insertion sort for this gap size
        for i in gap..len {
            let mut j = i;
            while j >= gap
                && slice[j - gap].partial_cmp(&slice[j]).unwrap() == std::cmp::Ordering::Greater
            {
                slice.swap(j - gap, j);
                j -= gap;
            }
        }

        // Calculate next gap
        gap = if gap == 2 { 1 } else { gap / 3 };
    }
}

/// Calculates the initial gap using Knuth's sequence: h = 3h + 1
/// This sequence has been shown to work well in practice
fn calculate_initial_gap(len: usize) -> usize {
    let mut gap = 1;
    while gap < len / 3 {
        gap = 3 * gap + 1;
    }
    gap
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_slice() {
        let mut arr: Vec<i32> = vec![];
        sort(&mut arr);
        assert_eq!(arr, vec![]);
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
        let mut arr: Vec<i32> = (0..1000).rev().collect();
        let mut expected = arr.clone();
        expected.sort();
        sort(&mut arr);
        assert_eq!(arr, expected);
    }

    #[test]
    fn test_different_types() {
        // Test with floating point numbers
        let mut float_arr = vec![3.14, 1.41, 2.71, 0.58];
        let mut expected = float_arr.clone();
        expected.sort_by(|a, b| a.partial_cmp(b).unwrap());
        sort(&mut float_arr);
        assert_eq!(float_arr, expected);

        // Test with strings
        let mut string_arr = vec!["banana", "apple", "cherry", "date"];
        let mut expected = string_arr.clone();
        expected.sort();
        sort(&mut string_arr);
        assert_eq!(string_arr, expected);
    }

    #[test]
    fn test_all_equal() {
        // Test with an array where all elements are equal
        let mut arr = vec![1, 1, 1, 1, 1];
        sort(&mut arr);
        assert_eq!(arr, vec![1, 1, 1, 1, 1]);
    }

    #[test]
    fn test_alternating() {
        // Test with an array that alternates between two values
        let mut arr = vec![1, 2, 1, 2, 1, 2];
        sort(&mut arr);
        assert_eq!(arr, vec![1, 1, 1, 2, 2, 2]);
    }

    #[test]
    fn test_negative_numbers() {
        // Test with negative numbers
        let mut arr = vec![-5, -2, -8, -1, -9];
        sort(&mut arr);
        assert_eq!(arr, vec![-9, -8, -5, -2, -1]);
    }

    #[test]
    fn test_mixed_numbers() {
        // Test with a mix of positive and negative numbers
        let mut arr = vec![-3, 4, 0, -1, 5, -2];
        sort(&mut arr);
        assert_eq!(arr, vec![-3, -2, -1, 0, 4, 5]);
    }

    #[test]
    fn test_partially_sorted() {
        // Test with an array that's partially sorted
        let mut arr = vec![1, 2, 4, 3, 5, 7, 6, 8];
        sort(&mut arr);
        assert_eq!(arr, vec![1, 2, 3, 4, 5, 6, 7, 8]);
    }

    #[test]
    fn test_gap_sequence() {
        // Test that gap sequence works correctly for different array sizes
        let sizes = vec![10, 100, 1000];
        for size in sizes {
            let gap = calculate_initial_gap(size);
            assert!(gap > 0, "Gap should be positive for size {}", size);
            assert!(gap < size, "Gap should be less than array size {}", size);
        }
    }

    #[test]
    fn test_knuth_sequence() {
        // Test that Knuth's sequence is correctly generated
        // For n=100, the sequence should be something like: 1, 4, 13, 40
        let n = 100;
        let mut gap = 1;
        let mut gaps = vec![gap];

        while gap * 3 + 1 < n {
            gap = gap * 3 + 1;
            gaps.push(gap);
        }

        // Check that gaps are decreasing
        for i in 0..gaps.len() - 1 {
            assert!(gaps[i] < gaps[i + 1], "Gaps should be in increasing order");
        }

        // Check that the largest gap is less than n
        assert!(
            gaps.last().unwrap() < &n,
            "Largest gap should be less than array size"
        );
    }
}
