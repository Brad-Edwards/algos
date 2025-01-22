/// Radix Sort implementation for sorting slices of unsigned integers.
///
/// # Algorithm Overview
/// Radix sort is a non-comparative integer sorting algorithm that:
/// 1. Takes each place value (digit) starting from least significant
/// 2. Groups numbers by the value at that digit
/// 3. Collects numbers maintaining relative order within each group
/// 4. Repeats for each digit up to the most significant
///
/// # Time Complexity
/// - Best Case: O(d * (n + b)) where d is number of digits and b is the base
/// - Average Case: O(d * (n + b))
/// - Worst Case: O(d * (n + b))
///
/// # Space Complexity
/// - O(n + b) auxiliary space where b is the base (typically 10 or 256)
///
/// # Stability
/// - Stable sort algorithm
///
/// # Advantages
/// - Linear time complexity for fixed number of digits
/// - Stable sorting algorithm
/// - Works well when the range of possible digits is small
/// - Can be faster than comparison-based sorts
///
/// # Limitations
/// - Only works with integers or strings
/// - Performance depends on number of digits and base
/// - Uses extra space
pub fn sort(slice: &mut [u32]) {
    if slice.len() <= 1 {
        return;
    }

    // Find the maximum number to know number of digits
    let max = find_max(slice);

    // Do counting sort for every digit
    let mut exp = 1;
    while max / exp > 0 {
        counting_sort_by_digit(slice, exp);
        exp *= 10;
    }
}

/// Performs counting sort on a specific digit (0-9)
fn counting_sort_by_digit(slice: &mut [u32], exp: u32) {
    let len = slice.len();

    // Create output array and count array
    let mut output = vec![0; len];
    let mut count = [0; 10]; // 10 possible digits (0-9)

    // Store count of occurrences of current digit
    for &num in slice.iter() {
        count[get_digit(num, exp)] += 1;
    }

    // Change count[i] so that it contains actual
    // position of this digit in output[]
    for i in 1..10 {
        count[i] += count[i - 1];
    }

    // Build the output array
    // Moving from end to start maintains stability
    for &num in slice.iter().rev() {
        let digit = get_digit(num, exp);
        count[digit] -= 1;
        output[count[digit]] = num;
    }

    // Copy the output array to slice[]
    slice.copy_from_slice(&output);
}

/// Gets the digit at a specific place value (exp)
fn get_digit(num: u32, exp: u32) -> usize {
    ((num / exp) % 10) as usize
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
    fn test_stability() {
        // Test stability by sorting pairs and checking if relative order is preserved
        #[derive(Debug, Clone, PartialEq)]
        struct Pair {
            key: u32,
            original_index: usize,
        }

        let pairs = vec![
            Pair {
                key: 501,
                original_index: 0,
            },
            Pair {
                key: 501,
                original_index: 1,
            },
            Pair {
                key: 502,
                original_index: 2,
            },
            Pair {
                key: 502,
                original_index: 3,
            },
        ];

        let mut values: Vec<u32> = pairs.iter().map(|p| p.key).collect();
        sort(&mut values);

        // Verify that relative order is preserved for equal keys
        for i in 0..pairs.len() - 1 {
            for j in i + 1..pairs.len() {
                if pairs[i].key == pairs[j].key {
                    let pos_i = values.iter().position(|&x| x == pairs[i].key).unwrap();
                    let pos_j = values.iter().rposition(|&x| x == pairs[j].key).unwrap();
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
    fn test_zero_and_max() {
        // Test with array containing zero and large values
        let mut arr = vec![0, 1000000, 5, 999999, 0];
        sort(&mut arr);
        assert_eq!(arr, vec![0, 0, 5, 999999, 1000000]);
    }

    #[test]
    fn test_find_max() {
        assert_eq!(find_max(&[1, 5, 3, 9, 2]), 9);
        assert_eq!(find_max(&[1]), 1);
        assert_eq!(find_max(&[1000000, 0, 5]), 1000000);
    }

    #[test]
    fn test_get_digit() {
        // Test digit extraction at different positions
        assert_eq!(get_digit(123, 1), 3); // ones place
        assert_eq!(get_digit(123, 10), 2); // tens place
        assert_eq!(get_digit(123, 100), 1); // hundreds place
        assert_eq!(get_digit(123, 1000), 0); // thousands place
    }

    #[test]
    fn test_different_digit_lengths() {
        // Test numbers with different numbers of digits
        let mut arr = vec![1, 10, 100, 1000, 10000];
        sort(&mut arr);
        assert_eq!(arr, vec![1, 10, 100, 1000, 10000]);
    }

    #[test]
    fn test_power_of_ten() {
        // Test with powers of ten
        let mut arr = vec![1, 10, 100, 1000, 10000];
        sort(&mut arr);
        assert_eq!(arr, vec![1, 10, 100, 1000, 10000]);
    }

    #[test]
    fn test_repeated_digits() {
        // Test numbers with repeated digits
        let mut arr = vec![111, 222, 111, 222];
        sort(&mut arr);
        assert_eq!(arr, vec![111, 111, 222, 222]);
    }

    #[test]
    fn test_single_digit_numbers() {
        // Test with single digit numbers
        let mut arr = vec![9, 8, 7, 6, 5, 4, 3, 2, 1, 0];
        sort(&mut arr);
        assert_eq!(arr, vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
    }
}
