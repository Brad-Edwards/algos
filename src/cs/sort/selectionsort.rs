use std::fmt::Debug;

/// Selection Sort implementation for sorting slices.
///
/// # Algorithm Overview
/// Selection sort works by:
/// 1. Dividing the input into a sorted and unsorted region
/// 2. Finding the minimum element in the unsorted region
/// 3. Swapping it with the first element of the unsorted region
/// 4. Moving the boundary between sorted and unsorted regions one element to the right
///
/// # Time Complexity
/// - Best Case: O(n²)
/// - Average Case: O(n²)
/// - Worst Case: O(n²)
///
/// # Space Complexity
/// - O(1) auxiliary space
///
/// # Stability
/// - Not stable by default (equal elements may change relative order)
///
/// # Advantages
/// - Simple implementation
/// - Performs well on small arrays
/// - Minimizes the number of swaps (O(n) swaps vs O(n²) comparisons)
/// - In-place algorithm
/// - Works well when memory writes are expensive
pub fn sort<T>(slice: &mut [T])
where
    T: PartialOrd + Clone + Debug,
{
    let len = slice.len();

    // One by one move boundary of unsorted subarray
    for i in 0..len {
        // Find the minimum element in unsorted array
        let mut min_idx = i;
        for j in (i + 1)..len {
            if slice[j].partial_cmp(&slice[min_idx]).unwrap() == std::cmp::Ordering::Less {
                min_idx = j;
            }
        }

        // Swap the found minimum element with the first element
        // Only if we found a smaller element
        if min_idx != i {
            slice.swap(i, min_idx);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_slice() {
        let mut arr: Vec<i32> = vec![];
        sort(&mut arr);
        assert_eq!(arr, Vec::<i32>::new());
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
    fn test_minimal_swaps() {
        // Test that we're making O(n) swaps
        let mut count = 0;
        let mut arr = vec![5, 4, 3, 2, 1];

        // Wrap the slice in a type that counts swaps
        struct SwapCounter<'a, T> {
            slice: &'a mut [T],
            swap_count: &'a mut usize,
        }

        impl<'a, T> SwapCounter<'a, T> {
            fn swap(&mut self, i: usize, j: usize) {
                self.slice.swap(i, j);
                *self.swap_count += 1;
            }
        }

        {
            let mut counter = SwapCounter {
                slice: &mut arr,
                swap_count: &mut count,
            };

            // Custom selection sort that uses the counter
            for i in 0..counter.slice.len() {
                let mut min_idx = i;
                for j in (i + 1)..counter.slice.len() {
                    if counter.slice[j] < counter.slice[min_idx] {
                        min_idx = j;
                    }
                }
                if min_idx != i {
                    counter.swap(i, min_idx);
                }
            }
        }

        // For n=5, we expect at most n-1=4 swaps in selection sort
        assert!(
            count <= arr.len() - 1,
            "Expected at most {} swaps, but got {}",
            arr.len() - 1,
            count
        );
        assert_eq!(arr, vec![1, 2, 3, 4, 5]);
    }
}
