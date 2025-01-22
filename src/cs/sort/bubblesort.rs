use std::fmt::Debug;

/// Bubblesort implementation for sorting slices.
///
/// # Algorithm Overview
/// Bubblesort is a simple sorting algorithm that:
/// 1. Repeatedly steps through the list
/// 2. Compares adjacent elements and swaps them if they are in the wrong order
/// 3. Continues until no swaps are needed
///
/// # Time Complexity
/// - Best Case: O(n) when array is already sorted
/// - Average Case: O(n²)
/// - Worst Case: O(n²)
///
/// # Space Complexity
/// - O(1) auxiliary space
///
/// # Stability
/// - Stable sort algorithm
pub fn sort<T>(slice: &mut [T])
where
    T: PartialOrd + Clone + Debug,
{
    if slice.len() <= 1 {
        return;
    }

    let len = slice.len();
    let mut swapped;

    // Optimization: Keep track of last swap position
    let mut new_len = len;

    loop {
        swapped = false;
        let mut last_swap = 0;

        for i in 0..new_len - 1 {
            if slice[i].partial_cmp(&slice[i + 1]).unwrap() == std::cmp::Ordering::Greater {
                slice.swap(i, i + 1);
                swapped = true;
                last_swap = i + 1;
            }
        }

        // If no swapping occurred, array is sorted
        if !swapped {
            break;
        }

        // Update new_len to last swap position
        new_len = last_swap;
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
    fn test_stability() {
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

        let mut items = vec![
            Item {
                key: 1,
                original_index: 0,
            },
            Item {
                key: 1,
                original_index: 1,
            },
            Item {
                key: 2,
                original_index: 2,
            },
            Item {
                key: 2,
                original_index: 3,
            },
        ];

        sort(&mut items);

        // Check if elements with equal keys maintain their relative order
        assert_eq!(items[0].original_index, 0);
        assert_eq!(items[1].original_index, 1);
        assert_eq!(items[2].original_index, 2);
        assert_eq!(items[3].original_index, 3);
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
    fn test_early_termination() {
        // Test that the algorithm terminates early when no swaps are needed
        let mut arr = vec![1, 2, 3, 4, 5];
        sort(&mut arr);
        assert_eq!(arr, vec![1, 2, 3, 4, 5]);
    }
}
