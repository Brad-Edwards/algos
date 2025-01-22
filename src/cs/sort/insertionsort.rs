use std::fmt::Debug;

/// Insertion Sort implementation for sorting slices.
///
/// # Algorithm Overview
/// Insertion sort builds the final sorted array one item at a time by:
/// 1. Starting with the first element as a sorted array of length 1
/// 2. Taking each subsequent element and inserting it into its correct position in the
///    sorted portion
/// 3. Shifting elements as needed to make space for the inserted element
///
/// # Time Complexity
/// - Best Case: O(n) when array is already sorted
/// - Average Case: O(n²)
/// - Worst Case: O(n²) when array is reverse sorted
///
/// # Space Complexity
/// - O(1) auxiliary space
///
/// # Stability
/// - Stable sort algorithm
///
/// # Advantages
/// - Simple implementation
/// - Efficient for small data sets
/// - Adaptive: O(n) when data is nearly sorted
/// - In-place: O(1) extra space
/// - Online: can sort a list as it receives it
pub fn sort<T>(slice: &mut [T])
where
    T: PartialOrd + Clone + Debug,
{
    for i in 1..slice.len() {
        let mut j = i;
        while j > 0 && slice[j - 1].partial_cmp(&slice[j]).unwrap() == std::cmp::Ordering::Greater {
            slice.swap(j - 1, j);
            j -= 1;
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
    fn test_nearly_sorted() {
        // Test with an array that's mostly sorted but has a few elements out of place
        let mut arr = vec![1, 2, 4, 3, 5, 6, 8, 7];
        sort(&mut arr);
        assert_eq!(arr, vec![1, 2, 3, 4, 5, 6, 7, 8]);
    }

    #[test]
    fn test_all_equal() {
        // Test with an array where all elements are equal
        let mut arr = vec![1, 1, 1, 1, 1];
        sort(&mut arr);
        assert_eq!(arr, vec![1, 1, 1, 1, 1]);
    }
}
