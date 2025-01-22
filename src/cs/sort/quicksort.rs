/// Sorts a mutable slice using the QuickSort algorithm.
///
/// # Algorithm Details
/// - Time Complexity: O(n log n) average case, O(n²) worst case
/// - Space Complexity: O(log n) average case for stack space
/// - Not stable: equal elements may be reordered
///
/// # Implementation Notes
/// - Uses median-of-three pivot selection to improve performance on partially sorted
///   arrays
/// - Switches to insertion sort for small subarrays (length < 10) to improve performance
/// - Employs tail-call optimization to prevent stack overflow
///
/// # Examples
/// ```
/// use algos::cs::sort::quicksort::sort;
/// let mut numbers = vec![3, 1, 4, 1, 5, 9, 2, 6, 5, 3];
/// sort(&mut numbers);
/// assert_eq!(numbers, vec![1, 1, 2, 3, 3, 4, 5, 5, 6, 9]);
/// ```
pub fn sort<T: Ord>(arr: &mut [T]) {
    // Threshold for switching to insertion sort
    const INSERTION_SORT_THRESHOLD: usize = 10;

    fn insertion_sort<T: Ord>(arr: &mut [T]) {
        for i in 1..arr.len() {
            let mut j = i;
            while j > 0 && arr[j - 1] > arr[j] {
                arr.swap(j - 1, j);
                j -= 1;
            }
        }
    }

    fn quicksort_internal<T: Ord>(arr: &mut [T]) {
        // Use iterative approach for tail-call optimization
        let mut stack = Vec::with_capacity(32); // log₂(usize::MAX) ≈ 32 for 64-bit systems
        stack.push((0, arr.len()));

        while let Some((start, end)) = stack.pop() {
            let len = end - start;

            if len <= 1 {
                continue;
            }

            if len < INSERTION_SORT_THRESHOLD {
                insertion_sort(&mut arr[start..end]);
                continue;
            }

            let pivot_idx = partition(&mut arr[start..end]) + start;

            // Push larger partition first to maintain O(log n) stack space
            if pivot_idx - start > end - (pivot_idx + 1) {
                stack.push((start, pivot_idx));
                stack.push((pivot_idx + 1, end));
            } else {
                stack.push((pivot_idx + 1, end));
                stack.push((start, pivot_idx));
            }
        }
    }

    quicksort_internal(arr);
}

fn partition<T: Ord>(arr: &mut [T]) -> usize {
    let len = arr.len();
    if len <= 1 {
        return 0;
    }

    // Median-of-three pivot selection
    let mid = len / 2;
    let last = len - 1;

    // Sort first, middle, and last elements
    if arr[0] > arr[mid] {
        arr.swap(0, mid);
    }
    if arr[mid] > arr[last] {
        arr.swap(mid, last);
    }
    if arr[0] > arr[mid] {
        arr.swap(0, mid);
    }

    // Move pivot to end
    arr.swap(mid, last - 1);
    let pivot_idx = last - 1;

    // Partition around pivot
    let mut i = 0;
    let mut j = pivot_idx;

    while i < j {
        while i < j && arr[i] <= arr[pivot_idx] {
            i += 1;
        }
        while i < j && arr[j - 1] > arr[pivot_idx] {
            j -= 1;
        }
        if i < j {
            arr.swap(i, j - 1);
        }
    }

    // Restore pivot
    if arr[i] > arr[pivot_idx] {
        arr.swap(i, pivot_idx);
        i
    } else {
        pivot_idx
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cmp::Ordering;

    // Basic functionality tests
    #[test]
    fn test_empty_array() {
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
    fn test_reverse_sorted_array() {
        let mut arr = vec![5, 4, 3, 2, 1];
        sort(&mut arr);
        assert_eq!(arr, vec![1, 2, 3, 4, 5]);
    }

    // Edge cases
    #[test]
    fn test_all_equal_elements() {
        let mut arr = vec![1, 1, 1, 1, 1];
        sort(&mut arr);
        assert_eq!(arr, vec![1, 1, 1, 1, 1]);
    }

    #[test]
    fn test_two_unique_elements_repeated() {
        let mut arr = vec![2, 1, 2, 1, 2, 1];
        sort(&mut arr);
        assert_eq!(arr, vec![1, 1, 1, 2, 2, 2]);
    }

    // Large array tests
    #[test]
    fn test_large_random_array() {
        let mut arr: Vec<i32> = (0..1000).map(|i| (i * 17 + 11) % 1000).collect();
        let mut expected = arr.clone();
        expected.sort();
        sort(&mut arr);
        assert_eq!(arr, expected);
    }

    #[test]
    fn test_large_mostly_sorted_array() {
        let mut arr: Vec<i32> = (0..1000).collect();
        // Introduce some disorder
        for i in 0..50 {
            arr.swap(i * 2, i * 2 + 1);
        }
        let mut expected = arr.clone();
        expected.sort();
        sort(&mut arr);
        assert_eq!(arr, expected);
    }

    // Custom type tests
    #[derive(Debug, Eq, PartialEq, Clone)]
    struct Person {
        name: String,
        age: u32,
    }

    impl PartialOrd for Person {
        fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
            Some(self.cmp(other))
        }
    }

    impl Ord for Person {
        fn cmp(&self, other: &Self) -> Ordering {
            self.age
                .cmp(&other.age)
                .then_with(|| self.name.cmp(&other.name))
        }
    }

    #[test]
    fn test_custom_type() {
        let mut people = vec![
            Person {
                name: "Alice".to_string(),
                age: 30,
            },
            Person {
                name: "Bob".to_string(),
                age: 25,
            },
            Person {
                name: "Charlie".to_string(),
                age: 35,
            },
            Person {
                name: "David".to_string(),
                age: 25,
            },
        ];

        sort(&mut people);

        assert_eq!(
            people,
            vec![
                Person {
                    name: "Bob".to_string(),
                    age: 25
                },
                Person {
                    name: "David".to_string(),
                    age: 25
                },
                Person {
                    name: "Alice".to_string(),
                    age: 30
                },
                Person {
                    name: "Charlie".to_string(),
                    age: 35
                },
            ]
        );
    }

    // Performance edge cases
    #[test]
    fn test_array_with_many_duplicates() {
        let mut arr = Vec::with_capacity(1000);
        for i in 0..1000 {
            arr.push(i % 4); // Only 4 unique values
        }
        let mut expected = arr.clone();
        expected.sort();
        sort(&mut arr);
        assert_eq!(arr, expected);
    }

    #[test]
    fn test_nearly_sorted_array() {
        let mut arr: Vec<i32> = (0..1000).collect();
        // Swap every 100th element
        for i in 0..10 {
            arr.swap(i * 100, i * 100 + 1);
        }
        let mut expected = arr.clone();
        expected.sort();
        sort(&mut arr);
        assert_eq!(arr, expected);
    }
}
