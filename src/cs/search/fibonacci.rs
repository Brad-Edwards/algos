use crate::cs::error::{Error, Result};

/// Performs a Fibonacci search on a sorted slice to find a target value.
/// Uses Fibonacci numbers to divide the search space, which can be more
/// efficient than binary search in some scenarios, particularly when
/// accessing memory/disk is expensive (fewer comparisons needed).
///
/// # Arguments
/// * `data` - A sorted slice of elements to search through
/// * `target` - The value to search for
///
/// # Returns
/// * `Ok(Some(index))` - The index where the target value was found
/// * `Ok(None)` - The target value was not found
/// * `Err(Error)` - An error occurred during the search (e.g., unsorted input)
///
/// # Examples
/// ```
/// # use algos::cs::search::fibonacci;
/// #
/// let numbers = vec![1, 2, 3, 4, 5, 6];
/// assert!(matches!(fibonacci::search(&numbers, &4).unwrap(), Some(3)));
/// assert!(matches!(fibonacci::search(&numbers, &7).unwrap(), None));
/// ```
///
/// # Performance
/// * Time: O(log n)
/// * Space: O(1)
/// * Fewer comparisons than binary search
///
/// # Type Requirements
/// * `T: Ord` - The type must support total ordering
pub fn search<T: Ord>(data: &[T], target: &T) -> Result<Option<usize>> {
    if data.is_empty() {
        return Ok(None);
    }

    // Verify the slice is sorted
    if !is_sorted(data) {
        return Err(Error::invalid_input(
            "Fibonacci search requires sorted input",
        ));
    }

    // Find the smallest Fibonacci number greater than or equal to len + 1
    let mut fib2 = 0; // (n-2)'th Fibonacci number
    let mut fib1 = 1; // (n-1)'th Fibonacci number
    let mut fib = fib1 + fib2; // n'th Fibonacci number

    while fib < data.len() {
        fib2 = fib1;
        fib1 = fib;
        fib = fib1 + fib2;
    }

    // Mark the eliminated range from front
    let mut offset = -1; // Marks the eliminated range from front

    // While there are elements to be inspected
    while fib > 1 {
        // Check if fib2 is a valid location
        let i = ((offset + fib2 as i32) as usize).min(data.len() - 1);

        match target.cmp(&data[i]) {
            std::cmp::Ordering::Less => {
                // The target lies in the first part
                fib = fib2;
                fib1 -= fib2;
                fib2 = fib - fib1;
            }
            std::cmp::Ordering::Greater => {
                // The target lies in the second part
                fib = fib1;
                fib1 = fib2;
                fib2 = fib - fib1;
                offset = i as i32;
            }
            std::cmp::Ordering::Equal => return Ok(Some(i)),
        }
    }

    // Compare last element
    if fib1 == 1 && (offset + 1) < data.len() as i32 {
        let i = (offset + 1) as usize;
        if target == &data[i] {
            return Ok(Some(i));
        }
    }

    Ok(None)
}

/// Checks if a slice is sorted in ascending order
fn is_sorted<T: Ord>(data: &[T]) -> bool {
    data.windows(2).all(|w| w[0] <= w[1])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_slice() {
        let data: Vec<i32> = vec![];
        assert!(matches!(search(&data, &5).unwrap(), None));
    }

    #[test]
    fn test_single_element_found() {
        let data = vec![5];
        assert!(matches!(search(&data, &5).unwrap(), Some(0)));
    }

    #[test]
    fn test_single_element_not_found() {
        let data = vec![5];
        assert!(matches!(search(&data, &3).unwrap(), None));
    }

    #[test]
    fn test_multiple_elements_found_first() {
        let data = vec![1, 2, 3, 4, 5];
        assert!(matches!(search(&data, &1).unwrap(), Some(0)));
    }

    #[test]
    fn test_multiple_elements_found_last() {
        let data = vec![1, 2, 3, 4, 5];
        assert!(matches!(search(&data, &5).unwrap(), Some(4)));
    }

    #[test]
    fn test_multiple_elements_found_middle() {
        let data = vec![1, 2, 3, 4, 5];
        assert!(matches!(search(&data, &3).unwrap(), Some(2)));
    }

    #[test]
    fn test_multiple_elements_not_found() {
        let data = vec![1, 2, 3, 4, 5];
        assert!(matches!(search(&data, &6).unwrap(), None));
    }

    #[test]
    fn test_with_duplicates() {
        let data = vec![1, 2, 2, 2, 3, 4];
        // Should find any occurrence of the duplicate value
        let result = search(&data, &2).unwrap().unwrap();
        assert!(result >= 1 && result <= 3);
    }

    #[test]
    fn test_unsorted_input() {
        let data = vec![3, 1, 4, 1, 5];
        assert!(matches!(search(&data, &4), Err(Error::InvalidInput(_))));
    }

    #[test]
    fn test_large_sorted_dataset() {
        let data: Vec<i32> = (0..10_000).collect();
        assert!(matches!(search(&data, &5000).unwrap(), Some(5000)));
        assert!(matches!(search(&data, &10_000).unwrap(), None));
    }

    #[test]
    fn test_with_strings() {
        let data = vec!["apple", "banana", "orange", "pear"];
        assert!(matches!(search(&data, &"orange").unwrap(), Some(2)));
        assert!(matches!(search(&data, &"grape").unwrap(), None));
    }

    #[test]
    fn test_boundary_values() {
        let data = vec![i32::MIN, -5, 0, 5, i32::MAX];
        assert!(matches!(search(&data, &i32::MIN).unwrap(), Some(0)));
        assert!(matches!(search(&data, &i32::MAX).unwrap(), Some(4)));
        assert!(matches!(search(&data, &0).unwrap(), Some(2)));
    }

    #[test]
    fn test_fibonacci_sequence_lengths() {
        // Test with array sizes that match Fibonacci numbers
        let data1: Vec<i32> = (0..3).collect(); // Fib: 3
        let data2: Vec<i32> = (0..5).collect(); // Fib: 5
        let data3: Vec<i32> = (0..8).collect(); // Fib: 8
        let data4: Vec<i32> = (0..13).collect(); // Fib: 13

        assert!(matches!(search(&data1, &1).unwrap(), Some(1)));
        assert!(matches!(search(&data2, &3).unwrap(), Some(3)));
        assert!(matches!(search(&data3, &5).unwrap(), Some(5)));
        assert!(matches!(search(&data4, &8).unwrap(), Some(8)));
    }

    #[test]
    fn test_non_fibonacci_lengths() {
        // Test with array sizes that don't match Fibonacci numbers
        let data1: Vec<i32> = (0..4).collect(); // Between Fib 3 and 5
        let data2: Vec<i32> = (0..7).collect(); // Between Fib 5 and 8
        let data3: Vec<i32> = (0..10).collect(); // Between Fib 8 and 13

        assert!(matches!(search(&data1, &2).unwrap(), Some(2)));
        assert!(matches!(search(&data2, &4).unwrap(), Some(4)));
        assert!(matches!(search(&data3, &6).unwrap(), Some(6)));
    }
}
