use crate::cs::error::{Error, Result};

/// Performs a binary search on a sorted slice to find a target value.
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
/// # use algos::cs::search::binary;
/// #
/// let numbers = vec![1, 2, 3, 4, 5, 6];
/// assert!(matches!(binary::search(&numbers, &4).unwrap(), Some(3)));
/// assert!(matches!(binary::search(&numbers, &7).unwrap(), None));
/// ```
///
/// # Performance
/// * Time: O(log n)
/// * Space: O(1)
///
/// # Type Requirements
/// * `T: Ord` - The type must support total ordering
pub fn search<T: Ord>(data: &[T], target: &T) -> Result<Option<usize>> {
    if data.is_empty() {
        return Ok(None);
    }

    // Verify the slice is sorted
    if !is_sorted(data) {
        return Err(Error::invalid_input("Binary search requires sorted input"));
    }

    let mut left = 0;
    let mut right = data.len();

    while left < right {
        let mid = left + (right - left) / 2;
        match data[mid].cmp(target) {
            std::cmp::Ordering::Equal => return Ok(Some(mid)),
            std::cmp::Ordering::Greater => right = mid,
            std::cmp::Ordering::Less => left = mid + 1,
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
}
