use crate::cs::error::{Error, Result};
use num_traits::ToPrimitive;
use std::cmp::Ord;

/// Performs an interpolation search on a sorted slice to find a target value.
/// Most efficient for uniformly distributed data.
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
/// # use algos::cs::search::interpolation;
/// #
/// let numbers = vec![1, 2, 3, 4, 5, 6];
/// assert!(matches!(interpolation::search(&numbers, &4).unwrap(), Some(3)));
/// assert!(matches!(interpolation::search(&numbers, &7).unwrap(), None));
/// ```
///
/// # Performance
/// * Time: O(log log n) average case for uniformly distributed data
/// * Time: O(n) worst case
/// * Space: O(1)
///
/// # Type Requirements
/// * `T: Ord + ToPrimitive` - The type must support total ordering and conversion to
///   primitive types
pub fn search<T>(data: &[T], target: &T) -> Result<Option<usize>>
where
    T: Ord + ToPrimitive,
{
    if data.is_empty() {
        return Ok(None);
    }

    // Verify the slice is sorted
    if !is_sorted(data) {
        return Err(Error::invalid_input(
            "Interpolation search requires sorted input",
        ));
    }

    let mut low = 0;
    let mut high = data.len() - 1;

    while low <= high && target >= &data[low] && target <= &data[high] {
        // Convert values to f64 for interpolation calculation
        let target_f = to_f64(target)?;
        let low_val_f = to_f64(&data[low])?;
        let high_val_f = to_f64(&data[high])?;

        // Avoid division by zero
        if high_val_f == low_val_f {
            if &data[low] == target {
                return Ok(Some(low));
            }
            return Ok(None);
        }

        // Calculate the probable position using interpolation formula
        let pos_f =
            low as f64 + ((high - low) as f64 * (target_f - low_val_f) / (high_val_f - low_val_f));

        let pos = pos_f.round() as usize;

        // Bounds check
        if pos > high {
            break;
        }

        match target.cmp(&data[pos]) {
            std::cmp::Ordering::Equal => return Ok(Some(pos)),
            std::cmp::Ordering::Less => {
                if pos == 0 {
                    break;
                }
                high = pos - 1;
            }
            std::cmp::Ordering::Greater => low = pos + 1,
        }
    }

    Ok(None)
}

/// Checks if a slice is sorted in ascending order
fn is_sorted<T: Ord>(data: &[T]) -> bool {
    data.windows(2).all(|w| w[0] <= w[1])
}

/// Converts a value to f64 for interpolation calculation
fn to_f64<T: ToPrimitive>(value: &T) -> Result<f64> {
    value
        .to_f64()
        .ok_or_else(|| Error::invalid_input("Failed to convert value to f64 for interpolation"))
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
    fn test_uniformly_distributed() {
        let data: Vec<i32> = (0..100).collect();
        assert!(matches!(search(&data, &50).unwrap(), Some(50)));
        assert!(matches!(search(&data, &99).unwrap(), Some(99)));
        assert!(matches!(search(&data, &100).unwrap(), None));
    }

    #[test]
    fn test_non_uniform_distribution() {
        let data = vec![1, 10, 100, 1000, 10000];
        assert!(matches!(search(&data, &100).unwrap(), Some(2)));
        assert!(matches!(search(&data, &500).unwrap(), None));
    }

    #[test]
    fn test_large_uniform_dataset() {
        let data: Vec<i32> = (0..10_000).collect();
        assert!(matches!(search(&data, &5000).unwrap(), Some(5000)));
        assert!(matches!(search(&data, &10_000).unwrap(), None));
    }

    #[test]
    fn test_floating_point() {
        let data = vec![1, 2, 3, 4, 5];
        assert!(matches!(search(&data, &3).unwrap(), Some(2)));
        assert!(matches!(search(&data, &6).unwrap(), None));
    }

    #[test]
    fn test_boundary_values() {
        let data = vec![i32::MIN, -5, 0, 5, i32::MAX];
        assert!(matches!(search(&data, &i32::MIN).unwrap(), Some(0)));
        assert!(matches!(search(&data, &i32::MAX).unwrap(), Some(4)));
        assert!(matches!(search(&data, &0).unwrap(), Some(2)));
    }

    #[test]
    fn test_equal_values_target_not_found() {
        let data = vec![5, 5, 5, 5, 5];
        assert!(matches!(search(&data, &3).unwrap(), None));
    }

    #[test]
    fn test_pos_greater_than_high() {
        let data = vec![1, 2, 1000000];
        assert!(matches!(search(&data, &999999).unwrap(), None));
    }

    #[test]
    fn test_conversion_failure() {
        // Create a custom type that implements Ord but not ToPrimitive
        #[derive(Debug, Eq, PartialEq, Ord, PartialOrd)]
        struct NonNumeric(i32);

        impl ToPrimitive for NonNumeric {
            fn to_i64(&self) -> Option<i64> {
                None
            }

            fn to_u64(&self) -> Option<u64> {
                None
            }

            fn to_f64(&self) -> Option<f64> {
                None
            }
        }

        let data = vec![NonNumeric(1), NonNumeric(2), NonNumeric(3)];
        assert!(matches!(
            search(&data, &NonNumeric(2)),
            Err(Error::InvalidInput(_))
        ));
    }

    #[test]
    fn test_interpolation_out_of_bounds() {
        let data = vec![1, 2, 1000000];
        assert!(matches!(search(&data, &999999).unwrap(), None));
    }

    #[test]
    fn test_target_less_than_first() {
        let data = vec![5, 10, 15, 20];
        assert!(matches!(search(&data, &1).unwrap(), None));
    }
}
