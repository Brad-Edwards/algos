use crate::cs::error::{Error, Result};

/// Performs a sublist search to find a pattern within a larger list.
/// Returns the starting index of the first occurrence of the pattern.
///
/// # Arguments
/// * `data` - The main list to search in
/// * `pattern` - The sublist pattern to search for
///
/// # Returns
/// * `Ok(Some(index))` - The starting index of the first occurrence of the pattern
/// * `Ok(None)` - The pattern was not found
/// * `Err(Error)` - An error occurred during the search
///
/// # Examples
/// ```
/// # use algos::cs::search::sublist;
/// #
/// let data = vec![1, 2, 3, 4, 5, 6];
/// let pattern = vec![3, 4, 5];
/// assert!(matches!(sublist::search(&data, &pattern).unwrap(), Some(2)));
///
/// let not_found = vec![7, 8];
/// assert!(matches!(sublist::search(&data, &not_found).unwrap(), None));
/// ```
///
/// # Performance
/// * Time: O(m×n) where m and n are lengths of the lists
/// * Space: O(1)
///
/// # Type Requirements
/// * `T: PartialEq` - The type must support equality comparison
pub fn search<T: PartialEq>(data: &[T], pattern: &[T]) -> Result<Option<usize>> {
    if pattern.is_empty() {
        return Err(Error::invalid_input("Pattern cannot be empty"));
    }

    if data.is_empty() {
        return Ok(None);
    }

    if pattern.len() > data.len() {
        return Ok(None);
    }

    // Iterate through possible starting positions in data
    for i in 0..=data.len() - pattern.len() {
        let mut found = true;

        // Check if pattern matches at current position
        for j in 0..pattern.len() {
            if data[i + j] != pattern[j] {
                found = false;
                break;
            }
        }

        if found {
            return Ok(Some(i));
        }
    }

    Ok(None)
}

/// Performs a sublist search using the KMP (Knuth-Morris-Pratt) algorithm.
/// This is more efficient than the naive approach for patterns with repeating elements.
///
/// # Arguments
/// * `data` - The main list to search in
/// * `pattern` - The sublist pattern to search for
///
/// # Returns
/// * `Ok(Some(index))` - The starting index of the first occurrence of the pattern
/// * `Ok(None)` - The pattern was not found
/// * `Err(Error)` - An error occurred during the search
///
/// # Examples
/// ```
/// # use algos::cs::search::sublist;
/// #
/// let data = vec![1, 2, 1, 2, 1, 2, 3];
/// let pattern = vec![1, 2, 3];
/// assert!(matches!(sublist::search_kmp(&data, &pattern).unwrap(), Some(4)));
/// ```
///
/// # Performance
/// * Time: O(m + n) where m and n are lengths of the lists
/// * Space: O(m) for the pattern preprocessing
///
/// # Type Requirements
/// * `T: PartialEq` - The type must support equality comparison
pub fn search_kmp<T: PartialEq>(data: &[T], pattern: &[T]) -> Result<Option<usize>> {
    if pattern.is_empty() {
        return Err(Error::invalid_input("Pattern cannot be empty"));
    }

    if data.is_empty() {
        return Ok(None);
    }

    if pattern.len() > data.len() {
        return Ok(None);
    }

    // Compute longest prefix suffix values for pattern
    let lps = compute_lps(pattern);
    let mut j = 0; // index for pattern
    let mut i = 0; // index for data

    while i < data.len() {
        if pattern[j] == data[i] {
            j += 1;
            i += 1;
        }

        if j == pattern.len() {
            return Ok(Some(i - j));
        } else if i < data.len() && pattern[j] != data[i] {
            if j != 0 {
                j = lps[j - 1];
            } else {
                i += 1;
            }
        }
    }

    Ok(None)
}

/// Computes the Longest Proper Prefix which is also Suffix array.
/// Used by the KMP algorithm for efficient pattern matching.
fn compute_lps<T: PartialEq>(pattern: &[T]) -> Vec<usize> {
    let mut lps = vec![0; pattern.len()];
    let mut len = 0;
    let mut i = 1;

    while i < pattern.len() {
        if pattern[i] == pattern[len] {
            len += 1;
            lps[i] = len;
            i += 1;
        } else if len != 0 {
            len = lps[len - 1];
        } else {
            lps[i] = 0;
            i += 1;
        }
    }

    lps
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_data() {
        let data: Vec<i32> = vec![];
        let pattern = vec![1, 2];
        assert!(matches!(search(&data, &pattern).unwrap(), None));
        assert!(matches!(search_kmp(&data, &pattern).unwrap(), None));
    }

    #[test]
    fn test_empty_pattern() {
        let data = vec![1, 2, 3];
        let pattern: Vec<i32> = vec![];
        assert!(matches!(
            search(&data, &pattern),
            Err(Error::InvalidInput(_))
        ));
        assert!(matches!(
            search_kmp(&data, &pattern),
            Err(Error::InvalidInput(_))
        ));
    }

    #[test]
    fn test_pattern_longer_than_data() {
        let data = vec![1, 2];
        let pattern = vec![1, 2, 3];
        assert!(matches!(search(&data, &pattern).unwrap(), None));
        assert!(matches!(search_kmp(&data, &pattern).unwrap(), None));
    }

    #[test]
    fn test_simple_match() {
        let data = vec![1, 2, 3, 4, 5];
        let pattern = vec![2, 3];
        assert!(matches!(search(&data, &pattern).unwrap(), Some(1)));
        assert!(matches!(search_kmp(&data, &pattern).unwrap(), Some(1)));
    }

    #[test]
    fn test_match_at_start() {
        let data = vec![1, 2, 3, 4, 5];
        let pattern = vec![1, 2];
        assert!(matches!(search(&data, &pattern).unwrap(), Some(0)));
        assert!(matches!(search_kmp(&data, &pattern).unwrap(), Some(0)));
    }

    #[test]
    fn test_match_at_end() {
        let data = vec![1, 2, 3, 4, 5];
        let pattern = vec![4, 5];
        assert!(matches!(search(&data, &pattern).unwrap(), Some(3)));
        assert!(matches!(search_kmp(&data, &pattern).unwrap(), Some(3)));
    }

    #[test]
    fn test_no_match() {
        let data = vec![1, 2, 3, 4, 5];
        let pattern = vec![2, 4];
        assert!(matches!(search(&data, &pattern).unwrap(), None));
        assert!(matches!(search_kmp(&data, &pattern).unwrap(), None));
    }

    #[test]
    fn test_with_repeating_elements() {
        let data = vec![1, 2, 1, 2, 1, 2, 3];
        let pattern = vec![1, 2, 3];
        assert!(matches!(search(&data, &pattern).unwrap(), Some(4)));
        assert!(matches!(search_kmp(&data, &pattern).unwrap(), Some(4)));
    }

    #[test]
    fn test_with_strings() {
        let data = vec!["apple", "banana", "cherry", "date"];
        let pattern = vec!["banana", "cherry"];
        assert!(matches!(search(&data, &pattern).unwrap(), Some(1)));
        assert!(matches!(search_kmp(&data, &pattern).unwrap(), Some(1)));
    }

    #[test]
    fn test_overlapping_pattern() {
        let data = vec![1, 1, 1, 1];
        let pattern = vec![1, 1];
        assert!(matches!(search(&data, &pattern).unwrap(), Some(0)));
        assert!(matches!(search_kmp(&data, &pattern).unwrap(), Some(0)));
    }

    #[test]
    fn test_single_element_pattern() {
        let data = vec![1, 2, 3, 4, 5];
        let pattern = vec![3];
        assert!(matches!(search(&data, &pattern).unwrap(), Some(2)));
        assert!(matches!(search_kmp(&data, &pattern).unwrap(), Some(2)));
    }

    #[test]
    fn test_pattern_equals_data() {
        let data = vec![1, 2, 3];
        let pattern = vec![1, 2, 3];
        assert!(matches!(search(&data, &pattern).unwrap(), Some(0)));
        assert!(matches!(search_kmp(&data, &pattern).unwrap(), Some(0)));
    }

    #[test]
    fn test_complex_pattern() {
        let data = vec![1, 2, 3, 1, 2, 4, 1, 2, 3, 1, 2, 3];
        let pattern = vec![1, 2, 3];
        assert!(matches!(search(&data, &pattern).unwrap(), Some(0)));
        assert!(matches!(search_kmp(&data, &pattern).unwrap(), Some(0)));
    }
}
