use crate::cs::error::{Error, Result};

/// Computes the Z-array for a given pattern.
/// The Z-array stores the length of the longest substring starting at each position
/// that matches a prefix of the string.
///
/// # Arguments
/// * `pattern` - The pattern to compute Z-array for
///
/// # Returns
/// * `Vec<usize>` - The Z-array where Z[i] is the length of the longest substring
///   starting at position i that matches a prefix of pattern
fn compute_z_array(pattern: &[u8]) -> Vec<usize> {
    let n = pattern.len();
    let mut z = vec![0; n];
    z[0] = n; // Z[0] is always n

    let mut left = 0;
    let mut right = 0;
    for i in 1..n {
        if i <= right {
            // We're within a Z-box, use previously computed values
            let k = i - left;
            let remaining = right - i + 1;
            if z[k] < remaining {
                z[i] = z[k];
                continue;
            }
            // Need to extend the match beyond the Z-box
            z[i] = remaining;
        }

        // Try to extend the match
        while i + z[i] < n && pattern[z[i]] == pattern[i + z[i]] {
            z[i] += 1;
        }

        // Update Z-box if we found a longer match
        if i + z[i] - 1 > right {
            left = i;
            right = i + z[i] - 1;
        }
    }

    z
}

/// Finds all occurrences of a pattern in the given text using the Z-algorithm.
///
/// # Arguments
/// * `text` - The text to search in
/// * `pattern` - The pattern to search for
///
/// # Returns
/// * `Result<Vec<usize>>` - A vector containing all starting positions where the pattern
///   occurs in the text
///
/// # Errors
/// * `Error::EmptyPattern` if the pattern is empty
/// * `Error::PatternTooLong` if pattern length exceeds text length
///
/// # Example
/// ```
/// use algos::cs::string::z_algorithm;
///
/// let text = "AABAACAADAABAAABAA";
/// let pattern = "AABA";
/// let positions = z_algorithm::find_all(text, pattern).unwrap();
/// assert_eq!(positions, vec![0, 9, 13]);
/// ```
pub fn find_all(text: impl AsRef<[u8]>, pattern: impl AsRef<[u8]>) -> Result<Vec<usize>> {
    let text = text.as_ref();
    let pattern = pattern.as_ref();

    // Validate inputs
    if pattern.is_empty() {
        return Err(Error::empty_pattern());
    }
    if pattern.len() > text.len() {
        return Err(Error::pattern_too_long(pattern.len(), text.len()));
    }

    let m = pattern.len();
    let n = text.len();
    let mut matches = Vec::new();

    // Edge case: empty text
    if n == 0 {
        return Ok(matches);
    }

    // Concatenate pattern and text with a sentinel character
    let mut combined = Vec::with_capacity(m + 1 + n);
    combined.extend_from_slice(pattern);
    combined.push(0); // Sentinel character
    combined.extend_from_slice(text);

    // Compute Z-array for the concatenated string
    let z = compute_z_array(&combined);

    // Find matches by looking for Z-values equal to pattern length
    for (i, &item) in z
        .iter()
        .enumerate()
        .skip(m + 1)
        .take(combined.len() - m - 1)
    {
        if item == m {
            matches.push(i - (m + 1));
        }
    }

    Ok(matches)
}

/// Finds the first occurrence of a pattern in the given text using the Z-algorithm.
///
/// # Arguments
/// * `text` - The text to search in
/// * `pattern` - The pattern to search for
///
/// # Returns
/// * `Result<Option<usize>>` - The starting position of the first occurrence if found,
///   None otherwise
///
/// # Errors
/// * `Error::EmptyPattern` if the pattern is empty
/// * `Error::PatternTooLong` if pattern length exceeds text length
///
/// # Example
/// ```
/// use algos::cs::string::z_algorithm;
///
/// let text = "AABAACAADAABAAABAA";
/// let pattern = "AABA";
/// let position = z_algorithm::find_first(text, pattern).unwrap();
/// assert_eq!(position, Some(0));
/// ```
pub fn find_first(text: impl AsRef<[u8]>, pattern: impl AsRef<[u8]>) -> Result<Option<usize>> {
    let text = text.as_ref();
    let pattern = pattern.as_ref();

    // Validate inputs
    if pattern.is_empty() {
        return Err(Error::empty_pattern());
    }
    if pattern.len() > text.len() {
        return Err(Error::pattern_too_long(pattern.len(), text.len()));
    }

    let m = pattern.len();
    let n = text.len();

    // Edge case: empty text
    if n == 0 {
        return Ok(None);
    }

    // Concatenate pattern and text with a sentinel character
    let mut combined = Vec::with_capacity(m + 1 + n);
    combined.extend_from_slice(pattern);
    combined.push(0); // Sentinel character
    combined.extend_from_slice(text);

    // Compute Z-array for the concatenated string
    let z = compute_z_array(&combined);

    // Find first match by looking for Z-value equal to pattern length
    for (i, &item) in z
        .iter()
        .enumerate()
        .skip(m + 1)
        .take(combined.len() - m - 1)
    {
        if item == m {
            return Ok(Some(i - (m + 1)));
        }
    }

    Ok(None)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_pattern() {
        let text = "hello";
        let pattern = "";
        assert!(matches!(find_all(text, pattern), Err(Error::EmptyPattern)));
    }

    #[test]
    fn test_pattern_too_long() {
        let text = "hi";
        let pattern = "hello";
        assert!(matches!(
            find_all(text, pattern),
            Err(Error::PatternTooLong { .. })
        ));
    }

    #[test]
    fn test_pattern_not_found() {
        let text = "hello world";
        let pattern = "xyz";
        assert_eq!(find_all(text, pattern).unwrap(), Vec::<usize>::new());
        assert_eq!(find_first(text, pattern).unwrap(), None);
    }

    #[test]
    fn test_single_match() {
        let text = "hello world";
        let pattern = "world";
        assert_eq!(find_all(text, pattern).unwrap(), vec![6]);
        assert_eq!(find_first(text, pattern).unwrap(), Some(6));
    }

    #[test]
    fn test_multiple_matches() {
        let text = "AABAACAADAABAAABAA";
        let pattern = "AABA";
        assert_eq!(find_all(text, pattern).unwrap(), vec![0, 9, 13]);
        assert_eq!(find_first(text, pattern).unwrap(), Some(0));
    }

    #[test]
    fn test_overlapping_matches() {
        let text = "AAAAA";
        let pattern = "AA";
        assert_eq!(find_all(text, pattern).unwrap(), vec![0, 1, 2, 3]);
        assert_eq!(find_first(text, pattern).unwrap(), Some(0));
    }

    #[test]
    fn test_match_at_start() {
        let text = "hello world";
        let pattern = "hello";
        assert_eq!(find_all(text, pattern).unwrap(), vec![0]);
        assert_eq!(find_first(text, pattern).unwrap(), Some(0));
    }

    #[test]
    fn test_match_at_end() {
        let text = "hello world";
        let pattern = "world";
        assert_eq!(find_all(text, pattern).unwrap(), vec![6]);
        assert_eq!(find_first(text, pattern).unwrap(), Some(6));
    }

    #[test]
    fn test_unicode_text() {
        let text = "Hello 世界!";
        let pattern = "世界";
        assert_eq!(
            find_all(text.as_bytes(), pattern.as_bytes()).unwrap(),
            vec![6]
        );
        assert_eq!(
            find_first(text.as_bytes(), pattern.as_bytes()).unwrap(),
            Some(6)
        );
    }

    #[test]
    fn test_empty_text() {
        let text = "";
        let pattern = "a";
        assert!(matches!(
            find_all(text, pattern),
            Err(Error::PatternTooLong { .. })
        ));
    }

    #[test]
    fn test_z_array() {
        let pattern = "aabaacd";
        let z = compute_z_array(pattern.as_bytes());
        assert_eq!(z, vec![7, 1, 0, 2, 1, 0, 0]);

        let pattern = "aaaaa";
        let z = compute_z_array(pattern.as_bytes());
        assert_eq!(z, vec![5, 4, 3, 2, 1]);
    }

    #[test]
    fn test_z_box_optimization() {
        // This test verifies that the Z-box optimization correctly reuses
        // previously computed values
        let text = "abababab";
        let pattern = "abab";
        assert_eq!(find_all(text, pattern).unwrap(), vec![0, 2, 4]);
    }
}
