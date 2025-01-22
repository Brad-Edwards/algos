use crate::cs::error::{Error, Result};

/// Computes the longest proper prefix which is also a suffix (LPS) array
/// for the Knuth-Morris-Pratt algorithm.
///
/// The LPS array helps skip characters when a mismatch occurs during pattern matching.
///
/// # Arguments
/// * `pattern` - The pattern to preprocess
///
/// # Returns
/// * `Vec<usize>` - The LPS array where each index i contains the length of the longest
///   proper prefix of pattern[0..i] which is also a suffix of pattern[0..i]
fn compute_lps(pattern: &[u8]) -> Vec<usize> {
    let n = pattern.len();
    let mut lps = vec![0; n];
    let mut len = 0;
    let mut i = 1;

    while i < n {
        if pattern[i] == pattern[len] {
            len += 1;
            lps[i] = len;
            i += 1;
        } else if len > 0 {
            len = lps[len - 1];
        } else {
            lps[i] = 0;
            i += 1;
        }
    }

    lps
}

/// Finds all occurrences of a pattern in the given text using the KMP algorithm.
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
/// use algos::cs::string::kmp;
///
/// let text = "AABAACAADAABAAABAA";
/// let pattern = "AABA";
/// let positions = kmp::find_all(text, pattern).unwrap();
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

    let lps = compute_lps(pattern);
    let mut matches = Vec::new();
    let mut i = 0; // index for text
    let mut j = 0; // index for pattern

    while i < text.len() {
        if pattern[j] == text[i] {
            i += 1;
            j += 1;
        }

        if j == pattern.len() {
            matches.push(i - j);
            j = lps[j - 1];
        } else if i < text.len() && pattern[j] != text[i] {
            if j > 0 {
                j = lps[j - 1];
            } else {
                i += 1;
            }
        }
    }

    Ok(matches)
}

/// Finds the first occurrence of a pattern in the given text using the KMP algorithm.
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
/// use algos::cs::string::kmp;
///
/// let text = "AABAACAADAABAAABAA";
/// let pattern = "AABA";
/// let position = kmp::find_first(text, pattern).unwrap();
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

    let lps = compute_lps(pattern);
    let mut i = 0; // index for text
    let mut j = 0; // index for pattern

    while i < text.len() {
        if pattern[j] == text[i] {
            i += 1;
            j += 1;
        }

        if j == pattern.len() {
            return Ok(Some(i - j));
        } else if i < text.len() && pattern[j] != text[i] {
            if j > 0 {
                j = lps[j - 1];
            } else {
                i += 1;
            }
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
    fn test_compute_lps() {
        let pattern = "AABAACAABAA";
        let lps = compute_lps(pattern.as_bytes());
        assert_eq!(lps, vec![0, 1, 0, 1, 2, 0, 1, 2, 3, 4, 5]);
    }
}
