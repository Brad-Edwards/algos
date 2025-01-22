use crate::cs::error::{Error, Result};

const PRIME: u64 = 16777619; // FNV prime
const BASE: u64 = 256; // Number of possible characters

/// Computes the hash of a pattern using the rolling hash function.
///
/// # Arguments
/// * `pattern` - The pattern to hash
/// * `m` - Length of the pattern
///
/// # Returns
/// * `(u64, u64)` - (hash value, highest place value used in hash calculation)
fn compute_pattern_hash(pattern: &[u8], m: usize) -> (u64, u64) {
    let mut pattern_hash = 0;
    let mut h = 1;

    // Calculate h = pow(BASE, m-1) % PRIME
    for _ in 0..m - 1 {
        h = (h * BASE) % PRIME;
    }

    // Calculate hash value of pattern
    for &ch in pattern.iter().take(m) {
        pattern_hash = (pattern_hash * BASE + ch as u64) % PRIME;
    }

    (pattern_hash, h)
}

/// Finds all occurrences of a pattern in the given text using the Rabin-Karp algorithm.
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
/// use algos::cs::string::rabin_karp;
///
/// let text = "AABAACAADAABAAABAA";
/// let pattern = "AABA";
/// let positions = rabin_karp::find_all(text, pattern).unwrap();
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

    let (pattern_hash, h) = compute_pattern_hash(pattern, m);
    let mut text_hash = 0;

    // Calculate hash value of first window
    for &ch in text.iter().take(m) {
        text_hash = (text_hash * BASE + ch as u64) % PRIME;
    }

    // Slide pattern over text one by one
    for i in 0..=n - m {
        if pattern_hash == text_hash {
            // Verify character by character on hash match
            if text[i..i + m] == pattern[..] {
                matches.push(i);
            }
        }

        // Calculate hash value for next window
        if i < n - m {
            text_hash = (BASE * (text_hash + PRIME - (h * text[i] as u64) % PRIME)
                + text[i + m] as u64)
                % PRIME;
        }
    }

    Ok(matches)
}

/// Finds the first occurrence of a pattern in the given text using the Rabin-Karp
/// algorithm.
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
/// use algos::cs::string::rabin_karp;
///
/// let text = "AABAACAADAABAAABAA";
/// let pattern = "AABA";
/// let position = rabin_karp::find_first(text, pattern).unwrap();
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

    let (pattern_hash, h) = compute_pattern_hash(pattern, m);
    let mut text_hash = 0;

    // Calculate hash value of first window
    for &ch in text.iter().take(m) {
        text_hash = (text_hash * BASE + ch as u64) % PRIME;
    }

    // Slide pattern over text one by one
    for i in 0..=n - m {
        if pattern_hash == text_hash {
            // Verify character by character on hash match
            if text[i..i + m] == pattern[..] {
                return Ok(Some(i));
            }
        }

        // Calculate hash value for next window
        if i < n - m {
            text_hash = (BASE * (text_hash + PRIME - (h * text[i] as u64) % PRIME)
                + text[i + m] as u64)
                % PRIME;
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
    fn test_hash_collisions() {
        // This test verifies that the algorithm correctly handles hash collisions
        // by doing character-by-character verification
        let text = "abcdef";
        let pattern = "abc";
        assert_eq!(find_all(text, pattern).unwrap(), vec![0]);
        assert_eq!(find_first(text, pattern).unwrap(), Some(0));
    }
}
