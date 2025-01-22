//! Manacher's algorithm for finding the longest palindromic substring in linear time.
//!
//! This implementation uses a common transform approach with special characters to handle
//! both odd and even-length palindromes uniformly.
//!
//! # Complexity
//! - Time: O(n), where n is the length of the input string
//! - Space: O(n) for the transformed string and auxiliary arrays
//!
//! # Example
//! ```rust
//! use algos::cs::string::manacher;
//!
//! let text = "babad";
//! let result = manacher::longest_palindrome(text).unwrap();
//! assert!(result == "bab" || result == "aba"); // both are valid results
//! ```

use crate::error::{Result, StringError};

/// Finds the longest palindromic substring in the given text using Manacher's algorithm.
///
/// This function implements Manacher's algorithm which efficiently finds the longest
/// palindromic substring in linear time. It handles both odd and even-length palindromes
/// by transforming the input string with special characters.
///
/// # Arguments
/// * `text` - The input text to search for palindromes
///
/// # Returns
/// * `Result<String>` - The longest palindromic substring found
///
/// # Examples
/// ```
/// use algos::cs::string::manacher;
///
/// let text = "babad";
/// let result = manacher::longest_palindrome(text).unwrap();
/// assert!(result == "bab" || result == "aba"); // both are valid
/// ```
pub fn longest_palindrome(text: &str) -> Result<String> {
    if text.is_empty() {
        return Ok(String::new());
    }

    // Transform the string to handle both odd and even length palindromes
    let transformed = preprocess(text);
    let chars: Vec<char> = transformed.chars().collect();
    let n = chars.len();

    let mut p = vec![0; n]; // palindrome radii
    let (mut center, mut right) = (0, 0);

    // Core Manacher's algorithm loop
    for i in 1..(n - 1) {
        let mirror = if i < center {
            // Safe subtraction: center is always >= i here
            2 * center - i
        } else {
            i
        };

        if i < right {
            // Safe subtraction: right is always >= i here
            p[i] = p[mirror].min(right - i);
        }

        // Attempt to expand around center i
        // Use checked arithmetic to prevent overflow
        while i + 1 + p[i] < n && i > p[i] {
            let right_pos = i + 1 + p[i];
            let left_pos = i.saturating_sub(1 + p[i]);
            if chars[right_pos] != chars[left_pos] {
                break;
            }
            p[i] += 1;
        }

        // Update center and right boundary if needed
        if i + p[i] > right {
            center = i;
            right = i + p[i];
        }
    }

    // Find the largest palindrome
    let (mut max_center, mut max_len) = (0, 0);
    for (i, &val) in p.iter().enumerate().skip(1).take(n - 2) {
        if val > max_len {
            max_center = i;
            max_len = val;
        }
    }

    // Convert indices back to original string
    // Use checked arithmetic to prevent overflow
    let start = max_center
        .checked_sub(1 + max_len)
        .map(|x| x / 2)
        .ok_or_else(|| StringError::invalid_input("Invalid palindrome position"))?;

    if start + max_len > text.len() {
        return Err(StringError::invalid_input("Invalid palindrome length"));
    }

    Ok(text[start..start + max_len].to_string())
}

/// Preprocesses the input string by adding special characters.
///
/// Transforms the input string by:
/// 1. Adding a boundary marker (^) at the start
/// 2. Inserting separators (#) between each character
/// 3. Adding a boundary marker ($) at the end
///
/// For example: "abc" becomes "^#a#b#c#$"
fn preprocess(s: &str) -> String {
    let mut result = String::with_capacity(s.len() * 2 + 3);
    result.push('^');
    for ch in s.chars() {
        result.push('#');
        result.push(ch);
    }
    result.push_str("#$");
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_string() {
        assert_eq!(longest_palindrome("").unwrap(), "");
    }

    #[test]
    fn test_single_char() {
        assert_eq!(longest_palindrome("a").unwrap(), "a");
    }

    #[test]
    fn test_odd_length_palindrome() {
        let result = longest_palindrome("babad").unwrap();
        assert!(result == "bab" || result == "aba");
    }

    #[test]
    fn test_even_length_palindrome() {
        assert_eq!(longest_palindrome("cbbd").unwrap(), "bb");
    }

    #[test]
    fn test_all_same_chars() {
        assert_eq!(longest_palindrome("aaaa").unwrap(), "aaaa");
    }

    #[test]
    fn test_no_palindrome() {
        let result = longest_palindrome("abcd").unwrap();
        assert_eq!(result.len(), 1);
    }

    #[test]
    fn test_complex_palindrome() {
        assert_eq!(
            longest_palindrome("forgeeksskeegfor").unwrap(),
            "geeksskeeg"
        );
    }
}
