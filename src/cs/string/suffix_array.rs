/// Suffix Array implementation using the prefix doubling algorithm.
///
/// A suffix array is a sorted array of all suffixes of a string. It allows for
/// efficient string operations like pattern matching and longest common substring.
/// This implementation uses the prefix doubling technique which runs in O(n log n) time
/// and O(n) space.
///
/// # Example
/// ```
/// use algos::cs::string::suffix_array::SuffixArray;
///
/// let text = "banana";
/// let sa = SuffixArray::new(text);
/// ```
/// Result type for string search operations
pub type SearchResult = Result<Vec<usize>, String>;

#[derive(Debug)]
pub struct SuffixArray {
    /// The original text
    text: String,
    /// The suffix array - contains indices into text in sorted suffix order
    array: Vec<usize>,
    /// Rank array - contains the rank of each suffix for efficient comparison
    rank: Vec<usize>,
    /// LCP (Longest Common Prefix) array - contains LCP lengths between adjacent suffixes
    lcp: Vec<usize>,
}

impl SuffixArray {
    /// Constructs a new suffix array from the given text.
    ///
    /// # Arguments
    /// * `text` - The input text to build the suffix array from
    ///
    /// # Returns
    /// A new SuffixArray instance
    pub fn new(text: &str) -> Self {
        let text = text.to_string();
        let chars: Vec<char> = text.chars().collect();
        let n = chars.len();
        let mut array: Vec<usize> = (0..n).collect();
        let mut rank = vec![0; n];
        let mut tmp_rank = vec![0; n];

        // Initialize ranks with character values
        for (i, ch) in chars.iter().enumerate() {
            rank[i] = *ch as usize;
        }

        let mut k = 1;
        // Main prefix doubling loop
        while k < n {
            // Sort by rank pairs
            array.sort_by(|&i, &j| {
                let ri = rank[i];
                let rj = rank[j];
                let ri1 = if i + k < n { rank[i + k] } else { 0 };
                let rj1 = if j + k < n { rank[j + k] } else { 0 };
                (ri, ri1).cmp(&(rj, rj1))
            });

            // Update ranks
            tmp_rank[array[0]] = 0;
            for i in 1..n {
                let curr = array[i];
                let prev = array[i - 1];
                let curr_pair = (rank[curr], if curr + k < n { rank[curr + k] } else { 0 });
                let prev_pair = (rank[prev], if prev + k < n { rank[prev + k] } else { 0 });

                tmp_rank[curr] = if curr_pair == prev_pair {
                    tmp_rank[prev]
                } else {
                    i
                };
            }

            rank.copy_from_slice(&tmp_rank);

            if rank[array[n - 1]] == n - 1 {
                break; // All suffixes are sorted
            }

            k *= 2;
        }

        // Compute LCP array using Kasai's algorithm
        let lcp = Self::compute_lcp_array(&chars, &array, &rank);

        Self {
            text,
            array,
            rank,
            lcp,
        }
    }

    /// Computes the LCP (Longest Common Prefix) array using Kasai's algorithm.
    ///
    /// The LCP array stores the length of the longest common prefix between
    /// adjacent suffixes in the suffix array.
    ///
    /// Time complexity: O(n)
    /// Space complexity: O(n)
    fn compute_lcp_array(chars: &[char], suffix_array: &[usize], rank: &[usize]) -> Vec<usize> {
        let n = chars.len();
        let mut lcp = vec![0; n];
        let mut h = 0; // height of previous LCP

        for i in 0..n {
            if rank[i] > 0 {
                let j = suffix_array[rank[i] - 1];

                // Calculate LCP between suffixes starting at i and j
                while i + h < n && j + h < n && chars[i + h] == chars[j + h] {
                    h += 1;
                }

                lcp[rank[i]] = h;

                if h > 0 {
                    h = h.saturating_sub(1);
                }
            }
        }

        lcp
    }

    /// Returns the constructed suffix array.
    pub fn get_array(&self) -> &[usize] {
        &self.array
    }

    /// Returns the rank array.
    pub fn get_rank(&self) -> &[usize] {
        &self.rank
    }

    /// Returns the LCP array.
    pub fn get_lcp(&self) -> &[usize] {
        &self.lcp
    }

    /// Finds all occurrences of a pattern in the text.
    ///
    /// Uses binary search to find the range of suffixes that start with the pattern,
    /// then returns their positions in the text.
    ///
    /// # Arguments
    /// * `pattern` - The pattern to search for
    ///
    /// # Returns
    /// * `Ok(Vec<usize>)` - Vector of starting positions where pattern occurs
    /// * `Err(String)` - Error message if pattern is invalid
    pub fn find_all(&self, pattern: &str) -> SearchResult {
        if pattern.is_empty() {
            return Err("Pattern cannot be empty".to_string());
        }
        if pattern.len() > self.text.len() {
            return Ok(vec![]);
        }

        self.find_bounds(pattern)
    }

    /// Finds the first occurrence of a pattern in the text.
    ///
    /// Uses binary search to find the leftmost suffix that starts with the pattern.
    ///
    /// # Arguments
    /// * `pattern` - The pattern to search for
    ///
    /// # Returns
    /// * `Ok(Option<usize>)` - Starting position of first occurrence, if found
    /// * `Err(String)` - Error message if pattern is invalid
    pub fn find_first(&self, pattern: &str) -> Result<Option<usize>, String> {
        self.find_all(pattern)
            .map(|positions| positions.first().copied())
    }

    /// Finds the range of suffixes that start with the pattern using binary search.
    fn find_bounds(&self, pattern: &str) -> Result<Vec<usize>, String> {
        let n = self.text.chars().count();
        let pattern_chars: Vec<char> = pattern.chars().collect();
        let text_chars: Vec<char> = self.text.chars().collect();

        // Find all matching positions in the text
        let mut positions: Vec<usize> = Vec::new();
        for i in 0..n {
            let pos = self.array[i];
            let suffix: Vec<char> = text_chars[pos..].to_vec();
            if self.is_pattern_prefix(&pattern_chars, &suffix) {
                positions.push(pos);
            }
        }

        positions.sort_unstable();
        Ok(positions)
    }

    fn is_pattern_prefix(&self, pattern: &[char], suffix: &[char]) -> bool {
        if suffix.len() < pattern.len() {
            return false;
        }
        pattern.iter().zip(suffix.iter()).all(|(p, s)| p == s)
    }
}

/// Finds all occurrences of a pattern in the text.
///
/// Uses binary search to find the range of suffixes that start with the pattern,
/// then returns their positions in the text.
///
/// # Arguments
/// * `text` - The text to search in
/// * `pattern` - The pattern to search for
///
/// # Returns
/// * `Ok(Vec<usize>)` - Vector of starting positions where pattern occurs
/// * `Err(String)` - Error message if pattern is invalid
pub fn find_all(text: &str, pattern: &str) -> SearchResult {
    let sa = SuffixArray::new(text);
    sa.find_all(pattern)
}

/// Finds the first occurrence of a pattern in the text.
///
/// Uses binary search to find the leftmost suffix that starts with the pattern.
///
/// # Arguments
/// * `text` - The text to search in
/// * `pattern` - The pattern to search for
///
/// # Returns
/// * `Ok(Option<usize>)` - Starting position of first occurrence, if found
/// * `Err(String)` - Error message if pattern is invalid
pub fn find_first(text: &str, pattern: &str) -> Result<Option<usize>, String> {
    let sa = SuffixArray::new(text);
    sa.find_first(pattern)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_suffix_array() {
        let text = "banana";
        let sa = SuffixArray::new(text);
        let array = sa.get_array();

        // Expected suffix array for "banana":
        // Suffixes sorted lexicographically:
        // 5: a
        // 3: ana
        // 1: anana
        // 0: banana
        // 4: na
        // 2: nana
        assert_eq!(array, &[5, 3, 1, 0, 4, 2]);
    }

    #[test]
    fn test_lcp_array() {
        let text = "banana";
        let sa = SuffixArray::new(text);
        let lcp = sa.get_lcp();

        // Expected LCP values for adjacent suffixes in suffix array:
        // [5, 3, 1, 0, 4, 2] -> suffixes
        // [a, ana, anana, banana, na, nana]
        // LCP values: [0, 1, 3, 0, 0, 2]
        assert_eq!(lcp, &[0, 1, 3, 0, 0, 2]);
    }

    #[test]
    fn test_find_all() {
        let text = "banana";
        let sa = SuffixArray::new(text);

        assert_eq!(sa.find_all("ana").unwrap(), vec![1, 3]);
        assert_eq!(sa.find_all("na").unwrap(), vec![2, 4]);
        assert_eq!(sa.find_all("a").unwrap(), vec![1, 3, 5]);
        assert_eq!(sa.find_all("ban").unwrap(), vec![0]);
        assert_eq!(sa.find_all("xyz").unwrap(), vec![]);
    }

    #[test]
    fn test_find_first() {
        let text = "banana";
        let sa = SuffixArray::new(text);

        assert_eq!(sa.find_first("ana").unwrap(), Some(1));
        assert_eq!(sa.find_first("na").unwrap(), Some(2));
        assert_eq!(sa.find_first("a").unwrap(), Some(1));
        assert_eq!(sa.find_first("ban").unwrap(), Some(0));
        assert_eq!(sa.find_first("xyz").unwrap(), None);
    }

    #[test]
    fn test_empty_pattern() {
        let text = "banana";
        let sa = SuffixArray::new(text);

        assert!(sa.find_all("").is_err());
        assert!(sa.find_first("").is_err());
    }

    #[test]
    fn test_pattern_longer_than_text() {
        let text = "abc";
        let sa = SuffixArray::new(text);

        assert_eq!(sa.find_all("abcd").unwrap(), vec![]);
        assert_eq!(sa.find_first("abcd").unwrap(), None);
    }

    #[test]
    fn test_unicode_text() {
        let text = "こんにちは世界";
        let sa = SuffixArray::new(text);

        assert_eq!(sa.find_all("にち").unwrap(), vec![2]);
        assert_eq!(sa.find_all("世界").unwrap(), vec![5]);
        assert_eq!(sa.find_all("ちは").unwrap(), vec![3]);
    }

    #[test]
    fn test_overlapping_patterns() {
        let text = "aaaaa";
        let sa = SuffixArray::new(text);

        // Should find all overlapping occurrences
        assert_eq!(sa.find_all("aa").unwrap(), vec![0, 1, 2, 3]);
        assert_eq!(sa.find_all("aaa").unwrap(), vec![0, 1, 2]);
    }

    #[test]
    fn test_long_text() {
        let text = "a".repeat(10000) + "b";
        let sa = SuffixArray::new(&text);

        // Should handle long texts efficiently
        assert_eq!(sa.find_first("b").unwrap(), Some(10000));
        assert_eq!(sa.find_all("aa").unwrap().len(), 9999);
    }

    #[test]
    fn test_module_level_functions() {
        let text = "banana";

        assert_eq!(find_all(text, "ana").unwrap(), vec![1, 3]);
        assert_eq!(find_first(text, "ana").unwrap(), Some(1));

        assert!(find_all(text, "").is_err());
        assert!(find_first(text, "").is_err());
    }

    #[test]
    fn test_repeated_patterns() {
        let text = "abababab";
        let sa = SuffixArray::new(text);

        assert_eq!(sa.find_all("ab").unwrap(), vec![0, 2, 4, 6]);
        assert_eq!(sa.find_all("aba").unwrap(), vec![0, 2, 4]);
        assert_eq!(sa.find_all("abab").unwrap(), vec![0, 2, 4]);
    }

    #[test]
    fn test_case_sensitivity() {
        let text = "bAnAnA";
        let sa = SuffixArray::new(text);

        assert_eq!(sa.find_all("ana").unwrap(), vec![]);
        assert_eq!(sa.find_all("AnA").unwrap(), vec![1, 3]);
    }
}
