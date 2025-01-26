//! lib.rs

/// Returns the length of the longest common subsequence (LCS) between `a` and `b`.
///
/// # Examples
///
/// ```
/// use algos::cs::dynamic::longest_common_subsequence::lcs_length;
///
/// let s1 = "ABCDGH";
/// let s2 = "AEDFHR";
/// assert_eq!(lcs_length(s1, s2), 3); // "ADH" is one possible LCS
/// ```
pub fn lcs_length(a: &str, b: &str) -> usize {
    let a_chars: Vec<char> = a.chars().collect();
    let b_chars: Vec<char> = b.chars().collect();
    let m = a_chars.len();
    let n = b_chars.len();

    let mut dp = vec![vec![0; n + 1]; m + 1];

    for i in 1..=m {
        for j in 1..=n {
            if a_chars[i - 1] == b_chars[j - 1] {
                dp[i][j] = dp[i - 1][j - 1] + 1;
            } else {
                dp[i][j] = dp[i - 1][j].max(dp[i][j - 1]);
            }
        }
    }

    dp[m][n]
}

/// Reconstructs and returns one actual LCS (Longest Common Subsequence)
/// between `a` and `b`.
///
/// If there are multiple subsequences with the same length, this returns
/// just one of them. Returns an empty string if there's no common subsequence.
///
/// # Examples
///
/// ```
/// use algos::cs::dynamic::longest_common_subsequence::lcs_sequence;
///
/// let s1 = "ABCDGH";
/// let s2 = "AEDFHR";
/// let lcs = lcs_sequence(s1, s2);
/// assert_eq!(lcs.len(), 3); // "ADH" is one possible LCS
/// ```
pub fn lcs_sequence(a: &str, b: &str) -> String {
    let a_chars: Vec<char> = a.chars().collect();
    let b_chars: Vec<char> = b.chars().collect();
    let m = a_chars.len();
    let n = b_chars.len();

    // Build DP table of lengths
    let mut dp = vec![vec![0; n + 1]; m + 1];

    for i in 1..=m {
        for j in 1..=n {
            if a_chars[i - 1] == b_chars[j - 1] {
                dp[i][j] = dp[i - 1][j - 1] + 1;
            } else {
                dp[i][j] = dp[i - 1][j].max(dp[i][j - 1]);
            }
        }
    }

    // Reconstruct the sequence from dp.
    let mut i = m;
    let mut j = n;
    let mut subsequence = Vec::new();

    // Traverse back from dp[m][n]
    while i > 0 && j > 0 {
        if a_chars[i - 1] == b_chars[j - 1] {
            // This character is part of an LCS
            subsequence.push(a_chars[i - 1]);
            i -= 1;
            j -= 1;
        } else if dp[i - 1][j] > dp[i][j - 1] {
            i -= 1;
        } else {
            j -= 1;
        }
    }

    // The subsequence is constructed backwards, so reverse it.
    subsequence.reverse();
    subsequence.into_iter().collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lcs_length() {
        assert_eq!(lcs_length("", ""), 0);
        assert_eq!(lcs_length("ABC", ""), 0);
        assert_eq!(lcs_length("", "ABC"), 0);

        assert_eq!(lcs_length("ABCBDAB", "BDCABA"), 4);
        assert_eq!(lcs_length("XMJYAUZ", "MZJAWXU"), 4);
        assert_eq!(lcs_length("BANANA", "ATANA"), 4);
    }

    #[test]
    fn test_lcs_sequence() {
        assert_eq!(lcs_sequence("", ""), "");
        assert_eq!(lcs_sequence("ABC", ""), "");
        assert_eq!(lcs_sequence("", "ABC"), "");

        let seq1 = lcs_sequence("ABCBDAB", "BDCABA");
        assert_eq!(seq1.len(), 4);

        // XMJYAUZ / MZJAWXU => possible LCS "MJAU"
        let seq2 = lcs_sequence("XMJYAUZ", "MZJAWXU");
        assert_eq!(seq2.len(), 4);
        // Check that seq2 is indeed a subsequence of both
        assert!(is_subsequence(&seq2, "XMJYAUZ"));
        assert!(is_subsequence(&seq2, "MZJAWXU"));
    }

    /// A helper to verify that `subseq` is a subsequence of `s`.
    /// Not visible outside of tests.
    fn is_subsequence(subseq: &str, s: &str) -> bool {
        let mut it = s.chars();
        for c in subseq.chars() {
            match it.find(|&x| x == c) {
                Some(_) => continue,
                None => return false,
            }
        }
        true
    }
}
