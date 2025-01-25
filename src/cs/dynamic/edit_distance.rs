/// lib.rs

/// Computes the Levenshtein (edit) distance between two string slices.
///
/// The Levenshtein distance is defined as the minimum number of single-character
/// edits (insertions, deletions, substitutions) required to change `a` into `b`.
///
/// # Examples
///
/// ```
/// use editdistance::levenshtein_distance;
///
/// assert_eq!(levenshtein_distance("", ""), 0);
/// assert_eq!(levenshtein_distance("kitten", "sitting"), 3);
/// assert_eq!(levenshtein_distance("gumbo", "gambol"), 2);
/// ```
pub fn levenshtein_distance(a: &str, b: &str) -> usize {
    // If either string is empty, distance is the length of the other.
    if a.is_empty() {
        return b.chars().count();
    } else if b.is_empty() {
        return a.chars().count();
    }

    compute_distance(a, b)
}

/// Internal helper implementing a standard dynamic programming approach.
/// This uses a single rolling array (two rows) for memory efficiency.
fn compute_distance(a: &str, b: &str) -> usize {
    let b_len = b.chars().count();
    let mut prev_row = (0..=b_len).collect::<Vec<usize>>();
    let mut curr_row = vec![0; b_len + 1];

    for (i, ca) in a.chars().enumerate() {
        curr_row[0] = i + 1;

        // We track position in `b` with j and also extract each char `cb`.
        for (j, cb) in b.chars().enumerate() {
            let cost = if ca == cb { 0 } else { 1 };

            // The recurrence relation:
            //   curr_row[j+1] = minimum of:
            //     1) prev_row[j+1] + 1   (deletion)
            //     2) curr_row[j] + 1     (insertion)
            //     3) prev_row[j] + cost  (substitution)
            curr_row[j + 1] = (prev_row[j + 1] + 1)
                .min(curr_row[j] + 1)
                .min(prev_row[j] + cost);
        }

        prev_row.copy_from_slice(&curr_row);
    }

    prev_row[b_len]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty() {
        assert_eq!(levenshtein_distance("", ""), 0);
        assert_eq!(levenshtein_distance("", "abc"), 3);
        assert_eq!(levenshtein_distance("abc", ""), 3);
    }

    #[test]
    fn test_basic_cases() {
        assert_eq!(levenshtein_distance("kitten", "sitting"), 3);
        assert_eq!(levenshtein_distance("sunday", "saturday"), 3);
        assert_eq!(levenshtein_distance("gumbo", "gambol"), 2);
        assert_eq!(levenshtein_distance("abc", "abc"), 0);
        assert_eq!(levenshtein_distance("flaw", "lawn"), 2);
    }

    #[test]
    fn test_unicode() {
        // Simple ASCII changes
        assert_eq!(levenshtein_distance("cafe", "coffee"), 3);
        // Slightly more complex with accented characters
        assert_eq!(levenshtein_distance("café", "cafe"), 1);
    }
}
