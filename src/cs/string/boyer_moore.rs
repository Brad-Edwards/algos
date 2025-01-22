use crate::cs::error::{Error, Result};
use std::collections::HashMap;

fn build_bad_char_table(pattern: &[u8]) -> HashMap<u8, usize> {
    let mut bad_char = HashMap::new();
    for (i, &c) in pattern.iter().enumerate() {
        bad_char.insert(c, i);
    }
    bad_char
}

pub fn find_all(text: impl AsRef<[u8]>, pattern: impl AsRef<[u8]>) -> Result<Vec<usize>> {
    let text = text.as_ref();
    let pattern = pattern.as_ref();

    if pattern.is_empty() {
        return Err(Error::empty_pattern());
    }
    if pattern.len() > text.len() {
        return Err(Error::pattern_too_long(pattern.len(), text.len()));
    }

    let m = pattern.len();
    let n = text.len();
    let mut matches = Vec::new();

    if n == 0 {
        return Ok(matches);
    }

    let bad_char = build_bad_char_table(pattern);

    let mut i = m - 1;
    while i < n {
        let mut j = m - 1;
        let mut k = i;
        let mut matched = true;

        // Match pattern from right to left
        while j > 0 && k > 0 {
            if pattern[j] != text[k] {
                matched = false;
                break;
            }
            j -= 1;
            k -= 1;
        }

        // Check the first character
        if matched && pattern[0] == text[k] {
            matches.push(k);
            // Move to next position after the start of current match
            i = k + m;
        } else {
            // Calculate shift using bad character rule
            let shift = match bad_char.get(&text[k]) {
                Some(&pos) => j.saturating_sub(pos),
                None => j + 1,
            };

            i += std::cmp::max(1, shift);
        }
    }

    Ok(matches)
}

pub fn find_first(text: impl AsRef<[u8]>, pattern: impl AsRef<[u8]>) -> Result<Option<usize>> {
    let text = text.as_ref();
    let pattern = pattern.as_ref();

    if pattern.is_empty() {
        return Err(Error::empty_pattern());
    }
    if pattern.len() > text.len() {
        return Err(Error::pattern_too_long(pattern.len(), text.len()));
    }

    let m = pattern.len();
    let n = text.len();

    if n == 0 {
        return Ok(None);
    }

    let bad_char = build_bad_char_table(pattern);

    let mut i = m - 1;
    while i < n {
        let mut j = m - 1;
        let mut k = i;
        let mut matched = true;

        while j > 0 && k > 0 {
            if pattern[j] != text[k] {
                matched = false;
                break;
            }
            j -= 1;
            k -= 1;
        }

        if matched && pattern[0] == text[k] {
            return Ok(Some(k));
        }

        let shift = match bad_char.get(&text[k]) {
            Some(&pos) => j.saturating_sub(pos),
            None => j + 1,
        };

        i += std::cmp::max(1, shift);
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
    fn test_bad_char_rule() {
        let pattern = "BAB";
        let bad_char = build_bad_char_table(pattern.as_bytes());
        assert_eq!(bad_char.get(&b'B'), Some(&2));
        assert_eq!(bad_char.get(&b'A'), Some(&1));
    }
}
