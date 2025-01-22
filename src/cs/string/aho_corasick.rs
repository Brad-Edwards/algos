use crate::cs::error::{Error, Result};
use std::{
    collections::{HashMap, VecDeque},
    fmt,
    sync::Arc,
};

/// Configuration options for pattern matching behavior.
#[derive(Clone)]
pub struct MatchConfig {
    /// Optional custom boundary checker: returns true if the character is considered a
    /// boundary.
    ///
    /// If `None`, no special boundary logic is applied.
    pub boundary_checker: Option<Arc<dyn Fn(char) -> bool + Send + Sync>>,
    /// Only report the longest match at each position.
    pub longest_match_only: bool,
}

// Manually implement Debug since `Arc<dyn Fn(...)>` doesn't implement Debug by default.
impl fmt::Debug for MatchConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MatchConfig")
            // We won't try to debug the actual closure. We just indicate its presence.
            .field(
                "boundary_checker",
                &match self.boundary_checker {
                    Some(_) => "Some(<fn>)",
                    None => "None",
                },
            )
            .field("longest_match_only", &self.longest_match_only)
            .finish()
    }
}

// Manual implementation needed because Arc<dyn Fn> doesn't implement Default
#[allow(clippy::derivable_impls)]
impl Default for MatchConfig {
    fn default() -> Self {
        Self {
            boundary_checker: None,
            longest_match_only: false,
        }
    }
}

/// Represents a match found by the Aho-Corasick algorithm.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Match {
    /// Index of the matched pattern in the original patterns vector.
    pub pattern_index: usize,
    /// Start position of the match in the text (byte index).
    pub start: usize,
    /// End position of the match in the text (byte index, exclusive).
    pub end: usize,
}

/// A node in the trie structure.
#[derive(Debug)]
struct TrieNode {
    /// Children nodes indexed by character.
    children: HashMap<char, usize>,
    /// Failure link to the longest proper suffix node.
    failure: Option<usize>,
    /// All pattern indices that end at this node.
    output: Vec<usize>,
    /// Depth in the trie (for match position calculation).
    depth: usize,
}

impl TrieNode {
    fn new(depth: usize) -> Self {
        Self {
            children: HashMap::new(),
            failure: None,
            output: Vec::new(),
            depth,
        }
    }
}

/// An implementation of the Aho-Corasick string matching algorithm.
#[derive(Debug)]
pub struct AhoCorasick {
    /// All nodes in the automaton.
    nodes: Vec<TrieNode>,
    /// Original patterns for reporting matches.
    patterns: Vec<String>,
    /// Root node index (always 0).
    root: usize,
    /// Configuration for pattern matching behavior.
    config: MatchConfig,
}

impl AhoCorasick {
    /// Creates a new Aho-Corasick automaton from the given patterns with default
    /// configuration.
    pub fn new(patterns: Vec<String>) -> Result<Self> {
        Self::with_config(patterns, MatchConfig::default())
    }

    /// Creates a new Aho-Corasick automaton with the specified configuration.
    pub fn with_config(patterns: Vec<String>, config: MatchConfig) -> Result<Self> {
        // Validate patterns.
        if patterns.is_empty() {
            return Err(Error::invalid_input("At least one pattern is required"));
        }
        if patterns.iter().any(|p| p.is_empty()) {
            return Err(Error::empty_pattern());
        }

        let mut ac = Self {
            nodes: vec![TrieNode::new(0)],
            patterns,
            root: 0,
            config,
        };

        // Build trie and failure links.
        ac.build_trie()?;
        ac.build_failure_links();
        Ok(ac)
    }

    /// Builds the initial trie structure from the patterns.
    fn build_trie(&mut self) -> Result<()> {
        for (pattern_idx, pattern) in self.patterns.iter().enumerate() {
            let mut current = self.root;

            // Follow/create path for each character.
            for ch in pattern.chars() {
                // Instead of using or_insert_with (which causes E0500),
                // we explicitly check if the child exists or not.
                if let Some(&next) = self.nodes[current].children.get(&ch) {
                    current = next;
                } else {
                    let new_idx = self.nodes.len();
                    self.nodes
                        .push(TrieNode::new(self.nodes[current].depth + 1));
                    self.nodes[current].children.insert(ch, new_idx);
                    current = new_idx;
                }
            }
            // Store the index of this pattern in the output list.
            self.nodes[current].output.push(pattern_idx);
        }
        Ok(())
    }

    /// Builds failure links using a breadth-first traversal of the trie.
    fn build_failure_links(&mut self) {
        let mut queue = VecDeque::new();

        // Initialize root's children.
        let root_children: Vec<_> = self.nodes[self.root].children.values().copied().collect();
        for child in root_children {
            self.nodes[child].failure = Some(self.root);
            queue.push_back(child);
        }

        // Process remaining nodes.
        while let Some(current) = queue.pop_front() {
            let current_failure = self.nodes[current].failure.unwrap_or(self.root);
            let children: Vec<(char, usize)> = self.nodes[current]
                .children
                .iter()
                .map(|(ch, &node)| (*ch, node))
                .collect();

            for (ch, child) in children {
                queue.push_back(child);

                // Find the failure link by following parent's failure.
                let mut fail_state = current_failure;
                let mut next_failure = self.root;
                while fail_state != self.root {
                    if let Some(&next) = self.nodes[fail_state].children.get(&ch) {
                        next_failure = next;
                        break;
                    }
                    fail_state = self.nodes[fail_state].failure.unwrap_or(self.root);
                }
                // Check root's children if needed.
                if fail_state == self.root {
                    if let Some(&next) = self.nodes[self.root].children.get(&ch) {
                        next_failure = next;
                    }
                }

                // Set failure link.
                self.nodes[child].failure = Some(next_failure);
                // Merge outputs from the failure link.
                let output_clone = self.nodes[next_failure].output.clone();
                self.nodes[child].output.extend_from_slice(&output_clone);
            }
        }
    }

    /// Finds the next trie state given the current state and an input character.
    fn find_next_state(&self, mut current: usize, ch: char) -> usize {
        while !self.nodes[current].children.contains_key(&ch) && current != self.root {
            current = self.nodes[current].failure.unwrap_or(self.root);
        }
        self.nodes[current]
            .children
            .get(&ch)
            .copied()
            .unwrap_or(self.root)
    }

    /// Helper function to check if a match is at a word boundary.
    ///
    /// If `boundary_checker` is `None`, we do no special check (always return true).
    fn is_word_boundary(&self, text: &str, start: usize, end: usize) -> bool {
        // If no boundary checker is provided, don't filter anything out.
        let Some(check_fn) = &self.config.boundary_checker else {
            return true;
        };

        let is_boundary_char = |c: char| check_fn(c);

        let before_is_boundary = start == 0
            || text[..start]
                .chars()
                .next_back()
                .is_none_or(is_boundary_char);
        let after_is_boundary =
            end >= text.len() || text[end..].chars().next().is_none_or(is_boundary_char);

        before_is_boundary && after_is_boundary
    }

    /// Finds all occurrences of any pattern in the given text.
    pub fn find_all<'a>(&'a self, text: &'a str) -> impl Iterator<Item = Match> + 'a {
        let mut matches = Vec::new();
        let mut current = self.root;

        // Convert text to (byte_offset, char).
        let chars: Vec<(usize, char)> = text.char_indices().collect();

        // If longest_match_only is set, we collect matches per position.
        let mut matches_at_pos = if self.config.longest_match_only {
            vec![Vec::new(); chars.len()]
        } else {
            Vec::new()
        };

        for (pos, (byte_offset, ch)) in chars.iter().enumerate() {
            current = self.find_next_state(current, *ch);

            // Check outputs for the current node.
            for &pattern_idx in &self.nodes[current].output {
                let pat_len = self.patterns[pattern_idx].chars().count();
                if pos + 1 >= pat_len {
                    let start_pos = pos + 1 - pat_len;
                    let start_byte = chars[start_pos].0;
                    let end_byte = byte_offset + ch.len_utf8();

                    // Check word boundaries if needed.
                    if self.is_word_boundary(text, start_byte, end_byte) {
                        let m = Match {
                            pattern_index: pattern_idx,
                            start: start_byte,
                            end: end_byte,
                        };
                        if self.config.longest_match_only {
                            matches_at_pos[start_pos].push(m);
                        } else {
                            matches.push(m);
                        }
                    }
                }
            }
        }

        // If we only want the longest match per start position.
        if self.config.longest_match_only {
            for pos_matches in matches_at_pos.into_iter().filter(|v| !v.is_empty()) {
                // Sort by (longest match first, then pattern index).
                let mut pos_matches = pos_matches;
                pos_matches
                    .sort_by_key(|m| (-(m.end as isize - m.start as isize), m.pattern_index));
                matches.push(pos_matches[0].clone());
            }
        }

        matches.into_iter()
    }

    /// Finds the first occurrence of any pattern in the given text.
    pub fn find_first(&self, text: &str) -> Option<Match> {
        self.find_all(text).next()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_single_pattern() {
        let ac = AhoCorasick::new(vec!["test".to_string()]).unwrap();
        let matches: Vec<_> = ac.find_all("this is a test case").collect();
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].pattern_index, 0);
        assert_eq!(matches[0].start, 10);
        assert_eq!(matches[0].end, 14);
    }

    #[test]
    fn test_multiple_patterns() {
        let patterns: Vec<String> = vec!["he", "she", "his", "hers"]
            .into_iter()
            .map(String::from)
            .collect();

        let ac = AhoCorasick::new(patterns.clone()).unwrap();
        let matches: Vec<_> = ac.find_all("she sells seashells").collect();
        // "she" at index 0 => "he" is found in "she", "sells", "seashells".
        assert_eq!(matches.len(), 4);

        // With boundary checker => only "she" at start is valid.
        let mut config = MatchConfig::default();
        config.boundary_checker = Some(Arc::new(|c: char| !c.is_alphanumeric()));

        let ac = AhoCorasick::with_config(patterns.clone(), config).unwrap();
        let matches: Vec<_> = ac.find_all("she sells seashells").collect();
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].pattern_index, 1);
        assert_eq!(matches[0].start, 0);
        assert_eq!(matches[0].end, 3);
    }

    #[test]
    fn test_overlapping_patterns() {
        let patterns: Vec<String> = vec!["ant", "ant colony", "colony"]
            .into_iter()
            .map(String::from)
            .collect();

        // Default config => all matches.
        let ac = AhoCorasick::new(patterns.clone()).unwrap();
        let matches: Vec<_> = ac.find_all("ant colony").collect();
        assert_eq!(matches.len(), 3);

        // Longest match only => "ant colony" and "colony".
        let mut config = MatchConfig::default();
        config.longest_match_only = true;
        let ac = AhoCorasick::with_config(patterns.clone(), config).unwrap();
        let matches: Vec<_> = ac.find_all("ant colony").collect();
        assert_eq!(matches.len(), 2);
    }

    #[test]
    fn test_unicode() {
        // Provide explicit type to avoid E0282
        let patterns: Vec<String> = vec!["🦀", "🦀🔧", "🔧"]
            .into_iter()
            .map(String::from)
            .collect();

        // Default config => all matches.
        let ac = AhoCorasick::new(patterns.clone()).unwrap();
        let matches: Vec<_> = ac.find_all("🦀🔧").collect();
        assert_eq!(matches.len(), 3);

        // Longest match only => "🦀🔧" and "🔧".
        let mut config = MatchConfig::default();
        config.longest_match_only = true;
        let ac = AhoCorasick::with_config(patterns, config).unwrap();
        let matches: Vec<_> = ac.find_all("🦀🔧").collect();
        assert_eq!(matches.len(), 2);
    }
}
