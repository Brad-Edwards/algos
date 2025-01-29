use std::collections::{HashMap, HashSet};

/// A state in the suffix automaton
#[derive(Debug, Clone)]
struct State {
    /// Length of the longest substring that leads to this state
    len: usize,
    /// Suffix link
    link: Option<usize>,
    /// Transitions
    next: HashMap<char, usize>,
    /// End positions for all substrings in this state
    end_pos: HashSet<usize>,
    /// Marks if this is a terminal state
    is_terminal: bool,
}

impl State {
    fn new(len: usize) -> Self {
        Self {
            len,
            link: None,
            next: HashMap::new(),
            end_pos: HashSet::new(),
            is_terminal: false,
        }
    }

    fn add_pos(&mut self, p: usize) {
        self.end_pos.insert(p);
    }
}

/// A suffix automaton for substring queries and occurrence finding.
#[derive(Debug)]
pub struct SuffixAutomaton {
    /// The states of the automaton
    states: Vec<State>,
    /// The last state that was added
    last: usize,
}

impl SuffixAutomaton {
    /// Constructs a suffix automaton from `text`.
    pub fn new(text: &str) -> Self {
        let mut sa = Self {
            states: vec![State::new(0)], // root: length=0
            last: 0,
        };

        for (i, ch) in text.chars().enumerate() {
            sa.extend(ch, i);
        }

        // Mark terminal states along the suffix chain from `last`
        sa.mark_terminals();
        // Propagate end positions in topological order
        sa.propagate_positions();
        sa
    }

    /// Extend the automaton with character `ch` at position `pos` in the original text.
    fn extend(&mut self, ch: char, pos: usize) {
        let mut p = self.last;
        // Create a new state for the extended substring
        let cur = self.states.len();
        self.states.push(State::new(self.states[p].len + 1));
        self.states[cur].add_pos(pos);

        // Add transitions back while no edge on `ch`
        while p != usize::MAX && !self.states[p].next.contains_key(&ch) {
            self.states[p].next.insert(ch, cur);
            p = self.states[p].link.unwrap_or(usize::MAX);
        }

        if p == usize::MAX {
            // If we fell off the root
            self.states[cur].link = Some(0);
        } else {
            let q = self.states[p].next[&ch];
            if self.states[p].len + 1 == self.states[q].len {
                // We can just link to `q`
                self.states[cur].link = Some(q);
            } else {
                // Need to clone
                let clone = self.states.len();
                self.states.push(State::new(self.states[p].len + 1));
                // Copy q's transitions and link
                self.states[clone].next = self.states[q].next.clone();
                self.states[clone].link = self.states[q].link;
                // The clone initially has no end positions; they'll be set by propagate_positions()

                // Redirect transitions that pointed to q
                while p != usize::MAX && self.states[p].next.get(&ch) == Some(&q) {
                    self.states[p].next.insert(ch, clone);
                    p = self.states[p].link.unwrap_or(usize::MAX);
                }
                // Fix suffix links
                self.states[q].link = Some(clone);
                self.states[cur].link = Some(clone);
            }
        }
        self.last = cur;
    }

    /// Mark all states on the link path from `last` as terminal.
    fn mark_terminals(&mut self) {
        let mut p = self.last;
        while p != 0 {
            self.states[p].is_terminal = true;
            p = self.states[p].link.unwrap_or(0);
        }
        // Root can also be considered terminal in some definitions
        self.states[0].is_terminal = true;
    }

    /// Propagate end positions up the suffix‐link tree in topological order (by length).
    fn propagate_positions(&mut self) {
        // Sort states by length ascending
        let mut order: Vec<usize> = (0..self.states.len()).collect();
        order.sort_by_key(|&i| self.states[i].len);

        // For each state from longest to shorter, unify positions with link
        for &i in order.iter().rev() {
            let positions: Vec<_> = self.states[i].end_pos.iter().copied().collect();
            if let Some(link) = self.states[i].link {
                // unify i's positions into its link
                for p in positions {
                    self.states[link].add_pos(p);
                }
            }
        }
    }

    /// Checks if a pattern is a substring by simply walking transitions.
    pub fn contains(&self, pattern: &str) -> bool {
        let mut s = 0;
        for ch in pattern.chars() {
            match self.states[s].next.get(&ch) {
                Some(&nx) => s = nx,
                None => return false,
            }
        }
        true
    }

    /// Find all start positions of `pattern` in the original text.
    pub fn find_all(&self, pattern: &str) -> Vec<usize> {
        if pattern.is_empty() {
            return Vec::new();
        }

        // Walk the automaton
        let mut s = 0;
        for ch in pattern.chars() {
            match self.states[s].next.get(&ch) {
                Some(&nx) => s = nx,
                None => return Vec::new(),
            }
        }

        // Only collect positions from the final state we reached
        let mut result: Vec<_> = self.states[s]
            .end_pos
            .iter()
            .filter_map(|&ep| {
                let pat_len = pattern.chars().count();
                if ep + 1 >= pat_len {
                    Some(ep + 1 - pat_len)
                } else {
                    None
                }
            })
            .collect();
        result.sort_unstable();
        result.dedup();
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_construction() {
        let text = "banana";
        let sa = SuffixAutomaton::new(text);

        assert!(sa.contains("ana"));
        assert!(sa.contains("ban"));
        assert!(sa.contains("na"));
        // "banana" DOES contain "nan"
        assert!(sa.contains("nan"));
    }

    #[test]
    fn test_find_all() {
        let text = "banana";
        let sa = SuffixAutomaton::new(text);

        assert_eq!(sa.find_all("ana"), vec![1, 3]);
        assert_eq!(sa.find_all("na"), vec![2, 4]);
        assert_eq!(sa.find_all("a"), vec![1, 3, 5]);
        assert_eq!(sa.find_all("ban"), vec![0]);
        assert_eq!(sa.find_all("xyz"), Vec::<usize>::new());
    }

    #[test]
    fn test_empty_pattern() {
        let text = "banana";
        let sa = SuffixAutomaton::new(text);

        assert!(sa.contains(""));
        assert_eq!(sa.find_all(""), Vec::<usize>::new());
    }

    #[test]
    fn test_unicode_text() {
        let text = "こんにちは世界";
        let sa = SuffixAutomaton::new(text);

        assert!(sa.contains("にち"));
        assert!(sa.contains("世界"));
        assert!(!sa.contains("世に"));

        assert_eq!(sa.find_all("にち"), vec![2]);
        assert_eq!(sa.find_all("世界"), vec![5]);
    }

    #[test]
    fn test_overlapping_patterns() {
        let text = "aaaaa";
        let sa = SuffixAutomaton::new(text);

        assert_eq!(sa.find_all("aa"), vec![0, 1, 2, 3]);
        assert_eq!(sa.find_all("aaa"), vec![0, 1, 2]);
    }

    #[test]
    fn test_long_text() {
        let text = "a".repeat(1000) + "b";
        let sa = SuffixAutomaton::new(&text);

        assert!(sa.contains("aaa"));
        assert!(sa.contains("b"));
        assert!(!sa.contains("c"));

        let positions = sa.find_all("aa");
        assert_eq!(positions.len(), 999);
    }

    #[test]
    fn test_case_sensitivity() {
        let text = "bAnAnA";
        let sa = SuffixAutomaton::new(text);

        assert!(!sa.contains("ana"));
        assert!(sa.contains("AnA"));

        assert_eq!(sa.find_all("ana"), Vec::<usize>::new());
        assert_eq!(sa.find_all("AnA"), vec![1, 3]);
    }
}
