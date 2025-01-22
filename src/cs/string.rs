pub mod aho_corasick;
pub mod boyer_moore;
pub mod kmp;
pub mod manacher;
pub mod rabin_karp;
pub mod rolling_hash;
pub mod suffix_array;
pub mod suffix_automaton;
pub mod suffix_tree;
pub mod z_algorithm;

// Re-export types
pub use aho_corasick::{AhoCorasick, Match, MatchConfig};
pub use rolling_hash::RollingHash;
pub use suffix_array::{SearchResult, SuffixArray};
pub use suffix_automaton::SuffixAutomaton;
pub use suffix_tree::{SuffixNode, SuffixTree};

// Re-export string matching functions
pub use boyer_moore::{find_all as boyer_moore_find_all, find_first as boyer_moore_find_first};
pub use kmp::{find_all as kmp_find_all, find_first as kmp_find_first};
pub use manacher::longest_palindrome;
pub use rabin_karp::{find_all as rabin_karp_find_all, find_first as rabin_karp_find_first};
pub use suffix_array::{find_all as suffix_array_find_all, find_first as suffix_array_find_first};
pub use z_algorithm::{find_all as z_algorithm_find_all, find_first as z_algorithm_find_first};
