pub mod bellman_equation;
pub mod coin_change;
pub mod edit_distance;
pub mod kadane;
pub mod knuth_optimization;
pub mod longest_common_subsequence;
pub mod longest_increasing_subsequence;
pub mod matrix_chain;
pub mod viterbi;
pub mod weighted_interval;

// Re-export dynamic programming algorithms with descriptive names
pub use bellman_equation::{value_iteration, MarkovDecisionProcess};
pub use coin_change::{count_change_ways, min_coins_for_change};
pub use edit_distance::levenshtein_distance;
pub use kadane::kadane;
pub use knuth_optimization::{min_merge_cost_knuth, reconstruct_optimal_merge};
pub use longest_common_subsequence::{lcs_length, lcs_sequence};
pub use longest_increasing_subsequence::{
    longest_increasing_subsequence, longest_increasing_subsequence_length,
};
pub use matrix_chain::optimal_matrix_chain_multiplication;
pub use weighted_interval::{best_weighted_schedule, max_weighted_schedule, WeightedInterval};
