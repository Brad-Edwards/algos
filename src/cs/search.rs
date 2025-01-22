pub mod bfs;
pub mod binary;
pub mod dfs;
pub mod exponential;
pub mod fibonacci;
pub mod interpolation;
pub mod jump;
pub mod linear;
pub mod sublist;
pub mod ternary;

// Re-export graph types
pub use bfs::Graph as BfsGraph;
pub use dfs::Graph as DfsGraph;

// Re-export search functions
pub use binary::search as binary_search;
pub use exponential::search as exponential_search;
pub use fibonacci::search as fibonacci_search;
pub use interpolation::search as interpolation_search;
pub use jump::search as jump_search;
pub use linear::search as linear_search;
pub use sublist::{search as sublist_search, search_kmp as sublist_search_kmp};
pub use ternary::search as ternary_search;
