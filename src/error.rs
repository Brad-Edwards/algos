//! Error types for the algos library.
//! This module re-exports the unified error types from cs/error.rs for backward
//! compatibility.

pub use crate::cs::error::{Error, Result};

// Type aliases for backward compatibility
pub type SortError = Error;
pub type SearchError = Error;
pub type StringError = Error;

pub type SortResult<T> = Result<T>;
pub type SearchResult<T> = Result<T>;
pub type StringResult<T> = Result<T>;

// Module-specific re-exports for backward compatibility
pub mod sort {
    pub use super::Error as SortError;
    pub type Result<T> = super::Result<T>;
}

pub mod search {
    pub use super::Error as SearchError;
    pub type Result<T> = super::Result<T>;
}

pub mod string {
    pub use super::Error as StringError;
    pub type Result<T> = super::Result<T>;
}
