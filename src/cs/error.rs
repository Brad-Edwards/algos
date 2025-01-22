use std::{collections::TryReserveError, fmt::Display};
use thiserror::Error;

/// Errors that can occur during algorithm operations
#[derive(Debug, Error)]
pub enum Error {
    /// The pattern is empty
    #[error("Pattern cannot be empty")]
    EmptyPattern,

    /// The pattern is longer than the text
    #[error("Pattern length {pattern_len} is longer than text length {text_len}")]
    PatternTooLong { pattern_len: usize, text_len: usize },

    /// The recursion depth exceeded the maximum
    #[error("Recursion depth {depth} exceeded maximum allowed depth of {max_depth}")]
    RecursionLimitExceeded { depth: usize, max_depth: usize },

    /// Failed to allocate memory
    #[error("Failed to allocate memory: {reason}")]
    AllocationFailed {
        reason: String,
        #[source]
        source: Option<TryReserveError>,
    },

    /// A parallel execution task failed
    #[error("Parallel execution failed: {reason}")]
    ParallelExecutionFailed { reason: String },

    /// The input was too large
    #[error("Input length {length} exceeds maximum supported length of {max_length}")]
    InputTooLarge { length: usize, max_length: usize },

    /// Index out of bounds
    #[error("Index out of bounds: {0}")]
    IndexOutOfBounds(String),

    /// Operation not supported
    #[error("Operation not supported: {0}")]
    Unsupported(String),

    /// Invalid input
    #[error("Invalid input: {0}")]
    InvalidInput(String),

    /// Vertex not found
    #[error("vertex not found")]
    VertexNotFound,

    /// A negative cycle was detected in the graph
    #[error("negative cycle detected in graph")]
    NegativeCycle,
}

/// Result type for algorithm operations
pub type Result<T> = std::result::Result<T, Error>;

/// Common error types for algorithm operations.
/// Note: Many modules define their own specific error types (e.g., HeapSortError)
/// for more precise error handling. These common errors are provided for
/// standardization across modules where appropriate.
impl Error {
    pub(crate) fn empty_pattern() -> Self {
        Self::EmptyPattern
    }

    pub(crate) fn pattern_too_long(pattern_len: usize, text_len: usize) -> Self {
        Self::PatternTooLong {
            pattern_len,
            text_len,
        }
    }

    pub(crate) fn recursion_limit_exceeded(depth: usize, max_depth: usize) -> Self {
        Self::RecursionLimitExceeded { depth, max_depth }
    }

    pub(crate) fn input_too_large(length: usize, max_length: usize) -> Self {
        Self::InputTooLarge { length, max_length }
    }

    pub(crate) fn invalid_input(msg: impl Display) -> Self {
        Self::InvalidInput(msg.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_messages() {
        let err = Error::empty_pattern();
        assert_eq!(err.to_string(), "Pattern cannot be empty");

        let err = Error::pattern_too_long(10, 5);
        assert_eq!(
            err.to_string(),
            "Pattern length 10 is longer than text length 5"
        );

        let err = Error::recursion_limit_exceeded(100, 50);
        assert_eq!(
            err.to_string(),
            "Recursion depth 100 exceeded maximum allowed depth of 50"
        );

        let err = Error::input_too_large(1_000_000, 100_000);
        assert_eq!(
            err.to_string(),
            "Input length 1000000 exceeds maximum supported length of 100000"
        );

        let err = Error::invalid_input("invalid UTF-8");
        assert_eq!(err.to_string(), "Invalid input: invalid UTF-8");
    }

    #[test]
    fn test_error_variants() {
        // Test that error variants exist but are handled by specific modules
        let err = Error::AllocationFailed {
            reason: "test".to_string(),
            source: None,
        };
        assert!(err.to_string().contains("Failed to allocate memory"));

        let err = Error::ParallelExecutionFailed {
            reason: "test".to_string(),
        };
        assert!(err.to_string().contains("Parallel execution failed"));

        let err = Error::IndexOutOfBounds("test".to_string());
        assert!(err.to_string().contains("Index out of bounds"));

        let err = Error::Unsupported("test".to_string());
        assert!(err.to_string().contains("Operation not supported"));
    }
}
