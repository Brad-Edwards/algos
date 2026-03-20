use thiserror::Error;

/// Errors that can occur during RL operations.
#[derive(Debug, Error)]
pub enum Error {
    /// Invalid configuration parameter.
    #[error("invalid configuration: {0}")]
    InvalidConfig(String),

    /// Dimension mismatch between expected and actual sizes.
    #[error("dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },

    /// Buffer does not have enough samples.
    #[error("insufficient samples: need {needed}, have {available}")]
    InsufficientSamples { needed: usize, available: usize },

    /// Environment returned an unexpected state.
    #[error("environment error: {0}")]
    Environment(String),

    /// Numerical computation failed (NaN, overflow, etc.).
    #[error("numerical error: {0}")]
    Numerical(String),
}

/// Result type for RL operations.
pub type Result<T> = std::result::Result<T, Error>;
