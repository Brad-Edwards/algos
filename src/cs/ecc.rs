//! Error correction code implementations.
//!
//! This module provides implementations of various error correction codes:
//! - Reed-Solomon codes
//! - Hamming codes
//! - Convolutional codes
//! - Turbo codes
//! - LDPC codes
//! - Fountain codes
//!
//! # Error Correction Algorithms
//!
//! Error correction codes are used to detect and correct errors in data
//! transmission and storage, making digital communications more reliable.
//!
//! Currently implemented:
//! - Reed-Solomon codes
//! - Hamming codes
//! - Convolutional codes
//!
//! # Examples
//!
//! ```rust
//! // Examples will be added as algorithms are implemented
//! ```

use crate::cs::error::Error;

/// Result type for error correction operations
pub type Result<T> = std::result::Result<T, Error>;

/// Trait for error correction code implementations
pub trait ErrorCorrection {
    /// Encode data with error correction symbols
    fn encode(&self, data: &[u8]) -> Result<Vec<u8>>;

    /// Decode data and correct errors if possible
    fn decode(&self, data: &[u8]) -> Result<Vec<u8>>;
}

/// Reed-Solomon error correction codes
pub mod reed_solomon;
pub use reed_solomon::{
    create_reed_solomon, reed_solomon_decode, reed_solomon_encode, GFElement, ReedSolomon,
};

/// Hamming error correction codes
pub mod hamming;
pub use hamming::{
    create_hamming, create_hamming_7_4, create_hamming_8_4, hamming_decode, hamming_encode,
    hamming_extended_decode, hamming_extended_encode, HammingCode,
};

/// Convolutional error correction codes
pub mod convolutional;
pub use convolutional::{
    convolutional_decode, convolutional_encode, create_convolutional_code,
    create_nasa_standard_code, create_rate_third_code, create_rate_two_thirds_code,
    ConvolutionalCode,
};
