//! Error correction code implementations.
//!
//! This module provides implementations of various error correction codes:
//! - Reed-Solomon codes
//! - Hamming codes
//! - Convolutional codes
//! - Turbo codes
//! - LDPC codes
//! - CRC (Cyclic Redundancy Check)
//! - BCH (Bose-Chaudhuri-Hocquenghem) codes
//! - Polar codes
//! - Fountain/Raptor codes
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
//! - Turbo codes
//! - LDPC (Low-Density Parity-Check) codes
//! - CRC (Cyclic Redundancy Check)
//! - BCH (Bose-Chaudhuri-Hocquenghem) codes
//! - Polar codes
//! - Fountain codes (LT codes)
//! - Raptor codes (LT codes with pre-coding)
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
    create_standard_viterbi_decoder, create_viterbi_decoder, viterbi_decode, ConvolutionalCode,
    ViterbiDecoder, ViterbiResult,
};

/// Turbo error correction codes
pub mod turbo;
pub use turbo::{
    create_custom_turbo_code, create_turbo_code, turbo_decode, turbo_encode, InterleaverType,
    TurboCode, TurboResult,
};

/// LDPC (Low-Density Parity-Check) error correction codes
pub mod ldpc;
pub use ldpc::{
    create_ldpc_code, create_optimized_ldpc_code, create_wifi_ldpc_code, ldpc_decode, ldpc_encode,
    LDPCCode, LDPCResult,
};

/// CRC (Cyclic Redundancy Check) codes
pub mod crc;
pub use crc::{calculate_crc16, calculate_crc32, calculate_crc8, Crc16, Crc32, Crc8, CrcAlgorithm};

/// BCH (Bose-Chaudhuri-Hocquenghem) error correction codes
pub mod bch;
pub use bch::{
    bch_decode, bch_encode, create_bch_15_7_2, create_bch_31_16_3, create_bch_63_45_3,
    create_bch_code, BchCode,
};

/// Polar error correction codes
pub mod polar;
pub use polar::{
    create_5g_polar_code, create_polar_code, create_polar_code_for_snr, polar_decode, polar_encode,
    PolarCode,
};

/// Fountain and Raptor error correction codes
pub mod fountain;
pub use fountain::{
    create_custom_lt_code, create_custom_raptor_code, create_lt_code, create_raptor_code,
    lt_decode, lt_encode, raptor_decode, raptor_encode, EncodedBlock, LTCode, LTParameters,
    RaptorCode, RaptorParameters,
};
