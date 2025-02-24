//! Compression algorithms implementation.
//!
//! This module provides implementations of various data compression algorithms:
//! - Lossless compression (Huffman, LZ77, LZ78, LZW)
//! - Dictionary-based compression
//! - Run-length encoding
//! - Delta encoding
//! - Arithmetic coding
//! - Burrows-Wheeler transform
//!
//! # Examples
//!
//! ```rust
//! // Examples will be added as algorithms are implemented
//! ```

use crate::cs::error::Error;

/// Result type for compression operations
pub type Result<T> = std::result::Result<T, Error>;

/// Trait for compression algorithms
pub trait Compression {
    /// Compress the input data
    fn compress(&self, data: &[u8]) -> Result<Vec<u8>>;

    /// Decompress the compressed data
    fn decompress(&self, data: &[u8]) -> Result<Vec<u8>>;
}

/// Trait for streaming compression algorithms
pub trait StreamingCompression {
    /// Process a chunk of input data
    fn process(&mut self, chunk: &[u8]) -> Result<Vec<u8>>;

    /// Finish processing and return any remaining data
    fn finish(&mut self) -> Result<Vec<u8>>;
}

pub mod huffman;
pub use huffman::{
    build_code_table, build_frequency_table, build_huffman_tree, decode, encode, huffman_decode,
    huffman_encode, HuffmanNode,
};

// Re-export specific implementations as they are added
// pub mod lz77;
// pub mod lz78;
// pub mod lzw;
// pub mod rle;
// pub mod arithmetic;
// pub mod bwt;
