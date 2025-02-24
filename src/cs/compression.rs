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

pub mod lz77;
pub use lz77::{compress as lz77_compress, decompress as lz77_decompress, Token as Lz77Token};

pub mod lz78;
pub use lz78::{compress as lz78_compress, decompress as lz78_decompress, Token as Lz78Token};

pub mod lzw;
pub use lzw::{compress as lzw_compress, decompress as lzw_decompress};

pub mod arithmetic;
pub use arithmetic::{
    arithmetic_decode, arithmetic_encode, FrequencyModel, ALPHABET_SIZE as ARITHMETIC_ALPHABET_SIZE,
};

pub mod deflate;
pub use deflate::algorithm::{deflate_compress, deflate_decompress, Token as DeflateToken};

pub mod bwt;
pub use bwt::{
    bwt_transform, bwt_inverse, bwt_transform_suffix_array, 
    bzip2_compress, bzip2_decompress,
    move_to_front_transform, move_to_front_inverse,
    run_length_encode, run_length_decode,
};

// Re-export specific implementations as they are added
// pub mod rle;
