//! Compression algorithms implementation.
//!
//! This module provides implementations of various data compression algorithms:
//! - Lossless compression (Huffman, LZ77, LZ78, LZW, LZMA)
//! - Dictionary-based compression
//! - Run-length encoding
//! - Delta encoding
//! - Arithmetic coding
//! - Burrows-Wheeler transform
//! - Prediction by Partial Matching (PPM)
//! - Range Asymmetric Numeral Systems (RANS)
//!
//! # Compression Algorithms
//!
//! This module provides various compression algorithms:
//!
//! - Huffman coding
//! - LZ77 (Lempel-Ziv 77)
//! - LZ78 (Lempel-Ziv 78)
//! - LZW (Lempel-Ziv-Welch)
//! - LZMA (Lempel-Ziv-Markov chain Algorithm)
//! - Burrows-Wheeler Transform (BWT)
//! - Run-Length Encoding (RLE)
//! - Move-to-Front Transform
//! - Prediction by Partial Matching (PPM)
//! - Range Asymmetric Numeral Systems (RANS)
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

pub mod lzma;
pub use lzma::{lzma_compress, lzma_decompress};

pub mod arithmetic;
pub use arithmetic::{
    arithmetic_decode, arithmetic_encode, FrequencyModel, ALPHABET_SIZE as ARITHMETIC_ALPHABET_SIZE,
};

pub mod deflate;
pub use deflate::algorithm::{deflate_compress, deflate_decompress, Token as DeflateToken};

pub mod bwt;
pub use bwt::{
    bwt_inverse, bwt_transform, bwt_transform_suffix_array, bzip2_compress, bzip2_decompress,
    move_to_front_inverse, move_to_front_transform, run_length_decode, run_length_encode,
};

pub mod ppm;
pub use ppm::{ppm_compress, ppm_decompress, ppm_star_compress, ppm_star_decompress};

pub mod rans;
pub use rans::{
    rans_decode, rans_encode, FrequencyModel as RansFrequencyModel, RansDecoder, RansEncoder,
};

// Re-export specific implementations as they are added
// pub mod rle;
