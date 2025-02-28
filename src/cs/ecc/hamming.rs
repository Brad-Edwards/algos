//! Hamming error correction code implementation.
//!
//! Hamming codes are a family of linear error-correcting codes developed by Richard Hamming in 1950.
//! They have the ability to detect up to two-bit errors or correct one-bit errors without detection
//! of uncorrected errors. The most common variant is (7,4) Hamming code, which encodes 4 data bits
//! into 7 bits by adding 3 parity bits.
//!
//! This implementation provides:
//! - Encoding of data with configurable Hamming codes
//! - Decoding with single-bit error correction
//! - Support for extended Hamming codes with additional parity bit
//!
//! # Applications
//!
//! - Computer memory (ECC RAM)
//! - Satellite communications
//! - Digital broadcasting
//! - Data storage systems

use crate::cs::ecc::Result;
use crate::cs::error::Error;
use bitvec::prelude::*;
use bitvec::view::BitView;
use std::cmp::min;

/// Represents a Hamming code configuration.
/// A Hamming(m,r) code encodes m data bits with r parity bits.
#[derive(Debug, Clone, Copy)]
pub struct HammingCode {
    /// Number of data bits per block
    data_bits: usize,
    /// Number of parity bits per block
    parity_bits: usize,
    /// Whether to use extended Hamming code with additional parity bit
    extended: bool,
}

impl HammingCode {
    /// Creates a new Hamming code configuration.
    ///
    /// # Arguments
    ///
    /// * `data_bits` - Number of data bits to encode in each block
    /// * `extended` - Whether to use extended Hamming code with additional parity bit
    ///
    /// # Returns
    ///
    /// A new `HammingCode` instance or an error if invalid parameters
    pub fn new(data_bits: usize, extended: bool) -> Result<Self> {
        if data_bits == 0 {
            return Err(Error::InvalidInput(
                "Data bits must be positive".to_string(),
            ));
        }

        // Determine required number of parity bits, where 2^r - r - 1 >= data_bits
        let mut parity_bits = 2;
        while (1 << parity_bits) - parity_bits - 1 < data_bits {
            parity_bits += 1;
        }

        Ok(HammingCode {
            data_bits,
            parity_bits,
            extended,
        })
    }

    /// Creates a standard (7,4) Hamming code
    pub fn standard_7_4() -> Self {
        // We know this is valid, no need to handle Result
        Self::new(4, false).unwrap()
    }

    /// Creates an extended (8,4) Hamming code
    pub fn extended_8_4() -> Self {
        // We know this is valid, no need to handle Result
        Self::new(4, true).unwrap()
    }

    /// Gets the total code word length (data bits + parity bits)
    pub fn total_bits(&self) -> usize {
        let base = self.data_bits + self.parity_bits;
        if self.extended {
            base + 1
        } else {
            base
        }
    }

    /// Gets the number of encoded bytes needed for a given number of input bytes
    pub fn encoded_bytes_needed(&self, input_bytes: usize) -> usize {
        let total_input_bits = input_bytes * 8;
        let total_blocks = total_input_bits.div_ceil(self.data_bits);
        let total_output_bits = total_blocks * self.total_bits();
        (total_output_bits + 7) / 8 // Round up to bytes
    }

    /// Encodes a byte slice using the configured Hamming code
    ///
    /// # Arguments
    ///
    /// * `data` - Data to encode
    ///
    /// # Returns
    ///
    /// The encoded data with parity bits
    pub fn encode(&self, data: &[u8]) -> Vec<u8> {
        if data.is_empty() {
            return Vec::new();
        }

        // Convert input bytes to bits
        let data_bits = data.view_bits::<Msb0>();

        // Calculate number of complete data blocks and required output size
        let data_blocks = data_bits.len().div_ceil(self.data_bits);
        let out_bits_len = data_blocks * self.total_bits();

        // Create output buffer
        let mut encoded = bitvec![u8, Msb0; 0; out_bits_len];

        // Process each block
        for block_idx in 0..data_blocks {
            let input_start = block_idx * self.data_bits;
            let output_start = block_idx * self.total_bits();

            // Encode one block
            self.encode_block(
                &data_bits[input_start..min(input_start + self.data_bits, data_bits.len())],
                &mut encoded[output_start..output_start + self.total_bits()],
            );
        }

        // Convert back to bytes
        encoded.as_raw_slice().to_vec()
    }

    /// Encodes a single block of data bits
    fn encode_block(&self, data_bits: &BitSlice<u8, Msb0>, output: &mut BitSlice<u8, Msb0>) {
        // We'll use a simple approach: position parity bits at powers of 2
        // and data bits at other positions

        // Copy data bits to correct positions in output (non-parity positions)
        let mut data_idx = 0;
        for i in 0..self.total_bits() {
            // Skip positions that are powers of 2 (0-indexed)
            if i + 1 > 0 && (i + 1).count_ones() == 1 {
                continue;
            }

            // Copy data bit if available, or use 0
            if data_idx < data_bits.len() {
                output.set(i, data_bits[data_idx]);
            } else {
                output.set(i, false);
            }
            data_idx += 1;
        }

        // Calculate parity bits
        for r in 0..self.parity_bits {
            let _parity_pos = (1 << r) - 1; // 0-indexed positions are 0, 1, 3, 7, etc.
            let mut parity = false;

            // Check all positions where bit r in the position is 1
            for i in 0..self.total_bits() {
                if i == _parity_pos {
                    continue;
                }

                // Check if bit r is set in the position (1-indexed for calculation)
                if ((i + 1) & (1 << r)) != 0 && output[i] {
                    parity = !parity;
                }
            }

            // Set parity bit
            output.set(_parity_pos, parity);
        }

        // For extended Hamming code, add overall parity bit
        if self.extended {
            let mut overall_parity = false;
            for i in 0..self.total_bits() - 1 {
                if output[i] {
                    overall_parity = !overall_parity;
                }
            }
            output.set(self.total_bits() - 1, overall_parity);
        }
    }

    /// Decodes Hamming-encoded data, correcting single-bit errors
    ///
    /// # Arguments
    ///
    /// * `encoded` - Data to decode
    ///
    /// # Returns
    ///
    /// The decoded data with errors corrected
    pub fn decode(&self, encoded: &[u8]) -> Result<Vec<u8>> {
        if encoded.is_empty() {
            return Ok(Vec::new());
        }

        // Check if we have enough data
        let min_bytes_needed = (self.total_bits() + 7) / 8;
        if encoded.len() < min_bytes_needed {
            return Err(Error::InvalidInput(format!(
                "Encoded data too short, need at least {} bytes",
                min_bytes_needed
            )));
        }

        // Convert input bytes to bits
        let encoded_bits = encoded.view_bits::<Msb0>();

        // Calculate number of complete blocks and required output size
        let blocks = encoded_bits.len() / self.total_bits();
        let out_bits_len = blocks * self.data_bits;

        // Create output buffer
        let mut decoded = bitvec![u8, Msb0; 0; out_bits_len];

        // Process each block
        for block_idx in 0..blocks {
            let input_start = block_idx * self.total_bits();
            let output_start = block_idx * self.data_bits;

            // Decode one block
            self.decode_block(
                &encoded_bits[input_start..input_start + self.total_bits()],
                &mut decoded[output_start..output_start + self.data_bits],
            )?;
        }

        // Convert back to bytes
        Ok(decoded.as_raw_slice().to_vec())
    }

    /// Decodes a single block of encoded bits
    fn decode_block(
        &self,
        encoded: &BitSlice<u8, Msb0>,
        output: &mut BitSlice<u8, Msb0>,
    ) -> Result<()> {
        // Calculate syndrome to detect errors
        let mut syndrome = 0;

        // Check each parity bit
        for r in 0..self.parity_bits {
            let _parity_pos = (1 << r) - 1;
            let mut parity = false;

            // Calculate parity across all bits covered by this parity bit
            for i in 0..encoded.len() {
                if ((i + 1) & (1 << r)) != 0 && encoded[i] {
                    parity = !parity;
                }
            }

            // If parity doesn't match, record in syndrome
            if parity {
                syndrome |= 1 << r;
            }
        }

        // Handle error correction
        if syndrome != 0 {
            // For extended code, check overall parity
            if self.extended {
                let mut overall_parity = false;
                for i in 0..encoded.len() - 1 {
                    if encoded[i] {
                        overall_parity = !overall_parity;
                    }
                }

                let expected_parity = encoded[encoded.len() - 1];

                // If overall parity matches but syndrome is non-zero, we have more than 1 error
                if overall_parity == expected_parity {
                    return Err(Error::InvalidInput(
                        "Detected more than one bit error in extended Hamming code".to_string(),
                    ));
                }
            }

            // For regular code, if syndrome is valid position, correct it
            if syndrome <= encoded.len() {
                // Create a corrected copy of the encoded data
                let mut corrected = encoded.to_bitvec();

                // Flip the bit at the error position
                let error_pos = syndrome - 1; // Convert to 0-indexed
                corrected.set(error_pos, !encoded[error_pos]);

                // Extract data bits to output
                let mut data_idx = 0;
                for i in 0..corrected.len() {
                    // Skip parity bit positions
                    if i + 1 > 0 && (i + 1).count_ones() == 1 {
                        continue;
                    }

                    // Skip overall parity bit for extended code
                    if self.extended && i == corrected.len() - 1 {
                        continue;
                    }

                    if data_idx < output.len() {
                        output.set(data_idx, corrected[i]);
                        data_idx += 1;
                    }
                }
            } else {
                return Err(Error::InvalidInput(format!(
                    "Invalid syndrome: {}, suggesting uncorrectable errors",
                    syndrome
                )));
            }
        } else {
            // No errors detected, just extract data bits
            let mut data_idx = 0;
            for i in 0..encoded.len() {
                // Skip parity bit positions
                if i + 1 > 0 && (i + 1).count_ones() == 1 {
                    continue;
                }

                // Skip overall parity bit for extended code
                if self.extended && i == encoded.len() - 1 {
                    continue;
                }

                if data_idx < output.len() {
                    output.set(data_idx, encoded[i]);
                    data_idx += 1;
                }
            }
        }

        Ok(())
    }
}

/// Creates a standard (7,4) Hamming code
pub fn create_hamming_7_4() -> HammingCode {
    HammingCode::standard_7_4()
}

/// Creates an extended (8,4) Hamming code
pub fn create_hamming_8_4() -> HammingCode {
    HammingCode::extended_8_4()
}

/// Creates a Hamming code with custom parameters
pub fn create_hamming(data_bits: usize, extended: bool) -> Result<HammingCode> {
    HammingCode::new(data_bits, extended)
}

/// Encodes data using standard (7,4) Hamming code
pub fn hamming_encode(data: &[u8]) -> Vec<u8> {
    let hamming = create_hamming_7_4();
    hamming.encode(data)
}

/// Decodes data using standard (7,4) Hamming code
pub fn hamming_decode(encoded: &[u8]) -> Result<Vec<u8>> {
    let hamming = create_hamming_7_4();
    hamming.decode(encoded)
}

/// Encodes data using extended (8,4) Hamming code
pub fn hamming_extended_encode(data: &[u8]) -> Vec<u8> {
    let hamming = create_hamming_8_4();
    hamming.encode(data)
}

/// Decodes data using extended (8,4) Hamming code
pub fn hamming_extended_decode(encoded: &[u8]) -> Result<Vec<u8>> {
    let hamming = create_hamming_8_4();
    hamming.decode(encoded)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hamming_creation() {
        // Standard (7,4) Hamming code
        let hamming = HammingCode::standard_7_4();
        assert_eq!(hamming.data_bits, 4);
        assert_eq!(hamming.parity_bits, 3);
        assert_eq!(hamming.extended, false);
        assert_eq!(hamming.total_bits(), 7);

        // Extended (8,4) Hamming code
        let hamming = HammingCode::extended_8_4();
        assert_eq!(hamming.data_bits, 4);
        assert_eq!(hamming.parity_bits, 3);
        assert_eq!(hamming.extended, true);
        assert_eq!(hamming.total_bits(), 8);

        // Custom Hamming code
        let hamming = HammingCode::new(8, false).unwrap();
        assert_eq!(hamming.data_bits, 8);
        assert_eq!(hamming.parity_bits, 4);
        assert_eq!(hamming.extended, false);
        assert_eq!(hamming.total_bits(), 12);
    }

    #[test]
    fn test_hamming_encode_decode_no_errors() {
        // Test with standard (7,4) code
        let data = b"Test data for Hamming code";
        let hamming = create_hamming_7_4();

        let encoded = hamming.encode(data);
        let decoded = hamming.decode(&encoded).unwrap();

        assert_eq!(decoded, data);

        // Test with extended (8,4) code
        let hamming = create_hamming_8_4();

        let encoded = hamming.encode(data);
        let decoded = hamming.decode(&encoded).unwrap();

        assert_eq!(decoded, data);
    }

    #[test]
    fn test_hamming_error_correction() {
        // Test with standard (7,4) code
        let data = b"Test";
        let hamming = create_hamming_7_4();

        let mut encoded = hamming.encode(data);

        // Introduce a single bit error in the first byte
        encoded[0] ^= 0x40; // Flip the second bit

        let decoded = hamming.decode(&encoded).unwrap();
        assert_eq!(decoded, data);

        // Test with extended (8,4) code
        let hamming = create_hamming_8_4();

        let mut encoded = hamming.encode(data);

        // Introduce a single bit error
        encoded[0] ^= 0x40; // Flip the second bit

        let decoded = hamming.decode(&encoded).unwrap();
        assert_eq!(decoded, data);
    }

    #[test]
    fn test_hamming_multi_bit_error_detection() {
        // Extended Hamming can detect (but not correct) 2-bit errors
        let data = b"Test";
        let hamming = create_hamming_8_4();

        let mut encoded = hamming.encode(data);

        // Introduce two bit errors in the first byte
        encoded[0] ^= 0x40; // Flip the second bit
        encoded[0] ^= 0x20; // Flip the third bit

        // Should detect the error but not be able to correct it
        let result = hamming.decode(&encoded);
        assert!(result.is_err());
    }

    #[test]
    fn test_empty_input() {
        let hamming = create_hamming_7_4();

        let encoded = hamming.encode(&[]);
        assert!(encoded.is_empty());

        let decoded = hamming.decode(&[]).unwrap();
        assert!(decoded.is_empty());
    }

    #[test]
    fn test_helper_functions() {
        let data = b"Test";

        // Test standard Hamming functions
        let encoded = hamming_encode(data);
        let decoded = hamming_decode(&encoded).unwrap();
        assert_eq!(decoded, data);

        // Test extended Hamming functions
        let encoded = hamming_extended_encode(data);
        let decoded = hamming_extended_decode(&encoded).unwrap();
        assert_eq!(decoded, data);
    }
}
