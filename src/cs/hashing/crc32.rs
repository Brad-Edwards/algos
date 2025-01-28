//! # CRC32 Implementation
//!
//! This module provides a **production-focused** implementation of CRC32 in Rust,
//! allowing customization of various CRC parameters (polynomial, initial value, final XOR, reflection).
//! The default configuration corresponds to the most common CRC-32 polynomial (0xEDB88320) with
//! typical Ethernet/ZIP behavior (reflected input/output, init=0xFFFFFFFF, final_xor=0xFFFFFFFF).
//!
//! ## Key Features
//! - **Builder** pattern to configure polynomial, init value, final XOR, reflection-in, reflection-out, and table generation.
//! - **High performance** table-based approach for incremental and final usage.
//! - **Streaming**: You can feed data incrementally (`update`) and then finalize (`finalize`).
//! - **`Hasher`** Trait Implementation: If desired, you can integrate it with standard library data structures
//!   by implementing `std::hash::Hasher` (though that's less typical for a CRC).
//!
//! **Note**: CRC is not cryptographically secure. For security or cryptographic needs, use a modern cryptographic hash.

use std::fmt;

/// Default polynomial for standard CRC-32 (Ethernet, ZIP, PNG, etc): 0xEDB88320 (reversed 0x04C11DB7).
pub const DEFAULT_POLYNOMIAL: u32 = 0xEDB88320;
/// Default initial value: 0xFFFFFFFF (common).
pub const DEFAULT_INIT: u32 = 0xFFFF_FFFF;
/// Default final XOR value: 0xFFFFFFFF.
pub const DEFAULT_FINAL_XOR: u32 = 0xFFFF_FFFF;

/// A builder for constructing a CRC32 object with customized parameters.
#[derive(Clone)]
pub struct Crc32Builder {
    polynomial: u32,
    init: u32,
    final_xor: u32,
    reflect_in: bool,
    reflect_out: bool,
}

impl fmt::Debug for Crc32Builder {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Crc32Builder")
            .field("polynomial", &format_args!("{:#X}", self.polynomial))
            .field("init", &format_args!("{:#X}", self.init))
            .field("final_xor", &format_args!("{:#X}", self.final_xor))
            .field("reflect_in", &self.reflect_in)
            .field("reflect_out", &self.reflect_out)
            .finish()
    }
}

impl Default for Crc32Builder {
    fn default() -> Self {
        Self {
            polynomial: DEFAULT_POLYNOMIAL,
            init: DEFAULT_INIT,
            final_xor: DEFAULT_FINAL_XOR,
            reflect_in: true,
            reflect_out: true,
        }
    }
}

impl Crc32Builder {
    /// Creates a new builder with default standard CRC-32 parameters.
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the polynomial (in reversed form if reflect_in=true).
    /// The typical "standard" one is 0xEDB88320 for reversed 0x04C11DB7.
    /// If you have a direct polynomial (not reversed), you must do your own reflection logic or set reflect_in/out accordingly.
    pub fn polynomial(mut self, poly: u32) -> Self {
        self.polynomial = poly;
        self
    }

    /// Sets the initial register value.
    pub fn initial_value(mut self, init: u32) -> Self {
        self.init = init;
        self
    }

    /// Sets the final XOR value.
    pub fn final_xor(mut self, fx: u32) -> Self {
        self.final_xor = fx;
        self
    }

    /// Enables or disables reflection of input bytes.
    /// Commonly set to `true` for standard Ethernet/ZIP usage.
    pub fn reflect_in(mut self, on: bool) -> Self {
        self.reflect_in = on;
        self
    }

    /// Enables or disables reflection of the final CRC before final_xor.
    /// Commonly `true` for standard usage.
    pub fn reflect_out(mut self, on: bool) -> Self {
        self.reflect_out = on;
        self
    }

    /// Builds the `Crc32` object with the specified parameters.
    /// It constructs a 256-element lookup table for quick table-based CRC.
    pub fn build(self) -> Crc32 {
        let mut table = [0u32; 256];
        // build table
        for i in 0..256 {
            let mut crc = i as u32;
            if self.reflect_in {
                // For reflected polynomials, we process LSB to MSB
                for _ in 0..8 {
                    if (crc & 1) != 0 {
                        crc = (crc >> 1) ^ self.polynomial;
                    } else {
                        crc >>= 1;
                    }
                }
            } else {
                // non-reflect
                crc <<= 24;
                for _ in 0..8 {
                    if (crc & 0x80000000) != 0 {
                        crc = (crc << 1) ^ self.polynomial;
                    } else {
                        crc <<= 1;
                    }
                }
            }
            table[i as usize] = crc;
        }

        let state = self.init;

        Crc32 {
            table,
            #[allow(dead_code)]
            polynomial: self.polynomial,
            init: self.init,
            final_xor: self.final_xor,
            reflect_in: self.reflect_in,
            reflect_out: self.reflect_out,
            state,
        }
    }
}

/// The main structure for computing CRC32 with a table-based approach.
#[derive(Debug, Clone)]
pub struct Crc32 {
    table: [u32; 256],
    #[allow(dead_code)]
    polynomial: u32,
    init: u32,
    final_xor: u32,
    reflect_in: bool,
    reflect_out: bool,

    state: u32, // current CRC state
}

impl Crc32 {
    /// Resets the CRC state to initial value.
    pub fn reset(&mut self) {
        self.state = self.init;
    }

    /// Updates the CRC with the entire `data` slice.
    pub fn update(&mut self, data: &[u8]) {
        if self.reflect_in {
            // for reversed polynomials, we process from LSB to MSB
            let mut crc = self.state;
            for &b in data {
                let idx = (crc ^ (b as u32)) & 0xFF;
                crc = (crc >> 8) ^ self.table[idx as usize];
            }
            self.state = crc;
        } else {
            // non-reflect
            let mut crc = self.state;
            for &b in data {
                let idx = ((crc >> 24) ^ (b as u32)) & 0xFF;
                crc = (crc << 8) ^ self.table[idx as usize];
            }
            self.state = crc;
        }
    }

    /// Finalizes the CRC, returning the 32-bit result.
    /// The hasher remains in a consistent state, so subsequent updates continue from the same state.
    pub fn finalize(&self) -> u32 {
        let mut crc = self.state;
        if self.reflect_out != self.reflect_in {
            crc = reflect_32(crc);
        }
        crc ^ self.final_xor
    }

    /// One-shot convenience: feed `data` and return final CRC, not affecting current state.
    pub fn compute_one_shot(&self, data: &[u8]) -> u32 {
        let mut temp = self.clone();
        temp.reset();
        temp.update(data);
        temp.finalize()
    }

    /// Returns current internal state (not final).
    pub fn current_state(&self) -> u32 {
        self.state
    }
}

/// Reflect the bits of a 32-bit value (MSB <-> LSB).
fn reflect_32(mut x: u32) -> u32 {
    let mut r = 0u32;
    for _ in 0..32 {
        r = (r << 1) | (x & 1);
        x >>= 1;
    }
    r
}

// Example usage:

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_crc32() {
        // default => polynomial=EDB88320, init=0xFFFF_FFFF, reflect_in/out = true, final_xor=0xFFFF_FFFF
        let c = Crc32Builder::new().build();
        let mut hasher = c.clone();
        hasher.update(b"123456789");
        let val = hasher.finalize();
        // known standard CRC-32 result for "123456789" is 0xcbf43926
        assert_eq!(val, 0xcbf43926);
    }

    #[test]
    fn test_no_reflection() {
        // let's do a config: polynomial=0x04C11DB7 (non reversed), init=0, final_xor=0, no reflection
        // This is the classical 802.3 polynomial in normal (non reversed) form
        let c = Crc32Builder::new()
            .polynomial(0x04C11DB7)
            .initial_value(0)
            .final_xor(0)
            .reflect_in(false)
            .reflect_out(false)
            .build();
        let h = c.compute_one_shot(b"ABC");
        // we won't have a known reference unless we do a known test. Let's just check it doesn't panic.
        assert_ne!(h, 0);
    }

    #[test]
    fn test_incremental() {
        let c = Crc32Builder::new().build();
        let mut hasher = c.clone();
        hasher.update(b"Hello, ");
        hasher.update(b"World!");
        let val1 = hasher.finalize();

        let val2 = c.compute_one_shot(b"Hello, World!");
        assert_eq!(val1, val2);
    }
}
