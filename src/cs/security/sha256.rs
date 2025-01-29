//! DISCLAIMER: This library is a toy example of the SHA-256 hash function in pure Rust.
//! It is *EXCLUSIVELY* for demonstration and educational purposes. Absolutely DO NOT use it
//! for real cryptographic or security-sensitive operations. It is not audited, not vetted,
//! and very likely insecure in practice. If you need SHA-256 or any cryptographic operations
//! in production, please use a vetted, well-reviewed cryptography library.

use core::convert::TryInto;

/// The output size of SHA-256 in bytes (256 bits).
pub const SHA256_OUTPUT_SIZE: usize = 32;

/// A toy SHA-256 state: 8 working variables (h0..h7), plus how many bytes have been processed, etc.
/// This is purely for demonstration. DO NOT use for real security.
#[derive(Debug, Clone)]
pub struct Sha256 {
    /// The working state (8 32-bit words).
    h: [u32; 8],
    /// Unprocessed bytes buffer (block buffer).
    buffer: [u8; 64],
    /// How many bytes so far modulo 64.
    buffer_len: usize,
    /// Total length in bits mod 2^64 (since we track 64-bit length).
    length_bits_low: u64,
}

/// The initial constants (fractional part of the square roots of primes).
static H_INIT: [u32; 8] = [
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
];

/// The round constants (fractional part of the cube roots of primes).
static K: [u32; 64] = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
];

impl Default for Sha256 {
    fn default() -> Self {
        Self::new()
    }
}

impl Sha256 {
    /// Creates a new SHA-256 context (toy).
    pub fn new() -> Self {
        Self {
            h: H_INIT,
            buffer: [0u8; 64],
            buffer_len: 0,
            length_bits_low: 0,
        }
    }

    /// Updates the hash context with `data`.
    /// This code is purely for demonstration and not secure.
    pub fn update(&mut self, data: &[u8]) {
        for &b in data {
            self.buffer[self.buffer_len] = b;
            self.buffer_len += 1;
            self.length_bits_low = self.length_bits_low.wrapping_add(8);

            if self.buffer_len == 64 {
                let buffer_copy = self.buffer;
                self.process_block(&buffer_copy);
                self.buffer_len = 0;
            }
        }
    }

    /// Finalizes the hash, returning a 32-byte result.
    /// After calling `finalize()`, the context should not be reused for more data.
    /// This code is for demonstration only.
    pub fn finalize(mut self) -> [u8; SHA256_OUTPUT_SIZE] {
        // padding
        let bit_len = self.length_bits_low; // only low 64 bits

        // append 0x80
        self.buffer[self.buffer_len] = 0x80;
        self.buffer_len += 1;

        // if not enough room for length (8 bytes), process block
        if self.buffer_len > 56 {
            // fill remainder with zero
            for i in self.buffer_len..64 {
                self.buffer[i] = 0;
            }
            let buffer_copy = self.buffer;
            self.process_block(&buffer_copy);
            self.buffer_len = 0;
        }
        // fill zeros until the last 8 bytes for length
        for i in self.buffer_len..56 {
            self.buffer[i] = 0;
        }
        // write bit length in big-endian
        self.buffer[56..64].copy_from_slice(&bit_len.to_be_bytes());

        let buffer_copy = self.buffer;
        self.process_block(&buffer_copy);

        // produce digest
        let mut output = [0u8; SHA256_OUTPUT_SIZE];
        for (i, val) in self.h.iter().enumerate() {
            output[(4 * i)..(4 * i + 4)].copy_from_slice(&val.to_be_bytes());
        }
        output
    }

    /// Processes a 512-bit (64-byte) block, updating the internal state.
    /// This is the core compression function of SHA-256 (toy).
    fn process_block(&mut self, block: &[u8]) {
        // parse block into 16 big-endian 32-bit words
        let mut w = [0u32; 64];
        for i in 0..16 {
            w[i] = u32::from_be_bytes(block[4 * i..4 * i + 4].try_into().unwrap());
        }
        // message schedule
        for i in 16..64 {
            let s0 = small_sigma0(w[i - 15]);
            let s1 = small_sigma1(w[i - 2]);
            w[i] = w[i - 16]
                .wrapping_add(s0)
                .wrapping_add(w[i - 7])
                .wrapping_add(s1);
        }

        // working variables
        let mut a = self.h[0];
        let mut b = self.h[1];
        let mut c = self.h[2];
        let mut d = self.h[3];
        let mut e = self.h[4];
        let mut f = self.h[5];
        let mut g = self.h[6];
        let mut h = self.h[7];

        for i in 0..64 {
            let temp1 = h
                .wrapping_add(big_sigma1(e))
                .wrapping_add(ch(e, f, g))
                .wrapping_add(K[i])
                .wrapping_add(w[i]);
            let temp2 = big_sigma0(a).wrapping_add(maj(a, b, c));

            h = g;
            g = f;
            f = e;
            e = d.wrapping_add(temp1);
            d = c;
            c = b;
            b = a;
            a = temp1.wrapping_add(temp2);
        }

        self.h[0] = self.h[0].wrapping_add(a);
        self.h[1] = self.h[1].wrapping_add(b);
        self.h[2] = self.h[2].wrapping_add(c);
        self.h[3] = self.h[3].wrapping_add(d);
        self.h[4] = self.h[4].wrapping_add(e);
        self.h[5] = self.h[5].wrapping_add(f);
        self.h[6] = self.h[6].wrapping_add(g);
        self.h[7] = self.h[7].wrapping_add(h);
    }
}

// SHA-256 utility functions:

#[inline(always)]
fn ch(x: u32, y: u32, z: u32) -> u32 {
    (x & y) ^ ((!x) & z)
}

#[inline(always)]
fn maj(x: u32, y: u32, z: u32) -> u32 {
    (x & y) ^ (x & z) ^ (y & z)
}

#[inline(always)]
fn big_sigma0(x: u32) -> u32 {
    x.rotate_right(2) ^ x.rotate_right(13) ^ x.rotate_right(22)
}

#[inline(always)]
fn big_sigma1(x: u32) -> u32 {
    x.rotate_right(6) ^ x.rotate_right(11) ^ x.rotate_right(25)
}

#[inline(always)]
fn small_sigma0(x: u32) -> u32 {
    x.rotate_right(7) ^ x.rotate_right(18) ^ (x >> 3)
}

#[inline(always)]
fn small_sigma1(x: u32) -> u32 {
    x.rotate_right(17) ^ x.rotate_right(19) ^ (x >> 10)
}

// A convenience function to compute SHA-256 digest in one-shot usage.
pub fn sha256_digest(data: &[u8]) -> [u8; SHA256_OUTPUT_SIZE] {
    let mut hasher = Sha256::new();
    hasher.update(data);
    hasher.finalize()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sha256_empty() {
        // from the FIPS example:
        // SHA256("") => e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
        let hash = sha256_digest(b"");
        assert_eq!(
            hex::encode(hash),
            "e3b0c44298fc1c149afbf4c8996fb924\
             27ae41e4649b934ca495991b7852b855"
        );
    }

    #[test]
    fn test_sha256_abc() {
        // from FIPS:
        // SHA256("abc") =>
        // ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad
        let hash = sha256_digest(b"abc");
        assert_eq!(
            hex::encode(hash),
            "ba7816bf8f01cfea414140de5dae2223\
             b00361a396177a9cb410ff61f20015ad"
        );
    }
}
