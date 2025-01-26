//! DISCLAIMER: This library is a toy example of the MD5 (legacy) hash function in pure Rust.
//! It is *EXCLUSIVELY* for demonstration and educational purposes. Absolutely DO NOT use it
//! for real cryptographic or security-sensitive operations. It is broken and insecure.
//! If you need a secure hash, use a vetted, modern library (e.g. SHA-2 or SHA-3 from RustCrypto).

use core::convert::TryInto;

/// The size of the MD5 digest in bytes (128 bits = 16 bytes).
pub const MD5_OUTPUT_SIZE: usize = 16;

/// A toy MD5 context for demonstration. DO NOT use in production or for any security purpose.
#[derive(Debug, Clone)]
pub struct Md5 {
    /// State (A, B, C, D) each 32 bits.
    a: u32,
    b: u32,
    c: u32,
    d: u32,
    /// 64-byte block buffer
    buffer: [u8; 64],
    /// Current buffer length
    buffer_len: usize,
    /// Total message length in bits mod 2^64
    length_bits_low: u64,
}

/// The initial values for (A, B, C, D) from the MD5 specification.
static INIT_A: u32 = 0x67452301;
static INIT_B: u32 = 0xEFCDAB89;
static INIT_C: u32 = 0x98BADCFE;
static INIT_D: u32 = 0x10325476;

/// The sine table constants (K) in MD5 (32 bits). 
/// K[i] = floor(2^32 * abs(sin(i+1))) for i=0..63
static K: [u32; 64] = [
    0xd76aa478, 0xe8c7b756, 0x242070db, 0xc1bdceee,
    0xf57c0faf, 0x4787c62a, 0xa8304613, 0xfd469501,
    0x698098d8, 0x8b44f7af, 0xffff5bb1, 0x895cd7be,
    0x6b901122, 0xfd987193, 0xa679438e, 0x49b40821,
    0xf61e2562, 0xc040b340, 0x265e5a51, 0xe9b6c7aa,
    0xd62f105d, 0x02441453, 0xd8a1e681, 0xe7d3fbc8,
    0x21e1cde6, 0xc33707d6, 0xf4d50d87, 0x455a14ed,
    0xa9e3e905, 0xfcefa3f8, 0x676f02d9, 0x8d2a4c8a,
    0xfffa3942, 0x8771f681, 0x6d9d6122, 0xfde5380c,
    0xa4beea44, 0x4bdecfa9, 0xf6bb4b60, 0xbebfbc70,
    0x289b7ec6, 0xeaa127fa, 0xd4ef3085, 0x04881d05,
    0xd9d4d039, 0xe6db99e5, 0x1fa27cf8, 0xc4ac5665,
    0xf4292244, 0x432aff97, 0xab9423a7, 0xfc93a039,
    0x655b59c3, 0x8f0ccc92, 0xffeff47d, 0x85845dd1,
    0x6fa87e4f, 0xfe2ce6e0, 0xa3014314, 0x4e0811a1,
    0xf7537e82, 0xbd3af235, 0x2ad7d2bb, 0xeb86d391,
];

/// The amount of left rotation performed in each MD5 round, grouped by step.
static S: [u32; 64] = [
    // Round 1
    7, 12, 17, 22,  7, 12, 17, 22,  7, 12, 17, 22,  7, 12, 17, 22,
    // Round 2
    5, 9, 14, 20,   5, 9, 14, 20,   5, 9, 14, 20,   5, 9, 14, 20,
    // Round 3
    4, 11, 16, 23,  4, 11, 16, 23,  4, 11, 16, 23,  4, 11, 16, 23,
    // Round 4
    6, 10, 15, 21,  6, 10, 15, 21,  6, 10, 15, 21,  6, 10, 15, 21,
];

impl Md5 {
    /// Creates a new MD5 context.
    pub fn new() -> Self {
        Self {
            a: INIT_A,
            b: INIT_B,
            c: INIT_C,
            d: INIT_D,
            buffer: [0u8; 64],
            buffer_len: 0,
            length_bits_low: 0,
        }
    }

    /// Updates the MD5 context with data. 
    /// This is purely for demonstration and is not secure.
    pub fn update(&mut self, data: &[u8]) {
        for &byte in data {
            self.buffer[self.buffer_len] = byte;
            self.buffer_len += 1;
            self.length_bits_low = self.length_bits_low.wrapping_add(8);

            if self.buffer_len == 64 {
                self.process_block(&self.buffer);
                self.buffer_len = 0;
            }
        }
    }

    /// Finalizes the MD5 hash, returning 16 bytes. 
    /// Do not reuse this context after finalize.
    /// This is purely for demonstration.
    pub fn finalize(mut self) -> [u8; MD5_OUTPUT_SIZE] {
        // append the 0x80
        self.buffer[self.buffer_len] = 0x80;
        self.buffer_len += 1;

        // if there's not enough room for 8-byte length, process block
        if self.buffer_len > 56 {
            for i in self.buffer_len..64 {
                self.buffer[i] = 0;
            }
            self.process_block(&self.buffer);
            self.buffer_len = 0;
        }
        // fill zero until last 8 bytes
        for i in self.buffer_len..56 {
            self.buffer[i] = 0;
        }
        // append the original length in bits, little-endian
        let length_le = self.length_bits_low.to_le_bytes();
        self.buffer[56..64].copy_from_slice(&length_le);

        self.process_block(&self.buffer);

        // produce digest in little-endian
        let mut output = [0u8; MD5_OUTPUT_SIZE];
        output[0..4].copy_from_slice(&self.a.to_le_bytes());
        output[4..8].copy_from_slice(&self.b.to_le_bytes());
        output[8..12].copy_from_slice(&self.c.to_le_bytes());
        output[12..16].copy_from_slice(&self.d.to_le_bytes());
        output
    }

    /// Processes a 512-bit (64-byte) block, updating the internal state.
    /// The block is divided into 16 32-bit words in little-endian.
    fn process_block(&mut self, block: &[u8]) {
        let mut w = [0u32; 16];
        for i in 0..16 {
            w[i] = u32::from_le_bytes(block[4*i..4*i+4].try_into().unwrap());
        }

        let (mut a, mut b, mut c, mut d) = (self.a, self.b, self.c, self.d);

        for i in 0..64 {
            let (f, g) = if i < 16 {
                // F function
                ((b & c) | ((!b) & d), i)
            } else if i < 32 {
                // G function
                ((b & d) | (c & (!d)), (5*i + 1) % 16)
            } else if i < 48 {
                // H function
                (b ^ c ^ d, (3*i + 5) % 16)
            } else {
                // I function
                (c ^ (b | (!d)), (7*i) % 16)
            };

            let temp = a
                .wrapping_add(f)
                .wrapping_add(w[g as usize])
                .wrapping_add(K[i]);
            let temp = temp.rotate_left(S[i]) .wrapping_add(b);

            a = d;
            d = c;
            c = b;
            b = temp;
        }

        self.a = self.a.wrapping_add(a);
        self.b = self.b.wrapping_add(b);
        self.c = self.c.wrapping_add(c);
        self.d = self.d.wrapping_add(d);
    }
}

/// Convenience function to compute MD5 digest in a single shot.
/// *Do not use for real security.* 
pub fn md5_digest(data: &[u8]) -> [u8; MD5_OUTPUT_SIZE] {
    let mut hasher = Md5::new();
    hasher.update(data);
    hasher.finalize()
}

#[cfg(test)]
mod tests {
    use super::*;

    // Known test vectors from RFC 1321

    #[test]
    fn test_md5_empty() {
        // MD5("") => d41d8cd98f00b204e9800998ecf8427e
        let digest = md5_digest(b"");
        assert_eq!(hex::encode(digest), "d41d8cd98f00b204e9800998ecf8427e");
    }

    #[test]
    fn test_md5_abc() {
        // MD5("abc") => 900150983cd24fb0d6963f7d28e17f72
        let digest = md5_digest(b"abc");
        assert_eq!(hex::encode(digest), "900150983cd24fb0d6963f7d28e17f72");
    }

    #[test]
    fn test_md5_message_digest() {
        // MD5("message digest") => f96b697d7cb7938d525a2f31aaf161d0
        let digest = md5_digest(b"message digest");
        assert_eq!(hex::encode(digest), "f96b697d7cb7938d525a2f31aaf161d0");
    }
}
