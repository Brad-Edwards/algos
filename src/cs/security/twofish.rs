//! DISCLAIMER: This library is a toy example of the Twofish block cipher in pure Rust.
//! It is *EXCLUSIVELY* for demonstration and educational purposes. Absolutely DO NOT use it
//! for real cryptographic or security-sensitive operations. It is not audited, not vetted,
//! and very likely insecure in practice. If you need Twofish or any cryptographic operations
//! in production, please use a vetted, well-reviewed cryptography library.

use core::convert::TryInto;

/// Twofish block size is 128 bits (16 bytes).
pub const TWOFISH_BLOCK_SIZE: usize = 16;

/// Twofish supports keys of 128, 192, or 256 bits. We'll represent them by length in bytes (16, 24, 32).
#[derive(Debug, Clone, Copy)]
pub enum TwofishKeySize {
    Bits128,
    Bits192,
    Bits256,
}

/// The maximum number of subkeys for Twofish:
/// - We generate 40 subkeys (each 32 bits) => 40 * 4 = 160 bytes total subkey space.
pub const TWOFISH_SUBKEY_COUNT: usize = 40;
/// MDS matrix dimension for polynomial multiplication in GF(256).
pub const MDS_DIM: usize = 4;

/// Toy structure holding the Twofish subkeys and internal key-dependent S-Boxes.
#[derive(Debug)]
pub struct TwofishKey {
    /// The expanded subkeys K[0..39], each is 4 bytes => total 40 * 4 = 160 bytes.
    pub subkeys: [u32; TWOFISH_SUBKEY_COUNT],
    /// The s-box keys used for the g() function, typically 4 words if 128-bit key, else 8 words, etc.
    /// We'll store enough for up to 8 in a simplified manner.
    sbox_keys: [u32; 8],
    /// The actual number of sbox_keys used depends on key size (kwords).
    kwords: usize,
}

/// A toy Twofish block encryption/decryption context.
/// Absolutely do not use for real security.
impl TwofishKey {
    /// Create a new TwofishKey from the raw key bytes and key size info.
    ///
    /// # Panics
    /// - If `key_data.len()` doesn't match the declared `TwofishKeySize`.
    /// - This code is purely for demonstration, so definitely not secure.
    pub fn new(key_data: &[u8], key_size: TwofishKeySize) -> Self {
        let (expected_len, kwords) = match key_size {
            TwofishKeySize::Bits128 => (16, 4),
            TwofishKeySize::Bits192 => (24, 6),
            TwofishKeySize::Bits256 => (32, 8),
        };
        if key_data.len() != expected_len {
            panic!("Invalid key length for TwofishKeySize");
        }

        // Initialize subkeys and sbox keys to zero
        let mut key = TwofishKey {
            subkeys: [0u32; TWOFISH_SUBKEY_COUNT],
            sbox_keys: [0u32; 8],
            kwords,
        };

        // Convert key bytes to words
        let mut key_words = vec![0u32; kwords];
        for i in 0..kwords {
            // each word is 4 bytes in big-endian or little-endian?
            // Twofish nominally uses little-endian for word assembly.
            let offset = i * 4;
            let word_le = u32::from_le_bytes(key_data[offset..offset + 4].try_into().unwrap());
            key_words[i] = word_le;
        }

        // Expand the key using the official Twofish approach (simplified here).
        key.twofish_key_schedule(&key_words);

        key
    }

    /// Encrypt a single 128-bit block in place.
    /// DO NOT use in production.
    pub fn encrypt_block(&self, block: &mut [u8; TWOFISH_BLOCK_SIZE]) {
        // parse input into four 32-bit words
        let mut x = [
            u32::from_le_bytes(block[0..4].try_into().unwrap()),
            u32::from_le_bytes(block[4..8].try_into().unwrap()),
            u32::from_le_bytes(block[8..12].try_into().unwrap()),
            u32::from_le_bytes(block[12..16].try_into().unwrap()),
        ];

        // input whitening
        x[0] ^= self.subkeys[0];
        x[1] ^= self.subkeys[1];
        x[2] ^= self.subkeys[2];
        x[3] ^= self.subkeys[3];

        // 16 rounds
        for r in 0..16 {
            let t0 = self.g_function(x[0], 0);
            let t1 = self.g_function(rol(x[1], 8), 1);

            let f0 = (t0.wrapping_add(t1)).wrapping_add(self.subkeys[8 + 2 * r]);
            let f1 = (t0.wrapping_add(2 * t1)).wrapping_add(self.subkeys[9 + 2 * r]);

            // apply f0, f1 to x[2], x[3]
            x[2] = rol(x[2] ^ f0, 1);
            x[3] = ror(x[3], 1) ^ f1;

            // rotate the block words left
            if r != 15 {
                let tmp = x[0];
                x[0] = x[2];
                x[2] = tmp;
                let tmp2 = x[1];
                x[1] = x[3];
                x[3] = tmp2;
            }
        }

        // output whitening
        x[2] ^= self.subkeys[4];
        x[3] ^= self.subkeys[5];
        x[0] ^= self.subkeys[6];
        x[1] ^= self.subkeys[7];

        // pack back
        block[0..4].copy_from_slice(&x[2].to_le_bytes());
        block[4..8].copy_from_slice(&x[3].to_le_bytes());
        block[8..12].copy_from_slice(&x[0].to_le_bytes());
        block[12..16].copy_from_slice(&x[1].to_le_bytes());
    }

    /// Decrypt a single 128-bit block in place.
    /// DO NOT use in production.
    pub fn decrypt_block(&self, block: &mut [u8; TWOFISH_BLOCK_SIZE]) {
        // parse input
        let mut x = [
            u32::from_le_bytes(block[0..4].try_into().unwrap()),
            u32::from_le_bytes(block[4..8].try_into().unwrap()),
            u32::from_le_bytes(block[8..12].try_into().unwrap()),
            u32::from_le_bytes(block[12..16].try_into().unwrap()),
        ];

        // undo output whitening
        let t0 = x[0];
        x[0] = x[2] ^ self.subkeys[4];
        x[2] = t0 ^ self.subkeys[6];
        let t1 = x[1];
        x[1] = x[3] ^ self.subkeys[5];
        x[3] = t1 ^ self.subkeys[7];

        // 16 rounds in reverse
        for r in (0..16).rev() {
            // unrotate?
            if r != 15 {
                let tmp = x[0];
                x[0] = x[2];
                x[2] = tmp;
                let tmp2 = x[1];
                x[1] = x[3];
                x[3] = tmp2;
            }

            let t0 = self.g_function(x[0], 0);
            let t1 = self.g_function(rol(x[1], 8), 1);

            let f0 = (t0.wrapping_add(t1)).wrapping_add(self.subkeys[8 + 2 * r]);
            let f1 = (t0.wrapping_add(2 * t1)).wrapping_add(self.subkeys[9 + 2 * r]);

            x[2] = ror(x[2], 1) ^ f0;
            x[3] = rol(x[3] ^ f1, 1);
        }

        // undo input whitening
        x[0] ^= self.subkeys[0];
        x[1] ^= self.subkeys[1];
        x[2] ^= self.subkeys[2];
        x[3] ^= self.subkeys[3];

        // pack back
        block[0..4].copy_from_slice(&x[0].to_le_bytes());
        block[4..8].copy_from_slice(&x[1].to_le_bytes());
        block[8..12].copy_from_slice(&x[2].to_le_bytes());
        block[12..16].copy_from_slice(&x[3].to_le_bytes());
    }

    // The g-function uses the key-dependent s-box, MDS matrix.
    // We'll do a partial approach here (toy).
    fn g_function(&self, x: u32, start: usize) -> u32 {
        // We'll do a simplified "h" function from Twofish specs, using the s-box keys, etc.
        // Real Twofish uses 4 bytes mapped through a key-based s-box, then MDS multiply.
        let mut b = [
            (x & 0xFF) as u8,
            ((x >> 8) & 0xFF) as u8,
            ((x >> 16) & 0xFF) as u8,
            ((x >> 24) & 0xFF) as u8,
        ];
        // apply key-based s-box step (toy)
        for i in 0..b.len() {
            b[i] = mds_q0q1(b[i]) ^ ((self.sbox_keys[start] as u8) & 0xFF);
        }
        // then MDS multiply (toy)
        let out = apply_mds(&b);
        out
    }

    /// The core key schedule for subkeys, sbox keys, etc.
    fn twofish_key_schedule(&mut self, key_words: &[u32]) {
        // We won't implement the full official approach in detail (just enough to show the structure).
        // Real Twofish does: generate sbox_keys from mekey, generate subkeys from moKey, etc.
        // We'll produce minimal plausible subkeys:

        // let's fill subkeys[0..8] as "input/output whitening"
        // subkeys[8..] as "round subkeys" for 16 rounds => total 32 round subkeys
        // 8 + 32 = 40 => subkeys

        for i in 0..self.kwords {
            self.sbox_keys[i] = key_words[i].rotate_left(3 * (i as u32)); // toy manip
        }

        for i in 0..TWOFISH_SUBKEY_COUNT {
            // toy expansion
            self.subkeys[i] = i as u32 ^ 0x9E3779B9; // example: some golden ratio constant
                                                     // in real code: compute subkey using polynomial-based approach or "RS" matrix, etc.
        }
        // add something from key
        for (i, &w) in key_words.iter().enumerate() {
            self.subkeys[i] ^= w;
        }
    }
}

// Toy MDS multiply with a single function. Real code uses a big matrix or references
fn apply_mds(b: &[u8; 4]) -> u32 {
    // This is a placeholder. Real MDS is a 4x4 matrix over GF(256).
    // We'll do some toy GF manipulation:
    let mut result = 0u32;
    for i in 0..4 {
        result ^= (b[i] as u32) << (8 * i);
    }
    result
}

// Toy Q-box function for demonstration
fn mds_q0q1(x: u8) -> u8 {
    // In real Twofish, there's a Q0, Q1 permutations. We'll just do a nibble swap or something.
    // This is purely to show structure, not correct for real Twofish.
    let hi = x >> 4;
    let lo = x & 0xF;
    (lo << 4) | hi
}

// rotate left and rotate right
fn rol(x: u32, n: u32) -> u32 {
    x.rotate_left(n)
}
fn ror(x: u32, n: u32) -> u32 {
    x.rotate_right(n)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_twofish_toy() {
        // We'll do a minimal test with a 128-bit key, a block of 16 zero bytes, just to see it "moves".
        let key_data = [0xAAu8; 16];
        let tf = TwofishKey::new(&key_data, TwofishKeySize::Bits128);

        let mut block = [0u8; TWOFISH_BLOCK_SIZE];
        let orig = block;
        tf.encrypt_block(&mut block);
        // check it changed
        assert_ne!(block, orig, "Twofish encrypt did not change the block");

        tf.decrypt_block(&mut block);
        assert_eq!(
            block, orig,
            "Twofish decrypt did not restore the original block"
        );
    }
}
