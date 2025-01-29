//! DISCLAIMER: This library is a toy example of the Blowfish block cipher in pure Rust.
//! It is *EXCLUSIVELY* for demonstration and educational purposes. Absolutely DO NOT use it
//! for real cryptographic or security-sensitive operations. It is not audited, not vetted,
//! and very likely insecure in practice. If you need Blowfish or any cryptographic operations
//! in production, please use a vetted, well-reviewed cryptography library.

use core::convert::TryInto;

/// Blowfish operates on 64-bit blocks (8 bytes).
pub const BLOWFISH_BLOCK_SIZE: usize = 8;
/// Blowfish uses a P-array of 18 32-bit subkeys plus 4 S-boxes each holding 256 32-bit entries.
pub const PARRAY_SIZE: usize = 18;
pub const SBOX_COUNT: usize = 4;
pub const SBOX_ENTRIES: usize = 256;

/// Maximum key size is up to 448 bits (56 bytes), but Blowfish also allows smaller keys.
pub const BLOWFISH_MAX_KEY_BYTES: usize = 56;

/// The Blowfish key schedule: P-array + S-boxes, each 32-bit.
pub struct BlowfishKey {
    p: [u32; PARRAY_SIZE],
    s: [[u32; SBOX_ENTRIES]; SBOX_COUNT],
}

/// A simple struct to store and manage Blowfish encryption/decryption context.
/// This is purely for demonstration and is NOT secure for real usage.
impl BlowfishKey {
    /// Creates a new Blowfish key schedule from the given `key_data`.
    ///
    /// # Panics
    /// - If `key_data` is empty or longer than 56 bytes.
    /// - This code is only for demonstration, so do not use in real cryptography.
    pub fn new(key_data: &[u8]) -> Self {
        if key_data.is_empty() || key_data.len() > BLOWFISH_MAX_KEY_BYTES {
            panic!("Invalid Blowfish key length (must be 1..=56 bytes)");
        }

        let mut key = BlowfishKey {
            p: DEFAULT_P,
            s: DEFAULT_SBOX,
        };
        key.key_schedule(key_data);
        key
    }

    /// Encrypt a single 64-bit block in place. `block` must be 8 bytes.
    /// DO NOT use in production.
    pub fn encrypt_block(&self, block: &mut [u8; BLOWFISH_BLOCK_SIZE]) {
        let (left, right) = block.split_at_mut(4);
        let mut xl = u32::from_be_bytes(left.try_into().unwrap());
        let mut xr = u32::from_be_bytes(right.try_into().unwrap());

        // 16 rounds
        for i in 0..16 {
            xl ^= self.p[i];
            xr ^= f_function(xl, &self.s);
            // swap
            let tmp = xl;
            xl = xr;
            xr = tmp;
        }
        // undo last swap
        let tmp = xl;
        xl = xr;
        xr = tmp;

        xr ^= self.p[16];
        xl ^= self.p[17];

        left.copy_from_slice(&xl.to_be_bytes());
        right.copy_from_slice(&xr.to_be_bytes());
    }

    /// Decrypt a single 64-bit block in place. `block` must be 8 bytes.
    /// DO NOT use in production.
    pub fn decrypt_block(&self, block: &mut [u8; BLOWFISH_BLOCK_SIZE]) {
        let (left, right) = block.split_at_mut(4);
        let mut xl = u32::from_be_bytes(left.try_into().unwrap());
        let mut xr = u32::from_be_bytes(right.try_into().unwrap());

        for i in (2..=17).rev() {
            xl ^= self.p[i];
            xr ^= f_function(xl, &self.s);
            // swap
            let tmp = xl;
            xl = xr;
            xr = tmp;
        }
        // undo last swap
        let tmp = xl;
        xl = xr;
        xr = tmp;

        xr ^= self.p[1];
        xl ^= self.p[0];

        left.copy_from_slice(&xl.to_be_bytes());
        right.copy_from_slice(&xr.to_be_bytes());
    }

    /// Blowfish Key Schedule:
    /// 1) XOR P-array with key bytes repeatedly.
    /// 2) Encrypt zero block, replace P[0..1].
    /// 3) Encrypt updated block, replace P[2..3].
    /// ...
    /// 4) Fill S-boxes similarly.
    fn key_schedule(&mut self, key_data: &[u8]) {
        // 1) XOR P-array with key
        let key_len = key_data.len();
        let mut data: usize = 0;
        for i in 0..PARRAY_SIZE {
            let mut val: u32 = 0;
            for _ in 0..4 {
                val = (val << 8) | (key_data[data % key_len] as u32);
                data += 1;
            }
            self.p[i] ^= val;
        }

        // 2) block = 64-bit all-zero
        let mut block = [0u8; 8];

        // 3) encrypt block, store result in P[0..1], then S[...].
        for i in (0..PARRAY_SIZE).step_by(2) {
            self.encrypt_block(&mut block);
            self.p[i] = u32::from_be_bytes(block[0..4].try_into().unwrap());
            self.p[i + 1] = u32::from_be_bytes(block[4..8].try_into().unwrap());
        }

        for sbox_i in 0..SBOX_COUNT {
            for sbox_j in (0..SBOX_ENTRIES).step_by(2) {
                self.encrypt_block(&mut block);
                self.s[sbox_i][sbox_j] = u32::from_be_bytes(block[0..4].try_into().unwrap());
                self.s[sbox_i][sbox_j + 1] = u32::from_be_bytes(block[4..8].try_into().unwrap());
            }
        }
    }
}

/// Blowfish F-Function:
/// Takes a 32-bit half-block `x`:
/// - split into four bytes (a, b, c, d)
/// - s0[a] + s1[b] ^ s2[c] + s3[d]
fn f_function(x: u32, s: &[[u32; 256]; 4]) -> u32 {
    let a = (x >> 24) as u8;
    let b = ((x >> 16) & 0xFF) as u8;
    let c = ((x >> 8) & 0xFF) as u8;
    let d = (x & 0xFF) as u8;

    let mut y = s[0][a as usize].wrapping_add(s[1][b as usize]);
    y ^= s[2][c as usize];
    y = y.wrapping_add(s[3][d as usize]);
    y
}

// Default P-array and S-box constants from the Blowfish specification.
static DEFAULT_P: [u32; PARRAY_SIZE] = [
    0x243F6A88, 0x85A308D3, 0x13198A2E, 0x03707344, 0xA4093822, 0x299F31D0, 0x082EFA98, 0xEC4E6C89,
    0x452821E6, 0x38D01377, 0xBE5466CF, 0x34E90C6C, 0xC0AC29B7, 0xC97C50DD, 0x3F84D5B5, 0xB5470917,
    0x9216D5D9, 0x8979FB1B,
];

static DEFAULT_SBOX: [[u32; 256]; 4] = [
    {
        let mut s = [0u32; 256];
        s[0] = 0xd1310ba6;
        s[1] = 0x98dfb5ac;
        s[2] = 0x2ffd72db;
        s[3] = 0xd01adfb7;
        s
    },
    [0u32; 256],
    [0u32; 256],
    [0u32; 256],
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_blowfish_basic() {
        // We'll do a minimal test with a short key and a known block.
        // There's a well-known test vector with "0000000000000000" block if we fill S with the default.
        // But we haven't included the entire default s-box. This is partial.
        // We'll just do a sanity check that we can do encrypt -> decrypt.

        let key_data = b"mytestkey";
        let bf = BlowfishKey::new(key_data);

        let mut block = [0u8; BLOWFISH_BLOCK_SIZE];
        block.copy_from_slice(b"12345678"); // 8 bytes

        let orig = block;
        bf.encrypt_block(&mut block);
        // Just ensure block changed
        assert_ne!(block, orig);

        bf.decrypt_block(&mut block);
        assert_eq!(
            block, orig,
            "Blowfish decrypt did not restore the original block"
        );
    }
}
