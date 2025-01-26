//! DISCLAIMER: This library is a toy example of the AES (Rijndael) block cipher in pure Rust.
//! It is *EXCLUSIVELY* for demonstration and educational purposes. Absolutely DO NOT use it
//! for real cryptographic or security-sensitive operations. It is not audited, not vetted,
//! and very likely insecure in practice. If you need AES or any cryptographic operations in
//! production, please use a vetted, well-reviewed cryptography library (e.g. RustCrypto).

use core::convert::TryInto;

/// AES block size in bytes (128 bits).
pub const AES_BLOCK_SIZE: usize = 16;

/// Represents key sizes for AES: 128, 192, or 256 bits.
#[derive(Debug, Clone, Copy)]
pub enum AesKeySize {
    Bits128,
    Bits192,
    Bits256,
}

/// An AES key schedule object, storing the round keys after expansion.
pub struct AesKey {
    pub round_keys: Vec<[u8; AES_BLOCK_SIZE]>,
    pub nr: usize, // number of rounds
}

impl AesKey {
    /// Create a new AES key from a given key material (raw bytes) and key size variant.
    /// 
    /// # Panics
    /// Panics if the key material length doesn't match the indicated `AesKeySize`.
    ///
    /// DO NOT USE THIS FOR REAL SECURITY.
    pub fn new(key_data: &[u8], key_size: AesKeySize) -> Self {
        let (key_len, nr, nk) = match key_size {
            AesKeySize::Bits128 => (16, 10, 4),
            AesKeySize::Bits192 => (24, 12, 6),
            AesKeySize::Bits256 => (32, 14, 8),
        };
        assert_eq!(key_data.len(), key_len, "Key length mismatch for AES key size");

        let expanded_len = AES_BLOCK_SIZE * (nr + 1);
        let mut round_keys = vec![0u8; expanded_len];
        // copy initial key
        round_keys[..key_len].copy_from_slice(key_data);

        key_expansion(&mut round_keys, nk, nr);

        // Convert round_keys to round-key blocks
        let mut round_blocks = Vec::with_capacity(nr + 1);
        for i in 0..(nr + 1) {
            let offset = i * AES_BLOCK_SIZE;
            let block: [u8; AES_BLOCK_SIZE] = round_keys[offset..offset + AES_BLOCK_SIZE]
                .try_into()
                .unwrap();
            round_blocks.push(block);
        }

        Self {
            round_keys: round_blocks,
            nr,
        }
    }
}

/// Encrypt a single 128-bit block `plaintext` in place using the provided AES key schedule.
/// *This is toy code. DO NOT use in production.*
///
/// # Panics
/// Panics if `plaintext.len() != 16`.
pub fn aes_encrypt_block(plaintext: &mut [u8; AES_BLOCK_SIZE], key: &AesKey) {
    add_round_key(plaintext, &key.round_keys[0]);

    for round in 1..key.nr {
        sub_bytes(plaintext);
        shift_rows(plaintext);
        mix_columns(plaintext);
        add_round_key(plaintext, &key.round_keys[round]);
    }

    // final round
    sub_bytes(plaintext);
    shift_rows(plaintext);
    add_round_key(plaintext, &key.round_keys[key.nr]);
}

/// Decrypt a single 128-bit block `ciphertext` in place using the provided AES key schedule.
/// *This is toy code. DO NOT use in production.*
///
/// # Panics
/// Panics if `ciphertext.len() != 16`.
pub fn aes_decrypt_block(ciphertext: &mut [u8; AES_BLOCK_SIZE], key: &AesKey) {
    add_round_key(ciphertext, &key.round_keys[key.nr]);
    inv_shift_rows(ciphertext);
    inv_sub_bytes(ciphertext);

    for round in (1..key.nr).rev() {
        add_round_key(ciphertext, &key.round_keys[round]);
        inv_mix_columns(ciphertext);
        inv_shift_rows(ciphertext);
        inv_sub_bytes(ciphertext);
    }

    // final
    add_round_key(ciphertext, &key.round_keys[0]);
}

// ---------------- Internal Implementation Details (toy) ---------------- //
// The following code includes S-Boxes, inverse S-Boxes, Rcon constants,
// and standard AES transformations. DO NOT rely on for real usage.

/// S-Box for AES subBytes
static SBOX: [u8; 256] = [
    0x63,0x7c,0x77,0x7b,0xf2,0x6b,0x6f,0xc5,0x30,0x01,0x67,0x2b,0xfe,0xd7,0xab,0x76,
    0xca,0x82,0xc9,0x7d,0xfa,0x59,0x47,0xf0,0xad,0xd4,0xa2,0xaf,0x9c,0xa4,0x72,0xc0,
    0xb7,0xfd,0x93,0x26,0x36,0x3f,0xf7,0xcc,0x34,0xa5,0xe5,0xf1,0x71,0xd8,0x31,0x15,
    0x04,0xc7,0x23,0xc3,0x18,0x96,0x05,0x9a,0x07,0x12,0x80,0xe2,0xeb,0x27,0xb2,0x75,
    0x09,0x83,0x2c,0x1a,0x1b,0x6e,0x5a,0xa0,0x52,0x3b,0xd6,0xb3,0x29,0xe3,0x2f,0x84,
    0x53,0xd1,0x00,0xed,0x20,0xfc,0xb1,0x5b,0x6a,0xcb,0xbe,0x39,0x4a,0x4c,0x58,0xcf,
    0xd0,0xef,0xaa,0xfb,0x43,0x4d,0x33,0x85,0x45,0xf9,0x02,0x7f,0x50,0x3c,0x9f,0xa8,
    0x51,0xa3,0x40,0x8f,0x92,0x9d,0x38,0xf5,0xbc,0xb6,0xda,0x21,0x10,0xff,0xf3,0xd2,
    0xcd,0x0c,0x13,0xec,0x5f,0x97,0x44,0x17,0xc4,0xa7,0x7e,0x3d,0x64,0x5d,0x19,0x73,
    0x60,0x81,0x4f,0xdc,0x22,0x2a,0x90,0x88,0x46,0xee,0xb8,0x14,0xde,0x5e,0x0b,0xdb,
    0xe0,0x32,0x3a,0x0a,0x49,0x06,0x24,0x5c,0xc2,0xd3,0xac,0x62,0x91,0x95,0xe4,0x79,
    0xe7,0xc8,0x37,0x6d,0x8d,0xd5,0x4e,0xa9,0x6c,0x56,0xf4,0xea,0x65,0x7a,0xae,0x08,
    0xba,0x78,0x25,0x2e,0x1c,0xa6,0xb4,0xc6,0xe8,0xdd,0x74,0x1f,0x4b,0xbd,0x8b,0x8a,
    0x70,0x3e,0xb5,0x66,0x48,0x03,0xf6,0x0e,0x61,0x35,0x57,0xb9,0x86,0xc1,0x1d,0x9e,
    0xe1,0xf8,0x98,0x11,0x69,0xd9,0x8e,0x94,0x9b,0x1e,0x87,0xe9,0xce,0x55,0x28,0xdf,
    0x8c,0xa1,0x89,0x0d,0xbf,0xe6,0x42,0x68,0x41,0x99,0x2d,0x0f,0xb0,0x54,0xbb,0x16,
];

/// Inverse S-Box for AES invSubBytes
static INV_SBOX: [u8; 256] = [
    0x52,0x09,0x6a,0xd5,0x30,0x36,0xa5,0x38,0xbf,0x40,0xa3,0x9e,0x81,0xf3,0xd7,0xfb,
    0x7c,0xe3,0x39,0x82,0x9b,0x2f,0xff,0x87,0x34,0x8e,0x43,0x44,0xc4,0xde,0xe9,0xcb,
    0x54,0x7b,0x94,0x32,0xa6,0xc2,0x23,0x3d,0xee,0x4c,0x95,0x0b,0x42,0xfa,0xc3,0x4e,
    0x08,0x2e,0xa1,0x66,0x28,0xd9,0x24,0xb2,0x76,0x5b,0xa2,0x49,0x6d,0x8b,0xd1,0x25,
    0x72,0xf8,0xf6,0x64,0x86,0x68,0x98,0x16,0xd4,0xa4,0x5c,0xcc,0x5d,0x65,0xb6,0x92,
    0x6c,0x70,0x48,0x50,0xfd,0xed,0xb9,0xda,0x5e,0x15,0x46,0x57,0xa7,0x8d,0x9d,0x84,
    0x90,0xd8,0xab,0x00,0x8c,0xbc,0xd3,0x0a,0xf7,0xe4,0x58,0x05,0xb8,0xb3,0x45,0x06,
    0xd0,0x2c,0x1e,0x8f,0xca,0x3f,0x0f,0x02,0xc1,0xaf,0xbd,0x03,0x01,0x13,0x8a,0x6b,
    0x3a,0x91,0x11,0x41,0x4f,0x67,0xdc,0xea,0x97,0xf2,0xcf,0xce,0xf0,0xb4,0xe6,0x73,
    0x96,0xac,0x74,0x22,0xe7,0xad,0x35,0x85,0xe2,0xf9,0x37,0xe8,0x1c,0x75,0xdf,0x6e,
    0x47,0xf1,0x1a,0x71,0x1d,0x29,0xc5,0x89,0x6f,0xb7,0x62,0x0e,0xaa,0x18,0xbe,0x1b,
    0xfc,0x56,0x3e,0x4b,0xc6,0xd2,0x79,0x20,0x9a,0xdb,0xc0,0xfe,0x78,0xcd,0x5a,0xf4,
    0x1f,0xdd,0xa8,0x33,0x88,0x07,0xc7,0x31,0xb1,0x12,0x10,0x59,0x27,0x80,0xec,0x5f,
    0x60,0x51,0x7f,0xa9,0x19,0xb5,0x4a,0x0d,0x2d,0xe5,0x7a,0x9f,0x93,0xc9,0x9c,0xef,
    0xa0,0xe0,0x3b,0x4d,0xae,0x2a,0xf5,0xb0,0xc8,0xeb,0xbb,0x3c,0x83,0x53,0x99,0x61,
    0x17,0x2b,0x04,0x7e,0xba,0x77,0xd6,0x26,0xe1,0x69,0x14,0x63,0x55,0x21,0x0c,0x7d,
];

/// Round constant for key expansion
static RCON: [u8; 255] = [
    0x00, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x1B, 0x36,
    // not all used, but let's keep them
    0x6C, 0xD8, 0xAB, 0x4D, 0x9A, 0x2F, 0x5E, 0xBC, 0x63, 0xC6, 0x97, 0x35, 0x6A, 0xD4,
    0xB3, 0x7D, 0xFA, 0xEF, 0xC5, 0x91, 0x39, 0x72, 0xE4, 0x9B, 0x2D, 0x5A, 0xB4, 0x7B,
    0xF6, 0xED, 0xC1, 0x81, 0x19, 0x32, 0x64, 0xC8, 0x8B, 0x01, 0x02, 0x04, /* ... */
    // only the first 11 are strictly needed for 128-bit AES. 
    // For 256-bit, we go further. This array extends to 255 for completeness in toy code.
    // We'll not fill them all out for brevity in this toy. 
    // This is enough for 14 rounds (AES-256). 
    0x08,0x10,0x20,0x40,0x80,0x1B,0x36,0x6C,0xD8,0xAB,0x4D,0x9A,0x2F,0x5E,0xBC,0x63,
    // etc...
];

fn sub_bytes(state: &mut [u8; AES_BLOCK_SIZE]) {
    for b in state.iter_mut() {
        *b = SBOX[*b as usize];
    }
}

fn inv_sub_bytes(state: &mut [u8; AES_BLOCK_SIZE]) {
    for b in state.iter_mut() {
        *b = INV_SBOX[*b as usize];
    }
}

fn shift_rows(state: &mut [u8; AES_BLOCK_SIZE]) {
    // row 1 shift by 1
    let row1 = [state[1], state[5], state[9], state[13]];
    state[1]  = row1[1];
    state[5]  = row1[2];
    state[9]  = row1[3];
    state[13] = row1[0];

    // row 2 shift by 2
    let row2 = [state[2], state[6], state[10], state[14]];
    state[2]  = row2[2];
    state[6]  = row2[3];
    state[10] = row2[0];
    state[14] = row2[1];

    // row 3 shift by 3
    let row3 = [state[3], state[7], state[11], state[15]];
    state[3]  = row3[3];
    state[7]  = row3[0];
    state[11] = row3[1];
    state[15] = row3[2];
}

fn inv_shift_rows(state: &mut [u8; AES_BLOCK_SIZE]) {
    // row 1 shift right by 1
    let row1 = [state[1], state[5], state[9], state[13]];
    state[1]  = row1[3];
    state[5]  = row1[0];
    state[9]  = row1[1];
    state[13] = row1[2];

    // row 2 shift right by 2
    let row2 = [state[2], state[6], state[10], state[14]];
    state[2]  = row2[2];
    state[6]  = row2[3];
    state[10] = row2[0];
    state[14] = row2[1];

    // row 3 shift right by 3
    let row3 = [state[3], state[7], state[11], state[15]];
    state[3]  = row3[1];
    state[7]  = row3[2];
    state[11] = row3[3];
    state[15] = row3[0];
}

fn xtime(x: u8) -> u8 {
    if (x & 0x80) != 0 {
        (x << 1) ^ 0x1B
    } else {
        x << 1
    }
}

fn mix_columns(state: &mut [u8; AES_BLOCK_SIZE]) {
    for col in 0..4 {
        let base = col * 4;
        let t = state[base] ^ state[base + 1] ^ state[base + 2] ^ state[base + 3];
        let temp0 = state[base];
        let temp1 = state[base + 1];
        let temp2 = state[base + 2];
        let temp3 = state[base + 3];

        state[base]     ^= t ^ xtime(temp0 ^ temp1);
        state[base + 1] ^= t ^ xtime(temp1 ^ temp2);
        state[base + 2] ^= t ^ xtime(temp2 ^ temp3);
        state[base + 3] ^= t ^ xtime(temp3 ^ temp0);
    }
}

fn inv_mix_columns(state: &mut [u8; AES_BLOCK_SIZE]) {
    // The standard approach is to multiply the state columns by the inverse of the MDS matrix
    // We'll do it in the typical inline approach:
    for col in 0..4 {
        let base = col * 4;
        let a0 = state[base];
        let a1 = state[base + 1];
        let a2 = state[base + 2];
        let a3 = state[base + 3];

        state[base]     = mul(a0, 0x0e) ^ mul(a1, 0x0b) ^ mul(a2, 0x0d) ^ mul(a3, 0x09);
        state[base + 1] = mul(a0, 0x09) ^ mul(a1, 0x0e) ^ mul(a2, 0x0b) ^ mul(a3, 0x0d);
        state[base + 2] = mul(a0, 0x0d) ^ mul(a1, 0x09) ^ mul(a2, 0x0e) ^ mul(a3, 0x0b);
        state[base + 3] = mul(a0, 0x0b) ^ mul(a1, 0x0d) ^ mul(a2, 0x09) ^ mul(a3, 0x0e);
    }
}

fn mul(x: u8, y: u8) -> u8 {
    // Galois Field (2^8) multiplication
    let mut r = 0;
    let mut a = x;
    let mut b = y;
    for _ in 0..8 {
        if (b & 1) == 1 {
            r ^= a;
        }
        let hi_bit_set = (a & 0x80) != 0;
        a <<= 1;
        if hi_bit_set {
            a ^= 0x1b;
        }
        b >>= 1;
    }
    r
}

fn add_round_key(state: &mut [u8; AES_BLOCK_SIZE], round_key: &[u8; AES_BLOCK_SIZE]) {
    for (s, k) in state.iter_mut().zip(round_key) {
        *s ^= *k;
    }
}

// Key Expansion routines
fn key_expansion(expanded: &mut [u8], nk: usize, nr: usize) {
    let key_size_bytes = nk * 4;
    let total_words = (nr + 1) * 4; // number of 32-bit words
    let mut i = nk;
    while i < total_words {
        let mut temp = [
            expanded[(i - 1) * 4],
            expanded[(i - 1) * 4 + 1],
            expanded[(i - 1) * 4 + 2],
            expanded[(i - 1) * 4 + 3],
        ];

        if i % nk == 0 {
            // rotate
            temp = [temp[1], temp[2], temp[3], temp[0]];
            // sub
            for t in temp.iter_mut() {
                *t = SBOX[*t as usize];
            }
            // rcon
            temp[0] ^= RCON[i / nk];
        } else if nk > 6 && i % nk == 4 {
            for t in temp.iter_mut() {
                *t = SBOX[*t as usize];
            }
        }

        let wprev = (i - nk) * 4;
        for (j, tj) in temp.iter().enumerate() {
            expanded[i * 4 + j] = expanded[wprev + j] ^ tj;
        }
        i += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Example test using NIST known test vectors for AES-128 single block
    // "Fips-197" example: key=2b7e151628aed2a6abf7158809cf4f3c, plaintext=6bc1bee22e409f96e93d7e117393172a
    // ciphertext=3ad77bb40d7a3660a89ecaf32466ef97
    #[test]
    fn test_aes128_encrypt_block() {
        let key_data = hex_to_bytes("2b7e151628aed2a6abf7158809cf4f3c");
        let mut block = hex_to_array("6bc1bee22e409f96e93d7e117393172a");

        let aes_key = AesKey::new(&key_data, AesKeySize::Bits128);

        aes_encrypt_block(&mut block, &aes_key);

        let expected = hex_to_array("3ad77bb40d7a3660a89ecaf32466ef97");
        assert_eq!(block, expected);
    }

    #[test]
    fn test_aes128_decrypt_block() {
        let key_data = hex_to_bytes("2b7e151628aed2a6abf7158809cf4f3c");
        let mut block = hex_to_array("3ad77bb40d7a3660a89ecaf32466ef97");

        let aes_key = AesKey::new(&key_data, AesKeySize::Bits128);

        aes_decrypt_block(&mut block, &aes_key);

        let expected = hex_to_array("6bc1bee22e409f96e93d7e117393172a");
        assert_eq!(block, expected);
    }

    // Helpers
    fn hex_to_bytes(s: &str) -> Vec<u8> {
        (0..s.len())
            .step_by(2)
            .map(|i| u8::from_str_radix(&s[i..i+2], 16).unwrap())
            .collect()
    }

    fn hex_to_array(s: &str) -> [u8; 16] {
        let bytes = hex_to_bytes(s);
        bytes.try_into().unwrap()
    }
}
