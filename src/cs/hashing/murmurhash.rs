//! # MurmurHash Implementation
//!
//! This module provides a **production-ready** implementation of **MurmurHash3** in Rust,
//! including both **32-bit (x86_32)** and **128-bit (x64_128)** variants. MurmurHash is a
//! non-cryptographic, high-performance hash function commonly used in many data structures
//! (like hash tables) and data processing pipelines where collision resistance beyond
//! typical use is not the primary concern.
//!
//! **Note**: MurmurHash is **not** cryptographically secure. If you need a security-grade hash,
//! consider a cryptographic algorithm like SHA-2 or BLAKE3.
//!
//! ## Key Features
//! - **Builder** pattern to choose between Murmur3 x86_32 or x64_128 variants.
//! - **Streaming** usage via a Hasher-like object (partial updates, then finalize).
//! - **One-shot** convenience functions for direct hashing of a byte slice.
//! - **Implements** `std::hash::Hasher` to integrate with Rust's standard `HashMap`, etc.
//!
//! # Usage Example
//! ```rust
//! use murmurhash::{MurmurBuilder, MurmurVariant};
//! use std::collections::HashMap;
//!
//! // Build a 128-bit x64 variant with a custom seed
//! let build_hasher = MurmurBuilder::new()
//!     .variant(MurmurVariant::Murmur3x64_128)
//!     .seed(12345)
//!     .build();
//!
//! // Use in a HashMap
//! let mut map = HashMap::with_hasher(build_hasher);
//! map.insert("hello", 42);
//! assert_eq!(map.get("hello"), Some(&42));
//! ```

use std::hash::{BuildHasher, Hasher};

/// Which variant of MurmurHash to use.
#[derive(Debug, Clone, Copy)]
pub enum MurmurVariant {
    /// Murmur3 x86_32 => 32-bit result
    Murmur3x86_32,
    /// Murmur3 x64_128 => 128-bit result (we store in a 128-bit state).
    Murmur3x64_128,
}

/// A builder for MurmurHash, letting you pick the variant and seed.
#[derive(Debug, Clone)]
pub struct MurmurBuilder {
    variant: MurmurVariant,
    seed: u32,
}

impl Default for MurmurBuilder {
    fn default() -> Self {
        Self {
            variant: MurmurVariant::Murmur3x86_32,
            seed: 0,
        }
    }
}

impl MurmurBuilder {
    /// Creates a new builder with default variant (Murmur3 x86_32) and seed=0.
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the variant: x86_32 or x64_128.
    pub fn variant(mut self, variant: MurmurVariant) -> Self {
        self.variant = variant;
        self
    }

    /// Sets the seed (default=0).
    pub fn seed(mut self, seed: u32) -> Self {
        self.seed = seed;
        self
    }

    /// Finalizes the builder, producing a `MurmurBuildHasher` that can create `MurmurHasher`.
    pub fn build(self) -> MurmurBuildHasher {
        MurmurBuildHasher {
            variant: self.variant,
            seed: self.seed,
        }
    }
}

/// A `BuildHasher` implementing standard library usage for MurmurHash.
/// Typically used in a `HashMap` or similar.
#[derive(Debug, Clone)]
pub struct MurmurBuildHasher {
    variant: MurmurVariant,
    seed: u32,
}

impl BuildHasher for MurmurBuildHasher {
    type Hasher = MurmurHasher;

    fn build_hasher(&self) -> Self::Hasher {
        MurmurHasher::new(self.variant, self.seed)
    }
}

/// The MurmurHasher implementing `std::hash::Hasher`.
///
/// This can represent either a Murmur3 x86_32 or a x64_128 state.
/// We'll store it in an internal enum or structure.
#[derive(Clone)]
pub struct MurmurHasher {
    variant: MurmurVariant,
    x86_32_state: Option<Murmur3x86_32>,
    x64_128_state: Option<Murmur3x64_128>,
}

impl std::fmt::Debug for MurmurHasher {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MurmurHasher")
            .field("variant", &self.variant)
            .finish()
    }
}

impl MurmurHasher {
    /// Creates a new `MurmurHasher` with the specified variant and seed.
    pub fn new(variant: MurmurVariant, seed: u32) -> Self {
        match variant {
            MurmurVariant::Murmur3x86_32 => Self {
                variant,
                x86_32_state: Some(Murmur3x86_32::new(seed)),
                x64_128_state: None,
            },
            MurmurVariant::Murmur3x64_128 => Self {
                variant,
                x86_32_state: None,
                x64_128_state: Some(Murmur3x64_128::new(seed as u64)),
            },
        }
    }
}

impl Hasher for MurmurHasher {
    fn write(&mut self, bytes: &[u8]) {
        match self.variant {
            MurmurVariant::Murmur3x86_32 => {
                self.x86_32_state.as_mut().unwrap().update(bytes);
            }
            MurmurVariant::Murmur3x64_128 => {
                self.x64_128_state.as_mut().unwrap().update(bytes);
            }
        }
    }

    fn finish(&self) -> u64 {
        match self.variant {
            MurmurVariant::Murmur3x86_32 => {
                let mut s = self.x86_32_state.clone().unwrap();
                let h32 = s.finish();
                // zero-extend to 64
                h32 as u64
            }
            MurmurVariant::Murmur3x64_128 => {
                let mut s = self.x64_128_state.clone().unwrap();
                let (low, _high) = s.finish128();
                // We'll return low 64 bits for Hasher usage
                low
            }
        }
    }
}

// ------- Implementation of Murmur3 x86_32 -------- //
#[derive(Debug, Clone)]
struct Murmur3x86_32 {
    h: u32,
    length: usize,
    // partial buffer leftover or we can store partial count, etc.
    buffer: Vec<u8>,
}

impl Murmur3x86_32 {
    fn new(seed: u32) -> Self {
        Self {
            h: seed,
            length: 0,
            buffer: Vec::new(),
        }
    }

    fn update(&mut self, data: &[u8]) {
        // We can do a block approach (4 bytes at a time). The partial leftover stored in `buffer`.
        self.buffer.extend_from_slice(data);
        let mut offset = 0;
        while self.buffer.len() - offset >= 4 {
            let block = &self.buffer[offset..offset + 4];
            let k = u32::from_le_bytes([block[0], block[1], block[2], block[3]]);
            self.length += 4;
            offset += 4;

            self.h = murmur3_x86_32_mix_k(self.h, k);
        }
        // remove processed part
        if offset > 0 {
            self.buffer.drain(0..offset);
        }
    }

    fn finish(mut self) -> u32 {
        let leftover = self.buffer.len();
        let mut k = 0u32;
        if leftover > 0 {
            let mut tail = [0u8; 4];
            tail[..leftover].copy_from_slice(&self.buffer[..leftover]);
            k = u32::from_le_bytes(tail);
            self.length += leftover;
            self.h = murmur3_x86_32_mix_k_partial(self.h, k, leftover as u32);
        }
        // final
        self.h ^= self.length as u32;
        self.h = fmix32(self.h);
        self.h
    }
}

#[inline]
fn murmur3_x86_32_mix_k(mut h: u32, mut k: u32) -> u32 {
    const C1: u32 = 0xcc9e2d51;
    const C2: u32 = 0x1b873593;

    k = k.wrapping_mul(C1);
    k = k.rotate_left(15);
    k = k.wrapping_mul(C2);

    h ^= k;
    h = h.rotate_left(13);
    h = h.wrapping_mul(5).wrapping_add(0xe6546b64);
    h
}

// partial block logic
#[inline]
fn murmur3_x86_32_mix_k_partial(mut h: u32, mut k: u32, size: u32) -> u32 {
    const C1: u32 = 0xcc9e2d51;
    const C2: u32 = 0x1b873593;

    // rotate, multiply depending on leftover
    k = match size {
        1 => {
            k = k & 0xff;
            k = k.wrapping_mul(C1);
            k = k.rotate_left(15);
            k = k.wrapping_mul(C2);
            k
        }
        2 => {
            k = k & 0xffff;
            k = k.wrapping_mul(C1);
            k = k.rotate_left(15);
            k = k.wrapping_mul(C2);
            k
        }
        3 => {
            k = k & 0xffffff;
            k = k.wrapping_mul(C1);
            k = k.rotate_left(15);
            k = k.wrapping_mul(C2);
            k
        }
        _ => k,
    };

    h ^= k;
    h
}

#[inline]
fn fmix32(mut h: u32) -> u32 {
    h ^= h >> 16;
    h = h.wrapping_mul(0x85ebca6b);
    h ^= h >> 13;
    h = h.wrapping_mul(0xc2b2ae35);
    h ^= h >> 16;
    h
}

// ------- Implementation of Murmur3 x64_128 ------- //
#[derive(Debug, Clone)]
struct Murmur3x64_128 {
    h1: u64,
    h2: u64,
    length: usize,
    buffer: Vec<u8>,
}

impl Murmur3x64_128 {
    fn new(seed: u64) -> Self {
        Self {
            h1: seed,
            h2: seed,
            length: 0,
            buffer: Vec::new(),
        }
    }

    fn update(&mut self, data: &[u8]) {
        self.buffer.extend_from_slice(data);
        let mut offset = 0;
        while self.buffer.len() - offset >= 16 {
            let block = &self.buffer[offset..offset + 16];
            let k1 = u64::from_le_bytes(block[0..8].try_into().unwrap());
            let k2 = u64::from_le_bytes(block[8..16].try_into().unwrap());
            offset += 16;
            self.length += 16;
            let (h1, h2) = murmur3_x64_128_mix_block(self.h1, self.h2, k1, k2);
            self.h1 = h1;
            self.h2 = h2;
        }
        if offset > 0 {
            self.buffer.drain(0..offset);
        }
    }

    fn finish128(mut self) -> (u64, u64) {
        let leftover = self.buffer.len();
        let (k1, k2) = murmur3_x64_128_handle_tail(&self.buffer, leftover);

        let mut h1 = self.h1;
        let mut h2 = self.h2;

        h1 ^= k1;
        h2 ^= k2;

        self.length += leftover;
        // finalize
        h1 ^= self.length as u64;
        h2 ^= self.length as u64;

        (h1, h2) = fmix128(h1, h2);
        (h1, h2)
    }
}

#[inline]
fn murmur3_x64_128_handle_tail(tail: &[u8], leftover: usize) -> (u64, u64) {
    let mut k1 = 0u64;
    let mut k2 = 0u64;
    if leftover > 8 {
        k2 = read_tail(&tail[8..], leftover - 8);
        k2 = k2.wrapping_mul(C2_128);
        k2 = k2.rotate_left(33);
        k2 = k2.wrapping_mul(C1_128);
    }
    if leftover > 0 {
        k1 = read_tail(tail, leftover.min(8));
        k1 = k1.wrapping_mul(C1_128);
        k1 = k1.rotate_left(31);
        k1 = k1.wrapping_mul(C2_128);
    }
    (k1, k2)
}

#[inline]
fn read_tail(data: &[u8], len: usize) -> u64 {
    let mut buf = [0u8; 8];
    buf[..len].copy_from_slice(&data[..len]);
    u64::from_le_bytes(buf)
}

const C1_128: u64 = 0x87c37b91114253d5;
const C2_128: u64 = 0x4cf5ad432745937f;

#[inline]
fn murmur3_x64_128_mix_block(mut h1: u64, mut h2: u64, mut k1: u64, mut k2: u64) -> (u64, u64) {
    // mix k1
    k1 = k1.wrapping_mul(C1_128);
    k1 = k1.rotate_left(31);
    k1 = k1.wrapping_mul(C2_128);
    h1 ^= k1;
    h1 = h1.rotate_left(27).wrapping_add(h2);
    h1 = h1.wrapping_mul(5).wrapping_add(0x52dce729);

    // mix k2
    k2 = k2.wrapping_mul(C2_128);
    k2 = k2.rotate_left(33);
    k2 = k2.wrapping_mul(C1_128);
    h2 ^= k2;
    h2 = h2.rotate_left(31).wrapping_add(h1);
    h2 = h2.wrapping_mul(5).wrapping_add(0x38495ab5);

    (h1, h2)
}

#[inline]
fn fmix128(mut h1: u64, mut h2: u64) -> (u64, u64) {
    h1 ^= h2;
    h2 ^= h1;

    h1 = fmix64(h1);
    h2 = fmix64(h2);

    h1 ^= h2;
    h2 ^= h1;
    (h1, h2)
}

#[inline]
fn fmix64(mut k: u64) -> u64 {
    k ^= k >> 33;
    k = k.wrapping_mul(0xff51afd7ed558ccd);
    k ^= k >> 33;
    k = k.wrapping_mul(0xc4ceb9fe1a85ec53);
    k ^= k >> 33;
    k
}

// -- One-shot convenience for x86_32 variant
pub fn murmur3_x86_32(data: &[u8], seed: u32) -> u32 {
    let mut s = Murmur3x86_32::new(seed);
    s.update(data);
    s.finish()
}

// -- One-shot convenience for x64_128 variant => returns (low64, high64)
pub fn murmur3_x64_128(data: &[u8], seed: u64) -> (u64, u64) {
    let mut s = Murmur3x64_128::new(seed);
    s.update(data);
    s.finish128()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_murmur3_x86_32() {
        let h = murmur3_x86_32(b"hello", 0);
        let h2 = murmur3_x86_32(b"hello", 0);
        assert_eq!(h, h2);
        assert_ne!(h, 0);

        let diff = murmur3_x86_32(b"Hello", 0);
        assert_ne!(h, diff);
    }

    #[test]
    fn test_murmur3_x64_128() {
        let (low, high) = murmur3_x64_128(b"hello", 0);
        let (low2, high2) = murmur3_x64_128(b"hello", 0);
        assert_eq!(low, low2);
        assert_eq!(high, high2);
        assert!(!(low == 0 && high == 0));

        let (low3, high3) = murmur3_x64_128(b"Hello", 0);
        assert!((low3 != low) || (high3 != high));
    }

    #[test]
    fn test_hasher_in_map() {
        use std::collections::HashMap;

        let buildh = MurmurBuilder::new()
            .variant(MurmurVariant::Murmur3x64_128)
            .seed(1234)
            .build();
        let mut map = HashMap::with_hasher(buildh);
        map.insert("foo", 1);
        map.insert("bar", 2);
        assert_eq!(map["foo"], 1);
    }
}
