//! # FNV Hash Implementation
//!
//! This module provides a **production-ready** FNV (Fowler–Noll–Vo) hash algorithm in Rust,
//! using modern coding patterns. FNV is a simple, fast non-cryptographic hash function
//! commonly used for hash-based lookups and data structures, especially where collisions
//! aren't a major concern or are well-handled by the container itself.
//!
//! **Note**: FNV is not cryptographically secure. If you need a secure hash, use a modern
//! cryptographic algorithm (e.g., SHA-2 or BLAKE3).
//!
//! ## Key Features
//! - **64-bit** version (`FNV-1` and `FNV-1a` variants) by default. Can optionally configure 32-bit if desired.
//! - **Builder** pattern to configure whether we use FNV-1 or FNV-1a, and 32 or 64 bits.  
//! - **High performance** insertion and retrieval in typical hash-based data structures.  
//! - **Implements** the standard `std::hash::Hasher` trait so it can be used as a drop-in with `BuildHasher` for `HashMap` or other structures.

use std::hash::{BuildHasher, Hasher};

/// Default offset basis and prime for 64-bit FNV-1 or FNV-1a.
const FNV64_OFFSET_BASIS: u64 = 0xcbf29ce484222325;
const FNV64_PRIME: u64 = 0x100000001b3;

/// Default offset basis and prime for 32-bit FNV-1 or FNV-1a.
const FNV32_OFFSET_BASIS: u32 = 0x811c9dc5;
const FNV32_PRIME: u32 = 16777619;

/// Which FNV variant: FNV1 or FNV1a.
#[derive(Debug, Clone, Copy)]
pub enum FnvVariant {
    /// FNV-1
    Fnv1,
    /// FNV-1a
    Fnv1a,
}

/// Which bit-size we use for FNV hashing: 32-bit or 64-bit.
#[derive(Debug, Clone, Copy)]
pub enum FnvBits {
    /// 32-bit hashing
    Fnv32,
    /// 64-bit hashing
    Fnv64,
}

/// A builder for the FNV hash, letting you configure the variant (Fnv1 or Fnv1a) and bit size (32, 64).
#[derive(Debug, Clone)]
pub struct FnvBuilder {
    variant: FnvVariant,
    bits: FnvBits,
}

impl Default for FnvBuilder {
    fn default() -> Self {
        // default to FNV-1a 64-bit
        Self {
            variant: FnvVariant::Fnv1a,
            bits: FnvBits::Fnv64,
        }
    }
}

impl FnvBuilder {
    /// Create a new builder with default (FNV-1a, 64-bit).
    pub fn new() -> Self {
        Default::default()
    }

    /// Sets the variant: FNV-1 or FNV-1a.
    pub fn variant(mut self, variant: FnvVariant) -> Self {
        self.variant = variant;
        self
    }

    /// Sets bit size: 32 or 64.
    pub fn bits(mut self, bits: FnvBits) -> Self {
        self.bits = bits;
        self
    }

    /// Builds a `FnvBuildHasher` that can produce `FnvHasher` objects implementing `std::hash::Hasher`.
    pub fn build(self) -> FnvBuildHasher {
        FnvBuildHasher {
            variant: self.variant,
            bits: self.bits,
        }
    }
}

/// The `BuildHasher` that can create `FnvHasher` objects for usage in `HashMap`, etc.
#[derive(Debug, Clone)]
pub struct FnvBuildHasher {
    variant: FnvVariant,
    bits: FnvBits,
}

impl BuildHasher for FnvBuildHasher {
    type Hasher = FnvHasher;

    fn build_hasher(&self) -> Self::Hasher {
        let (state64, state32) = match self.bits {
            FnvBits::Fnv64 => (Some(FnvState64::new(self.variant)), None),
            FnvBits::Fnv32 => (None, Some(FnvState32::new(self.variant))),
        };
        FnvHasher {
            variant: self.variant,
            bits: self.bits,
            state64,
            state32,
        }
    }
}

/// The main hasher that implements `std::hash::Hasher`.
#[derive(Debug, Clone)]
pub struct FnvHasher {
    variant: FnvVariant,
    bits: FnvBits,
    state64: Option<FnvState64>,
    state32: Option<FnvState32>,
}

impl Hasher for FnvHasher {
    fn finish(&self) -> u64 {
        match self.bits {
            FnvBits::Fnv64 => self.state64.unwrap().finish(),
            FnvBits::Fnv32 => self.state32.unwrap().finish() as u64,
        }
    }

    fn write(&mut self, bytes: &[u8]) {
        match self.bits {
            FnvBits::Fnv64 => {
                let st = self.state64.as_mut().unwrap();
                match self.variant {
                    FnvVariant::Fnv1 => st.update_fnv1(bytes),
                    FnvVariant::Fnv1a => st.update_fnv1a(bytes),
                }
            }
            FnvBits::Fnv32 => {
                let st = self.state32.as_mut().unwrap();
                match self.variant {
                    FnvVariant::Fnv1 => st.update_fnv1(bytes),
                    FnvVariant::Fnv1a => st.update_fnv1a(bytes),
                }
            }
        }
    }
}

/// Internal state for 64-bit FNV hashing.
#[derive(Debug, Clone)]
struct FnvState64 {
    state: u64,
    variant: FnvVariant,
}

impl FnvState64 {
    fn new(variant: FnvVariant) -> Self {
        Self {
            state: FNV64_OFFSET_BASIS,
            variant,
        }
    }

    fn finish(&self) -> u64 {
        self.state
    }

    fn update_fnv1(&mut self, data: &[u8]) {
        for &b in data {
            // FNV-1: state = state * prime, then state = state ^ b
            self.state = self.state.wrapping_mul(FNV64_PRIME);
            self.state ^= b as u64;
        }
    }

    fn update_fnv1a(&mut self, data: &[u8]) {
        for &b in data {
            // FNV-1a: state = state ^ b, then state = state * prime
            self.state ^= b as u64;
            self.state = self.state.wrapping_mul(FNV64_PRIME);
        }
    }
}

/// Internal state for 32-bit FNV hashing.
#[derive(Debug, Clone)]
struct FnvState32 {
    state: u32,
    variant: FnvVariant,
}

impl FnvState32 {
    fn new(variant: FnvVariant) -> Self {
        Self {
            state: FNV32_OFFSET_BASIS,
            variant,
        }
    }

    fn finish(&self) -> u32 {
        self.state
    }

    fn update_fnv1(&mut self, data: &[u8]) {
        for &b in data {
            self.state = self.state.wrapping_mul(FNV32_PRIME);
            self.state ^= b as u32;
        }
    }

    fn update_fnv1a(&mut self, data: &[u8]) {
        for &b in data {
            self.state ^= b as u32;
            self.state = self.state.wrapping_mul(FNV32_PRIME);
        }
    }
}

// --- Public convenience functions for typical usage ---

/// Returns a 64-bit FNV-1a hash of `data`.
pub fn fnv64a_hash(data: &[u8]) -> u64 {
    let mut st = FnvState64::new(FnvVariant::Fnv1a);
    st.update_fnv1a(data);
    st.finish()
}

/// Returns a 64-bit FNV-1 hash of `data`.
pub fn fnv64_hash(data: &[u8]) -> u64 {
    let mut st = FnvState64::new(FnvVariant::Fnv1);
    st.update_fnv1(data);
    st.finish()
}

/// Returns a 32-bit FNV-1a hash of `data`.
pub fn fnv32a_hash(data: &[u8]) -> u32 {
    let mut st = FnvState32::new(FnvVariant::Fnv1a);
    st.update_fnv1a(data);
    st.finish()
}

/// Returns a 32-bit FNV-1 hash of `data`.
pub fn fnv32_hash(data: &[u8]) -> u32 {
    let mut st = FnvState32::new(FnvVariant::Fnv1);
    st.update_fnv1(data);
    st.finish()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fnv64a_hash() {
        let h1 = fnv64a_hash(b"hello");
        let h2 = fnv64a_hash(b"hello");
        assert_eq!(h1, h2);
        assert_ne!(h1, 0);

        let h3 = fnv64a_hash(b"Hello");
        assert_ne!(h1, h3);
    }

    #[test]
    fn test_fnv32a_hash() {
        let h1 = fnv32a_hash(b"world");
        let h2 = fnv32a_hash(b"world");
        assert_eq!(h1, h2);
        assert_ne!(h1, 0);

        let h3 = fnv32a_hash(b"WORLD");
        assert_ne!(h1, h3);
    }

    #[test]
    fn test_builder_usage() {
        // e.g. 32-bit FNV-1
        let hasher_builder = FnvBuilder::new()
            .variant(FnvVariant::Fnv1)
            .bits(FnvBits::Fnv32)
            .build();

        let mut hasher = hasher_builder.build_hasher();
        hasher.write(b"abc");
        let val = hasher.finish();
        // let's do the direct function
        let direct = fnv32_hash(b"abc") as u64;
        assert_eq!(val, direct);
    }
}
