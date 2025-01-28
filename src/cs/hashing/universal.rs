//! # Universal Hashing
//!
//! This library provides a **universal hash** construction in Rust, following a
//! polynomial-based (Carter-Wegman style) approach. The goal is to map arbitrary input data to
//! a numeric hash value in a manner that *minimizes collisions* across different keys, under
//! random choice of parameters. By "universal," we mean that for distinct inputs `x` and `y`,
//! the probability of `hash(x) == hash(y)` is small (assuming a random selection of hashing parameters).
//!
//! **Note**: This implementation is designed to be robust, flexible, and perform well in
//! specialized contexts. However, it is **not** cryptographically secure and is **not** intended
//! as a cryptographic hash or for security-critical use cases. Instead, it serves as a high-performance
//! universal hashing approach for tasks such as randomized data structures, fingerprinting,
//! or load balancing with near-uniform distribution.
//!
//! # Overview
//!
//! We implement a polynomial hash over a prime field of 64 bits (using a prime just under `2^61`
//! to avoid overflow in 128-bit intermediate multiplication). The general idea:
//!
//! For an input byte sequence `X = [x_0, x_1, ..., x_{n-1}]`, we interpret these bytes as a series
//! of small coefficients in a polynomial mod a large prime `P`. Then we compute:
//!
//! ```text
//!   H(X) = ( ( (a * x_0 + x_1 ) * a + x_2 ) * a + ... ) + b ) mod P
//! ```
//!
//! where `a` and `b` are random parameters chosen at initialization to ensure universal behavior.
//! This approach ensures that for distinct inputs, collisions occur with low probability (1/P),
//! so long as `a` and `b` are chosen from the prime field at random.
//!
//! We produce a final 64-bit hash. By default, we reduce mod `P`, a prime near 2^61. Because 2^61
//! is smaller than 2^64, we can do multiplications in 128 bits and reduce carefully to avoid overflow
//! in 64-bit arithmetic.
//!
//! # Usage
//!
//! ```rust
//! use algos::cs::hashing::universal::{UniversalHash64, UniversalHashBuilder};
//!
//! // Build a universal hash instance with random parameters (for robust collision distribution).
//! let hasher = UniversalHashBuilder::new().build_64();
//!
//! // Hash some data
//! let data = b"Hello, world!";
//! let hash_value = hasher.hash(data);
//! println!("Hash = {:#x}", hash_value);
//! ```
//!
//! # Features
//! - Random parameter generation for `a` and `b`, or user can specify them.
//! - A safe prime `P` near 2^61 for minimal collision probability with up to 64-bit outputs.
//! - Streaming mode: you can feed data incrementally (`update`) and then finalize (`finish`).
//! - Multi-threads usage: each hasher is independent. For concurrency, create multiple hashers.
//!
//! This code attempts to be efficient by accumulating partial polynomial expansions and using
//! 128-bit intermediate multiplications carefully. If you need extremely large input or extremely
//! high performance, consider more specialized solutions or hardware acceleration.
//!
//! Universal hashing implementation.
//!
//! Example using the streaming interface:
//! ```rust
//! use algos::cs::hashing::universal::{UniversalHash64, UniversalHash64State};
//!
//! let mut state = UniversalHash64State::new();
//! state.write(b"hello");
//! let hash = state.finish();
//! ```
//!
//! Example using the builder:
//! ```rust
//! use algos::cs::hashing::universal::{UniversalHash64, UniversalHashBuilder};
//!
//! let builder = UniversalHashBuilder::new();
//! let hasher = builder.build_64();
//! let hash = hasher.hash(b"hello");
//! ```

use rand::{rngs::StdRng, Rng, SeedableRng};

/// A prime just under `2^61`. We use 2^61 - 1 here for convenience (which is not prime),
/// so let's pick a known prime near 2^61. We'll use 2305843009213693951 = 2^61 - 1 (which is Mersenne),
/// but 2^61-1 is actually prime if 61 is prime. Indeed, 2^61 - 1 is "Mersenne prime" if 61 is prime.
/// This is sometimes called "M61".
///
/// We'll define it here. We assume it's prime for the polynomial hashing approach.
/// (2^61 - 1 is indeed prime.)
const PRIME_61: u64 = 0x1FFFFFFFFFFFFFFF; // 2^61 - 1

/// A 64-bit universal hash state for incremental hashing.
///
/// Example:
/// ```rust
/// use algos::cs::hashing::universal::UniversalHash64State;
///
/// let mut state = UniversalHash64State::new();
/// state.write(b"hello");
/// let hash = state.finish();
/// ```
pub struct UniversalHash64State {
    p: u64,
    a: u64,
    b: u64,
    partial: u64,
}

impl Default for UniversalHash64State {
    fn default() -> Self {
        Self::new()
    }
}

/// A builder for constructing universal hash functions.
///
/// Example:
/// ```rust
/// use algos::cs::hashing::universal::{UniversalHash64, UniversalHashBuilder};
///
/// let builder = UniversalHashBuilder::new();
/// let hasher = builder.build_64();
/// let hash = hasher.hash(b"hello");
/// ```
pub struct UniversalHashBuilder {
    seed: Option<u64>,
    a: Option<u64>,
    b: Option<u64>,
    p: u64, // prime modulus for the field, default 2^61-1
}

impl Default for UniversalHashBuilder {
    fn default() -> Self {
        Self {
            seed: None,
            a: None,
            b: None,
            p: PRIME_61,
        }
    }
}

impl UniversalHashBuilder {
    /// Creates a new builder with default prime (2^61-1) and random parameters if not overridden.
    pub fn new() -> Self {
        Self::default()
    }

    /// Use a custom prime `p`. Must be < 2^63.
    /// For typical usage, 2^61-1 is recommended.
    /// # Panics
    /// - if `p <= 1`.
    pub fn prime(mut self, p: u64) -> Self {
        assert!(p > 1, "prime must be > 1");
        self.p = p;
        self
    }

    /// Sets an explicit seed for random parameter generation.
    /// If not called, a random seed will be used from the OS RNG.
    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Provide your own parameters `a` and `b`, bypassing random generation.
    /// Must be in `[1..p-1]` for `a`, `[0..p]` for `b`.
    pub fn params(mut self, a: u64, b: u64) -> Self {
        self.a = Some(a);
        self.b = Some(b);
        self
    }

    /// Builds a universal hasher returning 64-bit results.
    pub fn build_64(self) -> UniversalHash64 {
        let p = self.p;
        // check a, b or generate
        let (a, b) = if let (Some(a), Some(b)) = (self.a, self.b) {
            // validate
            if a == 0 || a >= p {
                panic!("param a must be in [1..p-1].");
            }
            if b >= p {
                panic!("param b must be in [0..p-1].");
            }
            (a % p, b % p) // Ensure parameters are reduced mod p
        } else {
            // random
            let mut rng = if let Some(s) = self.seed {
                StdRng::seed_from_u64(s)
            } else {
                StdRng::from_entropy()
            };
            // generate a in [1..p-1], b in [0..p-1].
            let a = rng.gen_range(1..p);
            let b = rng.gen_range(0..p);
            (a, b)
        };
        UniversalHash64 { p, a, b }
    }
}

/// A polynomial-based universal hash function that produces 64-bit outputs
/// by reducing mod `p` (where `p < 2^63`).
#[derive(Debug, Clone)]
pub struct UniversalHash64 {
    p: u64,
    a: u64,
    b: u64,
}

impl UniversalHash64 {
    /// Hashes the entire `data` in one shot, returning a 64-bit result in `[0..p)`.
    pub fn hash(&self, data: &[u8]) -> u64 {
        let mut h = 0u64;
        for &byte in data {
            // h = (h * a + byte) mod p
            h = mul_mod(h, self.a, self.p);
            h = add_mod(h, byte as u64, self.p);
        }
        // add b and ensure final result is reduced mod p
        add_mod(h, self.b, self.p)
    }

    /// Creates a streaming stateful hasher for partial updates. This allows large data sets
    /// without building the entire slice in memory at once.
    pub fn hasher(&self) -> UniversalHash64State {
        UniversalHash64State {
            p: self.p,
            a: self.a,
            b: self.b,
            partial: 0,
        }
    }
}

impl UniversalHash64State {
    /// Creates a new hash state with default parameters.
    pub fn new() -> Self {
        Self {
            p: PRIME_61,
            a: 1, // Default multiplier
            b: 0, // Default offset
            partial: 0,
        }
    }

    /// Updates the hash state with the provided `data`.
    pub fn write(&mut self, data: &[u8]) {
        for &byte in data {
            self.partial = mul_mod(self.partial, self.a, self.p);
            self.partial = add_mod(self.partial, byte as u64, self.p);
        }
    }

    /// Finalizes the hash, returning a 64-bit result. This consumes the state.
    pub fn finish(self) -> u64 {
        // Ensure final result is reduced mod p
        add_mod(self.partial, self.b, self.p)
    }
}

// ---------- Low-level ops for mod p < 2^63 ---------- //

#[inline]
fn add_mod(x: u64, y: u64, p: u64) -> u64 {
    // First reduce inputs modulo p to handle large values (like ASCII codes)
    let x = x % p;
    let y = y % p;
    // Now do the modular addition
    let s = x.wrapping_add(y);
    if s >= p {
        s.wrapping_sub(p)
    } else {
        s
    }
}

/// Multiplication mod p, p<2^63, so we can do 128-bit intermediate with standard Rust
/// then reduce.
#[inline]
fn mul_mod(x: u64, y: u64, p: u64) -> u64 {
    // We'll do x*y in 128 bits, reduce mod p.
    // We rely on stable 128-bit ops in modern Rust.
    let x = x % p; // Ensure inputs are reduced
    let y = y % p;
    let prod = (x as u128) * (y as u128);
    (prod % (p as u128)) as u64
}

// For test usage
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_universal_hash() {
        let uh = UniversalHashBuilder::new().prime(31).seed(12345).build_64();
        // prime=31 => a, b in that field
        // We'll see if it doesn't panic
        let h1 = uh.hash(b"abcd");
        let h2 = uh.hash(b"abce");
        // Just ensure no panic, we don't check distribution here
        assert!(h1 < 31);
        assert!(h2 < 31);
        assert_ne!(
            h1, h2,
            "Likely different for small input, though not guaranteed"
        );
    }

    #[test]
    fn test_streaming_equiv() {
        let data = b"Hello, universal hashing test data!";
        let uh = UniversalHashBuilder::new().build_64();

        let direct = uh.hash(data);

        let mut st = uh.hasher();
        st.write(&data[..10]);
        st.write(&data[10..]);
        let streaming = st.finish();

        assert_eq!(direct, streaming, "Streaming must match one-shot result");
    }

    #[test]
    fn test_zero_data() {
        let uh = UniversalHashBuilder::new().build_64();
        let val = uh.hash(b"");
        // It's just b mod p. That can be anything from [0..p)
        // But let's ensure it doesn't panic
        println!("Hash of empty = {}", val);
    }
}
