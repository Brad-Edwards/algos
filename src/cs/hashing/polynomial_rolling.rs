//! # Polynomial Rolling Hash
//!
//! This module implements a **polynomial rolling hash** for strings or byte slices in Rust,
//! suitable for use in **production** code bases that rely on a lightweight, fast, and
//! straightforward rolling hash. It's commonly used in substring search (e.g., Rabin-Karp),
//! string fingerprinting, or building various advanced data structures (e.g., rolling checksums).
//!
//! ## Key Features
//! - **Configurable base** (multiplier) and **modulus** to control collision probabilities and range.
//! - **64-bit** or **user-defined** modulus (less than 2^63 if using 128-bit intermediate arithmetic).
//! - **Streaming** usage: you can feed data incrementally (`update`) and retrieve the rolling hash
//!   via `current_hash`. You can also remove front data if you keep the sequence or track powers (optional).
//! - **Builder pattern** for easy parameter specification.  
//!
//! **Note**: This is **not** cryptographically secure. It's a polynomial rolling hash for typical
//! substring search or fingerprinting. For security-critical usage, rely on cryptographic hash
//! functions.

use std::ops::Mul;

/// Default base (multiplier) for polynomial rolling, e.g. ~257 for ASCII or 131542391 for distribution
const DEFAULT_BASE: u64 = 257;
/// Default modulus if none specified, e.g. a prime near 2^61 or 2^63. We'll pick 2^61-1 (a Mersenne prime).
const DEFAULT_MODULUS: u64 = 0x1FFFFFFFFFFFFFFF; // 2^61 - 1

/// A builder for polynomial rolling hash, allowing you to set base, modulus, etc.
#[derive(Debug, Clone)]
pub struct PolyHashBuilder {
    base: u64,
    modulus: u64,
}

impl Default for PolyHashBuilder {
    fn default() -> Self {
        Self {
            base: DEFAULT_BASE,
            modulus: DEFAULT_MODULUS,
        }
    }
}

impl PolyHashBuilder {
    /// Creates a new builder with default base/modulus.
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the base (the multiplier used in the polynomial).
    /// Typically near the size of the character set or bigger.
    pub fn with_base(mut self, base: u64) -> Self {
        assert!(base > 1, "base must be > 1");
        self.base = base;
        self
    }

    /// Sets the modulus for the polynomial.
    /// Must be < 2^63 to ensure we can do 128-bit intermediate safely if needed.
    /// Typically a large prime for fewer collisions.
    pub fn with_modulus(mut self, modulus: u64) -> Self {
        assert!(modulus > 1, "modulus must be > 1");
        self.modulus = modulus;
        self
    }

    /// Build the polynomial rolling hasher with the specified parameters.
    pub fn build(self) -> PolynomialRollingHash {
        PolynomialRollingHash {
            base: self.base,
            modulus: self.modulus,
            current_hash: 0,
            current_len: 0,
            current_power: 1, // base^0 = 1
        }
    }
}

/// A polynomial rolling hash struct that can be updated incrementally and returns a 64-bit hash in `[0..modulus)`.
#[derive(Debug, Clone)]
pub struct PolynomialRollingHash {
    base: u64,
    modulus: u64,
    /// The current polynomial hash value
    current_hash: u64,
    /// The number of "items" (e.g. characters) hashed so far.
    current_len: usize,
    /// The current power = base^(current_len) mod modulus, used if we want to remove from front (optional).
    current_power: u64,
}

impl PolynomialRollingHash {
    /// Creates a new polynomial rolling hash with default base/modulus.
    pub fn new() -> Self {
        PolyHashBuilder::new().build()
    }

    /// Resets the hasher state to empty.
    pub fn clear(&mut self) {
        self.current_hash = 0;
        self.current_len = 0;
        self.current_power = 1;
    }

    /// Returns the current hash value mod `modulus`.
    pub fn current_hash(&self) -> u64 {
        self.current_hash
    }

    /// Feeds the entire data slice in one shot. This is a convenience method.
    /// After calling, the `current_hash()` is updated for these bytes.
    pub fn hash_slice(&mut self, data: &[u8]) {
        for &b in data {
            self.update(b as u64);
        }
    }

    /// Streaming update with a single item (e.g. a byte or small integer).
    /// This increments the length by 1, multiplies the old hash by `base`, adds the new item, mod `modulus`.
    pub fn update(&mut self, x: u64) {
        let new_hash = mul_mod(self.current_hash, self.base, self.modulus);
        self.current_hash = add_mod(new_hash, x, self.modulus);

        // increment current_len
        self.current_len += 1;
        // update current_power = current_power * base mod modulus
        self.current_power = mul_mod(self.current_power, self.base, self.modulus);
    }

    /// If you want to remove the "oldest" item from the front (like a rolling window),
    /// you must pass the same `x` that was added first in the sequence.
    /// Then we do:
    /// `h' = h - x*base^(len-1), mod modulus`, and reduce length by 1.
    ///
    /// # Panics
    /// - if `self.current_len == 0`.
    pub fn remove_front(&mut self, x: u64) {
        assert!(self.current_len > 0, "No items to remove");
        // remove x*(base^(len-1)) from current_hash
        // we have current_power = base^(len)
        // so base^(len-1) = current_power / base => we can do a modular inverse, or keep a separate structure of powers?

        // We'll do an approach:
        // base^(len-1) = current_power / base mod => we do a modular inverse of base => potentially expensive.
        // or keep a dynamic array of powers. But for production usage, let's do a mod_inv approach or keep a separate approach.

        // We'll do the mod_inv approach for demonstration.
        // For large usage, you might keep an array or a rolling index.
        let inv_base = mod_inv(self.base, self.modulus)
            .expect("base is not invertible mod modulus (shouldn't happen if prime modulus and base < modulus).");

        let exponent = mul_mod(self.current_power, inv_base, self.modulus);
        // exponent = base^(len-1)

        // delta = x * exponent
        let delta = mul_mod(x, exponent, self.modulus);

        // new_hash = (current_hash - delta) mod
        self.current_hash = sub_mod(self.current_hash, delta, self.modulus);

        self.current_len -= 1;
        // current_power = current_power / base => multiply by inv_base
        self.current_power = mul_mod(self.current_power, inv_base, self.modulus);
    }
}

// internal ops

#[inline]
fn add_mod(a: u64, b: u64, m: u64) -> u64 {
    let s = a.wrapping_add(b);
    let res = if s >= m { s - m } else { s };
    res
}

#[inline]
fn sub_mod(a: u64, b: u64, m: u64) -> u64 {
    if a >= b {
        a - b
    } else {
        (a + m) - b
    }
}

#[inline]
fn mul_mod(a: u64, b: u64, m: u64) -> u64 {
    // do 128-bit multiplication, reduce mod m
    // must ensure m < 2^63
    let res = (a as u128).wrapping_mul(b as u128) % (m as u128);
    res as u64
}

/// Compute modular inverse of `x` mod `m` using extended Euclid. Panics if gcd != 1.
fn mod_inv(x: u64, m: u64) -> Option<u64> {
    // extended gcd approach
    // We do it in 128 bits for safety, or big ints. We'll do a small approach with i128.
    let (g, s, _) = extended_gcd(x as i128, m as i128);
    if g != 1 {
        return None;
    }
    let mm = m as i128;
    let inv = ((s % mm) + mm) % mm;
    Some(inv as u64)
}

fn extended_gcd(mut a: i128, mut b: i128) -> (i128, i128, i128) {
    let (mut x0, mut x1) = (1i128, 0i128);
    let (mut y0, mut y1) = (0i128, 1i128);

    while b != 0 {
        let q = a / b;
        let r = a % b;
        a = b;
        b = r;

        let tmpx = x0 - q * x1;
        x0 = x1;
        x1 = tmpx;

        let tmpy = y0 - q * y1;
        y0 = y1;
        y1 = tmpy;
    }
    (a, x0, y0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_usage() {
        let mut hasher = PolyHashBuilder::new()
            .with_base(257)
            .with_modulus(1_000_000_007)
            .build();

        hasher.hash_slice(b"hello");
        let h1 = hasher.current_hash();

        hasher.clear();
        hasher.hash_slice(b"hello");
        let h2 = hasher.current_hash();

        assert_eq!(h1, h2);
        assert_ne!(h1, 0, "Likely not zero for normal strings");
    }

    #[test]
    fn test_remove_front() {
        let mut hasher = PolynomialRollingHash::new();
        hasher.hash_slice(b"abcd");
        let h1 = hasher.current_hash();

        // remove front 'a'
        hasher.remove_front(b'a' as u64);
        let h2 = hasher.current_hash();
        // Now h2 should be the hash of "bcd"
        let mut alt = PolynomialRollingHash::new();
        alt.hash_slice(b"bcd");
        assert_eq!(h2, alt.current_hash());

        // remove front 'b'
        hasher.remove_front(b'b' as u64);
        let h3 = hasher.current_hash();
        alt.clear();
        alt.hash_slice(b"cd");
        assert_eq!(h3, alt.current_hash());
    }
}
