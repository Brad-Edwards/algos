use std::collections::VecDeque;

/// A generic rolling hash struct based on polynomial rolling.
///
/// # Type Parameters
/// * `B`: The base used in the polynomial hash (typically a prime number)
/// * `M`: The modulus for the hash (typically a large prime number)
#[derive(Debug, Clone)]
pub struct RollingHash<const B: u64, const M: u64> {
    /// The current rolling hash value
    hash: u64,
    /// Current power of base corresponding to the size of the window
    base_power: u64,
    /// Length of the current window
    window_size: usize,
    /// Queue of elements currently in the window
    elements: VecDeque<u8>,
}

impl<const B: u64, const M: u64> Default for RollingHash<B, M> {
    fn default() -> Self {
        Self::new()
    }
}

impl<const B: u64, const M: u64> RollingHash<B, M> {
    /// Creates a new RollingHash with an empty initial window.
    ///
    /// # Example
    /// ```
    /// use algos::cs::string::rolling_hash::RollingHash;
    ///
    /// let rh: RollingHash<257, 1_000_000_007> = RollingHash::new();
    /// ```
    pub fn new() -> Self {
        Self {
            hash: 0,
            base_power: 1,
            window_size: 0,
            elements: VecDeque::new(),
        }
    }

    /// Returns the current hash value of the window.
    ///
    /// # Example
    /// ```
    /// use algos::cs::string::rolling_hash::RollingHash;
    ///
    /// let mut rh: RollingHash<257, 1_000_000_007> = RollingHash::new();
    /// rh.push(b'a');
    /// let hash = rh.current_hash();
    /// ```
    #[inline]
    pub fn current_hash(&self) -> u64 {
        self.hash
    }

    /// Returns the current window size.
    #[inline]
    pub fn window_size(&self) -> usize {
        self.window_size
    }

    /// Pushes a new element (byte) to the window, updating the rolling hash.
    ///
    /// # Example
    /// ```
    /// use algos::cs::string::rolling_hash::RollingHash;
    ///
    /// let mut rh: RollingHash<257, 1_000_000_007> = RollingHash::new();
    /// rh.push(b'a');
    /// ```
    pub fn push(&mut self, byte: u8) {
        // New hash: hash * B + byte (mod M)
        let hash_term = (self.hash % M).wrapping_mul(B % M) % M;
        let byte_term = byte as u64 % M;
        self.hash = (hash_term + byte_term) % M;
        self.elements.push_back(byte);

        // Keep track of B^(window_size) mod M
        if self.window_size > 0 {
            self.base_power = (self.base_power % M).wrapping_mul(B % M) % M;
        }
        self.window_size += 1;
    }

    /// Pops an element from the front of the window, updating the rolling hash.
    ///
    /// # Returns
    /// * `Option<u8>` - The popped byte, or None if the window was empty
    ///
    /// # Example
    /// ```
    /// use algos::cs::string::rolling_hash::RollingHash;
    ///
    /// let mut rh: RollingHash<257, 1_000_000_007> = RollingHash::new();
    /// rh.push(b'a');
    /// rh.push(b'b');
    /// let popped = rh.pop(); // Some(b'a')
    /// ```
    pub fn pop(&mut self) -> Option<u8> {
        if self.window_size == 0 {
            return None;
        }

        let front = self.elements.pop_front()?;
        self.window_size -= 1;

        // Recompute hash to remove the contribution of the front element
        let front_term = (front as u64 % M).wrapping_mul(self.base_power % M) % M;
        self.hash = if self.hash >= front_term {
            self.hash - front_term
        } else {
            M - (front_term - self.hash) % M
        };

        // Adjust base_power down since the window is now one shorter
        if self.window_size > 0 {
            let b_inv =
                mod_inv(B % M, M).expect("B and M must be coprime for modular inverse to exist");
            self.base_power = (self.base_power % M).wrapping_mul(b_inv) % M;
        } else {
            self.base_power = 1;
        }

        Some(front)
    }

    /// Clears the rolling hash window.
    pub fn clear(&mut self) {
        self.hash = 0;
        self.base_power = 1;
        self.window_size = 0;
        self.elements.clear();
    }
}

/// Computes the modular multiplicative inverse of `a` under modulo `m`.
///
/// # Arguments
/// * `a` - The number to find the inverse for
/// * `m` - The modulus
///
/// # Returns
/// * `Option<u64>` - The modular multiplicative inverse if it exists
fn mod_inv(a: u64, m: u64) -> Option<u64> {
    let (g, x, _) = extended_gcd(a as i64, m as i64);
    if g != 1 {
        return None;
    }
    Some((((x % m as i64) + m as i64) % m as i64) as u64)
}

/// Extended GCD algorithm to find coefficients of Bézout's identity.
///
/// # Returns
/// * `(i64, i64, i64)` - (g, x, y) where g = gcd(a, b) and ax + by = g
fn extended_gcd(a: i64, b: i64) -> (i64, i64, i64) {
    if b == 0 {
        (a, 1, 0)
    } else {
        let (g, x1, y1) = extended_gcd(b, a % b);
        (g, y1, x1 - (a / b) * y1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rolling_hash_basics() {
        let mut rh: RollingHash<257, 1_000_000_007> = RollingHash::new();

        rh.push(b'a');
        assert_eq!(rh.window_size(), 1);
        let h1 = rh.current_hash();

        rh.push(b'b');
        assert_eq!(rh.window_size(), 2);
        let h2 = rh.current_hash();
        assert_ne!(h1, h2);

        let popped = rh.pop();
        assert_eq!(popped, Some(b'a'));
        assert_eq!(rh.window_size(), 1);

        let h3 = rh.current_hash();
        let mut rh_test: RollingHash<257, 1_000_000_007> = RollingHash::new();
        rh_test.push(b'b');
        assert_eq!(h3, rh_test.current_hash());
    }

    #[test]
    fn test_pop_on_empty() {
        let mut rh: RollingHash<257, 1_000_000_007> = RollingHash::new();
        assert_eq!(rh.pop(), None);
    }

    #[test]
    fn test_clear() {
        let mut rh: RollingHash<257, 1_000_000_007> = RollingHash::new();
        rh.push(b'a');
        rh.push(b'b');
        rh.clear();
        assert_eq!(rh.window_size(), 0);
        assert_eq!(rh.current_hash(), 0);
        assert_eq!(rh.pop(), None);
    }

    #[test]
    fn test_window_operations() {
        let mut rh: RollingHash<257, 1_000_000_007> = RollingHash::new();

        // Push sequence of characters
        let text = b"abcdef";
        for &byte in text {
            rh.push(byte);
        }
        assert_eq!(rh.window_size(), 6);

        // Pop characters one by one
        for &expected in text {
            assert_eq!(rh.pop(), Some(expected));
        }
        assert_eq!(rh.window_size(), 0);
    }

    #[test]
    fn test_hash_consistency() {
        let mut rh1: RollingHash<257, 1_000_000_007> = RollingHash::new();
        let mut rh2: RollingHash<257, 1_000_000_007> = RollingHash::new();

        // Same sequence should produce same hash
        for &byte in b"hello" {
            rh1.push(byte);
            rh2.push(byte);
        }
        assert_eq!(rh1.current_hash(), rh2.current_hash());
    }
}
