//! # Perfect Hashing
//!
//! This module implements a **minimal perfect hash** construction for a static set of distinct keys.
//! The resulting hash function maps each key to a unique index in the range `[0..keys.len())`
//! with **no collisions**. Once built, lookups for those keys complete in `O(1)` time.
//!
//! ## Overview of the Algorithm
//!
//! 1. **Collect Keys**: We assume we have a static set of distinct keys (e.g., `&[String]`).
//!
//! 2. **Hash Functions**: We define two universal hash functions `h1` and `h2` (each using a different seed).
//!    For each key `k`, the pair `(h1(k), h2(k))` indicates an undirected edge between the nodes
//!    `h1(k)` and `h2(k)` in a bipartite graph. Node indices live in `[0..n)`, where `n` is the number of keys.
//!
//! 3. **Building a Perfect Hash**: We process each connected component/cycle of the bipartite graph
//!    in a specific order. By assigning an integer offset (`g[u]`) for each node `u`, we can ensure
//!    that all edges (keys) map to unique results:
//!    \[ \text{index}(k) = \bigl(h1(k) + g[h1(k)] + g[h2(k)]\bigr) \mod n. \]
//!    A BFS or DFS approach visits each edge exactly once, allowing us to fix offsets in a way
//!    that yields no collisions among all keys.
//!
//! 4. **Lookup**: Once built, for any key `k` in the set, we compute its perfect-hash index as above.
//!    This index will be unique among all keys. If `k` was not in the original set, there is no collision
//!    guarantee—but typically you'd store the keys separately and compare if needed.
//!
//! ## Note
//! This implementation is suitable for **learning** or certain specialized use cases.
//! For large-scale or high-performance scenarios, advanced structures or existing libraries
//! (e.g. [phf], [fst], etc.) may be more efficient or robust.

use std::collections::{HashSet, VecDeque};
use std::hash::Hasher;

/// A minimal perfect hash function for a static set of distinct keys.
#[derive(Debug)]
pub struct PerfectHash {
    /// Number of distinct keys.
    pub size: usize,
    /// The seeds for hash1 and hash2.
    hash_seed1: u64,
    hash_seed2: u64,
    /// The integer offsets g[u] for each node u in `[0..size)`.
    pub g: Vec<u64>,
}

impl PerfectHash {
    /// Builds a minimal perfect hash for the given set of distinct keys.
    ///
    /// # Panics
    /// - If `keys` is empty.
    /// - If we fail to build a perfect hash after trying multiple seeds (unlikely for smaller sets).
    pub fn build(keys: &[String]) -> Self {
        assert!(!keys.is_empty(), "No keys provided.");

        // We'll pick random seeds for hash1, hash2 until we can build a collision-free mapping.
        // In practice, you might do more systematic or advanced approaches for large sets.
        // We'll do a simple loop over attempts.
        use rand::{rngs::StdRng, Rng, SeedableRng};

        let n = keys.len();
        let mut attempt_rng = StdRng::seed_from_u64(0xABCD1234);
        for _ in 0..1000 {
            let seed1 = attempt_rng.gen::<u64>();
            let seed2 = attempt_rng.gen::<u64>();

            match try_build_phf(keys, seed1, seed2) {
                Ok(g) => {
                    return PerfectHash {
                        size: n,
                        hash_seed1: seed1,
                        hash_seed2: seed2,
                        g,
                    };
                }
                Err(_) => {
                    // try next
                }
            }
        }
        panic!("Unable to build a perfect hash after many attempts. Increase attempts or refine approach.");
    }

    /// Returns the minimal perfect hash index in `[0..size)` for the given key.
    ///
    /// This index is guaranteed unique for any key in the original set used to build this structure.
    ///
    /// # Note
    /// If the key was *not* in the original set, this function still produces an index, but collisions
    /// are possible. Typically you'd store the keys or an additional data structure if membership checks
    /// are required.
    pub fn hash_index(&self, key: &str) -> usize {
        let i1 = hash_str(key, self.hash_seed1) % self.size as u64;
        let i2 = hash_str(key, self.hash_seed2) % self.size as u64;
        let idx = (i1 + self.g[i1 as usize] + self.g[i2 as usize]) % (self.size as u64);
        idx as usize
    }
}

/// Attempts to build the `g[]` array for the bipartite graph. If collisions or conflicts arise,
/// returns `Err(())`.
fn try_build_phf(keys: &[String], seed1: u64, seed2: u64) -> Result<Vec<u64>, ()> {
    let n = keys.len();

    // Special case for n=1: just map to index 0
    if n == 1 {
        return Ok(vec![0]);
    }

    // Build edges: each key defines an edge from h1(k) to h2(k)
    let mut edges = Vec::with_capacity(n);
    for (k_idx, k) in keys.iter().enumerate() {
        let i1 = hash_str(k, seed1) % (n as u64);
        let i2 = hash_str(k, seed2) % (n as u64);
        // Only reject self-loops for n>2
        if n > 2 && i1 == i2 {
            return Err(());
        }
        edges.push((i1 as usize, i2 as usize, k_idx));
    }

    // For n=2, we need to ensure different indices when used in hash_index
    if n == 2 {
        // Get hash values for both keys
        let (i1_0, i2_0) = (hash_str(&keys[0], seed1) % 2, hash_str(&keys[0], seed2) % 2);
        let (i1_1, i2_1) = (hash_str(&keys[1], seed1) % 2, hash_str(&keys[1], seed2) % 2);

        // Try all possible g values
        for g0 in 0..2u64 {
            for g1 in 0..2u64 {
                let g = vec![g0, g1];

                // Calculate actual indices using same formula as hash_index
                let idx0 = (i1_0 + g[i1_0 as usize] + g[i2_0 as usize]) % 2;
                let idx1 = (i1_1 + g[i1_1 as usize] + g[i2_1 as usize]) % 2;

                // If indices are different, we found a solution
                if idx0 != idx1 {
                    return Ok(g);
                }
            }
        }

        // No solution found with these seeds
        return Err(());
    }

    // For n>2, use the graph-based approach
    let mut adjacency = vec![Vec::new(); n];
    for &(u, v, e_idx) in &edges {
        adjacency[u].push((v, e_idx)); // Only add forward edges
    }

    let mut g = vec![0u64; n];
    let mut visited = vec![false; n];
    let mut used_edge = vec![false; n];

    // Process each component
    for start in 0..n {
        if visited[start] {
            continue;
        }

        visited[start] = true;
        let mut queue = VecDeque::new();
        queue.push_back(start);
        let mut used_indices = HashSet::new();

        while let Some(u) = queue.pop_front() {
            for &(v, key_idx) in &adjacency[u] {
                if used_edge[key_idx] {
                    continue;
                }

                let i1 = edges[key_idx].0;
                let i2 = edges[key_idx].1;
                let mut idx = (i1 as u64 + g[i1] + g[i2]) % (n as u64);

                if !used_indices.insert(idx) {
                    let mut found = false;
                    for new_g in 0..n as u64 {
                        if !visited[v] {
                            g[v] = new_g;
                            idx = (i1 as u64 + g[i1] + g[v]) % (n as u64);
                            if used_indices.insert(idx) {
                                found = true;
                                break;
                            }
                        }
                    }
                    if !found {
                        return Err(());
                    }
                }

                if !visited[v] {
                    visited[v] = true;
                    queue.push_back(v);
                }
                used_edge[key_idx] = true;
            }
        }
    }

    // Final verification using same index calculation as build
    let mut used = HashSet::new();
    for k in keys.iter() {
        let i1 = hash_str(k, seed1) % (n as u64);
        let i2 = hash_str(k, seed2) % (n as u64);
        let idx = (i1 + g[i1 as usize] + g[i2 as usize]) % (n as u64);
        if !used.insert(idx) {
            return Err(());
        }
    }

    Ok(g)
}

/// A basic string hash function using a `seed`, building on a 64-bit hasher approach.
///
/// This is not cryptographically secure, but it suffices to form distinct edges in the bipartite graph.
fn hash_str(s: &str, seed: u64) -> u64 {
    let mut h = Fnv64 {
        state: 0xcbf29ce484222325 ^ seed,
    };
    for (shift, b) in s.as_bytes().iter().enumerate() {
        h.write_u8(*b);
        // Rotate state to mix bits better
        h.state = h.state.rotate_left((shift & 63) as u32);
    }
    h.finish()
}

// A simple 64-bit FNV-1a hasher with a seed.
struct Fnv64 {
    state: u64,
}

impl Hasher for Fnv64 {
    fn finish(&self) -> u64 {
        self.state
    }
    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.write_u8(b);
        }
    }
    fn write_u8(&mut self, b: u8) {
        const FNV_PRIME_64: u64 = 0x100000001b3;
        self.state ^= b as u64;
        self.state = self.state.wrapping_mul(FNV_PRIME_64);
    }
}

/// # Example
///
/// ```rust
/// use perfect_hash::PerfectHash;
///
/// let keys = vec![
///     "apple".to_string(),
///     "banana".to_string(),
///     "orange".to_string(),
/// ];
///
/// // Build a minimal perfect hash for these keys
/// let ph = PerfectHash::build(&keys);
///
/// // Now each key has a unique index in [0..3).
/// for (i, k) in keys.iter().enumerate() {
///     let idx = ph.hash_index(k);
///     println!("Key = {:?}, index = {}", k, idx);
///     // Each i is unique in [0..3).
/// }
/// ```

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_small_phf() {
        let keys = vec![
            "apple".to_string(),
            "banana".to_string(),
            "cherry".to_string(),
            "date".to_string(),
            "eggplant".to_string(),
        ];
        let ph = PerfectHash::build(&keys);

        let mut seen = std::collections::HashSet::new();
        for k in &keys {
            let idx = ph.hash_index(k);
            assert!(seen.insert(idx), "Index collision for key {:?}", k);
        }
        assert_eq!(seen.len(), keys.len());
    }

    #[test]
    fn test_single_key() {
        let keys = vec!["solo".to_string()];
        let ph = PerfectHash::build(&keys);
        let idx = ph.hash_index("solo");
        assert_eq!(idx, 0);
    }

    #[test]
    fn test_two_keys() {
        let keys = vec!["one".to_string(), "two".to_string()];
        let ph = PerfectHash::build(&keys);
        let i1 = ph.hash_index("one");
        let i2 = ph.hash_index("two");
        assert!(i1 != i2);
        assert!(i1 < 2 && i2 < 2);
    }
}
