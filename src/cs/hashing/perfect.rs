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
use std::hash::{BuildHasher, Hasher};

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
    // Build adjacency list: each key defines an edge from h1(k) to h2(k).
    // We'll store all edges in a vector of (node1, node2, key_index).
    let mut edges = Vec::with_capacity(n);
    for (k_idx, k) in keys.iter().enumerate() {
        let i1 = hash_str(k, seed1) % (n as u64);
        let i2 = hash_str(k, seed2) % (n as u64);
        edges.push((i1 as usize, i2 as usize, k_idx));
    }

    // We'll need to track connected components. We'll store adjacency for each node in [0..n].
    let mut adjacency = vec![Vec::new(); n];
    for &(u, v, e_idx) in &edges {
        adjacency[u].push((v, e_idx));
        adjacency[v].push((u, e_idx));
    }

    // We want to assign an integer offset g[u] for each node u. We'll initialize them with 0.
    let mut g = vec![0u64; n];

    // We'll keep track which edges are "used" in BFS once we fix them. We'll process each connected
    // component in BFS manner, setting g[u] so that each edge in that component yields a unique index.
    let mut visited = vec![false; n];
    let mut used_edge = vec![false; n]; // track each key by index?

    // BFS over connected components
    for start_node in 0..n {
        if adjacency[start_node].is_empty() {
            // isolated node => no edges => g doesn't matter for it
            visited[start_node] = true;
            continue;
        }
        if visited[start_node] {
            continue;
        }

        // BFS queue
        let mut queue = VecDeque::new();
        visited[start_node] = true;
        queue.push_back(start_node);

        while let Some(u) = queue.pop_front() {
            // explore edges from u
            for &(v, key_idx) in &adjacency[u] {
                if used_edge[key_idx] {
                    continue;
                }
                // the key with index key_idx connects u..v
                // We want the resulting index for that key to be distinct.
                // The formula is index = (u + g[u] + g[v]) mod n => must be key_idx for a perfect mapping.
                // So we want: (g[u] + g[v]) mod n = (key_idx - u) mod n. (We'll store that difference in e).
                let required = mod_diff(key_idx as u64, u as u64, n as u64);

                // So g[v] = required - g[u], mod n
                // => g[v] = (required + n - g[u]) mod n
                // We'll fix g[v] if we haven't visited v yet, or check consistency otherwise.
                let needed = mod_diff(required, g[u], n as u64);

                if !visited[v] {
                    visited[v] = true;
                    g[v] = needed;
                    queue.push_back(v);
                } else {
                    // check if it conflicts
                    if g[v] != needed {
                        // conflict => can't build
                        return Err(());
                    }
                }
                // mark edge used
                used_edge[key_idx] = true;
            }
        }
    }

    // If we reach here, we assigned g[] consistently with no collisions => success
    Ok(g)
}

/// Helper: returns (a - b) mod n in [0..n-1].
fn mod_diff(a: u64, b: u64, n: u64) -> u64 {
    ((a + n) - b) % n
}

/// A basic string hash function using a `seed`, building on a 64-bit hasher approach.
///
/// This is not cryptographically secure, but it suffices to form distinct edges in the bipartite graph.
fn hash_str(s: &str, seed: u64) -> u64 {
    // We'll use a simple "Xorshift" or "FNV" style approach for demonstration.
    // You could do a `std::collections::hash_map::DefaultHasher` plus seed, or a custom approach.
    let mut h = Fnv64 {
        state: 0xcbf29ce484222325 ^ seed,
    };
    for b in s.as_bytes() {
        h.write_u8(*b);
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
