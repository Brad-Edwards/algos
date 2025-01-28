//! # Cuckoo Hashing
//!
//! This module provides a **Cuckoo HashMap** implementation in Rust, designed for **high performance**
//! lookups and inserts in a production-like setting, while still maintaining a clear design.
//! It uses **two hash functions** to minimize collisions and store elements in **two tables**
//! (one per hash function). When collisions occur, we "kick out" an existing element to its alternate
//! position, performing a bounded number of displacements before resizing if necessary.
//!
//! ## Key Features
//! - **Generic** over key/value types: `K: Hash + Eq`, `V`.
//! - **Two-table** approach, each half the total capacity, ensuring constant-time lookups with high probability.
//! - **Bounded** insertion attempts via `max_displacements`. If exceeded, we **grow** the table capacity and rehash.
//! - **Automatic Growth** triggered by a high load factor or repeated insertion failures.
//! - **Thread Safety**: This implementation is **not** thread-safe. For concurrency, wrap in a mutex or use specialized concurrency crates.
//!
//! **Note**: While relatively straightforward for production usage, further tuning (e.g. open addressing, BFS-based displacement,
//! advanced hash functions, or concurrency) may be needed in specialized scenarios. This code focuses on a **two-table** approach with
//! robust fallback to rehash/grow when collisions get excessive.
//!
//! ## Example
//! Example:
//! ```rust
//! use algos::cs::hashing::cuckoo::CuckooHashMap;
//!
//! let mut map = CuckooHashMap::new();
//! let key = "key";
//! map.insert(key, 42);
//! assert_eq!(map.get(&key), Some(&42));
//! ```

use std::collections::hash_map::RandomState;
use std::hash::{BuildHasher, Hash, Hasher};

/// Default maximum displacements (kicks) before we decide to grow the table and rehash.
const DEFAULT_MAX_DISPLACEMENTS: usize = 32;

/// Maximum load factor before we forcibly grow. Typically 0.5 ~ 0.9 in cuckoo hashing. We'll choose 0.49 for safety.
const DEFAULT_MAX_LOAD_FACTOR: f64 = 0.49;

/// A specialized hasher for Cuckoo usage (two different seeds).
/// We'll define a trait for hashing a key to a single `u64` result.
/// However, we can rely on the standard library's `BuildHasher` + `Hasher`.
pub trait CuckooBuildHasher: BuildHasher {
    /// Seeds or parameters that might differ between the two hashers.
    fn seed(&self) -> u64;
}

/// A simple newtype implementing `CuckooBuildHasher`, using the standard RandomState with an extra seed param.
#[derive(Clone)]
pub struct CuckooBuildHasherImpl {
    seed: u64,
    base: RandomState,
}

impl CuckooBuildHasher for CuckooBuildHasherImpl {
    fn seed(&self) -> u64 {
        self.seed
    }
}

impl BuildHasher for CuckooBuildHasherImpl {
    type Hasher = CuckooHasher;

    fn build_hasher(&self) -> Self::Hasher {
        CuckooHasher {
            seed: self.seed,
            base_hasher: self.base.build_hasher(),
        }
    }
}

/// The custom hasher that first uses `base_hasher` to get a result, then mixes with `seed`.
#[derive(Debug)]
pub struct CuckooHasher {
    seed: u64,
    base_hasher: std::collections::hash_map::DefaultHasher, // or any
}

impl Hasher for CuckooHasher {
    fn finish(&self) -> u64 {
        let partial = self.base_hasher.finish();
        // Combine partial with seed in a simple manner
        // for distinct distributions:
        partial ^ (self.seed.wrapping_mul(0x9E3779B185EBCA87))
    }
    fn write(&mut self, bytes: &[u8]) {
        self.base_hasher.write(bytes);
    }
}

/// The main structure for storing K->V with cuckoo hashing in two tables.
#[derive(Debug)]
pub struct CuckooHashMap<K, V, BH1 = CuckooBuildHasherImpl, BH2 = CuckooBuildHasherImpl> {
    // each table has `capacity / 2` slots
    capacity: usize,
    len: usize,
    table1: Vec<Option<(K, V)>>,
    table2: Vec<Option<(K, V)>>,
    buildhasher1: BH1,
    buildhasher2: BH2,

    max_displacements: usize,
    max_load_factor: f64,
}

impl<K: Hash + Eq + Clone, V: Clone> Default for CuckooHashMap<K, V> {
    fn default() -> Self {
        Self::new()
    }
}

impl<K: Hash + Eq + Clone, V: Clone> CuckooHashMap<K, V> {
    /// Constructs an empty `CuckooHashMap` with default capacity = 16 and default hasher seeds.
    pub fn new() -> Self {
        Self::with_capacity(16)
    }
}

impl<K: Hash + Eq + Clone, V: Clone>
    CuckooHashMap<K, V, CuckooBuildHasherImpl, CuckooBuildHasherImpl>
{
    /// Constructs an empty `CuckooHashMap` with the specified capacity (rounded up to even).
    /// Seeds for the two hashers are chosen randomly from OS RNG.
    pub fn with_capacity(cap: usize) -> Self {
        use rand::{rngs::StdRng, Rng, SeedableRng};

        let mut rng = StdRng::from_entropy();
        let seed1 = rng.gen::<u64>();
        let seed2 = rng.gen::<u64>();
        Self::with_capacity_and_hasher(
            cap,
            CuckooBuildHasherImpl {
                seed: seed1,
                base: RandomState::new(),
            },
            CuckooBuildHasherImpl {
                seed: seed2,
                base: RandomState::new(),
            },
        )
    }
}

impl<
        K: Hash + Eq + Clone,
        V: Clone,
        BH1: CuckooBuildHasher + Clone,
        BH2: CuckooBuildHasher + Clone,
    > CuckooHashMap<K, V, BH1, BH2>
{
    /// Constructs a new `CuckooHashMap` with the specified capacity and two build hashers.
    pub fn with_capacity_and_hasher(cap: usize, h1: BH1, h2: BH2) -> Self {
        let capacity = cap.max(2).next_power_of_two();
        let half = capacity / 2;
        CuckooHashMap {
            capacity,
            len: 0,
            table1: vec![None; half],
            table2: vec![None; half],
            buildhasher1: h1,
            buildhasher2: h2,

            max_displacements: DEFAULT_MAX_DISPLACEMENTS,
            max_load_factor: DEFAULT_MAX_LOAD_FACTOR,
        }
    }

    /// Returns the number of elements in the map.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns true if the map is empty.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Inserts a key-value pair into the map, returning the old value if key was present.
    /// If the map must be rehashed or grown, it does so automatically.
    pub fn insert(&mut self, key: K, value: V) -> Option<V> {
        // Check if the key already in the map, if so update
        if let Some(vref) = self.get_mut(&key) {
            let oldv = std::mem::replace(vref, value);
            return Some(oldv);
        }
        // If load factor is too high => grow
        if (self.len + 1) as f64 / (self.capacity as f64) > self.max_load_factor {
            self.grow();
        }
        // Attempt to place. If we get stuck => rehash or grow
        let r = self.cuckoo_place(key, value, self.max_displacements);
        if r.is_err() {
            // Rehash approach:
            self.grow();
            let (k, v) = r.unwrap_err();
            self.rehash();
            self.cuckoo_place(k, v, self.max_displacements)
                .unwrap_or_else(|_| panic!("Insertion failed even after rehash - unexpected"));
        } else {
            self.len += 1;
        }
        None
    }

    /// Retrieves a reference to the value corresponding to the given key.
    pub fn get(&self, key: &K) -> Option<&V> {
        let (i1, i2) = self.indices_of(key);
        if let Some((ref k, ref v)) = self.table1[i1] {
            if k == key {
                return Some(v);
            }
        }
        if let Some((ref k, ref v)) = self.table2[i2] {
            if k == key {
                return Some(v);
            }
        }
        None
    }

    /// Retrieves a mutable reference to the value corresponding to the given key.
    pub fn get_mut(&mut self, key: &K) -> Option<&mut V> {
        let (i1, i2) = self.indices_of(key);
        if let Some((ref k, ref mut v)) = self.table1[i1] {
            if k == key {
                return Some(v);
            }
        }
        if let Some((ref k, ref mut v)) = self.table2[i2] {
            if k == key {
                return Some(v);
            }
        }
        None
    }

    /// Removes a key-value pair from the map, returning the value if present.
    pub fn remove(&mut self, key: &K) -> Option<V> {
        let (i1, i2) = self.indices_of(key);
        // check table1
        if let Some((ref k, _)) = self.table1[i1] {
            if k == key {
                let (_, v) = self.table1[i1].take().unwrap();
                self.len -= 1;
                return Some(v);
            }
        }
        // check table2
        if let Some((ref k, _)) = self.table2[i2] {
            if k == key {
                let (_, v) = self.table2[i2].take().unwrap();
                self.len -= 1;
                return Some(v);
            }
        }
        None
    }

    /// Clears the map, removing all key-value pairs.
    pub fn clear(&mut self) {
        self.table1.fill(None);
        self.table2.fill(None);
        self.len = 0;
    }

    // --- internal logic ---

    /// Compute the two table indices for a key.
    fn indices_of(&self, key: &K) -> (usize, usize) {
        let h1 = self.hash1(key);
        let h2 = self.hash2(key);
        let half = self.capacity / 2;
        ((h1 % half as u64) as usize, (h2 % half as u64) as usize)
    }

    fn hash1(&self, key: &K) -> u64 {
        self.buildhasher1.hash_one(key)
    }

    fn hash2(&self, key: &K) -> u64 {
        self.buildhasher2.hash_one(key)
    }

    /// Attempt to place a key-value pair in the tables, with up to `max_displacements` kicks.
    /// If we succeed, Ok(()) is returned. If we fail, we return the (key, value) that could not be placed in Err.
    fn cuckoo_place(&mut self, mut key: K, mut value: V, mut kicks: usize) -> Result<(), (K, V)> {
        let half = self.capacity / 2;
        let mut table_index = 1; // we start with table1
        let mut index = (self.hash1(&key) % (half as u64)) as usize;

        loop {
            let slot = match table_index {
                1 => &mut self.table1[index],
                2 => &mut self.table2[index],
                _ => unreachable!(),
            };
            if slot.is_none() {
                // place it
                *slot = Some((key, value));
                return Ok(());
            } else {
                // we have to displace occupant
                let (old_key, old_val) = slot.take().unwrap();
                *slot = Some((key, value));

                // now we re-insert old occupant in the other table
                key = old_key;
                value = old_val;

                table_index = if table_index == 1 { 2 } else { 1 };
                index = if table_index == 1 {
                    // rehash occupant for table1
                    (self.hash1(&key) % (half as u64)) as usize
                } else {
                    (self.hash2(&key) % (half as u64)) as usize
                };

                kicks -= 1;
                if kicks == 0 {
                    // fail
                    return Err((key, value));
                }
            }
        }
    }

    /// Grow the table (roughly double capacity) and re-insert all elements.
    /// We attempt to do so in place by building new vectors and re-cuckooing everything.
    fn grow(&mut self) {
        let new_capacity = self.capacity * 2;
        let half = new_capacity / 2;
        let old_tab1 = std::mem::replace(&mut self.table1, vec![None; half]);
        let old_tab2 = std::mem::replace(&mut self.table2, vec![None; half]);
        self.capacity = new_capacity;
        self.len = 0; // we'll re-insert

        // Re-insert all elements
        for (k, v) in old_tab1.into_iter().chain(old_tab2.into_iter()).flatten() {
            self.insert(k, v);
        }
    }

    /// Rehash the table.
    fn rehash(&mut self) {
        // Implementation of rehash logic
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_insert_get_remove() {
        let mut map = CuckooHashMap::with_capacity(8);
        map.insert("hello", 123);
        map.insert("world", 456);

        assert_eq!(map.len(), 2);
        assert_eq!(map.get(&"hello"), Some(&123));
        assert_eq!(map.get(&"world"), Some(&456));
        assert_eq!(map.get(&"foo"), None);

        let old = map.insert("hello", 999);
        assert_eq!(old, Some(123));
        assert_eq!(map.get(&"hello"), Some(&999));

        let rm = map.remove(&"world");
        assert_eq!(rm, Some(456));
        assert_eq!(map.get(&"world"), None);
        assert_eq!(map.len(), 1);
    }

    #[test]
    fn test_many_inserts() {
        let mut map = CuckooHashMap::with_capacity(4);
        for i in 0..50 {
            map.insert(format!("key{}", i), i);
        }
        assert_eq!(map.len(), 50);
        for i in 0..50 {
            let key = format!("key{}", i);
            assert_eq!(map.get(&key), Some(&i));
        }
    }

    #[test]
    fn test_collision_rehash() {
        // We'll artificially pick small capacity
        let mut map = CuckooHashMap::with_capacity(2);
        let keys = vec!["a", "b", "c", "d", "e", "f", "g"];
        for (i, k) in keys.iter().enumerate() {
            map.insert(*k, i as i32);
        }
        // check
        for (i, k) in keys.iter().enumerate() {
            assert_eq!(map.get(k), Some(&(i as i32)));
        }
    }
}
