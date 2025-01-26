//! # Separate Chaining Hash Table
//!
//! This module implements a **HashMap** using **separate chaining** in Rust, aimed at production-like usage.
//! It supports:
//! - **Generic** key-value pairs (`K: Hash + Eq, V`).
//! - **Customizable** capacity growth and load factor threshold.
//! - **Configurable** hasher using `BuildHasher` traits, with user-defined or random seeds.
//! - **Insert**, **get**, **remove**, **iter** (basic iteration) operations with expected **O(1)** average performance.
//!
//! This is not a standard library replacement but is designed as a robust demonstration for real usage.

use std::collections::hash_map::RandomState;
use std::hash::{BuildHasher, Hash, Hasher};

/// Default initial capacity if none specified.
const DEFAULT_INITIAL_CAPACITY: usize = 16;

/// Default maximum load factor before resizing.
const DEFAULT_MAX_LOAD_FACTOR: f64 = 0.75;

/// A single entry in a chain: `(K, V)`.
#[derive(Debug)]
struct Entry<K, V> {
    key: K,
    value: V,
}

/// A "bucket" is a vector of entries for separate chaining.
type Bucket<K, V> = Vec<Entry<K, V>>;

/// A separate-chaining HashMap with generic `K, V` and a customizable hasher.
#[derive(Debug)]
pub struct ChainedHashMap<K, V, S = RandomState> {
    buckets: Vec<Bucket<K, V>>,
    /// The number of stored key-value pairs.
    len: usize,
    /// The capacity in terms of how many buckets we have.
    bucket_count: usize,
    /// The maximum load factor (ratio = len / bucket_count).
    max_load_factor: f64,
    /// Hasher builder.
    build_hasher: S,
}

/// A builder for the `ChainedHashMap`.
/// Typically you'll call `.with_hasher(...)`, `.with_capacity(...)`, etc., then `.build()`.
#[derive(Debug)]
pub struct ChainedHashMapBuilder<S> {
    capacity: usize,
    max_load_factor: f64,
    hasher: S,
}

impl Default for ChainedHashMapBuilder<RandomState> {
    fn default() -> Self {
        Self {
            capacity: DEFAULT_INITIAL_CAPACITY,
            max_load_factor: DEFAULT_MAX_LOAD_FACTOR,
            hasher: RandomState::new(),
        }
    }
}

impl ChainedHashMapBuilder<RandomState> {
    /// Creates a new builder with default capacity and default hasher (RandomState).
    pub fn new() -> Self {
        Default::default()
    }
}

impl<S: BuildHasher + Clone> ChainedHashMapBuilder<S> {
    /// Sets an explicit capacity (number of buckets).
    /// The actual data capacity for key-values is unbounded; we just store them in buckets.
    pub fn with_capacity(mut self, capacity: usize) -> Self {
        self.capacity = capacity.max(1);
        self
    }

    /// Sets the maximum load factor. If `len / bucket_count` exceeds it, we resize.
    pub fn with_max_load_factor(mut self, lf: f64) -> Self {
        assert!(lf > 0.0, "Load factor must be > 0");
        self.max_load_factor = lf;
        self
    }

    /// Sets a custom hasher builder.
    pub fn with_hasher<T: BuildHasher + Clone>(self, hasher: T) -> ChainedHashMapBuilder<T> {
        ChainedHashMapBuilder {
            capacity: self.capacity,
            max_load_factor: self.max_load_factor,
            hasher,
        }
    }

    /// Build the final `ChainedHashMap`.
    pub fn build<K: Hash + Eq, V>(self) -> ChainedHashMap<K, V, S> {
        let bucket_count = self.capacity;
        let mut buckets = Vec::with_capacity(bucket_count);
        buckets.resize_with(bucket_count, Default::default);

        ChainedHashMap {
            buckets,
            len: 0,
            bucket_count,
            max_load_factor: self.max_load_factor,
            build_hasher: self.hasher,
        }
    }
}

impl<K: Hash + Eq, V> ChainedHashMap<K, V> {
    /// Creates a new map with default capacity and default hasher.
    pub fn new() -> Self {
        ChainedHashMapBuilder::new().build()
    }

    /// Creates a new map with a specified initial capacity and default hasher.
    pub fn with_capacity(cap: usize) -> Self {
        ChainedHashMapBuilder::new().with_capacity(cap).build()
    }
}

impl<K: Hash + Eq, V, S: BuildHasher + Clone> ChainedHashMap<K, V, S> {
    /// Returns the number of key-value pairs in the map.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns true if the map is empty.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Inserts a key-value pair into the map.
    /// If the key already exists, its value is replaced and the old value returned.
    pub fn insert(&mut self, key: K, value: V) -> Option<V> {
        // If load factor exceeded, grow
        if (self.len + 1) as f64 / (self.bucket_count as f64) > self.max_load_factor {
            self.resize();
        }

        let bucket_index = self.bucket_index(&key);
        let bucket = &mut self.buckets[bucket_index];

        // Check if key exists
        for entry in bucket.iter_mut() {
            if entry.key == key {
                let oldv = std::mem::replace(&mut entry.value, value);
                return Some(oldv);
            }
        }
        // Insert
        bucket.push(Entry { key, value });
        self.len += 1;
        None
    }

    /// Returns a reference to the value corresponding to the key, if present.
    pub fn get(&self, key: &K) -> Option<&V> {
        let idx = self.bucket_index(key);
        for entry in &self.buckets[idx] {
            if &entry.key == key {
                return Some(&entry.value);
            }
        }
        None
    }

    /// Returns a mutable reference to the value corresponding to the key, if present.
    pub fn get_mut(&mut self, key: &K) -> Option<&mut V> {
        let idx = self.bucket_index(key);
        for entry in &mut self.buckets[idx] {
            if &entry.key == key {
                return Some(&mut entry.value);
            }
        }
        None
    }

    /// Removes and returns the value for the specified key, if present.
    pub fn remove(&mut self, key: &K) -> Option<V> {
        let idx = self.bucket_index(key);
        let bucket = &mut self.buckets[idx];
        let mut i = 0;
        while i < bucket.len() {
            if &bucket[i].key == key {
                let entry = bucket.swap_remove(i);
                self.len -= 1;
                return Some(entry.value);
            }
            i += 1;
        }
        None
    }

    /// Clears the map, removing all key-value pairs.
    pub fn clear(&mut self) {
        for bucket in &mut self.buckets {
            bucket.clear();
        }
        self.len = 0;
    }

    /// Returns an iterator over the key-value pairs in the map.
    pub fn iter(&self) -> impl Iterator<Item = (&K, &V)> {
        self.buckets
            .iter()
            .flat_map(|bucket| bucket.iter().map(|entry| (&entry.key, &entry.value)))
    }

    /// Internal function computing the bucket index for a given key.
    fn bucket_index(&self, key: &K) -> usize {
        let mut hasher = self.build_hasher.build_hasher();
        key.hash(&mut hasher);
        let h = hasher.finish();
        (h as usize) % self.bucket_count
    }

    /// Resize the map (roughly doubling the number of buckets) and re-insert existing entries.
    fn resize(&mut self) {
        let new_bucket_count = (self.bucket_count * 2).max(1);
        let mut new_buckets = Vec::with_capacity(new_bucket_count);
        new_buckets.resize_with(new_bucket_count, Default::default);

        // We'll re-insert everything
        for bucket in self.buckets.drain(..) {
            for entry in bucket {
                let mut hasher = self.build_hasher.build_hasher();
                entry.key.hash(&mut hasher);
                let h = hasher.finish() as usize % new_bucket_count;
                new_buckets[h].push(entry);
            }
        }
        self.bucket_count = new_bucket_count;
        self.buckets = new_buckets;
        // len remains the same
    }
}

// Additional convenience methods or iter types can be added if needed for production usage.

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_insert_get_remove() {
        let mut map = ChainedHashMap::with_capacity(4);
        assert_eq!(map.len(), 0);
        assert!(map.is_empty());

        // Insert
        let old = map.insert("foo", 123);
        assert_eq!(old, None);
        assert_eq!(map.len(), 1);
        assert!(!map.is_empty());

        // Insert second
        let old = map.insert("bar", 999);
        assert_eq!(old, None);
        assert_eq!(map.len(), 2);

        // Insert existing
        let old = map.insert("foo", 456);
        assert_eq!(old, Some(123));
        assert_eq!(map.len(), 2);

        // get
        assert_eq!(map.get(&"foo"), Some(&456));
        assert_eq!(map.get(&"bar"), Some(&999));
        assert_eq!(map.get(&"baz"), None);

        // remove
        let rm = map.remove(&"bar");
        assert_eq!(rm, Some(999));
        assert_eq!(map.len(), 1);
        assert_eq!(map.get(&"bar"), None);
    }

    #[test]
    fn test_resize() {
        let mut map = ChainedHashMap::with_capacity(2);
        for i in 0..10 {
            map.insert(format!("key{}", i), i);
        }
        for i in 0..10 {
            assert_eq!(map.get(&format!("key{}", i)), Some(&i));
        }
    }

    #[test]
    fn test_iter() {
        let mut map = ChainedHashMap::new();
        map.insert("one", 1);
        map.insert("two", 2);
        map.insert("three", 3);

        let mut items: Vec<_> = map.iter().map(|(k, v)| (*k, *v)).collect();
        items.sort_by_key(|x| x.0);
        assert_eq!(items, vec![("one", 1), ("three", 3), ("two", 2)]);
    }
}
