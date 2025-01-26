//! # Open Addressing Hash Table
//!
//! This module provides a **HashMap** using *open addressing* in Rust. It supports various probing strategies
//! (linear probing, quadratic probing, double hashing) with easy configuration. The design aims to be
//! **useful in production** for typical workloads, rather than just demonstrating a toy example.
//!
//! ## Key Features
//! - **Generic** key-value pairs (`K: Hash + Eq, V`).
//! - **Open Addressing**: Collisions are resolved by probing other slots in a single contiguous array.
//! - **Configurable Probing**: Choose from [linear, quadratic, double hashing] via `ProbingStrategy`.
//! - **Automatic Growth**: If load factor is exceeded, the table resizes (rehashes all entries) to maintain performance.
//! - **Tombstones**: Removal sets a "tombstone" marker, so subsequent probes can continue past old removed slots.
//!   If too many tombstones accumulate, a rehash can occur to maintain efficiency.
//! - **Customizable** hasher using `BuildHasher` traits (can specify seeds or rely on `RandomState`).
//!
//! **Note**: For concurrency or extremely large data sets, you may need a specialized approach or more advanced data structures.

use std::collections::hash_map::RandomState;
use std::hash::{BuildHasher, Hash, Hasher};

/// Default initial capacity if unspecified.
const DEFAULT_INITIAL_CAPACITY: usize = 16;
/// Default maximum load factor.
const DEFAULT_MAX_LOAD_FACTOR: f64 = 0.75;
/// Default threshold of tombstones vs. capacity to trigger rehash, e.g. 0.2 => if tombstones exceed 20% we rehash.
const DEFAULT_TOMBSTONE_THRESHOLD: f64 = 0.2;

/// An entry can be `Empty`, `Tombstone` (used to be occupied but removed), or `Occupied(key, value)`.
#[derive(Debug, Clone)]
enum Slot<K, V> {
    Empty,
    Tombstone,
    Occupied(K, V),
}

impl<K, V> Default for Slot<K, V> {
    fn default() -> Self {
        Slot::Empty
    }
}

/// The strategy used for collision resolution in open addressing.
#[derive(Debug, Clone, Copy)]
pub enum ProbingStrategy {
    /// Linear probing: next slot = (hash + step) mod capacity
    Linear,
    /// Quadratic probing: next slot = (hash + i^2) mod capacity
    Quadratic,
    /// Double hashing: next slot = (hash + i * hash2) mod capacity
    /// Requires an additional hash function or a second approach.
    DoubleHash,
}

/// A builder for the `OpenAddressingMap`, allowing you to specify capacity, load factor, probing strategy, etc.
#[derive(Debug)]
pub struct OpenAddressingBuilder<S> {
    capacity: usize,
    max_load_factor: f64,
    tombstone_threshold: f64,
    strategy: ProbingStrategy,
    hasher: S,          // For primary hash
    hasher2: Option<S>, // For double hashing
}

impl Default for OpenAddressingBuilder<RandomState> {
    fn default() -> Self {
        Self {
            capacity: DEFAULT_INITIAL_CAPACITY,
            max_load_factor: DEFAULT_MAX_LOAD_FACTOR,
            tombstone_threshold: DEFAULT_TOMBSTONE_THRESHOLD,
            strategy: ProbingStrategy::Linear,
            hasher: RandomState::new(),
            hasher2: None,
        }
    }
}

impl OpenAddressingBuilder<RandomState> {
    /// Create a new builder with default parameters and RandomState hasher.
    pub fn new() -> Self {
        Self::default()
    }
}

impl<S: BuildHasher + Clone> OpenAddressingBuilder<S> {
    /// Sets initial capacity (will be rounded up to next power of two).
    pub fn with_capacity(mut self, cap: usize) -> Self {
        self.capacity = cap.max(1);
        self
    }

    /// Sets the maximum load factor. If (len+1)/capacity > max_load_factor => rehash/grow.
    pub fn with_max_load_factor(mut self, lf: f64) -> Self {
        assert!(lf > 0.0 && lf < 1.0, "Load factor must be in (0,1)");
        self.max_load_factor = lf;
        self
    }

    /// Sets tombstone threshold ratio. If tombstones exceed that fraction of capacity, we rehash to clean them.
    pub fn with_tombstone_threshold(mut self, ratio: f64) -> Self {
        assert!(
            ratio >= 0.0 && ratio < 1.0,
            "Tombstone threshold must be in [0,1)"
        );
        self.tombstone_threshold = ratio;
        self
    }

    /// Sets the collision resolution strategy (linear, quadratic, or double hashing).
    pub fn with_strategy(mut self, strategy: ProbingStrategy) -> Self {
        self.strategy = strategy;
        self
    }

    /// For double hashing, you can optionally provide a second hasher builder.
    pub fn with_double_hasher<T: BuildHasher + Clone>(
        mut self,
        secondary: T,
    ) -> OpenAddressingBuilder<T> {
        let old = self.hasher2.take();
        let new2 = secondary;
        OpenAddressingBuilder {
            capacity: self.capacity,
            max_load_factor: self.max_load_factor,
            tombstone_threshold: self.tombstone_threshold,
            strategy: self.strategy,
            hasher: new2,
            hasher2: old.map(|_| panic!("Chaining secondary hasher conflicts with new type!")),
        }
    }

    /// Finalize building the `OpenAddressingMap` with `K: Hash + Eq, V`.
    /// If strategy=DoubleHash, we require `hasher2` to be Some. Otherwise we panic.
    pub fn build<K: Hash + Eq + Clone, V: Clone>(self) -> OpenAddressingMap<K, V, S> {
        let capacity = self.capacity.next_power_of_two();
        let mut slots = Vec::with_capacity(capacity);
        slots.resize_with(capacity, Default::default);

        // If double hashing, ensure we have a second hasher or we panic
        if let ProbingStrategy::DoubleHash = self.strategy {
            if self.hasher2.is_none() {
                panic!("DoubleHash strategy requires a second hasher. Use `.with_double_hasher(...)` first.");
            }
        }

        OpenAddressingMap {
            slots,
            capacity,
            len: 0,
            tombstones: 0,
            strategy: self.strategy,
            buildhasher1: self.hasher,
            buildhasher2: self.hasher2.clone(),
            max_load_factor: self.max_load_factor,
            tombstone_threshold: self.tombstone_threshold,
        }
    }
}

/// The main open addressing hash map with user-chosen probing strategy.
#[derive(Debug)]
pub struct OpenAddressingMap<K, V, S> {
    slots: Vec<Slot<K, V>>,
    capacity: usize,
    len: usize,
    tombstones: usize,

    strategy: ProbingStrategy,
    buildhasher1: S,
    buildhasher2: Option<S>,

    max_load_factor: f64,
    tombstone_threshold: f64,
}

impl<K: Hash + Eq + Clone, V: Clone> OpenAddressingMap<K, V, RandomState> {
    /// Creates a new map with default parameters and RandomState hasher (linear probing).
    pub fn new() -> Self {
        OpenAddressingBuilder::new().build()
    }
    /// Creates with given capacity.
    pub fn with_capacity(cap: usize) -> Self {
        OpenAddressingBuilder::new().with_capacity(cap).build()
    }
}

impl<K: Hash + Eq + Clone, V: Clone, S: BuildHasher + Clone> OpenAddressingMap<K, V, S> {
    /// Returns the number of key-value pairs in the map.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns true if the map is empty.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Insert a key-value pair. Returns the old value if the key existed.
    /// If we exceed load factor or tombstone ratio, we rehash/grow.
    pub fn insert(&mut self, key: K, value: V) -> Option<V> {
        // check load factor or tombstone ratio
        if (self.len + 1) as f64 / self.capacity as f64 > self.max_load_factor
            || self.tombstones as f64 / self.capacity as f64 > self.tombstone_threshold
        {
            self.grow();
        }

        // search or place
        let mut idx = self.primary_index(&key);
        let second_hash = self.secondary_hash(&key);
        let capacity_mask = self.capacity - 1;

        let mut probe_i = 0;
        loop {
            match &mut self.slots[idx] {
                Slot::Empty => {
                    // place new
                    self.slots[idx] = Slot::Occupied(key, value);
                    self.len += 1;
                    return None;
                }
                Slot::Occupied(k, v) => {
                    if k == &key {
                        // update
                        let oldv = std::mem::replace(v, value);
                        return Some(oldv);
                    }
                }
                Slot::Tombstone => {
                    // we can place new here, but we should keep track in case there's a later slot with same key
                    // but typical open addressing: we can just place it. But we might keep the index as "first tombstone".
                    let tomb_idx = idx;
                    // We can continue searching for the exact key to see if it exists, or place here if not found.
                    // But standard approach: place it immediately. Then we might find the key in a later slot though...
                    // We'll do "Robin Hood" or "classic"? Typically we can do: keep searching to see if key is present.
                    // For production approach, let's do "immediate insertion" (since a tombstone means no occupant).
                    // But we must check if the key is further. We'll do full approach: keep searching for occupant match, if not found, place in tomb_idx.

                    // We'll do a single pass approach: store tomb_idx, continue searching, if found no occupant, place in tomb_idx.
                    let free_spot = tomb_idx;
                    let replaced =
                        self.probe_find_or_place(key, value, second_hash, probe_i, free_spot)?;
                    return replaced;
                }
            }

            probe_i += 1;
            idx = self.next_index(idx, second_hash, probe_i, capacity_mask);
            if probe_i > self.capacity {
                // full or infinite loop
                self.grow();
                return self.insert(key, value);
            }
        }
    }

    /// Retrieve a reference to the value for `key`.
    pub fn get(&self, key: &K) -> Option<&V> {
        let mut idx = self.primary_index(key);
        let second_hash = self.secondary_hash(key);
        let capacity_mask = self.capacity - 1;

        let mut probe_i = 0;
        loop {
            match &self.slots[idx] {
                Slot::Empty => {
                    // no key
                    return None;
                }
                Slot::Occupied(k, v) => {
                    if k == key {
                        return Some(v);
                    }
                }
                Slot::Tombstone => {
                    // keep searching
                }
            }
            probe_i += 1;
            if probe_i > self.capacity {
                return None;
            }
            idx = self.next_index(idx, second_hash, probe_i, capacity_mask);
        }
    }

    /// Retrieve a mutable reference to the value for `key`.
    pub fn get_mut(&mut self, key: &K) -> Option<&mut V> {
        let mut idx = self.primary_index(key);
        let second_hash = self.secondary_hash(key);
        let capacity_mask = self.capacity - 1;

        let mut probe_i = 0;
        loop {
            match &mut self.slots[idx] {
                Slot::Empty => {
                    return None;
                }
                Slot::Occupied(k, v) => {
                    if k == key {
                        return Some(v);
                    }
                }
                Slot::Tombstone => {
                    // keep searching
                }
            }
            probe_i += 1;
            if probe_i > self.capacity {
                return None;
            }
            idx = self.next_index(idx, second_hash, probe_i, capacity_mask);
        }
    }

    /// Removes key from the table, returning the old value if present.
    pub fn remove(&mut self, key: &K) -> Option<V> {
        let mut idx = self.primary_index(key);
        let second_hash = self.secondary_hash(key);
        let capacity_mask = self.capacity - 1;

        let mut probe_i = 0;
        loop {
            match &mut self.slots[idx] {
                Slot::Empty => {
                    // not found
                    return None;
                }
                Slot::Occupied(k, _v) => {
                    if k == key {
                        let old = match std::mem::replace(&mut self.slots[idx], Slot::Tombstone) {
                            Slot::Occupied(_, v) => v,
                            _ => unreachable!(),
                        };
                        self.len -= 1;
                        self.tombstones += 1;
                        return Some(old);
                    }
                }
                Slot::Tombstone => {
                    // keep searching
                }
            }
            probe_i += 1;
            if probe_i > self.capacity {
                return None;
            }
            idx = self.next_index(idx, second_hash, probe_i, capacity_mask);
        }
    }

    /// Clears the table of all key-value pairs.
    pub fn clear(&mut self) {
        for slot in self.slots.iter_mut() {
            *slot = Slot::Empty;
        }
        self.len = 0;
        self.tombstones = 0;
    }

    /// Returns an iterator over all (key, value) pairs in the table.
    pub fn iter(&self) -> impl Iterator<Item = (&K, &V)> {
        self.slots.iter().filter_map(|slot| match slot {
            Slot::Occupied(k, v) => Some((k, v)),
            _ => None,
        })
    }

    /// Some private helpers:

    fn load_factor_exceeded(&self) -> bool {
        (self.len as f64) / (self.capacity as f64) > self.max_load_factor
    }

    fn tombstone_exceeded(&self) -> bool {
        (self.tombstones as f64) / (self.capacity as f64) > self.tombstone_threshold
    }

    /// Grow or rehash if load factor or tombstone ratio is exceeded
    fn grow(&mut self) {
        let new_cap = (self.capacity * 2).max(2);
        self.rehash(new_cap);
    }

    /// Rebuild the table with new capacity, re-inserting all Occupied slots.
    fn rehash(&mut self, new_cap: usize) {
        let old_slots = std::mem::replace(&mut self.slots, vec![Slot::Empty; new_cap]);
        self.capacity = new_cap;
        self.len = 0;
        self.tombstones = 0;

        for slot in old_slots {
            if let Slot::Occupied(k, v) = slot {
                let _ = self.insert(k, v);
            }
        }
    }

    /// Return primary index
    fn primary_index(&self, key: &K) -> usize {
        let mut hasher = self.buildhasher1.build_hasher();
        key.hash(&mut hasher);
        let h = hasher.finish() as usize;
        h & (self.capacity - 1)
    }

    /// For double hashing, we define a second index increment. For linear/quadratic, we can ignore or define 0.
    fn secondary_hash(&self, key: &K) -> usize {
        match self.strategy {
            ProbingStrategy::DoubleHash => {
                // use buildhasher2
                let bh2 = self.buildhasher2.as_ref().unwrap();
                let mut hasher2 = bh2.build_hasher();
                key.hash(&mut hasher2);
                let h2 = hasher2.finish() as usize;
                // we must ensure the increment is not zero => do (h2 | 1) or something
                let incr = (h2 & (self.capacity - 1)) | 1;
                incr
            }
            _ => 0,
        }
    }

    /// compute next index based on strategy
    fn next_index(
        &self,
        base_idx: usize,
        second_hash: usize,
        i: usize,
        capacity_mask: usize,
    ) -> usize {
        match self.strategy {
            ProbingStrategy::Linear => (base_idx + 1) & capacity_mask,
            ProbingStrategy::Quadratic => {
                // (base_idx + i^2) mod capacity
                (base_idx + i * i) & capacity_mask
            }
            ProbingStrategy::DoubleHash => {
                // base + i*second_hash mod capacity
                (base_idx + i * second_hash) & capacity_mask
            }
        }
    }

    /// Called from `insert` if we encountered a tombstone. We attempt to see if the key is further in the chain
    /// or place new entry. Return Some(...) if we found a replaced value, or None if we placed new, or an error to bubble up?
    fn probe_find_or_place(
        &mut self,
        key: K,
        value: V,
        second_hash: usize,
        start_i: usize,
        tomb_idx: usize,
    ) -> Option<Option<V>> {
        let capacity_mask = self.capacity - 1;
        let mut i = start_i + 1;
        let mut idx = self.next_index(tomb_idx, second_hash, i, capacity_mask);
        while i <= self.capacity {
            match &mut self.slots[idx] {
                Slot::Empty => {
                    // no occupant => so key not found. Place in tomb_idx
                    self.slots[tomb_idx] = Slot::Occupied(key, value);
                    self.len += 1;
                    self.tombstones -= 1; // we used up a tombstone
                    return None;
                }
                Slot::Occupied(k2, v2) => {
                    if k2 == &key {
                        // found existing key => update and place in tomb_idx
                        let oldv = std::mem::replace(v2, value);
                        // also we can move the occupant from idx to tomb_idx?
                        // But typically for correctness, we might do a small rearrangement.
                        // Let's do a simpler approach: we simply store occupant in tomb_idx, mark idx tombstone => or store new in tomb_idx?
                        // Actually simpler: we do an approach that we should do a small "swap" approach.
                        // But that can be complicated. We'll do the simpler approach: "the same key is found => we can do an immediate update."
                        // Then we can place a tombstone or not. Actually in open addressing, we typically just do an update in place => done.
                        // But we have a tomb_idx reserved. We are in the middle of an insertion. This is a corner scenario.
                        // If the key is found further, it's simpler just to put the new value in the found slot, ignoring tomb_idx.
                        // Then we won't fill the tomb_idx.
                        // The existing slot remains Occupied, we do not reduce len or anything.
                        // We effectively do a direct update.
                        // We'll do that approach:
                        return Some(Some(oldv));
                    }
                }
                Slot::Tombstone => {
                    // keep searching
                }
            }
            i += 1;
            idx = self.next_index(tomb_idx, second_hash, i, capacity_mask);
        }
        // if we can't find an empty or occupant key after full pass => fallback rehash
        // We'll place in tomb_idx and return None
        self.slots[tomb_idx] = Slot::Occupied(key, value);
        self.len += 1;
        self.tombstones -= 1;
        None
    }
}

// Demo tests
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_ops() {
        let mut map = OpenAddressingBuilder::new()
            .with_capacity(8)
            .with_strategy(ProbingStrategy::Linear)
            .build::<String, i32>();

        assert_eq!(map.len(), 0);
        assert!(map.is_empty());

        // Insert a few
        map.insert("hello".to_string(), 1);
        map.insert("world".to_string(), 2);
        map.insert("rust".to_string(), 3);
        assert_eq!(map.len(), 3);

        // get
        assert_eq!(map.get(&"hello".to_string()), Some(&1));
        assert_eq!(map.get(&"missing".to_string()), None);

        // remove
        assert_eq!(map.remove(&"world".to_string()), Some(2));
        assert_eq!(map.get(&"world".to_string()), None);
        assert_eq!(map.len(), 2);

        // update
        let old = map.insert("hello".to_string(), 10);
        assert_eq!(old, Some(1));
        assert_eq!(map.get(&"hello".to_string()), Some(&10));

        // multiple inserts to trigger resize
        for i in 0..20 {
            map.insert(format!("key{}", i), i);
        }
        for i in 0..20 {
            assert_eq!(map.get(&format!("key{}", i)), Some(&i));
        }
    }

    #[test]
    fn double_hashing() {
        let mut map = OpenAddressingBuilder::new()
            .with_capacity(8)
            .with_strategy(ProbingStrategy::DoubleHash)
            // we must supply a second hasher
            .with_double_hasher(RandomState::new())
            .build::<i32, i32>();

        map.insert(1, 10);
        map.insert(2, 20);
        map.insert(3, 30);

        assert_eq!(map.get(&1), Some(&10));
        assert_eq!(map.get(&2), Some(&20));
        assert_eq!(map.get(&99), None);

        // remove
        let rm = map.remove(&2);
        assert_eq!(rm, Some(20));
        assert_eq!(map.get(&2), None);
    }
}
