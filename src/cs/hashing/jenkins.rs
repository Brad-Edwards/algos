//! # Jenkins "One-At-A-Time" Hash
//!
//! This module provides a **production-oriented** implementation of the Jenkins "One-At-A-Time" hash function for **32-bit** results.
//! It's a simple, fast, non-cryptographic hash commonly used in hash tables or checksums where collisions are not a major concern
//! or are well-managed by the container itself.
//!
//! **Note**: This hash is not cryptographically secure; it's meant for general-purpose hashing in data structures or indexing.
//!
//! ## Key Points
//! - **32-bit** internal state, but can integrate with `std::hash::Hasher` (which requires `finish() -> u64`;
//!   we zero-extend the 32-bit result to 64 bits).
//! - A `Builder` pattern to create a `BuildHasher` for integrating with standard library `HashMap` or `HashSet`.
//! - One-shot function `jenkins_hash(data) -> u32` for direct usage.

use std::hash::{BuildHasher, Hasher};

/// A one-shot function computing Jenkins's one-at-a-time hash for a `data` slice, returning a 32-bit hash.
pub fn jenkins_hash(data: &[u8]) -> u32 {
    let mut h: u32 = 0;
    for &c in data {
        h = h.wrapping_add(c as u32);
        h = h.wrapping_add(h << 10);
        h ^= h >> 6;
    }
    h = h.wrapping_add(h << 3);
    h ^= h >> 11;
    h = h.wrapping_add(h << 15);
    h
}

/// A builder that configures how to create Jenkins hashers.
///
/// Currently Jenkins is quite simple, so there aren't many parameters to tweak:
/// It's basically the standard one-at-a-time approach. We might allow advanced
/// seeding or variations in the future.
#[derive(Debug, Clone, Default)]
pub struct JenkinsBuilder {
    /// If you'd like a "seed" to perturb the initial state, you can store it here.
    /// By default, we do no seed (0).
    pub seed: u32,
}

impl JenkinsBuilder {
    /// Creates a new Jenkins builder with default seed=0.
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets a "seed" for the hash. This modifies the initial state used by the hasher,
    /// changing the final hash distribution.
    pub fn with_seed(mut self, seed: u32) -> Self {
        self.seed = seed;
        self
    }

    /// Produces the `BuildHasher` object.
    pub fn build(self) -> JenkinsBuildHasher {
        JenkinsBuildHasher { seed: self.seed }
    }
}

/// The `BuildHasher` object that can create `JenkinsHasher` for usage in
/// `std::collections::HashMap` or others.
#[derive(Debug, Clone)]
pub struct JenkinsBuildHasher {
    seed: u32,
}

impl BuildHasher for JenkinsBuildHasher {
    type Hasher = JenkinsHasher;

    fn build_hasher(&self) -> Self::Hasher {
        JenkinsHasher::with_seed(self.seed)
    }
}

/// The Jenkins one-at-a-time Hasher implementing `std::hash::Hasher`.
/// It accumulates a 32-bit state, but returns 64 bits by zero-extending in `finish()`.
#[derive(Debug, Clone)]
pub struct JenkinsHasher {
    state: u32,
}

impl JenkinsHasher {
    /// Creates a new hasher with default initial state 0.
    pub fn new() -> Self {
        JenkinsHasher { state: 0 }
    }

    /// Creates a new hasher with a specific `seed`.
    /// This modifies the initial state, effectively injecting a "seed" into the final result.
    pub fn with_seed(seed: u32) -> Self {
        JenkinsHasher { state: seed }
    }

    /// Returns current 32-bit state (not final).
    pub fn current_jenkins(&self) -> u32 {
        self.state
    }
}

impl Hasher for JenkinsHasher {
    fn finish(&self) -> u64 {
        // We'll finalize the Jenkins steps if needed.
        // However, one-at-a-time typically is done after the final data.
        // We'll do a partial approach so that if the user never wrote anything,
        // we still produce a consistent result.
        // We might do "final mix" steps here if we haven't.
        // But in the Jenkins approach, we do the final mix after each chunk.
        // The typical approach is "One-At-A-Time" is a single pass,
        // but we can finalize now.
        // The user might have partial data...
        // We'll do a final step or we can do partial after each write.
        // Actually the standard approach is to do the final mixing only once at the end:
        let mut h = self.state;
        // We'll apply the final mixing if we didn't do it in `write`.
        // But we have been doing partial mixing after each byte in one-at-a-time.
        // Actually the "One-At-A-Time" described is the entire process.
        // We'll do that after each byte. So there's a final mixing step that is part of the last lines:
        // "h += h << 3; h ^= h >> 11; h += h << 15;".
        // We do it after all bytes in the standard approach.
        // But here we do it in `write`. We'll do it in the finish.
        // Actually the reference code says:
        //   for each byte:
        //      h += c
        //      h += h<<10
        //      h ^= h>>6
        //   h += h<<3
        //   h ^= h>>11
        //   h += h<<15
        // So the final lines are outside the loop.
        // Means we haven't done that yet if we do partial after each byte.
        // We'll do the final lines now:
        h = h.wrapping_add(h << 3);
        h ^= h >> 11;
        h = h.wrapping_add(h << 15);

        // zero-extend to 64
        h as u64
    }

    fn write(&mut self, bytes: &[u8]) {
        // We'll do partial approach for each byte, but not the final lines.
        for &b in bytes {
            self.state = self.state.wrapping_add(b as u32);
            self.state = self.state.wrapping_add(self.state << 10);
            self.state ^= self.state >> 6;
        }
    }
}

impl Default for JenkinsHasher {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_jenkins_hash_function() {
        let h1 = jenkins_hash(b"hello");
        let h2 = jenkins_hash(b"hello");
        assert_eq!(h1, h2);
        assert_ne!(h1, 0);

        let h3 = jenkins_hash(b"Hello");
        assert_ne!(h1, h3);
    }

    #[test]
    fn test_hasher() {
        let mut hasher = JenkinsHasher::new();
        hasher.write(b"abc");
        let val1 = hasher.finish();

        let h2 = jenkins_hash(b"abc");
        // the finish does the final lines => h2 will match val1
        assert_eq!(val1, h2 as u64);
    }

    #[test]
    fn test_with_seed() {
        let mut hasher = JenkinsHasher::with_seed(12345);
        hasher.write(b"abc");
        let val_seeded = hasher.finish();

        let val_no_seed = jenkins_hash(b"abc") as u64;
        assert_ne!(val_seeded, val_no_seed);
    }

    #[test]
    fn test_builder_in_hashmap() {
        use std::collections::HashMap;
        let buildh = JenkinsBuilder::new().with_seed(999).build();
        let mut map = HashMap::with_hasher(buildh);
        map.insert("foo", 1);
        map.insert("bar", 2);
        assert_eq!(map["foo"], 1);
    }
}
