//! DISCLAIMER: This library is a toy example of Diffie-Hellman Key Exchange in pure Rust.
//! It is *EXCLUSIVELY* for demonstration and educational purposes.
//! Absolutely DO NOT use it for real cryptographic or security-sensitive operations.
//! It is not audited, not vetted, and very likely insecure in practice.
//!
//! If you need Diffie-Hellman or any cryptographic operations in production, please use a
//! vetted, well-reviewed cryptography library.

use num_bigint::{BigUint, RandPrime};
use num_traits::{One, Zero};
use rand::{rngs::StdRng, Rng, SeedableRng};

/// DiffieHellmanParams holds the large prime `p` and generator `g`.
/// In a real system, these should be carefully chosen and possibly validated safe primes.
///
/// *This is for demonstration only. DO NOT use in real systems.*
#[derive(Debug, Clone)]
pub struct DiffieHellmanParams {
    /// A large prime modulus.
    pub p: BigUint,
    /// A generator base (also sometimes called `g`).
    pub g: BigUint,
}

/// An ephemeral keypair for Diffie-Hellman:
/// - `private_key`: a random integer `a` in `[1, p-1]`.
/// - `public_key`: `g^a mod p`.
///
/// *This is for demonstration only. DO NOT use in real systems.*
#[derive(Debug, Clone)]
pub struct DiffieHellmanKeyPair {
    /// The prime modulus, same as in `DiffieHellmanParams`.
    pub p: BigUint,
    /// The generator, same as in `DiffieHellmanParams`.
    pub g: BigUint,
    /// The private key exponent (`a`).
    pub private_key: BigUint,
    /// The corresponding public value (`A = g^a mod p`).
    pub public_key: BigUint,
}

/// Configuration to generate toy Diffie-Hellman parameters.
pub struct DHParamsConfig {
    /// The bit length of the prime `p`.
    pub prime_bits: usize,
    /// Optional RNG seed for reproducibility in toy examples.
    pub seed: Option<u64>,
}

/// Configuration for ephemeral key generation given a set of DH parameters.
pub struct DHKeyGenConfig {
    /// Optional RNG seed for reproducibility in toy examples.
    pub seed: Option<u64>,
}

impl DiffieHellmanParams {
    /// Generate toy Diffie-Hellman parameters (prime `p` and generator `g`).
    /// 
    /// # Warnings
    /// - This is a TOY function that picks a prime of the requested size but does no
    ///   advanced primality checks beyond `num-bigint`'s prime generation.
    /// - No validation for safe prime or generator is done. This is purely illustrative.
    pub fn generate(config: &DHParamsConfig) -> Self {
        // Create our RNG
        let mut rng = match config.seed {
            Some(s) => StdRng::seed_from_u64(s),
            None => StdRng::from_entropy(),
        };

        // Generate a random prime of `prime_bits` length
        let p = rng.gen_prime(config.prime_bits);
        
        // For demonstration, pick a small g:
        // In real usage, g must be carefully chosen. We'll do something naive here.
        let g = BigUint::from(2_u64);

        DiffieHellmanParams { p, g }
    }

    /// Create ephemeral key pair for a user, with random private key `a` in [1, p-1].
    pub fn generate_keypair(&self, config: &DHKeyGenConfig) -> DiffieHellmanKeyPair {
        let mut rng = match config.seed {
            Some(s) => StdRng::seed_from_u64(s),
            None => StdRng::from_entropy(),
        };

        // Private key `a` is a random integer in [1, p-1].
        // For simplicity, we pick a random `a` near the bit size of p.
        let bitlen = self.p.bits();
        let mut a = BigUint::zero();
        // ensure a is not 0
        while a.is_zero() {
            a = rng.gen_biguint(bitlen);
            if a >= self.p.clone() {
                // reduce mod p-1 or something
                a = &a % (&self.p - BigUint::one());
            }
        }

        // compute public key = g^a mod p
        let pubkey = self.g.modpow(&a, &self.p);

        DiffieHellmanKeyPair {
            p: self.p.clone(),
            g: self.g.clone(),
            private_key: a,
            public_key: pubkey,
        }
    }
}

impl DiffieHellmanKeyPair {
    /// Given another party's public key `other_pub`, compute the shared secret:
    ///   `S = other_pub^a mod p`.
    /// 
    /// # Warnings
    /// - This is naive exponentiation with no safety checks for malicious inputs.
    /// - No key derivation function is applied on top. It's purely the raw group element.
    pub fn compute_shared_secret(&self, other_pub: &BigUint) -> BigUint {
        other_pub.modpow(&self.private_key, &self.p)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_toy_diffie_hellman() {
        // Generate toy parameters: prime ~ 256 bits
        let dh_config = DHParamsConfig {
            prime_bits: 256,
            seed: Some(42),
        };
        let dh_params = DiffieHellmanParams::generate(&dh_config);
        
        // Alice keypair
        let alice_config = DHKeyGenConfig {
            seed: Some(100),
        };
        let alice = dh_params.generate_keypair(&alice_config);

        // Bob keypair
        let bob_config = DHKeyGenConfig {
            seed: Some(200),
        };
        let bob = dh_params.generate_keypair(&bob_config);

        // Each side computes shared secret
        let alice_secret = alice.compute_shared_secret(&bob.public_key);
        let bob_secret = bob.compute_shared_secret(&alice.public_key);

        // Check if they match
        assert_eq!(alice_secret, bob_secret, "Diffie-Hellman secrets must match");
    }
}
