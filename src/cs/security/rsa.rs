//! DISCLAIMER: This library is a toy example of RSA implemented in pure Rust.
//! It is *EXCLUSIVELY* for demonstration and educational purposes.
//! Absolutely DO NOT use it for real cryptographic or security-sensitive operations.
//! It is not audited, not vetted, and very likely insecure in practice.
//!
//! If you need RSA or any cryptographic operations in production, please use a
//! vetted, well-reviewed cryptography library.

use num_bigint::{BigInt, BigUint, RandPrime, ToBigInt, ToBigUint};
use num_integer::Integer;
use num_traits::{One, Zero};
use rand::{rngs::StdRng, Rng, SeedableRng};
use rand_core::RngCore;

/// Structure for an RSA public key.
#[derive(Debug, Clone)]
pub struct RSAPublicKey {
    pub n: BigUint,
    pub e: BigUint,
}

/// Structure for an RSA private key.
/// DO NOT use this for real cryptographic operations.
#[derive(Debug)]
pub struct RSAPrivateKey {
    pub n: BigUint,
    pub e: BigUint,
    pub d: BigUint,
    /// Additional factors for optimization (optional).
    pub p: BigUint,
    pub q: BigUint,
}

/// RSA KeyPair holds both public and private keys.
#[derive(Debug)]
pub struct RSAKeyPair {
    pub public_key: RSAPublicKey,
    pub private_key: RSAPrivateKey,
}

/// Configuration for RSA key generation (toy parameters).
/// DO NOT use for real cryptography.
pub struct RSAKeyGenConfig {
    /// Key size in bits (e.g. 1024, 2048).
    pub key_size: usize,
    /// Public exponent. Commonly 65537 in real usage, but let's keep it flexible.
    pub public_exponent: u64,
    /// Optional RNG seed for reproducibility in toy examples.
    pub seed: Option<u64>,
}

impl RSAKeyPair {
    /// Generate an RSA key pair with the given (TOY) configuration.
    ///
    /// # Warnings
    /// This is purely educational and insecure for production usage.
    pub fn generate(config: &RSAKeyGenConfig) -> Self {
        // Create our RNG
        let mut rng = match config.seed {
            Some(s) => StdRng::seed_from_u64(s),
            None => StdRng::from_entropy(),
        };

        // Generate two primes of ~ half key_size bits each
        let prime_bits = config.key_size / 2;
        let p = rng.gen_prime(prime_bits);
        let q = rng.gen_prime(prime_bits);

        let n = &p * &q;

        // Compute phi(n) = (p - 1)(q - 1)
        let phi = (p.clone() - BigUint::one()) * (q.clone() - BigUint::one());

        let e = BigUint::from(config.public_exponent);
        // Compute d = e^-1 mod phi(n)
        let d =
            mod_inverse(&e, &phi).expect("Could not find modular inverse; invalid e or primes?");

        let public_key = RSAPublicKey {
            n: n.clone(),
            e: e.clone(),
        };
        let private_key = RSAPrivateKey {
            n: n.clone(),
            e: e.clone(),
            d: d.clone(),
            p,
            q,
        };
        RSAKeyPair {
            public_key,
            private_key,
        }
    }
}

/// RSA Encrypt using the public key:
/// ciphertext = (plaintext^e) mod n
///
/// *This is purely for demonstration. DO NOT use in real systems.*
pub fn rsa_encrypt(public_key: &RSAPublicKey, plaintext: &BigUint) -> BigUint {
    plaintext.modpow(&public_key.e, &public_key.n)
}

/// RSA Decrypt using the private key:
/// plaintext = (ciphertext^d) mod n
///
/// *This is purely for demonstration. DO NOT use in real systems.*
pub fn rsa_decrypt(private_key: &RSAPrivateKey, ciphertext: &BigUint) -> BigUint {
    ciphertext.modpow(&private_key.d, &private_key.n)
}

/// RSA Sign using the private key:
/// signature = (message_hash^d) mod n
///
/// Typically, you'd apply a padding scheme like PSS. This is just raw RSA exponentiation
/// to show the concept. DO NOT use in real systems.
pub fn rsa_sign(private_key: &RSAPrivateKey, message_hash: &BigUint) -> BigUint {
    message_hash.modpow(&private_key.d, &private_key.n)
}

/// RSA Verify using the public key:
/// verified_hash = (signature^e) mod n
///
/// Compare `verified_hash` to the original `message_hash` to confirm authenticity.
/// DO NOT use in real systems without a proper padding/PKCS scheme.
pub fn rsa_verify(public_key: &RSAPublicKey, signature: &BigUint) -> BigUint {
    signature.modpow(&public_key.e, &public_key.n)
}

/// Finds the modular inverse of `a` modulo `m` using the Extended Euclidean Algorithm.
/// Returns `Some(x)` where x satisfies (a*x) mod m = 1, or `None` if no inverse exists.
/// DO NOT rely on this for real cryptography.
fn mod_inverse(a: &BigUint, m: &BigUint) -> Option<BigUint> {
    let (g, x, _) = extended_gcd(a, m);
    if g.is_one() {
        // (x mod m) is the inverse
        let x_mod_m = if x.is_negative() {
            // (m - (|x| mod m))
            let neg_x = (-x).to_biguint().unwrap();
            m - (&neg_x % m)
        } else {
            x.to_biguint().unwrap() % m
        };
        Some(x_mod_m)
    } else {
        None
    }
}

/// Extended Euclidean Algorithm in BigInts.
/// Returns (gcd(a, b), x, y) such that a*x + b*y = gcd(a,b).
/// DO NOT rely on this for real cryptography.
fn extended_gcd(a: &BigUint, b: &BigUint) -> (BigInt, BigInt, BigInt) {
    if b.is_zero() {
        return (a.to_bigint().unwrap(), BigInt::one(), BigInt::zero());
    }
    let (quotient, remainder) = a.div_mod_floor(b);
    let (g, x1, y1) = extended_gcd(b, &remainder);
    let x = &y1 - &quotient.to_bigint().unwrap() * &x1;
    let y = x1;
    (g, x, y)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_toy_rsa() {
        // Generate an extremely small RSA key (only 512 bits) for quick testing.
        // This is STILL only for demonstration and not secure at all.
        let config = RSAKeyGenConfig {
            key_size: 512,
            public_exponent: 65537,
            seed: Some(42),
        };
        let keypair = RSAKeyPair::generate(&config);

        // Basic check: encrypt/decrypt a small message
        let msg = BigUint::from(123456789_u64);
        let enc = rsa_encrypt(&keypair.public_key, &msg);
        let dec = rsa_decrypt(&keypair.private_key, &enc);

        assert_eq!(dec, msg, "RSA encryption/decryption mismatch");

        // Basic sign/verify check
        let hash = BigUint::from(987654321_u64);
        let sig = rsa_sign(&keypair.private_key, &hash);
        let recovered = rsa_verify(&keypair.public_key, &sig);
        assert_eq!(recovered, hash, "RSA sign/verify mismatch");
    }
}
