//! DISCLAIMER: This library is a toy example of RSA implemented in pure Rust.
//! It is *EXCLUSIVELY* for demonstration and educational purposes.
//! Absolutely DO NOT use it for real cryptographic or security-sensitive operations.
//! It is not audited, not vetted, and very likely insecure in practice.
//!
//! If you need RSA or any cryptographic operations in production, please use a
//! vetted, well-reviewed cryptography library.

use num_bigint_dig::{BigInt, BigUint, RandPrime, Sign, ToBigInt};
use num_integer::Integer;
use num_traits::{One, Zero};
use rand::{rngs::StdRng, SeedableRng};

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

        // Keep generating primes until we find suitable ones
        let mut p: BigUint;
        let mut q: BigUint;
        let mut phi: BigUint;
        let e = BigUint::from(config.public_exponent);

        loop {
            p = RandPrime::gen_prime(&mut rng, prime_bits);
            q = RandPrime::gen_prime(&mut rng, prime_bits);

            // Ensure p != q
            if p == q {
                continue;
            }

            // Compute phi(n) = (p - 1)(q - 1)
            phi = (p.clone() - BigUint::one()) * (q.clone() - BigUint::one());

            // Check if e and phi are coprime
            if mod_inverse(&e, &phi).is_some() {
                break;
            }
        }

        let n = &p * &q;
        let d = mod_inverse(&e, &phi).unwrap();

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
    let a_int = a.to_bigint().unwrap();
    let m_int = m.to_bigint().unwrap();
    let (g, x, _) = extended_gcd(&a_int, &m_int);
    if g.is_one() {
        // Make sure we return a positive value in [0, m-1]
        let mut result = x % &m_int;
        if result.sign() == Sign::Minus {
            result += &m_int;
        }
        Some(result.to_biguint().unwrap())
    } else {
        None
    }
}

/// Extended Euclidean Algorithm in BigInts.
/// Returns (gcd(a, b), x, y) such that a*x + b*y = gcd(a,b).
/// DO NOT rely on this for real cryptography.
fn extended_gcd(a: &BigInt, b: &BigInt) -> (BigInt, BigInt, BigInt) {
    if b.is_zero() {
        (a.clone(), BigInt::one(), BigInt::zero())
    } else {
        let (q, r) = a.div_rem(b);
        let (g, x, y) = extended_gcd(b, &r);
        (g, y.clone(), x - &q * y)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_toy_rsa() {
        // Generate an extremely small RSA key for quick testing.
        // This is STILL only for demonstration and not secure at all.
        let config = RSAKeyGenConfig {
            key_size: 128, // Use small but sufficient key for test
            public_exponent: 65537,
            seed: Some(42),
        };
        let keypair = RSAKeyPair::generate(&config);

        // Basic check: encrypt/decrypt a small message
        let msg = BigUint::from(42u64);
        let enc = rsa_encrypt(&keypair.public_key, &msg);

        // Verify message is smaller than modulus
        assert!(
            msg < keypair.public_key.n,
            "Message must be smaller than modulus"
        );

        let dec = rsa_decrypt(&keypair.private_key, &enc);
        assert_eq!(dec, msg, "RSA encryption/decryption mismatch");

        // Basic sign/verify check
        let hash = BigUint::from(24u64);
        // Verify hash is smaller than modulus
        assert!(
            hash < keypair.public_key.n,
            "Hash must be smaller than modulus"
        );

        let sig = rsa_sign(&keypair.private_key, &hash);
        let recovered = rsa_verify(&keypair.public_key, &sig);
        assert_eq!(recovered, hash, "RSA sign/verify mismatch");
    }
}
