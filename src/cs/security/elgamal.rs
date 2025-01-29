//! DISCLAIMER: This library is a toy example of ElGamal Encryption in pure Rust.
//! It is *EXCLUSIVELY* for demonstration and educational purposes.
//! Absolutely DO NOT use it for real cryptographic or security-sensitive operations.
//! It is not audited, not vetted, and very likely insecure in practice.
//!
//! If you need ElGamal or any cryptographic operations in production, please use a
//! vetted, well-reviewed cryptography library.

use num_bigint_dig::{BigUint, RandBigInt, RandPrime};
use num_traits::{One, Zero};
use rand::{rngs::StdRng, SeedableRng};

/// ElGamal parameters: a large prime `p` and a generator `g`.
/// In actual practice, these parameters must be carefully selected and validated.
///
/// *This is for demonstration only. DO NOT use in real systems.*
#[derive(Debug, Clone)]
pub struct ElGamalParams {
    /// A large prime modulus.
    pub p: BigUint,
    /// A generator for the multiplicative group modulo `p`.
    pub g: BigUint,
}

/// A structure holding the ElGamal public key:
/// - `p` and `g` from the system parameters.
/// - `y = g^x mod p`, where `x` is the secret exponent.
///
/// *This is for demonstration only. DO NOT use in real systems.*
#[derive(Debug, Clone)]
pub struct ElGamalPublicKey {
    pub p: BigUint,
    pub g: BigUint,
    pub y: BigUint,
}

/// A structure holding the ElGamal private key:
/// - the same `p`, `g`,
/// - plus the secret exponent `x`.
///
/// *This is for demonstration only. DO NOT use in real systems.*
#[derive(Debug)]
pub struct ElGamalPrivateKey {
    pub p: BigUint,
    pub g: BigUint,
    pub x: BigUint,
}

/// Combined keypair, storing both public and private halves together.
/// This can be split if needed.
#[derive(Debug)]
pub struct ElGamalKeyPair {
    pub public: ElGamalPublicKey,
    pub private: ElGamalPrivateKey,
}

/// A ciphertext in ElGamal encryption consists of two values, (c1, c2).
///
/// *This is for demonstration only. DO NOT use in real systems.*
#[derive(Debug, Clone)]
pub struct ElGamalCiphertext {
    pub c1: BigUint,
    pub c2: BigUint,
}

/// Configuration for generating ElGamal parameters. This is purely a toy example.
pub struct ElGamalParamsConfig {
    /// The bit length of the prime `p`.
    pub prime_bits: usize,
    /// Optional RNG seed for reproducibility in toy examples.
    pub seed: Option<u64>,
}

/// Configuration for keypair generation given certain ElGamal parameters.
pub struct ElGamalKeyGenConfig {
    /// Optional RNG seed for reproducibility in toy examples.
    pub seed: Option<u64>,
}

/// Configuration for encryption (toy ephemeral exponent).
/// Typically you want a random ephemeral `k`.
pub struct ElGamalEncryptConfig {
    /// Optional RNG seed for reproducibility in toy examples.
    pub seed: Option<u64>,
}

impl ElGamalParams {
    /// Generate toy ElGamal parameters with prime `p` of size `prime_bits`.
    /// We pick a trivial generator `g = 2` here, with zero validation, for demonstration.
    ///
    /// # Warnings
    /// - No advanced prime checks or generator validation is performed.
    /// - DO NOT USE FOR REAL CRYPTOGRAPHY.
    pub fn generate(config: &ElGamalParamsConfig) -> Self {
        // Create RNG
        let mut rng = match config.seed {
            Some(s) => StdRng::seed_from_u64(s),
            None => StdRng::from_entropy(),
        };

        // Generate a random prime ~ `prime_bits` in length.
        let p = rng.gen_prime(config.prime_bits);
        let g = BigUint::from(2_u64); // naive generator

        ElGamalParams { p, g }
    }

    /// Generate an ElGamal keypair: pick private x in [1, p-1], public y = g^x mod p.
    ///
    /// # Warnings
    /// - This is a toy function. No advanced checks or safe primes.
    /// - DO NOT USE FOR REAL CRYPTOGRAPHY.
    pub fn generate_keypair(&self, config: &ElGamalKeyGenConfig) -> ElGamalKeyPair {
        let mut rng = match config.seed {
            Some(s) => StdRng::seed_from_u64(s),
            None => StdRng::from_entropy(),
        };

        let bitlen = self.p.bits();
        let mut x = BigUint::zero();
        // ensure x != 0
        while x.is_zero() {
            x = rng.gen_biguint(bitlen);
            if x >= self.p.clone() {
                // reduce mod p-1 or something
                x = &x % (&self.p - BigUint::one());
            }
        }

        let y = self.g.modpow(&x, &self.p);

        let public = ElGamalPublicKey {
            p: self.p.clone(),
            g: self.g.clone(),
            y,
        };
        let private = ElGamalPrivateKey {
            p: self.p.clone(),
            g: self.g.clone(),
            x,
        };

        ElGamalKeyPair { public, private }
    }
}

/// Encrypt a message `m` using ElGamal public key and ephemeral exponent `k`.
///
/// # Arguments
/// - `public_key`: the ElGamal public key (p, g, y).
/// - `message`: a `BigUint` representing the message in [1, p-1].
/// - `config`: ephemeral random for k (toy).
///
/// # Returns
/// An `ElGamalCiphertext` (c1, c2) where:
/// c1 = g^k mod p
/// c2 = m * y^k mod p
///
/// # Warnings
/// - This is a raw ElGamal approach with no padding or advanced encoding.
/// - `message` must be < p.
/// - DO NOT USE FOR REAL CRYPTOGRAPHY.
pub fn elgamal_encrypt(
    public_key: &ElGamalPublicKey,
    message: &BigUint,
    config: &ElGamalEncryptConfig,
) -> ElGamalCiphertext {
    // ephemeral exponent k
    let mut rng = match config.seed {
        Some(s) => StdRng::seed_from_u64(s),
        None => StdRng::from_entropy(),
    };

    let bitlen = public_key.p.bits();
    let mut k = BigUint::zero();
    while k.is_zero() {
        k = rng.gen_biguint(bitlen);
        // reduce mod p-1 or something
        if k >= public_key.p.clone() {
            k = &k % (&public_key.p - BigUint::one());
        }
    }

    // c1 = g^k mod p
    let c1 = public_key.g.modpow(&k, &public_key.p);
    // c2 = message * y^k mod p
    let yk = public_key.y.modpow(&k, &public_key.p);
    let c2 = (message * &yk) % &public_key.p;

    ElGamalCiphertext { c1, c2 }
}

/// Decrypt an ElGamal ciphertext (c1, c2) using the private exponent x.
///
/// # Returns
/// The message = c2 * (c1^x)^(-1) mod p, or equivalently c2 * c1^(p-1 - x).
///
/// # Warnings
/// - Raw ElGamal decryption with no checks or side-channel protections.
/// - DO NOT USE FOR REAL CRYPTOGRAPHY.
pub fn elgamal_decrypt(private_key: &ElGamalPrivateKey, ciphertext: &ElGamalCiphertext) -> BigUint {
    // s = c1^x mod p
    let s = ciphertext.c1.modpow(&private_key.x, &private_key.p);
    // s_inv = s^(p-2) mod p (Fermat's little theorem) if p is prime
    let s_inv = s.modpow(
        &(private_key.p.clone() - BigUint::one() - BigUint::one()),
        &private_key.p,
    );

    // message = c2 * s_inv mod p
    (&ciphertext.c2 * &s_inv) % &private_key.p
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_toy_elgamal() {
        // Generate small parameters: prime ~256 bits
        let params_cfg = ElGamalParamsConfig {
            prime_bits: 256,
            seed: Some(42),
        };
        let params = ElGamalParams::generate(&params_cfg);

        // Keypair
        let key_cfg = ElGamalKeyGenConfig { seed: Some(100) };
        let keypair = params.generate_keypair(&key_cfg);

        // Prepare a message
        let message = BigUint::from(123456789_u64);
        // Check that message < p
        assert!(message < keypair.public.p, "Message must be < p");

        // Encrypt
        let enc_cfg = ElGamalEncryptConfig { seed: Some(200) };
        let ciphertext = elgamal_encrypt(&keypair.public, &message, &enc_cfg);

        // Decrypt
        let recovered = elgamal_decrypt(&keypair.private, &ciphertext);

        assert_eq!(recovered, message, "ElGamal encryption/decryption mismatch");
    }
}
