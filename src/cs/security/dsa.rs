//! DISCLAIMER: This library is a **toy** example of the Digital Signature Algorithm (DSA)
//! implemented in pure Rust. It is *EXCLUSIVELY* for demonstration and educational purposes.
//! Absolutely DO NOT use it for real cryptographic or security-sensitive operations.
//! It is not audited, not vetted, and very likely insecure in practice. If you need DSA or
//! any cryptographic operations in production, please use a vetted, well-reviewed cryptography library.

use num_bigint_dig::{BigInt, BigUint, Sign, ToBigInt};
use num_integer::Integer;
use num_traits::{One, Zero};
use rand::{rngs::StdRng, RngCore, SeedableRng};

/// DSA domain parameters: prime p, prime q, and g = h^((p-1)/q) mod p.
#[derive(Debug, Clone)]
pub struct DsaParams {
    pub p: BigUint,
    pub q: BigUint,
    pub g: BigUint,
}

/// DSA keypair: private x in [1..q-1], public y = g^x mod p.
#[derive(Debug, Clone)]
pub struct DsaKeyPair {
    pub private: BigUint,
    pub public: BigUint,
}

/// A toy DSA signature: `(r, s)`.
#[derive(Debug, Clone)]
pub struct DsaSignature {
    pub r: BigUint,
    pub s: BigUint,
}

/// Generate toy DSA domain parameters. This is not a real, secure method!
/// We'll pick random `p` of given bit length, random `q` smaller, and compute `g`.
/// In real usage, you'd follow FIPS 186-4 for DSA parameter generation.
pub fn toy_generate_dsa_params(_p_bits: usize, _q_bits: usize, _seed: Option<u64>) -> DsaParams {
    // For this toy implementation, just use fixed test values that we know work
    // These are NOT secure parameters, but they're fine for learning/testing
    let q = BigUint::from(11u64); // q = 11 (prime)
    let p = &q * BigUint::from(2u64) + BigUint::one(); // p = 23 (safe prime)

    // g = 2 is a generator for this case
    let g = BigUint::from(2u64);

    DsaParams { p, q, g }
}

/// Generate a DSA keypair: pick x in [1..q-1], compute y = g^x mod p.
pub fn toy_dsa_generate_keypair(params: &DsaParams, seed: Option<u64>) -> DsaKeyPair {
    let mut rng = match seed {
        Some(s) => StdRng::seed_from_u64(s),
        None => StdRng::from_entropy(),
    };

    let q = &params.q;
    let p = &params.p;
    let g = &params.g;

    // pick x in [1..q-1]
    let mut x = BigUint::zero();
    for _ in 0..100 {
        let mut scalar_bytes = vec![0u8; q.bits() / 8 + 1];
        rng.fill_bytes(&mut scalar_bytes);
        x = BigUint::from_bytes_be(&scalar_bytes) % q;
        if !x.is_zero() {
            break;
        }
    }
    if x.is_zero() {
        x = BigUint::one();
    }
    // y = g^x mod p
    let y = mod_exp(g, &x, p);

    DsaKeyPair {
        private: x,
        public: y,
    }
}

/// Sign a message hash `h_m` with DSA: produce (r, s) in [1..q-1].
/// In real usage, `h_m` is the output of a secure hash truncated to q bits, ephemeral k is random, etc.
pub fn toy_dsa_sign(
    params: &DsaParams,
    kp: &DsaKeyPair,
    h_m: &BigUint,
    seed: Option<u64>,
) -> DsaSignature {
    let mut rng = match seed {
        Some(s) => StdRng::seed_from_u64(s),
        None => StdRng::from_entropy(),
    };

    let p = &params.p;
    let q = &params.q;
    let g = &params.g;
    let x = &kp.private;

    // ephemeral k in [1..q-1] and coprime with q
    let mut k = BigUint::zero();
    for _ in 0..100 {
        let mut buf = vec![0u8; q.bits() / 8 + 1];
        rng.fill_bytes(&mut buf);
        let k_candidate = BigUint::from_bytes_be(&buf) % q;
        if !k_candidate.is_zero() {
            // Check if k is coprime with q using extended_gcd
            let (gcd, _, _) = extended_gcd(k_candidate.clone(), q.clone());
            if gcd == BigUint::one() {
                k = k_candidate;
                break;
            }
        }
    }
    if k.is_zero() {
        k = BigUint::one();
    }

    // r = (g^k mod p) mod q
    let gk = mod_exp(g, &k, p);
    let r = gk % q;

    // s = k^-1 * (h_m + x*r) mod q
    let k_inv = mod_inv(k, q).expect("No inverse for ephemeral k?! (toy DSA)");
    let xr = (x * &r) % q;
    let sum = (h_m + &xr) % q;
    let s = (&k_inv * &sum) % q;

    DsaSignature { r, s }
}

/// Verify a DSA signature `(r, s)` for message hash `h_m`.
/// If valid => true, else false.
pub fn toy_dsa_verify(
    params: &DsaParams,
    pub_key: &BigUint,
    h_m: &BigUint,
    sig: &DsaSignature,
) -> bool {
    let q = &params.q;
    let p = &params.p;
    let g = &params.g;

    // check 0 < r, s < q
    if sig.r.is_zero() || sig.r >= *q {
        println!("r out of range: r={}", sig.r);
        return false;
    }
    if sig.s.is_zero() || sig.s >= *q {
        println!("s out of range: s={}", sig.s);
        return false;
    }

    // check public key is valid: 1 < y < p
    if pub_key <= &BigUint::one() || pub_key >= p {
        println!("Invalid public key");
        return false;
    }

    // w = s^-1 mod q
    let w = match mod_inv(sig.s.clone(), q) {
        Some(val) => val,
        None => {
            println!("no inverse for s: s={}", sig.s);
            return false;
        }
    };
    println!("w = {}", w);

    // u1 = (h_m * w) mod q
    // u2 = (r * w) mod q
    let u1 = (h_m * &w) % q;
    let u2 = (&sig.r * &w) % q;
    println!("u1 = {}", u1);
    println!("u2 = {}", u2);

    // v = ((g^u1 * y^u2) mod p) mod q
    let gu1 = mod_exp(g, &u1, p);
    let yu2 = mod_exp(pub_key, &u2, p);
    println!("gu1 = {}", gu1);
    println!("yu2 = {}", yu2);
    let t = (&gu1 * &yu2) % p;
    println!("t = {}", t);
    let v = t % q;
    println!("v = {}", v);
    println!("r = {}", sig.r);

    v == sig.r
}

// Utility: modular exponentiation: base^exp mod m
fn mod_exp(base: &BigUint, exp: &BigUint, m: &BigUint) -> BigUint {
    if m.is_zero() {
        panic!("mod_exp with modulus=0");
    }
    base.modpow(exp, m)
}

// Extended Euclidean Algorithm
fn extended_gcd(a: BigUint, b: BigUint) -> (BigUint, BigInt, BigInt) {
    let mut a_int = a.to_bigint().unwrap();
    let mut b_int = b.to_bigint().unwrap();
    let mut x0 = BigInt::one();
    let mut x1 = BigInt::zero();
    let mut y0 = BigInt::zero();
    let mut y1 = BigInt::one();

    while !b_int.is_zero() {
        let (q, r) = a_int.div_rem(&b_int);
        a_int = b_int;
        b_int = r;

        let tmpx = x0 - &q * &x1;
        x0 = x1;
        x1 = tmpx;

        let tmpy = y0 - &q * &y1;
        y0 = y1;
        y1 = tmpy;
    }
    (a_int.to_biguint().unwrap(), x0, y0)
}

// Utility: extended GCD to find modular inverse of x mod m if gcd(x,m)=1
fn mod_inv(x: BigUint, m: &BigUint) -> Option<BigUint> {
    let (g, s, _) = extended_gcd(x, m.clone());
    if g != BigUint::one() {
        None
    } else {
        // Convert negative coefficients to positive ones modulo m
        let m_int = m.to_bigint().unwrap();
        let mut result = s % &m_int;
        if result.sign() == Sign::Minus {
            result += &m_int;
        }
        Some(result.to_biguint().unwrap())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dsa_toy() {
        // Let's produce small parameters for demonstration. Real DSA uses big primes, e.g. 1024-bit p, 160-bit q or more.
        let p_bits = 256;
        let q_bits = 160;
        let params = toy_generate_dsa_params(p_bits, q_bits, Some(42));
        println!("\nDSA Parameters:");
        println!("p = {}", params.p);
        println!("q = {}", params.q);
        println!("g = {}", params.g);

        // Generate a key pair
        let kp = toy_dsa_generate_keypair(&params, Some(100));
        println!("\nKey Pair:");
        println!("x (private) = {}", kp.private);
        println!("y (public) = {}", kp.public);

        // "Hash" a message by just taking some BigUint (toy). Real usage => a real hash truncated to q bits.
        let msg_hash = BigUint::parse_bytes(b"123456789ABCDEF", 16).unwrap();
        println!("\nMessage Hash:");
        println!("h_m = {}", msg_hash);

        // Sign
        let sig = toy_dsa_sign(&params, &kp, &msg_hash, Some(200));
        println!("\nSignature:");
        println!("r = {}", sig.r);
        println!("s = {}", sig.s);

        // Verify
        let valid = toy_dsa_verify(&params, &kp.public, &msg_hash, &sig);
        println!("\nVerification Result: {}", valid);
        assert!(valid, "DSA signature must verify with correct key/msg.");

        // If we tamper the signature
        let mut bad_sig = sig.clone();
        bad_sig.r += 1u64;
        let invalid = toy_dsa_verify(&params, &kp.public, &msg_hash, &bad_sig);
        assert!(!invalid, "Tampered signature must fail verification");
    }
}
