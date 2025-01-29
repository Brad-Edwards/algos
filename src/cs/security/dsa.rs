//! DISCLAIMER: This library is a **toy** example of the Digital Signature Algorithm (DSA)
//! implemented in pure Rust. It is *EXCLUSIVELY* for demonstration and educational purposes.
//! Absolutely DO NOT use it for real cryptographic or security-sensitive operations.
//! It is not audited, not vetted, and very likely insecure in practice. If you need DSA or
//! any cryptographic operations in production, please use a vetted, well-reviewed cryptography library.

use num_bigint_dig::{BigUint, RandPrime};
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
pub fn toy_generate_dsa_params(p_bits: usize, q_bits: usize, seed: Option<u64>) -> DsaParams {
    let mut rng = match seed {
        Some(s) => StdRng::seed_from_u64(s),
        None => StdRng::from_entropy(),
    };

    // 1) generate prime q of q_bits
    let q = rng.gen_prime(q_bits);

    // 2) generate prime p of p_bits
    let mut p = BigUint::zero();
    for _attempt in 0..1000 {
        let candidate = rng.gen_prime(p_bits);
        // check if candidate-1 is multiple of q
        if (&candidate - BigUint::one()) % &q == BigUint::zero() {
            p = candidate;
            break;
        }
    }
    if p.is_zero() {
        panic!("Failed to find a p with (p-1) multiple of q in toy DSA param generation!");
    }

    // 3) compute g = h^((p-1)/q) mod p for some small h
    let mut h = BigUint::from(2u64);
    let pm1_div_q = (&p - BigUint::one()) / &q;
    let mut g = mod_exp(&h, &pm1_div_q, &p);
    // if g=1, increment h and retry in a toy manner
    for _ in 0..100 {
        if g == BigUint::one() {
            h += 1u64;
            g = mod_exp(&h, &pm1_div_q, &p);
        } else {
            break;
        }
    }
    if g == BigUint::one() {
        panic!("Failed to find g != 1 for toy DSA param generation after tries.");
    }

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
    let mut scalar_bytes = vec![0u8; q.bits() as usize / 8 + 1];
    rng.fill_bytes(&mut scalar_bytes);
    let mut x = BigUint::from_bytes_be(&scalar_bytes);
    x = x % q;
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

    // ephemeral k in [1..q-1]
    let mut k = BigUint::zero();
    for _ in 0..100 {
        let mut buf = vec![0u8; q.bits() as usize / 8 + 1];
        rng.fill_bytes(&mut buf);
        k = BigUint::from_bytes_be(&buf) % q;
        if !k.is_zero() {
            break;
        }
    }
    if k.is_zero() {
        k = BigUint::one();
    }

    // r = (g^k mod p) mod q
    let gk = mod_exp(g, &k, p);
    let r = &gk % q;

    // s = k^-1 * (h_m + x*r) mod q
    let k_inv = mod_inv(k, q).expect("No inverse for ephemeral k?! (toy DSA)");
    let xr = (x * &r) % q;
    let sum = (h_m + xr) % q;
    let s = (k_inv * sum) % q;

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
        return false;
    }
    if sig.s.is_zero() || sig.s >= *q {
        return false;
    }

    // w = s^-1 mod q
    let w = match mod_inv(sig.s.clone(), q) {
        Some(val) => val,
        None => return false,
    };

    // u1 = (h_m * w) mod q
    // u2 = (r * w) mod q
    let u1 = (h_m * &w) % q;
    let u2 = (&sig.r * &w) % q;

    // v = ((g^u1 * y^u2) mod p) mod q
    let gu1 = mod_exp(g, &u1, p);
    let yu2 = mod_exp(pub_key, &u2, p);
    let t = (gu1 * yu2) % p;
    let v = &t % q;

    v == sig.r
}

// Utility: modular exponentiation: base^exp mod m
fn mod_exp(base: &BigUint, exp: &BigUint, m: &BigUint) -> BigUint {
    if m.is_zero() {
        panic!("mod_exp with modulus=0");
    }
    let mut result = BigUint::one();
    let mut cur_base = base % m;
    for bit in exp.to_bytes_be() {
        for i in 0..8 {
            result = result.modpow(&BigUint::from(2u32), m);
            if ((bit >> (7 - i)) & 1) == 1 {
                result = (result * &cur_base) % m;
            }
            cur_base = (&cur_base * &cur_base) % m;
        }
    }
    result
}

// Utility: extended GCD to find modular inverse of x mod m if gcd(x,m)=1
fn mod_inv(x: BigUint, m: &BigUint) -> Option<BigUint> {
    let (g, s, _) = extended_gcd(x, m.clone());
    if g != BigUint::one() {
        None
    } else {
        Some(((s % m) + m) % m)
    }
}

// Extended Euclidean Algorithm
fn extended_gcd(mut a: BigUint, mut b: BigUint) -> (BigUint, BigUint, BigUint) {
    let (mut x0, mut x1) = (BigUint::one(), BigUint::zero());
    let (mut y0, mut y1) = (BigUint::zero(), BigUint::one());

    while b != BigUint::zero() {
        let q = &a / &b;
        let r = &a % &b;
        a = b;
        b = r;

        let tmpx = &x0 - &q * &x1;
        x0 = x1;
        x1 = tmpx;

        let tmpy = &y0 - &q * &y1;
        y0 = y1;
        y1 = tmpy;
    }
    (a, x0, y0)
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

        // Generate a key pair
        let kp = toy_dsa_generate_keypair(&params, Some(100));

        // "Hash" a message by just taking some BigUint (toy). Real usage => a real hash truncated to q bits.
        let msg_hash = BigUint::parse_bytes(b"123456789ABCDEF", 16).unwrap();

        // Sign
        let sig = toy_dsa_sign(&params, &kp, &msg_hash, Some(200));

        // Verify
        let valid = toy_dsa_verify(&params, &kp.public, &msg_hash, &sig);
        assert!(valid, "DSA signature must verify with correct key/msg.");

        // If we tamper the signature
        let mut bad_sig = sig.clone();
        bad_sig.r += 1u64;
        let invalid = toy_dsa_verify(&params, &kp.public, &msg_hash, &bad_sig);
        assert!(!invalid, "Tampered signature must fail verification");
    }
}
