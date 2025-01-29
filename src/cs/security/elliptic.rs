//! DISCLAIMER: This library is a **toy** example of elliptic curve cryptography (ECC) in pure Rust.
//! It is *EXCLUSIVELY* for demonstration and educational purposes. Absolutely DO NOT use it
//! for real cryptographic or security-sensitive operations. It is not audited, not vetted,
//! and very likely insecure in practice. If you need ECC or any cryptographic operations
//! in production, please use a vetted, well-reviewed cryptography library.

//! # Overview
//! This toy library shows a minimal subset of ECC operations on a short-Weierstrass curve,
//! specifically something akin to the secp256k1 prime curve. We demonstrate:
//! 1. The curve parameters (p, a, b, G, n) in a `ToyCurve` struct.
//! 2. A `Point` type representing a coordinate pair or "Infinity" as the identity.
//! 3. Functions for point addition, doubling, and scalar multiplication.
//! 4. A "keypair" mechanism to produce a private scalar `d` and a public point `Q = d*G`.
//! 5. A toy "ECDH" function that given two private keys can produce a shared secret.
//!
//! **Again, do not use in real security!**

use num_bigint::{BigInt, BigUint, Sign, ToBigInt};
use num_integer::Integer;
use num_traits::{One, Zero};

/// A point on the elliptic curve in short Weierstrass form (x, y) mod p,
/// plus a special "Infinity" variant for the identity element.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Point {
    /// The point at infinity (identity).
    Infinity,
    /// A projective/affine coordinate pair (x, y).
    Coord { x: BigUint, y: BigUint },
}

/// A toy ECC curve parameters, roughly mimicking secp256k1 (but not exactly).
///
/// For a real library, you'd confirm all domain parameters thoroughly.
#[derive(Debug, Clone)]
pub struct ToyCurve {
    /// The prime modulus p.
    pub p: BigUint,
    /// Coefficient a in the curve equation y^2 = x^3 + a*x + b (mod p).
    pub a: BigUint,
    /// Coefficient b in the curve equation y^2 = x^3 + a*x + b (mod p).
    pub b: BigUint,
    /// The base point (generator) G on the curve.
    pub g: Point,
    /// The order n of the base point G.
    pub n: BigUint,
}

/// A toy ECC keypair: private scalar `d`, public point `Q = d*G`.
#[derive(Debug, Clone)]
pub struct ToyKeyPair {
    pub private: BigUint,
    pub public: Point,
}

/// Minimal example of domain parameters akin to secp256k1.
/// Real secp256k1 is p=2^256 - 2^32 - 977, etc. We only illustrate here.
pub fn toy_secp256k1_curve() -> ToyCurve {
    // Real secp256k1:
    // p = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F
    // a = 0
    // b = 7
    // Gx=79be667ef..., Gy=483ada77...
    // n=FFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFE BAAEDCE6AF48A03B BFD25E8CD0364141
    // We'll do a smaller toy version just for demonstration.

    let p = BigUint::parse_bytes(
        b"FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F",
        16,
    )
    .unwrap_or_else(|| BigUint::parse_bytes(b"0", 10).unwrap());
    let a = BigUint::zero(); // a=0
    let b = BigUint::from(7u64); // b=7
                                 // Let's pretend G is some point, we do the real secp256k1 G:
    let gx_str = "79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798";
    let gy_str = "483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8";

    let gx = BigUint::parse_bytes(gx_str.as_bytes(), 16).unwrap_or(BigUint::zero());
    let gy = BigUint::parse_bytes(gy_str.as_bytes(), 16).unwrap_or(BigUint::zero());
    let g = Point::Coord { x: gx, y: gy };

    let n_str = "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141";
    let n = BigUint::parse_bytes(n_str.as_bytes(), 16).unwrap_or(BigUint::zero());

    ToyCurve { p, a, b, g, n }
}

/// Helper function for modular subtraction that avoids overflow
fn mod_sub(a: &BigUint, b: &BigUint, m: &BigUint) -> BigUint {
    let a_mod = a % m;
    let b_mod = b % m;
    if a_mod >= b_mod {
        (a_mod - b_mod) % m
    } else {
        (m - (b_mod - a_mod) % m) % m
    }
}

/// Helper function for modular addition
fn mod_add(a: &BigUint, b: &BigUint, m: &BigUint) -> BigUint {
    let a_mod = a % m;
    let b_mod = b % m;
    (a_mod + b_mod) % m
}

/// Helper function for modular multiplication
fn mod_mul(a: &BigUint, b: &BigUint, m: &BigUint) -> BigUint {
    let a_mod = a % m;
    let b_mod = b % m;
    (a_mod * b_mod) % m
}

/// We define the "point add" operation in short Weierstrass form:
/// - handle Infinity cases
/// - if x1==x2, check if y1== -y2 => Infinity or else point doubling
/// - standard formula for slope, etc.
pub fn point_add(c: &ToyCurve, p1: &Point, p2: &Point) -> Point {
    match (p1, p2) {
        (Point::Infinity, _) => p2.clone(),
        (_, Point::Infinity) => p1.clone(),
        (Point::Coord { x: x1, y: y1 }, Point::Coord { x: x2, y: y2 }) => {
            if x1 == x2 {
                // if y1 = -y2 => Infinity
                // in mod p, -y2 means p-y2
                let neg_y2 = mod_sub(&c.p, y2, &c.p);
                if y1 == &neg_y2 {
                    return Point::Infinity;
                }
                // else p1==p2 => use doubling
                return point_double(c, p1);
            }
            // slope = (y2 - y1) * inv(x2 - x1) mod p
            let dx = mod_sub(x2, x1, &c.p);
            let dy = mod_sub(y2, y1, &c.p);
            let dx_inv = mod_inv(dx, &c.p).expect("No inverse, degenerate case?");
            let slope = mod_mul(&dy, &dx_inv, &c.p);

            // x3 = slope^2 - x1 - x2
            let slope_squared = mod_mul(&slope, &slope, &c.p);
            let mut x3 = mod_sub(&slope_squared, x1, &c.p);
            x3 = mod_sub(&x3, x2, &c.p);

            // y3 = slope*(x1 - x3) - y1
            let x1_minus_x3 = mod_sub(x1, &x3, &c.p);
            let slope_times_diff = mod_mul(&slope, &x1_minus_x3, &c.p);
            let y3 = mod_sub(&slope_times_diff, y1, &c.p);

            Point::Coord { x: x3, y: y3 }
        }
    }
}

/// Point doubling: slope = (3*x^2+a)/(2y)
pub fn point_double(c: &ToyCurve, p: &Point) -> Point {
    match p {
        Point::Infinity => Point::Infinity,
        Point::Coord { x, y } => {
            if y.is_zero() {
                return Point::Infinity;
            }
            let two = BigUint::from(2u64);
            let three = BigUint::from(3u64);

            // slope = (3x^2 + a) / (2y)
            let x_squared = mod_mul(x, x, &c.p);
            println!("x_squared = {}", x_squared);

            let three_x_squared = mod_mul(&three, &x_squared, &c.p);
            println!("three_x_squared = {}", three_x_squared);

            let numerator = mod_add(&three_x_squared, &c.a, &c.p);
            println!("numerator = {}", numerator);

            let two_y = mod_mul(&two, y, &c.p);
            println!("two_y = {}", two_y);

            let denom_inv = mod_inv(two_y, &c.p).expect("no inverse in doubling?");
            println!("denom_inv = {}", denom_inv);

            let slope = mod_mul(&numerator, &denom_inv, &c.p);
            println!("slope = {}", slope);

            // x3 = slope^2 - 2x
            let slope_squared = mod_mul(&slope, &slope, &c.p);
            println!("slope_squared = {}", slope_squared);

            let two_x = mod_mul(&two, x, &c.p);
            println!("two_x = {}", two_x);

            let x3 = mod_sub(&slope_squared, &two_x, &c.p);
            println!("x3 = {}", x3);

            // y3 = slope*(x - x3) - y
            let x_minus_x3 = mod_sub(x, &x3, &c.p);
            println!("x_minus_x3 = {}", x_minus_x3);

            let slope_times_diff = mod_mul(&slope, &x_minus_x3, &c.p);
            println!("slope_times_diff = {}", slope_times_diff);

            let y3 = mod_sub(&slope_times_diff, y, &c.p);
            println!("y3 = {}", y3);

            Point::Coord { x: x3, y: y3 }
        }
    }
}

/// Scalar multiplication: compute s*P. We use a simple double-and-add here. Not optimized.
pub fn point_mul(c: &ToyCurve, p: &Point, scalar: &BigUint) -> Point {
    let mut result = Point::Infinity;
    let base = p.clone();
    for bit in scalar.to_bytes_be() {
        // process each byte, then each bit
        for i in 0..8 {
            result = point_double(c, &result);
            if ((bit >> (7 - i)) & 1) == 1 {
                result = point_add(c, &result, &base);
            }
        }
    }
    result
}

/// Creates a toy ECC key pair: pick a random scalar d in [1..n-1], then Q = dG.
use rand::{rngs::StdRng, RngCore, SeedableRng};

pub fn toy_generate_keypair(c: &ToyCurve, seed: Option<u64>) -> ToyKeyPair {
    let mut rng = match seed {
        Some(s) => StdRng::seed_from_u64(s),
        None => StdRng::from_entropy(),
    };

    // produce a random scalar in [1..n-1]
    let mut scalar_bytes = vec![0u8; c.n.bits() as usize / 8 + 1];
    rng.fill_bytes(&mut scalar_bytes);

    let mut d = BigUint::from_bytes_be(&scalar_bytes);
    d %= &c.n;
    if d.is_zero() {
        d = BigUint::one();
    }
    let public = point_mul(c, &c.g, &d);
    ToyKeyPair { private: d, public }
}

/// A toy ECDH: each side has a private key (d1, d2). They produce shared = d1 * Q2 = d1 * (d2 * G).
/// which = d1*d2 * G = d2 * Q1, etc.
pub fn toy_ecdh(c: &ToyCurve, kp1: &ToyKeyPair, kp2: &ToyKeyPair) -> Point {
    // user1 computes S = d1 * Q2
    let s1 = point_mul(c, &kp2.public, &kp1.private);
    // user2 computes S = d2 * Q1
    let s2 = point_mul(c, &kp1.public, &kp2.private);
    // Should be the same
    assert_eq!(s1, s2, "ECDH mismatch?! In correct math they match.");
    s1
}

/// Extended Euclidean for (a, b) => (g, x, y) with ax + by = g
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

/// Compute modular inverse of x mod m using extended Euclid. Return None if no inverse exists.
fn mod_inv(x: BigUint, m: &BigUint) -> Option<BigUint> {
    if x.is_zero() {
        return None;
    }
    let (g, s, _t) = extended_gcd(x.clone(), m.clone());
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
    fn test_toy_ecc_basic() {
        // Use a small curve for testing
        let p = BigUint::from(23u64); // Small prime
        let a = BigUint::from(1u64); // y^2 = x^3 + x + 1
        let b = BigUint::from(1u64);

        // Point (3,10) is on the curve y^2 = x^3 + x + 1 mod 23
        let gx = BigUint::from(3u64);
        let gy = BigUint::from(10u64);
        let g = Point::Coord { x: gx, y: gy };

        // Order of the curve is 28
        let n = BigUint::from(28u64);

        let curve = ToyCurve { p, a, b, g, n };

        // Check G + Infinity => G
        let r1 = point_add(&curve, &curve.g, &Point::Infinity);
        assert_eq!(r1, curve.g);

        // Doubling G: 2G
        let two_g = point_double(&curve, &curve.g);
        assert!(
            two_g != Point::Infinity,
            "2G shouldn't be Infinity in normal curve usage"
        );

        // Key pairs
        let kp1 = toy_generate_keypair(&curve, Some(42));
        let kp2 = toy_generate_keypair(&curve, Some(123));

        let shared1 = toy_ecdh(&curve, &kp1, &kp2);
        let shared2 = toy_ecdh(&curve, &kp2, &kp1);
        assert_eq!(shared1, shared2, "Shared secrets must match");
        assert_ne!(
            shared1,
            Point::Infinity,
            "Should be a valid point if d1, d2 != 0"
        );
    }
}
