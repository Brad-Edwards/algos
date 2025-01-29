pub mod aes;
pub mod blowfish;
pub mod diffie_hellman;
pub mod dsa;
pub mod elgamal;
pub mod elliptic;
pub mod md5;
pub mod rsa;
pub mod sha256;
pub mod twofish;

// Re-export AES functionality
pub use aes::{AesKey, AesKeySize, AES_BLOCK_SIZE};

// Re-export Blowfish functionality
pub use blowfish::{BlowfishKey, BLOWFISH_BLOCK_SIZE, BLOWFISH_MAX_KEY_BYTES};

// Re-export DSA functionality
pub use dsa::{
    toy_dsa_generate_keypair, toy_dsa_sign, toy_dsa_verify, toy_generate_dsa_params, DsaKeyPair,
    DsaParams, DsaSignature,
};

// Re-export SHA-256 functionality
pub use sha256::{Sha256, SHA256_OUTPUT_SIZE};

// Re-export RSA functionality
pub use rsa::{RSAKeyGenConfig, RSAKeyPair, RSAPrivateKey, RSAPublicKey};

// Re-export MD5 functionality
pub use md5::{md5_digest, Md5, MD5_OUTPUT_SIZE};

// Re-export Twofish functionality
pub use twofish::{TwofishKey, TwofishKeySize, TWOFISH_BLOCK_SIZE, TWOFISH_SUBKEY_COUNT};

// Re-export Diffie-Hellman functionality
pub use diffie_hellman::{
    DHKeyGenConfig, DHParamsConfig, DiffieHellmanKeyPair, DiffieHellmanParams,
};

// Re-export ElGamal functionality
pub use elgamal::{
    elgamal_decrypt, elgamal_encrypt, ElGamalCiphertext, ElGamalEncryptConfig, ElGamalKeyGenConfig,
    ElGamalKeyPair, ElGamalParams, ElGamalParamsConfig, ElGamalPrivateKey, ElGamalPublicKey,
};

// Re-export Elliptic Curve functionality
pub use elliptic::{
    point_add, point_double, point_mul, toy_ecdh, toy_generate_keypair, toy_secp256k1_curve, Point,
    ToyCurve, ToyKeyPair,
};
