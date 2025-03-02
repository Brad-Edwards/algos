use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;

fn main() {
    // Print out the available methods on ChaCha20Rng to see how to properly initialize it
    println!("Testing ChaCha20Rng initialization");

    // Try different ways to initialize
    let _rng1 = ChaCha20Rng::from_seed([0; 32]);
    println!("Created rng1 with from_seed");

    let _rng2 = ChaCha20Rng::from_entropy();
    println!("Created rng2 with from_entropy");

    let _rng3 = ChaCha20Rng::seed_from_u64(123);
    println!("Created rng3 with seed_from_u64");

    println!("Success!");
}
