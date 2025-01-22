use rand::prelude::*;

pub fn generate_data(len: usize, distribution: &str) -> Vec<i32> {
    let mut rng = thread_rng();
    let mut data: Vec<i32> = (0..len as i32).collect();

    match distribution {
        "sorted" => { /* already sorted */ }
        "reverse" => data.reverse(),
        "random" => data.shuffle(&mut rng),
        "nearly_sorted" => {
            for i in 0..(len / 100) {
                let j = i * 100;
                if j + 1 < len {
                    data.swap(j, j + 1);
                }
            }
        }
        "few_unique" => {
            for i in 0..len {
                data[i] = (i % 10) as i32;
            }
        }
        _ => {}
    };
    data
}

pub const SIZES: [usize; 4] = [1_000, 10_000, 100_000, 1_000_000];
pub const DISTRIBUTIONS: [&str; 5] = ["sorted", "reverse", "random", "nearly_sorted", "few_unique"];
