use rand::Rng;

/// Returns a reservoir sample of k items from the given iterator.
/// This function processes the iterator in a single pass and selects k items uniformly at random.
pub fn reservoir_sampling<T, I>(iter: I, k: usize) -> Vec<T>
where
    I: Iterator<Item = T>,
{
    let mut reservoir = Vec::with_capacity(k);
    let mut rng = rand::thread_rng();
    for (i, item) in iter.enumerate() {
        if i < k {
            reservoir.push(item);
        } else {
            let r = rng.gen_range(0..=i);
            if r < k {
                reservoir[r] = item;
            }
        }
    }
    reservoir
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_reservoir_sampling() {
        let data = 1..101;
        let sample = reservoir_sampling(data, 10);
        assert_eq!(sample.len(), 10);
        // Ensure all selected values are in the expected range.
        for &x in &sample {
            assert!(x >= 1 && x <= 100);
        }
    }
}
