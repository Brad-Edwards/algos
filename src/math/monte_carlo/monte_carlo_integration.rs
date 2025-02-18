use rand::Rng;

/// Performs Monte Carlo integration of the function `f` over the interval [a, b] using the specified number of samples.
pub fn monte_carlo_integration<F>(f: F, a: f64, b: f64, samples: usize) -> f64
where
    F: Fn(f64) -> f64,
{
    let mut rng = rand::thread_rng();
    let mut sum = 0.0;
    for _ in 0..samples {
        let x = rng.gen_range(a..b);
        sum += f(x);
    }
    let avg = sum / samples as f64;
    (b - a) * avg
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_monte_carlo_integration() {
        // Integrate f(x) = x over [0,1]. The exact value is 0.5.
        let result = monte_carlo_integration(|x| x, 0.0, 1.0, 100_000);
        assert!((result - 0.5).abs() < 0.01);
    }
}
