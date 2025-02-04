#[derive(Debug, Clone)]
pub struct Item {
    weight: f64,
    value: f64,
}

impl Item {
    pub fn new(weight: f64, value: f64) -> Self {
        Self { weight, value }
    }
}

#[derive(Debug, Clone)]
pub struct KnapsackInstance {
    items: Vec<Item>,
    capacity: f64,
}

impl KnapsackInstance {
    pub fn new(items: Vec<Item>, capacity: f64) -> Self {
        Self { items, capacity }
    }
}

/// Implements a Polynomial-Time Approximation Scheme (PTAS) for the Knapsack problem.
///
/// This algorithm provides a (1 + ε)-approximation for any ε > 0. It works by:
/// 1. Rounding item values to create a smaller range of possible values
/// 2. Solving the rounded problem exactly using dynamic programming
/// 3. Converting the solution back to the original problem
///
/// # Arguments
///
/// * `instance` - The knapsack instance containing items and capacity
/// * `epsilon` - The approximation parameter (smaller means better approximation)
///
/// # Returns
///
/// * A tuple containing:
///   - Vector of indices of selected items
///   - Total value of the solution
pub fn solve(instance: &KnapsackInstance, epsilon: f64) -> (Vec<usize>, f64) {
    assert!(epsilon > 0.0, "Epsilon must be positive");

    if instance.items.is_empty() {
        return (Vec::new(), 0.0);
    }

    // Find maximum item value for scaling
    let max_value = instance
        .items
        .iter()
        .map(|item| item.value)
        .fold(f64::NEG_INFINITY, f64::max);

    // Scale and round values
    let scale = epsilon * max_value / instance.items.len() as f64;
    let scaled_items: Vec<_> = instance
        .items
        .iter()
        .map(|item| Item::new(item.weight, (item.value / scale).floor() * scale))
        .collect();

    // Solve rounded problem using dynamic programming
    let scaled_instance = KnapsackInstance::new(scaled_items, instance.capacity);
    let solution = solve_exact(&scaled_instance);

    // Calculate actual value using original item values
    let total_value = solution.iter().map(|&idx| instance.items[idx].value).sum();

    (solution, total_value)
}

fn solve_exact(instance: &KnapsackInstance) -> Vec<usize> {
    let n = instance.items.len();
    let max_scaled_value: f64 = instance.items.iter().map(|item| item.value).sum();
    let max_value_int = max_scaled_value.ceil() as usize;

    // dp[i][v] = minimum weight needed to achieve value v using first i items
    let mut dp = vec![vec![f64::INFINITY; max_value_int + 1]; n + 1];
    dp[0][0] = 0.0;

    // For each item
    for i in 0..n {
        let item = &instance.items[i];

        // For each possible value
        for v in 0..=max_value_int {
            // Don't take item i
            dp[i + 1][v] = dp[i][v];

            // Take item i if possible
            let prev_v = v.saturating_sub(item.value.floor() as usize);
            if dp[i][prev_v] + item.weight <= instance.capacity {
                dp[i + 1][v] = dp[i + 1][v].min(dp[i][prev_v] + item.weight);
            }
        }
    }

    // Find maximum achievable value
    let mut max_value = 0;
    for v in 0..=max_value_int {
        if dp[n][v] <= instance.capacity {
            max_value = v;
        }
    }

    // Reconstruct solution
    let mut solution = Vec::new();
    let mut remaining_value = max_value;
    let mut i = n;

    while remaining_value > 0 && i > 0 {
        if dp[i][remaining_value] != dp[i - 1][remaining_value] {
            solution.push(i - 1);
            remaining_value =
                remaining_value.saturating_sub(instance.items[i - 1].value.floor() as usize);
        }
        i -= 1;
    }

    solution.reverse();
    solution
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_knapsack() {
        let items = vec![
            Item::new(2.0, 3.0),
            Item::new(3.0, 4.0),
            Item::new(4.0, 5.0),
        ];
        let instance = KnapsackInstance::new(items, 6.0);

        let (solution, value) = solve(&instance, 0.1);

        // Verify capacity constraint
        let total_weight: f64 = solution.iter().map(|&idx| instance.items[idx].weight).sum();
        assert!(total_weight <= instance.capacity);

        // Verify solution quality (should be close to optimal)
        assert!(value >= 7.0); // optimal is 7.0 (items 0 and 1)
    }

    #[test]
    fn test_empty_knapsack() {
        let instance = KnapsackInstance::new(Vec::new(), 10.0);
        let (solution, value) = solve(&instance, 0.1);

        assert!(solution.is_empty());
        assert_eq!(value, 0.0);
    }
}
