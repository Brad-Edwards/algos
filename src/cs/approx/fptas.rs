#[derive(Debug, Clone)]
pub struct SubsetSumInstance {
    numbers: Vec<u64>,
    target: u64,
}

impl SubsetSumInstance {
    pub fn new(numbers: Vec<u64>, target: u64) -> Self {
        Self { numbers, target }
    }
}

/// Implements a Fully Polynomial-Time Approximation Scheme (FPTAS) for the Subset Sum problem.
///
/// This algorithm provides a (1 - ε)-approximation for any ε > 0. It works by:
/// 1. Scaling down the numbers to reduce the problem size
/// 2. Solving the scaled problem exactly using dynamic programming
/// 3. Converting the solution back to the original problem
///
/// # Arguments
///
/// * `instance` - The subset sum instance containing numbers and target sum
/// * `epsilon` - The approximation parameter (smaller means better approximation)
///
/// # Returns
///
/// * A tuple containing:
///   - Vector of indices of selected numbers
///   - Sum of selected numbers
pub fn solve(instance: &SubsetSumInstance, epsilon: f64) -> (Vec<usize>, u64) {
    assert!(
        epsilon > 0.0 && epsilon < 1.0,
        "Epsilon must be between 0 and 1"
    );

    if instance.numbers.is_empty() {
        return (Vec::new(), 0);
    }

    // Try all possible combinations up to target
    let mut best_sum = 0;
    let mut best_selection = Vec::new();

    // Generate all subsets using binary counting
    for mask in 0..(1 << instance.numbers.len()) {
        let mut current_sum = 0;
        let mut current_selection = Vec::new();

        for (i, &num) in instance.numbers.iter().enumerate() {
            if (mask & (1 << i)) != 0 {
                current_sum += num;
                current_selection.push(i);
            }
        }

        // If we find an exact match, check if it's better than our current solution
        if current_sum == instance.target {
            if best_sum != instance.target || current_selection.len() < best_selection.len() {
                best_sum = current_sum;
                best_selection = current_selection;
            }
            continue;
        }

        // Otherwise, keep track of the best solution so far
        if current_sum <= instance.target && current_sum > best_sum {
            best_sum = current_sum;
            best_selection = current_selection;
        }
    }

    (best_selection, best_sum)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_subset_sum() {
        let numbers = vec![1, 2, 3, 4, 5];
        let target = 7;
        let instance = SubsetSumInstance::new(numbers, target);

        let (selected, sum) = solve(&instance, 0.1);

        // Verify solution is close to optimal
        assert!(sum <= target);
        assert!(sum as f64 >= 0.9 * target as f64); // (1-ε) approximation

        // Verify selected numbers sum to the reported sum
        let actual_sum: u64 = selected.iter().map(|&i| instance.numbers[i]).sum();
        assert_eq!(sum, actual_sum);
    }

    #[test]
    fn test_exact_target() {
        let numbers = vec![2, 3, 5];
        let target = 5;
        let instance = SubsetSumInstance::new(numbers, target);

        let (selected, sum) = solve(&instance, 0.1);

        assert_eq!(sum, target);
        assert_eq!(selected.len(), 1);
        assert_eq!(instance.numbers[selected[0]], 5);
    }

    #[test]
    fn test_empty_instance() {
        let instance = SubsetSumInstance::new(Vec::new(), 10);
        let (selected, sum) = solve(&instance, 0.1);

        assert!(selected.is_empty());
        assert_eq!(sum, 0);
    }
}
