use std::collections::{HashMap, HashSet};

#[derive(Debug, Clone)]
pub struct SetCoverInstance {
    sets: Vec<HashSet<usize>>,
    weights: Vec<f64>,
    universe: HashSet<usize>,
}

impl SetCoverInstance {
    pub fn new(sets: Vec<HashSet<usize>>, weights: Vec<f64>, universe: HashSet<usize>) -> Self {
        assert_eq!(sets.len(), weights.len(), "Each set must have a weight");
        Self {
            sets,
            weights,
            universe,
        }
    }
}

/// Implements the LP Rounding algorithm for the Set Cover problem.
///
/// This algorithm provides an O(log n)-approximation for weighted set cover. It works by:
/// 1. Solving the LP relaxation using the Primal-Dual method
/// 2. Rounding the fractional solution to an integral solution
///
/// # Arguments
///
/// * `instance` - The set cover instance
///
/// # Returns
///
/// * A tuple containing:
///   - Vector of indices of selected sets
///   - Total weight of the solution
pub fn solve(instance: &SetCoverInstance) -> (Vec<usize>, f64) {
    // Solve LP relaxation using Primal-Dual method
    let fractional_solution = solve_lp_relaxation(instance);

    // Round fractional solution
    let selected_sets = round_solution(&fractional_solution, instance);

    // Calculate total weight
    let total_weight: f64 = selected_sets.iter().map(|&idx| instance.weights[idx]).sum();

    (selected_sets, total_weight)
}

fn solve_lp_relaxation(instance: &SetCoverInstance) -> Vec<f64> {
    let mut x = vec![0.0; instance.sets.len()];
    let mut uncovered: HashSet<_> = instance.universe.clone();

    // Initialize dual variables (y_e for each element e)
    let mut duals: HashMap<usize, f64> = HashMap::new();
    for &e in &instance.universe {
        duals.insert(e, 0.0);
    }

    // Track remaining slack for each set
    let mut slacks: Vec<f64> = instance.weights.clone();

    // Track which elements are in which sets for efficient lookup
    let mut element_to_sets: HashMap<usize, Vec<usize>> = HashMap::new();
    for (set_idx, set) in instance.sets.iter().enumerate() {
        for &e in set {
            element_to_sets.entry(e).or_default().push(set_idx);
        }
    }

    while !uncovered.is_empty() {
        // Find minimum slack per uncovered element ratio
        let mut min_ratio = f64::INFINITY;
        let mut min_ratio_set = 0;

        for &e in &uncovered {
            for &set_idx in element_to_sets.get(&e).unwrap_or(&Vec::new()) {
                if slacks[set_idx] <= 0.0 {
                    continue;
                }
                let ratio = slacks[set_idx] / instance.sets[set_idx].len() as f64;
                if ratio < min_ratio {
                    min_ratio = ratio;
                    min_ratio_set = set_idx;
                }
            }
        }

        // Update fractional solution and dual variables
        x[min_ratio_set] += min_ratio;
        for &e in &instance.sets[min_ratio_set] {
            if uncovered.contains(&e) {
                *duals.get_mut(&e).unwrap() += min_ratio;

                // Update slacks of sets containing e
                for &set_idx in element_to_sets.get(&e).unwrap_or(&Vec::new()) {
                    slacks[set_idx] -= min_ratio;
                }

                uncovered.remove(&e);
            }
        }
    }

    x
}

fn round_solution(fractional_solution: &[f64], instance: &SetCoverInstance) -> Vec<usize> {
    let ln_n = (instance.universe.len() as f64).ln();
    let threshold = 1.0 / ln_n;

    let mut selected_sets = Vec::new();
    let mut uncovered: HashSet<_> = instance.universe.clone();

    // First round: select sets with large fractional values
    for (idx, &value) in fractional_solution.iter().enumerate() {
        if value >= threshold {
            selected_sets.push(idx);
            for &e in &instance.sets[idx] {
                uncovered.remove(&e);
            }
        }
    }

    // Second round: greedily cover remaining elements
    while !uncovered.is_empty() {
        // Find set covering most uncovered elements per unit cost
        let mut best_ratio = 0.0;
        let mut best_idx = 0;

        for (idx, set) in instance.sets.iter().enumerate() {
            if selected_sets.contains(&idx) {
                continue;
            }

            let uncovered_count = set.intersection(&uncovered).count();
            if uncovered_count == 0 {
                continue;
            }

            let ratio = uncovered_count as f64 / instance.weights[idx];
            if ratio > best_ratio {
                best_ratio = ratio;
                best_idx = idx;
            }
        }

        selected_sets.push(best_idx);
        for &e in &instance.sets[best_idx] {
            uncovered.remove(&e);
        }
    }

    selected_sets
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_set_cover() {
        let mut sets = Vec::new();
        sets.push([1, 2].iter().cloned().collect());
        sets.push([2, 3].iter().cloned().collect());
        sets.push([3, 4].iter().cloned().collect());

        let weights = vec![1.0, 1.0, 1.0];
        let universe: HashSet<_> = (1..=4).collect();

        let instance = SetCoverInstance::new(sets, weights, universe);
        let (solution, _weight) = solve(&instance);

        // Verify solution covers all elements
        let mut covered = HashSet::new();
        for &idx in &solution {
            covered.extend(&instance.sets[idx]);
        }
        assert_eq!(covered, instance.universe);

        // Verify approximation ratio
        let opt = 2.0; // optimal solution uses 2 sets
        assert!(_weight <= opt * (instance.universe.len() as f64).ln());
    }

    #[test]
    fn test_weighted_set_cover() {
        let mut sets = Vec::new();
        sets.push([1, 2, 3].iter().cloned().collect());
        sets.push([1].iter().cloned().collect());
        sets.push([2, 3].iter().cloned().collect());

        let weights = vec![10.0, 1.0, 3.0];
        let universe: HashSet<_> = (1..=3).collect();

        let instance = SetCoverInstance::new(sets, weights, universe);
        let (solution, _weight) = solve(&instance);

        // Verify solution covers all elements
        let mut covered = HashSet::new();
        for &idx in &solution {
            covered.extend(&instance.sets[idx]);
        }
        assert_eq!(covered, instance.universe);

        // Should prefer cheaper sets
        assert!(!solution.contains(&0));
    }

    #[test]
    fn test_empty_instance() {
        let instance = SetCoverInstance::new(Vec::new(), Vec::new(), HashSet::new());
        let (solution, _weight) = solve(&instance);

        assert!(solution.is_empty());
        assert_eq!(_weight, 0.0);
    }
}
