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

/// Implements the Primal-Dual approximation algorithm for the Set Cover problem.
///
/// This algorithm provides an f-approximation for weighted set cover, where f is
/// the maximum frequency of any element (number of sets containing it). It works by:
/// 1. Maintaining dual variables for each element
/// 2. Increasing dual variables uniformly until some set becomes tight
/// 3. Including tight sets in the solution
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
    let mut selected_sets = Vec::new();
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
        // Find minimum slack among sets containing uncovered elements
        let mut min_slack = f64::INFINITY;
        let mut min_slack_set = 0;

        for &e in &uncovered {
            for &set_idx in element_to_sets.get(&e).unwrap_or(&Vec::new()) {
                if slacks[set_idx] < min_slack {
                    min_slack = slacks[set_idx];
                    min_slack_set = set_idx;
                }
            }
        }

        // Increase dual variables of uncovered elements
        for &e in &uncovered {
            let increase = min_slack;
            *duals.get_mut(&e).unwrap() += increase;

            // Update slacks of sets containing e
            for &set_idx in element_to_sets.get(&e).unwrap_or(&Vec::new()) {
                slacks[set_idx] -= increase;
            }
        }

        // Add tight set to solution
        selected_sets.push(min_slack_set);

        // Update uncovered elements
        for &e in &instance.sets[min_slack_set] {
            uncovered.remove(&e);
        }
    }

    // Calculate total weight
    let total_weight: f64 = selected_sets.iter().map(|&idx| instance.weights[idx]).sum();

    (selected_sets, total_weight)
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
        let max_freq = 2; // each element appears in at most 2 sets
        assert!(_weight <= max_freq as f64 * 2.0); // optimal is 2.0
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
