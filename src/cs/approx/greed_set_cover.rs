use std::collections::HashSet;

#[derive(Debug, Clone)]
pub struct SetCoverInstance<T: Clone + Eq + std::hash::Hash> {
    universe: HashSet<T>,
    sets: Vec<HashSet<T>>,
    weights: Vec<f64>,
}

impl<T: Clone + Eq + std::hash::Hash> SetCoverInstance<T> {
    pub fn new(universe: HashSet<T>, sets: Vec<HashSet<T>>, weights: Vec<f64>) -> Self {
        assert_eq!(
            sets.len(),
            weights.len(),
            "Each set must have a corresponding weight"
        );
        Self {
            universe,
            sets,
            weights,
        }
    }
}

/// Implements the Greedy Set Cover algorithm.
///
/// This algorithm provides an H_n approximation for the weighted set cover problem,
/// where H_n is the nth harmonic number. It works by repeatedly selecting the set
/// with the minimum cost-effectiveness ratio (cost per newly covered element).
///
/// # Arguments
///
/// * `instance` - The set cover instance containing the universe, sets, and their weights
///
/// # Returns
///
/// * A vector of indices representing the selected sets in the cover
pub fn solve<T: Clone + Eq + std::hash::Hash>(instance: &SetCoverInstance<T>) -> Vec<usize> {
    let mut selected_sets = Vec::new();
    let mut covered: HashSet<T> = HashSet::new();

    // Continue until all elements are covered
    while covered.len() < instance.universe.len() {
        let mut best_ratio = f64::INFINITY;
        let mut best_idx = 0;
        let mut best_new_elements = HashSet::new();

        // Find set with best cost-effectiveness ratio
        for (idx, set) in instance.sets.iter().enumerate() {
            if selected_sets.contains(&idx) {
                continue;
            }

            let new_elements: HashSet<_> = set.difference(&covered).cloned().collect();
            if new_elements.is_empty() {
                continue;
            }

            let ratio = instance.weights[idx] / new_elements.len() as f64;
            if ratio < best_ratio {
                best_ratio = ratio;
                best_idx = idx;
                best_new_elements = new_elements;
            }
        }

        // Add the best set to our solution
        selected_sets.push(best_idx);
        covered.extend(best_new_elements);
    }

    selected_sets
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_set_cover() {
        let universe: HashSet<_> = (1..=4).collect();
        let sets = vec![
            [1, 2].iter().cloned().collect(),
            [2, 3].iter().cloned().collect(),
            [3, 4].iter().cloned().collect(),
        ];
        let weights = vec![1.0, 1.0, 1.0];

        let instance = SetCoverInstance::new(universe, sets, weights);
        let solution = solve(&instance);

        // Should need at most 3 sets to cover all elements
        assert!(solution.len() <= 3);

        // Verify that all elements are covered
        let mut covered: HashSet<_> = HashSet::new();
        for &idx in &solution {
            covered.extend(&instance.sets[idx]);
        }
        assert_eq!(covered, instance.universe);
    }

    #[test]
    fn test_weighted_set_cover() {
        let universe: HashSet<_> = (1..=3).collect();
        let sets = vec![
            [1, 2, 3].iter().cloned().collect(), // expensive set covering everything
            [1].iter().cloned().collect(),       // cheap set covering one element
            [2, 3].iter().cloned().collect(),    // medium set covering two elements
        ];
        let weights = vec![10.0, 1.0, 3.0];

        let instance = SetCoverInstance::new(universe, sets, weights);
        let solution = solve(&instance);

        // Should prefer the combination of cheap sets over the expensive set
        assert!(!solution.contains(&0));

        // Verify that all elements are covered
        let mut covered: HashSet<_> = HashSet::new();
        for &idx in &solution {
            covered.extend(&instance.sets[idx]);
        }
        assert_eq!(covered, instance.universe);
    }
}
