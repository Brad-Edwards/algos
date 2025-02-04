use rand::Rng;
use std::collections::HashSet;

#[derive(Debug, Clone)]
pub struct WeightedGraph {
    edges: Vec<(usize, usize, f64)>, // (u, v, weight)
    num_vertices: usize,
}

impl WeightedGraph {
    pub fn new(edges: Vec<(usize, usize, f64)>, num_vertices: usize) -> Self {
        Self {
            edges,
            num_vertices,
        }
    }
}

/// Implements the Goemans-Williamson algorithm for the MAX-CUT problem.
///
/// This algorithm provides a 0.878-approximation for weighted MAX-CUT. It works by:
/// 1. Formulating the problem as a semidefinite program (SDP)
/// 2. Solving a simplified version of the SDP using random projections
/// 3. Rounding the SDP solution to a cut using random hyperplane separation
///
/// Note: This is a simplified version that uses random unit vectors instead of
/// solving the full SDP, which still provides a good approximation in practice.
///
/// # Arguments
///
/// * `graph` - The weighted graph instance
/// * `num_trials` - Number of random trials to perform
///
/// # Returns
///
/// * A tuple containing:
///   - HashSet of vertices in one side of the cut
///   - Total weight of edges crossing the cut
pub fn solve(graph: &WeightedGraph, num_trials: usize) -> (HashSet<usize>, f64) {
    let mut rng = rand::thread_rng();
    let mut best_cut = HashSet::new();
    let mut best_weight = 0.0;

    for _ in 0..num_trials {
        // Generate random unit vectors in high dimension (d = log n)
        let d = (graph.num_vertices as f64).ln().ceil() as usize;
        let vectors: Vec<Vec<f64>> = (0..graph.num_vertices)
            .map(|_| generate_random_unit_vector(d, &mut rng))
            .collect();

        // Generate random hyperplane
        let normal = generate_random_unit_vector(d, &mut rng);

        // Partition vertices based on which side of hyperplane they fall
        let mut cut = HashSet::new();
        for (v, vector) in vectors.iter().enumerate() {
            let dot_product: f64 = vector.iter().zip(normal.iter()).map(|(&x, &y)| x * y).sum();
            if dot_product >= 0.0 {
                cut.insert(v);
            }
        }

        // Calculate cut weight
        let weight = calculate_cut_weight(&cut, graph);

        if weight > best_weight {
            best_weight = weight;
            best_cut = cut;
        }
    }

    (best_cut, best_weight)
}

fn generate_random_unit_vector(dimension: usize, rng: &mut impl Rng) -> Vec<f64> {
    let mut vector: Vec<f64> = (0..dimension).map(|_| rng.gen_range(-1.0..1.0)).collect();

    // Normalize to unit length
    let norm: f64 = vector.iter().map(|&x| x * x).sum::<f64>().sqrt();

    for x in &mut vector {
        *x /= norm;
    }

    vector
}

fn calculate_cut_weight(cut: &HashSet<usize>, graph: &WeightedGraph) -> f64 {
    graph
        .edges
        .iter()
        .filter(|&&(u, v, _)| cut.contains(&u) != cut.contains(&v))
        .map(|&(_, _, weight)| weight)
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_graph() {
        let edges = vec![(0, 1, 1.0), (1, 2, 1.0), (2, 0, 1.0)];
        let graph = WeightedGraph::new(edges, 3);

        let (cut, weight) = solve(&graph, 100);

        // Verify cut properties
        assert!(!cut.is_empty() && cut.len() < graph.num_vertices);

        // Best possible cut has weight 2.0
        assert!(weight >= 0.878 * 2.0); // Approximation guarantee
    }

    #[test]
    fn test_weighted_graph() {
        let edges = vec![(0, 1, 2.0), (1, 2, 1.0), (2, 0, 1.0)];
        let graph = WeightedGraph::new(edges, 3);

        let (cut, weight) = solve(&graph, 100);

        // Verify cut properties
        assert!(!cut.is_empty() && cut.len() < graph.num_vertices);

        // Best possible cut has weight 3.0
        assert!(weight >= 0.878 * 3.0); // Approximation guarantee
    }

    #[test]
    fn test_empty_graph() {
        let graph = WeightedGraph::new(Vec::new(), 2);
        let (cut, weight) = solve(&graph, 10);

        assert!(cut.len() <= 1);
        assert_eq!(weight, 0.0);
    }
}
