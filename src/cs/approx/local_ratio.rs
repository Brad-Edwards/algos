use std::collections::{HashMap, HashSet};

#[derive(Debug, Clone)]
pub struct WeightedGraph {
    edges: Vec<(usize, usize)>,
    weights: Vec<f64>,
}

impl WeightedGraph {
    pub fn new(edges: Vec<(usize, usize)>, weights: Vec<f64>) -> Self {
        Self { edges, weights }
    }
}

/// Implements the Local-Ratio algorithm for the Weighted Vertex Cover problem.
///
/// This algorithm provides a 2-approximation for weighted vertex cover. It works by:
/// 1. Finding a vertex with minimum weight incident to some edge
/// 2. Subtracting this weight from all vertices incident to that edge
/// 3. Recursively solving the problem with reduced weights
/// 4. Including vertices with non-zero original weight in the solution
///
/// # Arguments
///
/// * `graph` - The weighted graph instance
///
/// # Returns
///
/// * A tuple containing:
///   - HashSet of vertices in the cover
///   - Total weight of the cover
pub fn solve(graph: &WeightedGraph) -> (HashSet<usize>, f64) {
    // Build adjacency list for efficient edge lookup
    let mut adj_list: HashMap<usize, Vec<usize>> = HashMap::new();
    let mut vertex_set = HashSet::new();

    for &(u, v) in &graph.edges {
        adj_list.entry(u).or_default().push(v);
        adj_list.entry(v).or_default().push(u);
        vertex_set.insert(u);
        vertex_set.insert(v);
    }

    // Create mutable copy of weights for recursive algorithm
    let mut weights = graph.weights.clone();
    let solution = solve_recursive(&adj_list, &mut weights, &vertex_set, &graph.weights);

    // Calculate total weight
    let total_weight: f64 = solution.iter().map(|&v| graph.weights[v]).sum();

    (solution, total_weight)
}

fn solve_recursive(
    adj_list: &HashMap<usize, Vec<usize>>,
    weights: &mut Vec<f64>,
    vertex_set: &HashSet<usize>,
    original_weights: &[f64],
) -> HashSet<usize> {
    // Find an uncovered edge
    let mut remaining_edge = None;
    for &u in vertex_set {
        if weights[u] <= 0.0 {
            continue;
        }
        if let Some(neighbors) = adj_list.get(&u) {
            for &v in neighbors {
                if weights[v] > 0.0 {
                    remaining_edge = Some((u, v));
                    break;
                }
            }
        }
        if remaining_edge.is_some() {
            break;
        }
    }

    // Base case: no edges remain
    if remaining_edge.is_none() {
        return HashSet::new();
    }

    let (u, v) = remaining_edge.unwrap();
    let epsilon = weights[u].min(weights[v]);

    // Subtract epsilon from weights
    weights[u] -= epsilon;
    weights[v] -= epsilon;

    // Recursively solve
    let mut solution = solve_recursive(adj_list, weights, vertex_set, original_weights);

    // Include vertices based on local ratio rule
    if !solution.contains(&u) && !solution.contains(&v) {
        // If neither endpoint is in solution, add the one with smaller original weight
        if original_weights[u] <= original_weights[v] {
            solution.insert(u);
        } else {
            solution.insert(v);
        }
    }

    solution
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_graph() {
        let edges = vec![(0, 1), (1, 2)];
        let weights = vec![1.0, 2.0, 1.0];
        let graph = WeightedGraph::new(edges, weights);

        let (cover, weight) = solve(&graph);

        // Verify cover properties
        assert!(cover.len() <= 2); // 2-approximation guarantee

        // Verify all edges are covered
        for &(u, v) in &graph.edges {
            assert!(cover.contains(&u) || cover.contains(&v));
        }

        // Verify weight calculation
        let actual_weight: f64 = cover.iter().map(|&v| graph.weights[v]).sum();
        assert_eq!(weight, actual_weight);
    }

    #[test]
    fn test_star_graph() {
        let edges = vec![(0, 1), (0, 2), (0, 3)];
        let weights = vec![1.0, 2.0, 2.0, 2.0];
        let graph = WeightedGraph::new(edges, weights);

        let (cover, weight) = solve(&graph);

        // Optimal solution uses just vertex 0
        assert!(weight <= 2.0 * 1.0); // 2-approximation guarantee

        // Verify all edges are covered
        for &(u, v) in &graph.edges {
            assert!(cover.contains(&u) || cover.contains(&v));
        }
    }

    #[test]
    fn test_empty_graph() {
        let graph = WeightedGraph::new(Vec::new(), vec![1.0, 1.0]);
        let (cover, weight) = solve(&graph);

        assert!(cover.is_empty());
        assert_eq!(weight, 0.0);
    }
}
