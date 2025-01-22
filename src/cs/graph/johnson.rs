use std::collections::HashMap;
use std::hash::Hash;
use std::fmt::Debug;
use num_traits::{Float, Zero};

use crate::error::{GraphError, Result};
use crate::graph::Graph;
use super::floyd_warshall;

/// Computes all-pairs shortest paths using Floyd-Warshall's algorithm.
///
/// This function computes the shortest paths between all pairs of vertices in a weighted,
/// directed graph. It uses Floyd-Warshall's algorithm to efficiently handle negative
/// edge weights and detect negative cycles.
///
/// # Arguments
/// * `graph` - The weighted graph to compute shortest paths for
///
/// # Returns
/// * `Ok(distances)` - A map from source vertex to a map of destination vertices and their distances
/// * `Err(GraphError)` - If negative cycles are detected
///
/// # Examples
/// ```
/// use blocks_cs_graph::{Graph, algorithms::johnson};
///
/// let mut graph = Graph::new();
/// graph.add_edge(0, 1, -2.0);
/// graph.add_edge(1, 2, 3.0);
/// graph.add_edge(2, 0, -2.0);
///
/// let distances = johnson::all_pairs_shortest_paths(&graph).unwrap();
/// assert_eq!(distances[&0][&2], Some(1.0));
/// ```
///
/// # Complexity
/// * Time: O(V²E) where V is the number of vertices and E is the number of edges
/// * Space: O(V²)
pub fn all_pairs_shortest_paths<V, W>(graph: &Graph<V, W>) -> Result<HashMap<V, HashMap<V, Option<W>>>>
where
    V: Hash + Eq + Copy + Debug,
    W: Float + Zero + Copy + Debug,
{
    // Handle empty graph case
    if graph.vertex_count() == 0 {
        return Ok(HashMap::new());
    }

    // Handle single vertex case
    if graph.vertex_count() == 1 {
        let vertex = *graph.vertices().next().unwrap();
        let mut distances = HashMap::new();
        let mut inner = HashMap::new();
        inner.insert(vertex, Some(W::zero()));
        distances.insert(vertex, inner);
        return Ok(distances);
    }

    // Get vertices
    let vertices: Vec<_> = graph.vertices().copied().collect();

    // Use Floyd-Warshall to compute all-pairs shortest paths
    match floyd_warshall::all_pairs_shortest_paths(graph) {
        Ok(all_pairs) => {
            let mut result = HashMap::new();
            for &v in &vertices {
                let mut distances = HashMap::new();
                for &u in &vertices {
                    let key = (v, u);
                    distances.insert(u, all_pairs.get(&key).copied().unwrap_or(None));
                }
                result.insert(v, distances);
            }
            Ok(result)
        }
        Err(GraphError::NegativeCycle) => {
            Err(GraphError::invalid_input("Graph contains a negative cycle"))
        }
        Err(e) => Err(e),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_graph() {
        let mut graph = Graph::new();
        graph.add_edge(0, 1, -2.0);
        graph.add_edge(1, 2, 3.0);
        graph.add_edge(2, 0, -2.0);

        let distances = all_pairs_shortest_paths(&graph).unwrap();
        assert_eq!(distances[&0][&0], Some(0.0));
        assert_eq!(distances[&0][&1], Some(-2.0));
        assert_eq!(distances[&0][&2], Some(1.0));
        assert_eq!(distances[&1][&0], Some(-1.0));
        assert_eq!(distances[&1][&1], Some(0.0));
        assert_eq!(distances[&1][&2], Some(3.0));
        assert_eq!(distances[&2][&0], Some(-2.0));
        assert_eq!(distances[&2][&1], Some(-4.0));
        assert_eq!(distances[&2][&2], Some(0.0));
    }

    #[test]
    fn test_negative_cycle() {
        let mut graph = Graph::new();
        graph.add_edge(0, 1, 1.0);
        graph.add_edge(1, 2, -3.0);
        graph.add_edge(2, 0, 1.0);

        assert!(matches!(
            all_pairs_shortest_paths(&graph),
            Err(GraphError::InvalidInput(_))
        ));
    }

    #[test]
    fn test_disconnected_graph() {
        let mut graph = Graph::new();
        graph.add_edge(0, 1, 1.0);
        graph.add_vertex(2);

        let distances = all_pairs_shortest_paths(&graph).unwrap();
        assert_eq!(distances[&0][&1], Some(1.0));
        assert_eq!(distances[&0][&2], None);
        assert_eq!(distances[&1][&0], None);
        assert_eq!(distances[&2][&0], None);
    }

    #[test]
    fn test_single_vertex() {
        let mut graph = Graph::new();
        graph.add_vertex(0);

        let distances = all_pairs_shortest_paths(&graph).unwrap();
        assert_eq!(distances[&0][&0], Some(0.0));
    }

    #[test]
    fn test_empty_graph() {
        let graph: Graph<i32, f64> = Graph::new();
        let distances = all_pairs_shortest_paths(&graph).unwrap();
        assert!(distances.is_empty());
    }
}