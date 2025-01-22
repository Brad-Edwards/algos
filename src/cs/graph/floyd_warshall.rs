use std::collections::HashMap;
use std::hash::Hash;
use std::fmt::Debug;
use num_traits::{Float, Zero};

use crate::error::{GraphError, Result};
use crate::graph::Graph;

/// Computes all-pairs shortest paths using the Floyd-Warshall algorithm.
///
/// # Arguments
/// * `graph` - The weighted graph to compute shortest paths for
///
/// # Returns
/// * `Ok(distances)` - A map from (source, target) pairs to the shortest distance between them
/// * `Err(GraphError)` - If a negative cycle is detected
///
/// # Examples
/// ```
/// use blocks_cs_graph::{Graph, algorithms::floyd_warshall};
///
/// let mut graph = Graph::new();
/// graph.add_vertex(0);
/// graph.add_vertex(1);
/// graph.add_vertex(2);
/// graph.add_vertex(3);
/// graph.add_edge(0, 1, 4.0);
/// graph.add_edge(0, 2, 2.0);
/// graph.add_edge(1, 3, -3.0); // Negative weights are allowed
/// graph.add_edge(2, 1, 1.0);
/// graph.add_edge(2, 3, 5.0);
///
/// let distances = floyd_warshall::all_pairs_shortest_paths(&graph)
///     .expect("Graph should be valid for Floyd-Warshall algorithm");
///
/// assert_eq!(distances[&(0, 3)], Some(0.0)); // Path 0->2->1->3 with total weight 0
/// ```
///
/// # Complexity
/// * Time: O(V³) where V is the number of vertices
/// * Space: O(V²)
///
/// # Errors
/// * `NegativeCycle` if a negative cycle is detected
pub fn all_pairs_shortest_paths<V, W>(graph: &Graph<V, W>) -> Result<HashMap<(V, V), Option<W>>>
where
    V: Hash + Eq + Copy + Debug,
    W: Float + Zero + Copy + Debug,
{
    // Initialize distances with infinity (None) for all vertex pairs
    let mut distances = HashMap::new();
    let vertices: Vec<_> = graph.vertices().copied().collect();

    // Initialize with direct edges
    for u in &vertices {
        for v in &vertices {
            let dist = if u == v {
                Some(W::zero())
            } else {
                // For parallel edges, use the minimum weight
                let mut min_weight = graph.edge_weight(u, v);
                for (src, dst, weight) in graph.edges() {
                    if u == src && v == dst && Some(weight) < min_weight {
                        min_weight = Some(weight);
                    }
                }
                min_weight
            };
            distances.insert((*u, *v), dist);
        }
    }

    // Floyd-Warshall algorithm
    for k in &vertices {
        for i in &vertices {
            for j in &vertices {
                if let (Some(dist_ik), Some(dist_kj)) = (distances[&(*i, *k)], distances[&(*k, *j)]) {
                    let new_dist = dist_ik + dist_kj;
                    let better = match distances[&(*i, *j)] {
                        None => true,
                        Some(current) => new_dist < current,
                    };

                    if better {
                        distances.insert((*i, *j), Some(new_dist));
                    }
                }
            }
        }
    }

    // Check for negative cycles
    for v in &vertices {
        if let Some(dist) = distances[&(*v, *v)] {
            if dist < W::zero() {
                return Err(GraphError::NegativeCycle);
            }
        }
    }

    Ok(distances)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_path() {
        let mut graph: Graph<i32, f64> = Graph::new();
        graph.add_vertex(0);
        graph.add_vertex(1);
        graph.add_vertex(2);
        graph.add_edge(0, 1, 1.0);
        graph.add_edge(1, 2, 2.0);
        graph.add_edge(0, 2, 4.0);

        let distances = all_pairs_shortest_paths(&graph).unwrap();
        assert_eq!(distances[&(0, 0)], Some(0.0));
        assert_eq!(distances[&(0, 1)], Some(1.0));
        assert_eq!(distances[&(0, 2)], Some(3.0));
        assert_eq!(distances[&(1, 2)], Some(2.0));
        assert_eq!(distances[&(2, 1)], None);
    }

    #[test]
    fn test_negative_weights() {
        let mut graph: Graph<i32, f64> = Graph::new();
        graph.add_vertex(0);
        graph.add_vertex(1);
        graph.add_vertex(2);
        graph.add_vertex(3);
        graph.add_edge(0, 1, 4.0);
        graph.add_edge(0, 2, 2.0);
        graph.add_edge(1, 3, -3.0);
        graph.add_edge(2, 1, 1.0);
        graph.add_edge(2, 3, 5.0);

        let distances = all_pairs_shortest_paths(&graph).unwrap();
        assert_eq!(distances[&(0, 0)], Some(0.0));
        assert_eq!(distances[&(0, 1)], Some(3.0));
        assert_eq!(distances[&(0, 2)], Some(2.0));
        assert_eq!(distances[&(0, 3)], Some(0.0));
    }

    #[test]
    fn test_negative_cycle() {
        let mut graph: Graph<i32, f64> = Graph::new();
        graph.add_vertex(0);
        graph.add_vertex(1);
        graph.add_vertex(2);
        graph.add_vertex(3);
        graph.add_edge(0, 1, 1.0);
        graph.add_edge(1, 2, -1.0);
        graph.add_edge(2, 3, -1.0);
        graph.add_edge(3, 1, -1.0);

        assert!(matches!(
            all_pairs_shortest_paths(&graph),
            Err(GraphError::NegativeCycle)
        ));
    }

    #[test]
    fn test_unreachable_vertices() {
        let mut graph: Graph<i32, f64> = Graph::new();
        graph.add_vertex(0);
        graph.add_vertex(1);
        graph.add_vertex(2);
        graph.add_edge(0, 1, 1.0);

        let distances = all_pairs_shortest_paths(&graph).unwrap();
        assert_eq!(distances[&(0, 0)], Some(0.0));
        assert_eq!(distances[&(0, 1)], Some(1.0));
        assert_eq!(distances[&(0, 2)], None);
        assert_eq!(distances[&(1, 2)], None);
        assert_eq!(distances[&(2, 0)], None);
    }

    #[test]
    fn test_undirected_graph() {
        let mut graph: Graph<i32, f64> = Graph::new_undirected();
        graph.add_vertex(0);
        graph.add_vertex(1);
        graph.add_vertex(2);
        graph.add_edge(0, 1, 1.0);
        graph.add_edge(1, 2, 2.0);

        let distances = all_pairs_shortest_paths(&graph).unwrap();
        assert_eq!(distances[&(0, 2)], Some(3.0));
        assert_eq!(distances[&(2, 0)], Some(3.0));
    }

    #[test]
    fn test_cycle() {
        let mut graph: Graph<i32, f64> = Graph::new();
        graph.add_vertex(0);
        graph.add_vertex(1);
        graph.add_vertex(2);
        graph.add_edge(0, 1, 1.0);
        graph.add_edge(1, 2, 2.0);
        graph.add_edge(2, 0, 3.0);

        let distances = all_pairs_shortest_paths(&graph).unwrap();
        assert_eq!(distances[&(0, 0)], Some(0.0));
        assert_eq!(distances[&(0, 1)], Some(1.0));
        assert_eq!(distances[&(0, 2)], Some(3.0));
        assert_eq!(distances[&(1, 0)], Some(5.0));
    }

    #[test]
    fn test_self_loop() {
        let mut graph: Graph<i32, f64> = Graph::new();
        graph.add_vertex(0);
        graph.add_vertex(1);
        graph.add_edge(0, 0, 1.0);
        graph.add_edge(0, 1, 2.0);

        let distances = all_pairs_shortest_paths(&graph).unwrap();
        assert_eq!(distances[&(0, 0)], Some(0.0)); // Self-loop doesn't affect shortest path to self
        assert_eq!(distances[&(0, 1)], Some(2.0));
    }

    #[test]
    fn test_parallel_edges() {
        let mut graph: Graph<i32, f64> = Graph::new();
        graph.add_vertex(0);
        graph.add_vertex(1);
        // Add two edges between same vertices
        graph.add_edge(0, 1, 2.0);
        graph.add_edge(0, 1, 1.0);

        let distances = all_pairs_shortest_paths(&graph).unwrap();
        assert_eq!(distances[&(0, 0)], Some(0.0));
        assert_eq!(distances[&(0, 1)], Some(1.0)); // Should use the shorter edge
    }

    #[test]
    fn test_large_graph() {
        let mut graph: Graph<i32, f64> = Graph::new();
        // Create a line graph with 100 vertices
        for i in 0..100 {
            graph.add_vertex(i);
            if i > 0 {
                graph.add_edge(i - 1, i, 1.0);
            }
        }

        let distances = all_pairs_shortest_paths(&graph).unwrap();
        assert_eq!(distances[&(0, 0)], Some(0.0));
        assert_eq!(distances[&(0, 50)], Some(50.0));
        assert_eq!(distances[&(0, 99)], Some(99.0));
    }
}