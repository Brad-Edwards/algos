use std::collections::HashMap;
use std::hash::Hash;
use std::fmt::Debug;
use num_traits::{Float, Zero};

use crate::error::{GraphError, Result};
use crate::graph::Graph;

/// Computes shortest paths from a source vertex to all other vertices using the Bellman-Ford algorithm.
///
/// # Arguments
/// * `graph` - The weighted graph to search
/// * `source` - The source vertex to compute paths from
///
/// # Returns
/// * `Ok(distances)` - A map from vertex to the shortest distance from the source
/// * `Err(GraphError)` - If the source vertex doesn't exist or if a negative cycle is detected
///
/// # Examples
/// ```
/// use blocks_cs_graph::{Graph, algorithms::bellman_ford};
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
/// let distances = bellman_ford::shortest_paths(&graph, &0)
///     .expect("Graph should be valid for Bellman-Ford algorithm");
///
/// assert_eq!(distances[&3], Some(0.0)); // Path 0->2->1->3 with total weight 0
/// ```
///
/// # Complexity
/// * Time: O(VE) where V is the number of vertices and E is the number of edges
/// * Space: O(V)
///
/// # Errors
/// * `VertexNotFound` if the source vertex doesn't exist in the graph
/// * `NegativeCycle` if a negative cycle is detected
pub fn shortest_paths<V, W>(graph: &Graph<V, W>, source: &V) -> Result<HashMap<V, Option<W>>>
where
    V: Hash + Eq + Copy + Debug,
    W: Float + Zero + Copy + Debug,
{
    // Validate source vertex exists
    if !graph.has_vertex(source) {
        return Err(GraphError::VertexNotFound);
    }

    // Initialize distances with infinity (None) for all vertices
    let mut distances = HashMap::new();
    for v in graph.vertices() {
        distances.insert(*v, if v == source { Some(W::zero()) } else { None });
    }

    // Relax edges |V| - 1 times
    let vertex_count = graph.vertex_count();
    for _ in 0..vertex_count - 1 {
        let mut updated = false;

        // Check each edge
        for (u, v, weight) in graph.edges() {
            if let Some(Some(dist_u)) = distances.get(u) {
                let new_dist = *dist_u + weight;
                let better = match distances.get(v) {
                    None => true,
                    Some(None) => true,
                    Some(Some(current)) => new_dist < *current,
                };

                if better {
                    distances.insert(*v, Some(new_dist));
                    updated = true;
                }
            }
        }

        // Early termination if no updates were made
        if !updated {
            break;
        }
    }

    // Check for negative cycles
    for (u, v, weight) in graph.edges() {
        if let Some(Some(dist_u)) = distances.get(u) {
            let new_dist = *dist_u + weight;
            if let Some(Some(dist_v)) = distances.get(v) {
                if new_dist < *dist_v {
                    return Err(GraphError::NegativeCycle);
                }
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

        let distances = shortest_paths(&graph, &0).unwrap();
        assert_eq!(distances[&0], Some(0.0));
        assert_eq!(distances[&1], Some(1.0));
        assert_eq!(distances[&2], Some(3.0));
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

        let distances = shortest_paths(&graph, &0).unwrap();
        assert_eq!(distances[&0], Some(0.0));
        assert_eq!(distances[&1], Some(3.0));
        assert_eq!(distances[&2], Some(2.0));
        assert_eq!(distances[&3], Some(0.0));
    }

    #[test]
    fn test_negative_cycle() {
        let mut graph: Graph<i32, f64> = Graph::new();
        graph.add_edge(0, 1, 1.0);
        graph.add_edge(1, 2, -1.0);
        graph.add_edge(2, 3, -1.0);
        graph.add_edge(3, 1, -1.0);

        assert!(matches!(
            shortest_paths(&graph, &0),
            Err(GraphError::NegativeCycle)
        ));
    }

    #[test]
    fn test_vertex_not_found() {
        let graph: Graph<i32, f64> = Graph::new();
        assert!(matches!(
            shortest_paths(&graph, &0),
            Err(GraphError::VertexNotFound)
        ));
    }

    #[test]
    fn test_unreachable_vertices() {
        let mut graph: Graph<i32, f64> = Graph::new();
        graph.add_edge(0, 1, 1.0);
        graph.add_vertex(2);

        let distances = shortest_paths(&graph, &0).unwrap();
        assert_eq!(distances[&0], Some(0.0));
        assert_eq!(distances[&1], Some(1.0));
        assert_eq!(distances[&2], None);
    }

    #[test]
    fn test_undirected_graph() {
        let mut graph: Graph<i32, f64> = Graph::new_undirected();
        graph.add_edge(0, 1, 1.0);
        graph.add_edge(1, 2, 2.0);

        let distances = shortest_paths(&graph, &0).unwrap();
        assert_eq!(distances[&0], Some(0.0));
        assert_eq!(distances[&1], Some(1.0));
        assert_eq!(distances[&2], Some(3.0));

        // Test from another source
        let distances = shortest_paths(&graph, &2).unwrap();
        assert_eq!(distances[&0], Some(3.0));
        assert_eq!(distances[&1], Some(2.0));
        assert_eq!(distances[&2], Some(0.0));
    }

    #[test]
    fn test_cycle() {
        let mut graph: Graph<i32, f64> = Graph::new();
        graph.add_edge(0, 1, 1.0);
        graph.add_edge(1, 2, 2.0);
        graph.add_edge(2, 0, 3.0);

        let distances = shortest_paths(&graph, &0).unwrap();
        assert_eq!(distances[&0], Some(0.0));
        assert_eq!(distances[&1], Some(1.0));
        assert_eq!(distances[&2], Some(3.0));
    }

    #[test]
    fn test_self_loop() {
        let mut graph: Graph<i32, f64> = Graph::new();
        graph.add_vertex(0);
        graph.add_vertex(1);
        graph.add_edge(0, 0, 1.0);
        graph.add_edge(0, 1, 2.0);

        let distances = shortest_paths(&graph, &0).unwrap();
        assert_eq!(distances[&0], Some(0.0)); // Self-loop doesn't affect shortest path to self
        assert_eq!(distances[&1], Some(2.0));
    }

    #[test]
    fn test_parallel_edges() {
        let mut graph: Graph<i32, f64> = Graph::new();
        graph.add_vertex(0);
        graph.add_vertex(1);
        // Add two edges between same vertices
        graph.add_edge(0, 1, 2.0);
        graph.add_edge(0, 1, 1.0);

        let distances = shortest_paths(&graph, &0).unwrap();
        assert_eq!(distances[&0], Some(0.0));
        assert_eq!(distances[&1], Some(1.0)); // Should use the shorter edge
    }

    #[test]
    fn test_large_graph() {
        let mut graph: Graph<i32, f64> = Graph::new();
        // Create a line graph with 1000 vertices
        for i in 0..999 {
            graph.add_edge(i, i + 1, 1.0);
        }

        let distances = shortest_paths(&graph, &0).unwrap();
        assert_eq!(distances[&0], Some(0.0));
        assert_eq!(distances[&500], Some(500.0));
        assert_eq!(distances[&999], Some(999.0));
    }
}