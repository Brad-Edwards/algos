use num_traits::{Float, Zero};
use std::collections::HashMap;
use std::fmt::Debug;
use std::hash::Hash;

use crate::cs::error::{Error, Result};
use crate::cs::graph::Graph;

/// Computes the transitive closure of a directed graph using Warshall's algorithm.
///
/// The transitive closure of a directed graph G is a graph G' with the same vertices as G,
/// and an edge from vertex i to vertex j if and only if there exists a path from i to j in G.
///
/// # Arguments
/// * `graph` - The directed graph to compute transitive closure for
///
/// # Returns
/// * `Ok(reachability)` - A map from (source, target) pairs to a boolean indicating if target is reachable from source
/// * `Err(Error)` - If the graph is undirected
///
/// # Examples
/// ```
/// use blocks_cs_graph::{Graph, algorithms::warshall};
///
/// let mut graph = Graph::new();
/// graph.add_edge(0, 1, 1.0);
/// graph.add_edge(1, 2, 1.0);
/// // Note: No direct edge from 0 to 2, but there is a path
///
/// let closure = warshall::transitive_closure(&graph).unwrap();
/// assert!(closure[&(0, 2)]); // 0 can reach 2 through 1
/// ```
///
/// # Complexity
/// * Time: O(V³) where V is the number of vertices
/// * Space: O(V²)
pub fn transitive_closure<V, W>(graph: &Graph<V, W>) -> Result<HashMap<(V, V), bool>>
where
    V: Hash + Eq + Copy + Debug,
    W: Float + Zero + Copy + Debug,
{
    // Validate graph is directed
    if !graph.is_directed() {
        return Err(Error::invalid_input("Warshall's algorithm requires a directed graph"));
    }

    let vertices: Vec<_> = graph.vertices().copied().collect();
    let mut reachability = HashMap::new();

    // Initialize reachability matrix
    for &i in &vertices {
        for &j in &vertices {
            // A vertex can always reach itself
            let reachable = i == j || graph.has_edge(&i, &j);
            reachability.insert((i, j), reachable);
        }
    }

    // Warshall's algorithm: if i->k and k->j then i->j
    for &k in &vertices {
        for &i in &vertices {
            for &j in &vertices {
                if let (Some(&ik), Some(&kj)) = (reachability.get(&(i, k)), reachability.get(&(k, j))) {
                    if ik && kj {
                        reachability.insert((i, j), true);
                    }
                }
            }
        }
    }

    Ok(reachability)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_warshall_simple_path() {
        let mut graph: Graph<i32, f64> = Graph::new();
        graph.add_vertex(0);
        graph.add_vertex(1);
        graph.add_vertex(2);
        graph.add_edge(0, 1, 1.0);
        graph.add_edge(1, 2, 1.0);

        let closure = transitive_closure(&graph).unwrap();
        assert!(closure[&(0, 0)], "vertex should reach itself");
        assert!(closure[&(1, 1)], "vertex should reach itself");
        assert!(closure[&(2, 2)], "vertex should reach itself");
        assert!(closure[&(0, 1)], "direct edge");
        assert!(closure[&(1, 2)], "direct edge");
        assert!(closure[&(0, 2)], "transitive path");
        assert!(!closure[&(2, 1)], "no backward path");
        assert!(!closure[&(2, 0)], "no backward path");
        assert!(!closure[&(1, 0)], "no backward path");
    }

    #[test]
    fn test_warshall_cycle() {
        let mut graph: Graph<i32, f64> = Graph::new();
        for i in 0..3 {
            graph.add_vertex(i);
        }
        graph.add_edge(0, 1, 1.0);
        graph.add_edge(1, 2, 1.0);
        graph.add_edge(2, 0, 1.0);

        let closure = transitive_closure(&graph).unwrap();
        // Every vertex can reach every other vertex in a cycle
        for i in 0..3 {
            for j in 0..3 {
                assert!(closure[&(i, j)], "vertex {i} should reach {j} in cycle");
            }
        }
    }

    #[test]
    fn test_warshall_disconnected() {
        let mut graph: Graph<i32, f64> = Graph::new();
        for i in 0..3 {
            graph.add_vertex(i);
        }
        graph.add_edge(0, 1, 1.0);

        let closure = transitive_closure(&graph).unwrap();
        assert!(closure[&(0, 0)], "vertex should reach itself");
        assert!(closure[&(1, 1)], "vertex should reach itself");
        assert!(closure[&(2, 2)], "vertex should reach itself");
        assert!(closure[&(0, 1)], "direct edge");
        assert!(!closure[&(0, 2)], "no path to isolated vertex");
        assert!(!closure[&(1, 2)], "no path to isolated vertex");
        assert!(!closure[&(2, 0)], "no path from isolated vertex");
        assert!(!closure[&(2, 1)], "no path from isolated vertex");
    }

    #[test]
    fn test_warshall_self_loop() {
        let mut graph: Graph<i32, f64> = Graph::new();
        for i in 0..2 {
            graph.add_vertex(i);
        }
        graph.add_edge(0, 0, 1.0);
        graph.add_edge(0, 1, 1.0);

        let closure = transitive_closure(&graph).unwrap();
        assert!(closure[&(0, 0)], "vertex should reach itself + self loop");
        assert!(closure[&(1, 1)], "vertex should reach itself");
        assert!(closure[&(0, 1)], "direct edge");
        assert!(!closure[&(1, 0)], "no backward path");
    }

    #[test]
    fn test_warshall_undirected_graph() {
        let mut graph: Graph<i32, f64> = Graph::new_undirected();
        graph.add_edge(0, 1, 1.0);

        assert!(matches!(
            transitive_closure(&graph),
            Err(Error::InvalidInput(_))
        ));
    }

    #[test]
    fn test_warshall_empty_graph() {
        let graph: Graph<i32, f64> = Graph::new();
        let closure = transitive_closure(&graph).unwrap();
        assert!(closure.is_empty());
    }

    #[test]
    fn test_warshall_single_vertex() {
        let mut graph: Graph<i32, f64> = Graph::new();
        graph.add_vertex(0);

        let closure = transitive_closure(&graph).unwrap();
        assert!(closure[&(0, 0)], "vertex should reach itself");
    }

    #[test]
    fn test_warshall_complex_graph() {
        let mut graph: Graph<i32, f64> = Graph::new();
        // Create a more complex graph structure
        for i in 0..6 {
            graph.add_vertex(i);
        }
        graph.add_edge(0, 1, 1.0);
        graph.add_edge(1, 2, 1.0);
        graph.add_edge(2, 3, 1.0);
        graph.add_edge(3, 1, 1.0); // Creates a cycle 1->2->3->1
        graph.add_edge(0, 4, 1.0);
        graph.add_edge(4, 5, 1.0);

        let closure = transitive_closure(&graph).unwrap();
        // Check self-reachability
        for i in 0..6 {
            assert!(closure[&(i, i)], "vertex {i} should reach itself");
        }
        // Check cycle reachability
        for i in 1..=3 {
            for j in 1..=3 {
                assert!(closure[&(i, j)], "vertex {i} should reach {j} in cycle");
            }
        }
        // Check path reachability
        assert!(closure[&(0, 5)], "should reach 5 through 4");
        assert!(!closure[&(5, 0)], "cannot reach 0 from 5");
        assert!(closure[&(0, 3)], "should reach 3 through path 0->1->2->3");
    }

    #[test]
    fn test_warshall_parallel_paths() {
        let mut graph: Graph<i32, f64> = Graph::new();
        for i in 0..3 {
            graph.add_vertex(i);
        }
        // Two paths from 0 to 2: 0->1->2 and 0->2
        graph.add_edge(0, 1, 1.0);
        graph.add_edge(1, 2, 1.0);
        graph.add_edge(0, 2, 1.0);

        let closure = transitive_closure(&graph).unwrap();
        for i in 0..3 {
            assert!(closure[&(i, i)], "vertex {i} should reach itself");
        }
        assert!(closure[&(0, 1)], "direct edge");
        assert!(closure[&(1, 2)], "direct edge");
        assert!(closure[&(0, 2)], "should be reachable through either path");
        assert!(!closure[&(2, 0)], "no backward path");
        assert!(!closure[&(2, 1)], "no backward path");
    }
}
