use crate::cs::error::{Error, Result};
use crate::cs::graph::Graph;
use crate::cs::graph::bellman_ford;
use crate::cs::graph::dijkstra;
use std::collections::HashMap;
use std::fmt::Debug;
use std::hash::Hash;
use num_traits::{Float, Zero};

/// Computes all-pairs shortest paths using Johnson's algorithm.
/// Returns a map of (source, target) pairs to their shortest path distances.
/// Returns None for unreachable vertices.
/// Returns an error if the graph contains a negative cycle.
///
/// # Arguments
/// * `graph` - A directed graph
///
/// # Returns
/// * `Ok(HashMap<(V, V), Option<W>>)` - A map of vertex pairs to their shortest path distances
/// * `Err(Error)` - If the graph contains a negative cycle or is not directed
///
/// # Complexity
/// * Time: O(VE log V) where V is the number of vertices and E is the number of edges
/// * Space: O(V²)
pub fn all_pairs_shortest_paths<V, W>(
    graph: &Graph<V, W>,
) -> Result<HashMap<(V, V), Option<W>>>
where
    V: Hash + Eq + Copy + Debug + Ord,
    W: Float + Zero + Copy + Debug,
{
    if !graph.is_directed() {
        return Err(Error::InvalidInput("Graph must be directed".to_string()));
    }

    let vertices: Vec<_> = graph.vertices().copied().collect();
    if vertices.is_empty() {
        return Ok(HashMap::new());
    }

    // Add a new vertex q and zero-weight edges to all other vertices
    let mut g = graph.clone();
    let q = match vertices.iter().max() {
        Some(&max_v) => max_v,
        None => return Ok(HashMap::new()),
    };
    for &v in &vertices {
        g.add_edge(q, v, W::zero());
    }

    // Run Bellman-Ford from q to get vertex potentials
    let potentials = match bellman_ford::shortest_paths(&g, &q) {
        Ok(p) => p,
        Err(_) => return Err(Error::NegativeCycle),
    };

    // Remove vertex q and its edges
    g = graph.clone();

    // Reweight edges using potentials
    let mut reweighted_graph = Graph::new();
    for &v in &vertices {
        reweighted_graph.add_vertex(v);
    }

    for (u, v, weight) in graph.edges() {
        if let (Some(Some(h_u)), Some(Some(h_v))) = (potentials.get(u), potentials.get(v)) {
            let reweighted = weight + *h_u - *h_v;
            reweighted_graph.add_edge(*u, *v, reweighted);
        }
    }

    // Compute shortest paths for each vertex
    let mut distances = HashMap::new();
    for &source in &vertices {
        let shortest_paths = match dijkstra::shortest_paths(&reweighted_graph, &source) {
            Ok(paths) => paths,
            Err(_) => continue,
        };
    
        for &target in &vertices {
            let dist = if source == target {
                Some(W::zero())
            } else {
                match (
                    shortest_paths.get(&target),  // Option<&Option<W>>
                    potentials.get(&source),      // Option<&Option<W>>
                    potentials.get(&target),      // Option<&Option<W>>
                ) {
                    (Some(Some(d)), Some(Some(h_source)), Some(Some(h_target))) => {
                        let potential_diff = *h_target - *h_source;
                        Some(*d + potential_diff) // Now d is a &W
                    }
                    _ => None,
                }
            };
            distances.insert((source, target), dist);
        }
    }
    Ok(distances)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_johnson_simple_graph() {
        let mut graph = Graph::new();
        for i in 0..3 {
            graph.add_vertex(i);
        }
        graph.add_edge(0, 1, -2.0);
        graph.add_edge(1, 2, 3.0);
        graph.add_edge(2, 0, -2.0);

        let distances = all_pairs_shortest_paths(&graph).unwrap();
        assert_eq!(distances[&(0, 0)], Some(0.0));
        assert_eq!(distances[&(0, 1)], Some(-2.0));
        assert_eq!(distances[&(0, 2)], Some(1.0));
        assert_eq!(distances[&(1, 0)], Some(-2.0));
        assert_eq!(distances[&(1, 1)], Some(0.0));
        assert_eq!(distances[&(1, 2)], Some(3.0));
        assert_eq!(distances[&(2, 0)], Some(-2.0));
        assert_eq!(distances[&(2, 1)], Some(-4.0));
        assert_eq!(distances[&(2, 2)], Some(0.0));
    }

    #[test]
    fn test_johnson_negative_cycle() {
        let mut graph = Graph::new();
        for i in 0..3 {
            graph.add_vertex(i);
        }
        graph.add_edge(0, 1, 1.0);
        graph.add_edge(1, 2, -3.0);
        graph.add_edge(2, 0, 1.0);

        assert!(matches!(
            all_pairs_shortest_paths(&graph),
            Err(Error::NegativeCycle)
        ));
    }

    #[test]
    fn test_johnson_disconnected_graph() {
        let mut graph = Graph::new();
        for i in 0..3 {
            graph.add_vertex(i);
        }
        graph.add_edge(0, 1, 1.0);

        let distances = all_pairs_shortest_paths(&graph).unwrap();
        assert_eq!(distances[&(0, 0)], Some(0.0));
        assert_eq!(distances[&(0, 1)], Some(1.0));
        assert_eq!(distances[&(0, 2)], None);
        assert_eq!(distances[&(1, 0)], None);
        assert_eq!(distances[&(1, 1)], Some(0.0));
        assert_eq!(distances[&(1, 2)], None);
        assert_eq!(distances[&(2, 0)], None);
        assert_eq!(distances[&(2, 1)], None);
        assert_eq!(distances[&(2, 2)], Some(0.0));
    }

    #[test]
    fn test_johnson_single_vertex() {
        let mut graph = Graph::new();
        graph.add_vertex(0);

        let distances = all_pairs_shortest_paths(&graph).unwrap();
        assert_eq!(distances[&(0, 0)], Some(0.0));
    }

    #[test]
    fn test_johnson_empty_graph() {
        let graph: Graph<i32, f64> = Graph::new();
        let distances = all_pairs_shortest_paths(&graph).unwrap();
        assert!(distances.is_empty());
    }
}
