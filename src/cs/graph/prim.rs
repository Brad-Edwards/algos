use num_traits::{Float, Zero};
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashSet};
use std::fmt::Debug;
use std::hash::Hash;

use crate::error::{GraphError, Result};
use crate::graph::Graph;

/// Entry in the priority queue for Prim's algorithm
#[derive(Copy, Clone, Debug)]
struct Edge<V, W> {
    vertex: V,
    cost: W,
    parent: V,
}

impl<V: Eq, W: PartialOrd> Eq for Edge<V, W> {}

impl<V: Eq, W: PartialOrd> PartialEq for Edge<V, W> {
    fn eq(&self, other: &Self) -> bool {
        self.vertex == other.vertex
    }
}

impl<V: Eq, W: PartialOrd> PartialOrd for Edge<V, W> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        // Reverse ordering for min-heap
        other.cost.partial_cmp(&self.cost)
    }
}

impl<V: Eq, W: PartialOrd> Ord for Edge<V, W> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap_or(Ordering::Equal)
    }
}

/// Computes the minimum spanning tree (MST) of an undirected graph using Prim's algorithm.
///
/// # Arguments
/// * `graph` - The undirected graph to find MST in
/// * `start` - The starting vertex for the algorithm
///
/// # Returns
/// * `Ok((total_weight, edges))` - The total weight of the MST and a vector of edges in the MST
/// * `Err(GraphError)` - If the graph is directed or vertices are not found
///
/// # Examples
/// ```
/// use algos::cs::graph::{Graph, prim};
///
/// let mut graph = Graph::new_undirected();
/// graph.add_edge(0, 1, 4.0);
/// graph.add_edge(0, 2, 2.0);
/// graph.add_edge(1, 2, 1.0);
///
/// let (weight, edges) = prim::minimum_spanning_tree(&graph, &0).unwrap();
/// ```
///
/// # Complexity
/// * Time: O((V + E) log V) where V is the number of vertices and E is the number of edges
/// * Space: O(V)
///
/// # Errors
/// * `InvalidInput` if the graph is directed or contains negative weights
/// * `VertexNotFound` if the start vertex doesn't exist
/// * `InvalidInput` if the graph is not connected
pub fn minimum_spanning_tree<V, W>(graph: &Graph<V, W>, start: &V) -> Result<(W, Vec<(V, V, W)>)>
where
    V: Hash + Eq + Copy + Debug,
    W: Float + Zero + Copy + Debug,
{
    // Validate graph is undirected
    if graph.is_directed() {
        return Err(GraphError::invalid_input(
            "Prim's algorithm requires an undirected graph",
        ));
    }

    // Validate start vertex exists
    if !graph.has_vertex(start) {
        return Err(GraphError::VertexNotFound);
    }

    // Validate graph is connected
    if !graph.is_connected() {
        return Err(GraphError::invalid_input(
            "Prim's algorithm requires a connected graph",
        ));
    }

    let mut total_weight = W::zero();
    let mut mst_edges = Vec::new();
    let mut visited = HashSet::new();
    let mut heap = BinaryHeap::new();

    // Initialize with start vertex
    visited.insert(*start);
    if let Ok(neighbors) = graph.neighbors(start) {
        for (neighbor, weight) in neighbors {
            // Validate non-negative weights
            if weight < W::zero() {
                return Err(GraphError::invalid_input(
                    "Prim's algorithm requires non-negative weights",
                ));
            }
            heap.push(Edge {
                vertex: *neighbor,
                cost: weight,
                parent: *start,
            });
        }
    }

    // Process edges until MST is complete
    while let Some(Edge {
        vertex,
        cost,
        parent,
    }) = heap.pop()
    {
        if visited.insert(vertex) {
            total_weight = total_weight + cost;
            mst_edges.push((parent, vertex, cost));

            // Add edges to unvisited neighbors
            if let Ok(neighbors) = graph.neighbors(&vertex) {
                for (neighbor, weight) in neighbors {
                    if !visited.contains(neighbor) {
                        if weight < W::zero() {
                            return Err(GraphError::invalid_input(
                                "Prim's algorithm requires non-negative weights",
                            ));
                        }
                        heap.push(Edge {
                            vertex: *neighbor,
                            cost: weight,
                            parent: vertex,
                        });
                    }
                }
            }
        }
    }

    Ok((total_weight, mst_edges))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prim_simple_mst() {
        let mut graph = Graph::new_undirected();
        graph.add_edge(0, 1, 1.0);
        graph.add_edge(1, 2, 2.0);
        graph.add_edge(0, 2, 3.0);

        let (weight, edges) = minimum_spanning_tree(&graph, &0).unwrap();
        assert_eq!(weight, 3.0);
        assert_eq!(edges.len(), 2);
    }

    #[test]
    fn test_prim_directed_graph() {
        let mut graph = Graph::new();
        graph.add_edge(0, 1, 1.0);

        assert!(matches!(
            minimum_spanning_tree(&graph, &0),
            Err(GraphError::InvalidInput(_))
        ));
    }

    #[test]
    fn test_prim_disconnected_graph() {
        let mut graph = Graph::new_undirected();
        graph.add_edge(0, 1, 1.0);
        graph.add_vertex(2); // Disconnected vertex

        assert!(matches!(
            minimum_spanning_tree(&graph, &0),
            Err(GraphError::InvalidInput(_))
        ));
    }

    #[test]
    fn test_prim_negative_weights() {
        let mut graph = Graph::new_undirected();
        graph.add_edge(0, 1, -1.0);

        assert!(matches!(
            minimum_spanning_tree(&graph, &0),
            Err(GraphError::InvalidInput(_))
        ));
    }

    #[test]
    fn test_prim_vertex_not_found() {
        let graph: Graph<i32, f64> = Graph::new_undirected();
        assert!(matches!(
            minimum_spanning_tree(&graph, &0),
            Err(GraphError::VertexNotFound)
        ));
    }

    #[test]
    fn test_prim_complex_graph() {
        let mut graph = Graph::new_undirected();
        graph.add_edge(0, 1, 4.0);
        graph.add_edge(0, 2, 2.0);
        graph.add_edge(1, 2, 1.0);
        graph.add_edge(1, 3, 3.0);
        graph.add_edge(2, 3, 5.0);

        let (weight, edges) = minimum_spanning_tree(&graph, &0).unwrap();
        assert_eq!(weight, 6.0);
        assert_eq!(edges.len(), 3);
    }

    #[test]
    fn test_prim_cycle() {
        let mut graph = Graph::new_undirected();
        graph.add_edge(0, 1, 1.0);
        graph.add_edge(1, 2, 2.0);
        graph.add_edge(2, 0, 3.0);

        let (weight, edges) = minimum_spanning_tree(&graph, &0).unwrap();
        assert_eq!(weight, 3.0);
        assert_eq!(edges.len(), 2);
    }

    #[test]
    fn test_prim_parallel_edges() {
        let mut graph = Graph::new_undirected();
        graph.add_edge(0, 1, 2.0);
        graph.add_edge(0, 1, 1.0); // Parallel edge with lower weight

        let (weight, edges) = minimum_spanning_tree(&graph, &0).unwrap();
        assert_eq!(weight, 1.0); // Should use the lower weight edge
        assert_eq!(edges.len(), 1);
    }

    #[test]
    fn test_prim_large_graph() {
        let mut graph = Graph::new_undirected();
        // Create a circular graph with 1000 vertices
        for i in 0..999 {
            graph.add_edge(i, i + 1, 1.0);
        }
        graph.add_edge(999, 0, 1.0);

        let (weight, edges) = minimum_spanning_tree(&graph, &0).unwrap();
        assert_eq!(weight, 999.0);
        assert_eq!(edges.len(), 999);
    }
}
