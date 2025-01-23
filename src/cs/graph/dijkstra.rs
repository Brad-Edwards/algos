use num_traits::{Float, Zero};
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};
use std::fmt::Debug;
use std::hash::Hash;

use crate::error::{GraphError, Result};
use crate::graph::Graph;

/// Entry in the priority queue for Dijkstra's algorithm
#[derive(Copy, Clone, Debug)]
struct State<V, W> {
    vertex: V,
    cost: W,
}

impl<V: Eq, W: PartialOrd> Eq for State<V, W> {}

impl<V: Eq, W: PartialOrd> PartialEq for State<V, W> {
    fn eq(&self, other: &Self) -> bool {
        self.vertex == other.vertex
    }
}

impl<V: Eq, W: PartialOrd> PartialOrd for State<V, W> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        // Reverse ordering for min-heap
        other.cost.partial_cmp(&self.cost)
    }
}

impl<V: Eq, W: PartialOrd> Ord for State<V, W> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap_or(Ordering::Equal)
    }
}

/// Computes shortest paths from a source vertex to all other vertices using Dijkstra's algorithm.
///
/// # Arguments
/// * `graph` - The weighted graph to search
/// * `source` - The source vertex to compute paths from
///
/// # Returns
/// * `Ok(distances)` - A map from vertex to the shortest distance from the source
/// * `Err(GraphError)` - If the source vertex doesn't exist or if negative weights are present
///
/// # Examples
/// ```
/// use algos::cs::graph::{Graph, dijkstra};
///
/// let mut graph = Graph::new();
/// graph.add_edge(0, 1, 4.0);
/// graph.add_edge(0, 2, 2.0);
/// graph.add_edge(1, 3, 3.0);
/// graph.add_edge(2, 1, 1.0);
///
/// let distances = dijkstra::shortest_paths(&graph, &0).unwrap();
/// ```
///
/// # Complexity
/// * Time: O((V + E) log V) where V is the number of vertices and E is the number of edges
/// * Space: O(V)
///
/// # Errors
/// * `VertexNotFound` if the source vertex doesn't exist in the graph
/// * `InvalidInput` if a negative weight is found
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

    let mut heap = BinaryHeap::new();
    heap.push(State {
        vertex: *source,
        cost: W::zero(),
    });

    while let Some(State { vertex, cost }) = heap.pop() {
        // Skip if we've found a better path
        if let Some(Some(best)) = distances.get(&vertex) {
            if *best < cost {
                continue;
            }
        }

        // Check all neighbors
        if let Ok(neighbors) = graph.neighbors(&vertex) {
            for (neighbor, edge_cost) in neighbors {
                // Validate non-negative weights
                if edge_cost < W::zero() {
                    return Err(GraphError::invalid_input(
                        "Dijkstra's algorithm requires non-negative weights",
                    ));
                }

                let next = State {
                    vertex: *neighbor,
                    cost: cost + edge_cost,
                };

                // Update distance if we found a better path
                let update = match distances.get(neighbor) {
                    None => true,
                    Some(None) => true,
                    Some(Some(best)) => next.cost < *best,
                };

                if update {
                    distances.insert(*neighbor, Some(next.cost));
                    heap.push(next);
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
    fn test_dijkstra_simple_path() {
        let mut graph = Graph::new();
        graph.add_edge(0, 1, 1.0);
        graph.add_edge(1, 2, 2.0);
        graph.add_edge(0, 2, 4.0);

        let distances = shortest_paths(&graph, &0).unwrap();
        assert_eq!(distances[&0], Some(0.0));
        assert_eq!(distances[&1], Some(1.0));
        assert_eq!(distances[&2], Some(3.0));
    }

    #[test]
    fn test_dijkstra_unreachable_vertices() {
        let mut graph = Graph::new();
        graph.add_edge(0, 1, 1.0);
        graph.add_vertex(2);

        let distances = shortest_paths(&graph, &0).unwrap();
        assert_eq!(distances[&0], Some(0.0));
        assert_eq!(distances[&1], Some(1.0));
        assert_eq!(distances[&2], None);
    }

    #[test]
    fn test_dijkstra_negative_weights() {
        let mut graph = Graph::new();
        graph.add_edge(0, 1, -1.0);

        assert!(matches!(
            shortest_paths(&graph, &0),
            Err(GraphError::InvalidInput(_))
        ));
    }

    #[test]
    fn test_dijkstra_vertex_not_found() {
        let graph: Graph<i32, f64> = Graph::new();
        assert!(matches!(
            shortest_paths(&graph, &0),
            Err(GraphError::VertexNotFound)
        ));
    }

    #[test]
    fn test_dijkstra_multiple_paths() {
        let mut graph = Graph::new();
        graph.add_edge(0, 1, 4.0);
        graph.add_edge(0, 2, 2.0);
        graph.add_edge(1, 3, 3.0);
        graph.add_edge(2, 1, 1.0);
        graph.add_edge(2, 3, 5.0);

        let distances = shortest_paths(&graph, &0).unwrap();
        assert_eq!(distances[&0], Some(0.0));
        assert_eq!(distances[&1], Some(3.0)); // via 0->2->1
        assert_eq!(distances[&2], Some(2.0));
        assert_eq!(distances[&3], Some(6.0)); // via 0->2->1->3
    }

    #[test]
    fn test_dijkstra_undirected_graph() {
        let mut graph = Graph::new_undirected();
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
    fn test_dijkstra_cycle() {
        let mut graph = Graph::new();
        graph.add_edge(0, 1, 1.0);
        graph.add_edge(1, 2, 2.0);
        graph.add_edge(2, 0, 3.0);

        let distances = shortest_paths(&graph, &0).unwrap();
        assert_eq!(distances[&0], Some(0.0));
        assert_eq!(distances[&1], Some(1.0));
        assert_eq!(distances[&2], Some(3.0));
    }

    #[test]
    fn test_dijkstra_self_loop() {
        let mut graph = Graph::new();
        graph.add_edge(0, 0, 1.0);
        graph.add_edge(0, 1, 2.0);

        let distances = shortest_paths(&graph, &0).unwrap();
        assert_eq!(distances[&0], Some(0.0)); // Self-loop doesn't affect shortest path to self
        assert_eq!(distances[&1], Some(2.0));
    }

    #[test]
    fn test_dijkstra_parallel_edges() {
        let mut graph = Graph::new();
        // Add two edges between same vertices
        graph.add_edge(0, 1, 2.0);
        graph.add_edge(0, 1, 1.0);

        let distances = shortest_paths(&graph, &0).unwrap();
        assert_eq!(distances[&0], Some(0.0));
        assert_eq!(distances[&1], Some(1.0)); // Should use the shorter edge
    }

    #[test]
    fn test_dijkstra_large_graph() {
        let mut graph = Graph::new();
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
