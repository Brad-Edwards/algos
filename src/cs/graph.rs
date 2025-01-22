pub mod bellman_ford;
pub mod dijkstra;
pub mod floyd_warshall;
pub mod johnson;
pub mod kadane;
pub mod kosaraju;
pub mod kruskal;
pub mod prim;
pub mod tarjan;
pub mod topological_sort;
pub mod warshall;

use num_traits::{Float, Zero};
use std::collections::{HashMap, HashSet};
use std::fmt::Debug;
use std::hash::Hash;

use crate::cs::error::{Error, Result};

pub use bellman_ford::shortest_paths as bellman_ford_shortest_paths;
pub use dijkstra::shortest_paths as dijkstra_shortest_paths;
pub use floyd_warshall::all_pairs_shortest_paths as floyd_warshall_all_pairs_shortest_paths;
pub use johnson::all_pairs_shortest_paths as johnson_all_pairs_shortest_paths;
pub use kadane::kadane as kadane_maximum_subarray;
pub use kosaraju::kosaraju as kosaraju_strongly_connected_components;
pub use kruskal::kruskal as kruskal_minimum_spanning_tree;
pub use prim::minimum_spanning_tree as prim_minimum_spanning_tree;
pub use tarjan::strongly_connected_components as tarjan_strongly_connected_components;
pub use topological_sort::sort as topological_sort;
pub use warshall::transitive_closure as warshall_transitive_closure;

/// A weighted graph implementation supporting both directed and undirected graphs.
#[derive(Debug, Clone)]
pub struct Graph<V, W> {
    /// Adjacency list representation: vertex -> list of (target vertex, weight)
    edges: HashMap<V, Vec<(V, W)>>,
    /// Whether the graph is directed
    directed: bool,
}

impl<V, W> Default for Graph<V, W>
where
    V: Hash + Eq + Copy + Debug,
    W: Float + Zero + Copy + Debug,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<V, W> Graph<V, W>
where
    V: Hash + Eq + Copy + Debug,
    W: Float + Zero + Copy + Debug,
{
    /// Creates a new directed graph.
    pub fn new() -> Self {
        Self {
            edges: HashMap::new(),
            directed: true,
        }
    }

    /// Creates a new undirected graph.
    pub fn new_undirected() -> Self {
        Self {
            edges: HashMap::new(),
            directed: false,
        }
    }

    /// Adds a vertex to the graph.
    pub fn add_vertex(&mut self, vertex: V) {
        self.edges.entry(vertex).or_default();
    }

    /// Adds an edge to the graph with the given weight.
    pub fn add_edge(&mut self, from: V, to: V, weight: W) {
        self.edges.entry(from).or_default().push((to, weight));
        if !self.directed {
            self.edges.entry(to).or_default().push((from, weight));
        }
    }

    /// Returns true if the graph contains the vertex.
    pub fn has_vertex(&self, vertex: &V) -> bool {
        self.edges.contains_key(vertex)
    }

    /// Returns true if the graph contains an edge from `from` to `to`.
    pub fn has_edge(&self, from: &V, to: &V) -> bool {
        self.edges
            .get(from)
            .map(|edges| edges.iter().any(|(v, _)| v == to))
            .unwrap_or(false)
    }

    /// Returns the weight of the edge from `from` to `to`, if it exists.
    pub fn edge_weight(&self, from: &V, to: &V) -> Option<W> {
        self.edges
            .get(from)
            .and_then(|edges| edges.iter().find(|(v, _)| v == to).map(|(_, w)| *w))
    }

    /// Returns an iterator over all vertices in the graph.
    pub fn vertices(&self) -> impl Iterator<Item = &V> {
        self.edges.keys()
    }

    /// Returns an iterator over all edges in the graph.
    pub fn edges(&self) -> impl Iterator<Item = (&V, &V, W)> {
        self.edges
            .iter()
            .flat_map(|(from, edges)| edges.iter().map(move |(to, weight)| (from, to, *weight)))
    }

    /// Returns an iterator over all neighbors of a vertex.
    pub fn neighbors(&self, vertex: &V) -> Result<impl Iterator<Item = (&V, W)>> {
        if !self.has_vertex(vertex) {
            return Err(Error::VertexNotFound);
        }
        Ok(self.edges[vertex].iter().map(|(v, w)| (v, *w)))
    }

    /// Returns the number of vertices in the graph.
    pub fn vertex_count(&self) -> usize {
        self.edges.len()
    }

    /// Returns the number of edges in the graph.
    pub fn edge_count(&self) -> usize {
        if self.directed {
            self.edges.values().map(|edges| edges.len()).sum()
        } else {
            self.edges.values().map(|edges| edges.len()).sum::<usize>() / 2
        }
    }

    /// Returns true if the graph is directed.
    pub fn is_directed(&self) -> bool {
        self.directed
    }

    /// Returns true if the graph is empty (has no vertices).
    pub fn is_empty(&self) -> bool {
        self.edges.is_empty()
    }

    /// Validates that all vertices in an iterator exist in the graph.
    #[allow(dead_code)]
    pub(crate) fn validate_vertices<'a, I>(&self, vertices: I) -> Result<()>
    where
        I: IntoIterator<Item = &'a V>,
        V: 'a,
    {
        for vertex in vertices {
            if !self.has_vertex(vertex) {
                return Err(Error::VertexNotFound);
            }
        }
        Ok(())
    }

    /// Returns true if the graph is connected (ignoring direction if directed).
    pub fn is_connected(&self) -> bool {
        if self.is_empty() {
            return true;
        }

        let start = self.vertices().next().unwrap();
        let mut visited = HashSet::new();
        self.dfs_visit(*start, &mut visited);

        visited.len() == self.vertex_count()
    }

    /// Helper function for depth-first search traversal.
    fn dfs_visit(&self, vertex: V, visited: &mut HashSet<V>) {
        if visited.insert(vertex) {
            // For directed graphs, consider both outgoing and incoming edges
            if self.is_directed() {
                // Check outgoing edges
                if let Ok(neighbors) = self.neighbors(&vertex) {
                    for (neighbor, _) in neighbors {
                        self.dfs_visit(*neighbor, visited);
                    }
                }
                // Check incoming edges
                for v in self.vertices() {
                    if let Ok(mut neighbors) = self.neighbors(v) {
                        if neighbors.any(|(n, _)| n == &vertex) {
                            self.dfs_visit(*v, visited);
                        }
                    }
                }
            } else {
                // For undirected graphs, just check neighbors
                if let Ok(neighbors) = self.neighbors(&vertex) {
                    for (neighbor, _) in neighbors {
                        self.dfs_visit(*neighbor, visited);
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64;

    #[test]
    fn test_new_graph() {
        let graph: Graph<i32, f64> = Graph::new();
        assert!(graph.is_empty());
        assert!(graph.is_directed());
        assert_eq!(graph.vertex_count(), 0);
        assert_eq!(graph.edge_count(), 0);
    }

    #[test]
    fn test_new_undirected_graph() {
        let graph: Graph<i32, f64> = Graph::new_undirected();
        assert!(!graph.is_directed());
        assert!(graph.is_empty());
    }

    #[test]
    fn test_add_vertex() {
        let mut graph: Graph<i32, f64> = Graph::new();
        graph.add_vertex(1);
        assert!(graph.has_vertex(&1));
        assert_eq!(graph.vertex_count(), 1);
    }

    #[test]
    fn test_add_edge_directed() {
        let mut graph: Graph<i32, f64> = Graph::new();
        graph.add_edge(1, 2, 1.0);
        assert!(graph.has_edge(&1, &2));
        assert!(!graph.has_edge(&2, &1));
        assert_eq!(graph.edge_count(), 1);
    }

    #[test]
    fn test_add_edge_undirected() {
        let mut graph: Graph<i32, f64> = Graph::new_undirected();
        graph.add_edge(1, 2, 1.0);
        assert!(graph.has_edge(&1, &2));
        assert!(graph.has_edge(&2, &1));
        assert_eq!(graph.edge_count(), 1);
    }

    #[test]
    fn test_edge_weight() {
        let mut graph: Graph<i32, f64> = Graph::new();
        graph.add_edge(1, 2, 1.5);
        assert_eq!(graph.edge_weight(&1, &2), Some(1.5));
        assert_eq!(graph.edge_weight(&2, &1), None);
    }

    #[test]
    fn test_vertices_iterator() {
        let mut graph: Graph<i32, f64> = Graph::new();
        graph.add_vertex(1);
        graph.add_vertex(2);
        graph.add_vertex(3);
        let vertices: HashSet<_> = graph.vertices().copied().collect();
        assert_eq!(vertices, vec![1, 2, 3].into_iter().collect());
    }

    #[test]
    fn test_edges_iterator() {
        let mut graph: Graph<i32, f64> = Graph::new();
        graph.add_edge(1, 2, 1.0);
        graph.add_edge(2, 3, 2.0);
        let edges: Vec<_> = graph.edges().collect();
        assert_eq!(edges.len(), 2);
        assert!(edges.contains(&(&1, &2, 1.0)));
        assert!(edges.contains(&(&2, &3, 2.0)));
    }

    #[test]
    fn test_neighbors() {
        let mut graph: Graph<i32, f64> = Graph::new();
        graph.add_edge(1, 2, 1.0);
        graph.add_edge(1, 3, 2.0);
        let neighbors: Vec<_> = graph.neighbors(&1).unwrap().collect();
        assert_eq!(neighbors.len(), 2);
        assert!(neighbors.contains(&(&2, 1.0)));
        assert!(neighbors.contains(&(&3, 2.0)));
    }

    #[test]
    fn test_neighbors_error() {
        let graph: Graph<i32, f64> = Graph::new();
        let result = graph.neighbors(&1);
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_vertices() {
        let mut graph: Graph<i32, f64> = Graph::new();
        graph.add_vertex(1);
        graph.add_vertex(2);
        assert!(graph.validate_vertices([&1, &2]).is_ok());
        assert!(matches!(
            graph.validate_vertices([&1, &3]).unwrap_err(),
            Error::VertexNotFound
        ));
    }

    #[test]
    fn test_is_connected() {
        let mut graph: Graph<i32, f64> = Graph::new_undirected();
        assert!(graph.is_connected()); // Empty graph is connected

        graph.add_edge(1, 2, 1.0);
        graph.add_edge(2, 3, 1.0);
        assert!(graph.is_connected());

        graph.add_vertex(4); // Isolated vertex
        assert!(!graph.is_connected());
    }

    #[test]
    fn test_directed_graph_connectivity() {
        let mut graph: Graph<i32, f64> = Graph::new();
        graph.add_edge(1, 2, 1.0);
        graph.add_edge(2, 3, 1.0);
        graph.add_edge(3, 1, 1.0); // Add cycle to make it strongly connected
        assert!(graph.is_connected());
    }
}
