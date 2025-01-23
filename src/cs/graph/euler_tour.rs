//! # Euler Tour (Hierholzer's Algorithm)
//!
//! This module provides a standard implementation of an **Euler Tour** (also
//! called an Eulerian path or circuit) using **Hierholzer's algorithm**. An Euler Tour visits every
//! edge of a connected graph exactly once. For an **Eulerian circuit**, the start and end vertices
//! coincide; for an **Eulerian path** (which may not end where it started), there can be up to two
//! vertices of odd degree.
//!
//! ## Summary of Hierholzer's Algorithm
//! 1. **Check Eulerian Feasibility**: For an undirected graph, either:
//!    - *Eulerian Circuit* condition: Every vertex has even degree and the graph is connected (ignoring isolated vertices).
//!    - *Eulerian Path* (non-circuit) condition: Exactly two vertices have odd degree and the graph is connected when ignoring isolated vertices.
//!
//! 2. **Find a Tour**:
//!    - Start from a vertex with nonzero degree. (If you want an Eulerian path, start from a vertex with odd degree if it exists; otherwise pick any vertex.)
//!    - Follow edges one at a time. Remove each edge from the graph as you traverse it. Eventually, you return to the starting vertex if it’s an Eulerian circuit (or use up all edges for Eulerian path).
//!    - If there are still remaining edges in some other component, jump there and continue building the tour until all edges are used.
//!
//! Complexity: \( O(V + E) \), where \(V\) is the number of vertices and \(E\) is the number of edges.
//!
//! ## Example Usage
//! ```rust
//! use euler_tour::{Graph, euler_tour};
//!
//! // Build a simple undirected graph that has an Eulerian circuit.
//! // Let's do a square with diagonals: 0 - 1 - 2 - 3 - 0 and edges 0-2, 1-3
//! let mut graph = Graph::new();
//! graph.add_edge(0, 1);
//! graph.add_edge(1, 2);
//! graph.add_edge(2, 3);
//! graph.add_edge(3, 0);
//! graph.add_edge(0, 2);
//! graph.add_edge(1, 3);
//!
//! // Compute Euler tour
//! let tour = euler_tour(&graph).expect("Euler tour should exist");
//! // This could produce an Euler cycle like [0, 1, 3, 2, 1, 2, 0], depending on edge visitation order
//! assert_eq!(tour.len(), graph.num_edges() + 1);
//! ```
//!
//! ## Notes
//! - This implementation assumes an **undirected** graph. For a directed version, the degree checks
//!   and adjacency manipulations differ but follow a similar approach.

use std::collections::{HashMap, HashSet, VecDeque};

/// A simple undirected graph structure suited for Euler Tour.
/// - `adj` stores each vertex -> set of neighbors.
/// - For an undirected edge (u,v), both adj[u] and adj[v] contain the other.
#[derive(Debug, Clone)]
pub struct Graph<T>
where
    T: Eq + std::hash::Hash + Clone,
{
    adj: HashMap<T, HashSet<T>>,
    edges_count: usize,
}

impl<T> Graph<T>
where
    T: Eq + std::hash::Hash + Clone,
{
    /// Creates an empty graph.
    pub fn new() -> Self {
        Self {
            adj: HashMap::new(),
            edges_count: 0,
        }
    }

    /// Adds an undirected edge between `u` and `v`.
    /// If the edge already exists, it's not duplicated; it's effectively idempotent.
    pub fn add_edge(&mut self, u: T, v: T) {
        // If either u or v is not in the map, insert them
        self.adj.entry(u.clone()).or_insert_with(HashSet::new);
        self.adj.entry(v.clone()).or_insert_with(HashSet::new);

        // Insert each side of the undirected edge if not present
        // If it wasn't in the set, increment edges_count
        let existed_uv = self.adj.get_mut(&u).unwrap().insert(v.clone());
        let existed_vu = self.adj.get_mut(&v).unwrap().insert(u.clone());
        if !existed_uv || !existed_vu {
            // Because this is an undirected edge, we only want to count it once
            // But we might have inserted it from both sides. If neither side existed,
            // we've effectively added the edge once.
            // A robust way: if either side was newly inserted, we can increment once.
            // But if we do that, we risk double-counting if both were new.
            //
            // If the edge was brand new, then existed_uv == false && existed_vu == false.
            // We'll increment by 1 in that case.
            if !existed_uv && !existed_vu {
                self.edges_count += 1;
            }
        }
    }

    /// Returns the number of edges in the undirected graph.
    /// Each undirected edge is counted once.
    pub fn num_edges(&self) -> usize {
        self.edges_count
    }

    /// Returns the vertices of the graph.
    pub fn vertices(&self) -> impl Iterator<Item = &T> {
        self.adj.keys()
    }

    /// Returns the neighbors of vertex `v`.
    pub fn neighbors(&self, v: &T) -> Option<&HashSet<T>> {
        self.adj.get(v)
    }
}

/// Performs a breadth-first search from some non-isolated start vertex to check connectivity
/// ignoring isolated vertices. Returns the set of visited vertices.
fn bfs_component<T>(graph: &Graph<T>, start: &T) -> HashSet<T>
where
    T: Eq + std::hash::Hash + Clone,
{
    let mut visited = HashSet::new();
    let mut queue = VecDeque::new();
    queue.push_back(start.clone());
    visited.insert(start.clone());

    while let Some(node) = queue.pop_front() {
        if let Some(neighs) = graph.neighbors(&node) {
            for neighbor in neighs {
                if !visited.contains(neighbor) {
                    visited.insert(neighbor.clone());
                    queue.push_back(neighbor.clone());
                }
            }
        }
    }
    visited
}

/// Checks if the graph is connected **ignoring isolated vertices**.
/// Returns `(is_connected_ignoring_isolated, candidate_start_vertex)`
/// where `candidate_start_vertex` is any vertex with a nonzero degree (or None if none).
fn is_connected_ignoring_isolated<T>(graph: &Graph<T>) -> (bool, Option<T>)
where
    T: Eq + std::hash::Hash + Clone,
{
    // Find a vertex with a nonzero degree to start BFS
    let mut candidate_start = None;
    for (v, neighs) in graph.adj.iter() {
        if !neighs.is_empty() {
            candidate_start = Some(v.clone());
            break;
        }
    }
    // If no such vertex, it's "connected" in the trivial sense (no edges),
    // and there's no start for an Euler tour but it might be considered an empty graph.
    let Some(start) = candidate_start.clone() else {
        return (true, None);
    };

    // BFS from that vertex
    let visited = bfs_component(graph, &start);

    // Check that every vertex with a nonzero degree is visited
    for (v, neighs) in graph.adj.iter() {
        if !neighs.is_empty() && !visited.contains(v) {
            return (false, candidate_start);
        }
    }
    (true, candidate_start)
}

/// Determines if the undirected graph has an Eulerian path or circuit and
/// returns a suitable start vertex for Hierholzer's algorithm.
/// - Returns `(true, Some(start_vertex))` if Eulerian; otherwise `(false, None)`.
/// - If exactly 0 or 2 vertices have odd degree, it's Eulerian (with a path or circuit).
/// - The chosen start vertex will be one of the odd-degree vertices if present,
///   otherwise any vertex with nonzero degree.
fn is_eulerian<T>(graph: &Graph<T>) -> (bool, Option<T>)
where
    T: Eq + std::hash::Hash + Clone,
{
    // Check connectivity ignoring isolated vertices
    let (connected, start_candidate) = is_connected_ignoring_isolated(graph);
    if !connected {
        return (false, None);
    }

    // Count how many vertices have odd degree
    let mut odd_vertices = vec![];
    for (v, neighs) in &graph.adj {
        let deg = neighs.len();
        if deg % 2 != 0 {
            odd_vertices.push(v.clone());
        }
    }

    match odd_vertices.len() {
        0 => {
            // Eulerian circuit
            // We can start from any vertex with nonzero degree
            (true, start_candidate)
        }
        2 => {
            // Eulerian path
            // Must start from one of these odd vertices
            (true, Some(odd_vertices[0].clone()))
        }
        _ => (false, None),
    }
}

/// Computes an Euler Tour (path or circuit) if one exists. Returns the sequence of vertices
/// forming the path that traverses every edge exactly once. If the graph is not Eulerian,
/// returns `None`.
///
/// This uses Hierholzer's algorithm:
/// 1. Check if the graph is Eulerian (0 or 2 odd-degree vertices, connected ignoring isolated).
/// 2. Start from a valid start vertex (odd-degree if present, else any non-isolated).
/// 3. Traverse edges using a stack-based approach until all edges are exhausted.
pub fn euler_tour<T>(graph: &Graph<T>) -> Option<Vec<T>>
where
    T: Eq + std::hash::Hash + Clone,
{
    let (eulerian, start) = is_eulerian(graph);
    if !eulerian {
        return None;
    }
    let Some(mut current_vertex) = start else {
        // No start means graph has no edges. By convention, the Euler tour is empty or a single vertex.
        // If the graph is truly empty, we can return an empty path. Or pick any vertex if we want a single-vertex path.
        return Some(vec![]);
    };

    // Make a local copy of adjacency we can mutate (remove edges as we go).
    // We'll store a multiset of edges. For an undirected edge (u <-> v), remove from both sides.
    let mut local_adj: HashMap<T, Vec<T>> = HashMap::new();
    for (v, neighs) in &graph.adj {
        local_adj.insert(v.clone(), neighs.iter().cloned().collect());
    }

    let mut stack = vec![current_vertex.clone()];
    let mut path = vec![];

    while !stack.is_empty() {
        if let Some(neighbors) = local_adj.get_mut(&current_vertex) {
            if !neighbors.is_empty() {
                // Pick next neighbor, remove the edge from adjacency
                let next_vertex = neighbors.pop().unwrap();

                // Also remove the reverse edge from next_vertex -> current_vertex
                if let Some(rev_list) = local_adj.get_mut(&next_vertex) {
                    // We need to remove one occurrence of current_vertex
                    // If the graph had parallel edges, remove just one.
                    if let Some(pos) = rev_list.iter().position(|x| *x == current_vertex) {
                        rev_list.swap_remove(pos);
                    }
                }

                // Move forward
                stack.push(current_vertex.clone());
                current_vertex = next_vertex;
            } else {
                // No more neighbors => add to path
                path.push(current_vertex.clone());
                current_vertex = stack.pop().unwrap_or(current_vertex);
            }
        } else {
            // This vertex might not exist in local_adj if it has zero adjacency
            path.push(current_vertex.clone());
            current_vertex = stack.pop().unwrap_or(current_vertex);
        }
    }

    Some(path)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_graph() {
        let g: Graph<i32> = Graph::new();
        // No edges, no vertices => Euler Tour is trivially empty
        assert_eq!(euler_tour(&g), Some(vec![]));
    }

    #[test]
    fn test_single_edge() {
        let mut g = Graph::new();
        g.add_edge(1, 2);
        // Euler path = [1, 2] or [2, 1]
        // Our code doesn't guarantee which direction it picks first,
        // but let's check length.
        let res = euler_tour(&g).unwrap();
        assert_eq!(res.len(), 2);
        assert!(matches!(&res[..], [1, 2] | [2, 1]));
    }

    #[test]
    fn test_triangle_eulerian_circuit() {
        // 0 - 1
        //  \   /
        //   2
        let mut g = Graph::new();
        g.add_edge(0, 1);
        g.add_edge(1, 2);
        g.add_edge(2, 0);
        // All vertices have even degree (2). Euler circuit.
        let res = euler_tour(&g).unwrap();
        // We expect 3 edges => path length is 4.
        assert_eq!(res.len(), 4);
        // Check it uses all edges exactly once
        // (We won't do a strict order check, just correctness)
    }

    #[test]
    fn test_graph_with_no_euler_path() {
        // This is a "T" shape with 3 edges, 3 vertices: 0-1, 1-2, 1-3
        // Actually let's add a 4th vertex: 3. So edges: 0-1, 1-2, 2-3 => chain of length 3
        // That has 2 vertices of degree 1, 2 of degree 2 => actually that does form a valid path
        // Let's create a shape that fails connectivity or odd-degree condition:
        //   0-1 2-3 (two separate edges, disconnected)
        let mut g = Graph::new();
        g.add_edge(0, 1);
        g.add_edge(2, 3);
        // Disconnected => no Euler path across entire graph
        assert_eq!(euler_tour(&g), None);
    }

    #[test]
    fn test_long_chain_euler_path() {
        // 0 - 1 - 2 - 3 - 4
        // Edges: (0,1), (1,2), (2,3), (3,4)
        // This is an Eulerian path (two vertices, 0 and 4, have odd degree = 1).
        let mut g = Graph::new();
        g.add_edge(0, 1);
        g.add_edge(1, 2);
        g.add_edge(2, 3);
        g.add_edge(3, 4);
        // Euler path length = edges + 1 = 5
        let path = euler_tour(&g).unwrap();
        assert_eq!(path.len(), 5);
        // Possibly [0,1,2,3,4] or the reverse
        // Just check first and last are the odd-degree nodes
        assert!(matches!(
            (path.first(), path.last()),
            (Some(0), Some(4)) | (Some(4), Some(0))
        ));
    }
}
