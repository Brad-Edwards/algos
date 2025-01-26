//! # Hierholzer's Algorithm for Eulerian Paths/Cycles
//!
//! This module provides a **production-grade** implementation of **Hierholzer's Algorithm**
//! to find an **Eulerian cycle** (or path) in a graph. We assume the graph is either undirected
//! or directed, with the typical constraints for an Eulerian cycle/path:
//! - **Undirected**: Each vertex has an even degree for a cycle, or exactly two vertices of odd degree for a path.
//! - **Directed**: Each vertex's in-degree equals out-degree for a cycle, or differs by 1 for exactly two vertices for a path, etc.
//!
//! For simplicity, we demonstrate **undirected** usage by default. If you need a directed
//! version, you can adapt or store separate in/out edges, or add a mode. The core Hierholzer
//! procedure is similar in both cases (just mind how you remove edges).
//!
//! ## Key Features
//! - **Graph** structure storing adjacency in a straightforward manner.
//! - **Add edges** easily, including parallel edges if needed.
//! - **Check** connectivity on the subgraph of vertices that have edges (we skip isolated ones).
//! - **Find Eulerian cycle or path** if it exists. If not, returns an error or an empty result.
//! - **Production**: The code is designed with robust checks, flexible usage, and typical
//!   adjacency-based approach for real use on moderate-sized graphs.
//!
//! ### Example (Undirected)
//! ```rust
//! use algos::cs::graph::hierholzer::{UndirectedGraph, hierholzer_eulerian_path};
//!
//! // Build a small undirected graph with an Eulerian cycle
//! let mut g = UndirectedGraph::new(4);
//! g.add_edge(0,1);
//! g.add_edge(1,2);
//! g.add_edge(2,3);
//! g.add_edge(3,0);
//!
//! let cycle = hierholzer_eulerian_path(&mut g, true).expect("Eulerian cycle not found");
//! println!("Eulerian cycle: {:?}", cycle);
//! // cycle might be [0,1,2,3,0], for example
//! ```

use std::collections::VecDeque;

/// A simple undirected graph structure supporting multiple edges, storing adjacency in memory.
#[derive(Debug, Clone)]
pub struct UndirectedGraph {
    /// Number of vertices
    pub n: usize,
    /// Adjacency list: `adj[v]` is a list of edges from `v` to some neighbor.
    /// We store `(neighbor, used)` so we can mark edges used during Hierholzer's.
    /// In an undirected sense, each edge is stored twice in adjacency.
    /// So the actual "edges" are half of what adjacency might store if counting each direction.
    pub adj: Vec<Vec<(usize, bool)>>,
}

impl UndirectedGraph {
    /// Creates a new empty graph with `n` vertices (`0..n-1`).
    #[must_use]
    pub fn new(n: usize) -> Self {
        Self {
            n,
            adj: vec![Vec::new(); n],
        }
    }

    /// Adds an undirected edge between `u` and `v`.
    /// If you want parallel edges, you can call this multiple times.
    /// # Panics
    /// - if `u` or `v` out of range
    pub fn add_edge(&mut self, u: usize, v: usize) {
        assert!(u < self.n && v < self.n, "vertex out of range");
        self.adj[u].push((v, false));
        self.adj[v].push((u, false));
    }

    /// Gets the total degree of a vertex (used for Eulerian property check)
    #[must_use]
    fn total_degree(&self, v: usize) -> usize {
        self.adj[v].len()
    }

    /// Gets the number of unused edges from a vertex (used during path construction)
    #[must_use]
    fn unused_edges(&self, v: usize) -> usize {
        self.adj[v].iter().filter(|&&(_, used)| !used).count()
    }
}

/// Finds an Eulerian path or cycle in the given **undirected** graph using Hierholzer's algorithm.
/// If `require_cycle=true`, we require an Eulerian cycle; if false, we allow a path if exactly two vertices have odd degree.
/// Returns `Ok(vec_of_vertices_in_order)` or `Err(...)` if no Eulerian path/cycle is possible.
///
/// # Steps
/// 1. Check if the graph is connected ignoring isolated vertices, or if there's a separate subgraph with edges => fail.
/// 2. Check degrees for odd count of vertices. If `require_cycle` is true, we need 0 odd-degree vertices. If false, we need 0 or 2.
/// 3. Start from a vertex that has edges (if path and there's 2 odd-degree, start from one of them).
/// 4. Use Hierholzer: follow edges marking them used until you return to the start (a cycle or partial).
/// 5. If there's leftover edges not used, incorporate them by splicing in sub-tours until all edges used.
///
/// # Errors
/// Returns `Err` with a descriptive message if:
/// - The graph is not connected (ignoring isolated vertices)
/// - The graph has wrong number of odd-degree vertices for requested path/cycle
/// - For cycles: any vertex has odd degree
/// - For paths: number of odd-degree vertices is not 0 or 2
///
/// # Examples
/// ```rust
/// use algos::cs::graph::hierholzer::{UndirectedGraph, hierholzer_eulerian_path};
///
/// // Build a small undirected graph with an Eulerian cycle
/// let mut g = UndirectedGraph::new(4);
/// g.add_edge(0,1);
/// g.add_edge(1,2);
/// g.add_edge(2,3);
/// g.add_edge(3,0);
///
/// let cycle = hierholzer_eulerian_path(&mut g, true).expect("Eulerian cycle not found");
/// println!("Eulerian cycle: {:?}", cycle);
/// // cycle might be [0,1,2,3,0], for example
/// ```
///
/// **Note**: The function modifies the graph's adjacency to mark edges used. It's recommended
/// to clone if you need the graph intact.
#[must_use = "This function returns a Result containing the Eulerian path/cycle that should be handled"]
pub fn hierholzer_eulerian_path(
    g: &mut UndirectedGraph,
    require_cycle: bool,
) -> Result<Vec<usize>, String> {
    // Step 1: check connectivity on subgraph of non-isolated vertices
    let comp_check = check_connectivity(g);
    if !comp_check {
        return Err(
            "Graph is not connected (in the subgraph that has edges). No Eulerian path/cycle."
                .into(),
        );
    }

    // Step 2: check degrees
    let mut odd_vertices = Vec::new();
    for v in 0..g.n {
        let deg = g.total_degree(v);
        if deg % 2 != 0 {
            odd_vertices.push(v);
        }
    }
    if require_cycle && !odd_vertices.is_empty() {
        return Err(format!(
            "Require Eulerian cycle, but found {} vertices with odd degree",
            odd_vertices.len()
        ));
    }
    if !(require_cycle || odd_vertices.is_empty() || odd_vertices.len() == 2) {
        return Err(format!(
            "Eulerian path requires 0 or 2 odd-degree vertices, found {}",
            odd_vertices.len()
        ));
    }

    // Step 3: pick start
    let start = odd_vertices.first().copied().unwrap_or_else(|| {
        // find any vertex with edges
        (0..g.n).find(|&v| !g.adj[v].is_empty()).unwrap_or(0)
    });

    // We'll store the final path in `circuit`.
    let mut circuit = Vec::new();
    let mut stack = Vec::new();
    stack.push(start);

    while let Some(u) = stack.last().copied() {
        // find an unused edge from u
        if g.unused_edges(u) > 0 {
            if let Some(e_idx) = g.adj[u].iter().position(|&(_, used)| !used) {
                // use that edge
                let v = g.adj[u][e_idx].0;
                g.adj[u][e_idx].1 = true;

                // mark ONE corresponding reverse edge as used
                if let Some(rev_edge) = g.adj[v].iter_mut().find(|(w, used)| *w == u && !*used) {
                    rev_edge.1 = true;
                }

                stack.push(v);
            }
        } else {
            // no unused edges from u => pop from stack to circuit
            stack.pop();
            circuit.push(u);
        }
    }

    circuit.reverse();
    Ok(circuit)
}

/// Check if the subgraph with edges is connected ignoring isolated vertices.
/// i.e. pick a vertex with edges, BFS, see if we can reach all vertices that have edges.
fn check_connectivity(g: &UndirectedGraph) -> bool {
    // find a vertex with edges if any
    let start_opt = (0..g.n).find(|&v| !g.adj[v].is_empty());
    if start_opt.is_none() {
        // no edges => trivially we have an Eulerian cycle with 0 edges or it's an empty graph
        return true;
    }
    let start = start_opt.unwrap();
    let mut visited = vec![false; g.n];
    let mut queue = VecDeque::new();
    visited[start] = true;
    queue.push_back(start);
    let mut count = 1usize;
    while let Some(u) = queue.pop_front() {
        for &(nbr, _) in &g.adj[u] {
            if !visited[nbr] {
                visited[nbr] = true;
                queue.push_back(nbr);
                count += 1;
            }
        }
    }
    // check all vertices that have edges are visited
    let mut total_with_edges = 0;
    for v in 0..g.n {
        if !g.adj[v].is_empty() {
            total_with_edges += 1;
        }
    }
    count == total_with_edges
}

#[test]
fn test_odd_degree_vertices() {
    // Modify the edges so **all 4 vertices** become odd-degree.
    // E.g. let every vertex have degree 3 => total of 6 edges.
    // Each vertex is connected to 3 edges => all 4 are odd-degree => should fail for both path & cycle.
    let mut g = UndirectedGraph::new(4);

    // Now each vertex has degree 3:
    // 0 connected to 1, 2, 3
    // 1 connected to 0, 2, 3
    // 2 connected to 0, 1, 3
    // 3 connected to 0, 1, 2
    g.add_edge(0, 1);
    g.add_edge(1, 2);
    g.add_edge(2, 3);
    g.add_edge(3, 0);
    g.add_edge(0, 2);
    g.add_edge(1, 3);

    let r = hierholzer_eulerian_path(&mut g, false);
    assert!(
        r.is_err(),
        "Should fail when 4 vertices have odd degree for Eulerian path"
    );

    let r = hierholzer_eulerian_path(&mut g, true);
    assert!(
        r.is_err(),
        "Should fail when 4 vertices have odd degree for Eulerian cycle"
    );
}

#[test]
fn test_multiple_edges() {
    // Add one more parallel edge to make all degrees even, which is required for an Eulerian cycle.

    let mut g = UndirectedGraph::new(3);
    g.add_edge(0, 1);
    g.add_edge(0, 1); // parallel edge
    g.add_edge(1, 2);
    g.add_edge(2, 0);

    // Originally this gave vertex degrees 0=3, 1=3, 2=2 (two odd, one even).
    // Add one more 0->1 so that 0=4, 1=4, 2=2 => all even => Eulerian cycle possible.
    g.add_edge(0, 1); // fix parity

    let cycle = hierholzer_eulerian_path(&mut g, true).expect("should find cycle");
    assert_eq!(cycle.len(), 6, "5 edges + return-to-start => length 6 path");
    assert_eq!(cycle.first(), cycle.last());
}

#[test]
fn test_vertex_out_of_range() {
    let _g = UndirectedGraph::new(2);
    let result = std::panic::catch_unwind(|| {
        UndirectedGraph::new(2).add_edge(0, 2) // vertex 2 is out of range
    });
    assert!(result.is_err());
}
