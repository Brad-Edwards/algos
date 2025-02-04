use std::collections::{HashMap, HashSet};

#[derive(Debug, Clone)]
pub struct Graph {
    edges: Vec<(usize, usize)>,
}

impl Graph {
    pub fn new(edges: Vec<(usize, usize)>) -> Self {
        Self { edges }
    }
}

/// Implements a 2-approximation algorithm for the Vertex Cover problem.
///
/// This algorithm provides a vertex cover that is at most twice the size of
/// the optimal solution. It works by repeatedly selecting both endpoints of
/// an uncovered edge until all edges are covered.
///
/// # Arguments
///
/// * `graph` - The input graph represented by edges and number of vertices
///
/// # Returns
///
/// * A HashSet containing the vertices in the approximate vertex cover
pub fn solve(graph: &Graph) -> HashSet<usize> {
    let mut vertex_cover = HashSet::new();
    let mut remaining_edges: HashSet<_> = graph.edges.iter().cloned().collect();

    // Build adjacency list for efficient edge lookup
    let mut adj_list: HashMap<usize, Vec<usize>> = HashMap::new();
    for &(u, v) in &graph.edges {
        adj_list.entry(u).or_default().push(v);
        adj_list.entry(v).or_default().push(u);
    }

    while let Some(&(u, v)) = remaining_edges.iter().next() {
        // Add both endpoints to the cover
        vertex_cover.insert(u);
        vertex_cover.insert(v);

        // Remove all edges incident to u and v
        let edges_to_remove: Vec<_> = remaining_edges
            .iter()
            .filter(|&&(x, y)| x == u || x == v || y == u || y == v)
            .cloned()
            .collect();

        for edge in edges_to_remove {
            remaining_edges.remove(&edge);
        }
    }

    vertex_cover
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_graph() {
        // Create a simple graph: path of length 2 (3 vertices, 2 edges)
        let edges = vec![(0, 1), (1, 2)];
        let graph = Graph::new(edges);

        let cover = solve(&graph);

        // Middle vertex (1) should be sufficient to cover all edges
        assert!(cover.len() <= 2); // 2-approximation guarantee

        // Verify that all edges are covered
        for &(u, v) in &graph.edges {
            assert!(cover.contains(&u) || cover.contains(&v));
        }
    }

    #[test]
    fn test_star_graph() {
        // Create a star graph with center vertex 0 and 4 leaves
        let edges = vec![(0, 1), (0, 2), (0, 3), (0, 4)];
        let graph = Graph::new(edges);

        let cover = solve(&graph);

        // Optimal solution uses just the center vertex
        assert!(cover.len() <= 2); // 2-approximation guarantee

        // Verify that all edges are covered
        for &(u, v) in &graph.edges {
            assert!(cover.contains(&u) || cover.contains(&v));
        }
    }
}
