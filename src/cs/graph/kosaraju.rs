/// A simple directed graph representation using adjacency lists.
#[derive(Debug, Clone)]
pub struct DirectedGraph {
    adjacency_list: Vec<Vec<usize>>,
}

impl DirectedGraph {
    /// Creates a new directed graph with `num_nodes` vertices (0..num_nodes-1).
    pub fn new(num_nodes: usize) -> Self {
        Self {
            adjacency_list: vec![Vec::new(); num_nodes],
        }
    }

    /// Adds a directed edge from `src` to `dst`.
    pub fn add_edge(&mut self, src: usize, dst: usize) {
        self.adjacency_list[src].push(dst);
    }

    /// Returns the total number of vertices in the graph.
    pub fn num_nodes(&self) -> usize {
        self.adjacency_list.len()
    }

    /// Returns the adjacency list of the graph (for debugging/inspection).
    pub fn adjacency_list(&self) -> &Vec<Vec<usize>> {
        &self.adjacency_list
    }

    /// Produces the transpose of this directed graph:
    /// a graph with all edges reversed.
    pub fn transpose(&self) -> Self {
        let mut transposed = Self::new(self.num_nodes());
        for (u, neighbors) in self.adjacency_list.iter().enumerate() {
            for &v in neighbors {
                transposed.add_edge(v, u);
            }
        }
        transposed
    }
}

/// Kosaraju's algorithm to find all strongly connected components (SCCs) in a directed graph.
///
/// Returns a vector of SCCs, where each SCC is represented by a vector of node indices.
/// The order of SCCs and the order of nodes within each SCC is not strictly defined.
pub fn kosaraju(graph: &DirectedGraph) -> Vec<Vec<usize>> {
    let n = graph.num_nodes();

    // First DFS pass to determine finishing times (stored on a stack).
    // We process nodes in ascending numerical order, but any order is fine
    // as long as we do a full DFS on unvisited nodes.
    let mut visited = vec![false; n];
    let mut stack = Vec::with_capacity(n);

    fn dfs1(graph: &DirectedGraph, node: usize, visited: &mut [bool], stack: &mut Vec<usize>) {
        visited[node] = true;
        for &neighbor in &graph.adjacency_list[node] {
            if !visited[neighbor] {
                dfs1(graph, neighbor, visited, stack);
            }
        }
        // Post-order push to record finishing time
        stack.push(node);
    }

    for node in 0..n {
        if !visited[node] {
            dfs1(graph, node, &mut visited, &mut stack);
        }
    }

    // Transpose the graph
    let transposed = graph.transpose();

    // Second DFS pass on the transposed graph in decreasing order of finishing times.
    visited.fill(false);
    let mut sccs = Vec::new();

    fn dfs2(graph: &DirectedGraph, node: usize, visited: &mut [bool], component: &mut Vec<usize>) {
        visited[node] = true;
        component.push(node);
        for &neighbor in &graph.adjacency_list[node] {
            if !visited[neighbor] {
                dfs2(graph, neighbor, visited, component);
            }
        }
    }

    // Pop from the stack to get nodes in decreasing finish time order.
    while let Some(node) = stack.pop() {
        if !visited[node] {
            let mut component = Vec::new();
            dfs2(&transposed, node, &mut visited, &mut component);
            sccs.push(component);
        }
    }

    sccs
}

#[cfg(test)]
mod tests {
    use super::*;

    // Helper to check if each node belongs to exactly one of the SCCs
    fn verify_partition(num_nodes: usize, sccs: &[Vec<usize>]) {
        let mut seen = vec![false; num_nodes];
        let mut count = 0;
        for comp in sccs {
            for &node in comp {
                seen[node] = true;
                count += 1;
            }
        }
        assert_eq!(count, num_nodes, "Not all nodes were placed in components");
        for (i, &val) in seen.iter().enumerate() {
            assert!(val, "Node {} was not in any component", i);
        }
    }

    #[test]
    fn test_empty_graph() {
        let graph = DirectedGraph::new(0);
        let sccs = kosaraju(&graph);
        assert!(sccs.is_empty(), "No SCCs expected for an empty graph");
    }

    #[test]
    fn test_single_node() {
        let graph = DirectedGraph::new(1);
        let sccs = kosaraju(&graph);
        // A single node is by definition its own SCC
        assert_eq!(sccs.len(), 1, "Expected exactly one SCC");
        assert_eq!(sccs[0], vec![0], "That SCC should contain the only node");
        verify_partition(1, &sccs);
    }

    #[test]
    fn test_no_edges_multiple_nodes() {
        let graph = DirectedGraph::new(3);
        let sccs = kosaraju(&graph);
        // With no edges, each node is an isolated SCC
        assert_eq!(sccs.len(), 3, "Expected each node to form its own SCC");
        verify_partition(3, &sccs);
    }

    #[test]
    fn test_simple_cycle() {
        // 0 -> 1, 1 -> 2, 2 -> 0 forms a single cycle with 3 nodes
        let mut graph = DirectedGraph::new(3);
        graph.add_edge(0, 1);
        graph.add_edge(1, 2);
        graph.add_edge(2, 0);

        let sccs = kosaraju(&graph);
        // All 3 should be in a single component
        assert_eq!(sccs.len(), 1, "All nodes in one cycle => exactly 1 SCC");
        let comp = &sccs[0];
        // Sorting for stable comparison
        let mut sorted_comp = comp.clone();
        sorted_comp.sort_unstable();
        assert_eq!(sorted_comp, vec![0, 1, 2]);
        verify_partition(3, &sccs);
    }

    #[test]
    fn test_two_components() {
        // Component 1: 0 -> 1 -> 2 -> 0
        // Component 2: 3 -> 4 -> 3
        // No edges between these components.
        let mut graph = DirectedGraph::new(5);
        // First SCC (0,1,2)
        graph.add_edge(0, 1);
        graph.add_edge(1, 2);
        graph.add_edge(2, 0);
        // Second SCC (3,4)
        graph.add_edge(3, 4);
        graph.add_edge(4, 3);

        let sccs = kosaraju(&graph);
        assert_eq!(sccs.len(), 2, "Two distinct SCCs expected");
        verify_partition(5, &sccs);

        // Each SCC should contain 3 nodes or 2 nodes
        let sizes: Vec<usize> = sccs.iter().map(|c| c.len()).collect();
        // Sort so we can reliably match [2, 3]
        let mut sorted_sizes = sizes.clone();
        sorted_sizes.sort_unstable();
        assert_eq!(sorted_sizes, vec![2, 3]);
    }

    #[test]
    fn test_chain_like_graph() {
        // 0 -> 1 -> 2 -> 3 -> 4
        // No cycles => each node is its own SCC.
        let mut graph = DirectedGraph::new(5);
        graph.add_edge(0, 1);
        graph.add_edge(1, 2);
        graph.add_edge(2, 3);
        graph.add_edge(3, 4);

        let sccs = kosaraju(&graph);
        assert_eq!(sccs.len(), 5, "No cycles => every node is its own SCC");
        verify_partition(5, &sccs);
    }

    #[test]
    fn test_complex_graph() {
        // This graph has several cross-links to create partial cycles.
        // We’ll expect some multi-node SCCs. Structure:
        //
        // 0 -> 1 -> 2
        // ^         |
        // |         v
        // 4 <- 3 <-- (plus 2 -> 3, 3 -> 4, 4 -> 0 forms a cycle)
        // 5 is isolated from the cycle but has a self-loop => separate 1-node SCC or is it a self-loop SCC?
        //
        // Let's define edges:
        // 0 -> 1, 1 -> 2, 2 -> 3, 3 -> 4, 4 -> 0 (SCC of [0,1,2,3,4])
        // 5 -> 5 (self-loop => its own SCC)

        let mut graph = DirectedGraph::new(6);
        graph.add_edge(0, 1);
        graph.add_edge(1, 2);
        graph.add_edge(2, 3);
        graph.add_edge(3, 4);
        graph.add_edge(4, 0);
        graph.add_edge(5, 5);

        let sccs = kosaraju(&graph);
        // Expect 2 SCCs: [0,1,2,3,4] and [5]
        assert_eq!(sccs.len(), 2);
        verify_partition(6, &sccs);

        // Let's see which one has length 5 and which has length 1
        let mut sorted_sccs: Vec<Vec<usize>> = sccs
            .into_iter()
            .map(|mut comp| {
                comp.sort_unstable();
                comp
            })
            .collect();
        // Sort the outer vector by length
        sorted_sccs.sort_by_key(|v| v.len());

        assert_eq!(sorted_sccs[0], vec![5], "Node 5 by itself");
        assert_eq!(sorted_sccs[1], vec![0, 1, 2, 3, 4], "Main cycle of 5 nodes");
    }
}
