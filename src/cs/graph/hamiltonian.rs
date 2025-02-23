/// A backtracking approach to find all Hamiltonian cycles in an undirected graph.
/// Each cycle is a sequence of vertices v0..v_{n-1} with an edge between consecutive
/// vertices, plus an edge from v_{n-1} back to v0, visiting all vertices exactly once.
///
/// # Example
/// ```
/// use algos::cs::graph::hamiltonian::Graph;
///
/// // A 4-node cycle: edges (0-1, 1-2, 2-3, 3-0).
/// let mut g = Graph::new(4);
/// g.add_edge(0, 1);
/// g.add_edge(1, 2);
/// g.add_edge(2, 3);
/// g.add_edge(3, 0);
///
/// // This graph has a single Hamiltonian cycle [0,1,2,3] (and its rotations).
/// // `all_hamiltonian_cycles` returns each cycle in a canonical form
/// // (smallest vertex first), so effectively just one unique cycle here.
/// let cycles = g.all_hamiltonian_cycles();
/// assert_eq!(cycles, vec![vec![0, 1, 2, 3]]);
/// ```

/// Simple undirected graph with adjacency lists.
#[derive(Clone, Debug)]
pub struct Graph {
    /// For each vertex `v`, `edges[v]` is a list of neighbors (undirected).
    edges: Vec<Vec<usize>>,
    /// Number of vertices.
    pub n: usize,
}

impl Graph {
    /// Create an undirected graph with `n` vertices (0..n-1) and no edges.
    pub fn new(n: usize) -> Self {
        let edges = vec![Vec::new(); n];
        Graph { edges, n }
    }

    /// Add an undirected edge between `u` and `v`.
    pub fn add_edge(&mut self, u: usize, v: usize) {
        assert!(u < self.n && v < self.n, "Invalid vertex index");
        self.edges[u].push(v);
        self.edges[v].push(u);
    }

    /// Returns all Hamiltonian cycles using backtracking.
    /// Each cycle is given in a canonical order (rotated so that the smallest vertex is first).
    /// No two cycles differ only by rotation/reversal; each distinct cycle is reported once.
    pub fn all_hamiltonian_cycles(&self) -> Vec<Vec<usize>> {
        if self.n == 0 {
            return vec![];
        }
        if self.n == 1 {
            // Single vertex can only form a cycle if there's a loop from v to itself,
            // but for a standard undirected graph we don't usually store self-loops.
            // So, typically that yields no Hamiltonian cycle. We'll return empty.
            return vec![];
        }

        let mut visited = vec![false; self.n];
        let mut path = Vec::with_capacity(self.n);
        let mut results = Vec::new();

        // We can fix the starting vertex at 0 for convenience, since we only want unique cycles.
        path.push(0);
        visited[0] = true;

        self.dfs_hamiltonian(0, &mut visited, &mut path, &mut results);

        // Remove duplicates or rotations
        unique_cycles(&mut results);
        results
    }

    fn dfs_hamiltonian(
        &self,
        current: usize,
        visited: &mut [bool],
        path: &mut Vec<usize>,
        results: &mut Vec<Vec<usize>>,
    ) {
        // If path contains all vertices, check if there's an edge back to start => cycle
        if path.len() == self.n {
            // path[0] is the start
            let start = path[0];
            if self.edges[current].contains(&start) {
                // We found a Hamiltonian cycle
                let cycle = path.clone();
                results.push(cycle);
            }
            return;
        }

        // Otherwise, try to go to each neighbor of 'current'
        for &next in &self.edges[current] {
            if !visited[next] {
                visited[next] = true;
                path.push(next);

                self.dfs_hamiltonian(next, visited, path, results);

                // backtrack
                path.pop();
                visited[next] = false;
            }
        }
    }
}

/// Rotate each cycle so that the smallest vertex is at the front,
/// and remove duplicates (including mirrored cycles).
fn unique_cycles(cycles: &mut Vec<Vec<usize>>) {
    // Normalize each cycle by rotating to place the smallest vertex at index 0.
    // Also, for undirected cycles, [0,1,2,3] is the same as [0,3,2,1] in reverse order.
    for cycle in cycles.iter_mut() {
        rotate_cycle_to_smallest(cycle);

        // Ensure we store each cycle in ascending wrap vs. its reverse
        // e.g., compare [0,1,2,3] to [0,3,2,1], pick lexicographically smaller.
        let mut reversed = cycle.clone();
        reversed.reverse();
        rotate_cycle_to_smallest(&mut reversed);

        if reversed < *cycle {
            *cycle = reversed;
        }
    }

    // Sort and deduplicate
    cycles.sort();
    cycles.dedup();
}

/// Rotate the cycle so that the smallest vertex is first.
/// E.g. [2,3,0,1] -> [0,1,2,3].
fn rotate_cycle_to_smallest(cycle: &mut Vec<usize>) {
    if cycle.is_empty() {
        return;
    }
    let min_val = *cycle.iter().min().unwrap();
    if let Some(pos) = cycle.iter().position(|&x| x == min_val) {
        cycle.rotate_left(pos);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_square() {
        // 4-cycle: edges (0-1, 1-2, 2-3, 3-0).
        let mut g = Graph::new(4);
        g.add_edge(0, 1);
        g.add_edge(1, 2);
        g.add_edge(2, 3);
        g.add_edge(3, 0);

        let cycles = g.all_hamiltonian_cycles();
        // Only one unique cycle: [0,1,2,3]
        // (the reversed or rotated forms are equivalent).
        assert_eq!(cycles, vec![vec![0, 1, 2, 3]]);
    }

    #[test]
    fn test_no_cycle() {
        // A tree: 0-1, 1-2 => no Hamiltonian cycle (cannot visit all nodes exactly once and return).
        let mut g = Graph::new(3);
        g.add_edge(0, 1);
        g.add_edge(1, 2);
        let cycles = g.all_hamiltonian_cycles();
        assert!(cycles.is_empty());
    }

    #[test]
    fn test_triangle() {
        // 0-1, 1-2, 2-0 => single triangle cycle [0,1,2].
        let mut g = Graph::new(3);
        g.add_edge(0, 1);
        g.add_edge(1, 2);
        g.add_edge(2, 0);
        let cycles = g.all_hamiltonian_cycles();
        assert_eq!(cycles, vec![vec![0, 1, 2]]);
    }

    #[test]
    fn test_multiple_edges() {
        // 0-1, 1-2, 2-0, plus an extra chord 0-2
        // Still the only Hamiltonian cycle is [0,1,2].
        let mut g = Graph::new(3);
        g.add_edge(0, 1);
        g.add_edge(1, 2);
        g.add_edge(2, 0);
        g.add_edge(0, 2);
        let cycles = g.all_hamiltonian_cycles();
        assert_eq!(cycles, vec![vec![0, 1, 2]]);
    }
} 