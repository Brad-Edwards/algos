use rand::Rng;

pub struct UnionFind {
    parent: Vec<usize>,
    rank: Vec<usize>,
}

impl UnionFind {
    pub fn new(n: usize) -> Self {
        Self {
            parent: (0..n).collect(),
            rank: vec![0; n],
        }
    }

    pub fn find(&mut self, x: usize) -> usize {
        if self.parent[x] != x {
            self.parent[x] = self.find(self.parent[x]);
        }
        self.parent[x]
    }

    pub fn union(&mut self, x: usize, y: usize) -> bool {
        let x_root = self.find(x);
        let y_root = self.find(y);
        if x_root == y_root {
            return false;
        }
        match self.rank[x_root].cmp(&self.rank[y_root]) {
            std::cmp::Ordering::Less => self.parent[x_root] = y_root,
            std::cmp::Ordering::Greater => self.parent[y_root] = x_root,
            std::cmp::Ordering::Equal => {
                self.parent[y_root] = x_root;
                self.rank[x_root] += 1;
            }
        }
        true
    }
}

/// Implements Karger's randomized min cut algorithm.
///
/// # Arguments
/// - `num_vertices`: Number of vertices in the graph.
/// - `edges`: Slice of edges as (u, v) pairs (0-indexed). The graph is undirected.
/// - `trials`: Number of independent trials to run (the more, the higher the chance to find the minimum cut).
///
/// # Returns
/// The estimated minimum cut value.
pub fn karger_min_cut(num_vertices: usize, edges: &[(usize, usize)], trials: usize) -> usize {
    let mut best_cut = usize::MAX;
    let mut rng = rand::thread_rng();
    for _ in 0..trials {
        let mut uf = UnionFind::new(num_vertices);
        let mut remaining = num_vertices;
        let mut current_edges = edges.to_vec();
        while remaining > 2 {
            let idx = rng.gen_range(0..current_edges.len());
            let (u, v) = current_edges[idx];
            let set_u = uf.find(u);
            let set_v = uf.find(v);
            if set_u == set_v {
                // Remove self-loop edge.
                current_edges.remove(idx);
                continue;
            }
            uf.union(set_u, set_v);
            remaining -= 1;
            // Remove self-loops.
            current_edges.retain(|&(a, b)| uf.find(a) != uf.find(b));
        }
        let cut = edges
            .iter()
            .filter(|&&(u, v)| uf.find(u) != uf.find(v))
            .count();
        if cut < best_cut {
            best_cut = cut;
        }
    }
    best_cut
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_karger_min_cut_triangle() {
        // Triangle graph: 3 vertices, 3 edges; minimum cut is 2.
        let vertices = 3;
        let edges = vec![(0, 1), (1, 2), (2, 0)];
        let cut = karger_min_cut(vertices, &edges, 100);
        assert_eq!(cut, 2);
    }

    #[test]
    fn test_karger_min_cut_square() {
        // Square graph with a diagonal:
        // Vertices: 0,1,2,3; Edges: (0,1), (1,2), (2,3), (3,0), (0,2)
        // Minimum cut should be small (often 2 or 3).
        let vertices = 4;
        let edges = vec![(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)];
        let cut = karger_min_cut(vertices, &edges, 200);
        // Due to randomness, allow a small tolerance.
        assert!(cut <= 3);
    }
}
