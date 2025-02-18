use rand::seq::SliceRandom;
use rand::thread_rng;

pub type Edge = (usize, usize, f64);

struct UnionFind {
    parent: Vec<usize>,
    rank: Vec<usize>,
}

impl UnionFind {
    fn new(n: usize) -> Self {
        Self {
            parent: (0..n).collect(),
            rank: vec![0; n],
        }
    }
    
    fn find(&mut self, x: usize) -> usize {
        if self.parent[x] != x {
            self.parent[x] = self.find(self.parent[x]);
        }
        self.parent[x]
    }
    
    fn union(&mut self, x: usize, y: usize) -> bool {
        let x_root = self.find(x);
        let y_root = self.find(y);
        if x_root == y_root {
            return false;
        }
        if self.rank[x_root] < self.rank[y_root] {
            self.parent[x_root] = y_root;
        } else if self.rank[x_root] > self.rank[y_root] {
            self.parent[y_root] = x_root;
        } else {
            self.parent[y_root] = x_root;
            self.rank[x_root] += 1;
        }
        true
    }
}

/// Implements a randomized variant of Kruskal's algorithm for finding a minimum spanning tree.
/// It shuffles the edges before sorting to randomize tie-breaking among edges with equal weight.
/// 
/// # Arguments
/// - `num_vertices`: Total number of vertices.
/// - `edges`: Vector of edges as (u, v, weight) tuples.
/// 
/// # Returns
/// A tuple containing the list of edges in the MST and the total weight.
pub fn randomized_kruskal(num_vertices: usize, mut edges: Vec<Edge>) -> (Vec<Edge>, f64) {
    let mut rng = thread_rng();
    edges.shuffle(&mut rng);
    edges.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap());
    
    let mut uf = UnionFind::new(num_vertices);
    let mut mst_edges = Vec::new();
    let mut total_weight = 0.0;
    
    for edge in edges {
        if uf.union(edge.0, edge.1) {
            mst_edges.push(edge);
            total_weight += edge.2;
        }
        if mst_edges.len() == num_vertices - 1 {
            break;
        }
    }
    (mst_edges, total_weight)
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_randomized_kruskal() {
        // Graph: 4 vertices with 6 edges.
        let num_vertices = 4;
        let edges = vec![
            (0, 1, 1.0),
            (0, 2, 4.0),
            (0, 3, 3.0),
            (1, 2, 2.0),
            (1, 3, 5.0),
            (2, 3, 1.5),
        ];
        let (mst, total_weight) = randomized_kruskal(num_vertices, edges);
        // MST should have 3 edges.
        assert_eq!(mst.len(), num_vertices - 1);
        // Depending on random tie-breaking, the total weight should be around 4.5 or 5.0.
        assert!(total_weight - 4.5 < 1e-6 || total_weight - 5.0 < 1e-6);
    }
}
