/// Represents an undirected, weighted edge in a graph.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Edge {
    pub src: usize,
    pub dst: usize,
    pub weight: i32,
}

/// Disjoint-set (union-find) for cycle detection in Kruskal's algorithm.
pub struct UnionFind {
    parent: Vec<usize>,
    rank: Vec<usize>,
}

impl UnionFind {
    /// Initializes a union-find for `n` elements (0..n-1).
    pub fn new(n: usize) -> Self {
        Self {
            parent: (0..n).collect(),
            rank: vec![0; n],
        }
    }

    /// Finds the representative (root) of the set containing `x`.  
    /// Uses path compression.
    pub fn find(&mut self, x: usize) -> usize {
        if self.parent[x] != x {
            self.parent[x] = self.find(self.parent[x]);
        }
        self.parent[x]
    }

    /// Unites the sets containing `x` and `y`.  
    /// Returns `true` if a union actually occurred (i.e., they were disjoint).
    pub fn union(&mut self, x: usize, y: usize) -> bool {
        let rx = self.find(x);
        let ry = self.find(y);

        if rx != ry {
            if self.rank[rx] > self.rank[ry] {
                self.parent[ry] = rx;
            } else if self.rank[rx] < self.rank[ry] {
                self.parent[rx] = ry;
            } else {
                self.parent[ry] = rx;
                self.rank[rx] += 1;
            }
            true
        } else {
            false
        }
    }
}

/// Kruskal's algorithm to compute the MST for an undirected, weighted graph.
///
/// - `num_nodes` is the number of vertices in the graph (assumed labeled 0..(num_nodes-1)).
/// - `edges` is a mutable slice of [`Edge`].
///
/// Returns the edges that form the MST.  
/// If the graph is disconnected, this will return a spanning forest of all connected components.
pub fn kruskal(num_nodes: usize, edges: &mut [Edge]) -> Vec<Edge> {
    // Sort edges by non-decreasing weight
    edges.sort_by_key(|e| e.weight);

    let mut uf = UnionFind::new(num_nodes);
    let mut mst = Vec::new();

    // Keep track of number of edges in MST
    let mut edges_in_mst = 0;
    let target_edges = num_nodes.saturating_sub(1);

    for edge in edges {
        // Check if adding this edge would create a cycle
        if uf.find(edge.src) != uf.find(edge.dst) {
            // Add edge to MST and merge components
            mst.push(edge.clone());
            uf.union(edge.src, edge.dst);
            edges_in_mst += 1;

            // Early exit if we have enough edges
            if edges_in_mst == target_edges {
                break;
            }
        }
    }
    mst
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_graph() {
        let mut edges = vec![];
        let mst = kruskal(0, &mut edges);
        assert!(mst.is_empty(), "MST of empty graph should be empty");
    }

    #[test]
    fn test_single_vertex_no_edges() {
        let mut edges = vec![];
        let mst = kruskal(1, &mut edges);
        assert!(
            mst.is_empty(),
            "MST of single vertex with no edges should be empty"
        );
    }

    #[test]
    fn test_single_edge() {
        let mut edges = vec![Edge {
            src: 0,
            dst: 1,
            weight: 2,
        }];
        let mst = kruskal(2, &mut edges);
        assert_eq!(mst.len(), 1);
        assert_eq!(
            mst[0],
            Edge {
                src: 0,
                dst: 1,
                weight: 2
            }
        );
    }

    #[test]
    fn test_disconnected_components() {
        let mut edges = vec![
            Edge {
                src: 0,
                dst: 1,
                weight: 1,
            },
            Edge {
                src: 2,
                dst: 3,
                weight: 2,
            },
        ];
        // 4 nodes: (0,1) is disconnected from (2,3)
        let mst = kruskal(4, &mut edges);
        assert_eq!(
            mst.len(),
            2,
            "Should return one edge per connected component minus 1 edge each."
        );
        // Check that each edge is in the MST
        assert!(mst.contains(&Edge {
            src: 0,
            dst: 1,
            weight: 1
        }));
        assert!(mst.contains(&Edge {
            src: 2,
            dst: 3,
            weight: 2
        }));
    }

    #[test]
    fn test_standard_graph() {
        // A small graph with 4 vertices
        // (0)---10---(1)
        //  | \       /
        //  6  5    15
        //  |   \   /
        // (2)---4---(3)
        let mut edges = vec![
            Edge {
                src: 0,
                dst: 1,
                weight: 10,
            },
            Edge {
                src: 0,
                dst: 2,
                weight: 6,
            },
            Edge {
                src: 0,
                dst: 3,
                weight: 5,
            },
            Edge {
                src: 1,
                dst: 3,
                weight: 15,
            },
            Edge {
                src: 2,
                dst: 3,
                weight: 4,
            },
        ];
        let mst = kruskal(4, &mut edges);

        // MST should be edges: (2-3, 0-3, 0-2) or (2-3, 0-2, 0-1) depending on sorting, etc.
        // In either case, it will have 3 edges.
        assert_eq!(mst.len(), 3);

        // Check minimal total weight or check known edges:
        let total_weight: i32 = mst.iter().map(|e| e.weight).sum();
        // The MST includes edges with weights 4,5,6 or 4,5,10 or 4,6,10.
        // Sorted edges are (2-3=4), (0-3=5), (0-2=6), (0-1=10), (1-3=15).
        // Typical pick is 4 + 5 + 6 = 15.
        // It's also possible 4 + 5 + 10 = 19 if the union-find picks incorrectly,
        // but given standard logic, it should always pick the smaller edges first.
        assert_eq!(
            total_weight, 15,
            "Kruskal's MST should have a total weight of 15"
        );
    }

    #[test]
    fn test_negative_weights() {
        let mut edges = vec![
            Edge {
                src: 0,
                dst: 1,
                weight: -2,
            },
            Edge {
                src: 1,
                dst: 2,
                weight: -3,
            },
            Edge {
                src: 0,
                dst: 2,
                weight: -1,
            },
            Edge {
                src: 2,
                dst: 3,
                weight: 2,
            },
        ];
        let mst = kruskal(4, &mut edges);
        // MST for 4 vertices will have 3 edges.
        assert_eq!(mst.len(), 3);
        let total_weight: i32 = mst.iter().map(|e| e.weight).sum();
        // The smallest edges by weight are -3, -2, -1 => but we can only pick 2 or 3 of them
        // depending on whether they form a cycle. Actually, -3 (1-2), -2 (0-1), and 2 (2-3)
        // or -3, -1, 2 or -2, -1, 2. Let's see:
        //   sort: (-3 -> 1-2), (-2 -> 0-1), (-1 -> 0-2), (2 -> 2-3)
        // picks: (-3 -> 1-2), (-2 -> 0-1) => next is (-1 -> 0-2) but that forms a cycle (0,1,2).
        // so it picks (2 -> 2-3). total = -3 + -2 + 2 = -3.
        assert_eq!(
            total_weight, -3,
            "MST should properly handle negative weights"
        );
    }

    #[test]
    fn test_parallel_edges() {
        // Graph with parallel edges between the same vertices
        let mut edges = vec![
            Edge {
                src: 0,
                dst: 1,
                weight: 10,
            },
            Edge {
                src: 0,
                dst: 1,
                weight: 1,
            }, // parallel edge, smaller weight
            Edge {
                src: 1,
                dst: 2,
                weight: 5,
            },
        ];
        // 3 vertices (0,1,2). MST should pick the edge with weight 1, not 10
        let mst = kruskal(3, &mut edges);

        assert_eq!(mst.len(), 2);
        // Check that the smaller parallel edge was chosen
        assert!(mst.contains(&Edge {
            src: 0,
            dst: 1,
            weight: 1
        }));
        // The other edge to connect all nodes is (1-2)
        assert!(mst.contains(&Edge {
            src: 1,
            dst: 2,
            weight: 5
        }));
    }
}
