/// A simple undirected graph using adjacency lists.
#[derive(Clone, Debug)]
pub struct Graph {
    n: usize,
    adj: Vec<Vec<usize>>,
}

impl Graph {
    /// Create an empty graph with `n` vertices (labeled 0..n-1).
    pub fn new(n: usize) -> Self {
        Graph {
            n,
            adj: vec![Vec::new(); n],
        }
    }

    /// Number of vertices.
    pub fn len(&self) -> usize {
        self.n
    }

    /// Add an undirected edge between vertices `u` and `v`.
    pub fn add_edge(&mut self, u: usize, v: usize) {
        assert!(u < self.n && v < self.n, "Invalid vertex index");
        self.adj[u].push(v);
        self.adj[v].push(u);
    }
}

/// Implementation of Edmonds' Blossom algorithm for finding maximum matchings in undirected graphs.
/// A matching in a graph is a set of edges where no two edges share a vertex. A maximum matching
/// is a matching that contains the largest possible number of edges.
///
/// # Algorithm Overview
/// The algorithm works by repeatedly finding augmenting paths in the graph. An augmenting path
/// is a path that starts and ends at unmatched vertices and alternates between unmatched and
/// matched edges. The key insight of Edmonds' algorithm is handling "blossoms" - odd-length
/// cycles that need to be contracted during the search.
///
/// # Complexity
/// - Time complexity: O(V³), where V is the number of vertices
/// - Space complexity: O(V), for storing the matching and auxiliary data structures
///
/// # Example
/// ```
/// use algos::cs::graph::edmonds_blossom::{Graph, edmonds_blossom_max_matching};
///
/// // Create a simple path of 3 vertices
/// let mut g = Graph::new(3);
/// g.add_edge(0, 1);
/// g.add_edge(1, 2);
///
/// // Find maximum matching
/// let matching = edmonds_blossom_max_matching(&g);
/// // The matching will contain one edge, matching[i] gives the vertex matched to i
/// assert_eq!(matching.len(), 3);
/// assert!(matching.iter().filter(|x| x.is_some()).count() == 2); // 2 vertices matched
/// ```
/// Computes a maximum matching in the given undirected graph using Edmonds' Blossom algorithm.
/// Vertices that are unmatched are stored as `None` in the output vector.
/// Internally, we use `INF = n` as a sentinel value.
pub fn edmonds_blossom_max_matching(g: &Graph) -> Vec<Option<usize>> {
    let n = g.len();
    let inf = n; // sentinel value: no match
    let mut matchv = vec![inf; n];
    let mut p = vec![inf; n];
    let mut base: Vec<usize> = (0..n).collect();
    let _used = vec![false; n];
    let _blossom = vec![false; n];

    /// Finds the least common ancestor (LCA) of vertices v and w in the alternating tree.
    /// This is used to identify the base of a blossom when one is found during the search.
    /// The alternating tree is implicitly defined by the parent array p and the current matching.
    fn lca(
        v: usize,
        w: usize,
        p: &Vec<usize>,
        base: &Vec<usize>,
        matchv: &Vec<usize>,
        inf: usize,
    ) -> usize {
        let n = p.len();
        let mut used_flag = vec![false; n];
        let mut v = v;
        loop {
            v = base[v];
            used_flag[v] = true;
            if matchv[v] == inf {
                break;
            }
            v = p[matchv[v]];
        }
        let mut w = w;
        while !used_flag[base[w]] {
            w = p[matchv[w]];
        }
        base[w]
    }

    /// Marks the vertices in a blossom for contraction.
    /// When a blossom is found (an odd cycle), we need to contract it into a single vertex
    /// for the purpose of finding an augmenting path. This function marks all vertices that
    /// are part of the blossom by following the alternating path from v up to the base vertex b.
    fn mark_path(
        v: usize,
        b: usize,
        mut x: usize,
        p: &mut Vec<usize>,
        base: &mut Vec<usize>,
        _used: &mut Vec<bool>,
        matchv: &Vec<usize>,
        blossom: &mut Vec<bool>,
        _inf: usize,
    ) {
        let mut v = v;
        while base[v] != b {
            blossom[base[v]] = true;
            blossom[base[matchv[v]]] = true;
            p[v] = x;
            // Move one step up in the tree.
            x = matchv[v];
            v = p[matchv[v]];
        }
    }

    /// Finds an augmenting path starting from an unmatched vertex using BFS.
    /// An augmenting path alternates between unmatched and matched edges, starting and ending
    /// at unmatched vertices. When found, such a path can be used to increase the size of the matching.
    /// The function handles blossom contraction when odd cycles are encountered during the search.
    fn find_path(
        start: usize,
        g: &Graph,
        matchv: &Vec<usize>,
        p: &mut Vec<usize>,
        base: &mut Vec<usize>,
        inf: usize,
    ) -> Option<usize> {
        let n = g.len();
        let mut used = vec![false; n];
        let mut q = std::collections::VecDeque::new();
        q.push_back(start);
        used[start] = true;
        for i in 0..n {
            p[i] = inf;
        }
        while let Some(v) = q.pop_front() {
            for &to in &g.adj[v] {
                if base[v] == base[to] || matchv[v] == to {
                    continue;
                }
                if to == start || (matchv[to] != inf && p[matchv[to]] != inf) {
                    let cur = lca(v, to, p, base, matchv, inf);
                    let mut blossom_flag = vec![false; n];
                    mark_path(
                        v,
                        cur,
                        to,
                        p,
                        base,
                        &mut used,
                        matchv,
                        &mut blossom_flag,
                        inf,
                    );
                    mark_path(
                        to,
                        cur,
                        v,
                        p,
                        base,
                        &mut used,
                        matchv,
                        &mut blossom_flag,
                        inf,
                    );
                    for i in 0..n {
                        if blossom_flag[base[i]] {
                            base[i] = cur;
                            if !used[i] {
                                used[i] = true;
                                q.push_back(i);
                            }
                        }
                    }
                } else if p[to] == inf {
                    p[to] = v;
                    if matchv[to] == inf {
                        return Some(to);
                    } else {
                        used[matchv[to]] = true;
                        q.push_back(matchv[to]);
                    }
                }
            }
        }
        None
    }

    /// Augments the matching along the found path.
    /// Given an augmenting path from start to finish, this function flips the matched/unmatched
    /// status of edges along the path to increase the size of the matching by one.
    fn augment_path(
        start: usize,
        finish: usize,
        matchv: &mut Vec<usize>,
        p: &Vec<usize>,
        inf: usize,
    ) {
        let mut cur = finish;
        while cur != start {
            let prev = p[cur];
            let nxt = matchv[prev];
            matchv[cur] = prev;
            matchv[prev] = cur;
            cur = nxt;
            if cur == inf {
                break;
            }
        }
    }

    // Main loop: for each free vertex, try to find an augmenting path.
    for v in 0..n {
        if matchv[v] == inf {
            if let Some(finish) = find_path(v, g, &matchv, &mut p, &mut base, inf) {
                augment_path(v, finish, &mut matchv, &p, inf);
            }
        }
    }

    // Convert sentinel inf back to None.
    matchv
        .into_iter()
        .map(|x| if x == inf { None } else { Some(x) })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// In a triangle (3-cycle), a maximum matching has one edge.
    #[test]
    fn test_simple_triangle() {
        let mut g = Graph::new(3);
        g.add_edge(0, 1);
        g.add_edge(1, 2);
        g.add_edge(2, 0);
        let matching = edmonds_blossom_max_matching(&g);
        let matched_edges: Vec<_> = matching
            .iter()
            .enumerate()
            .filter_map(|(v, &m)| {
                if let Some(u) = m {
                    if v < u {
                        Some((v, u))
                    } else {
                        None
                    }
                } else {
                    None
                }
            })
            .collect();
        assert_eq!(
            matched_edges.len(),
            1,
            "Triangle should yield 1 matched edge"
        );
    }

    /// In a square with diagonals, the maximum matching has two edges.
    #[test]
    fn test_square() {
        let mut g = Graph::new(4);
        g.add_edge(0, 1);
        g.add_edge(1, 2);
        g.add_edge(2, 3);
        g.add_edge(3, 0);
        g.add_edge(0, 2);
        g.add_edge(1, 3);
        let matching = edmonds_blossom_max_matching(&g);
        let matched_edges: Vec<_> = matching
            .iter()
            .enumerate()
            .filter_map(|(v, &m)| {
                if let Some(u) = m {
                    if v < u {
                        Some((v, u))
                    } else {
                        None
                    }
                } else {
                    None
                }
            })
            .collect();
        assert_eq!(
            matched_edges.len(),
            2,
            "Square with diagonals should yield 2 matched edges"
        );
    }

    /// A classic blossom case: a 5-cycle with an extra chord.
    #[test]
    fn test_blossom_case() {
        let mut g = Graph::new(5);
        g.add_edge(0, 1);
        g.add_edge(1, 2);
        g.add_edge(2, 3);
        g.add_edge(3, 4);
        g.add_edge(4, 0);
        g.add_edge(1, 3);
        let matching = edmonds_blossom_max_matching(&g);
        let matched_edges: Vec<_> = matching
            .iter()
            .enumerate()
            .filter_map(|(v, &m)| {
                if let Some(u) = m {
                    if v < u {
                        Some((v, u))
                    } else {
                        None
                    }
                } else {
                    None
                }
            })
            .collect();
        // For this blossom case, a maximum matching has two edges.
        assert_eq!(
            matched_edges.len(),
            2,
            "Blossom case should yield 2 matched edges"
        );
    }
}
