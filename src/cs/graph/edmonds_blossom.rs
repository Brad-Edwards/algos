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

/// Computes a maximum matching in the given undirected graph using Edmonds' Blossom algorithm.
/// Vertices that are unmatched are stored as `None` in the output vector.
/// Internally, we use `INF = n` as a sentinel value.
pub fn edmonds_blossom_max_matching(g: &Graph) -> Vec<Option<usize>> {
    let n = g.len();
    let INF = n; // sentinel value: no match
    let mut matchv = vec![INF; n];
    let mut p = vec![INF; n];
    let mut base: Vec<usize> = (0..n).collect();
    let mut used = vec![false; n];
    let mut blossom = vec![false; n];

    // Finds the least common ancestor of v and w in the alternating tree.
    fn lca(v: usize, w: usize, p: &Vec<usize>, base: &Vec<usize>, matchv: &Vec<usize>, INF: usize) -> usize {
        let n = p.len();
        let mut used_flag = vec![false; n];
        let mut v = v;
        loop {
            v = base[v];
            used_flag[v] = true;
            if matchv[v] == INF {
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

    // Marks the blossom along the path.
    fn mark_path(
        v: usize,
        b: usize,
        mut x: usize,
        p: &mut Vec<usize>,
        base: &mut Vec<usize>,
        used: &mut Vec<bool>,
        matchv: &Vec<usize>,
        blossom: &mut Vec<bool>,
        INF: usize,
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

    // Finds an augmenting path starting from 'start' using BFS.
    fn find_path(start: usize, g: &Graph, matchv: &Vec<usize>, p: &mut Vec<usize>, base: &mut Vec<usize>, INF: usize) -> Option<usize> {
        let n = g.len();
        let mut used = vec![false; n];
        let mut q = std::collections::VecDeque::new();
        q.push_back(start);
        used[start] = true;
        for i in 0..n {
            p[i] = INF;
        }
        while let Some(v) = q.pop_front() {
            for &to in &g.adj[v] {
                if base[v] == base[to] || matchv[v] == to {
                    continue;
                }
                if to == start || (matchv[to] != INF && p[matchv[to]] != INF) {
                    let cur = lca(v, to, p, base, matchv, INF);
                    let mut blossom_flag = vec![false; n];
                    mark_path(v, cur, to, p, base, &mut used, matchv, &mut blossom_flag, INF);
                    mark_path(to, cur, v, p, base, &mut used, matchv, &mut blossom_flag, INF);
                    for i in 0..n {
                        if blossom_flag[base[i]] {
                            base[i] = cur;
                            if !used[i] {
                                used[i] = true;
                                q.push_back(i);
                            }
                        }
                    }
                } else if p[to] == INF {
                    p[to] = v;
                    if matchv[to] == INF {
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

    // Augments the matching along the found path.
    fn augment_path(start: usize, finish: usize, matchv: &mut Vec<usize>, p: &Vec<usize>, INF: usize) {
        let mut cur = finish;
        while cur != start {
            let prev = p[cur];
            let nxt = matchv[prev];
            matchv[cur] = prev;
            matchv[prev] = cur;
            cur = nxt;
            if cur == INF {
                break;
            }
        }
    }

    // Main loop: for each free vertex, try to find an augmenting path.
    for v in 0..n {
        if matchv[v] == INF {
            if let Some(finish) = find_path(v, g, &matchv, &mut p, &mut base, INF) {
                augment_path(v, finish, &mut matchv, &p, INF);
            }
        }
    }

    // Convert sentinel INF back to None.
    matchv.into_iter().map(|x| if x == INF { None } else { Some(x) }).collect()
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
        let matched_edges: Vec<_> = matching.iter().enumerate().filter_map(|(v, &m)| {
            if let Some(u) = m {
                if v < u { Some((v, u)) } else { None }
            } else { None }
        }).collect();
        assert_eq!(matched_edges.len(), 1, "Triangle should yield 1 matched edge");
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
        let matched_edges: Vec<_> = matching.iter().enumerate().filter_map(|(v, &m)| {
            if let Some(u) = m {
                if v < u { Some((v, u)) } else { None }
            } else { None }
        }).collect();
        assert_eq!(matched_edges.len(), 2, "Square with diagonals should yield 2 matched edges");
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
        let matched_edges: Vec<_> = matching.iter().enumerate().filter_map(|(v, &m)| {
            if let Some(u) = m {
                if v < u { Some((v, u)) } else { None }
            } else { None }
        }).collect();
        // For this blossom case, a maximum matching has two edges.
        assert_eq!(matched_edges.len(), 2, "Blossom case should yield 2 matched edges");
    }
}
