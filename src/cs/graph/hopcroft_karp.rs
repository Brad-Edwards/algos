use std::collections::{HashMap, VecDeque};

#[derive(Debug, Clone)]
pub struct HopcroftKarp {
    // Stores edges from left partition to right partition
    graph: HashMap<usize, Vec<usize>>,
    n: usize, // size of left partition
    m: usize, // size of right partition
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hopcroft_karp_max_matching() {
        let mut hopcroft_karp = HopcroftKarp::new(4, 4);
        // Add edges only from left to right partition
        hopcroft_karp.add_edge(0, 1);
        hopcroft_karp.add_edge(1, 2);
        hopcroft_karp.add_edge(2, 3);
        let max_matching = hopcroft_karp.max_matching();
        // We can match: 0->1, 1->2, 2->3
        assert_eq!(max_matching, 3);
    }

    #[test]
    fn test_hopcroft_karp_max_matching_2() {
        let mut hopcroft_karp = HopcroftKarp::new(5, 5);
        hopcroft_karp.add_edge(0, 1);
        hopcroft_karp.add_edge(0, 2);
        hopcroft_karp.add_edge(1, 2);
        hopcroft_karp.add_edge(1, 3);
        hopcroft_karp.add_edge(2, 4);
        let max_matching = hopcroft_karp.max_matching();
        assert_eq!(max_matching, 3);
    }

    #[test]
    fn test_hopcroft_karp_max_matching_empty() {
        let mut hopcroft_karp = HopcroftKarp::new(0, 0);
        let max_matching = hopcroft_karp.max_matching();
        assert_eq!(max_matching, 0);
    }
}

impl HopcroftKarp {
    /// Creates a new HopcroftKarp instance for bipartite matching
    /// n: size of left partition
    /// m: size of right partition
    pub fn new(n: usize, m: usize) -> Self {
        HopcroftKarp {
            graph: HashMap::new(),
            n,
            m,
        }
    }

    /// Adds an edge from vertex u in left partition to vertex v in right partition
    pub fn add_edge(&mut self, u: usize, v: usize) {
        self.graph.entry(u).or_default().push(v);
    }

    /// Finds the maximum matching in the bipartite graph
    pub fn max_matching(&mut self) -> usize {
        // pair[v] = u means vertex v from right is matched with vertex u from left
        let mut pair = vec![-1; self.m];
        // matched[u] = v means vertex u from left is matched with vertex v from right
        let mut matched = vec![-1; self.n];
        let mut result = 0;

        loop {
            let mut queue = VecDeque::new();
            let mut used = vec![false; self.n];
            let mut dist = vec![-1; self.n];

            // Initialize queue with unmatched vertices from left partition
            for u in 0..self.n {
                if matched[u] == -1 && self.graph.contains_key(&u) {
                    dist[u] = 0;
                    queue.push_back(u);
                }
            }

            // BFS to find shortest augmenting paths
            while let Some(u) = queue.pop_front() {
                if let Some(edges) = self.graph.get(&u) {
                    for &v in edges {
                        if pair[v] == -1 {
                            // Found an augmenting path
                            continue;
                        }
                        let next_u = pair[v] as usize;
                        if dist[next_u] == -1 {
                            dist[next_u] = dist[u] + 1;
                            queue.push_back(next_u);
                        }
                    }
                }
            }

            // Try to find augmenting paths for unmatched vertices
            let mut found_path = false;
            for u in 0..self.n {
                if matched[u] == -1
                    && self.graph.contains_key(&u)
                    && !used[u]
                    && self.dfs(u, &mut pair, &mut matched, &mut used, &dist)
                {
                    found_path = true;
                    result += 1;
                }
            }

            if !found_path {
                break;
            }
        }

        result
    }

    fn dfs(
        &self,
        u: usize,
        pair: &mut Vec<i32>,
        matched: &mut Vec<i32>,
        used: &mut Vec<bool>,
        dist: &Vec<i32>,
    ) -> bool {
        used[u] = true;
        if let Some(edges) = self.graph.get(&u) {
            for &v in edges {
                let next_u = pair[v];
                if next_u == -1
                    || (!used[next_u as usize]
                        && dist[next_u as usize] == dist[u] + 1
                        && self.dfs(next_u as usize, pair, matched, used, dist))
                {
                    pair[v] = u as i32;
                    matched[u] = v as i32;
                    return true;
                }
            }
        }
        false
    }
}
