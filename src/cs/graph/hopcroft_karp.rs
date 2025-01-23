use std::collections::{HashMap, VecDeque};

#[derive(Debug, Clone)]
pub struct HopcroftKarp {
    graph: HashMap<usize, Vec<usize>>,
    n: usize,
    m: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hopcroft_karp_max_matching() {
        let mut hopcroft_karp = HopcroftKarp::new(4, 4);
        hopcroft_karp.add_edge(0, 1);
        hopcroft_karp.add_edge(1, 0);
        hopcroft_karp.add_edge(1, 2);
        hopcroft_karp.add_edge(2, 1);
        hopcroft_karp.add_edge(2, 3);
        hopcroft_karp.add_edge(3, 2);
        let max_matching = hopcroft_karp.max_matching();
        assert_eq!(max_matching, 2);
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
    pub fn new(n: usize, m: usize) -> Self {
        HopcroftKarp {
            graph: HashMap::new(),
            n,
            m,
        }
    }

    pub fn add_edge(&mut self, u: usize, v: usize) {
        self.graph.entry(u).or_default().push(v);
    }

    pub fn max_matching(&mut self) -> usize {
        let mut matching = vec![-1; self.m];
        let mut result = 0;

        while self.bfs(&mut matching) {
            for u in 0..self.n {
                if !self.graph.contains_key(&u) {
                    continue;
                }
                if self.dfs(u, &mut matching, &mut vec![false; self.n]) {
                    result += 1;
                }
            }
        }
        result
    }

    fn bfs(&self, matching: &mut Vec<i32>) -> bool {
        let mut dist = vec![-1; self.n];
        let mut queue = VecDeque::new();

        for u in 0..self.n {
            if !self.graph.contains_key(&u) {
                continue;
            }
            let mut matched = false;
            for v in &self.graph[&u] {
                if matching[*v] as usize == u {
                    matched = true;
                    break;
                }
            }
            if !matched {
                dist[u] = 0;
                queue.push_back(u);
            }
        }

        let mut found_path = false;
        while let Some(u) = queue.pop_front() {
            if !self.graph.contains_key(&u) {
                continue;
            }
            for v in &self.graph[&u] {
                if matching[*v] == -1 {
                    found_path = true;
                } else if dist[matching[*v] as usize] == -1 {
                    dist[matching[*v] as usize] = dist[u] + 1;
                    queue.push_back(matching[*v] as usize);
                }
            }
        }
        found_path
    }

    fn dfs(&self, u: usize, matching: &mut Vec<i32>, visited: &mut Vec<bool>) -> bool {
        if visited[u] {
            return false;
        }
        visited[u] = true;
        if !self.graph.contains_key(&u) {
            return false;
        }
        for v in &self.graph[&u] {
            if matching[*v] == -1
                || (!visited[matching[*v] as usize]
                    && self.dfs(matching[*v] as usize, matching, visited))
            {
                matching[*v] = u as i32;
                return true;
            }
        }
        false
    }
}
