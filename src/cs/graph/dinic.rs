use std::cmp::min;
use std::collections::VecDeque;

#[derive(Debug, Clone)]
struct Edge {
    to: usize,
    capacity: i32,
    flow: i32,
    rev: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dinic_max_flow() {
        let mut dinic = Dinic::new(6);
        dinic.add_edge(0, 1, 10);
        dinic.add_edge(0, 2, 10);
        dinic.add_edge(1, 3, 4);
        dinic.add_edge(1, 4, 8);
        dinic.add_edge(2, 4, 9);
        dinic.add_edge(3, 5, 10);
        dinic.add_edge(4, 3, 6);
        dinic.add_edge(4, 5, 10);

        let max_flow = dinic.max_flow(0, 5);
        assert_eq!(max_flow, 19);
    }

    #[test]
    fn test_dinic_max_flow_disconnected() {
        let mut dinic = Dinic::new(4);
        dinic.add_edge(0, 1, 10);
        dinic.add_edge(2, 3, 5);
        let max_flow = dinic.max_flow(0, 3);
        assert_eq!(max_flow, 0);
    }

    #[test]
    fn test_dinic_max_flow_no_flow() {
        let mut dinic = Dinic::new(3);
        dinic.add_edge(0, 1, 10);
        let max_flow = dinic.max_flow(0, 2);
        assert_eq!(max_flow, 0);
    }

    #[test]
    fn test_dinic_max_flow_multiple_paths() {
        let mut dinic = Dinic::new(4);
        dinic.add_edge(0, 1, 10);
        dinic.add_edge(0, 2, 5);
        dinic.add_edge(1, 3, 10);
        dinic.add_edge(2, 3, 5);
        let max_flow = dinic.max_flow(0, 3);
        assert_eq!(max_flow, 15);
    }

    #[test]
    fn test_dinic_max_flow_complex() {
        let mut dinic = Dinic::new(7);
        dinic.add_edge(0, 1, 10);
        dinic.add_edge(0, 2, 5);
        dinic.add_edge(1, 3, 9);
        dinic.add_edge(1, 4, 3);
        dinic.add_edge(2, 4, 7);
        dinic.add_edge(2, 5, 2);
        dinic.add_edge(3, 6, 10);
        dinic.add_edge(4, 6, 10);
        dinic.add_edge(5, 6, 5);
        let max_flow = dinic.max_flow(0, 6);
        assert_eq!(max_flow, 15);
    }
}

#[derive(Debug, Clone)]
pub struct Dinic {
    graph: Vec<Vec<Edge>>,
    n: usize,
}

impl Dinic {
    pub fn new(n: usize) -> Self {
        Dinic {
            graph: vec![Vec::new(); n],
            n,
        }
    }

    pub fn add_edge(&mut self, from: usize, to: usize, capacity: i32) {
        let to_len = self.graph[to].len();
        let from_len = self.graph[from].len();
        self.graph[from].push(Edge {
            to,
            capacity,
            flow: 0,
            rev: to_len,
        });
        self.graph[to].push(Edge {
            to: from,
            capacity: 0,
            flow: 0,
            rev: from_len,
        });
    }

    fn bfs(&self, s: usize, t: usize, level: &mut [i32]) -> bool {
        level.fill(-1);
        level[s] = 0;
        let mut queue = VecDeque::new();
        queue.push_back(s);

        while let Some(u) = queue.pop_front() {
            for edge in &self.graph[u] {
                if edge.capacity - edge.flow > 0 && level[edge.to] == -1 {
                    level[edge.to] = level[u] + 1;
                    queue.push_back(edge.to);
                }
            }
        }
        level[t] != -1
    }

    fn dfs(
        &mut self,
        u: usize,
        t: usize,
        level: &Vec<i32>,
        flow: i32,
        start: &mut Vec<usize>,
    ) -> i32 {
        if u == t {
            return flow;
        }
        while start[u] < self.graph[u].len() {
            let i = start[u];
            let (capacity, to, rev) = {
                let edge = &self.graph[u][i];
                (edge.capacity - edge.flow, edge.to, edge.rev)
            };

            if capacity > 0 && level[to] == level[u] + 1 {
                let pushed = self.dfs(to, t, level, min(flow, capacity), start);
                if pushed > 0 {
                    self.graph[u][i].flow += pushed;
                    self.graph[to][rev].flow -= pushed;
                    return pushed;
                }
            }
            start[u] += 1;
        }
        0
    }

    pub fn max_flow(&mut self, s: usize, t: usize) -> i32 {
        let mut total_flow = 0;
        let mut level = vec![0; self.n];
        while self.bfs(s, t, &mut level) {
            let mut start = vec![0; self.n];
            while let Some(flow) = Some(self.dfs(s, t, &level, i32::MAX, &mut start)) {
                if flow == 0 {
                    break;
                }
                total_flow += flow;
            }
        }
        total_flow
    }
}
