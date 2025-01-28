use std::collections::VecDeque;

/// Performs a BFS on the residual graph to find an augmenting path from `s` to `t`.
/// Returns `true` if a path is found, and populates `parent[v]` with the predecessor
/// of `v` in the path. Otherwise returns `false`.
fn bfs(r_graph: &[Vec<i32>], s: usize, t: usize, parent: &mut [Option<usize>]) -> bool {
    let n = r_graph.len();
    let mut visited = vec![false; n];
    let mut queue = VecDeque::new();

    visited[s] = true;
    parent[s] = None;
    queue.push_back(s);

    while let Some(u) = queue.pop_front() {
        for (v, &cap) in r_graph[u].iter().enumerate() {
            if !visited[v] && cap > 0 {
                visited[v] = true;
                parent[v] = Some(u);
                queue.push_back(v);

                if v == t {
                    return true; // We reached the sink, no need to continue BFS
                }
            }
        }
    }
    false
}

/// Computes the maximum flow from `s` to `t` using the Ford-Fulkerson (Edmond-Karp) method.
/// The input `graph` is the capacity matrix where `graph[u][v]` gives the capacity of edge `u -> v`.
pub fn ford_fulkerson(graph: &[Vec<i32>], s: usize, t: usize) -> i32 {
    let n = graph.len();

    // Make a copy of the capacity graph to use as the residual graph.
    let mut r_graph = graph.to_vec();

    // This will store the path found by BFS.
    let mut parent = vec![None; n];

    let mut max_flow = 0;

    // While we can find an augmenting path in the residual graph...
    while bfs(&r_graph, s, t, &mut parent) {
        // Find the minimum residual capacity (bottleneck) along the path we just found.
        let mut path_flow = i32::MAX;
        let mut v = t;
        while let Some(u) = parent[v] {
            path_flow = path_flow.min(r_graph[u][v]);
            v = u;
        }

        // Update residual capacities along the path
        let mut v = t;
        while let Some(u) = parent[v] {
            r_graph[u][v] -= path_flow;
            r_graph[v][u] += path_flow;
            v = u;
        }

        // Add this path's flow to the total
        max_flow += path_flow;
    }

    max_flow
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ford_fulkerson() {
        let graph = vec![
            vec![0, 16, 13, 0, 0, 0],
            vec![0, 0, 10, 12, 0, 0],
            vec![0, 4, 0, 0, 14, 0],
            vec![0, 0, 9, 0, 0, 20],
            vec![0, 0, 0, 7, 0, 4],
            vec![0, 0, 0, 0, 0, 0],
        ];

        // From the example, max flow from 0 to 5 should be 23.
        let result = ford_fulkerson(&graph, 0, 5);
        assert_eq!(result, 23);
    }

    #[test]
    fn test_simple_path() {
        let graph = vec![vec![0, 10, 0], vec![0, 0, 10], vec![0, 0, 0]];
        assert_eq!(ford_fulkerson(&graph, 0, 2), 10);
    }

    #[test]
    fn test_no_path() {
        let graph = vec![vec![0, 0, 0], vec![10, 0, 10], vec![0, 0, 0]];
        assert_eq!(ford_fulkerson(&graph, 0, 2), 0);
    }

    #[test]
    fn test_parallel_paths() {
        let graph = vec![
            vec![0, 10, 10, 0],
            vec![0, 0, 0, 10],
            vec![0, 0, 0, 10],
            vec![0, 0, 0, 0],
        ];
        assert_eq!(ford_fulkerson(&graph, 0, 3), 20);
    }

    #[test]
    fn test_diamond_graph() {
        // Source -> (A,B) -> Sink
        // Two paths that share start and end
        let graph = vec![
            vec![0, 10, 10, 0],
            vec![0, 0, 0, 8],
            vec![0, 0, 0, 8],
            vec![0, 0, 0, 0],
        ];
        assert_eq!(ford_fulkerson(&graph, 0, 3), 16);
    }

    #[test]
    fn test_backward_flow() {
        // Tests if the algorithm correctly handles backward flow
        let graph = vec![
            vec![0, 5, 5, 0], // Source can send 5 to A and 5 to B
            vec![0, 0, 3, 5], // A can send 3 to B and 5 to Sink
            vec![0, 0, 0, 5], // B can send 5 to Sink
            vec![0, 0, 0, 0], // Sink
        ];
        assert_eq!(ford_fulkerson(&graph, 0, 3), 10); // Changed from 8 to 10
    }
}
