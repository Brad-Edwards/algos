use crate::cs::error::Error;
use std::cmp::min;
use std::collections::VecDeque;

pub fn edmond_karp(capacity: &[Vec<i32>], source: usize, sink: usize) -> Result<i32, Error> {
    let n = capacity.len();
    if n == 0 {
        return Ok(0);
    }

    if source >= n || sink >= n {
        return Err(Error::InvalidVertex);
    }

    // Initialize flow network
    let mut flow = vec![vec![0; n]; n];
    let mut max_flow = 0;

    // While there exists an augmenting path
    while let Some((path, path_flow)) = find_augmenting_path(capacity, &flow, source, sink) {
        // Update flow along the path
        for i in 0..path.len() - 1 {
            let u = path[i];
            let v = path[i + 1];
            flow[u][v] += path_flow;
            flow[v][u] -= path_flow; // Reverse edge
        }
        max_flow += path_flow;
    }

    Ok(max_flow)
}

fn find_augmenting_path(
    capacity: &[Vec<i32>],
    flow: &[Vec<i32>],
    source: usize,
    sink: usize,
) -> Option<(Vec<usize>, i32)> {
    let n = capacity.len();
    let mut visited = vec![false; n];
    let mut parent = vec![None; n];
    let mut queue = VecDeque::new();

    queue.push_back(source);
    visited[source] = true;

    // BFS to find augmenting path
    while let Some(u) = queue.pop_front() {
        for v in 0..n {
            let residual_capacity = capacity[u][v] - flow[u][v];
            if !visited[v] && residual_capacity > 0 {
                parent[v] = Some(u);
                visited[v] = true;

                if v == sink {
                    return construct_path(capacity, flow, &parent, source, sink);
                }
                queue.push_back(v);
            }
        }
    }
    None
}

fn construct_path(
    capacity: &[Vec<i32>],
    flow: &[Vec<i32>],
    parent: &[Option<usize>],
    source: usize,
    sink: usize,
) -> Option<(Vec<usize>, i32)> {
    let mut path = Vec::new();
    let mut curr = sink;
    let mut min_flow = i32::MAX;

    // Trace back the path and find minimum residual capacity
    while curr != source {
        let prev = parent[curr].unwrap();
        path.push(curr);
        min_flow = min(min_flow, capacity[prev][curr] - flow[prev][curr]);
        curr = prev;
    }
    path.push(source);
    path.reverse();
    Some((path, min_flow))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_edmond_karp_empty_graph() {
        let capacity = vec![vec![0; 0]; 0];
        let source = 0;
        let sink = 0;
        let result = edmond_karp(&capacity, source, sink);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 0);
    }

    #[test]
    fn test_edmond_karp_simple_graph() {
        let capacity = vec![
            vec![0, 16, 13, 0, 0, 0],
            vec![0, 0, 10, 12, 0, 0],
            vec![0, 4, 0, 0, 14, 0],
            vec![0, 0, 9, 0, 0, 20],
            vec![0, 0, 0, 7, 0, 4],
            vec![0, 0, 0, 0, 0, 0],
        ];
        let source = 0;
        let sink = 5;
        let result = edmond_karp(&capacity, source, sink);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 23);
    }

    #[test]
    fn test_edmond_karp_multiple_paths() {
        let capacity = vec![
            vec![0, 10, 10, 0, 0],
            vec![0, 0, 4, 8, 0],
            vec![0, 0, 0, 9, 0],
            vec![0, 0, 0, 0, 10],
            vec![0, 0, 0, 0, 0],
        ];
        let source = 0;
        let sink = 4;
        let result = edmond_karp(&capacity, source, sink);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 10);
    }

    #[test]
    fn test_edmond_karp_no_path() {
        let capacity = vec![
            vec![0, 10, 0, 0],
            vec![0, 0, 0, 0],
            vec![0, 0, 0, 10],
            vec![0, 0, 0, 0],
        ];
        let source = 0;
        let sink = 3;
        let result = edmond_karp(&capacity, source, sink);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 0);
    }

    #[test]
    fn test_edmond_karp_invalid_vertex() {
        let capacity = vec![vec![0, 10], vec![0, 0]];
        let source = 0;
        let sink = 2;
        let result = edmond_karp(&capacity, source, sink);
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), Error::InvalidVertex));
    }
}
