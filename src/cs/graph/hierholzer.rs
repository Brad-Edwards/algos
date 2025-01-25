use std::collections::HashMap;
use crate::cs::error::{Error, Result};

pub fn find_eulerian_path<V>(
    edges: &HashMap<V, Vec<V>>,
    start_node: V,
) -> Result<Vec<V>>
where
    V: std::hash::Hash + Eq + Copy + std::fmt::Debug,
{
    // Quick checks to align with your tests and typical Hierholzer assumptions:
    if edges.is_empty() {
        return Err(Error::NoEulerianPath);
    }
    if !edges.contains_key(&start_node) {
        return Err(Error::NoEulerianPath);
    }

    // Work on a mutable clone so we can pop edges as we go.
    let mut graph = edges.clone();
    let mut path = Vec::new();
    let mut stack = vec![start_node];

    // Standard Hierholzer stack-based routine.
    while let Some(&top) = stack.last() {
        if let Some(neighbors) = graph.get_mut(&top) {
            if !neighbors.is_empty() {
                let next = neighbors.pop().unwrap();
                stack.push(next);
            } else {
                path.push(stack.pop().unwrap());
            }
        } else {
            path.push(stack.pop().unwrap());
        }
    }

    // Verify all edges used.
    let mut edge_count = 0;
    for neighbors in edges.values() {
        edge_count += neighbors.len();
    }
    let mut path_edge_count = 0;
    for i in 0..path.len() - 1 {
        if let Some(neighbors) = edges.get(&path[i]) {
            if neighbors.contains(&path[i + 1]) {
                path_edge_count += 1;
            }
        }
    }
    if path_edge_count != edge_count {
        return Err(Error::NoEulerianPath);
    }

    path.reverse();
    Ok(path)
}
