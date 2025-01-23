use num_traits::{Float, Zero};
use std::collections::{HashMap, HashSet};
use std::fmt::Debug;
use std::hash::Hash;

use crate::cs::error::{Error, Result};
use crate::cs::graph::Graph;

pub fn find_cycles<V, W>(graph: &Graph<V, W>) -> Result<Vec<Vec<V>>>
where
    V: Hash + Eq + Copy + Debug + Clone,
    W: Float + Zero + Copy + Debug + PartialOrd,
{
    let mut cycles = Vec::new();
    for start_node in graph.vertices() {
        let mut visited = HashSet::new();
        let mut stack = Vec::new();
        find_cycles_recursive(
            graph,
            *start_node,
            *start_node,
            &mut visited,
            &mut stack,
            &mut cycles,
        )?;
    }
    Ok(cycles)
}

fn find_cycles_recursive<V, W>(
    graph: &Graph<V, W>,
    current_node: V,
    start_node: V,
    visited: &mut HashSet<V>,
    stack: &mut Vec<V>,
    cycles: &mut Vec<Vec<V>>,
) -> Result<()>
where
    V: Hash + Eq + Copy + Debug + Clone,
    W: Float + Zero + Copy + Debug + PartialOrd,
{
    visited.insert(current_node);
    stack.push(current_node);

    if let Ok(neighbors) = graph.neighbors(&current_node) {
        for (neighbor, _) in neighbors {
            if *neighbor == start_node {
                // Cycle detected
                let mut cycle = stack.clone();
                cycles.push(cycle);
            } else if !visited.contains(neighbor) {
                find_cycles_recursive(graph, *neighbor, start_node, visited, stack, cycles)?;
            }
        }
    }

    stack.pop();
    visited.remove(&current_node);
    Ok(())
}
