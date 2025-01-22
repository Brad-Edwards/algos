use num_traits::{Float, Zero};
use std::collections::{HashMap, HashSet};
use std::fmt::Debug;
use std::hash::Hash;

use crate::error::{GraphError, Result};
use crate::graph::Graph;

/// Computes a topological ordering of vertices in a directed acyclic graph (DAG).
///
/// A topological ordering is a linear ordering of vertices such that for every directed edge (u,v),
/// vertex u comes before v in the ordering. A topological ordering exists if and only if the graph
/// has no directed cycles (i.e., is a DAG).
///
/// # Arguments
/// * `graph` - The directed graph to sort topologically
///
/// # Returns
/// * `Ok(order)` - A vector of vertices in topological order
/// * `Err(GraphError)` - If the graph is undirected or contains a cycle
///
/// # Examples
/// ```
/// use blocks_cs_graph::{Graph, algorithms::topological_sort};
///
/// let mut graph = Graph::new();
/// graph.add_edge(0, 1, 1.0); // Task 0 must be done before Task 1
/// graph.add_edge(1, 2, 1.0); // Task 1 must be done before Task 2
/// graph.add_edge(0, 2, 1.0); // Task 0 must also be done before Task 2
///
/// let order = topological_sort::sort(&graph).unwrap();
/// assert_eq!(order[0], 0); // Task 0 comes first
/// assert!(order.iter().position(|&x| x == 1) < order.iter().position(|&x| x == 2));
/// ```
///
/// # Complexity
/// * Time: O(V + E) where V is the number of vertices and E is the number of edges
/// * Space: O(V)
pub fn sort<V, W>(graph: &Graph<V, W>) -> Result<Vec<V>>
where
    V: Hash + Eq + Copy + Debug,
    W: Float + Zero + Copy + Debug,
{
    // Validate graph is directed
    if !graph.is_directed() {
        return Err(GraphError::invalid_input(
            "Topological sort requires a directed graph",
        ));
    }

    let mut visited = HashSet::new();
    let mut temp_mark = HashSet::new();
    let mut order = Vec::new();

    // Visit each unvisited vertex
    for &v in graph.vertices() {
        if !visited.contains(&v) {
            visit(v, graph, &mut visited, &mut temp_mark, &mut order)?;
        }
    }

    // Reverse to get correct topological order
    order.reverse();
    Ok(order)
}

/// Helper function for depth-first search traversal.
fn visit<V, W>(
    v: V,
    graph: &Graph<V, W>,
    visited: &mut HashSet<V>,
    temp_mark: &mut HashSet<V>,
    order: &mut Vec<V>,
) -> Result<()>
where
    V: Hash + Eq + Copy + Debug,
    W: Float + Zero + Copy + Debug,
{
    // Check for cycle
    if temp_mark.contains(&v) {
        return Err(GraphError::invalid_input(
            "Graph contains a cycle, topological sort not possible",
        ));
    }

    // Skip if already visited
    if visited.contains(&v) {
        return Ok(());
    }

    // Mark temporarily for cycle detection
    temp_mark.insert(v);

    // Visit all neighbors
    if let Ok(neighbors) = graph.neighbors(&v) {
        for (w, _) in neighbors {
            visit(*w, graph, visited, temp_mark, order)?;
        }
    }

    // Mark as permanently visited and add to order
    temp_mark.remove(&v);
    visited.insert(v);
    order.push(v);

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_topological_sort_simple_path() {
        let mut graph: Graph<i32, f64> = Graph::new();
        graph.add_edge(0, 1, 1.0);
        graph.add_edge(1, 2, 1.0);

        let order = sort(&graph).unwrap();
        assert_eq!(order, vec![0, 1, 2]);
    }

    #[test]
    fn test_topological_sort_dag() {
        let mut graph: Graph<i32, f64> = Graph::new();
        // Create a more complex DAG
        graph.add_edge(0, 1, 1.0);
        graph.add_edge(0, 2, 1.0);
        graph.add_edge(1, 3, 1.0);
        graph.add_edge(2, 3, 1.0);

        let order = sort(&graph).unwrap();
        assert_eq!(order[0], 0);
        assert!(order.iter().position(|&x| x == 1) < order.iter().position(|&x| x == 3));
        assert!(order.iter().position(|&x| x == 2) < order.iter().position(|&x| x == 3));
    }

    #[test]
    fn test_topological_sort_cycle() {
        let mut graph: Graph<i32, f64> = Graph::new();
        graph.add_edge(0, 1, 1.0);
        graph.add_edge(1, 2, 1.0);
        graph.add_edge(2, 0, 1.0); // Creates a cycle

        assert!(matches!(sort(&graph), Err(GraphError::InvalidInput(_))));
    }

    #[test]
    fn test_topological_sort_self_loop() {
        let mut graph: Graph<i32, f64> = Graph::new();
        graph.add_edge(0, 0, 1.0); // Self-loop is a cycle

        assert!(matches!(sort(&graph), Err(GraphError::InvalidInput(_))));
    }

    #[test]
    fn test_topological_sort_undirected_graph() {
        let mut graph: Graph<i32, f64> = Graph::new_undirected();
        graph.add_edge(0, 1, 1.0);

        assert!(matches!(sort(&graph), Err(GraphError::InvalidInput(_))));
    }

    #[test]
    fn test_topological_sort_empty_graph() {
        let graph: Graph<i32, f64> = Graph::new();
        let order = sort(&graph).unwrap();
        assert!(order.is_empty());
    }

    #[test]
    fn test_topological_sort_single_vertex() {
        let mut graph: Graph<i32, f64> = Graph::new();
        graph.add_vertex(0);

        let order = sort(&graph).unwrap();
        assert_eq!(order, vec![0]);
    }

    #[test]
    fn test_topological_sort_disconnected() {
        let mut graph: Graph<i32, f64> = Graph::new();
        graph.add_edge(0, 1, 1.0);
        graph.add_edge(2, 3, 1.0); // Separate component

        let order = sort(&graph).unwrap();
        assert_eq!(order.len(), 4);
        assert!(order.iter().position(|&x| x == 0) < order.iter().position(|&x| x == 1));
        assert!(order.iter().position(|&x| x == 2) < order.iter().position(|&x| x == 3));
    }

    #[test]
    fn test_topological_sort_multiple_paths() {
        let mut graph: Graph<i32, f64> = Graph::new();
        // Multiple paths from 0 to 3:
        // 0->1->3
        // 0->2->3
        graph.add_edge(0, 1, 1.0);
        graph.add_edge(0, 2, 1.0);
        graph.add_edge(1, 3, 1.0);
        graph.add_edge(2, 3, 1.0);

        let order = sort(&graph).unwrap();
        assert_eq!(order[0], 0);
        assert_eq!(order[3], 3);
        // Both 1 and 2 must come after 0 and before 3
        assert!(order.iter().position(|&x| x == 1) > order.iter().position(|&x| x == 0));
        assert!(order.iter().position(|&x| x == 2) > order.iter().position(|&x| x == 0));
        assert!(order.iter().position(|&x| x == 1) < order.iter().position(|&x| x == 3));
        assert!(order.iter().position(|&x| x == 2) < order.iter().position(|&x| x == 3));
    }
}
