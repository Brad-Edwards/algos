use num_traits::{Float, Zero};
use std::collections::{HashMap, HashSet, VecDeque};
use std::fmt::Debug;
use std::hash::Hash;

use crate::error::{GraphError, Result};
use crate::graph::Graph;

/// Computes strongly connected components (SCCs) using Tarjan's algorithm.
///
/// A strongly connected component is a maximal subset of vertices in a directed graph
/// where every vertex is reachable from every other vertex in the subset.
///
/// # Arguments
/// * `graph` - The directed graph to find SCCs in
///
/// # Returns
/// * `Ok(components)` - A vector of vectors, where each inner vector contains the vertices of one SCC
/// * `Err(GraphError)` - If the graph is undirected
///
/// # Examples
/// ```
/// use blocks_cs_graph::{Graph, algorithms::tarjan};
///
/// let mut graph = Graph::new();
/// graph.add_edge(0, 1, 1.0);
/// graph.add_edge(1, 2, 1.0);
/// graph.add_edge(2, 0, 1.0); // Forms a cycle 0->1->2->0
/// graph.add_edge(2, 3, 1.0);
/// graph.add_edge(3, 4, 1.0);
/// graph.add_edge(4, 3, 1.0); // Forms a cycle 3<->4
///
/// let components = tarjan::strongly_connected_components(&graph).unwrap();
/// assert_eq!(components.len(), 2); // Two SCCs: {0,1,2} and {3,4}
/// ```
///
/// # Complexity
/// * Time: O(V + E) where V is the number of vertices and E is the number of edges
/// * Space: O(V)
pub fn strongly_connected_components<V, W>(graph: &Graph<V, W>) -> Result<Vec<Vec<V>>>
where
    V: Hash + Eq + Copy + Debug,
    W: Float + Zero + Copy + Debug,
{
    // Validate graph is directed
    if !graph.is_directed() {
        return Err(GraphError::invalid_input(
            "Tarjan's algorithm requires a directed graph",
        ));
    }

    let mut index = 0;
    let mut stack = VecDeque::new();
    let mut indices = HashMap::new();
    let mut lowlinks = HashMap::new();
    let mut on_stack = HashSet::new();
    let mut components = Vec::new();

    // Process each vertex
    for &v in graph.vertices() {
        if !indices.contains_key(&v) {
            strong_connect(
                v,
                &mut index,
                &mut indices,
                &mut lowlinks,
                &mut stack,
                &mut on_stack,
                &mut components,
                graph,
            )?;
        }
    }

    Ok(components)
}

/// Helper function for Tarjan's algorithm that processes a single vertex.
fn strong_connect<V, W>(
    v: V,
    index: &mut usize,
    indices: &mut HashMap<V, usize>,
    lowlinks: &mut HashMap<V, usize>,
    stack: &mut VecDeque<V>,
    on_stack: &mut HashSet<V>,
    components: &mut Vec<Vec<V>>,
    graph: &Graph<V, W>,
) -> Result<()>
where
    V: Hash + Eq + Copy + Debug,
    W: Float + Zero + Copy + Debug,
{
    // Set depth index for v
    indices.insert(v, *index);
    lowlinks.insert(v, *index);
    *index += 1;
    stack.push_back(v);
    on_stack.insert(v);

    // Consider successors of v
    if let Ok(neighbors) = graph.neighbors(&v) {
        for (w, _) in neighbors {
            if !indices.contains_key(w) {
                // Successor w has not yet been visited; recurse on it
                strong_connect(
                    *w, index, indices, lowlinks, stack, on_stack, components, graph,
                )?;
                lowlinks.insert(v, std::cmp::min(lowlinks[&v], lowlinks[w]));
            } else if on_stack.contains(w) {
                // Successor w is in stack and hence in the current SCC
                lowlinks.insert(v, std::cmp::min(lowlinks[&v], indices[w]));
            }
        }
    }

    // If v is a root node, pop the stack and generate an SCC
    if lowlinks[&v] == indices[&v] {
        let mut component = Vec::new();
        loop {
            let w = stack.pop_back().unwrap();
            on_stack.remove(&w);
            component.push(w);
            if w == v {
                break;
            }
        }
        components.push(component);
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tarjan_simple_cycle() {
        let mut graph: Graph<i32, f64> = Graph::new();
        graph.add_edge(0, 1, 1.0);
        graph.add_edge(1, 2, 1.0);
        graph.add_edge(2, 0, 1.0);

        let components = strongly_connected_components(&graph).unwrap();
        assert_eq!(components.len(), 1);
        assert_eq!(components[0].len(), 3);
    }

    #[test]
    fn test_tarjan_multiple_components() {
        let mut graph: Graph<i32, f64> = Graph::new();
        // First component: 0->1->2->0
        graph.add_edge(0, 1, 1.0);
        graph.add_edge(1, 2, 1.0);
        graph.add_edge(2, 0, 1.0);
        // Second component: 3<->4
        graph.add_edge(2, 3, 1.0);
        graph.add_edge(3, 4, 1.0);
        graph.add_edge(4, 3, 1.0);

        let components = strongly_connected_components(&graph).unwrap();
        assert_eq!(components.len(), 2);
        // Components are returned in reverse topological order
        assert_eq!(components[0].len(), 2); // {3,4}
        assert_eq!(components[1].len(), 3); // {0,1,2}
    }

    #[test]
    fn test_tarjan_single_vertex() {
        let mut graph: Graph<i32, f64> = Graph::new();
        graph.add_vertex(0);

        let components = strongly_connected_components(&graph).unwrap();
        assert_eq!(components.len(), 1);
        assert_eq!(components[0], vec![0]);
    }

    #[test]
    fn test_tarjan_no_cycles() {
        let mut graph: Graph<i32, f64> = Graph::new();
        graph.add_edge(0, 1, 1.0);
        graph.add_edge(1, 2, 1.0);
        graph.add_edge(2, 3, 1.0);

        let components = strongly_connected_components(&graph).unwrap();
        assert_eq!(components.len(), 4);
        for component in components {
            assert_eq!(component.len(), 1);
        }
    }

    #[test]
    fn test_tarjan_self_loop() {
        let mut graph: Graph<i32, f64> = Graph::new();
        graph.add_edge(0, 0, 1.0);
        graph.add_edge(0, 1, 1.0);

        let components = strongly_connected_components(&graph).unwrap();
        assert_eq!(components.len(), 2);
        assert!(components.iter().any(|c| c == &vec![0]));
        assert!(components.iter().any(|c| c == &vec![1]));
    }

    #[test]
    fn test_tarjan_undirected_graph() {
        let mut graph = Graph::new_undirected();
        graph.add_edge(0, 1, 1.0);

        assert!(matches!(
            strongly_connected_components(&graph),
            Err(GraphError::InvalidInput(_))
        ));
    }

    #[test]
    fn test_tarjan_empty_graph() {
        let graph: Graph<i32, f64> = Graph::new();
        let components = strongly_connected_components(&graph).unwrap();
        assert!(components.is_empty());
    }

    #[test]
    fn test_tarjan_complex_graph() {
        let mut graph: Graph<i32, f64> = Graph::new();
        // Component 1: 0->1->2->0
        graph.add_edge(0, 1, 1.0);
        graph.add_edge(1, 2, 1.0);
        graph.add_edge(2, 0, 1.0);
        // Component 2: 3->4->5->3
        graph.add_edge(3, 4, 1.0);
        graph.add_edge(4, 5, 1.0);
        graph.add_edge(5, 3, 1.0);
        // Bridge between components
        graph.add_edge(2, 3, 1.0);

        let components = strongly_connected_components(&graph).unwrap();
        assert_eq!(components.len(), 2);
        assert_eq!(components[0].len(), 3); // {3,4,5}
        assert_eq!(components[1].len(), 3); // {0,1,2}
    }

    #[test]
    fn test_tarjan_disconnected_components() {
        let mut graph: Graph<i32, f64> = Graph::new();
        // First component
        graph.add_edge(0, 1, 1.0);
        graph.add_edge(1, 0, 1.0);
        // Second component (disconnected)
        graph.add_edge(2, 3, 1.0);
        graph.add_edge(3, 2, 1.0);

        let components = strongly_connected_components(&graph).unwrap();
        assert_eq!(components.len(), 2);
        assert_eq!(components[0].len(), 2);
        assert_eq!(components[1].len(), 2);
    }
}
