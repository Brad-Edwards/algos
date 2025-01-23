use num_traits::{Float, Zero};
use std::collections::{HashMap, VecDeque};
use std::fmt::Debug;
use std::hash::Hash;

use crate::cs::error::{Error, Result};
use crate::cs::graph::Graph;

pub fn max_flow<V, W>(graph: &Graph<V, W>, source: &V, sink: &V) -> Result<W>
where
    V: Hash + Eq + Copy + Debug,
    W: Float + Zero + Copy + Debug + PartialOrd,
{
    if !graph.has_vertex(source) {
        return Err(Error::VertexNotFound);
    }
    if !graph.has_vertex(sink) {
        return Err(Error::VertexNotFound);
    }
    if source == sink {
        return Err(Error::InvalidInput(
            "Source and sink cannot be the same vertex".to_string(),
        ));
    }

    let mut residual_graph = create_residual_graph(graph);
    let mut max_flow = W::zero();

    loop {
        if let Some(path) = bfs(&residual_graph, source, sink) {
            let path_flow = find_path_flow(&residual_graph, &path);
            max_flow = max_flow + path_flow;
            update_residual_graph(&mut residual_graph, &path, path_flow);
        } else {
            break; // No augmenting path found
        }
    }

    Ok(max_flow)
}

fn create_residual_graph<V, W>(graph: &Graph<V, W>) -> Graph<V, W>
where
    V: Hash + Eq + Copy + Debug,
    W: Float + Zero + Copy + Debug,
{
    let mut residual_graph = Graph::new();
    for vertex in graph.vertices() {
        residual_graph.add_vertex(*vertex);
    }
    for (from, to, weight) in graph.edges() {
        residual_graph.add_edge(*from, *to, weight);
        residual_graph.add_edge(*to, *from, W::zero()); // Backwards edge with zero capacity
    }
    residual_graph
}

fn bfs<V, W>(graph: &Graph<V, W>, source: &V, sink: &V) -> Option<Vec<V>>
where
    V: Hash + Eq + Copy + Debug,
    W: Float + Zero + Copy + Debug + PartialOrd,
{
    let epsilon = W::from(1e-9).unwrap();
    let mut queue = VecDeque::new();
    queue.push_back(*source);
    let mut visited = HashMap::new();
    visited.insert(*source, None); // Start node has no parent

    while let Some(current_vertex) = queue.pop_front() {
        if &current_vertex == sink {
            break; // Path found
        }
        if let Ok(neighbors) = graph.neighbors(&current_vertex) {
            for (neighbor, weight) in neighbors {
                if !visited.contains_key(neighbor) && weight > epsilon {
                    visited.insert(*neighbor, Some(current_vertex));
                    queue.push_back(*neighbor);
                }
            }
        }
    }

    if visited.contains_key(sink) {
        // Reconstruct path from sink to source
        let mut path = vec![*sink];
        let mut current = *sink;
        while let Some(&Some(parent)) = visited.get(&current) {
            path.push(parent);
            current = parent;
            if current == *source {
                break;
            }
        }
        path.reverse(); // Reverse to get path from source to sink
        Some(path)
    } else {
        None // No path found
    }
}

fn find_path_flow<V, W>(graph: &Graph<V, W>, path: &[V]) -> W
where
    V: Hash + Eq + Copy + Debug,
    W: Float + Zero + Copy + Debug + PartialOrd,
{
    let epsilon = W::from(1e-9).unwrap();
    let mut path_flow: Option<W> = None;
    for i in 0..(path.len() - 1) {
        let from = &path[i];
        let to = &path[i + 1];
        if let Some(weight) = graph.edge_weight(from, to) {
            if weight > epsilon {
                path_flow = Some(match path_flow {
                    Some(current_flow) => current_flow.min(weight),
                    None => weight,
                });
            }
        }
    }
    path_flow.unwrap_or(W::zero())
}

fn update_residual_graph<V, W>(graph: &mut Graph<V, W>, path: &[V], flow: W)
where
    V: Hash + Eq + Copy + Debug,
    W: Float + Zero + Copy + Debug + PartialOrd,
{
    let epsilon = W::from(1e-9).unwrap();
    for i in 0..(path.len() - 1) {
        let from = &path[i];
        let to = &path[i + 1];

        // Update forward edge
        let current_capacity = graph.edge_weight(from, to).unwrap();
        let new_capacity = current_capacity - flow;
        // Always update the edge, even if capacity is zero
        graph.add_edge(
            *from,
            *to,
            if new_capacity < epsilon {
                W::zero()
            } else {
                new_capacity
            },
        );

        // Update backward edge
        let back_capacity = graph.edge_weight(to, from).unwrap_or(W::zero());
        let new_back_capacity = back_capacity + flow;
        // Always update the back edge with the new capacity
        graph.add_edge(*to, *from, new_back_capacity);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cs::error::Error;

    #[test]
    fn test_ford_fulkerson_simple() {
        let mut graph: Graph<char, f64> = Graph::new();
        graph.add_vertex('A');
        graph.add_vertex('B');
        graph.add_vertex('C');
        graph.add_vertex('D');
        graph.add_vertex('E');
        graph.add_vertex('F');

        graph.add_edge('A', 'B', 10.0);
        graph.add_edge('A', 'C', 10.0);
        graph.add_edge('B', 'C', 2.0);
        graph.add_edge('B', 'D', 8.0);
        graph.add_edge('C', 'E', 9.0);
        graph.add_edge('D', 'F', 10.0);
        graph.add_edge('E', 'D', 4.0);
        graph.add_edge('E', 'F', 10.0);

        let max_flow_val = max_flow(&graph, &'A', &'F').unwrap();
        assert_eq!(max_flow_val, 19.0);
    }

    #[test]
    fn test_ford_fulkerson_disconnected() {
        let mut graph: Graph<char, f64> = Graph::new();
        graph.add_vertex('A');
        graph.add_vertex('B');
        graph.add_vertex('C');
        graph.add_vertex('D');

        graph.add_edge('A', 'B', 10.0);
        graph.add_edge('C', 'D', 10.0);

        let max_flow_val = max_flow(&graph, &'A', &'B').unwrap();
        assert_eq!(max_flow_val, 10.0);

        let max_flow_val_ac = max_flow(&graph, &'A', &'C').unwrap();
        assert_eq!(max_flow_val_ac, 0.0); // No path from A to C
    }

    #[test]
    fn test_ford_fulkerson_no_path() {
        let mut graph: Graph<char, f64> = Graph::new();
        graph.add_vertex('A');
        graph.add_vertex('B');
        graph.add_vertex('C');

        graph.add_edge('A', 'B', 10.0);
        graph.add_edge('B', 'C', 10.0);

        let max_flow_val = max_flow(&graph, &'C', &'A').unwrap();
        assert_eq!(max_flow_val, 0.0); // No path from C to A
    }

    #[test]
    fn test_ford_fulkerson_source_sink_same() {
        let mut graph: Graph<char, f64> = Graph::new();
        graph.add_vertex('A');
        let result = max_flow(&graph, &'A', &'A');
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), Error::InvalidInput(_)));
    }

    #[test]
    fn test_ford_fulkerson_vertex_not_found() {
        let graph: Graph<char, f64> = Graph::new();
        let result_source = max_flow(&graph, &'A', &'B');
        let result_sink = max_flow(&graph, &'C', &'A');

        assert!(result_source.is_err());
        assert!(matches!(result_source.unwrap_err(), Error::VertexNotFound));
        assert!(result_sink.is_err());
        assert!(matches!(result_sink.unwrap_err(), Error::VertexNotFound));
    }

    #[test]
    fn test_ford_fulkerson_complex_graph() {
        let mut graph = Graph::new();
        graph.add_vertex('S');
        graph.add_vertex('A');
        graph.add_vertex('B');
        graph.add_vertex('C');
        graph.add_vertex('D');
        graph.add_vertex('T');

        graph.add_edge('S', 'A', 10.0);
        graph.add_edge('S', 'B', 5.0);
        graph.add_edge('A', 'C', 10.0);
        graph.add_edge('A', 'D', 5.0);
        graph.add_edge('B', 'C', 5.0);
        graph.add_edge('B', 'D', 10.0);
        graph.add_edge('C', 'T', 15.0);
        graph.add_edge('D', 'T', 10.0);

        let max_flow_val = max_flow(&graph, &'S', &'T').unwrap();
        assert_eq!(max_flow_val, 20.0);
    }

    #[test]
    fn test_ford_fulkerson_residual_graph_updates() {
        let mut graph = Graph::new();
        graph.add_vertex('S');
        graph.add_vertex('T');
        graph.add_edge('S', 'T', 10.0);

        let mut residual_graph = create_residual_graph(&graph);
        assert_eq!(residual_graph.edge_weight(&'S', &'T'), Some(10.0));
        assert_eq!(residual_graph.edge_weight(&'T', &'S'), Some(0.0));

        let path = bfs(&residual_graph, &'S', &'T').unwrap();
        let path_flow = find_path_flow(&residual_graph, &path);
        update_residual_graph(&mut residual_graph, &path, path_flow);

        assert_eq!(residual_graph.edge_weight(&'T', &'S'), Some(10.0)); // Back edge capacity increased
        assert_eq!(residual_graph.edge_weight(&'S', &'T'), Some(0.0)); // Forward edge has zero capacity
    }
}
