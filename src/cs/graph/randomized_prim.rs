use rand::thread_rng;
use rand::Rng;

pub type Edge = (usize, usize, f64);

/// Implements a randomized Prim's algorithm for computing a minimum spanning tree.
/// The graph is represented as an adjacency list: Vec<Vec<(usize, f64)>>, where each entry is (neighbor, weight).
pub fn randomized_prim(num_vertices: usize, graph: &[Vec<(usize, f64)>]) -> (Vec<Edge>, f64) {
    if num_vertices == 0 {
        return (Vec::new(), 0.0);
    }
    let mut rng = thread_rng();
    let start = rng.gen_range(0..num_vertices);
    let mut in_tree = vec![false; num_vertices];
    in_tree[start] = true;
    let mut tree_edges = Vec::new();
    let mut total_weight = 0.0;
    let mut candidate_edges: Vec<Edge> = graph[start].iter().map(|&(v, w)| (start, v, w)).collect();

    while tree_edges.len() < num_vertices - 1 {
        if candidate_edges.is_empty() {
            break;
        }
        let min_index = candidate_edges
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.2.partial_cmp(&b.2).unwrap())
            .map(|(i, _)| i)
            .unwrap();
        let edge = candidate_edges.remove(min_index);
        let (_, v, w) = edge;
        if in_tree[v] {
            continue;
        }
        in_tree[v] = true;
        tree_edges.push(edge);
        total_weight += w;
        for &(nbr, weight) in &graph[v] {
            if !in_tree[nbr] {
                candidate_edges.push((v, nbr, weight));
            }
        }
    }
    (tree_edges, total_weight)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_randomized_prim() {
        let num_vertices = 4;
        let mut graph = vec![Vec::new(); num_vertices];
        // Construct a simple graph.
        graph[0].push((1, 1.0));
        graph[0].push((2, 4.0));
        graph[1].push((0, 1.0));
        graph[1].push((2, 2.0));
        graph[1].push((3, 5.0));
        graph[2].push((0, 4.0));
        graph[2].push((1, 2.0));
        graph[2].push((3, 3.0));
        graph[3].push((1, 5.0));
        graph[3].push((2, 3.0));
        let (mst_edges, total_weight) = randomized_prim(num_vertices, &graph);
        assert_eq!(mst_edges.len(), num_vertices - 1);
        assert!((total_weight - 6.0).abs() < 1e-6);
    }
}
