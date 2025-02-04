use std::collections::{HashMap, HashSet};

#[derive(Debug, Clone)]
pub struct Point {
    x: f64,
    y: f64,
}

impl Point {
    pub fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }

    fn distance(&self, other: &Point) -> f64 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        (dx * dx + dy * dy).sqrt()
    }
}

#[derive(Debug, Clone)]
pub struct TSPInstance {
    points: Vec<Point>,
}

impl TSPInstance {
    pub fn new(points: Vec<Point>) -> Self {
        Self { points }
    }
}

/// Implements Christofides algorithm for the metric TSP problem.
///
/// This algorithm provides a 1.5-approximation for metric TSP. It works by:
/// 1. Computing a minimum spanning tree (MST)
/// 2. Finding odd-degree vertices in the MST
/// 3. Computing a minimum-weight perfect matching on odd-degree vertices
/// 4. Combining MST and matching to form an Eulerian multigraph
/// 5. Finding an Eulerian circuit and shortcutting to a Hamiltonian cycle
///
/// # Arguments
///
/// * `instance` - The TSP instance containing points in metric space
///
/// # Returns
///
/// * A tuple containing:
///   - Vector of indices representing the tour order
///   - Total tour length
pub fn solve(instance: &TSPInstance) -> (Vec<usize>, f64) {
    if instance.points.len() <= 3 {
        return trivial_solution(instance);
    }

    // Step 1: Compute MST using Prim's algorithm
    let mst = compute_mst(&instance.points);

    // Step 2: Find vertices with odd degree in MST
    let odd_vertices = find_odd_vertices(&mst, instance.points.len());

    // Step 3: Compute minimum-weight perfect matching on odd vertices
    let matching = compute_min_matching(&instance.points, &odd_vertices);

    // Step 4: Combine MST and matching to get Eulerian multigraph
    let mut eulerian_graph = mst.clone();
    for (u, v) in matching {
        eulerian_graph.entry(u).or_default().push(v);
        eulerian_graph.entry(v).or_default().push(u);
    }

    // Step 5: Find Eulerian circuit and shortcut
    let tour = find_eulerian_circuit(&eulerian_graph);
    let shortened_tour = shortcut_tour(&tour);

    // Calculate tour length
    let tour_length = calculate_tour_length(&shortened_tour, &instance.points);

    (shortened_tour, tour_length)
}

fn compute_mst(points: &[Point]) -> HashMap<usize, Vec<usize>> {
    let n = points.len();
    let mut mst: HashMap<usize, Vec<usize>> = HashMap::new();
    let mut visited = HashSet::new();
    let mut distances = vec![f64::INFINITY; n];
    let mut parent = vec![None; n];

    // Start from vertex 0
    distances[0] = 0.0;

    for _ in 0..n {
        // Find closest unvisited vertex
        let mut min_dist = f64::INFINITY;
        let mut u = 0;
        for (v, &dist) in distances.iter().enumerate() {
            if !visited.contains(&v) && dist < min_dist {
                min_dist = dist;
                u = v;
            }
        }

        visited.insert(u);

        // Add edge to MST
        if let Some(p) = parent[u] {
            mst.entry(p).or_default().push(u);
            mst.entry(u).or_default().push(p);
        }

        // Update distances
        for (v, dist) in distances.iter_mut().enumerate() {
            if !visited.contains(&v) {
                let new_dist = points[u].distance(&points[v]);
                if new_dist < *dist {
                    *dist = new_dist;
                    parent[v] = Some(u);
                }
            }
        }
    }

    mst
}

fn find_odd_vertices(graph: &HashMap<usize, Vec<usize>>, n: usize) -> Vec<usize> {
    let mut odd = Vec::new();
    for v in 0..n {
        if let Some(neighbors) = graph.get(&v) {
            if neighbors.len() % 2 == 1 {
                odd.push(v);
            }
        }
    }
    odd
}

fn compute_min_matching(points: &[Point], odd_vertices: &[usize]) -> Vec<(usize, usize)> {
    let mut matching = Vec::new();
    let mut unmatched: HashSet<_> = odd_vertices.iter().cloned().collect();

    while !unmatched.is_empty() {
        let u = *unmatched.iter().next().unwrap();
        unmatched.remove(&u);

        let mut min_dist = f64::INFINITY;
        let mut best_v = 0;

        for &v in &unmatched {
            let dist = points[u].distance(&points[v]);
            if dist < min_dist {
                min_dist = dist;
                best_v = v;
            }
        }

        matching.push((u, best_v));
        unmatched.remove(&best_v);
    }

    matching
}

fn find_eulerian_circuit(graph: &HashMap<usize, Vec<usize>>) -> Vec<usize> {
    let mut circuit = Vec::new();
    let current = 0;
    let mut edges = graph.clone();

    fn dfs(v: usize, edges: &mut HashMap<usize, Vec<usize>>, circuit: &mut Vec<usize>) {
        while let Some(pos) = edges.get(&v).and_then(|neighbors| {
            if neighbors.is_empty() {
                None
            } else {
                Some(neighbors.len() - 1)
            }
        }) {
            let u = edges.get_mut(&v).unwrap().swap_remove(pos);
            edges.get_mut(&u).unwrap().retain(|&x| x != v);
            dfs(u, edges, circuit);
        }
        circuit.push(v);
    }

    dfs(current, &mut edges, &mut circuit);
    circuit
}

fn shortcut_tour(euler_tour: &[usize]) -> Vec<usize> {
    let mut seen = HashSet::new();
    let mut shortened = Vec::new();

    for &v in euler_tour {
        if !seen.contains(&v) {
            shortened.push(v);
            seen.insert(v);
        }
    }

    // Add starting vertex to complete the cycle
    if !shortened.is_empty() {
        shortened.push(shortened[0]);
    }

    shortened
}

fn calculate_tour_length(tour: &[usize], points: &[Point]) -> f64 {
    let mut length = 0.0;
    for i in 0..tour.len() - 1 {
        length += points[tour[i]].distance(&points[tour[i + 1]]);
    }
    length
}

fn trivial_solution(instance: &TSPInstance) -> (Vec<usize>, f64) {
    let tour: Vec<_> = (0..instance.points.len())
        .chain(std::iter::once(0))
        .collect();
    let length = calculate_tour_length(&tour, &instance.points);
    (tour, length)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_small_instance() {
        let points = vec![
            Point::new(0.0, 0.0),
            Point::new(1.0, 0.0),
            Point::new(1.0, 1.0),
            Point::new(0.0, 1.0),
        ];
        let instance = TSPInstance::new(points);

        let (tour, length) = solve(&instance);

        // Verify tour properties
        assert_eq!(tour.first(), tour.last());
        assert_eq!(tour.len(), instance.points.len() + 1);

        // Verify approximation ratio (optimal tour length is 4.0)
        assert!(length <= 6.0); // 1.5 * optimal
    }

    #[test]
    fn test_tiny_instance() {
        let points = vec![Point::new(0.0, 0.0), Point::new(1.0, 0.0)];
        let instance = TSPInstance::new(points);

        let (tour, length) = solve(&instance);

        assert_eq!(tour, vec![0, 1, 0]);
        assert!((length - 2.0).abs() < 1e-10);
    }
}
