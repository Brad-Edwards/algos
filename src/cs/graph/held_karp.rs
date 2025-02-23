/// Held-Karp implementation for the Traveling Salesman Problem (TSP) using Dynamic Programming.
/// Assumes a complete graph with `dist[i][j]` giving the cost from node i to node j.
/// Returns the minimal cost and one optimal route (starting and ending at node 0).
///
/// # Example
/// ```
/// use algos::cs::graph::held_karp::held_karp;
///
/// // A small 4-node complete graph (0->1=10, 0->2=15, 0->3=20, etc.).
/// // dist[i][j] is the cost of traveling from i to j.
/// let dist = vec![
///     vec![0, 10, 15, 20],
///     vec![10, 0, 35, 25],
///     vec![15, 35, 0, 30],
///     vec![20, 25, 30, 0],
/// ];
///
/// let (cost, path) = held_karp(&dist);
/// // This returns (80, [0, 1, 3, 2, 0]) or another path with the same cost.
/// assert_eq!(cost, 80);
/// assert_eq!(path.len(), 5);
/// assert_eq!(path[0], 0);
/// assert_eq!(path[4], 0);
/// ```

/// Held-Karp TSP solver.
/// Returns (minimum_cost, path_including_start_end).
pub fn held_karp(dist: &[Vec<u64>]) -> (u64, Vec<usize>) {
    let n = dist.len();
    if n == 0 {
        return (0, vec![]);
    }
    if n == 1 {
        return (0, vec![0, 0]);
    }

    // Use a large 'infinity' to avoid overflow.
    let inf = u64::MAX / 4;
    // dp[mask][j] = minimal cost to reach subset 'mask' ending at 'j'.
    let mut dp = vec![vec![inf; n]; 1 << n];
    // parent[mask][j] = predecessor of 'j' on the optimal path for dp[mask][j].
    let mut parent = vec![vec![usize::MAX; n]; 1 << n];

    // Base case: direct paths from node 0
    for j in 1..n {
        dp[1 << j][j] = dist[0][j];
        parent[1 << j][j] = 0;
    }

    // For each subset size
    for size in 2..n {
        // For each subset of size 'size'
        for mask in 0..(1 << n) {
            if mask & 1 != 0 {
                // Skip if subset includes node 0
                continue;
            }

            // Count set bits
            let mut bits = 0;
            let mut m = mask;
            while m > 0 {
                bits += m & 1;
                m >>= 1;
            }
            if bits != size {
                continue;
            }

            // For each possible last node in this subset
            for j in 1..n {
                if (mask & (1 << j)) == 0 {
                    continue;
                }

                // Try all possible second-to-last nodes
                let prev_mask = mask & !(1 << j);
                for i in 1..n {
                    if i == j || (mask & (1 << i)) == 0 {
                        continue;
                    }

                    let new_cost = dp[prev_mask][i].saturating_add(dist[i][j]);
                    if new_cost < dp[mask][j] {
                        dp[mask][j] = new_cost;
                        parent[mask][j] = i;
                    }
                }
            }
        }
    }

    // Handle the final subset that includes all nodes except 0
    let all_but_zero = ((1 << n) - 1) & !(1);
    for j in 1..n {
        let prev_mask = all_but_zero & !(1 << j);
        for i in 1..n {
            if i == j {
                continue;
            }
            let new_cost = dp[prev_mask][i].saturating_add(dist[i][j]);
            if new_cost < dp[all_but_zero][j] {
                dp[all_but_zero][j] = new_cost;
                parent[all_but_zero][j] = i;
            }
        }
    }

    // Find the optimal cost and last node
    let mut best_cost = inf;
    let mut best_end = 0;
    for j in 1..n {
        let final_cost = dp[all_but_zero][j];
        if final_cost != inf {
            let tour_cost = final_cost.saturating_add(dist[j][0]);
            if tour_cost < best_cost {
                best_cost = tour_cost;
                best_end = j;
            }
        }
    }

    // If no valid tour was found
    if best_cost == inf {
        return (0, vec![]);
    }

    // Reconstruct the path in reverse, then fix its order
    let mut path = Vec::new();
    let mut curr = best_end;
    let mut mask = all_but_zero;

    // Walk backward until we reach 0
    while curr != usize::MAX {
        path.push(curr);
        if curr == 0 {
            break;
        }
        let p = parent[mask][curr];
        mask &= !(1 << curr);
        curr = p;
    }
    // Now path is [end, ..., 0]. Reverse it:
    path.reverse();
    // Finally, append 0 again to complete the cycle
    path.push(0);

    (best_cost, path)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_small_tsp() {
        let dist = vec![
            vec![0, 10, 15, 20],
            vec![10, 0, 35, 25],
            vec![15, 35, 0, 30],
            vec![20, 25, 30, 0],
        ];
        let (cost, path) = held_karp(&dist);
        // Minimal tour cost is 80 for this matrix, e.g. 0->1->3->2->0
        assert_eq!(cost, 80);
        assert_eq!(path.len(), 5);
        assert_eq!(path[0], 0);
        assert_eq!(path[path.len() - 1], 0);
    }

    #[test]
    fn test_single_node() {
        let dist = vec![vec![0]];
        let (cost, path) = held_karp(&dist);
        // Only one node, cost is zero, path is [0, 0].
        assert_eq!(cost, 0);
        assert_eq!(path, vec![0, 0]);
    }

    #[test]
    fn test_empty() {
        let dist: Vec<Vec<u64>> = vec![];
        let (cost, path) = held_karp(&dist);
        assert_eq!(cost, 0);
        assert_eq!(path.len(), 0);
    }

    #[test]
    fn test_three_nodes() {
        let dist = vec![vec![0, 10, 15], vec![10, 0, 20], vec![15, 20, 0]];
        let (cost, path) = held_karp(&dist);
        assert_eq!(cost, 45);
        assert_eq!(path.len(), 4);
        assert_eq!(path[0], 0);
        assert_eq!(path[path.len() - 1], 0);
    }

    #[test]
    fn test_asymmetric_costs() {
        let dist = vec![vec![0, 10, 15], vec![20, 0, 25], vec![30, 35, 0]];
        let (cost, path) = held_karp(&dist);
        assert_eq!(cost, 65); // 0->1->2->0 = 10 + 25 + 30
        let expected_path = vec![0, 1, 2, 0];
        assert_eq!(path.len(), expected_path.len());
        assert_eq!(path[0], 0);
        assert_eq!(path[path.len() - 1], 0);

        // Calculate the actual cost of the path
        let mut actual_cost = 0;
        for i in 0..path.len() - 1 {
            actual_cost += dist[path[i]][path[i + 1]];
        }
        assert_eq!(actual_cost, 65);
    }
}
