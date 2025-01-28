//! lib.rs

/// Computes the minimum cost to merge consecutive segments (files) with sizes given in `weights`
/// using Knuth Optimization for an O(n^2) solution.
///
/// # Arguments
///
/// - `weights`: a slice of non-negative sizes (e.g., file sizes).
///
/// # Returns
///
/// - `dp[0][n-1]`: the minimal total cost to merge all segments into one.
/// - an optional 2D array `opt` that records the partition points (the "k" values).
///   You can use `reconstruct_optimal_merge` (below) to build an actual merge order
///   if you desire, though many applications only need the minimal cost.
///
/// # Examples
///
/// ```
/// use algos::cs::dynamic::knuth_optimization::min_merge_cost_knuth;
///
/// let weights = vec![10, 20, 30];
/// let (cost, _) = min_merge_cost_knuth(&weights);
/// assert_eq!(cost, 90);
/// // Explanation:
/// // - Merge (10, 20) cost=30, new array [30, 30]
/// // - Merge (30, 30) cost=60
/// // total=90
/// ```
pub fn min_merge_cost_knuth(weights: &[usize]) -> (usize, Vec<Vec<usize>>) {
    let n = weights.len();
    if n == 0 {
        return (0, Vec::new());
    }
    if n == 1 {
        return (0, vec![vec![0; 1]; 1]); // no cost to "merge" a single item
    }

    // Prefix sums for quick subarray cost calculation: prefix_sum[i] = sum of weights[..i].
    // So sum of weights[i..j] = prefix_sum[j] - prefix_sum[i].
    let mut prefix_sum = vec![0; n + 1];
    for i in 0..n {
        prefix_sum[i + 1] = prefix_sum[i] + weights[i];
    }

    // dp[i][j] = minimal cost to merge subarray i..j.
    let mut dp = vec![vec![0_usize; n]; n];
    // opt[i][j] = the best "k" that achieves the min cost for dp[i][j].
    // We'll fill it in to enable Knuth's bounding technique.
    let mut opt = vec![vec![0_usize; n]; n];

    // Base case: dp[i][i] = 0 (no cost to have a single element).
    for i in 0..n {
        dp[i][i] = 0;
        opt[i][i] = i;
    }

    // cost function
    let cost = |i: usize, j: usize| prefix_sum[j + 1] - prefix_sum[i];

    // For subarray length from 2 to n
    for length in 2..=n {
        for i in 0..=n - length {
            let j = i + length - 1;

            // We only need to check k from opt[i][j-1]..=opt[i+1][j]
            // Because of Knuth's monotonic queue bounding.
            // But we must ensure these indices are within [i..j-1].
            let start_k = opt[i][j.saturating_sub(1)].max(i);
            let end_k = opt
                .get(i + 1)
                .map(|row| row.get(j).cloned().unwrap_or(j.saturating_sub(1)))
                .unwrap_or(j.saturating_sub(1))
                .min(j.saturating_sub(1));

            let mut min_val = usize::MAX;
            let mut best_k = start_k;
            for k in start_k..=end_k {
                let val = dp[i][k] + dp[k + 1][j];
                if val < min_val {
                    min_val = val;
                    best_k = k;
                }
            }
            dp[i][j] = min_val + cost(i, j);
            opt[i][j] = best_k;
        }
    }

    (dp[0][n - 1], opt)
}

/// Reconstructs one optimal merge sequence using the `opt` table from `min_merge_cost_knuth`.
///
/// Returns a list of merges in the form `(start, mid, end)` meaning "merge subarray `[start..=mid]` with `[mid+1..=end]`".
/// This is one way to represent the merge strategy, but in practice you might
/// only need the minimal cost.
///
/// # Arguments
/// - `opt`: the 2D table from `min_merge_cost_knuth`
/// - `i`: start index of the subarray
/// - `j`: end index of the subarray
///
/// # Examples
///
/// ```
/// use algos::cs::dynamic::knuth_optimization::{min_merge_cost_knuth, reconstruct_optimal_merge};
///
/// let weights = vec![10, 20, 30];
/// let (_, s) = min_merge_cost_knuth(&weights);
/// let merges = reconstruct_optimal_merge(&s, 0, weights.len() - 1);
/// assert_eq!(merges.len(), 2); // Two merge operations needed
/// ```
pub fn reconstruct_optimal_merge(
    opt: &Vec<Vec<usize>>,
    i: usize,
    j: usize,
) -> Vec<(usize, usize, usize)> {
    let mut result = Vec::new();
    reconstruct_optimal_merge_rec(opt, i, j, &mut result);
    result
}

fn reconstruct_optimal_merge_rec(
    opt: &Vec<Vec<usize>>,
    i: usize,
    j: usize,
    merges: &mut Vec<(usize, usize, usize)>,
) {
    if i >= j {
        return;
    }
    let k = opt[i][j];
    // Record the merge of [i..=k] with [k+1..=j]
    merges.push((i, k, j));
    reconstruct_optimal_merge_rec(opt, i, k, merges);
    reconstruct_optimal_merge_rec(opt, k + 1, j, merges);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_single_file() {
        let weights = vec![10];
        let (cost, _opt) = min_merge_cost_knuth(&weights);
        assert_eq!(cost, 0, "Merging 1 file is zero cost");
    }

    #[test]
    fn test_two_files() {
        let weights = vec![10, 20];
        let (cost, _opt) = min_merge_cost_knuth(&weights);
        // Merging [10, 20] => cost=10+20=30
        assert_eq!(cost, 30);
    }

    #[test]
    fn test_three_files() {
        // Common example: [10, 20, 30]
        // (10 + 20)=30 => cost=30
        // merges to [30, 30], final merge cost=60 => total=90
        let weights = vec![10, 20, 30];
        let (cost, opt) = min_merge_cost_knuth(&weights);
        assert_eq!(cost, 90);

        let merges = reconstruct_optimal_merge(&opt, 0, weights.len() - 1);
        // merges might be something like:
        //    (0,0,1) => merges subarray [0..=0] with [1..=1]
        //    (0,1,2) => merges subarray [0..=1] with [2..=2]
        // The order can vary depending on how ties are broken.
        assert_eq!(merges.len(), 2);
    }

    #[test]
    fn test_four_files() {
        // Example: [1,2,3,4]
        // There's more than one merge order but let's just check the final cost is correct.
        // Known minimal cost is 19:
        // e.g., merge(1,2)=3 -> cost=3, new array [3,3,4], merge(3,3)=6 -> cost=6, new array [6,4], final merge=10 => total=3+6+10=19
        let weights = vec![1, 2, 3, 4];
        let (cost, _opt) = min_merge_cost_knuth(&weights);
        assert_eq!(cost, 19);
    }
}
