/// lib.rs

/// Represents a single weighted interval in scheduling.
#[derive(Debug, Clone)]
pub struct WeightedInterval {
    pub start: usize,
    pub end: usize,
    pub weight: usize,
}

impl WeightedInterval {
    /// Creates a new `WeightedInterval`.
    ///
    /// # Panics
    ///
    /// Panics if `start > end`.
    pub fn new(start: usize, end: usize, weight: usize) -> Self {
        assert!(start <= end, "start cannot be greater than end");
        Self { start, end, weight }
    }
}

/// Computes the maximum total weight of a set of non-overlapping intervals.
///
/// # Examples
///
/// ```
/// use winterval::WeightedInterval;
/// use winterval::max_weighted_schedule;
///
/// let intervals = vec![
///     WeightedInterval::new(0, 3, 4),
///     WeightedInterval::new(1, 5, 2),
///     WeightedInterval::new(4, 6, 5),
///     WeightedInterval::new(5, 9, 6),
/// ];
///
/// // One optimal schedule is intervals 0 and 3 => weight = 4 + 6 = 10
/// assert_eq!(max_weighted_schedule(&intervals), 10);
/// ```
pub fn max_weighted_schedule(intervals: &[WeightedInterval]) -> usize {
    if intervals.is_empty() {
        return 0;
    }

    // Sort intervals by their end time (ascending).
    let mut sorted = intervals.to_vec();
    sorted.sort_by_key(|iv| iv.end);

    // Precompute p(i): the index of the rightmost interval j < i that doesn't overlap i.
    let p = compute_predecessors(&sorted);

    // dp[i] = maximum weight of scheduling intervals among the first i
    // In code, dp[i] will represent the result for sorted[..i], so it's 1-based indexing.
    let n = sorted.len();
    let mut dp = vec![0_usize; n + 1];

    for i in 1..=n {
        let weight_i = sorted[i - 1].weight;
        let p_i = p[i - 1];
        dp[i] = dp[i - 1].max(weight_i + dp[(p_i as usize) + 1]);
    }

    dp[n]
}

/// Reconstructs an actual optimal schedule for Weighted Interval Scheduling.
///
/// Returns a vector of intervals (in ascending order by end time) that yields
/// the maximum total weight. If multiple schedules have the same weight, only
/// one is returned.
///
/// # Examples
///
/// ```
/// use winterval::{WeightedInterval, best_weighted_schedule};
///
/// let intervals = vec![
///     WeightedInterval::new(0, 3, 4),
///     WeightedInterval::new(1, 5, 2),
///     WeightedInterval::new(4, 6, 5),
///     WeightedInterval::new(5, 9, 6),
/// ];
///
/// // One optimal solution is intervals 0 and 3 => total weight 10
/// let best_set = best_weighted_schedule(&intervals);
/// let total_weight: usize = best_set.iter().map(|iv| iv.weight).sum();
/// assert_eq!(total_weight, 10);
/// ```
pub fn best_weighted_schedule(intervals: &[WeightedInterval]) -> Vec<WeightedInterval> {
    if intervals.is_empty() {
        return Vec::new();
    }

    let mut sorted = intervals.to_vec();
    sorted.sort_by_key(|iv| iv.end);

    // Compute predecessor table
    let p = compute_predecessors(&sorted);
    let n = sorted.len();

    // dp[i] will store the maximum weight among intervals[0..i]
    let mut dp = vec![0_usize; n + 1];

    // We'll store decisions for reconstruction:
    // chosen[i] = true if interval i is included in the optimal schedule
    // (with i in 0-based index for sorted)
    let mut chosen = vec![false; n];

    for i in 1..=n {
        let weight_i = sorted[i - 1].weight;
        let p_i = p[i - 1];
        let with_current = weight_i + dp[(p_i as usize) + 1];
        let without_current = dp[i - 1];

        if with_current > without_current {
            dp[i] = with_current;
            chosen[i - 1] = true;
        } else {
            dp[i] = without_current;
        }
    }

    // Reconstruct intervals by backtracking
    let mut result = Vec::new();
    let mut i = n;
    while i > 0 {
        if chosen[i - 1] {
            result.push(sorted[i - 1].clone());
            i = (p[i - 1] as usize) + 1;
        } else {
            i -= 1;
        }
    }

    result.reverse();
    result
}

/// Computes the predecessor array p where p[i] is the index of the rightmost interval
/// that does not overlap `intervals[i]`, or -1 if none exists.
///
/// Precondition: `intervals` must be sorted by their end time.
fn compute_predecessors(intervals: &[WeightedInterval]) -> Vec<isize> {
    let n = intervals.len();
    let mut p = vec![-1; n];

    // For each interval i, we binary search among intervals[0..i] to find
    // the rightmost interval j < i such that intervals[j].end <= intervals[i].start.
    for i in 0..n {
        let start_i = intervals[i].start;
        // We want to find the largest j < i with intervals[j].end <= start_i.
        // We'll do a binary search on the end times.
        let mut lo = 0_usize;
        let mut hi = i; // exclusive upper bound

        while lo < hi {
            let mid = (lo + hi) / 2;
            if intervals[mid].end <= start_i {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }

        // After this, lo is the smallest index such that intervals[lo].end > start_i,
        // so lo-1 is the index of the rightmost interval with end <= start_i.
        if lo > 0 {
            p[i] = (lo - 1) as isize;
        } else {
            p[i] = -1;
        }
    }
    p
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_max_weighted_schedule_basic() {
        let intervals = vec![
            WeightedInterval::new(0, 3, 4),
            WeightedInterval::new(1, 5, 2),
            WeightedInterval::new(4, 6, 5),
            WeightedInterval::new(5, 9, 6),
        ];
        // Best schedule is either: (0..3, weight=4) + (5..9, weight=6) => total=10
        assert_eq!(max_weighted_schedule(&intervals), 10);
    }

    #[test]
    fn test_best_weighted_schedule_basic() {
        let intervals = vec![
            WeightedInterval::new(0, 3, 4),
            WeightedInterval::new(1, 5, 2),
            WeightedInterval::new(4, 6, 5),
            WeightedInterval::new(5, 9, 6),
        ];
        let result = best_weighted_schedule(&intervals);
        let total_weight: usize = result.iter().map(|iv| iv.weight).sum();
        assert_eq!(total_weight, 10);
        // Check non-overlapping
        for i in 0..result.len() {
            for j in i + 1..result.len() {
                assert!(result[i].end <= result[j].start || result[j].end <= result[i].start);
            }
        }
    }

    #[test]
    fn test_edge_cases() {
        // Empty input
        let intervals: Vec<WeightedInterval> = vec![];
        assert_eq!(max_weighted_schedule(&intervals), 0);
        assert!(best_weighted_schedule(&intervals).is_empty());

        // One interval
        let intervals = vec![WeightedInterval::new(2, 4, 10)];
        assert_eq!(max_weighted_schedule(&intervals), 10);
        let result = best_weighted_schedule(&intervals);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].weight, 10);
    }
}
