/// Kadane's algorithm to find the maximum subarray sum in a slice of `i32`.
///
/// If the input slice is empty, returns `None`. Otherwise, returns `Some(<max sum>)`.
///
/// # Examples
///
/// ```
/// use algos::cs::graph::kadane;
///
/// let arr = [1, -2, 3, 5, -1];
/// let result = kadane(&arr);
/// assert_eq!(result, Some(8)); // The subarray [3, 5] has sum 8
/// ```
pub fn kadane(arr: &[i32]) -> Option<i32> {
    if arr.is_empty() {
        return None;
    }
    let mut current_sum = arr[0];
    let mut max_sum = arr[0];

    for &val in &arr[1..] {
        // Either extend the current subarray or start a new one at `val`
        current_sum = current_sum.max(0) + val;
        max_sum = max_sum.max(current_sum);
    }

    Some(max_sum)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_array() {
        let arr: [i32; 0] = [];
        assert_eq!(kadane(&arr), None, "Empty array should return None");
    }

    #[test]
    fn test_single_element() {
        let arr = [42];
        assert_eq!(
            kadane(&arr),
            Some(42),
            "Single-element array should return that element"
        );
    }

    #[test]
    fn test_all_negative() {
        let arr = [-8, -3, -6, -2, -5, -4];
        // Kadane's should pick the largest single negative element: -2
        assert_eq!(kadane(&arr), Some(-2));
    }

    #[test]
    fn test_mixed_values() {
        let arr = [1, -2, 3, 5, -1];
        // The subarray [3, 5] or [3, 5, -1] if we consider the negative is still beneficial or not:
        //   3 + 5 = 8, 3 + 5 + (-1) = 7, so max subarray sum is 8
        assert_eq!(kadane(&arr), Some(8));
    }

    #[test]
    fn test_large_positive() {
        let arr = [2, 2, 2, 2, 2];
        // All positives, so entire array is the max subarray
        assert_eq!(kadane(&arr), Some(10));
    }

    #[test]
    fn test_subarray_in_the_middle() {
        let arr = [-1, -2, 4, 5, -1, -2];
        // Max subarray is [4,5] => sum=9
        assert_eq!(kadane(&arr), Some(9));
    }

    #[test]
    fn test_subarray_at_the_end() {
        let arr = [-5, -1, 2, 3, 7];
        // Max subarray is [2,3,7] => sum=12
        assert_eq!(kadane(&arr), Some(12));
    }

    #[test]
    fn test_large_fluctuations() {
        let arr = [10, -5, 2, -1, 15, -20, 25, -2];
        // A plausible max subarray is [10, -5, 2, -1, 15, -20, 25] or shorter; let's break it down:
        //   partial sums: 10 -> 5 -> 7 -> 6 -> 21 -> 1 -> 26 -> 24
        // The largest encountered is 26
        assert_eq!(kadane(&arr), Some(26));
    }
}
