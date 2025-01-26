//! lib.rs

/// Returns the length of the Longest Increasing Subsequence (LIS) in `numbers`.
///
/// # Examples
///
/// ```
/// use algos::cs::dynamic::longest_increasing_subsequence::longest_increasing_subsequence_length;
///
/// let arr = vec![10, 9, 2, 5, 3, 7, 101, 18];
/// assert_eq!(longest_increasing_subsequence_length(&arr), 4);
/// // One possible LIS is [2, 5, 7, 101]
/// ```
pub fn longest_increasing_subsequence_length(numbers: &[i32]) -> usize {
    let mut tails = Vec::with_capacity(numbers.len());

    for &num in numbers {
        // Binary search in tails to find where 'num' should be placed.
        match tails.binary_search(&num) {
            // If 'num' is greater than or equal to tails[mid], we replace the first
            // larger element. (>= for strictly increasing, you might use > if you
            // want non-decreasing subsequence, etc.)
            Ok(pos) | Err(pos) => {
                if pos == tails.len() {
                    tails.push(num);
                } else {
                    tails[pos] = num;
                }
            }
        }
    }
    tails.len()
}

/// Returns one actual Longest Increasing Subsequence (LIS) in `numbers`.
///
/// This uses the same method as the length calculation, but keeps track of
/// the predecessor indices so that one valid subsequence can be reconstructed.
///
/// If there are multiple LIS with the same length, this returns just one of them.
///
/// # Examples
///
/// ```
/// use algos::cs::dynamic::longest_increasing_subsequence::longest_increasing_subsequence;
///
/// let arr = vec![10, 9, 2, 5, 3, 7, 101, 18];
/// let lis = longest_increasing_subsequence(&arr);
/// assert_eq!(lis.len(), 4);
/// // One possible LIS is [2, 5, 7, 101]
/// ```
pub fn longest_increasing_subsequence(numbers: &[i32]) -> Vec<i32> {
    if numbers.is_empty() {
        return Vec::new();
    }

    // tails[len] = index in `numbers` of the last element of an increasing subsequence of length len+1
    let mut tails_index = Vec::with_capacity(numbers.len());
    // Store the predecessor of each element in the LIS chain
    let mut prev_index = vec![-1; numbers.len()];

    let mut length = 0_usize;

    for (i, &num) in numbers.iter().enumerate() {
        // Binary search in the slice of `tails_index[..length]` using numbers[tails_index[pos]] for comparison
        let pos =
            match tails_index[..length].binary_search_by(|&idx: &usize| numbers[idx].cmp(&num)) {
                Ok(pos) | Err(pos) => pos,
            };

        // If pos equals current LIS length, we extend tails_index by one
        if pos == length {
            tails_index.push(i);
            length += 1;
        } else {
            // Otherwise, we update an existing position
            tails_index[pos] = i;
        }

        // Link this element to its predecessor (for reconstruction)
        if pos > 0 {
            let pred_index = tails_index[pos - 1];
            prev_index[i] = pred_index as isize;
        }
    }

    // Reconstruct the subsequence by backtracking from tails_index[length - 1]
    let mut lis = Vec::with_capacity(length);
    let mut curr_index = tails_index[length - 1];
    while curr_index != usize::MAX {
        lis.push(numbers[curr_index]);
        let pi = prev_index[curr_index];
        if pi < 0 {
            break;
        }
        curr_index = pi as usize;
    }
    lis.reverse();
    lis
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lis_length_empty() {
        assert_eq!(longest_increasing_subsequence_length(&[]), 0);
    }

    #[test]
    fn test_lis_sequence_empty() {
        let seq = longest_increasing_subsequence(&[]);
        assert_eq!(seq.len(), 0);
    }

    #[test]
    fn test_lis_length_basic() {
        // Example from LeetCode 300: [10,9,2,5,3,7,101,18] -> length is 4
        let nums = [10, 9, 2, 5, 3, 7, 101, 18];
        assert_eq!(longest_increasing_subsequence_length(&nums), 4);

        let nums2 = [0, 1, 0, 3, 2, 3];
        assert_eq!(longest_increasing_subsequence_length(&nums2), 4);
    }

    #[test]
    fn test_lis_sequence_basic() {
        let nums = [10, 9, 2, 5, 3, 7, 101, 18];
        let seq = longest_increasing_subsequence(&nums);
        // There's more than one correct LIS, but length must be 4
        assert_eq!(seq.len(), 4);

        // Quick check that seq is strictly increasing
        for win in seq.windows(2) {
            assert!(win[0] < win[1]);
        }
    }

    #[test]
    fn test_lis_additional() {
        let nums = [3, 1, 2, 1, 8, 6, 7];
        let length = longest_increasing_subsequence_length(&nums);
        assert_eq!(length, 4);
        let seq = longest_increasing_subsequence(&nums);
        assert_eq!(seq.len(), 4);
        for w in seq.windows(2) {
            assert!(w[0] < w[1]);
        }
    }
}
