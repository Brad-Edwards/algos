/// Johnson-Trotter permutation generator in Rust.
/// Produces all permutations of distinct items in "adjacent swap" (Johnson-Trotter) order.
///
/// # Example
/// ```
/// use algos::cs::combinatorial::johnson_trotter;
///
/// let data = vec![1, 2, 3];
/// let perms = johnson_trotter(&data);
/// assert_eq!(perms, vec![
///     vec![1, 2, 3],
///     vec![1, 3, 2],
///     vec![3, 1, 2],
///     vec![3, 2, 1],
///     vec![2, 3, 1],
///     vec![2, 1, 3],
/// ]);
/// ```

/// Johnson-Trotter algorithm for generating permutations of distinct items.
/// Returns a vector of permutations in adjacent-swap order.
pub fn johnson_trotter<T: Copy + Ord>(items: &[T]) -> Vec<Vec<T>> {
    if items.is_empty() {
        return vec![vec![]];
    }

    // We'll work from a local copy and track directions for each position.
    let mut elements = items.to_vec();
    // Directions: -1 for left, +1 for right. Start all pointing left.
    let mut dirs = vec![-1; elements.len()];

    // Collect all permutations in this vector.
    let mut result = Vec::new();
    result.push(elements.clone());

    loop {
        // 1. Find the largest mobile element (an element whose adjacent in its direction is smaller).
        let mut mobile_index = None;
        let mut mobile_value = None;

        for i in 0..elements.len() {
            let dir = dirs[i];
            let adj = (i as isize + dir as isize) as usize;
            // Check bounds and "mobility".
            if adj < elements.len() && elements[i] > elements[adj] {
                // If it's bigger than the adjacent, it's mobile.
                // Track the largest such element.
                if mobile_value.map_or(true, |val| elements[i] > val) {
                    mobile_index = Some(i);
                    mobile_value = Some(elements[i]);
                }
            }
        }

        // 2. If no mobile element, we're done.
        let Some(i) = mobile_index else {
            break;
        };

        // 3. Swap with the adjacent element in its direction.
        let dir = dirs[i];
        let j = (i as isize + dir as isize) as usize;
        elements.swap(i, j);
        dirs.swap(i, j);

        // 4. After the swap, reverse directions of all elements larger than the chosen one.
        if let Some(val) = mobile_value {
            for d_i in 0..elements.len() {
                if elements[d_i] > val {
                    dirs[d_i] = -dirs[d_i];
                }
            }
        }

        // 5. Record the new permutation.
        result.push(elements.clone());
    }

    result
}

#[cfg(test)]
mod tests {
    use super::johnson_trotter;

    #[test]
    fn test_jt_basic() {
        let data = vec![1, 2, 3];
        let perms = johnson_trotter(&data);
        // The known Johnson-Trotter order for [1,2,3] is:
        let expected = vec![
            vec![1, 2, 3],
            vec![1, 3, 2],
            vec![3, 1, 2],
            vec![3, 2, 1],
            vec![2, 3, 1],
            vec![2, 1, 3],
        ];
        assert_eq!(perms, expected);
    }

    #[test]
    fn test_jt_empty() {
        let data: Vec<i32> = vec![];
        let perms = johnson_trotter(&data);
        assert_eq!(perms, vec![vec![]]);
    }

    #[test]
    fn test_jt_single() {
        let data = vec![42];
        let perms = johnson_trotter(&data);
        assert_eq!(perms, vec![vec![42]]);
    }
} 