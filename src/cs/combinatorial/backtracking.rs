/// A collection of reference backtracking algorithms for permutations and combinations.
/// These functions demonstrate a simple, modern Rust approach to generate all possible
/// permutations or combinations of a given collection.
///
/// # Examples
/// ```
/// use algos::cs::combinatorial::backtracking::{permutations, combinations};
/// let data = vec![1, 2, 3];
/// let perms = permutations(&data);
/// assert_eq!(perms.len(), 6);
///
/// let combos = combinations(&data, 2);
/// assert_eq!(combos.len(), 3);
/// ```

/// Returns all permutations of the input slice using backtracking.
///
/// # Example
/// ```
/// use algos::cs::combinatorial::backtracking::permutations;
///
/// let items = vec!['a', 'b', 'c'];
/// let perms = permutations(&items);
/// let mut expected = vec![
///     vec!['a', 'b', 'c'],
///     vec!['a', 'c', 'b'],
///     vec!['b', 'a', 'c'],
///     vec!['b', 'c', 'a'],
///     vec!['c', 'b', 'a'],
///     vec!['c', 'a', 'b'],
/// ];
/// expected.sort();
/// let mut perms_sorted = perms.clone();
/// perms_sorted.sort();
/// assert_eq!(perms_sorted, expected);
/// assert_eq!(perms.len(), 6);
/// ```
pub fn permutations<T: Clone>(items: &[T]) -> Vec<Vec<T>> {
    let mut results = Vec::new();
    let mut temp = items.to_vec();
    backtrack_permutations(0, &mut temp, &mut results);
    results
}

fn backtrack_permutations<T: Clone>(start: usize, current: &mut [T], results: &mut Vec<Vec<T>>) {
    if start == current.len() {
        results.push(current.to_vec());
        return;
    }
    for i in start..current.len() {
        current.swap(start, i);
        backtrack_permutations(start + 1, current, results);
        current.swap(start, i);
    }
}

/// Returns all combinations of the input slice, choosing `k` items each time.
///
/// # Example
/// ```
/// use algos::cs::combinatorial::backtracking::combinations;
///
/// let items = vec![1, 2, 3, 4];
/// let combos = combinations(&items, 2);
/// assert_eq!(combos, vec![
///     vec![1, 2],
///     vec![1, 3],
///     vec![1, 4],
///     vec![2, 3],
///     vec![2, 4],
///     vec![3, 4],
/// ]);
/// ```
pub fn combinations<T: Clone>(items: &[T], k: usize) -> Vec<Vec<T>> {
    let mut results = Vec::new();
    let mut combo = Vec::with_capacity(k);
    backtrack_combinations(items, 0, k, &mut combo, &mut results);
    results
}

fn backtrack_combinations<T: Clone>(
    items: &[T],
    start: usize,
    k: usize,
    combo: &mut Vec<T>,
    results: &mut Vec<Vec<T>>,
) {
    if combo.len() == k {
        results.push(combo.clone());
        return;
    }
    for i in start..items.len() {
        combo.push(items[i].clone());
        backtrack_combinations(items, i + 1, k, combo, results);
        combo.pop();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_permutations_empty() {
        let items: Vec<i32> = vec![];
        let perms = permutations(&items);
        assert_eq!(perms, vec![vec![]]);
    }

    #[test]
    fn test_permutations_single() {
        let items = vec![1];
        let perms = permutations(&items);
        assert_eq!(perms, vec![vec![1]]);
    }

    #[test]
    fn test_permutations_two() {
        let items = vec![1, 2];
        let perms = permutations(&items);
        assert_eq!(perms, vec![vec![1, 2], vec![2, 1]]);
    }

    #[test]
    fn test_permutations_three() {
        let items = vec!['a', 'b', 'c'];
        let mut perms = permutations(&items);
        let mut expected = vec![
            vec!['a', 'b', 'c'],
            vec!['a', 'c', 'b'],
            vec!['b', 'a', 'c'],
            vec!['b', 'c', 'a'],
            vec!['c', 'a', 'b'],
            vec!['c', 'b', 'a'],
        ];
        perms.sort();
        expected.sort();
        assert_eq!(perms, expected);
    }

    #[test]
    fn test_combinations_empty() {
        let items: Vec<i32> = vec![];
        let combos = combinations(&items, 0);
        assert_eq!(combos, vec![vec![]]);
    }

    #[test]
    fn test_combinations_k_zero() {
        let items = vec![1, 2, 3];
        let combos = combinations(&items, 0);
        assert_eq!(combos, vec![vec![]]);
    }

    #[test]
    fn test_combinations_k_one() {
        let items = vec![1, 2, 3];
        let combos = combinations(&items, 1);
        assert_eq!(combos, vec![vec![1], vec![2], vec![3]]);
    }

    #[test]
    fn test_combinations_k_two() {
        let items = vec![1, 2, 3, 4];
        let combos = combinations(&items, 2);
        assert_eq!(combos, vec![
            vec![1, 2],
            vec![1, 3],
            vec![1, 4],
            vec![2, 3],
            vec![2, 4],
            vec![3, 4],
        ]);
    }

    #[test]
    fn test_combinations_k_all() {
        let items = vec![1, 2, 3];
        let combos = combinations(&items, 3);
        assert_eq!(combos, vec![vec![1, 2, 3]]);
    }

    #[test]
    fn test_combinations_k_too_large() {
        let items = vec![1, 2, 3];
        let combos = combinations(&items, 4);
        assert!(combos.is_empty());
    }
} 