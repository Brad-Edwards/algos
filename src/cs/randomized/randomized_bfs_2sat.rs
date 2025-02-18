use rand::seq::SliceRandom;
use rand::thread_rng;
use rand::Rng;
use std::collections::{HashSet, VecDeque};

/// Checks if the given assignment satisfies the 2-SAT formula.
/// Clauses are represented as (i32, i32) where a positive integer represents a literal (variable is true)
/// and a negative integer represents its negation (variable is false).
fn satisfies(assignment: &[bool], clauses: &[(i32, i32)]) -> bool {
    for &(a, b) in clauses {
        let lit_a = if a > 0 {
            assignment[(a - 1) as usize]
        } else {
            !assignment[(-a - 1) as usize]
        };
        let lit_b = if b > 0 {
            assignment[(b - 1) as usize]
        } else {
            !assignment[(-b - 1) as usize]
        };
        if !(lit_a || lit_b) {
            return false;
        }
    }
    true
}

/// Converts a boolean assignment to a unique key (assuming num_vars ≤ 64).
fn assignment_to_key(assignment: &[bool]) -> u64 {
    let mut key = 0u64;
    for &b in assignment {
        key = (key << 1) | (if b { 1 } else { 0 });
    }
    key
}

/// Solves a 2-SAT instance using a randomized BFS over the space of assignments.
///
/// # Arguments
/// - `clauses`: A slice of clauses, each represented as a tuple (i32, i32).
///   A positive literal i represents variable i being true; a negative literal -i represents variable i being false.
/// - `num_vars`: The total number of variables.
///
/// # Returns
/// Returns Some(assignment) where assignment is a Vec<bool> (index 0 corresponds to variable 1), if a satisfying assignment is found.
/// Returns None if the formula is unsatisfiable.
pub fn randomized_bfs_2sat(clauses: &[(i32, i32)], num_vars: usize) -> Option<Vec<bool>> {
    let mut rng = thread_rng();
    let mut queue = VecDeque::new();
    // Generate a random initial assignment.
    let initial: Vec<bool> = (0..num_vars).map(|_| rng.gen()).collect();
    queue.push_back(initial.clone());
    let mut visited = HashSet::new();
    visited.insert(assignment_to_key(&initial));

    while !queue.is_empty() {
        // Randomly select an assignment from the queue.
        let idx = rng.gen_range(0..queue.len());
        let current = queue.remove(idx).unwrap();
        if satisfies(&current, clauses) {
            return Some(current);
        }
        // Generate neighbors by flipping each variable.
        let mut neighbors = Vec::new();
        for i in 0..num_vars {
            let mut neighbor = current.clone();
            neighbor[i] = !neighbor[i];
            let key = assignment_to_key(&neighbor);
            if !visited.contains(&key) {
                visited.insert(key);
                neighbors.push(neighbor);
            }
        }
        neighbors.shuffle(&mut rng);
        for n in neighbors {
            queue.push_back(n);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_randomized_bfs_2sat_satisfiable() {
        // Example 2-SAT formula:
        // (x1 OR x2) ∧ (¬x1 OR x3) ∧ (¬x2 OR ¬x3)
        // Represented as: [(1, 2), (-1, 3), (-2, -3)]
        let clauses = vec![(1, 2), (-1, 3), (-2, -3)];
        let num_vars = 3;
        let assignment = randomized_bfs_2sat(&clauses, num_vars);
        assert!(assignment.is_some());
        let asgn = assignment.unwrap();
        assert!(satisfies(&asgn, &clauses));
    }

    #[test]
    fn test_randomized_bfs_2sat_unsatisfiable() {
        // Unsatisfiable formula: (x1) ∧ (¬x1)
        let clauses = vec![(1, 1), (-1, -1)];
        let num_vars = 1;
        let assignment = randomized_bfs_2sat(&clauses, num_vars);
        // The method should return None for an unsatisfiable formula.
        assert!(assignment.is_none());
    }
}
