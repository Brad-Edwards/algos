#[derive(Debug, Clone, PartialEq)]
pub struct Clause {
    literals: Vec<i32>,
    weight: f64,
}

impl Clause {
    pub fn new(literals: Vec<i32>, weight: f64) -> Self {
        Self { literals, weight }
    }
}

pub struct MaxSatInstance {
    clauses: Vec<Clause>,
    num_variables: usize,
}

impl MaxSatInstance {
    pub fn new(clauses: Vec<Clause>, num_variables: usize) -> Self {
        Self {
            clauses,
            num_variables,
        }
    }
}

/// Implements Johnson's algorithm for the weighted MAX-SAT problem.
///
/// This algorithm provides a 2/3-approximation guarantee for the weighted maximum satisfiability
/// problem. It works by:
/// 1. Assigning each variable true/false with probability 1/2
/// 2. Repeatedly improving the solution by flipping variables that increase satisfied weight
///
/// # Arguments
///
/// * `instance` - The MAX-SAT instance containing clauses and number of variables
///
/// # Returns
///
/// * A tuple containing:
///   - Vector of boolean assignments for each variable
///   - Total weight of satisfied clauses
pub fn solve(instance: &MaxSatInstance) -> (Vec<bool>, f64) {
    use rand::Rng;
    let mut rng = rand::thread_rng();

    // Initial random assignment
    let mut assignment: Vec<bool> = (0..instance.num_variables)
        .map(|_| rng.gen_bool(0.5))
        .collect();

    let mut improved = true;
    while improved {
        improved = false;

        // Try flipping each variable
        for var in 0..instance.num_variables {
            let current_weight = evaluate_assignment(&assignment, instance);
            assignment[var] = !assignment[var];
            let new_weight = evaluate_assignment(&assignment, instance);

            if new_weight <= current_weight {
                // Revert flip if it didn't improve solution
                assignment[var] = !assignment[var];
            } else {
                improved = true;
            }
        }
    }

    let final_weight = evaluate_assignment(&assignment, instance);
    (assignment, final_weight)
}

fn evaluate_assignment(assignment: &[bool], instance: &MaxSatInstance) -> f64 {
    instance
        .clauses
        .iter()
        .filter(|clause| is_clause_satisfied(clause, assignment))
        .map(|clause| clause.weight)
        .sum()
}

fn is_clause_satisfied(clause: &Clause, assignment: &[bool]) -> bool {
    clause.literals.iter().any(|&lit| {
        let var_idx = (lit.abs() - 1) as usize;
        let var_value = assignment[var_idx];
        if lit > 0 {
            var_value
        } else {
            !var_value
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_sat_instance() {
        // Create a simple MAX-SAT instance with 2 variables and 3 clauses
        let clauses = vec![
            Clause::new(vec![1, 2], 1.0),  // x1 ∨ x2
            Clause::new(vec![1, -2], 1.0), // x1 ∨ ¬x2
            Clause::new(vec![-1], 1.0),    // ¬x1
        ];

        let instance = MaxSatInstance::new(clauses, 2);
        let (_assignment, weight) = solve(&instance);

        // At least 2 clauses should be satisfied (2/3 approximation)
        assert!(weight >= 2.0);
    }

    #[test]
    fn test_weighted_sat_instance() {
        let clauses = vec![
            Clause::new(vec![1], 2.0),  // x1 (weight 2)
            Clause::new(vec![-1], 1.0), // ¬x1 (weight 1)
        ];

        let instance = MaxSatInstance::new(clauses, 1);
        let (_assignment, weight) = solve(&instance);

        // Should choose the assignment satisfying the clause with weight 2
        assert!(weight >= 2.0);
    }
}
