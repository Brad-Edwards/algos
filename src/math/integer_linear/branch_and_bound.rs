use crate::math::integer_linear::{ILPSolution, ILPSolver, ILPStatus, IntegerLinearProgram};
use crate::math::optimization::simplex::{minimize, LinearProgram};
use crate::math::optimization::OptimizationConfig;
use std::error::Error;

pub struct BranchAndBoundSolver {
    max_iterations: usize,
    tolerance: f64,
}

impl BranchAndBoundSolver {
    pub fn new(max_iterations: usize, tolerance: f64) -> Self {
        Self {
            max_iterations,
            tolerance,
        }
    }

    fn is_integer(&self, value: f64) -> bool {
        (value - value.round()).abs() < self.tolerance
    }

    fn solve_relaxation(
        &self,
        problem: &IntegerLinearProgram,
    ) -> Result<ILPSolution, Box<dyn Error>> {
        let lp = LinearProgram {
            // For maximization, we need to negate the objective since minimize will negate it again
            objective: problem.objective.iter().map(|&x| -x).collect(),
            constraints: problem.constraints.clone(),
            rhs: problem.bounds.clone(),
        };

        let config = OptimizationConfig {
            max_iterations: self.max_iterations,
            tolerance: self.tolerance,
            learning_rate: 1.0,
        };
        let result = minimize(&lp, &config);
        let optimal_point = result.optimal_point.clone();

        Ok(ILPSolution {
            values: optimal_point.clone(),
            objective_value: -result.optimal_value, // Negate back since we're maximizing
            status: if !result.converged {
                ILPStatus::MaxIterationsReached
            } else if optimal_point.is_empty() {
                ILPStatus::Infeasible
            } else {
                ILPStatus::Optimal
            },
        })
    }

    fn branch(
        &self,
        problem: &IntegerLinearProgram,
        var_idx: usize,
        value: f64,
    ) -> (IntegerLinearProgram, IntegerLinearProgram) {
        let mut lower_branch = problem.clone();
        let mut upper_branch = problem.clone();

        // Add constraint x_i <= floor(value) to lower branch
        let mut lower_constraint = vec![0.0; problem.objective.len()];
        lower_constraint[var_idx] = 1.0;
        lower_branch.constraints.push(lower_constraint.clone());
        lower_branch.bounds.push(value.floor());

        // Add constraint x_i >= ceil(value) to upper branch
        let mut upper_constraint = vec![0.0; problem.objective.len()];
        upper_constraint[var_idx] = -1.0; // Negative coefficient for >= constraint
        upper_branch.constraints.push(upper_constraint);
        upper_branch.bounds.push(-value.ceil()); // Negative RHS for >= constraint

        (lower_branch, upper_branch)
    }
}

impl ILPSolver for BranchAndBoundSolver {
    fn solve(&self, problem: &IntegerLinearProgram) -> Result<ILPSolution, Box<dyn Error>> {
        let mut best_solution = None;
        let mut best_objective = f64::NEG_INFINITY; // Start with negative infinity for maximization
        let mut nodes = vec![problem.clone()];
        let mut iterations = 0;

        while !nodes.is_empty() && iterations < self.max_iterations {
            iterations += 1;
            let current = nodes.pop().unwrap();

            // Solve LP relaxation
            let relaxation = match self.solve_relaxation(&current) {
                Ok(sol) => sol,
                Err(_) => continue,
            };

            // Check if solution is feasible and better than best known
            if relaxation.status != ILPStatus::Optimal
                || relaxation.objective_value <= best_objective
            {
                // Changed comparison for maximization
                continue;
            }

            // Check if solution is integer feasible
            let mut all_integer = true;
            let mut first_fractional = None;
            for (i, &value) in relaxation.values.iter().enumerate() {
                if problem.integer_vars.contains(&i) && !self.is_integer(value) {
                    all_integer = false;
                    first_fractional = Some((i, value));
                    break;
                }
            }

            if all_integer {
                best_solution = Some(relaxation.clone());
                best_objective = relaxation.objective_value;
            } else if let Some((var_idx, value)) = first_fractional {
                let (lower, upper) = self.branch(&current, var_idx, value);
                nodes.push(upper); // Push upper branch first to explore larger values
                nodes.push(lower);
            }
        }

        match best_solution {
            Some(solution) => Ok(solution),
            None => Ok(ILPSolution {
                values: vec![],
                objective_value: f64::NEG_INFINITY,
                status: if iterations >= self.max_iterations {
                    ILPStatus::MaxIterationsReached
                } else {
                    ILPStatus::Infeasible
                },
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_ilp() -> Result<(), Box<dyn Error>> {
        // Simple ILP: maximize x + y subject to:
        // x + y <= 5
        // x, y >= 0
        // x, y integer
        let problem = IntegerLinearProgram {
            objective: vec![1.0, 1.0],
            constraints: vec![
                vec![1.0, 1.0], // x + y <= 5
                vec![1.0, 0.0], // x >= 0
                vec![0.0, 1.0], // y >= 0
            ],
            bounds: vec![5.0, 0.0, 0.0],
            integer_vars: vec![0, 1],
        };

        let solver = BranchAndBoundSolver::new(1000, 1e-6);
        let solution = solver.solve(&problem)?;

        assert_eq!(solution.status, ILPStatus::Optimal);
        assert!((solution.objective_value - 5.0).abs() < 1e-6);
        assert!(solution.values.len() == 2);
        assert!((solution.values[0].round() - solution.values[0]).abs() < 1e-6);
        assert!((solution.values[1].round() - solution.values[1]).abs() < 1e-6);

        Ok(())
    }

    #[test]
    fn test_infeasible_ilp() -> Result<(), Box<dyn Error>> {
        // Infeasible ILP: maximize x + y subject to:
        // x + y <= 5
        // x + y >= 6
        // x, y >= 0
        // x, y integer
        let problem = IntegerLinearProgram {
            objective: vec![1.0, 1.0],
            constraints: vec![
                vec![1.0, 1.0], // x + y <= 5
                vec![1.0, 1.0], // x + y >= 6
                vec![1.0, 0.0], // x >= 0
                vec![0.0, 1.0], // y >= 0
            ],
            bounds: vec![5.0, 6.0, 0.0, 0.0],
            integer_vars: vec![0, 1],
        };

        let solver = BranchAndBoundSolver::new(1000, 1e-6);
        let solution = solver.solve(&problem)?;

        assert_eq!(solution.status, ILPStatus::Infeasible);
        Ok(())
    }
}
