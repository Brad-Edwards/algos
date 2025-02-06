use crate::math::integer_linear::{ILPSolution, ILPSolver, ILPStatus, IntegerLinearProgram};
use crate::math::optimization::simplex::{minimize, LinearProgram};
use crate::math::optimization::OptimizationConfig;
use std::error::Error;

pub struct BranchAndReduceSolver {
    max_iterations: usize,
    tolerance: f64,
}

impl BranchAndReduceSolver {
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

        Ok(ILPSolution {
            values: result.optimal_point,
            objective_value: -result.optimal_value, // Negate back since we're maximizing
            status: if result.converged {
                ILPStatus::Optimal
            } else {
                ILPStatus::Infeasible
            },
        })
    }

    fn reduce_problem(&self, problem: &mut IntegerLinearProgram) -> bool {
        let mut reduced = false;
        let n = problem.objective.len();
        let mut i = 0;

        while i < problem.constraints.len() {
            // Check if constraint is redundant
            let mut is_redundant = false;
            let current = &problem.constraints[i];

            // Simple redundancy check: if all coefficients are zero
            if current.iter().all(|&x| x.abs() < self.tolerance) {
                is_redundant = true;
            }

            // Check if constraint is dominated by another constraint
            for j in 0..problem.constraints.len() {
                if i != j {
                    let other = &problem.constraints[j];
                    if self.dominates(other, current, problem.bounds[j], problem.bounds[i]) {
                        is_redundant = true;
                        break;
                    }
                }
            }

            if is_redundant {
                problem.constraints.remove(i);
                problem.bounds.remove(i);
                reduced = true;
            } else {
                i += 1;
            }
        }

        // Variable fixing: if we can determine a variable must be at its bound
        for j in 0..n {
            if !problem.integer_vars.contains(&j) {
                continue;
            }

            let mut min_val = f64::NEG_INFINITY;
            let mut max_val = f64::INFINITY;

            for (i, constraint) in problem.constraints.iter().enumerate() {
                if constraint[j].abs() > self.tolerance {
                    let bound = problem.bounds[i] / constraint[j];
                    if constraint[j] > 0.0 {
                        max_val = max_val.min(bound);
                    } else {
                        min_val = min_val.max(bound);
                    }
                }
            }

            if min_val.ceil() == max_val.floor() {
                // Variable can be fixed
                let fixed_val = min_val.ceil();
                let mut new_constraint = vec![0.0; n];
                new_constraint[j] = 1.0;
                problem.constraints.push(new_constraint);
                problem.bounds.push(fixed_val);
                reduced = true;
            }
        }

        reduced
    }

    fn dominates(&self, a: &[f64], b: &[f64], a_bound: f64, b_bound: f64) -> bool {
        // Check if constraint a dominates constraint b
        let scale = b_bound / a_bound;
        a.iter()
            .zip(b.iter())
            .all(|(&x, &y)| (x * scale - y).abs() < self.tolerance)
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
        lower_branch.constraints.push(lower_constraint);
        lower_branch.bounds.push(value.floor());

        // Add constraint x_i >= ceil(value) to upper branch
        let mut upper_constraint = vec![0.0; problem.objective.len()];
        upper_constraint[var_idx] = 1.0;
        upper_branch.constraints.push(upper_constraint);
        upper_branch.bounds.push(value.ceil());

        (lower_branch, upper_branch)
    }
}

impl ILPSolver for BranchAndReduceSolver {
    fn solve(&self, problem: &IntegerLinearProgram) -> Result<ILPSolution, Box<dyn Error>> {
        let mut best_solution = None;
        let mut best_objective = f64::INFINITY;
        let mut nodes = vec![problem.clone()];
        let mut iterations = 0;

        while !nodes.is_empty() && iterations < self.max_iterations {
            iterations += 1;
            let mut current = nodes.pop().unwrap();

            // Try to reduce the problem
            let reduced = self.reduce_problem(&mut current);

            // Solve LP relaxation
            let relaxation = match self.solve_relaxation(&current) {
                Ok(sol) => sol,
                Err(_) => continue,
            };

            // Check if solution is worse than best known
            if relaxation.status != ILPStatus::Optimal
                || relaxation.objective_value >= best_objective
            {
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
                nodes.push(lower);
                nodes.push(upper);
            }

            // If problem was reduced, also explore the reduced problem
            if reduced {
                nodes.push(current);
            }
        }

        match best_solution {
            Some(solution) => Ok(solution),
            None => Ok(ILPSolution {
                values: vec![],
                objective_value: 0.0,
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

        let solver = BranchAndReduceSolver::new(1000, 1e-6);
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

        let solver = BranchAndReduceSolver::new(1000, 1e-6);
        let solution = solver.solve(&problem)?;

        assert_eq!(solution.status, ILPStatus::Infeasible);
        Ok(())
    }
}
