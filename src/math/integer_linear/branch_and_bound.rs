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
        // Convert "maximize f(x)" into "minimize -f(x)"
        let neg_objective: Vec<f64> = problem.objective.iter().map(|&c| -c).collect();

        // LP in standard form: A x <= b
        let lp = LinearProgram {
            objective: neg_objective,
            constraints: problem.constraints.clone(),
            rhs: problem.bounds.clone(),
        };

        let config = OptimizationConfig {
            max_iterations: self.max_iterations,
            tolerance: self.tolerance,
            learning_rate: 1.0,
        };
        let result = minimize(&lp, &config);

        // Basic checks
        if !result.converged
            || result.optimal_point.is_empty()
            || result.optimal_value.is_infinite()
            || result.optimal_value.is_nan()
        {
            return Ok(ILPSolution {
                values: vec![],
                objective_value: f64::NEG_INFINITY,
                status: ILPStatus::Infeasible,
            });
        }

        // Check feasibility: A x <= b + tol
        for (constraint, &b) in problem.constraints.iter().zip(problem.bounds.iter()) {
            let lhs: f64 = constraint
                .iter()
                .zip(&result.optimal_point)
                .map(|(a, &x)| a * x)
                .sum();
            if lhs > b + self.tolerance {
                return Ok(ILPSolution {
                    values: vec![],
                    objective_value: f64::NEG_INFINITY,
                    status: ILPStatus::Infeasible,
                });
            }
        }

        // Everything is feasible; negate minimized value to get original "maximize" objective
        Ok(ILPSolution {
            values: result.optimal_point.clone(),
            objective_value: -result.optimal_value,
            status: ILPStatus::Optimal,
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

        // x_i <= floor(value)
        let mut lower_constraint = vec![0.0; problem.objective.len()];
        lower_constraint[var_idx] = 1.0;
        lower_branch.constraints.push(lower_constraint);
        lower_branch.bounds.push(value.floor());

        // x_i >= ceil(value) => -x_i <= -ceil(value)
        let mut upper_constraint = vec![0.0; problem.objective.len()];
        upper_constraint[var_idx] = -1.0;
        upper_branch.constraints.push(upper_constraint);
        upper_branch.bounds.push(-value.ceil());

        (lower_branch, upper_branch)
    }
}

impl ILPSolver for BranchAndBoundSolver {
    fn solve(&self, problem: &IntegerLinearProgram) -> Result<ILPSolution, Box<dyn Error>> {
        let mut best_solution = None;
        let mut best_objective = f64::NEG_INFINITY;
        let mut stack = vec![problem.clone()];
        let mut iterations = 0;

        while let Some(node) = stack.pop() {
            iterations += 1;
            if iterations > self.max_iterations {
                break;
            }

            // Solve LP relaxation on this node
            let relaxation = match self.solve_relaxation(&node) {
                Ok(sol) if sol.status == ILPStatus::Optimal => sol,
                _ => continue,
            };

            // If the relaxation's best bound <= current best, prune
            if relaxation.objective_value <= best_objective {
                continue;
            }

            // Check integer feasibility
            let mut all_integer = true;
            let mut most_fractional = None;
            let mut max_frac_diff = 0.0;

            for (i, &v) in relaxation.values.iter().enumerate() {
                if node.integer_vars.contains(&i) && !self.is_integer(v) {
                    all_integer = false;
                    let frac = (v - v.floor()).abs();
                    let diff = (frac - 0.5).abs();
                    if diff > max_frac_diff {
                        max_frac_diff = diff;
                        most_fractional = Some((i, v));
                    }
                }
            }

            if all_integer {
                // Update best solution if it improves
                if relaxation.objective_value > best_objective {
                    best_objective = relaxation.objective_value;
                    best_solution = Some(relaxation);
                }
            } else if let Some((idx, val)) = most_fractional {
                // Branch on the most fractional variable
                let (lower, upper) = self.branch(&node, idx, val);
                stack.push(lower);
                stack.push(upper);
            }
        }

        Ok(best_solution.unwrap_or(ILPSolution {
            values: vec![],
            objective_value: f64::NEG_INFINITY,
            status: ILPStatus::Infeasible,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_ilp() -> Result<(), Box<dyn Error>> {
        // maximize 2x + y
        // subject to:
        //   x + y <= 4
        //   x <= 2
        //   x, y >= 0 and integer
        let problem = IntegerLinearProgram {
            objective: vec![2.0, 1.0], // Prefer x over y
            constraints: vec![
                vec![1.0, 1.0], // x + y <= 4
                vec![1.0, 0.0], // x <= 2
            ],
            bounds: vec![4.0, 2.0],
            integer_vars: vec![0, 1],
        };

        let solver = BranchAndBoundSolver::new(100, 1e-6);
        let solution = solver.solve(&problem)?;

        assert_eq!(solution.status, ILPStatus::Optimal);
        // Optimal solution should be x=2, y=2 giving value of 6
        assert!((solution.objective_value - 6.0).abs() < 1e-6);

        // Check integer feasibility
        for &v in &solution.values {
            assert!((v - v.round()).abs() < 1e-6);
            assert!(v >= 0.0); // Check non-negativity
        }

        Ok(())
    }

    #[test]
    fn test_infeasible_ilp() -> Result<(), Box<dyn Error>> {
        // maximize x + y
        // subject to:
        //   x + y <= 5
        //   x + y >= 6
        //   x, y >= 0 and integer
        let problem = IntegerLinearProgram {
            objective: vec![1.0, 1.0],
            constraints: vec![
                vec![1.0, 1.0],   // x + y <= 5
                vec![-1.0, -1.0], // -(x + y) <= -6  (x + y >= 6)
                vec![-1.0, 0.0],  // -x <= 0  (x >= 0)
                vec![0.0, -1.0],  // -y <= 0  (y >= 0)
            ],
            bounds: vec![5.0, -6.0, 0.0, 0.0],
            integer_vars: vec![0, 1],
        };

        let solver = BranchAndBoundSolver::new(100, 1e-6);
        let solution = solver.solve(&problem)?;

        assert_eq!(solution.status, ILPStatus::Infeasible);
        Ok(())
    }
}
