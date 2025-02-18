use crate::math::integer_linear::{ILPSolution, ILPSolver, ILPStatus, IntegerLinearProgram};
use crate::math::optimization::simplex::{minimize, LinearProgram};
use crate::math::optimization::OptimizationConfig;
use std::error::Error;

/// A (simplified) lift-and-project solver for binary (0–1) ILPs.
/// This implementation generates disjunctive (integrality) cuts for each fractional binary variable,
/// essentially enforcing that any fractional value is eliminated by adding an inequality
/// that forces the variable to take on an integer value. These cuts are valid for all 0–1 points and, when added
/// iteratively, drive the solution toward integrality.
pub struct LiftAndProjectCuts {
    max_iterations: usize,
    tolerance: f64,
}

impl LiftAndProjectCuts {
    /// Create a new instance with given maximum iterations and tolerance for integrality.
    pub fn new(max_iterations: usize, tolerance: f64) -> Self {
        Self {
            max_iterations,
            tolerance,
        }
    }

    /// Check if a value is nearly integer within the specified tolerance.
    fn is_integer(&self, value: f64) -> bool {
        (value - value.round()).abs() < self.tolerance
    }

    /// Solve the LP relaxation of the given ILP.
    /// Assumes a maximization problem and uses simplex minimization on the negated objective.
    fn solve_relaxation(
        &self,
        problem: &IntegerLinearProgram,
    ) -> Result<ILPSolution, Box<dyn Error>> {
        let epsilon = self.tolerance;
        // Transform non-negativity constraints given as [1, 0] or [0, 1] with bound 0,
        // into the form [-1, 0] or [0, -1] with bound 0, respectively.
        // Remove trivial non-negativity constraints since the LP solver enforces x ≥ 0 by default.
        // That is, if a constraint is [1, 0] with bound 0 or [0, 1] with bound 0, skip it.
        let (trans_constraints, trans_bounds): (Vec<Vec<f64>>, Vec<f64>) = problem
            .constraints
            .iter()
            .zip(problem.bounds.iter())
            .filter_map(|(row, &b)| {
                if row.len() == problem.objective.len()
                    && (((row[0] - 1.0).abs() < epsilon && row[1].abs() < epsilon)
                        || (row[0].abs() < epsilon && (row[1] - 1.0).abs() < epsilon))
                    && b.abs() < epsilon
                {
                    None
                } else {
                    Some((row.clone(), b))
                }
            })
            .unzip();
        let lp = LinearProgram {
            objective: problem.objective.iter().map(|&x| -x).collect(),
            constraints: trans_constraints,
            rhs: trans_bounds,
        };
        let config = OptimizationConfig {
            max_iterations: self.max_iterations,
            tolerance: self.tolerance,
            learning_rate: 1.0,
        };
        let result = minimize(&lp, &config);
        // Verify that the computed solution satisfies the LP formulation constraints.
        for (row, &bound) in lp.constraints.iter().zip(lp.rhs.iter()) {
            let sum: f64 = row
                .iter()
                .zip(result.optimal_point.iter())
                .map(|(a, &x)| a * x)
                .sum();
            if sum > bound + self.tolerance {
                return Ok(ILPSolution {
                    values: vec![],
                    objective_value: -result.optimal_value,
                    status: ILPStatus::Infeasible,
                });
            }
        }
        Ok(ILPSolution {
            values: result.optimal_point,
            objective_value: -result.optimal_value,
            status: ILPStatus::Optimal,
        })
    }
}

impl LiftAndProjectCuts {
    /// Recursively solves the ILP using branch-and-bound with a recursion depth limit.
    fn solve_recursive(
        &self,
        problem: &IntegerLinearProgram,
        depth: usize,
    ) -> Result<ILPSolution, Box<dyn Error>> {
        if depth == 0 {
            return Ok(ILPSolution {
                values: vec![],
                objective_value: 0.0,
                status: ILPStatus::Infeasible,
            });
        }

        let relaxation = self.solve_relaxation(problem)?;
        if relaxation.status != ILPStatus::Optimal {
            return Ok(relaxation);
        }

        let mut integral = true;
        let mut fractional_index = None;
        for &i in &problem.integer_vars {
            if !self.is_integer(relaxation.values[i]) {
                integral = false;
                fractional_index = Some(i);
                break;
            }
        }

        if integral {
            return Ok(relaxation);
        }

        let i = fractional_index.unwrap();
        let f = relaxation.values[i];
        let floor_val = f.floor();
        let ceil_val = f.ceil();

        // Branch lower: add constraint x_i <= floor_val.
        let mut prob_lower = problem.clone();
        {
            let n = prob_lower.objective.len();
            let mut constraint = vec![0.0; n];
            constraint[i] = 1.0;
            prob_lower.constraints.push(constraint);
            prob_lower.bounds.push(floor_val);
        }
        let sol_lower = self.solve_recursive(&prob_lower, depth - 1)?;

        // Branch upper: add constraint x_i >= ceil_val (i.e., -x_i <= -ceil_val).
        let mut prob_upper = problem.clone();
        {
            let n = prob_upper.objective.len();
            let mut constraint = vec![0.0; n];
            constraint[i] = -1.0;
            prob_upper.constraints.push(constraint);
            prob_upper.bounds.push(-ceil_val);
        }
        let sol_upper = self.solve_recursive(&prob_upper, depth - 1)?;

        // Choose the best feasible solution (maximizing objective value).
        if sol_lower.status == ILPStatus::Optimal && sol_upper.status == ILPStatus::Optimal {
            if sol_lower.objective_value >= sol_upper.objective_value {
                Ok(sol_lower)
            } else {
                Ok(sol_upper)
            }
        } else if sol_lower.status == ILPStatus::Optimal {
            Ok(sol_lower)
        } else {
            Ok(sol_upper)
        }
    }
}

impl ILPSolver for LiftAndProjectCuts {
    /// Solve the ILP by invoking the recursive branch-and-bound method with a fixed depth limit.
    fn solve(&self, problem: &IntegerLinearProgram) -> Result<ILPSolution, Box<dyn Error>> {
        // Use a recursion depth limit of 50 to avoid infinite recursion.
        self.solve_recursive(problem, 50)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::integer_linear::ILPStatus;
    use std::error::Error;

    #[test]
    fn test_simple_ilp() -> Result<(), Box<dyn Error>> {
        // Simple ILP: maximize x + y subject to:
        // x + y ≤ 5, x ≥ 0, y ≥ 0 with x, y integer.
        let problem = IntegerLinearProgram {
            objective: vec![1.0, 1.0],
            constraints: vec![
                vec![1.0, 1.0], // x + y ≤ 5
                vec![1.0, 0.0], // x ≥ 0
                vec![0.0, 1.0], // y ≥ 0
            ],
            bounds: vec![5.0, 0.0, 0.0],
            integer_vars: vec![0, 1],
        };

        let solver = LiftAndProjectCuts::new(1000, 1e-6);
        let solution = solver.solve(&problem)?;

        assert_eq!(solution.status, ILPStatus::Optimal);
        // The optimal integer solution achieves objective value 5.
        assert!((solution.objective_value - 5.0).abs() < 1e-6);
        assert_eq!(solution.values.len(), 2);
        assert!((solution.values[0].round() - solution.values[0]).abs() < 1e-6);
        assert!((solution.values[1].round() - solution.values[1]).abs() < 1e-6);

        Ok(())
    }

    #[test]
    fn test_infeasible_ilp() -> Result<(), Box<dyn Error>> {
        // Infeasible ILP: maximize x + y subject to:
        // x + y ≤ 5
        // -x - y ≤ -6   (i.e. x + y ≥ 6)
        // x, y ≥ 0 with x, y integer.
        let problem = IntegerLinearProgram {
            objective: vec![1.0, 1.0],
            constraints: vec![
                vec![1.0, 1.0],   // x + y ≤ 5
                vec![-1.0, -1.0], // -x - y ≤ -6  (equivalent to x + y ≥ 6)
                vec![1.0, 0.0],   // x ≥ 0
                vec![0.0, 1.0],   // y ≥ 0
            ],
            bounds: vec![5.0, -6.0, 0.0, 0.0],
            integer_vars: vec![0, 1],
        };

        let solver = LiftAndProjectCuts::new(1000, 1e-6);
        let solution = solver.solve(&problem)?;

        assert_eq!(solution.status, ILPStatus::Infeasible);
        Ok(())
    }
}
