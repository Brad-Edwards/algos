use crate::math::integer_linear::{ILPSolution, ILPSolver, ILPStatus, IntegerLinearProgram};
use crate::math::optimization::simplex::{minimize, LinearProgram};
use crate::math::optimization::OptimizationConfig;
use std::error::Error;

pub struct BranchAndPriceSolver {
    max_iterations: usize,
    tolerance: f64,
    max_columns_per_node: usize,
}

impl BranchAndPriceSolver {
    pub fn new(max_iterations: usize, tolerance: f64, max_columns_per_node: usize) -> Self {
        Self {
            max_iterations,
            tolerance,
            max_columns_per_node,
        }
    }

    fn is_integer(&self, value: f64) -> bool {
        (value - value.round()).abs() < self.tolerance
    }

    fn solve_relaxation(
        &self,
        problem: &IntegerLinearProgram,
    ) -> Result<ILPSolution, Box<dyn Error>> {
        let original_objective = problem.objective.clone();
        let n = problem.objective.len();
        let m = problem.constraints.len();

        // Convert the maximization problem into a minimization one by negating the objective.
        let mut lp_objective: Vec<f64> = original_objective.iter().map(|x| -x).collect();
        lp_objective.extend(vec![0.0; m]);

        let lp_constraints: Vec<Vec<f64>> = problem
            .constraints
            .iter()
            .enumerate()
            .map(|(i, row)| {
                let mut new_row = row.clone();
                new_row.resize(n + m, 0.0);
                new_row[n + i] = 1.0;
                new_row
            })
            .collect();

        let lp = LinearProgram {
            objective: lp_objective,
            constraints: lp_constraints,
            rhs: problem.bounds.clone(),
        };

        let config = OptimizationConfig {
            max_iterations: self.max_iterations,
            tolerance: self.tolerance,
            learning_rate: 1.0,
        };
        let result = minimize(&lp, &config);

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

        // Retrieve the values corresponding to the original variables.
        let x = result.optimal_point[..n].to_vec();
        // Check feasibility against original constraints.
        for (i, constraint) in problem.constraints.iter().enumerate() {
            let lhs: f64 = constraint.iter().zip(&x).map(|(&a, &xi)| a * xi).sum();
            if lhs > problem.bounds[i] + self.tolerance {
                return Ok(ILPSolution {
                    values: vec![],
                    objective_value: f64::NEG_INFINITY,
                    status: ILPStatus::Infeasible,
                });
            }
        }

        // Re-negate the optimal value to convert back into a maximization result.
        Ok(ILPSolution {
            values: x,
            objective_value: -result.optimal_value,
            status: ILPStatus::Optimal,
        })
    }

    fn solve_pricing_problem(
        &self,
        dual_values: &[f64],
        problem: &IntegerLinearProgram,
    ) -> Option<Vec<f64>> {
        // In a real implementation, this would solve a problem-specific pricing problem
        // to find columns with negative reduced cost. Here we use a simple heuristic.
        let mut best_column = None;
        let mut best_reduced_cost = -self.tolerance;
        // Try unit vectors as potential columns.
        for i in 0..problem.constraints.len() {
            let mut new_column = vec![0.0; problem.constraints.len()];
            new_column[i] = 1.0;
            // Calculate reduced cost for this column.
            let mut reduced_cost = -1.0; // Assume unit contribution to objective.
            for (j, &dual) in dual_values.iter().enumerate() {
                reduced_cost += dual * new_column[j];
            }
            if reduced_cost < best_reduced_cost {
                // Verify the column satisfies all constraints.
                let mut feasible = true;
                for (constraint, &bound) in problem.constraints.iter().zip(problem.bounds.iter()) {
                    let lhs: f64 = constraint
                        .iter()
                        .zip(&new_column)
                        .map(|(&a, &x)| a * x)
                        .sum();
                    if lhs > bound + self.tolerance {
                        feasible = false;
                        break;
                    }
                }
                if feasible {
                    best_reduced_cost = reduced_cost;
                    best_column = Some(new_column);
                }
            }
        }
        best_column
    }

    fn generate_columns(&self, problem: &mut IntegerLinearProgram, solution: &ILPSolution) -> bool {
        if problem.objective.len() >= self.max_columns_per_node {
            return false;
        }
        let mut columns_added = 0;
        let mut dual_values = vec![0.0; problem.constraints.len()];
        // Calculate dual values from solution.
        for (i, constraint) in problem.constraints.iter().enumerate() {
            let lhs: f64 = constraint
                .iter()
                .zip(&solution.values)
                .map(|(&a, &x)| a * x)
                .sum();
            let slack = problem.bounds[i] - lhs;
            if slack.abs() < self.tolerance {
                dual_values[i] = if problem.bounds[i] > 0.0 { 1.0 } else { -1.0 };
            }
        }
        // Try to generate a single improving column.
        if let Some(new_column) = self.solve_pricing_problem(&dual_values, problem) {
            // Add new column to problem.
            for (i, constraint) in problem.constraints.iter_mut().enumerate() {
                constraint.push(new_column[i]);
            }
            // Calculate objective coefficient.
            let obj_coeff = new_column
                .iter()
                .zip(problem.objective.iter())
                .map(|(&a, &c)| a * c)
                .sum::<f64>();
            problem.objective.push(obj_coeff);
            problem.integer_vars.push(problem.objective.len() - 1);
            columns_added += 1;
        }
        columns_added > 0
    }

    fn branch(
        &self,
        problem: &IntegerLinearProgram,
        var_idx: usize,
        value: f64,
    ) -> (IntegerLinearProgram, IntegerLinearProgram) {
        let mut lower_branch = problem.clone();
        let mut upper_branch = problem.clone();
        // Add constraint x_i <= floor(value) to lower branch.
        let mut lower_constraint = vec![0.0; problem.objective.len()];
        lower_constraint[var_idx] = 1.0;
        lower_branch.constraints.push(lower_constraint);
        lower_branch.bounds.push(value.floor());
        // Add constraint x_i >= ceil(value) to upper branch.
        let mut upper_constraint = vec![0.0; problem.objective.len()];
        upper_constraint[var_idx] = -1.0; // Use -1.0 to convert >= to <=.
        upper_branch.constraints.push(upper_constraint);
        upper_branch.bounds.push(-value.ceil()); // Negate bound for <=.
        (lower_branch, upper_branch)
    }
}

impl ILPSolver for BranchAndPriceSolver {
    fn solve(&self, problem: &IntegerLinearProgram) -> Result<ILPSolution, Box<dyn Error>> {
        let mut stack = vec![problem.clone()];
        let mut iterations = 0;
        let mut best_objective = f64::NEG_INFINITY;
        let mut best_solution = None;

        while let Some(node) = stack.pop() {
            iterations += 1;
            if iterations > self.max_iterations {
                break;
            }

            // Solve LP relaxation on this node.
            let relaxation = match self.solve_relaxation(&node) {
                Ok(sol) if sol.status == ILPStatus::Optimal => sol,
                _ => continue,
            };

            // If the LP relaxation yields an integer solution for the original variables, return it.
            let mut solution_is_integer = true;
            for &i in &node.integer_vars {
                if (relaxation.values[i] - relaxation.values[i].round()).abs() > self.tolerance {
                    solution_is_integer = false;
                    break;
                }
            }
            if solution_is_integer {
                return Ok(relaxation);
            }

            // Generate additional columns if possible.
            let mut current_node = node.clone();
            if self.generate_columns(&mut current_node, &relaxation) {
                stack.push(current_node);
                continue;
            }

            // Prune if the relaxation's objective is not better than the best known.
            if relaxation.objective_value <= best_objective {
                continue;
            }

            // Branch on the most fractional variable.
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
            if !all_integer {
                if let Some((idx, val)) = most_fractional {
                    let (lower, upper) = self.branch(&node, idx, val);
                    stack.push(lower);
                    stack.push(upper);
                }
            }

            // Update best solution if applicable.
            if all_integer && relaxation.objective_value > best_objective {
                best_objective = relaxation.objective_value;
                best_solution = Some(relaxation);
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
        //   x + y ≤ 4
        //   x ≤ 2
        //   x, y ≥ 0 and integer
        let problem = IntegerLinearProgram {
            objective: vec![2.0, 1.0],
            constraints: vec![
                vec![1.0, 1.0],
                vec![1.0, 0.0],
                vec![-1.0, 0.0],
                vec![0.0, -1.0],
            ],
            bounds: vec![4.0, 2.0, 0.0, 0.0],
            integer_vars: vec![0, 1],
        };
        let solver = BranchAndPriceSolver::new(100, 1e-6, 5);
        let solution = solver.solve(&problem)?;
        assert_eq!(solution.status, ILPStatus::Optimal);
        // Optimal solution should be x=2, y=2, value of 6.
        assert!((solution.objective_value - 6.0).abs() < 1e-1);
        for &v in &solution.values {
            assert!((v - v.round()).abs() < 1e-6 && v >= 0.0);
        }
        Ok(())
    }

    #[test]
    fn test_infeasible_ilp() -> Result<(), Box<dyn Error>> {
        // maximize x + y
        // subject to:
        //   x + y ≤ 5
        //   x + y ≥ 6  (represented as -(x + y) ≤ -6)
        //   x, y ≥ 0 and integer
        let problem = IntegerLinearProgram {
            objective: vec![1.0, 1.0],
            constraints: vec![
                vec![1.0, 1.0],
                vec![-1.0, -1.0],
                vec![-1.0, 0.0],
                vec![0.0, -1.0],
            ],
            bounds: vec![5.0, -6.0, 0.0, 0.0],
            integer_vars: vec![0, 1],
        };
        let solver = BranchAndPriceSolver::new(100, 1e-6, 5);
        let solution = solver.solve(&problem)?;
        assert_eq!(solution.status, ILPStatus::Infeasible);
        Ok(())
    }
}
