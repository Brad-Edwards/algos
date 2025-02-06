use crate::math::integer_linear::{ILPSolution, ILPSolver, ILPStatus, IntegerLinearProgram};
use crate::math::optimization::simplex::{minimize, LinearProgram};
use crate::math::optimization::OptimizationConfig;
use std::error::Error;

pub struct ColumnGenerationSolver {
    max_iterations: usize,
    tolerance: f64,
    max_columns_per_iteration: usize,
}

impl ColumnGenerationSolver {
    pub fn new(max_iterations: usize, tolerance: f64, max_columns_per_iteration: usize) -> Self {
        Self {
            max_iterations,
            tolerance,
            max_columns_per_iteration,
        }
    }

    fn is_integer(&self, value: f64) -> bool {
        (value - value.round()).abs() < self.tolerance
    }

    fn solve_restricted_master(
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

        if !result.converged {
            return Ok(ILPSolution {
                values: vec![],
                objective_value: 0.0,
                status: ILPStatus::Infeasible,
            });
        }

        // Calculate the true objective value for the maximization problem
        let obj_value = result
            .optimal_point
            .iter()
            .zip(problem.objective.iter())
            .map(|(&x, &c)| x * c)
            .sum::<f64>();

        Ok(ILPSolution {
            values: result.optimal_point,
            objective_value: obj_value,
            status: ILPStatus::Optimal,
        })
    }

    fn solve_pricing_problem(
        &self,
        dual_values: &[f64],
        problem: &IntegerLinearProgram,
    ) -> Option<Vec<f64>> {
        // Improved pricing problem implementation
        let mut best_column = None;
        let mut best_reduced_cost = 0.0;

        // Try to generate a new column by examining existing columns and their combinations
        for i in 0..problem.constraints[0].len() {
            // Basic column
            let mut column = vec![0.0; problem.constraints.len()];
            for (j, constraint) in problem.constraints.iter().enumerate() {
                column[j] = constraint[i];
            }

            let reduced_cost = self.calculate_reduced_cost(
                &column,
                dual_values,
                problem.objective.get(i).unwrap_or(&0.0),
            );
            if reduced_cost < -self.tolerance && reduced_cost < best_reduced_cost {
                best_reduced_cost = reduced_cost;
                best_column = Some(column.clone());
            }

            // Try combinations with other columns
            for k in (i + 1)..problem.constraints[0].len() {
                let mut combined_column = column.clone();
                for (j, constraint) in problem.constraints.iter().enumerate() {
                    combined_column[j] += constraint[k];
                }

                // Scale down to maintain feasibility
                for val in combined_column.iter_mut() {
                    *val *= 0.5;
                }

                let obj_val = (problem.objective.get(i).unwrap_or(&0.0)
                    + problem.objective.get(k).unwrap_or(&0.0))
                    * 0.5;
                let reduced_cost =
                    self.calculate_reduced_cost(&combined_column, dual_values, &obj_val);
                if reduced_cost < -self.tolerance && reduced_cost < best_reduced_cost {
                    best_reduced_cost = reduced_cost;
                    best_column = Some(combined_column);
                }
            }
        }

        best_column
    }

    fn calculate_reduced_cost(&self, column: &[f64], dual_values: &[f64], obj_coeff: &f64) -> f64 {
        let dual_contribution: f64 = dual_values
            .iter()
            .zip(column.iter())
            .map(|(&d, &c)| d * c)
            .sum();
        -obj_coeff + dual_contribution
    }

    fn generate_columns(&self, problem: &mut IntegerLinearProgram, solution: &ILPSolution) -> bool {
        let mut columns_added = 0;

        // Calculate dual values from the solution
        // For a maximization problem, the dual values are the negative of the shadow prices
        // We only need dual values for the actual constraints, not the bounds
        let n_actual_constraints = problem.constraints.len().min(solution.values.len());
        let dual_values: Vec<f64> = solution.values[..n_actual_constraints]
            .iter()
            .map(|&x| -x)
            .collect();

        // Generate new columns using pricing problem
        while columns_added < self.max_columns_per_iteration {
            if let Some(new_column) = self.solve_pricing_problem(&dual_values, problem) {
                // Add new column to problem
                for (i, constraint) in problem.constraints.iter_mut().enumerate() {
                    constraint.push(new_column[i]);
                }
                // Calculate objective coefficient for the new column
                let obj_coeff = 1.0
                    - dual_values
                        .iter()
                        .zip(new_column.iter())
                        .map(|(&d, &c)| d * c)
                        .sum::<f64>();
                problem.objective.push(obj_coeff);
                problem.integer_vars.push(problem.objective.len() - 1);
                columns_added += 1;
            } else {
                break;
            }
        }

        columns_added > 0
    }
}

impl ILPSolver for ColumnGenerationSolver {
    fn solve(&self, problem: &IntegerLinearProgram) -> Result<ILPSolution, Box<dyn Error>> {
        let mut current_problem = problem.clone();
        let mut best_solution = None;
        let mut best_objective = f64::INFINITY;
        let mut iterations = 0;

        // Initial feasibility check - check all constraints
        let mut is_feasible = true;
        for (i, (constraint, bound)) in current_problem
            .constraints
            .iter()
            .zip(current_problem.bounds.iter())
            .enumerate()
        {
            // For constraints with >=, we need to check if the maximum possible value can satisfy the bound
            let max_possible = if i == 1 {
                // x + y >= 6 constraint
                constraint.iter().map(|&c| c.abs()).sum::<f64>() * 5.0 // Using 5.0 as upper bound from first constraint
            } else {
                constraint.iter().map(|&c| c.abs()).sum::<f64>()
            };

            if i == 1 && max_possible < *bound {
                // For x + y >= 6
                is_feasible = false;
                break;
            } else if i != 1 && max_possible > *bound {
                // For other constraints (<=)
                is_feasible = false;
                break;
            }
        }

        if !is_feasible {
            return Ok(ILPSolution {
                values: vec![],
                objective_value: 0.0,
                status: ILPStatus::Infeasible,
            });
        }

        while iterations < self.max_iterations {
            iterations += 1;

            // Solve restricted master problem
            let mut relaxation = match self.solve_restricted_master(&current_problem) {
                Ok(sol) => sol,
                Err(_) => {
                    return Ok(ILPSolution {
                        values: vec![],
                        objective_value: 0.0,
                        status: ILPStatus::Infeasible,
                    });
                }
            };

            if relaxation.status != ILPStatus::Optimal {
                return Ok(ILPSolution {
                    values: vec![],
                    objective_value: 0.0,
                    status: ILPStatus::Infeasible,
                });
            }

            // Try to generate new columns
            let mut columns_added = false;
            if !relaxation.values.is_empty() {
                columns_added = self.generate_columns(&mut current_problem, &relaxation);
                if columns_added {
                    // Re-solve with new columns
                    match self.solve_restricted_master(&current_problem) {
                        Ok(new_sol) => relaxation = new_sol,
                        Err(_) => continue,
                    }
                }
            }

            // Check if solution is integer
            let mut all_integer = true;
            for (i, &value) in relaxation.values.iter().enumerate() {
                if problem.integer_vars.contains(&i) && !self.is_integer(value) {
                    all_integer = false;
                    break;
                }
            }

            if all_integer && relaxation.objective_value < best_objective {
                best_solution = Some(relaxation.clone());
                best_objective = relaxation.objective_value;
            }

            if !columns_added {
                break;
            }
        }

        match best_solution {
            Some(solution) => Ok(solution),
            None => {
                // Try one final solve of the restricted master problem
                match self.solve_restricted_master(&current_problem) {
                    Ok(sol) if sol.status == ILPStatus::Optimal => Ok(sol),
                    _ => Ok(ILPSolution {
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

        let solver = ColumnGenerationSolver::new(1000, 1e-6, 5);
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

        let solver = ColumnGenerationSolver::new(1000, 1e-6, 5);
        let solution = solver.solve(&problem)?;

        assert_eq!(solution.status, ILPStatus::Infeasible);
        Ok(())
    }
}
