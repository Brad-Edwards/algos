use crate::math::integer_linear::{ILPSolution, ILPSolver, ILPStatus, IntegerLinearProgram};
use crate::math::optimization::simplex::{minimize, LinearProgram};
use crate::math::optimization::OptimizationConfig;
use std::error::Error;

pub struct BranchAndCutSolver {
    max_iterations: usize,
    tolerance: f64,
    max_cuts_per_node: usize,
}

impl BranchAndCutSolver {
    pub fn new(max_iterations: usize, tolerance: f64, max_cuts_per_node: usize) -> Self {
        Self {
            max_iterations,
            tolerance,
            max_cuts_per_node,
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

    fn generate_gomory_cut(
        &self,
        tableau: &[Vec<f64>],
        basic_var: usize,
    ) -> Option<(Vec<f64>, f64)> {
        let n = tableau[0].len() - 1;
        let mut cut = vec![0.0; n - 1];
        let mut rhs = 0.0;

        // Extract the row corresponding to the basic variable
        let row = &tableau[basic_var];

        // Generate cut coefficients
        for j in 0..n - 1 {
            let frac_part = row[j] - row[j].floor();
            if frac_part > self.tolerance {
                cut[j] = frac_part;
            }
        }

        // Calculate RHS
        let frac_part = row[n - 1] - row[n - 1].floor();
        if frac_part > self.tolerance {
            rhs = frac_part;
        }

        // Return cut if non-trivial
        if cut.iter().any(|&x| x.abs() > self.tolerance) {
            Some((cut, rhs))
        } else {
            None
        }
    }

    fn add_cuts(&self, problem: &mut IntegerLinearProgram, solution: &ILPSolution) -> bool {
        let mut cuts_added = 0;
        let mut tableau = vec![vec![0.0; problem.objective.len() + 1]];

        // Build initial tableau from solution
        tableau[0][problem.objective.len()] = solution.objective_value;
        for (i, &val) in solution.values.iter().enumerate() {
            tableau[0][i] = val;
        }

        // Try to generate cuts for each fractional variable
        for (i, &value) in solution.values.iter().enumerate() {
            if problem.integer_vars.contains(&i) && !self.is_integer(value) {
                if let Some((cut, rhs)) = self.generate_gomory_cut(&tableau, i) {
                    problem.constraints.push(cut);
                    problem.bounds.push(rhs);
                    cuts_added += 1;

                    if cuts_added >= self.max_cuts_per_node {
                        break;
                    }
                }
            }
        }

        cuts_added > 0
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

impl ILPSolver for BranchAndCutSolver {
    fn solve(&self, problem: &IntegerLinearProgram) -> Result<ILPSolution, Box<dyn Error>> {
        let mut best_solution = None;
        let mut best_objective = f64::INFINITY;
        let mut nodes = vec![problem.clone()];
        let mut iterations = 0;

        while !nodes.is_empty() && iterations < self.max_iterations {
            iterations += 1;
            let mut current = nodes.pop().unwrap();

            // Solve LP relaxation
            let mut relaxation = match self.solve_relaxation(&current) {
                Ok(sol) => sol,
                Err(_) => continue,
            };

            // Check if solution is worse than best known
            if relaxation.status != ILPStatus::Optimal
                || relaxation.objective_value >= best_objective
            {
                continue;
            }

            // Try to add cutting planes
            let mut cuts_added = false;
            if !relaxation.values.is_empty() {
                cuts_added = self.add_cuts(&mut current, &relaxation);
                if cuts_added {
                    // Re-solve with new cuts
                    match self.solve_relaxation(&current) {
                        Ok(new_sol) => relaxation = new_sol,
                        Err(_) => continue,
                    }
                }
            }

            // If no cuts were added or solution is still fractional, branch
            if !cuts_added {
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
            } else {
                // If cuts were added, continue exploring this node
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

        let solver = BranchAndCutSolver::new(1000, 1e-6, 5);
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

        let solver = BranchAndCutSolver::new(1000, 1e-6, 5);
        let solution = solver.solve(&problem)?;

        assert_eq!(solution.status, ILPStatus::Infeasible);
        Ok(())
    }
}
