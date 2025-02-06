use crate::math::integer_linear::{ILPSolution, ILPSolver, ILPStatus, IntegerLinearProgram};
use crate::math::optimization::simplex::{minimize, LinearProgram};
use crate::math::optimization::OptimizationConfig;
use std::error::Error;

pub struct GomoryCuttingPlanes {
    max_iterations: usize,
    tolerance: f64,
    max_cuts_per_iteration: usize,
}

impl GomoryCuttingPlanes {
    pub fn new(max_iterations: usize, tolerance: f64, max_cuts_per_iteration: usize) -> Self {
        Self {
            max_iterations,
            tolerance,
            max_cuts_per_iteration,
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
        solution: &[f64],
        row: &[f64],
        rhs: f64,
    ) -> Option<(Vec<f64>, f64)> {
        // Find a basic variable with fractional value
        let mut fractional_idx = None;
        for (i, &value) in solution.iter().enumerate() {
            if !self.is_integer(value) {
                fractional_idx = Some(i);
                break;
            }
        }

        if fractional_idx.is_some() {
            // Generate Gomory cut coefficients
            let mut cut = vec![0.0; row.len()];
            let mut cut_rhs = 0.0;

            for (i, &coeff) in row.iter().enumerate() {
                let frac_part = coeff - coeff.floor();
                if frac_part > self.tolerance {
                    cut[i] = frac_part;
                }
            }

            let rhs_frac = rhs - rhs.floor();
            if rhs_frac > self.tolerance {
                cut_rhs = rhs_frac;
            }

            // Return cut if non-trivial
            if cut.iter().any(|&x| x.abs() > self.tolerance) {
                Some((cut, cut_rhs))
            } else {
                None
            }
        } else {
            None
        }
    }

    fn add_cuts(&self, problem: &mut IntegerLinearProgram, solution: &ILPSolution) -> bool {
        let mut cuts_added = 0;
        let n = problem.constraints.len();
        let mut new_cuts = Vec::new();
        let mut new_bounds = Vec::new();

        // Try to generate cuts from each constraint
        for i in 0..n {
            if let Some((cut, rhs)) = self.generate_gomory_cut(
                &solution.values,
                &problem.constraints[i],
                problem.bounds[i],
            ) {
                new_cuts.push(cut);
                new_bounds.push(rhs);
                cuts_added += 1;

                if cuts_added >= self.max_cuts_per_iteration {
                    break;
                }
            }
        }

        // Add all generated cuts at once
        problem.constraints.extend(new_cuts);
        problem.bounds.extend(new_bounds);

        cuts_added > 0
    }
}

impl ILPSolver for GomoryCuttingPlanes {
    fn solve(&self, problem: &IntegerLinearProgram) -> Result<ILPSolution, Box<dyn Error>> {
        let mut current_problem = problem.clone();
        let mut best_solution = None;
        let mut best_objective = f64::INFINITY;
        let mut iterations = 0;

        while iterations < self.max_iterations {
            iterations += 1;

            // Solve LP relaxation
            let relaxation = match self.solve_relaxation(&current_problem) {
                Ok(sol) => sol,
                Err(_) => break,
            };

            if relaxation.status != ILPStatus::Optimal {
                break;
            }

            // Check if solution is integer
            let mut all_integer = true;
            for (i, &value) in relaxation.values.iter().enumerate() {
                if problem.integer_vars.contains(&i) && !self.is_integer(value) {
                    all_integer = false;
                    break;
                }
            }

            if all_integer {
                if relaxation.objective_value < best_objective {
                    best_solution = Some(relaxation.clone());
                    best_objective = relaxation.objective_value;
                }
            }

            // Add Gomory cuts
            if !self.add_cuts(&mut current_problem, &relaxation) {
                // No more cuts can be generated
                break;
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

        let solver = GomoryCuttingPlanes::new(1000, 1e-6, 5);
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

        let solver = GomoryCuttingPlanes::new(1000, 1e-6, 5);
        let solution = solver.solve(&problem)?;

        assert_eq!(solution.status, ILPStatus::Infeasible);
        Ok(())
    }
}
