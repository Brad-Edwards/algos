use crate::math::integer_linear::{ILPSolution, ILPSolver, ILPStatus, IntegerLinearProgram};
use crate::math::optimization::simplex::{minimize, LinearProgram};
use crate::math::optimization::OptimizationConfig;
use std::error::Error;

pub struct MixedIntegerRoundingCuts {
    max_iterations: usize,
    tolerance: f64,
    max_cuts_per_iteration: usize,
}

impl MixedIntegerRoundingCuts {
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

    fn generate_mir_cut(
        &self,
        row: &[f64],
        rhs: f64,
        integer_vars: &[usize],
    ) -> Option<(Vec<f64>, f64)> {
        // Find the fractional part of the RHS
        let f0 = rhs - rhs.floor();
        if f0 < self.tolerance || (1.0 - f0) < self.tolerance {
            return None;
        }

        let mut cut = vec![0.0; row.len()];

        // Process integer variables
        for (j, &coeff) in row.iter().enumerate() {
            if integer_vars.contains(&j) {
                let fj = coeff - coeff.floor();
                if fj <= f0 {
                    cut[j] = fj / f0;
                } else {
                    cut[j] = (1.0 - fj) / (1.0 - f0);
                }
            } else {
                // Continuous variables
                if coeff > 0.0 {
                    cut[j] = coeff / f0;
                } else if coeff < 0.0 {
                    cut[j] = coeff / (1.0 - f0);
                }
            }
        }

        // Return cut if non-trivial
        if cut.iter().any(|&x| x.abs() > self.tolerance) {
            Some((cut, 1.0))
        } else {
            None
        }
    }

    fn add_cuts(&self, problem: &mut IntegerLinearProgram) -> bool {
        let mut cuts_added = 0;
        let n = problem.constraints.len();
        let mut new_cuts = Vec::new();
        let mut new_bounds = Vec::new();

        // Try to generate MIR cuts from each constraint
        for i in 0..n {
            if let Some((cut, rhs)) = self.generate_mir_cut(
                &problem.constraints[i],
                problem.bounds[i],
                &problem.integer_vars,
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

impl ILPSolver for MixedIntegerRoundingCuts {
    fn solve(&self, problem: &IntegerLinearProgram) -> Result<ILPSolution, Box<dyn Error>> {
        let mut current_problem = problem.clone();
        let mut best_solution = None;
        let mut best_objective = f64::NEG_INFINITY;
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
                if relaxation.objective_value > best_objective {
                    best_solution = Some(relaxation.clone());
                    best_objective = relaxation.objective_value;
                }
            }

            // Add MIR cuts
            if !self.add_cuts(&mut current_problem) {
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

        let solver = MixedIntegerRoundingCuts::new(1000, 1e-6, 5);
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

        let solver = MixedIntegerRoundingCuts::new(1000, 1e-6, 5);
        let solution = solver.solve(&problem)?;

        assert_eq!(solution.status, ILPStatus::Infeasible);
        Ok(())
    }
}
