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
        (value - value.round()).abs() < f64::max(self.tolerance, 1e-4)
    }

    fn solve_relaxation(
        &self,
        problem: &IntegerLinearProgram,
    ) -> Result<ILPSolution, Box<dyn Error>> {
        // First check for obviously conflicting constraints
        for i in 0..problem.constraints.len() {
            for j in i + 1..problem.constraints.len() {
                let c1 = &problem.constraints[i];
                let c2 = &problem.constraints[j];

                // Check if constraints are parallel (same direction)
                let parallel = c1
                    .iter()
                    .zip(c2.iter())
                    .all(|(&a, &b)| (a.abs() - b.abs()).abs() < self.tolerance);

                if parallel {
                    let b1 = problem.bounds[i];
                    let b2 = problem.bounds[j];
                    let is_c1_geq = c1.iter().any(|&x| x < 0.0);
                    let is_c2_geq = c2.iter().any(|&x| x < 0.0);

                    // If both are <= and lower bound > upper bound, infeasible
                    // If both are >= and upper bound < lower bound, infeasible
                    // If one is <= and one is >= and they conflict, infeasible
                    if (!is_c1_geq && !is_c2_geq && b1 < b2 - self.tolerance)
                        || (is_c1_geq && is_c2_geq && -b1 > -b2 + self.tolerance)
                        || (is_c1_geq != is_c2_geq && b1 < b2 - self.tolerance)
                    {
                        return Ok(ILPSolution {
                            values: vec![],
                            objective_value: f64::NEG_INFINITY,
                            status: ILPStatus::Infeasible,
                        });
                    }
                }
            }
        }

        // Convert constraints to standard form (Ax <= b)
        let mut std_constraints = Vec::new();
        let mut std_bounds = Vec::new();

        // Process each constraint
        for (i, constraint) in problem.constraints.iter().enumerate() {
            if constraint.iter().any(|&x| x < 0.0) {
                // This is a >= constraint (negative coefficients), negate it
                let negated: Vec<f64> = constraint.iter().map(|&x| -x).collect();
                std_constraints.push(negated);
                std_bounds.push(-problem.bounds[i]);
            } else {
                // This is a <= constraint, keep as is
                std_constraints.push(constraint.clone());
                std_bounds.push(problem.bounds[i]);
            }
        }

        let lp = LinearProgram {
            objective: problem.objective.iter().map(|x| -x).collect(),
            constraints: std_constraints,
            rhs: std_bounds,
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

        // Everything is feasible; result.optimal_value is already what we want
        Ok(ILPSolution {
            values: result.optimal_point.clone(),
            objective_value: -result.optimal_value,
            status: ILPStatus::Optimal,
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

        // Add constraint x_i >= ceil(value) to upper branch with conversion: -x_i <= -ceil(value)
        let mut upper_constraint = vec![0.0; problem.objective.len()];
        upper_constraint[var_idx] = -1.0;
        upper_branch.constraints.push(upper_constraint);
        upper_branch.bounds.push(-value.ceil());

        (lower_branch, upper_branch)
    }
}

impl ILPSolver for BranchAndCutSolver {
    fn solve(&self, problem: &IntegerLinearProgram) -> Result<ILPSolution, Box<dyn Error>> {
        let mut best_solution = None;
        let mut best_objective = f64::NEG_INFINITY;
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
                || relaxation.objective_value <= best_objective
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

        let solver = BranchAndCutSolver::new(100, 1e-6, 5);
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
