use crate::math::integer_linear::{ILPSolution, ILPSolver, ILPStatus, IntegerLinearProgram};
use crate::math::optimization::simplex::{minimize, LinearProgram};
use crate::math::optimization::OptimizationConfig;
use std::error::Error;

// Helper function to perform brute force rounding of an LP solution to a feasible
// integer solution by enumerating floor and ceiling choices for integer variables.
fn brute_force_round(
    solution: &[f64],
    problem: &IntegerLinearProgram,
    tolerance: f64,
) -> Option<Vec<f64>> {
    let mut best_solution = None;
    let mut best_obj = -f64::INFINITY;
    let int_indices = &problem.integer_vars;
    let base_solution = solution.to_vec();
    let k = int_indices.len();
    let choices = 1 << k; // 2^k combinations
    for mask in 0..choices {
        let mut candidate = base_solution.clone();
        for (j, &i) in int_indices.iter().enumerate() {
            let floor_val = solution[i].floor();
            let ceil_val = solution[i].ceil();
            if (solution[i] - floor_val).abs() < tolerance {
                candidate[i] = floor_val;
            } else if (ceil_val - solution[i]).abs() < tolerance {
                candidate[i] = ceil_val;
            } else if ((mask >> j) & 1) == 0 {
                candidate[i] = floor_val;
            } else {
                candidate[i] = ceil_val;
            }
        }
        // Check feasibility of candidate: each constraint dot(candidate) <= bound (+ tolerance)
        let mut feasible = true;
        for (i, constraint) in problem.constraints.iter().enumerate() {
            let dot: f64 = constraint
                .iter()
                .zip(candidate.iter())
                .map(|(&a, &x)| a * x)
                .sum();
            if dot > problem.bounds[i] + tolerance {
                feasible = false;
                break;
            }
        }
        // Also enforce non-negativity
        for &x in candidate.iter() {
            if x < -tolerance {
                feasible = false;
                break;
            }
        }
        if !feasible {
            continue;
        }
        // Compute objective value.
        let obj: f64 = problem
            .objective
            .iter()
            .zip(candidate.iter())
            .map(|(&c, &x)| c * x)
            .sum();
        if obj > best_obj {
            best_obj = obj;
            best_solution = Some(candidate);
        }
    }
    best_solution
}

pub struct DantzigWolfeDecomposition {
    max_iterations: usize,
    tolerance: f64,
}

impl DantzigWolfeDecomposition {
    pub fn new(max_iterations: usize, tolerance: f64) -> Self {
        Self {
            max_iterations,
            tolerance,
        }
    }
}

impl ILPSolver for DantzigWolfeDecomposition {
    fn solve(&self, problem: &IntegerLinearProgram) -> Result<ILPSolution, Box<dyn Error>> {
        println!("Dantzig-Wolfe: Starting to solve ILP");
        println!("Original problem:");
        println!("  Objective: {:?}", problem.objective);
        println!("  Constraints: {:?}", problem.constraints);
        println!("  Bounds: {:?}", problem.bounds);
        println!("  Integer variables: {:?}", problem.integer_vars);

        let lp = LinearProgram {
            objective: problem.objective.iter().map(|&c| -c).collect(),
            constraints: problem.constraints.clone(),
            rhs: problem.bounds.clone(),
        };

        println!("\nSolving LP relaxation:");
        println!("  Minimization objective: {:?}", lp.objective);
        println!("  LP constraints: {:?}", lp.constraints);
        println!("  LP bounds: {:?}", lp.rhs);

        let config = OptimizationConfig {
            max_iterations: self.max_iterations,
            tolerance: self.tolerance,
            learning_rate: 1.0,
        };

        let result = minimize(&lp, &config);

        println!("\nLP solution results:");
        println!("  Converged: {}", result.converged);
        println!("  Iterations: {}", result.iterations);
        println!("  Optimal value: {}", -result.optimal_value);
        println!("  Solution point: {:?}", result.optimal_point);

        if !result.converged {
            println!("LP relaxation failed to converge");
            return Ok(ILPSolution {
                values: vec![],
                objective_value: 0.0,
                status: ILPStatus::Infeasible,
            });
        }

        if result.optimal_point.is_empty() {
            println!("LP solution is empty");
            return Ok(ILPSolution {
                values: vec![],
                objective_value: 0.0,
                status: ILPStatus::Infeasible,
            });
        }

        let n = problem.objective.len();
        let mut solution = vec![0.0; n];
        for (i, value) in solution.iter_mut().enumerate().take(n) {
            if i < result.optimal_point.len() {
                *value = result.optimal_point[i];
                println!("Variable {}: {}", i, value);
            }
        }

        println!("\nRounding solution to integer values using brute force:");
        if let Some(rounded_solution) = brute_force_round(&solution, problem, self.tolerance) {
            println!("Rounded solution: {:?}", rounded_solution);
            let obj_value: f64 = problem
                .objective
                .iter()
                .zip(rounded_solution.iter())
                .map(|(&c, &x)| c * x)
                .sum();
            println!("\nFinal objective value: {}", obj_value);
            Ok(ILPSolution {
                values: rounded_solution,
                objective_value: obj_value,
                status: ILPStatus::Optimal,
            })
        } else {
            println!("\nNo feasible integer rounding found.");
            Ok(ILPSolution {
                values: vec![],
                objective_value: 0.0,
                status: ILPStatus::Infeasible,
            })
        }
    }
}
