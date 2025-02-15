use std::error::Error;

pub mod benders;
pub mod branch_and_bound;
pub mod branch_and_cut;
pub mod branch_and_price;
pub mod branch_and_reduce;
pub mod column_generation;
pub mod dantzig_wolfe;
pub mod gomory;
pub mod lift_and_project;
pub mod mixed_integer_rounding;

#[derive(Debug, Clone)]
pub struct IntegerLinearProgram {
    pub objective: Vec<f64>,
    pub constraints: Vec<Vec<f64>>,
    pub bounds: Vec<f64>,
    pub integer_vars: Vec<usize>,
}

#[derive(Debug, Clone)]
pub struct ILPSolution {
    pub values: Vec<f64>,
    pub objective_value: f64,
    pub status: ILPStatus,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ILPStatus {
    Optimal,
    Infeasible,
    Unbounded,
    MaxIterationsReached,
}

pub trait ILPSolver {
    fn solve(&self, problem: &IntegerLinearProgram) -> Result<ILPSolution, Box<dyn Error>>;
}

pub use benders::BendersDecomposition;
pub use branch_and_bound::BranchAndBoundSolver;
pub use branch_and_cut::BranchAndCutSolver;
pub use branch_and_price::BranchAndPriceSolver;
pub use branch_and_reduce::BranchAndReduceSolver;
pub use column_generation::ColumnGenerationSolver;
pub use dantzig_wolfe::DantzigWolfeDecomposition;
pub use gomory::GomoryCuttingPlanes;
pub use lift_and_project::LiftAndProjectCuts;
pub use mixed_integer_rounding::MixedIntegerRoundingCuts;
