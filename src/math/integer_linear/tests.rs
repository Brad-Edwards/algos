use crate::math::integer_linear::{
    DantzigWolfeDecomposition,
    IntegerLinearProgram,
    ILPStatus,
    ILPSolver,
};

#[test]
fn test_dantzig_wolfe_simple() {
    // Simple maximization problem:
    // max x + y
    // s.t. x + y <= 5
    //      x, y >= 0
    //      x, y integer
    let problem = IntegerLinearProgram {
        objective: vec![1.0, 1.0],
        constraints: vec![
            vec![1.0, 1.0],  // x + y <= 5
            vec![-1.0, 0.0], // -x <= 0
            vec![0.0, -1.0], // -y <= 0
        ],
        bounds: vec![5.0, 0.0, 0.0],
        integer_vars: vec![0, 1],
    };

    let solver = DantzigWolfeDecomposition::new(100, 1e-6, 10);
    let result = solver.solve(&problem).unwrap();

    assert_eq!(result.status, ILPStatus::Optimal);
    assert!((result.objective_value - 5.0).abs() < 1e-6);
    
    // Check solution values
    assert!(result.values.len() == 2);
    let sum: f64 = result.values.iter().sum();
    assert!((sum - 5.0).abs() < 1e-6);
    
    // Check integrality
    for &value in &result.values {
        assert!((value - value.round()).abs() < 1e-6);
    }
}

#[test]
fn test_dantzig_wolfe_infeasible() {
    // Infeasible problem:
    // max x + y
    // s.t. x + y <= -1
    //      x, y >= 0
    let problem = IntegerLinearProgram {
        objective: vec![1.0, 1.0],
        constraints: vec![
            vec![1.0, 1.0],   // x + y <= -1
            vec![-1.0, 0.0],  // -x <= 0
            vec![0.0, -1.0],  // -y <= 0
        ],
        bounds: vec![-1.0, 0.0, 0.0],
        integer_vars: vec![0, 1],
    };

    let solver = DantzigWolfeDecomposition::new(100, 1e-6, 10);
    let result = solver.solve(&problem).unwrap();

    assert_eq!(result.status, ILPStatus::Infeasible);
} 