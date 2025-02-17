// Mixed Integer Rounding Cuts Implementation
//
// This module provides a reference implementation of mixed integer rounding (MIR) cuts,
// which are used in integer linear programming to derive valid inequalities from fractional LP solutions.
//
// Given an inequality of the form:
//     ∑ a_j * x_j <= b
// where b is fractional, the MIR cut is constructed by computing:
//     r = fractional part of b (i.e. b - floor(b)).
// Then, for each coefficient a_j, the MIR coefficient is:
//     a'_j = min( fractional part of a_j, r )
// yielding the inequality:
//     ∑ a'_j * x_j <= r.
//
// This implementation assumes that x_j >= 0 and that the input inequality has been normalized.

pub struct MixedIntegerRoundingCuts;

impl MixedIntegerRoundingCuts {
    /// Computes the Mixed Integer Rounding (MIR) cut for a given inequality.
    ///
    /// # Arguments
    ///
    /// * `coeffs` - Slice of coefficients of the inequality.
    /// * `rhs` - Right-hand side of the inequality.
    ///
    /// # Returns
    ///
    /// Returns `Some((cut_coeffs, r))` where `cut_coeffs` is the vector of coefficients for the MIR cut
    /// and `r` is the fractional part of `rhs`. Returns `None` if `rhs` is an integer,
    /// meaning that no MIR cut is applicable.
    ///
    /// # Examples
    ///
    /// ```
    /// use algos::integer_linear::MixedIntegerRoundingCuts;
    /// let coeffs = vec![1.2, 3.7, 4.3];
    /// let rhs = 7.5; // fractional part r = 0.5
    /// let result = MixedIntegerRoundingCuts::compute(&coeffs, rhs);
    /// assert_eq!(result, Some((vec![0.2, 0.5, 0.3], 0.5)));
    /// ```
    pub fn compute(coeffs: &[f64], rhs: f64) -> Option<(Vec<f64>, f64)> {
        let r = rhs.fract();
        if r == 0.0 {
            return None;
        }
        let cut_coeffs: Vec<f64> = coeffs.iter().map(|&a| {
            let frac_a = a.fract();
            if frac_a < r { frac_a } else { r }
        }).collect();
        Some((cut_coeffs, r))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_approx_eq(expected: f64, actual: f64) {
        let tol = 1e-12;
        assert!((expected - actual).abs() < tol, "expected: {}, got: {}", expected, actual);
    }

    fn assert_approx_eq_vec(expected: &[f64], actual: &[f64]) {
        assert_eq!(expected.len(), actual.len(), "Vectors have different lengths");
        for (&e, &a) in expected.iter().zip(actual.iter()) {
            assert_approx_eq(e, a);
        }
    }

    #[test]
    fn test_no_cut_when_rhs_integer() {
        let coeffs = vec![2.0, 3.5, 4.2];
        let rhs = 10.0;
        assert_eq!(MixedIntegerRoundingCuts::compute(&coeffs, rhs), None);
    }

    #[test]
    fn test_basic_mir_cut() {
        let coeffs = vec![1.2, 3.7, 4.3];
        let rhs = 7.5; // fractional part r = 0.5
        let result = MixedIntegerRoundingCuts::compute(&coeffs, rhs);
        assert!(result.is_some());
        let (cut, r) = result.unwrap();
        let expected_cut = vec![0.2, 0.5, 0.3];
        assert_approx_eq_vec(&expected_cut, &cut);
        assert_approx_eq(0.5, r);
    }

    #[test]
    fn test_edge_case() {
        let coeffs = vec![2.0, 3.99, 4.50];
        let rhs = 8.3; // fractional part r = 0.3
        let result = MixedIntegerRoundingCuts::compute(&coeffs, rhs);
        assert!(result.is_some());
        let (cut, r) = result.unwrap();
        // For 2.0, fractional part = 0.0;
        // For 3.99, fractional part = 0.99 -> min(0.99, 0.3) = 0.3;
        // For 4.50, fractional part = 0.5 -> min(0.5, 0.3) = 0.3.
        let expected_cut = vec![0.0, 0.3, 0.3];
        assert_approx_eq_vec(&expected_cut, &cut);
        assert_approx_eq(0.3, r);
    }
}
