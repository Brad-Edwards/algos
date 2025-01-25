/// lib.rs

/// A basic matrix implementation using `Vec<f64>`.
#[derive(Debug, Clone)]
pub struct Matrix {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f64>,
}

impl Matrix {
    /// Creates a new matrix with given dimensions and data.
    ///
    /// # Panics
    ///
    /// Panics if `data.len() != rows * cols`.
    pub fn new(rows: usize, cols: usize, data: Vec<f64>) -> Self {
        assert_eq!(data.len(), rows * cols);
        Self { rows, cols, data }
    }

    /// Multiplies two matrices if their dimensions are compatible.
    ///
    /// Returns an error if the matrices cannot be multiplied.
    pub fn multiply(&self, other: &Matrix) -> Result<Matrix, String> {
        if self.cols != other.rows {
            return Err("Incompatible dimensions for multiplication".into());
        }

        let mut result = vec![0.0; self.rows * other.cols];
        for i in 0..self.rows {
            for j in 0..other.cols {
                let mut sum = 0.0;
                for k in 0..self.cols {
                    sum += self.data[i * self.cols + k] * other.data[k * other.cols + j];
                }
                result[i * other.cols + j] = sum;
            }
        }

        Ok(Matrix {
            rows: self.rows,
            cols: other.cols,
            data: result,
        })
    }
}

/// Computes the optimal parenthesization for a chain of matrices based on their dimensions.
///
/// `dims` is a slice where `dims[i]` is the number of rows of matrix `i`,
/// and `dims[i+1]` is the number of columns of matrix `i`.
/// So for `n` matrices, `dims.len() == n + 1`.
///
/// Returns two tables:
/// 1. `m`: Cost table where `m[i][j]` holds the minimum cost for multiplying matrices i..j.
/// 2. `s`: Split table where `s[i][j]` indicates the index at which the optimal split occurs.
fn matrix_chain_order(dims: &[usize]) -> (Vec<Vec<usize>>, Vec<Vec<usize>>) {
    let n = dims.len() - 1; // number of matrices
    let mut m = vec![vec![0; n]; n];
    let mut s = vec![vec![0; n]; n];

    // chain_length is the length of the chain of matrices being considered.
    for chain_length in 2..=n {
        for i in 0..=n - chain_length {
            let j = i + chain_length - 1;
            m[i][j] = usize::MAX;
            for k in i..j {
                let q = m[i][k]
                    + m[k + 1][j]
                    + dims[i] * dims[k + 1] * dims[j + 1];
                if q < m[i][j] {
                    m[i][j] = q;
                    s[i][j] = k;
                }
            }
        }
    }

    (m, s)
}

/// Recursively reconstructs the optimal multiplication order as indices.
/// This is an internal utility function.
fn construct_optimal_parens(s: &Vec<Vec<usize>>, i: usize, j: usize) -> Vec<(usize, usize)> {
    // If there's only one matrix, no split needed
    if i == j {
        return vec![(i, i)];
    }
    let k = s[i][j];
    let left = construct_optimal_parens(s, i, k);
    let right = construct_optimal_parens(s, k + 1, j);
    [left, right].concat()
}

/// Multiplies a slice of matrices using the optimal parenthesization determined by dynamic programming.
///
/// Returns an error if dimensions don't match at any step.
///
/// # Example
///
/// ```
/// // Suppose you have 3 matrices:
/// // A: 10 x 20, B: 20 x 5, C: 5 x 15
/// // dims = [10, 20, 5, 15]
/// // The best order is typically (A x (B x C)) or (A x B) x C, depending on costs.
/// // This function will compute that order automatically.
/// ```
pub fn optimal_matrix_chain_multiplication(matrices: &[Matrix]) -> Result<Matrix, String> {
    if matrices.is_empty() {
        return Err("No matrices provided".to_string());
    }
    if matrices.len() == 1 {
        return Ok(matrices[0].clone());
    }

    // Build dimension array
    let mut dims = Vec::with_capacity(matrices.len() + 1);
    dims.push(matrices[0].rows);
    for mat in matrices.iter() {
        if dims.last().unwrap() != &mat.rows {
            return Err("Dimension mismatch in consecutive matrices".into());
        }
        dims.push(mat.cols);
    }

    let n = matrices.len();
    let (_, s) = matrix_chain_order(&dims);

    // Reconstruct multiplication order
    // Example: s = split table. Then we walk it to find the actual multiplication steps
    let order = construct_optimal_parens(&s, 0, n - 1);

    // The `order` slice effectively lists each matrix as a segment [start, end].
    // We'll multiply from left to right in that list (though it's a bit more nuanced).
    // A simpler approach is to do a top-down simulation of the DP splits.
    multiply_chain_rec(&matrices, &s, 0, n - 1)
}

/// Recursively multiply sub-chains based on split table.
fn multiply_chain_rec(
    matrices: &[Matrix],
    s: &Vec<Vec<usize>>,
    i: usize,
    j: usize,
) -> Result<Matrix, String> {
    if i == j {
        return Ok(matrices[i].clone());
    }
    let k = s[i][j];
    let left = multiply_chain_rec(matrices, s, i, k)?;
    let right = multiply_chain_rec(matrices, s, k + 1, j)?;
    left.multiply(&right)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_optimal_matrix_chain_multiplication() {
        // Example matrices with dimensions:
        // A: 10 x 30
        // B: 30 x 5
        // C: 5 x 60
        // dims = [10, 30, 5, 60]
        // Minimum cost parenthesization is A(BC) with cost 10*30*5 + 10*5*60 + 30*5*60 (or computed by DP)
        let a = Matrix::new(10, 30, vec![1.0; 10 * 30]);
        let b = Matrix::new(30, 5, vec![1.0; 30 * 5]);
        let c = Matrix::new(5, 60, vec![1.0; 5 * 60]);

        let matrices = vec![a, b, c];
        let result = optimal_matrix_chain_multiplication(&matrices);

        assert!(result.is_ok());
        let res_matrix = result.unwrap();
        // The final matrix after multiplication should have dimensions 10 x 60
        assert_eq!(res_matrix.rows, 10);
        assert_eq!(res_matrix.cols, 60);
    }
}
