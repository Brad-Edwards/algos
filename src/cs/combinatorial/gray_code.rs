/// Generate an n-bit Gray code sequence using the standard i ^ (i >> 1) method.
/// Returns a Vec of length 2^n, where each element is the Gray code for i in [0 .. 2^n).
///
/// # Example
/// ```
/// use algos::cs::combinatorial::gray_code;
///
/// let codes = gray_code(2);
/// // For n=2, the sequence is [0, 1, 3, 2].
/// assert_eq!(codes, vec![0, 1, 3, 2]);
/// ```

/// Returns a vector of 2^n Gray codes, each stored as a `u64`.
pub fn gray_code(n: usize) -> Vec<u64> {
    let size = 1 << n;
    let mut result = Vec::with_capacity(size);
    for i in 0..size {
        // Standard formula: GrayCode(i) = i ^ (i >> 1)
        let gray = (i ^ (i >> 1)) as u64;
        result.push(gray);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::gray_code;

    #[test]
    fn test_gray_code_n2() {
        let codes = gray_code(2);
        assert_eq!(codes, vec![0, 1, 3, 2]);
    }

    #[test]
    fn test_gray_code_n3() {
        let codes = gray_code(3);
        // The classic n=3 sequence: 0,1,3,2,6,7,5,4
        assert_eq!(codes, vec![0, 1, 3, 2, 6, 7, 5, 4]);
    }

    #[test]
    fn test_gray_code_n0() {
        // 2^0 = 1 code: [0]
        let codes = gray_code(0);
        assert_eq!(codes, vec![0]);
    }
}
