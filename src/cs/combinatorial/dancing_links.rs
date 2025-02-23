/// A minimal Dancing Links (Algorithm X) reference implementation in Rust.
/// This solves the "Exact Cover" problem by selecting rows so that every
/// column is covered exactly once. Knuth's Algorithm X + Dancing Links
/// efficiently backtracks through solutions.
///
/// # Example
/// ```
/// // In this small example, we have 4 columns (C0..C3) and 6 possible rows.
/// // Each row says which columns it covers with '1'.
/// // We want to find all ways to cover every column exactly once.
/// // Matrix rows (R0..R5):
/// //   R0: [1, 0, 1, 0]
/// //   R1: [1, 0, 0, 1]
/// //   R2: [0, 1, 1, 0]
/// //   R3: [0, 1, 0, 1]
/// //   R4: [1, 0, 1, 0]  (same as R0, just to illustrate multiple coverage)
/// //   R5: [0, 1, 0, 1]  (same as R3)
/// //
/// // The columns are "exactly covered" by choosing e.g. R1 + R2 or R0 + R3, etc.
/// use algos::cs::combinatorial::dancing_links::DancingLinks;
///
/// let matrix = vec![
///     vec![true,  false, true,  false], // R0
///     vec![true,  false, false, true ], // R1
///     vec![false, true,  true,  false], // R2
///     vec![false, true,  false, true ], // R3
///     vec![true,  false, true,  false], // R4
///     vec![false, true,  false, true ], // R5
/// ];
///
/// // Build the DLX structure and search for solutions.
/// let mut dlx = DancingLinks::new(&matrix);
/// let solutions = dlx.solve_all(); // Each solution is a set of row indices
///
/// // Print solutions (order of solutions/rows may vary).
/// // e.g. one possible solution is [1, 2] which covers all columns.
/// println!("{:?}", solutions);
/// ```

/// Each dancing-links node links up/down/left/right in a circular list.
#[derive(Clone, Debug)]
struct Node {
    left: usize,
    right: usize,
    up: usize,
    down: usize,
    column: usize, // Which column does this node belong to?
}

/// Each column header tracks how many nodes (rows) cover this column.
#[derive(Clone, Debug)]
struct Column {
    size: usize,
}

/// A Dancing Links solver for exact cover.
pub struct DancingLinks {
    root: usize,
    nodes: Vec<Node>,
    cols: Vec<Column>,
    /// Number of columns (excluding the special root header).
    num_cols: usize,
    /// Which row does a given node correspond to? (for reconstruction)
    row_id: Vec<usize>,
    /// Partial solution stack used by the search.
    solution_stack: Vec<usize>,
    // Col names are optional; we just track count by index
}

impl DancingLinks {
    /// Constructs the dancing-links structure from a boolean matrix:
    /// * `matrix[r][c] = true` means row r covers column c.
    /// The rows are 0..matrix.len(), columns 0..matrix[0].len().
    pub fn new(matrix: &[Vec<bool>]) -> Self {
        let rows = matrix.len();
        let cols = if rows == 0 { 0 } else { matrix[0].len() };

        // One extra for the root header.
        let mut dlx = DancingLinks {
            root: 0,
            nodes: vec![],
            cols: vec![],
            num_cols: cols,
            row_id: vec![],
            solution_stack: vec![],
        };

        // For empty matrix, just return with root node pointing to itself
        if cols == 0 {
            dlx.nodes.push(Node {
                left: 0,
                right: 0,
                up: 0,
                down: 0,
                column: 0,
            });
            dlx.cols.push(Column { size: 0 });
            return dlx;
        }

        // Initialize column headers + root
        // We'll store columns from 1..=cols, plus node[0] is the "root".
        dlx.nodes.reserve(1 + cols); // root + each col header
        dlx.cols.reserve(cols + 1);

        // Create the "root" node
        dlx.nodes.push(Node {
            left: 0,
            right: 0,
            up: 0,
            down: 0,
            column: 0,
        });
        dlx.cols.push(Column { size: 0 });

        // Link column headers in a left-right ring
        for c in 1..=cols {
            dlx.nodes.push(Node {
                left: if c == 1 { 0 } else { c - 1 },
                right: if c == cols { 0 } else { c + 1 },
                up: c,
                down: c,
                column: c,
            });
            dlx.cols.push(Column { size: 0 });
        }
        dlx.nodes[0].left = cols; // root's left = last col
        dlx.nodes[0].right = 1; // root's right = first col

        // Add rows
        let mut current_node_idx = 1 + cols; // next free node index
        for r in 0..rows {
            let mut first_in_row: Option<usize> = None;
            for c in 0..cols {
                if !matrix[r][c] {
                    continue;
                }
                // Insert a new node
                let col_header_idx = c + 1; // column header node index

                // Get the necessary values before mutating
                let up_idx = dlx.nodes[col_header_idx].up;

                // Create the new node
                let node_idx = current_node_idx;
                current_node_idx += 1;

                dlx.nodes.push(Node {
                    column: col_header_idx,
                    up: up_idx,
                    down: col_header_idx,
                    left: node_idx,
                    right: node_idx,
                });
                dlx.row_id.push(r);

                // Fix up-down links
                dlx.nodes[node_idx].down = col_header_idx;
                dlx.nodes[up_idx].down = node_idx;
                dlx.nodes[col_header_idx].up = node_idx;

                // Increment column size
                dlx.cols[col_header_idx].size += 1;

                // Link left-right within the row
                if let Some(first) = first_in_row {
                    let left_idx = dlx.nodes[first].left;
                    // Insert node to the left of 'first' so we form a ring
                    dlx.nodes[node_idx].right = first;
                    dlx.nodes[node_idx].left = left_idx;
                    dlx.nodes[left_idx].right = node_idx;
                    dlx.nodes[first].left = node_idx;
                } else {
                    first_in_row = Some(node_idx);
                }
            }
        }

        dlx
    }

    /// Solve the exact cover problem, returning ALL solutions (each solution is a list of row indices).
    pub fn solve_all(&mut self) -> Vec<Vec<usize>> {
        let mut solutions = Vec::new();
        self.search(&mut solutions);
        solutions
    }

    fn search(&mut self, solutions: &mut Vec<Vec<usize>>) {
        // If root is its own right, no columns remain => found a solution
        if self.nodes[self.root].right == self.root {
            // Collect row indices from solution_stack
            let mut sol_rows = Vec::with_capacity(self.solution_stack.len());
            for &node_idx in &self.solution_stack {
                // The node itself belongs to some row
                let r = self.row_id[node_idx - (1 + self.num_cols)];
                sol_rows.push(r);
            }
            solutions.push(sol_rows);
            return;
        }

        // Choose the column with fewest rows (heuristic)
        let col = {
            let mut c = self.nodes[self.root].right;
            let mut best = c;
            let mut best_size = self.cols[c].size;
            while c != self.root {
                if self.cols[c].size < best_size {
                    best = c;
                    best_size = self.cols[c].size;
                    if best_size == 0 {
                        break; // No solution down this path
                    }
                }
                c = self.nodes[c].right;
            }
            best
        };

        // If that column has no nodes, no solution
        if self.cols[col].size == 0 {
            return;
        }

        // Cover this column
        self.cover(col);

        // For each row in col
        let mut row_node = self.nodes[col].down;
        while row_node != col {
            // Push this row in the solution
            self.solution_stack.push(row_node);

            // Cover all columns in this row
            let mut right_node = self.nodes[row_node].right;
            while right_node != row_node {
                self.cover(self.nodes[right_node].column);
                right_node = self.nodes[right_node].right;
            }

            // Recurse
            self.search(solutions);

            // Uncover
            let mut left_node = self.nodes[row_node].left;
            while left_node != row_node {
                self.uncover(self.nodes[left_node].column);
                left_node = self.nodes[left_node].left;
            }

            self.solution_stack.pop();
            row_node = self.nodes[row_node].down;
        }

        // Uncover the chosen column
        self.uncover(col);
    }

    fn cover(&mut self, col: usize) {
        // Remove column header from row
        let left_col = self.nodes[col].left;
        let right_col = self.nodes[col].right;
        self.nodes[left_col].right = right_col;
        self.nodes[right_col].left = left_col;

        // For each row in the column
        let mut row_node = self.nodes[col].down;
        while row_node != col {
            // For each node in that row
            let mut node = self.nodes[row_node].right;
            while node != row_node {
                // Unlink this node from its column
                let up = self.nodes[node].up;
                let down = self.nodes[node].down;
                self.nodes[up].down = down;
                self.nodes[down].up = up;
                self.cols[self.nodes[node].column].size -= 1;
                node = self.nodes[node].right;
            }
            row_node = self.nodes[row_node].down;
        }
    }

    fn uncover(&mut self, col: usize) {
        // For each row in reverse order
        let mut row_node = self.nodes[col].up;
        while row_node != col {
            // For each node in that row
            let mut node = self.nodes[row_node].left;
            while node != row_node {
                let up = self.nodes[node].up;
                let down = self.nodes[node].down;
                self.nodes[up].down = node;
                self.nodes[down].up = node;
                self.cols[self.nodes[node].column].size += 1;
                node = self.nodes[node].left;
            }
            row_node = self.nodes[row_node].up;
        }
        // Re-link the column header
        let left_col = self.nodes[col].left;
        let right_col = self.nodes[col].right;
        self.nodes[left_col].right = col;
        self.nodes[right_col].left = col;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_small_exact_cover() {
        // Columns: 4
        // Rows: R0..R5
        // R0 -> [1, 0, 1, 0]
        // R1 -> [1, 0, 0, 1]
        // R2 -> [0, 1, 1, 0]
        // R3 -> [0, 1, 0, 1]
        // R4 -> [1, 0, 1, 0]
        // R5 -> [0, 1, 0, 1]
        let matrix = vec![
            vec![true, false, true, false],
            vec![true, false, false, true],
            vec![false, true, true, false],
            vec![false, true, false, true],
            vec![true, false, true, false],
            vec![false, true, false, true],
        ];
        let mut dlx = DancingLinks::new(&matrix);
        let solutions = dlx.solve_all();
        // At least one valid solution: e.g. row1 + row2 covers all 4 columns
        assert!(!solutions.is_empty());

        // Check that each solution truly covers each column exactly once
        for sol in solutions {
            let mut covered = vec![false; 4];
            for r in sol {
                for (c, &val) in matrix[r].iter().enumerate() {
                    if val {
                        assert!(!covered[c], "Column {} covered twice", c);
                        covered[c] = true;
                    }
                }
            }
            assert!(covered.iter().all(|&x| x), "Not all columns covered");
        }
    }

    #[test]
    fn test_empty() {
        // No rows or columns
        let matrix: Vec<Vec<bool>> = vec![];
        let mut dlx = DancingLinks::new(&matrix);
        let solutions = dlx.solve_all();
        // By convention, there's exactly one solution if no columns remain
        assert_eq!(solutions.len(), 1);
        assert!(solutions[0].is_empty());
    }

    #[test]
    fn test_single_col() {
        // 1 column, 2 rows
        // Row0 covers the column, Row1 does not
        let matrix = vec![vec![true], vec![false]];
        let mut dlx = DancingLinks::new(&matrix);
        let solutions = dlx.solve_all();
        // The only solution is picking row0
        assert_eq!(solutions, vec![vec![0]]);
    }
}
