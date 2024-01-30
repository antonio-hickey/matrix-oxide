use std::fmt::Debug;
use std::ops::{Add, Mul, Sub};

/// Non resizeable MxN Matrix
pub struct Matrix<T, const R: usize, const C: usize> {
    pub data: Vec<T>,
}
impl<T: Default + Clone, const R: usize, const C: usize> Matrix<T, R, C> {
    /// Construct a new *non-empty* and *sized* `Matrix`
    pub fn new() -> Self {
        Matrix {
            data: vec![T::default(); R * C],
        }
    }

    /// Try to get a reference to the value at a given row and column from the matrix
    pub fn get(&self, row: usize, col: usize) -> Option<&T> {
        if row < R && col < C {
            Some(&self.data[col + row * C])
        } else {
            None
        }
    }

    /// Get a vector of the diagonal elements of the matrix
    pub fn get_diagonal(&self) -> Vec<T> {
        (0..C)
            .filter_map(|col_idx| self.get(col_idx, col_idx).cloned())
            .collect()
    }

    /// Try to get a mutable reference to the value at a given row and column from the matrix
    pub fn get_mut(&mut self, row: usize, col: usize) -> Option<&mut T> {
        if row < R && col < C {
            Some(&mut self.data[col + row * C])
        } else {
            None
        }
    }

    /// Try to set a value at a given row and column in the matrix
    pub fn set(&mut self, row: usize, column: usize, value: T) -> bool {
        if let Some(cell) = self.get_mut(row, column) {
            *cell = value;
            true
        } else {
            false
        }
    }

    /// Try to get all the values for a given column
    ///
    /// NOTE: If you pass a column value larger than the number of columns
    /// this function will return None.
    pub fn try_get_column(&self, column: usize) -> Option<Vec<T>> {
        // Bounds check
        if column >= C {
            return None;
        }

        // Iterate over all the rows grabbing a specific column each time
        let col_data: Vec<T> = (0..R)
            .map(|row| self.data[row * C + column].clone())
            .collect();

        Some(col_data)
    }

    /// Try to get all the values for a given row
    ///
    /// NOTE: If you pass a row value larger than the number of rows
    /// this function will return None.
    pub fn try_get_row(&self, row: usize) -> Option<Vec<T>> {
        // Bounds check
        if row >= R {
            return None;
        }

        // Iterate over all the rows grabbing a specific column each time
        let row_data: Vec<T> = (0..C).map(|col| self.data[row * C + col].clone()).collect();

        Some(row_data)
    }

    /// Perform a transpose operation (swap rows for columns and vice versa)
    /// Example:
    ///  [[1, 2, 3]       [[1, 4]
    ///   [4, 5, 6]]   ->  [2, 5]
    ///                    [3, 6]]
    pub fn transpose(&self) -> Matrix<T, R, C> {
        Matrix {
            data: (0..C)
                .flat_map(|col| (0..R).map(move |row| self.data[row * C + col].clone()))
                .collect(),
        }
    }
}
impl<T: Default + Clone, const R: usize, const C: usize> Default for Matrix<T, R, C> {
    /// Create a default `Matrix` instance
    fn default() -> Self {
        Self::new()
    }
}
impl<T: Default + Clone + Debug, const R: usize, const C: usize> Add for Matrix<T, R, C>
where
    T: Add<Output = T> + Clone,
{
    type Output = Matrix<T, R, C>;

    /// Matrix addition
    /// NOTE: the matrices you add MUST have the same dimensionality
    fn add(self, rhs: Self) -> Matrix<T, R, C> {
        let data: Vec<T> = (0..R)
            .flat_map(|row| {
                let row_a = self.try_get_row(row).expect("Invalid row in self");
                let row_b = rhs.try_get_row(row).expect("Invalid row in rhs");
                row_a.into_iter().zip(row_b).map(|(a, b)| a + b)
            })
            .collect();

        Matrix { data }
    }
}
impl<T, const R: usize, const C: usize> Matrix<T, R, C>
where
    T: Default + Clone + Add<Output = T>,
{
    /// Perform the trace operation that computes the sum of all diagonal
    /// elements in the matrix.
    ///
    /// NOTE: off-diagnonal elements do NOT contribute to the trace of the
    /// matrix, so 2 very different matrices can have the same trace.
    pub fn trace(&self) -> T {
        self.get_diagonal()
            .into_iter()
            .fold(T::default(), |acc, diagonal| acc + diagonal)
    }
}
impl<T: Default + Clone + Debug, const R: usize, const C: usize> Sub for Matrix<T, R, C>
where
    T: Sub<Output = T> + Clone,
{
    type Output = Matrix<T, R, C>;

    /// Subtract a matrix by another matrix
    /// NOTE: the matrix you subtract by MUST have the same dimensionality
    /// Example:
    ///  [[1, 2, 3]     [[6, 5, 4],    [[-5, -3, -1]
    ///   [4, 5, 6]]  -  [3, 2, 1]]  =  [1, 3, 5]]
    fn sub(self, rhs: Self) -> Matrix<T, R, C> {
        let data: Vec<T> = (0..R)
            .flat_map(|row| {
                let row_a = self.try_get_row(row).expect("Invalid row in self");
                let row_b = rhs.try_get_row(row).expect("Invalid row in rhs");
                row_a.into_iter().zip(row_b).map(|(a, b)| a - b)
            })
            .collect();

        Matrix { data }
    }
}
impl<T, const R: usize, const C: usize> Matrix<T, R, C>
where
    T: Mul<Output = T> + Clone,
{
    /// Multiply a matrix by a single number (scalar)
    /// NOTE: The scalar type MUST match the matrix type.
    pub fn scalar_multiply(&self, scalar: T) -> Matrix<T, R, C> {
        let data = self
            .data
            .iter()
            .map(|value| value.clone() * scalar.clone())
            .collect();
        Matrix { data }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_martrix_trace() {
        let matrix = Matrix::<i32, 2, 2> {
            data: vec![1, 2, 3, 4],
        };

        let expected: i32 = 5;
        let result = matrix.trace();
        assert_eq!(result, expected);
    }

    #[test]
    fn test_martrix_diagonal() {
        let matrix = Matrix::<i32, 2, 2> {
            data: vec![1, 2, 3, 4],
        };

        let expected: Vec<i32> = vec![1, 4];
        let result = matrix.get_diagonal();
        assert_eq!(result, expected);
    }

    #[test]
    fn test_matrix_scalar_multiplication() {
        let matrix = Matrix::<i32, 2, 2> {
            data: vec![1, 2, 3, 4],
        };

        let expected = Matrix::<i32, 2, 2> {
            data: vec![2, 4, 6, 8],
        };
        let result = matrix.scalar_multiply(2);

        assert_eq!(result.data, expected.data);
    }

    #[test]
    fn test_matrix_subtraction() {
        let matrix_a = Matrix::<i32, 2, 3> {
            data: vec![1, 2, 3, 4, 5, 6],
        };
        let matrix_b = Matrix::<i32, 2, 3> {
            data: vec![6, 5, 4, 3, 2, 1],
        };

        let expected = Matrix::<i32, 2, 3> {
            data: vec![-5, -3, -1, 1, 3, 5],
        };
        let result = matrix_a - matrix_b;
        assert_eq!(result.data, expected.data);
    }

    #[test]
    fn test_matrix_addition() {
        let matrix_a = Matrix::<i32, 2, 2> {
            data: vec![1, 2, 3, 4],
        };
        let matrix_b = Matrix::<i32, 2, 2> {
            data: vec![4, 3, 2, 1],
        };

        let expected = Matrix::<i32, 2, 2> {
            data: vec![5, 5, 5, 5],
        };

        let result = matrix_a + matrix_b;
        assert_eq!(result.data, expected.data);
    }

    #[test]
    fn new_matrix_has_correct_size() {
        let matrix: Matrix<i32, 2, 3> = Matrix::new();
        assert_eq!(matrix.data.len(), 6);
    }

    #[test]
    fn default_matrix_is_same_as_new() {
        let default_matrix: Matrix<i32, 2, 3> = Matrix::default();
        let new_matrix: Matrix<i32, 2, 3> = Matrix::new();
        assert_eq!(default_matrix.data, new_matrix.data);
    }

    #[test]
    fn try_get_column_valid() {
        let matrix: Matrix<i32, 2, 3> = Matrix::new();
        let column = matrix.try_get_column(1);
        assert!(column.is_some());
        assert_eq!(column.unwrap(), vec![0, 0]);
    }

    #[test]
    fn try_get_column_invalid() {
        let matrix: Matrix<i32, 2, 3> = Matrix::new();
        let column = matrix.try_get_column(3);
        assert!(column.is_none());
    }

    #[test]
    fn try_get_row_valid() {
        let matrix: Matrix<i32, 2, 3> = Matrix::new();
        let row = matrix.try_get_row(0);
        assert!(row.is_some());
        assert_eq!(row.unwrap(), vec![0, 0, 0]);
    }

    #[test]
    fn try_get_row_invalid() {
        let matrix: Matrix<i32, 2, 3> = Matrix::new();
        let row = matrix.try_get_row(2);
        assert!(row.is_none());
    }

    #[test]
    fn transpose_works_correctly() {
        let mut matrix: Matrix<i32, 2, 3> = Matrix::new();
        for i in 0..matrix.data.len() {
            matrix.data[i] = i as i32;
        }
        let transposed = matrix.transpose();
        assert_eq!(transposed.data, vec![0, 3, 1, 4, 2, 5]);
    }
}
