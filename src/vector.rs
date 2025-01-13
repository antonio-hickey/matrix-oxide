use crate::numbers::Numeric;
use std::ops::{Add, Mul};

pub trait VectorOps<T>
where
    T: Numeric + Default + Mul<Output = T> + Add<Output = T> + Clone,
{
    /// Compute the dot product between 2 vectors. This outputs a single number
    /// that provides information about the relationship between the 2 vectors.
    ///
    /// NOTE: The two vectors MUST have the same dimensionality in order to
    /// compute the dot product for them.
    fn dot_product(&self, b: &[T]) -> Option<T>;

    /// Compute the dot product of a vector and itself.
    fn squared_norm(&self) -> T;
}
impl<T> VectorOps<T> for Vec<T>
where
    T: Numeric + Default + Mul<Output = T> + Add<Output = T> + Clone,
{
    /// Compute the dot product between 2 vectors. This outputs a single number
    /// that provides information about the relationship between the 2 vectors.
    ///
    /// NOTE: The two vectors MUST have the same dimensionality in order to
    /// compute the dot product for them.
    fn dot_product(&self, b: &[T]) -> Option<T> {
        // In order to compute the dot product between two
        // vectors they most have the same dimensionality
        if self.len() != b.len() {
            return None;
        }

        // Element wise multiplication on the vectors while accumulating a sum
        // of the products, which after summing each product is the dot product.
        Some(self.iter().zip(b).fold(T::default(), |acc, (ai, bi)| {
            (ai.clone() * bi.clone()) + acc
        }))
    }

    /// Compute the squared norm of a vector, the dot product of a vector and itself.
    fn squared_norm(&self) -> T {
        self.iter()
            .fold(T::default(), |acc, x| (x.clone() * x.clone()) + acc)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dot_product_integers() {
        let vec1 = vec![1, 2, 3];
        let vec2 = vec![4, 5, 6];
        let result = vec1.dot_product(&vec2).unwrap();
        assert_eq!(result, 32);
    }

    #[test]
    fn test_dot_product_floats() {
        let vec1 = vec![1.0, 2.0, 3.0];
        let vec2 = vec![4.0, 5.0, 6.0];
        let result = vec1.dot_product(&vec2).unwrap();
        assert!(((result - 32.0) as f32).abs() < f32::EPSILON);
    }

    #[test]
    fn test_dot_product_empty_vectors() {
        let vec1: Vec<i32> = vec![];
        let vec2: Vec<i32> = vec![];
        let result = vec1.dot_product(&vec2).unwrap();
        assert_eq!(result, 0);
    }

    #[test]
    fn test_dot_product_mismatched_lengths() {
        let vec1 = vec![1, 2];
        let vec2 = vec![3, 4, 5];
        let result = vec1.dot_product(&vec2);
        assert_eq!(result, None);
    }

    #[test]
    fn test_dot_product_single_element() {
        let vec1 = vec![7];
        let vec2 = vec![3];
        let result = vec1.dot_product(&vec2).unwrap();
        assert_eq!(result, 21);
    }

    #[test]
    fn test_dot_product_with_zeros() {
        let vec1 = vec![0, 0, 0];
        let vec2 = vec![1, 2, 3];
        let result = vec1.dot_product(&vec2).unwrap();
        assert_eq!(result, 0);
    }

    #[test]
    fn test_dot_product_negative_numbers() {
        let vec1 = vec![-1, -2, -3];
        let vec2 = vec![4, 5, 6];
        let result = vec1.dot_product(&vec2).unwrap();
        assert_eq!(result, -32);
    }

    #[test]
    fn test_squared_norm_empty() {
        let v: Vec<i32> = vec![];
        assert_eq!(v.squared_norm(), 0);
    }

    #[test]
    fn test_squared_norm_single_element() {
        let v = vec![5];
        assert_eq!(v.squared_norm(), 25);
    }

    #[test]
    fn test_squared_norm_multiple_integers() {
        let v = vec![2, -3, 4];
        assert_eq!(v.squared_norm(), 29);
    }

    #[test]
    fn test_squared_norm_with_negatives() {
        let v = vec![-1, -2, -3];
        assert_eq!(v.squared_norm(), 14);
    }

    #[test]
    fn test_squared_norm_large_values() {
        let v = vec![1000, 2000, 3000];
        assert_eq!(v.squared_norm(), 14_000_000);
    }

    #[test]
    fn test_squared_norm_floats() {
        let v = vec![1.0_f64, 2.5_f64, 3.2_f64];
        let result = v.squared_norm();
        let expected = 17.49;
        let epsilon = 1e-10;
        assert!((result - expected).abs() < epsilon);
    }
}
