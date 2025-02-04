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

    /// Compute a shrunken version of the vector.
    ///
    /// NOTE: The scalar (λ) MUST meet the following condition
    /// to shrink a vector: 0 < λ < 1. (The scalar must be
    /// greater than 0 and also less than 1).
    ///
    /// NOTE: The shrunken vector will always be of type f64
    /// no matter the original type of the vector pre shrink.
    fn shrink(&self, scalar: f64) -> Option<Vec<f64>>;

    /// Compute a stretched version of the vector.
    ///
    /// NOTE: The scalar (λ) MUST meet the following condition
    /// to str a vector: 0 > λ > 1. (The scalar must be
    /// less than 0 and also greater than 1).
    ///
    /// NOTE: A vector stretched with a negative scalar means
    /// geometrically the vector will do a 180 and then be stretched.
    ///
    /// NOTE: The stretched vector will always be of type f64
    /// no matter the original type of the vector pre stretch.
    fn stretch<U>(&self, scalar: U) -> Option<Vec<f64>>
    where
        U: Into<f64> + PartialOrd + Copy;
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

    /// Compute a shrunken version of the vector.
    ///
    /// NOTE: The scalar (λ) MUST meet the following condition
    /// to shrink a vector: 0 < λ < 1. (The scalar must be
    /// greater than 0 and also less than 1).
    ///
    /// NOTE: The shrunken vector will always be of type f64
    /// no matter the original type of the vector pre shrink.
    fn shrink(&self, scalar: f64) -> Option<Vec<f64>> {
        // In order to shrink a vector the scalar must
        // be greater than 0.0 and less than 1.0
        if !(0.0..1.0).contains(&scalar) {
            return None;
        }

        let shrunken_vector = self.iter().map(|x| x.as_f64() * scalar).collect();

        Some(shrunken_vector)
    }

    /// Compute a stretched version of the vector.
    ///
    /// NOTE: The scalar (λ) MUST meet the following condition
    /// to str a vector: 0 > λ > 1. (The scalar must be
    /// less than 0 and also greater than 1).
    ///
    /// NOTE: A vector stretched with a negative scalar means
    /// geometrically the vector will do a 180 and then be stretched.
    ///
    /// NOTE: The stretched vector will always be of type f64
    /// no matter the original type of the vector pre stretch.
    fn stretch<U>(&self, scalar: U) -> Option<Vec<f64>>
    where
        U: Into<f64> + PartialOrd + Copy,
    {
        let scalar: f64 = scalar.into();

        // In order to stretch a vector the scalar must
        // be less than 0.0 and greater than 1.0
        if (0.0..1.0).contains(&scalar) {
            return None;
        }

        let stretched_vector = self.iter().map(|x| x.as_f64() * scalar).collect();

        Some(stretched_vector)
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

    #[test]
    fn test_shrink_invalid_scalar() {
        let v = vec![1_i32, 2, 3];
        // Scalar is 1.0 (not less than 1.0) → should return None.
        assert_eq!(v.shrink(1.0), None);

        // Negative scalar → should return None.
        assert_eq!(v.shrink(-0.1), None);
    }

    #[test]
    fn test_shrink_i8() {
        let v: Vec<i8> = vec![10, 20, 30];
        let scalar = 0.5;
        let result = v.shrink(scalar).expect("Expected valid shrink result");
        // Each element should be multiplied by 0.5: 10 * 0.5 = 5.0, etc.
        assert_eq!(result, vec![5.0, 10.0, 15.0]);
    }

    #[test]
    fn test_shrink_i32() {
        let v: Vec<i32> = vec![1000, 2000, 3000];
        let scalar = 0.2;
        let result = v.shrink(scalar).expect("Expected valid shrink result");
        assert_eq!(result, vec![200.0, 400.0, 600.0]);
    }

    #[test]
    fn test_shrink_i64() {
        let v: Vec<i64> = vec![1000, 2000, 3000];
        let scalar = 0.1;
        let result = v.shrink(scalar).expect("Expected valid shrink result");
        assert_eq!(result, vec![100.0, 200.0, 300.0]);
    }

    #[test]
    fn test_shrink_u64() {
        let v: Vec<u64> = vec![1000, 2000, 3000];
        let scalar = 0.1;
        let result = v.shrink(scalar).expect("Expected valid shrink result");
        assert_eq!(result, vec![100.0, 200.0, 300.0]);
    }

    #[test]
    fn test_shrink_f32() {
        let v: Vec<f32> = vec![10.0, 20.0, 30.0];
        let scalar = 0.5;
        let result = v.shrink(scalar).expect("Expected valid shrink result");
        assert_eq!(result, vec![5.0, 10.0, 15.0]);
    }

    #[test]
    fn test_shrink_f64() {
        let v: Vec<f64> = vec![10.0, 20.0, 30.0];
        let scalar = 0.5;
        let result = v.shrink(scalar).expect("Expected valid shrink result");
        assert_eq!(result, vec![5.0, 10.0, 15.0]);
    }

    #[test]
    fn test_stretch_with_scalar_in_range_returns_none() {
        let vec: Vec<f64> = vec![1.0, 2.0, 3.0];
        assert_eq!(vec.stretch(0.5), None);
    }

    #[test]
    fn test_stretch_with_scalar_equal_to_one_returns_same_vector() {
        let vec: Vec<f64> = vec![1.0, 2.0, 3.0];
        let expected: Vec<f64> = vec![1.0, 2.0, 3.0];
        assert_eq!(vec.stretch(1.0), Some(expected));
    }

    #[test]
    fn test_stretch_with_scalar_greater_than_one() {
        let vec: Vec<f64> = vec![1.0, 2.0, 3.0];
        let scalar = 2.0;
        let expected: Vec<f64> = vec![2.0, 4.0, 6.0];
        assert_eq!(vec.stretch(scalar), Some(expected));
    }

    #[test]
    fn test_stretch_with_integer_scalar() {
        let vec: Vec<f64> = vec![1.0, 2.0, 3.0];
        let scalar = 3;
        let expected: Vec<f64> = vec![3.0, 6.0, 9.0];
        assert_eq!(vec.stretch(scalar), Some(expected));
    }

    #[test]
    fn test_stretch_with_negative_scalar() {
        let vec: Vec<f64> = vec![1.0, 2.0, 3.0];
        let scalar = -2.0;
        let expected: Vec<f64> = vec![-2.0, -4.0, -6.0];
        assert_eq!(vec.stretch(scalar), Some(expected));
    }
}
