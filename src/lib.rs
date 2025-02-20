//! Matrix Oxide
//! ===
//!
//! A simple, lightweight, and from scratch linear algebra library for Rust. Currently still under active development with goals at becoming more of a deep learning library.
//!
//! Installation
//! ---
//! Use cargo CLI:
//! ```
//! cargo install matrix-oxide
//! ```
//!
//! Or manually add it into your Cargo.toml:
//! ```
//! [dependencies]
//! matrix-oxide = "0.1.2"
//! ```
//!
//! Usage
//! ---
//!
//! For more thorough information, read the [docs](https://docs.rs/matrix-oxide/latest/matrix_oxide/).
//!
//!
//! Example: Multiply 2 random 2x2 matrices.
//! ```
//! let matrix_a = Matrix::<i32>::new_random(2, 2);
//! let matrix_b = Matrix::<i32>::new_random(2, 2);
//!
//! let matrix_ab = matrix_a.multiply(&matrix_b);
//! ```

pub mod activation;
pub mod matrix;
pub mod numbers;
pub mod random;
pub mod vector;

// expose `Matrix` at the crates root level
pub use matrix::Matrix;
