use matrix_oxide::Matrix;
use std::time::{Duration, Instant};

fn main() {
    bench_matrix_multiplication();
    bench_matrix_inversion();
}

fn bench_matrix_multiplication() {
    println!("Benchmarking matrix multiplication...");

    let b: Matrix<i64> = Matrix::new_random(100, 100);
    let a: Matrix<i64> = Matrix::new_random(100, 100);
    let start = Instant::now();
    let _ab = a.multiply(&b).expect(
        "Matrix Multiplication To Work. If you're reading this open an issue:
        https://github.com/antonio-hickey/matrix-oxide/issues",
    );
    let duration = start.elapsed();
    println!(
        "Ran mat mul op on 100x100 matrix of i64's in: {:?}",
        duration
    );

    let b: Matrix<f64> = Matrix::new_random(100, 100);
    let a: Matrix<f64> = Matrix::new_random(100, 100);
    let start = Instant::now();
    let _ab = a.multiply(&b).expect(
        "Matrix Multiplication To Work. If you're reading this open an issue:
        https://github.com/antonio-hickey/matrix-oxide/issues",
    );
    let duration = start.elapsed();
    println!(
        "Ran mat mul op on 100x100 matrix of f64's in: {:?}",
        duration
    );

    let duration: u64 = (0..100).fold(0, |acc, _| {
        let b: Matrix<i64> = Matrix::new_random(100, 100);
        let a: Matrix<i64> = Matrix::new_random(100, 100);

        let start = Instant::now();
        let _ab = a.multiply(&b).expect(
            "Matrix Multiplication To Work. If you're reading this open an issue:
            https://github.com/antonio-hickey/matrix-oxide/issues",
        );

        let sub_duration = start.elapsed().as_millis() as u64;

        acc + sub_duration
    });
    println!(
        "Ran 100 mat mul op's on 100x100 matrix of i64's in: {:?}",
        Duration::from_millis(duration)
    );

    let duration: u64 = (0..100).fold(0, |acc, _| {
        let b: Matrix<f64> = Matrix::new_random(100, 100);
        let a: Matrix<f64> = Matrix::new_random(100, 100);

        let start = Instant::now();
        let _ab = a.multiply(&b).expect(
            "Matrix Multiplication To Work. If you're reading this open an issue:
            https://github.com/antonio-hickey/matrix-oxide/issues",
        );

        let sub_duration = start.elapsed().as_millis() as u64;

        acc + sub_duration
    });
    println!(
        "Ran 100 mat mul op's on 100x100 matrix of f64's in: {:?}",
        Duration::from_millis(duration)
    );
}

fn bench_matrix_inversion() {
    println!("Benchmarking matrix inversion...");

    // When dealing with random matrices it's possible to get
    // a matrix that's not full rank (r != M), so we want to loop
    // until we find one that is and then run the benchmark.
    let mut inverse: Option<Matrix<f64>> = None;
    while inverse.is_none() {
        let a: Matrix<f64> = Matrix::new_random(100, 100);
        let start = Instant::now();
        inverse = a.inverse();
        let duration = start.elapsed();
        println!(
            "Ran matrix inversion op on 100x100 matrix of i64's in: {:?}",
            duration
        );
    }

    let duration: u64 = (0..100).fold(0, |acc, _| {
        // When dealing with random matrices it's possible to get
        // a matrix that's not full rank (r != M), so we want to loop
        // until we find one that is and then run the benchmark.
        let mut sub_duration: u64 = 0;
        let mut inverse: Option<Matrix<f64>> = None;
        while inverse.is_none() {
            let a: Matrix<f64> = Matrix::new_random(100, 100);
            let start = Instant::now();
            inverse = a.inverse();
            sub_duration = start.elapsed().as_millis() as u64;
        }

        acc + sub_duration
    });
    println!(
        "Ran 100 matric inversion op's on 100x100 matrix of f64's in: {:?}",
        Duration::from_millis(duration)
    );
}
