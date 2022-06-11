use synap::algebra::{algorithms::mat_mul_naive_threaded, matrix::Matrix};

fn main() {
    let size = 3;
    dbg!(mat_mul_naive_threaded(
        &Matrix::new(vec![vec![1; size]; size]),
        &Matrix::new(vec![vec![2; size]; size]),
    ));
}
