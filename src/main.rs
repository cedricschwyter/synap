use synap::algebra::{algorithms::inverse_naive, matrix::Matrix};

fn main() {
    let matrix = Matrix::new(vec![vec![1, 0, 0], vec![0, 1, 0], vec![1, 0, 1]]);
    dbg!(inverse_naive(&matrix));
}
