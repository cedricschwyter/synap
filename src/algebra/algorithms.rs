use crate::algebra::matrix::*;

pub fn gauss_elim<T: MatrixElement<T>>(matrix: &Matrix<T>) -> Matrix<T> {
    unreachable!();
}

pub fn lu_decomp<T: MatrixElement<T>>(matrix: &Matrix<T>) -> (Matrix<T>, Matrix<T>) {
    unreachable!();
}

pub fn qr_decomp<T: MatrixElement<T>>(matrix: &Matrix<T>) -> (Matrix<T>, Matrix<T>) {
    unreachable!();
}

pub fn det<T: MatrixElement<T>>(matrix: &Matrix<T>) -> T {
    unreachable!();
}

pub fn rank<T: MatrixElement<T>>(matrix: &Matrix<T>) -> u32 {
    unreachable!();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore]
    fn det_square_matrix() {
        let matrix = Matrix::<i32>::new(vec![vec![1, 2], vec![3, 4]]);
        assert_eq!(det(&matrix), -2);
        let matrix = Matrix::<i32>::new(vec![
            vec![4, 3, 2, 1],
            vec![1, 2, 3, 4],
            vec![3, 4, 1, 2],
            vec![2, 1, 4, 4],
        ]);
        assert_eq!(det(&matrix), 20);
    }
}
