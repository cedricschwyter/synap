use crate::algebra::matrix::*;

pub fn vec_vec_mul_naive<T: MatrixElement<T>>(lhs: &Matrix<T>, rhs: &Matrix<T>) -> Matrix<T> {
    let mut value: T = lhs[0][0] * rhs[0][0];
    for i in 1..lhs.width() {
        value = value + lhs[0][i] * rhs[i][0];
    }
    Matrix::<T>::scalar(value)
}

pub fn mat_mat_mul_naive<T: MatrixElement<T>>(lhs: &Matrix<T>, rhs: &Matrix<T>) -> Matrix<T> {
    let mut elements = Vec::<Vec<T>>::new();
    for row in 0..lhs.height() {
        elements.push(Vec::<T>::new());
        for col in 0..rhs.width() {
            let mut r = Vec::<T>::new();
            for i in 0..rhs.height() {
                r.push(rhs[i][col]);
            }
            elements[row].push(
                (Matrix::<T>::vector(lhs[row].to_vec()).transpose() * Matrix::<T>::vector(r))
                    .to_scalar(),
            );
        }
    }
    Matrix::<T>::new(elements)
}

pub fn gauss_elim_naive<T: MatrixElement<T>>(matrix: &Matrix<T>) -> Matrix<T> {
    unreachable!();
}

pub fn lu_decomp_naive<T: MatrixElement<T>>(matrix: &Matrix<T>) -> (Matrix<T>, Matrix<T>) {
    unreachable!();
}

pub fn qr_decomp_naive<T: MatrixElement<T>>(matrix: &Matrix<T>) -> (Matrix<T>, Matrix<T>) {
    unreachable!();
}

pub fn det_naive<T: MatrixElement<T>>(matrix: &Matrix<T>) -> T {
    unreachable!();
}

pub fn rank_naive<T: MatrixElement<T>>(matrix: &Matrix<T>) -> u32 {
    unreachable!();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore]
    fn det_square_matrix() {
        let matrix = Matrix::<i32>::new(vec![vec![1, 2], vec![3, 4]]);
        assert_eq!(det_naive(&matrix), -2);
        let matrix = Matrix::<i32>::new(vec![
            vec![4, 3, 2, 1],
            vec![1, 2, 3, 4],
            vec![3, 4, 1, 2],
            vec![2, 1, 4, 4],
        ]);
        assert_eq!(det_naive(&matrix), 20);
    }
}
