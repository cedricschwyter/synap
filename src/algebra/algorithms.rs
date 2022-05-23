//! Functions of this module are meant to be 'raw' implementations of the respective algorithm. The
//! functions generally expect that all relevant checks (i.e., matrix/vector dimension validation)
//! are performed by the caller. This is in the hopes to be able to save computing resources during
//! actual execution of an algorithm.
//!
//! # NOTE:
//! If you are a client of the library it is recommended to use the methods on the [`Matrix`](super::matrix::Matrix) struct
//! that implement the required behaviour as these functions perform the necessary assertions and
//! generally choose the most appropriate algorithm for the job automatically.
//! Unless you wish to explicitely execute a specific algorithm, you are most likely looking in the
//! wrong place right now.

use num::{pow, Float};

use crate::algebra::matrix::*;

/// A straightforward implementation of the euclidean scalar product
/// $\langle x, y \rangle = x^T y$ of two vectors $x, y$ that returns a scalar matrix (a 1 x 1 matrix) corresponding to the
/// numerical value of the scalar product.
///
/// * `lhs` - corresponds to $x^T$ above, is expected to be in transposed form ($x^T$) already,
/// thus of size $1 \times d$
/// * `rhs` - corresponds to $y$ above, expected to be of size $d \times 1$
pub fn euclidean_scalar_product_naive<T: FieldElement<T>>(
    lhs: &Matrix<T>,
    rhs: &Matrix<T>,
) -> Matrix<T> {
    let mut value: T = lhs[0][0] * rhs[0][0];
    for i in 1..lhs.width() {
        value = value + lhs[0][i] * rhs[i][0];
    }
    Matrix::<T>::scalar(value)
}

/// A naive implementation of the euclidean norm (colloquially, the 'length')
/// $\lvert \lvert x \rvert \rvert = \sqrt{\langle x, x \rangle}$ of a vector $x$ that returns a
/// scalar matrix (a 1 x 1 matrix) corresponding to the numerical value of the norm.
///
/// * `matrix` - corresponds to $x$ above, expected to be of size $\dim x \times 1$
pub fn euclidean_norm_naive<T: FieldElement<T> + Float>(matrix: &Matrix<T>) -> Matrix<T> {
    Matrix::<T>::scalar(
        euclidean_scalar_product_naive(matrix, matrix)
            .to_scalar()
            .sqrt(),
    )
}

/// A naive implementation of the more general $p$-norm: $\lvert \lvert x \rvert \rvert_p =
/// (\lvert x_1 \rvert ^ p + ... + \lvert x_n \rvert ^ p)^\frac{1}{p}$.
///
/// * `matrix` - corresponds to the $x$ above, expected to be of size $\dim x \times 1$
pub fn p_norm_naive<T: FieldElement<T> + Float>(matrix: &Matrix<T>, p: usize) -> Matrix<T> {
    let mut value: T = num::zero();
    for i in 0..matrix.height() {
        value = value + pow(matrix[i][0], p);
    }
    unimplemented!()
}

/// A naive implementation of the matrix multiplication algorithm, a classic $O(n^3)$
/// implementation.
/// It multiplies together matrices $A, B$ of sizes $n \times m$ and $m \times p$ respectively and
/// returns a matrix of size $n \times p$.
///
/// * `lhs` - corresponds to $A$ above, expected to be of size $n \times m$
/// * `rhs` - corresponds to $B$ above, expected to be of size $m \times p$
pub fn mat_mul_naive<T: FieldElement<T>>(lhs: &Matrix<T>, rhs: &Matrix<T>) -> Matrix<T> {
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

pub fn gauss_elim_naive<T: FieldElement<T>>(matrix: &Matrix<T>) -> Matrix<T> {
    unreachable!();
}

pub fn lu_decomp_naive<T: FieldElement<T>>(matrix: &Matrix<T>) -> (Matrix<T>, Matrix<T>) {
    unreachable!();
}

pub fn qr_decomp_naive<T: FieldElement<T>>(matrix: &Matrix<T>) -> (Matrix<T>, Matrix<T>) {
    unreachable!();
}

pub fn det_naive<T: FieldElement<T>>(matrix: &Matrix<T>) -> T {
    unreachable!();
}

pub fn rank_naive<T: FieldElement<T>>(matrix: &Matrix<T>) -> usize {
    unreachable!();
}

pub fn gram_schmidt<T: FieldElement<T>>(matrix: &Matrix<T>) -> Matrix<T> {
    unimplemented!();
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

    #[test]
    #[ignore]
    fn euclidean_scalar_product() {
        unimplemented!();
    }

    #[test]
    #[ignore]
    fn euclidean_norm() {
        unimplemented!();
    }
}
